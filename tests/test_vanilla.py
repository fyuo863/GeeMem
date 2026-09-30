from concurrent.futures import ThreadPoolExecutor
import sqlite3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd, AMLSearch, create_app
from memory.vanilla import VanillaMemory, bm25, chunks
from memory.store import Conflict


class Embedder:
    identity = 'test-three-dimensions-v1'
    def __init__(self):
        self.calls = 0
        self.query_texts = []
    def documents(self, texts):
        self.calls += 1
        return np.array([[1 + ('marathon' in t), 1 + ('Tokyo' in t), 1] for t in texts])
    def queries(self, texts):
        self.query_texts.extend(texts)
        return self.documents(texts)


def add(user='u', rid='r', session='s', content='marathon training'):
    return AMLAdd(user_id=user, request_id=rid, session_id=session,
                  messages=[dict(role='user', content=content)])


def search(user='u', **kwargs):
    return AMLSearch(user_id=user, query='marathon', top_k=100, **kwargs)


def test_chunks_preserve_substrings_and_cover_input():
    text = '  one  two, three! 杭州马拉松。 last  '
    parts = chunks(text, 5, 2)
    assert all(p in text for p in parts)
    assert parts[0] == 'one  two, three!'
    assert parts[-1].endswith('last')
    assert chunks(text, 100, 2) == [text]
    with pytest.raises(ValueError): chunks(text, 5, 5)


def test_hybrid_formula_and_options(tmp_path):
    e = Embedder()
    e.documents = lambda texts: np.tile([0., 0., 1.], (len(texts), 1))
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'), RAG_RESULT_WINDOW='0'), e)
    for i, text in enumerate(['nothing here', 'rareword rareword', 'other document']):
        store.add(add(rid=str(i), content=text))
    hits = store.search(AMLSearch(user_id='u', query='rareword', top_k=3, options=['Tokyo']))['data']
    assert hits[0]['content'] == 'rareword rareword'
    assert hits[0]['score'] == pytest.approx(1/62 + 0.5/61)
    assert e.query_texts[-1] == 'rareword\nOptions:\nTokyo'
    assert bm25(['x', 'x x', 'y'], 'missing').tolist() == [0,0,0]


def test_persistence_retry_conflict_and_user_scoping(tmp_path):
    cfg = dict(RAG_MEMORY_DB=str(tmp_path/'db'))
    e = Embedder(); store = VanillaMemory(cfg, e)
    store.add(add()); store.add(add())
    assert e.calls == 1
    with pytest.raises(Conflict): store.add(add(content='changed'))
    store.add(add(user='other', content='Tokyo'))
    reopened = VanillaMemory(cfg, Embedder())
    assert [h['content'] for h in reopened.search(search())['data']] == ['marathon training']
    assert reopened.search(search('missing')) == {'data': []}
    assert 'created_at' not in reopened.search(search())['data'][0]
    assert reopened.search(search()) == store.search(search())
    with pytest.raises(ValueError): VanillaMemory(dict(cfg,RAG_CHUNK_TOKENS='400'), e)


def test_window_is_session_scoped_and_deduplicated(tmp_path):
    cfg = dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW_SEED_K='1')
    store = VanillaMemory(cfg, Embedder())
    for i, (session, content) in enumerate([('a','neighbor before'),('a','marathon'),('b','other session')]):
        store.add(add(rid=str(i),session=session,content=content))
    hits = store.search(AMLSearch(user_id='u',query='marathon',top_k=2))['data']
    assert [h['content'] for h in hits] == ['marathon','neighbor before']
    assert len({h['id'] for h in hits}) == 2


def test_invalid_vectors_are_atomic(tmp_path):
    e = Embedder(); e.documents = lambda texts: np.array([[np.nan,1,0]])
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),e)
    with pytest.raises(ValueError): store.add(add())
    with sqlite3.connect(store.path) as db:
        assert db.execute('SELECT count(*) FROM rag_requests').fetchone()[0] == 0


def test_concurrent_retries(tmp_path):
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedder())
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda _:store.add(add()),range(8)))
    assert len(store.search(search())['data']) == 1


def test_official_api_with_vanilla(tmp_path):
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedder())
    app = create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test'),backend=store)
    headers = {'Authorization':'Bearer test'}
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/add',json=add().model_dump()).status_code == 401
        payload = add().model_dump(); payload['messages'][0]['timestamp'] = 0
        result = client.post('/add',json=payload,headers=headers)
        assert result.json() == dict(success=True,user_id='u',session_id='s',request_id='r')
        hits = client.post('/search',json=search().model_dump(),headers=headers).json()['data']
        assert hits[0]['content'] == 'marathon training'
        assert hits[0]['created_at'] == '1970-01-01T00:00:00Z'
        payload['messages'][0]['content']='changed'
        assert client.post('/add',json=payload,headers=headers).status_code==409
        assert client.post('/search',json=search('none').model_dump(),headers=headers).json()=={'data':[]}


def test_graph_db_cannot_be_reused(tmp_path):
    path=tmp_path/'graph.db'
    with sqlite3.connect(path) as db: db.execute('CREATE TABLE messages(id TEXT)')
    with pytest.raises(ValueError,match='separate database'):
        VanillaMemory(dict(RAG_MEMORY_DB=str(path)),Embedder())
