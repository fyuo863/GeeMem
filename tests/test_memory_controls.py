from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import sqlite3
import pytest
from memory.aml_api import AMLAdd,AMLSearch
from memory.vanilla import VanillaMemory
from memory.read_versions import READ_VERSION
from memory.search_service import SearchService
from memory.supplemental import supplemental_query
from test_vanilla import Embedder


def payload(r='a',text='Alice works at Atlas.'):
    return AMLAdd(user_id='u',request_id=r,session_id=r,messages=[dict(role='user',content=text,timestamp=1704067200000)])


def backend(tmp_path,**cfg):
    return VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'m.db'),RAG_VERSION_MODE='on',**cfg),embedder=Embedder())


def test_failed_route_hidden_retry_publishes_once(tmp_path):
    b=backend(tmp_path)
    def fail(*a): raise ValueError('index failed')
    b.route_writer=SimpleNamespace(process=fail)
    with pytest.raises(ValueError): b.add(payload())
    q=AMLSearch(user_id='u',query='Alice',top_k=10)
    assert not b.search(q)['data']
    b.route_writer=None;b.add(payload());v=b.versions.latest('u')
    b.add(payload());assert b.versions.latest('u')==v
    assert len(b.search(q)['data'])==1


def test_pinned_read_does_not_see_new_publication(tmp_path):
    b=backend(tmp_path);b.add(payload())
    t=READ_VERSION.set(('u',b.versions.latest('u')))
    try:
        b.add(payload('b','Alice moved to Rome.'))
        result=b._search_direct(AMLSearch(user_id='u',query='Alice',top_k=10))
        assert len(result['data'])==1
    finally:READ_VERSION.reset(t)
    assert len(b.search(AMLSearch(user_id='u',query='Alice',top_k=10))['data'])==2


def test_snapshot_copied_into_parallel_workers():
    seen=[]
    class Planner:
        mode='llm'
        def run(self,p,retrieve,rerank,trace):
            with ThreadPoolExecutor(2) as pool:
                list(pool.map(retrieve,[p,p]))
            return {'data':[]}
    def retrieve(p):seen.append(READ_VERSION.get());return {'data':[]}
    t=READ_VERSION.set(('u',12))
    try:SearchService(retrieve,multihop=Planner()).search(AMLSearch(user_id='u',query='Alice',top_k=1))
    finally:READ_VERSION.reset(t)
    assert seen==[('u',12),('u',12)]


def test_masking_before_reranker_and_view_mapping(tmp_path):
    b=backend(tmp_path,RAG_DISCLOSURE_MODE='mask')
    raw='Alice works at Atlas. Email alice@example.com phone 13800138000 password=demo-pass'
    b.add(payload(text=raw))
    seen=[]
    b.reranker=SimpleNamespace(score=lambda q,texts:seen.extend(texts) or [1]*len(texts))
    r=b.search(AMLSearch(user_id='u',query='Where does Alice work?',top_k=1))
    text=r['data'][0]['content']
    assert r['data'][0]['id'].startswith('view_')
    for secret in ['alice@example.com','13800138000','demo-pass']:
        assert secret not in text and all(secret not in s for s in seen)
    assert 'Atlas' in text
    with sqlite3.connect(tmp_path/'m.db') as db:
        source=db.execute('select source_id from rag_disclosure_views where user_id=?',('u',)).fetchone()[0]
        assert db.execute('select content from rag_memories where id=?',(source,)).fetchone()[0]==raw
    assert not b.search(AMLSearch(user_id='other',query='Alice',top_k=1))['data']


def test_long_question_and_irrelevant_question_skip_supplemental():
    assert supplemental_query('What is your refund policy?')[0]=='rule'
    assert supplemental_query('行程后来有什么变化？')[0]=='stage'
    assert supplemental_query('What is Alice allergic to?') is None
    assert supplemental_query('rule'+'x'*1200) is None


def test_version_survives_restart(tmp_path):
    b=backend(tmp_path);b.add(payload());v=b.versions.latest('u')
    b=backend(tmp_path)
    assert b.versions.latest('u')==v
    assert len(b.search(AMLSearch(user_id='u',query='Alice',top_k=1))['data'])==1


def test_writes_while_pinning_off_remain_visible_when_enabled(tmp_path):
    b=VanillaMemory({'RAG_MEMORY_DB':str(tmp_path/'m.db')},embedder=Embedder())
    b.add(payload())
    b=backend(tmp_path)
    assert len(b.search(AMLSearch(user_id='u',query='Alice',top_k=1))['data'])==1


@pytest.mark.parametrize('budget,expected',[(2,1),(3,1)])
def test_supplemental_respects_shared_query_budget(budget,expected):
    from memory.multihop import MultiHop,Route,Review
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,*a):return Review(sufficient=False,supports=[],missing='unknown',queries=[])
    calls=[];trace={}
    def retrieve(p):calls.append(p.query);return {'data':[]}
    MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_SUPPLEMENTAL_MODE':'on','RAG_MULTIHOP_QUERIES':str(budget)},Planner()).run(
        AMLSearch(user_id='u',query='What is the refund rule?',top_k=3),retrieve,lambda q,d:[],trace)
    assert len(calls)==1+expected and trace.get('supplemental_calls',0)==expected
