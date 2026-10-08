from contextlib import closing
import json
import numpy as np
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd,create_app
from memory.vanilla import VanillaMemory
from memory.write_gate import MemoryTypeSelector
from memory.event import EventRetriever
from test_multilabel_judge import Fake,output


class Embedder:
    identity='event-source-test'
    calls=0
    def documents(self,texts):
        self.calls+=1
        return np.ones((len(texts),3))
    def queries(self,texts):return np.ones((len(texts),3))


def test_http_event_is_source_only_and_idempotent(tmp_path,monkeypatch):
    def forbid(*a,**kw):raise AssertionError('Event builder must not call LLM')
    monkeypatch.setattr('memory.llm.LLM.complete',forbid)
    llm=Fake(output({'event':[2]})); embedder=Embedder()
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_BUILD_MODE='on'),embedder,
                          write_selector=MemoryTypeSelector(llm))
    payload=dict(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='user',content='Trip?',speaker='A'),dict(role='assistant',content='Shanghai?',speaker='B'),
        dict(role='user',content='The Shanghai trip was cancelled.',speaker='A',timestamp=1704067200000)])
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'t'},backend=backend)) as client:
        for _ in range(2):
            assert client.post('/add',json=payload,headers={'Authorization':'Bearer t'}).status_code==200
        assert client.post('/search',json=dict(user_id='u',query='Shanghai',top_k=3),headers={'Authorization':'Bearer t'}).status_code==200
    assert llm.calls==1 and embedder.calls==1
    with closing(backend.connect()) as db:
        assert db.execute('SELECT count(*) FROM event_sources').fetchone()[0]==1
        assert db.execute("SELECT count(*) FROM sqlite_master WHERE name='event_records'").fetchone()[0]==0
        row=db.execute('SELECT * FROM rag_memory_routes').fetchone()
        assert row['extraction'] is None and row['status']=='completed'
        assert json.loads(row['audit'])[0]['model_calls']==0
    hits=backend.event_retriever.search('u','cancelled',fallback=False)
    assert hits[0]['content']==payload['messages'][2]['content']
    assert hits[0]['message_timestamp']==1704067200000
    assert len(hits[0]['context'])==2
    assert backend.event_retriever.search('other','cancelled')==[]
    assert backend.event_retriever.search('u','cancelled',session_id='other')==[]


def test_fallback_rescues_unclassified_source(tmp_path):
    b=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_BUILD_MODE='on'),Embedder())
    b.add(AMLAdd(user_id='u',session_id='s',request_id='r',messages=[dict(role='user',content='I visited Tokyo yesterday.')]))
    assert b.event_retriever.search('u','Tokyo',fallback=False)==[]
    assert b.event_retriever.search('u','Tokyo')[0]['memory_type']=='fallback'


def test_missing_source_rolls_back_index(tmp_path):
    b=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_BUILD_MODE='on'),Embedder())
    p=AMLAdd(user_id='u',session_id='s',request_id='r',messages=[dict(role='user',content='trip')])
    with pytest.raises(ValueError,match='persisted'):
        b.route_writer.event_builder.write(p,[0])
    with closing(b.connect()) as db:
        assert db.execute('SELECT count(*) FROM event_sources').fetchone()[0]==0


@pytest.mark.parametrize("kind", ['profile', 'relationship', 'rule', 'event'])
def test_all_types_share_original_vectors(tmp_path, monkeypatch, kind):
    def forbid(*args, **kwargs): raise AssertionError('No generation after classification')
    monkeypatch.setattr('memory.llm.LLM.complete', forbid)
    fake=Fake(output({k:[2] for k in ['profile','relationship','rule','event']}))
    embedder=Embedder()
    b=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'), RAG_BUILD_MODE='on'),embedder,
                    write_selector=MemoryTypeSelector(fake))
    p=AMLAdd(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='user',content='context one'),dict(role='assistant',content='context two'),
        dict(role='user',content='My friend likes football.',speaker='Alice')])
    b.add(p);b.add(p)
    assert fake.calls==1 and embedder.calls==1
    with closing(b.connect()) as db:
        assert db.execute('SELECT count(*) FROM rag_memories').fetchone()[0]==3
        assert db.execute(f'SELECT count(*) FROM {kind}_sources').fetchone()[0]==1
        assert all(r['extraction'] is None for r in db.execute('SELECT * FROM rag_memory_routes'))
        assert not db.execute("SELECT name FROM sqlite_master WHERE name IN ('profile_facts','rule_records','relationship_assertions')").fetchall()
    retriever=getattr(b,kind+'_retriever')
    hit=retriever.search('u','football',fallback=False)[0]
    assert hit['content']==p.messages[2].content and hit['memory_type']==kind
    assert len(hit['context'])==2 and hit['speaker']=='Alice'
    assert retriever.search('other','football')==[]
    assert retriever.search('u','football',session_id='other')==[]


@pytest.mark.parametrize("kind", ['profile', 'relationship', 'rule'])
def test_typed_fallback(tmp_path,kind):
    b=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_BUILD_MODE='on'),Embedder())
    b.add(AMLAdd(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content='football')]))
    r=getattr(b,kind+'_retriever')
    assert r.search('u','football',fallback=False)==[]
    assert r.search('u','football')[0]['memory_type']=='fallback'
