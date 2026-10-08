import sqlite3
import numpy as np
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd, create_app
from memory.vanilla import VanillaMemory
from memory.selector import JudgeResult
from memory.store import Conflict

class Embedding:
    identity = 'gate-test'
    def documents(self, texts): return np.ones((len(texts), 3))
    queries = documents

class Tags:
    identity = 'gate-tags'
    def __init__(self): self.calls = 0
    def extract(self, texts, **kwargs):
        self.calls += 1
        return [['hello'] for _ in texts]

class Selector:
    def __init__(self, label): self.label=label; self.calls=0
    def select(self, payload):
        self.calls += 1
        if self.label == 'error': raise ValueError('invalid model output')
        return JudgeResult(label=self.label, scores=[dict(option='valuable',score=0.5),
            dict(option='vector_only',score=0.5)],confidence=0.9,
            evidence=payload.messages[0].content,reason='test')

def payload():
    return AMLAdd(user_id='u',session_id='s',request_id='r',messages=[dict(role='user',content='hello')])

@pytest.mark.parametrize('route,queued,tags', [('valuable',1,1),('vector_only',0,0)])
def test_add_routes_are_atomic_idempotent_and_searchable(tmp_path,route,queued,tags):
    selector=Selector(route); tagger=Tags()
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE='semantic_rank'),
        Embedding(),tagger=tagger,write_selector=selector)
    app=create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test'),backend=backend)
    headers={'Authorization':'Bearer test'}
    with TestClient(app) as client:
        for _ in range(2):
            response=client.post('/add',json=payload().model_dump(),headers=headers)
            assert response.status_code==200
            assert response.json()['success'] is True
        response=client.post('/search',json=dict(user_id='u',query='hello',top_k=1),headers=headers)
        assert response.status_code==200
        assert response.json()['data'][0]['content']=='hello'
    assert selector.calls==1
    with sqlite3.connect(backend.path) as db:
        assert db.execute('SELECT count(*) FROM rag_memory_queue').fetchone()[0]==queued
        assert db.execute('SELECT count(*) FROM rag_tags').fetchone()[0]==tags
        assert db.execute('SELECT decision FROM rag_write_decisions').fetchone()[0]==route
        assert db.execute('SELECT count(*) FROM rag_memories').fetchone()[0]==1
    changed=payload().model_copy(update={'session_id':'different'})
    with pytest.raises(Conflict): backend.add(changed)
    assert selector.calls==1

@pytest.mark.parametrize('failure',['judge','embedding'])
def test_write_failure_does_not_leave_queue_or_partial_records(tmp_path,failure):
    embedder=Embedding()
    if failure=='embedding': embedder.documents=lambda texts: np.array([[float('nan'),1,1]])
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),embedder,
        write_selector=Selector('error' if failure=='judge' else 'valuable'))
    with pytest.raises(ValueError): backend.add(payload())
    with sqlite3.connect(backend.path) as db:
        for table in ['rag_requests','rag_memories','rag_write_decisions','rag_memory_queue']:
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0

def test_value_selector_passes_whole_conversation_and_roles():
    from memory.write_gate import MemoryValueSelector
    class LLM:
        def complete(self, instruction, data, schema):
            assert data['text'] == '你好\n\n我对海鲜过敏'
            assert [m['role'] for m in data['metadata']['messages']] == ['assistant','user']
            assert data['require_evidence'] is False
            return JudgeResult(label='valuable',scores=[dict(option='valuable',score=1),
                dict(option='vector_only',score=0)],confidence=1,evidence='',reason='health constraint')
    request=AMLAdd(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='assistant',content='你好'),dict(role='user',content='我对海鲜过敏')])
    assert MemoryValueSelector(LLM()).select(request).label=='valuable'

@pytest.mark.parametrize('recover,status,records',[(True,200,1),(False,502,0)])
def test_api_judge_retries_before_atomic_write(tmp_path,monkeypatch,recover,status,records):
    from memory.write_gate import MemoryValueSelector
    from memory.llm import LLMError
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    class Flaky:
        calls=0
        def complete(self,instruction,data,schema):
            self.calls+=1
            if not recover or self.calls<3: raise LLMError('temporary failure')
            return JudgeResult(label='valuable',scores=[dict(option='valuable',score=1),
                dict(option='vector_only',score=0)],confidence=1,evidence='',reason='test')
    llm=Flaky()
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedding(),
        write_selector=MemoryValueSelector(llm))
    app=create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test'),backend=backend)
    with TestClient(app) as client:
        response=client.post('/add',json=payload().model_dump(),headers={'Authorization':'Bearer test'})
    assert response.status_code==status
    assert llm.calls==3
    with sqlite3.connect(backend.path) as db:
        for table in ['rag_requests','rag_memories','rag_write_decisions','rag_memory_queue']:
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0]==records
