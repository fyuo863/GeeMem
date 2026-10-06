from contextlib import closing
import json
import pytest
from memory.selector import MultiLabelJudge, MultiLabelResult
from memory.write_gate import MemoryTypeSelector, MEMORY_TYPE_CONFIG
from memory.vanilla import VanillaMemory
from memory.aml_api import AMLAdd, create_app
from fastapi.testclient import TestClient
import numpy as np


def output(labels, **overrides):
    rows=[dict(label=item['name'],selected=item['name'] in labels,
        score=0.9 if item['name'] in labels else 0.1,
        message_indices=labels.get(item['name'],[]),reason='test') for item in MEMORY_TYPE_CONFIG['labels']]
    for name,values in overrides.items():
        next(r for r in rows if r['label']==name).update(values)
    return MultiLabelResult(assessments=rows)

class Fake:
    def __init__(self,result): self.result=result; self.calls=0
    def complete(self,instruction,payload,schema):
        self.calls+=1
        assert [m['index'] for m in payload['messages']]==list(range(len(payload['messages'])))
        return self.result

@pytest.mark.parametrize('kind',['exclusive','range','missing','duplicate','empty','unselected'])
def test_multilabel_invalid_outputs_retry_and_fail(monkeypatch,kind):
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    result=output({'profile':[0]})
    if kind=='exclusive': result=output({'profile':[0],'vector_only':[0]})
    if kind=='range': result=output({'profile':[1]})
    if kind=='missing': result.assessments.pop()
    if kind=='duplicate': result=output({'profile':[0,0]})
    if kind=='empty': result=output({})
    if kind=='unselected': result=output({'profile':[0]},event={'message_indices':[0]})
    llm=Fake(result)
    with pytest.raises(ValueError): MultiLabelJudge(MEMORY_TYPE_CONFIG,llm).judge([dict(content='test')])
    assert llm.calls==3

def test_multilabel_recovers_and_binds_source_ids(monkeypatch):
    from memory.provenance import source_id
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    class Flaky(Fake):
        def complete(self,*args):
            self.calls+=1
            return output({'profile':[99]}) if self.calls==1 else output({'profile':[0,1],'event':[1]})
    llm=Flaky(None)
    payload=AMLAdd(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='assistant',content='你住杭州吗？'),dict(role='user',content='对，下周去上海出差。')])
    result=MemoryTypeSelector(llm).select(payload)
    assert result.label=='valuable' and llm.calls==2
    assert result.classification.labels==['profile','event']
    assert result.routes[0].source_ids==[source_id('u','r',0),source_id('u','r',1)]

@pytest.mark.parametrize('indices',[[True],['0'],[0.5]])
def test_indices_are_strict_integers(indices):
    with pytest.raises(ValueError): output({'profile':indices})

@pytest.mark.parametrize('labels',[{'profile':[0],'event':[0]},{'vector_only':[0]}])
def test_multilabel_api_routes_transactionally(tmp_path,labels):
    class Embedder:
        identity='type-test'
        def documents(self,texts): return np.ones((len(texts),3))
    llm=Fake(output(labels))
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedder(),write_selector=MemoryTypeSelector(llm))
    request=dict(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content='我住杭州，下周去上海出差。')])
    with TestClient(create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test'),backend=backend)) as client:
        for _ in range(2):
            assert client.post('/add',json=request,headers={'Authorization':'Bearer test'}).status_code==200
    assert llm.calls==1
    with closing(backend.connect()) as db:
        routes=db.execute('SELECT * FROM rag_memory_routes').fetchall()
        assert {r['memory_type'] for r in routes}==set(labels)-{'vector_only'}
        assert all(json.loads(r['message_indices'])==[0] for r in routes)
        decision=json.loads(db.execute('SELECT result FROM rag_write_decisions').fetchone()[0])
        assert decision['version']=='memory-types-v2'
        assert db.execute('SELECT count(*) FROM rag_memory_queue').fetchone()[0]==int('vector_only' not in labels)

def test_enabled_gate_uses_multilabel_default(tmp_path):
    class Embedder:
        identity='default-test'
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_WRITE_GATE_MODE='on'),Embedder())
    assert isinstance(backend.write_selector,MemoryTypeSelector)


def test_failed_route_insert_rolls_back_everything(tmp_path):
    class Embedder:
        identity='rollback-test'
        def documents(self,texts): return np.ones((len(texts),3))
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedder(),
        write_selector=MemoryTypeSelector(Fake(output({'profile':[0],'event':[0]}))))
    with closing(backend.connect()) as db,db:
        db.execute("CREATE TRIGGER fail_route BEFORE INSERT ON rag_memory_routes BEGIN SELECT RAISE(ABORT, 'route failure'); END")
    import sqlite3
    with pytest.raises(sqlite3.IntegrityError):
        backend.add(AMLAdd(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content='test')]))
    with closing(backend.connect()) as db:
        for table in ['rag_memory_routes','rag_memory_queue','rag_memories','rag_requests','rag_write_decisions','rag_sources']:
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0


def test_label_order_is_normalized_without_retry():
    result=output({'profile':[0],'event':[0]})
    result.assessments.reverse()
    llm=Fake(result)
    classified=MultiLabelJudge(MEMORY_TYPE_CONFIG,llm).judge([dict(content='text')])
    assert classified.labels==['profile','event']
    assert llm.calls==1

@pytest.mark.parametrize('selected,contexts,combined',[
    ([0],[],[0]),([1],[0],[0,1]),([3],[1,2],[1,2,3]),
    ([2,3],[0,1],[0,1,2,3]),([4,1],[0,2,3],[0,1,2,3,4]),
])
def test_context_windows_keep_direct_evidence_separate(selected,contexts,combined):
    from memory.write_gate import bind_route
    from memory.provenance import source_id
    payload=AMLAdd(user_id='u',session_id='s',request_id='r',messages=[
        dict(role='user',content=str(i)) for i in range(5)])
    route=bind_route(payload,'profile',selected)
    assert route.message_indices==sorted(selected)
    assert route.context_indices==contexts
    assert route.context_source_ids==[source_id('u','r',i) for i in contexts]
    assert [m['message_index'] for m in route.builder_messages]==combined
    assert [m['message_index'] for m in route.builder_messages if m['source_kind']=='evidence']==sorted(selected)
    assert all(m['content']==str(m['message_index']) for m in route.builder_messages)


def test_context_is_persisted_and_never_crosses_requests(tmp_path):
    class Embedder:
        identity='context-test'
        def documents(self,texts): return np.ones((len(texts),3))
    llm=Fake(output({'profile':[1]}))
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db')),Embedder(),write_selector=MemoryTypeSelector(llm))
    request=AMLAdd(user_id='u',session_id='s',request_id='r',messages=[
        dict(role='assistant',content='你住杭州吗？'),dict(role='user',content='是的。')])
    backend.add(request)
    with closing(backend.connect()) as db:
        route=db.execute('SELECT * FROM rag_memory_routes').fetchone()
        assert json.loads(route['message_indices'])==[1]
        assert json.loads(route['context_indices'])==[0]
        messages=json.loads(route['builder_messages'])
        assert [(m['content'],m['source_kind']) for m in messages]==[('你住杭州吗？','context'),('是的。','evidence')]
        for m in messages:
            assert db.execute('SELECT 1 FROM rag_sources WHERE source_id=?',(m['source_id'],)).fetchone()
    llm.result=output({'profile':[0]})
    # Even in the same session, only this request's preceding messages are used.
    backend.add(AMLAdd(user_id='u',session_id='s',request_id='new',messages=[dict(role='user',content='新消息')]))
    with closing(backend.connect()) as db:
        route=db.execute("SELECT * FROM rag_memory_routes WHERE request_id='new'").fetchone()
        assert json.loads(route['context_indices'])==[]
        assert len(json.loads(route['builder_messages']))==1


def test_old_route_schema_migrates_without_erasing_rows(tmp_path):
    import sqlite3
    class Embedder:
        identity='migration-test'
    path=tmp_path/'db'
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(path)),Embedder())
    with closing(backend.connect()) as db,db:
        db.execute('DROP TABLE rag_memory_routes')
        db.execute('CREATE TABLE rag_memory_routes(user_id TEXT,request_id TEXT,memory_type TEXT,message_indices TEXT,source_ids TEXT,status TEXT,PRIMARY KEY(user_id,request_id,memory_type))')
        db.execute("INSERT INTO rag_memory_routes VALUES ('u','old','profile','[0]','[\"old-id\"]','pending')")
    reopened=VanillaMemory(dict(RAG_MEMORY_DB=str(path)),Embedder())
    with closing(reopened.connect()) as db:
        row=db.execute('SELECT * FROM rag_memory_routes').fetchone()
        assert row['request_id']=='old' and row['source_ids']=='["old-id"]'
        assert row['builder_messages']=='[]' and row['context_indices']=='[]'
