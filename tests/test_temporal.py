import sqlite3
from types import SimpleNamespace
from datetime import datetime, timezone
from memory.temporal import TemporalRanker, TemporalNormalizer
from memory.temporal_index import TemporalIndex


def rows():
    return [dict(timestamp=9999999999999, time_mentions=[dict(start=t, precision='day')])
            for t in ['2024-01-01','2024-03-01','2023-01-01','2022-01-01']]


def test_no_message_time_fallback():
    rs=[dict(timestamp=t) for t in (100,200,300)]
    assert TemporalRanker.order('latest trip',rs,[0,1,2],[.8,.79,.1])==[0,1,2]


def test_tie_and_strong_relevance():
    assert TemporalRanker.order('latest trip',rows(),[0,1,2,3],[.8,.79,.2,.1])[:2]==[1,0]
    assert TemporalRanker.order('latest trip',rows(),[0,1,2,3],[1,.2,.1,0])==[0,1,2,3]


def test_ordered_queries_and_unknown_are_preserved():
    assert TemporalRanker.operator('order from earliest to latest')=='ordered'
    assert TemporalRanker.order('order from earliest to latest',rows(),[0,1,2,3],[.8,.79,.2,.1])==[0,1,2,3]
    rs=rows(); rs[1]['time_mentions']=[]
    assert TemporalRanker.order('latest trip',rs,[0,1,2,3],[.8,.79,.2,.1])==[0,1,2,3]
    assert TemporalRanker.order('my hobby',rows(),[0,1,2,3],[.8,.79,.2,.1])==[0,1,2,3]


def test_position_bound_and_sqlite_rows():
    rs=[dict(time_mentions=[dict(start=f'2024-01-{i+1:02}',precision='day')]) for i in range(20)]
    result=TemporalRanker.order('latest trip',rs,list(range(20)),[1-i*.0001 for i in range(19)]+[0])
    assert all(abs(i-p)<=2 for p,i in enumerate(result))
    db=sqlite3.connect(':memory:'); db.row_factory=sqlite3.Row
    native=db.execute('select 100 as timestamp').fetchall()
    assert TemporalRanker.order('latest trip',native,[0],[1])==[0]


def test_normalization_is_conservative():
    stamp=int(datetime(2024,3,1,tzinfo=timezone.utc).timestamp()*1000)
    assert TemporalNormalizer.normalize('yesterday',stamp)['start']=='2024-02-29'
    assert TemporalNormalizer.normalize('last month',stamp)['end']=='2024-02-29'
    assert TemporalNormalizer.normalize('2024-02-30',stamp)['precision']=='unknown'
    assert TemporalNormalizer.normalize('May 15',stamp)['precision']=='unknown'
    assert TemporalNormalizer.normalize('yesterday')['precision']=='unknown'
    assert TemporalNormalizer.normalize('May 15, 2020')['start']=='2020-05-15'


def test_index_budget_idempotence_and_failure(tmp_path):
    def connect():
        db=sqlite3.connect(tmp_path/'index.db'); db.row_factory=sqlite3.Row; return db
    class Fake:
        calls=0
        def annotate(self, sources):
            self.calls+=1
            raise ValueError('bad output')
    fake=Fake(); index=TemporalIndex(connect,annotator=fake)
    payload=SimpleNamespace(user_id='u',request_id='r',messages=[SimpleNamespace(content='On 2024-02-01 I travelled.',timestamp=None)]*5)
    index.write(payload); index.write(payload)
    assert fake.calls==2
    with connect() as db:
        assert db.execute('select count(*) from rag_time_mentions').fetchone()[0]==5
        assert db.execute("select count(*) from rag_time_mentions where status='budget_skipped'").fetchone()[0]==3
    assert index.enrich('other',[dict(source_id='s',char_start=0,char_end=100)])[0]['time_mentions']==[]


def test_add_search_time_index_preserves_original_evidence(tmp_path):
    import numpy as np
    from fastapi.testclient import TestClient
    from memory.aml_api import create_app
    from memory.vanilla import VanillaMemory
    class Embedder:
        identity='temporal-integration-test'
        def documents(self, texts): return np.tile([1.,0.], (len(texts),1))
        queries=documents
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'mem.db'),RAG_TEMPORAL_MODE='on',RAG_RESULT_WINDOW='0',
             AML_AUTH_MODE='bearer',AML_API_KEY='test')
    memory=VanillaMemory(cfg,Embedder())
    content='On 2024-02-01 I visited Berlin.'
    body=dict(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content=content)])
    with TestClient(create_app(settings=cfg,backend=memory)) as client:
        client.headers['Authorization']='Bearer test'
        assert client.post('/add',json=body).status_code==200
        assert client.post('/add',json=body).status_code==200
        result=client.post('/search',json=dict(user_id='u',query='When did I visit Berlin?',top_k=5))
        assert result.status_code==200
        assert result.json()['data'][0]['content']==content
        assert client.post('/search',json=dict(user_id='other',query='When did I visit Berlin?',top_k=5)).json()['data']==[]
    with memory.connect() as db:
        import json
        row=db.execute('select * from rag_time_mentions').fetchall()
        assert len(row)==1
        mention=json.loads(row[0]['mentions'])[0]
        assert content[mention['char_start']:mention['char_end']]=='2024-02-01'
        assert mention['start']=='2024-02-01'
