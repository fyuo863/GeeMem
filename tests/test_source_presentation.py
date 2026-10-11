from copy import deepcopy
from datetime import datetime
import json

import pytest
from fastapi.testclient import TestClient

from memory.source_presentation import present_sources
from memory.aml_api import create_app
from memory.multihop import MultiHop
from memory.vanilla import VanillaMemory
from test_vanilla import Embedder


@pytest.mark.parametrize('timestamp', ['2023-01-20T16:04:00Z','2023-01-21T00:04:00+08:00'])
def test_time_is_visible_without_rewriting_or_converting_source(timestamp):
    hit=dict(id='original',content='Lost my job as a banker yesterday.',score=.9,created_at=timestamp)
    before=deepcopy(hit);trace={}
    result=present_sources({'data':[hit]},[],trace)['data'][0]
    assert timestamp in result['content']
    assert result['content'].split('\n',1)[1]==hit['content']
    assert 'not event time' in result['content']
    assert all(result[k]==hit[k] for k in ['id','score','created_at'])
    assert hit==before and trace['source_presentation']['anchored_source_ids']==['original']


@pytest.mark.parametrize('timestamp',[None,'bad-date','2023-01-20','2023-01-20T16:04:00',0,
                                    '1970-01-01T00:00:00Z','1970-01-01T08:00:00+08:00'])
def test_unknown_or_timezone_free_timestamp_is_not_guessed(timestamp):
    result={'data':[dict(id='a',content='Yesterday.',score=1,created_at=timestamp)]}
    assert present_sources(result,[],{})==result


def test_bundle_dates_and_oversized_single_source_are_preserved():
    original='Yesterday. '*1000
    result={'data':[dict(id='bundle',content='[source_time="2023-01-20"] A\n[source_time="2024-01-02"] B',score=2),
                    dict(id='long',content=original,score=1,created_at='2023-01-20T16:04:00Z')]}
    before=deepcopy(result)
    out=present_sources(result,[dict(id='bundle',source_ids=['a','b'])],{})
    assert out['data'][0]==result['data'][0]
    assert out['data'][1]['content'].split('\n',1)[1]==original
    assert result==before


@pytest.mark.parametrize('planning_failure',[False,True])
def test_api_single_source_time_survives_atomic_and_planner_timeout(tmp_path,planning_failure):
    embedder=Embedder()
    backend=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'test.db'),RAG_RESULT_WINDOW='0',
        RAG_SEARCH_AUDIT_PATH=str(tmp_path/'audit.jsonl')),embedder)
    if planning_failure:
        class TimeoutPlanner:
            def route(self,*args):raise TimeoutError('route timeout')
        backend.search_service.multihop=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw'),TimeoutPlanner())
    original='Lost my job as a banker yesterday.'
    timestamp=int(datetime.fromisoformat('2023-01-20T16:04:00+00:00').timestamp()*1000)
    with TestClient(create_app(settings={'AML_API_KEY':'test'},backend=backend)) as client:
        headers={'Authorization':'Bearer test'}
        response=client.post('/add',headers=headers,json=dict(user_id='u',session_id='s',request_id='r',
            messages=[dict(role='user',content=original,timestamp=timestamp)]))
        assert response.status_code==200
        response=client.post('/search',headers=headers,json=dict(user_id='u',query='When was the job lost?',top_k=1))
        assert response.status_code==200
        hit=response.json()['data'][0]
        assert '2023-01-20T16:04:00Z' in hit['content'] and original in hit['content']
        assert set(hit)=={'id','content','score','created_at'}
        assert client.post('/search',headers=headers,json=dict(user_id='other',query='job',top_k=1)).json()['data']==[]
    with backend.connect() as db:
        assert db.execute('SELECT content FROM rag_memories').fetchone()[0]==original
    assert all('source_time=' not in q for q in embedder.query_texts)
    event=json.loads((tmp_path/'audit.jsonl').read_text(encoding='utf8').splitlines()[0])
    assert event['trace']['source_presentation']['anchored_source_ids']==[hit['id']]
    assert event['response']['data'][0]==hit
