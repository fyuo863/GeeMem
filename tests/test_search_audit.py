import json
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
import pytest

from memory.aml_api import AMLAdd, AMLSearch
from memory.vanilla import VanillaMemory


class Embedder:
    identity = 'audit-test'

    def documents(self, texts):
        return np.ones((len(texts), 3))

    queries = documents


def make_store(tmp_path):
    return VanillaMemory({'RAG_MEMORY_DB': str(tmp_path/'test.db'),
                          'RAG_SEARCH_AUDIT_PATH': str(tmp_path/'audit.jsonl'),
                          'RAG_POSITION_LOG_PATH': str(tmp_path/'positions.jsonl'),
                          'RAG_RESULT_WINDOW': '0'}, Embedder())


def events(tmp_path):
    return [json.loads(s) for s in (tmp_path/'audit.jsonl').read_text(encoding='utf8').splitlines()]


def test_real_storage_search_audit_and_parallel_isolation(tmp_path):
    store = make_store(tmp_path)
    store.add(AMLAdd(user_id='u', session_id='s', request_id='r',
                     messages=[{'role':'user', 'content':'小林喜欢足球。'}]))
    def search(i):
        return store.search(AMLSearch(user_id='u', query=f'问题{i}', top_k=1))
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(search, range(8)))
    rows = events(tmp_path)
    assert len(rows) == len({e['trace_id'] for e in rows}) == 8
    assert {e['request']['query'] for e in rows} == {f'问题{i}' for i in range(8)}
    assert all(e['response'] == responses[0] and e['status'] == 'success' for e in rows)
    assert all(e['trace']['retrieval_queries'][0]['query'] == e['request']['query'] for e in rows)
    positions = [json.loads(s) for s in (tmp_path/'positions.jsonl').read_text(encoding='utf8').splitlines()]
    assert {e['trace_id'] for e in positions} == {e['trace_id'] for e in rows}


def test_failed_search_keeps_status_not_credentials(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    def fail(*args, **kwargs):
        response = httpx.Response(502, request=httpx.Request('POST', 'https://host/?key=SECRET'))
        raise httpx.HTTPStatusError('Bearer SECRET', request=response.request, response=response)
    monkeypatch.setattr(store, '_search_request', fail)
    with pytest.raises(httpx.HTTPStatusError):
        store.search(AMLSearch(user_id='u', query='失败问题', top_k=10))
    row = events(tmp_path)[0]
    assert row['error']['http_status'] == 502
    assert row['status'] == 'error' and row['response'] is None
    assert 'SECRET' not in (tmp_path/'audit.jsonl').read_text(encoding='utf8')


def test_audit_keeps_review_and_fallback_and_public_bundle(tmp_path, monkeypatch):
    store = make_store(tmp_path)
    response = {'data':[{'id':'bundle', 'content':'[a] 原文甲\n[b] 原文乙', 'score':1.0}]}
    def run(payload, trace):
        trace.update(initial_plan={'strategy':'chain'},
                     audit_review_inputs=[{'round':0,'evidence':[{'id':'a','content':'原文甲'}]}],
                     rounds=[{'round':0,'review':{'sufficient':False,'missing':'缺少老师信息'}}],
                     bundle_sources={'bundle':['a','b']}, fallback=True, error_type='LLMError',
                     error_phase='review', cause_type='HTTPStatusError', http_status=502)
        return response
    monkeypatch.setattr(store, '_search_request', run)
    assert store.search(AMLSearch(user_id='u', query='兄弟的老师是谁', top_k=10)) == response
    row = events(tmp_path)[0]
    assert row['trace']['http_status'] == 502
    assert row['trace']['audit_review_inputs'][0]['evidence'][0]['content'] == '原文甲'
    assert row['trace']['rounds'][0]['review']['missing'] == '缺少老师信息'
    assert row['trace']['final_source_positions']['b'][0]['result_id'] == 'bundle'
    assert row['response'] == response


def test_audit_disk_failure_does_not_break_search(tmp_path, monkeypatch, caplog):
    store = make_store(tmp_path)
    def fail(*args):
        raise OSError('SECRET path')
    monkeypatch.setattr(store.search_audit.handler, '_open', fail)
    assert store.search(AMLSearch(user_id='u', query='PRIVATE QUERY', top_k=10)) == {'data': []}
    assert 'search_audit_write_failed' in caplog.text
    assert 'PRIVATE' not in caplog.text and 'SECRET' not in caplog.text


def test_audit_disabled_by_default_and_rotation(tmp_path):
    from memory.search_audit import SearchAudit
    assert not SearchAudit({}).enabled
    sink = SearchAudit({'RAG_SEARCH_AUDIT_PATH':str(tmp_path/'rotate.jsonl'),
                        'RAG_SEARCH_AUDIT_MAX_BYTES':'100', 'RAG_SEARCH_AUDIT_BACKUPS':'2'})
    for i in range(6):
        sink.write(AMLSearch(user_id='u',query=str(i),top_k=10), {'position_trace_id':str(i)},
                   {'data':[]}, None, 'start', 1)
    sink.handler.close()
    assert len(list(tmp_path.glob('rotate.jsonl*'))) == 3
    assert json.loads((tmp_path/'rotate.jsonl').read_text(encoding='utf8'))['request']['query'] == '5'


def test_multihop_audit_records_actual_review_packets(tmp_path, monkeypatch):
    from test_multihop import ChainPlanner, request, hit
    from memory.multihop import MultiHop
    from memory.search_service import SearchService
    planner = ChainPlanner()
    def retrieve(p):
        if p.query == 'Who is Alice married to?':
            return {'data':[hit('spouse','Alice is married to Bob.')]}
        if p.query == 'Where does Bob work?':
            return {'data':[hit('employer','Bob works at Atlas.')]}
        return {'data':[hit('noise','Alice likes hiking.')]}
    class Ranker:
        def score(self, q, docs): return [1.0]*len(docs)
    service = SearchService(retrieve, MultiHop({'RAG_MULTIHOP_MODE':'llm'}, planner), Ranker())
    store = make_store(tmp_path)
    monkeypatch.setattr(store, '_search_request', service.search)
    result = store.search(request(k=2))
    event = events(tmp_path)[0]
    trace = event['trace']
    assert trace['initial_plan']['strategy'] == 'chain'
    assert [s['evidence'] for s in trace['audit_review_inputs']] == planner.inputs
    assert trace['audit_review_inputs'][-1]['model_output']['sufficient'] is True
    assert trace['rounds'][-1]['review']['sufficient'] is True
    assert trace['retrieval_queries'][-1]['query'] == 'Where does Bob work?'
    assert {h['id'] for h in result['data']} == {'spouse','employer'}
    assert event['response'] == result
