from datetime import date
from memory.query_time import QueryTime
from memory.temporal_window import window_candidates, apply_temporal_fusion, soft_quota


TARGET = QueryTime('today', date(2023, 3, 31), date(2023, 3, 31))


def test_window_promotes_outside_pool_and_retains_unknown_global():
    rows = [dict(time_mentions=[]) for _ in range(20)]
    rows[18]['time_mentions'] = [dict(start='2023-03-30', end='2023-04-01')]
    pool, trace = window_candidates(rows, list(range(20)), TARGET, 10)
    assert len(pool) == len(set(pool)) == 10
    assert 18 in pool and {0, 1, 2}.issubset(pool)
    assert trace['promoted_indices'] == [18]


def test_no_reference_or_no_overlap_preserves_baseline():
    rows = [dict(timestamp=1680220800000, time_mentions=[]) for _ in range(20)]
    for target in (None, TARGET):
        assert window_candidates(rows, list(range(20)), target, 10)[0] == list(range(10))


def test_invalid_uncertain_dates_and_scope_do_not_leak():
    rows = [dict(time_mentions=[dict(start='2023-03-31', end='2023-03-31')]) for _ in range(10)]
    rows[8]['time_mentions'][0]['confidence'] = .5
    rows[9]['time_mentions'][0]['end'] = 'invalid'
    pool, trace = window_candidates(rows, [8, 9], TARGET, 5)
    assert pool == [8, 9]
    assert trace['matched_count'] == 0


def test_short_pool_and_one_slot_keep_budget():
    rows = [dict(time_mentions=[]) for _ in range(3)]
    assert window_candidates(rows, [0, 1, 2], TARGET, 10)[0] == [0, 1, 2]
    assert window_candidates(rows, [0, 1, 2], TARGET, 1)[0] == [0]


def test_temporal_fusion_and_soft_quota_are_bounded():
    rows = [dict(session_id='s', time_mentions=[]),
            dict(session_id='s', time_mentions=[dict(start='2023-03-31', end='2023-03-31')]),
            dict(session_id='s', time_mentions=[])]
    order, scores, trace = apply_temporal_fusion(rows, [0, 1, 2], [1., .99, .98], TARGET, .1, 1)
    assert order[1] == 1 and trace['matched_count'] == 1
    assert scores[1] > .99
    quota, qtrace = soft_quota(rows, order, TARGET, 2, .5)
    assert quota[0] == 1 and qtrace['window_selected'] == 1


def test_http_window_promotes_evidence_without_timestamp_filter(tmp_path):
    import numpy as np
    from fastapi.testclient import TestClient
    from memory.aml_api import AMLAdd, create_app
    from memory.vanilla import VanillaMemory

    class E:
        identity = 'window-test'
        def documents(self, texts):
            return np.tile([1., 0.], (len(texts), 1))
        queries = documents

    class R:
        seen = []
        def score(self, query, docs):
            self.seen = docs
            return [10. if 'target artist' in d else 1. for d in docs]

    cfg = dict(RAG_MEMORY_DB=str(tmp_path/'test.db'), RAG_TEMPORAL_MODE='on',
               RAG_TEMPORAL_EXPERIMENT='window', RAG_RETRIEVAL_MODE='dense',
               RAG_RERANK_CANDIDATES='5', RAG_RESULT_WINDOW='0',
               AML_AUTH_MODE='bearer', AML_API_KEY='test')
    reranker = R()
    store = VanillaMemory(cfg, E(), reranker=reranker)
    texts = ['Undated discussion '+str(i) for i in range(12)] + [
        'On 2023-03-31 I started listening to target artist.']
    store.add(AMLAdd(user_id='u', request_id='r', session_id='s', messages=[
        dict(role='user', content=t, timestamp=1680652800000) for t in texts]))
    with TestClient(create_app(settings=cfg, backend=store)) as client:
        client.headers['Authorization'] = 'Bearer test'
        query = dict(user_id='u', query='Which artist last Friday?', top_k=3,
                     reference_time=1680652800000)
        result = client.post('/search', json=query)
        assert result.status_code == 200
        assert result.json()['data'][0]['content'] == texts[-1]
        assert len(reranker.seen) == 5
        assert any('Undated discussion' in d for d in reranker.seen)
        # No reference => original semantic pool. The late target cannot enter.
        query.pop('reference_time')
        result = client.post('/search', json=query)
        assert all(h['content'] != texts[-1] for h in result.json()['data'])
        query['user_id'] = 'other'
        assert client.post('/search', json=query).json()['data'] == []
