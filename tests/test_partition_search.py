from types import SimpleNamespace
from fastapi.testclient import TestClient
from memory.aml_api import create_app
from memory.partition_search import merge_candidates, PartitionSearch
from test_shared_typed_retrieval import setup


class Selector:
    def judge(self, messages):
        assert messages[0]['content']=='football'
        return SimpleNamespace(labels=['profile'],model_dump=lambda:{'labels':['profile']})


def test_budget_and_partition_rescue():
    assert merge_candidates(list(range(20)),{15,16,17},6)==[0,1,2,15,16,17]
    assert merge_candidates(list(range(20)),set(),6)==list(range(6))
    assert merge_candidates(list(range(3)),{1},6)==[0,1,2]


def test_http_dual_reuses_embedding_and_one_rerank(tmp_path):
    b,r=setup(tmp_path)
    b.rerank_candidates=2
    b.search_service.partition_search=PartitionSearch({'RAG_PARTITION_MODE':'dual'},Selector())
    calls=[]
    original=b.embedder.queries
    b.embedder.queries=lambda texts:(calls.append(texts) or original(texts))
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'t'},backend=b)) as client:
        response=client.post('/search',headers={'Authorization':'Bearer t'},
                             json={'user_id':'u','query':'football','top_k':1})
    assert response.status_code==200
    assert len(calls)==1 and len(r.calls)==1 and len(r.calls[0][1])==2
    assert set(response.json()['data'][0]) <= {'id','content','score','created_at'}


def test_router_failure_falls_back_without_rewriting(tmp_path):
    b,r=setup(tmp_path)
    class Broken:
        def judge(self,messages): raise RuntimeError('not logged')
    b.search_service.partition_search=PartitionSearch({'RAG_PARTITION_MODE':'strict'},Broken())
    trace={}
    p=SimpleNamespace(user_id='u',query='football',top_k=2,options=None)
    result=b.search(p,trace=trace)
    assert len(result['data'])==2 and trace['routing_error']=='RuntimeError'
    assert trace['partitions']==[] and r.calls[0][0]=='football'


def test_strict_route_and_dual_trace(tmp_path):
    b,_=setup(tmp_path)
    p=SimpleNamespace(user_id='u',query='football',top_k=3,options=None)
    route=PartitionSearch({'RAG_PARTITION_MODE':'strict'},Selector())
    b.search_service.partition_search=route
    assert len(b.search(p)['data'])==2
    route.mode='dual'
    trace={}
    assert len(b.search(p,trace=trace)['data'])==3
    assert trace['candidate_count']==3 and trace['typed_candidate_count']==2
