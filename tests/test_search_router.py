from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import pytest
from memory.search_service import SearchService
from memory.partition_search import PartitionSearch

@pytest.mark.parametrize('strategy', ['direct','split','chain'])
def test_every_query_uses_scoped_atomic_retrieval(strategy):
    seen=[]
    barrier=Barrier(2)
    class Selector:
        def judge(self,messages):
            q=messages[0]['content']
            if strategy=='split' and q in ('brother','team'): barrier.wait(timeout=3)
            label='relationship' if q=='brother' else 'profile'
            return SimpleNamespace(labels=[label],model_dump=lambda:{'labels':[label]})
    def atomic(p):
        assert p.user_id=='u' and p.options==['option'] and p.top_k==5
        assert p.dual_channel
        seen.append((p.query,p.memory_types))
        return {'data':[{'id':p.query,'content':p.query,'score':1}]}
    class Planner:
        mode='llm'
        def run(self,p,retrieve,rerank,trace):
            trace.update(strategy=strategy,mode='llm')
            retrieve(p)
            if strategy=='split':
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(lambda q:retrieve(SimpleNamespace(**{**vars(p),'query':q})),['brother','team']))
            elif strategy=='chain':
                for q in ['brother','team']:
                    retrieve(SimpleNamespace(**{**vars(p),'query':q}))
            trace['stop']='sufficient'
            return {'data':[]}
    trace={}
    service=SearchService(atomic,Planner(),partition_search=PartitionSearch({'RAG_PARTITION_MODE':'dual'},Selector()))
    service.search(SimpleNamespace(query='original',user_id='u',options=['option'],top_k=5),trace)
    assert trace['strategy']==strategy and trace['mode']=='llm'
    assert len(trace['partition_queries'])==(1 if strategy=='direct' else 3)
    if strategy!='direct':
        assert ('brother',('relationship',)) in seen
        assert ('team',('profile',)) in seen

def test_failed_scope_selection_uses_global_inside_planning():
    class Selector:
        def judge(self,*a): raise ValueError('private detail')
    def atomic(p):
        assert p.memory_types==() and p.fallback and not p.dual_channel
        return {'data':['baseline']}
    class Planner:
        mode='llm'
        def run(self,p,retrieve,*args): return retrieve(p)
    trace={}
    service=SearchService(atomic,Planner(),partition_search=PartitionSearch({'RAG_PARTITION_MODE':'dual'},Selector()))
    assert service.search(SimpleNamespace(query='q',user_id='u',top_k=2),trace)=={'data':['baseline']}
    assert trace['partition_queries'][0]['routing_error']=='ValueError'
    assert 'private' not in str(trace)
