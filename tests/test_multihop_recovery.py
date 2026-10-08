from memory.multihop import MultiHop, Query, Route, Review, Support
from test_multihop import request, hit


def test_chain_recovery_runs_initial_query_variants_in_parallel():
    calls=[]
    class Planner:
        def route(self,*args):
            return Route(strategy='chain', queries=[Query(query='Who is Alice married to?', source_id='__question__', bridge='Alice')])
        def review(self,q,options,strategy,evidence,history,timeout):
            if any(e['id']=='spouse' for e in evidence):
                return Review(sufficient=False, supports=[Support(source_id='spouse',quote='Alice is married to Bob.',needed_for='identity')], missing='Bob employer', queries=[Query(query='Where does Bob work?',source_id='spouse',bridge='Bob')])
            return Review(sufficient=False,supports=[],missing='spouse',queries=[])
    def retrieve(p):
        calls.append(p.query)
        if 'Context question' in p.query:
            return {'data':[hit('spouse','Alice is married to Bob.')]}
        if 'Bob work' in p.query:
            return {'data':[hit('employer','Bob works at Atlas.')]}
        return {'data':[hit('noise','Alice likes hiking.')]} 
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_RECOVERY':'on','RAG_MULTIHOP_QUERIES':'6'},Planner()).run(
        request(k=3),retrieve,lambda q,docs:[1]*len(docs),trace)
    assert any('Context question' in q for q in calls)
    assert trace['recovery_queries'] >= 1
    assert trace['recovery_sources'] >= 1
    assert {'spouse','employer'} <= {h['id'] for h in result['data']}


def test_recovery_keeps_original_baseline_when_all_variants_fail():
    class Planner:
        def route(self,*args):
            return Route(strategy='chain',queries=[Query(query='Who is Alice married to?',source_id='__question__',bridge='Alice')])
        def review(self,*args):
            return Review(sufficient=False,supports=[],missing='spouse',queries=[])
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_RECOVERY':'on'},Planner()).run(
        request(k=2),lambda p:{'data':[hit('base','Alice lives in Paris.')]},lambda q,docs:[1]*len(docs),trace)
    assert result['data'][0]['id']=='base'
    assert trace['fallback'] is False
    assert trace['recovery_queries']>=1
