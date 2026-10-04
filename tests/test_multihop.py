from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
import json

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from memory.aml_api import AMLAdd, AMLSearch, create_app
from memory.llm import LLMError
from memory.multihop import MultiHop, Planner, Route, Review, Query, Support
from memory.vanilla import VanillaMemory
from test_vanilla import Embedder


def request(query='Where does the spouse of Alice work?', k=3):
    return AMLSearch(user_id='u',query=query,top_k=k)


def hit(uid,text,score=1):
    return dict(id=uid,content=text,score=score)


class ChainPlanner:
    def __init__(self): self.inputs=[]
    def route(self,q,options,timeout):
        return Route(strategy='chain',queries=[Query(query='Who is Alice married to?',source_id='__question__',bridge='Alice')])
    def review(self,q,options,strategy,evidence,history,timeout):
        self.inputs.append(evidence)
        supports=[Support(source_id=e['id'],quote=e['content'],needed_for='relationship')
                  for e in evidence if e['id'] in ('spouse','employer')]
        done=any(e['id']=='employer' for e in evidence)
        return Review(sufficient=done,supports=supports,missing='' if done else 'Employer of Bob',
            queries=[] if done else [Query(query='Where does Bob work?',source_id='spouse',bridge='Bob')])


def test_chain_uses_discovered_entity_and_keeps_intermediate_raw_evidence():
    calls=[]
    def retrieve(p):
        calls.append(p)
        if p.query=='Who is Alice married to?': return {'data':[hit('spouse','Alice is married to Bob.')]}
        if p.query=='Where does Bob work?': return {'data':[hit('employer','Bob works at Atlas.')]}
        return {'data':[hit('noise','Alice likes hiking.',8)]}
    planner=ChainPlanner(); trace={}
    out=MultiHop({'RAG_MULTIHOP_MODE':'llm'},planner).run(request(k=2),retrieve,
        lambda q,ds: [9 if 'hiking' in d else -8 for d in ds],trace)['data']
    assert {h['id'] for h in out}=={'spouse','employer'}
    assert [p.query for p in calls]==[request().query,'Who is Alice married to?','Where does Bob work?']
    assert all(p.user_id=='u' for p in calls)
    assert trace['stop']=='sufficient' and trace['supports_fit']
    assert out[0]['score']>=out[1]['score']
    assert all(h['content'] in ('Alice is married to Bob.','Bob works at Atlas.') for h in out)


def test_independent_queries_execute_concurrently():
    barrier=Barrier(2,timeout=5)
    class Split:
        def route(self,q,options,timeout):
            return Route(strategy='split',queries=[Query(query=f'Price of {x}?',source_id='__question__',bridge=x) for x in ('bike','tent')])
        def review(self,*args):
            return Review(sufficient=True,supports=[Support(source_id=x,quote=f'{x} costs $50.',needed_for='price') for x in ('bike','tent')],missing='',queries=[])
    def retrieve(p):
        if p.query.startswith('Price of'):
            barrier.wait()
            x='bike' if 'bike' in p.query else 'tent'
            return {'data':[hit(x,f'{x} costs $50.')]}
        return {'data':[]}
    trace={}
    out=MultiHop({'RAG_MULTIHOP_MODE':'llm'},Split()).run(request('Total price of bike and tent?',2),retrieve,lambda q,d:[1]*len(d),trace)
    assert len(out['data'])==2 and trace['search_calls']==3


def test_invalid_bridges_quotes_and_duplicate_queries_are_rejected():
    class Bad:
        def route(self,q,*args):
            return Route(strategy='chain',queries=[Query(query='Secret employer',source_id='other-user',bridge='Secret')])
        def review(self,*args):
            return Review(sufficient=True,supports=[Support(source_id='a',quote='invented fact',needed_for='answer')],missing='',
                queries=[Query(query=request().query.upper(),source_id='__question__',bridge='Alice')])
    trace={}
    out=MultiHop({'RAG_MULTIHOP_MODE':'llm'},Bad()).run(request(),lambda p:{'data':[hit('a','Alice lives in Paris.')]},lambda q,d:[1]*len(d),trace)
    assert trace['rejected_queries']==2 and trace['rejected_supports']==1
    assert trace['stop']=='no_grounded_new_query' and trace['search_calls']==1
    assert out['data'][0]['content']=='Alice lives in Paris.'


@pytest.mark.parametrize('stage',['route','review'])
def test_model_failure_returns_exact_baseline(stage):
    class Failed(ChainPlanner):
        def route(self,*args):
            if stage=='route': raise LLMError('unavailable')
            return super().route(*args)
        def review(self,*args): raise LLMError('unavailable')
    base=[hit('a','original evidence',7)]
    trace={}
    out=MultiHop({'RAG_MULTIHOP_MODE':'llm'},Failed()).run(request(),lambda p:{'data':base},lambda q,d:[1]*len(d),trace)
    assert out=={'data':base} and trace['fallback']


def test_query_budget_and_topk_are_enforced_even_if_not_sufficient():
    trace={}
    runner=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_QUERIES':'2'},ChainPlanner())
    def retrieve(p):
        return {'data':[hit('spouse','Alice is married to Bob.')]}
    out=runner.run(request(k=1),retrieve,lambda q,d:[1]*len(d),trace)
    assert trace['search_calls']==2 and trace['stop']=='query_budget' and len(out['data'])==1


def test_api_scope_add_does_not_call_llm_and_off_is_identical(tmp_path):
    class Direct:
        def __init__(self):self.seen=[]
        def route(self,*args): return Route(strategy='direct')
        def review(self,q,options,strategy,evidence,history,timeout):
            self.seen.extend(evidence)
            e=evidence[0]
            return Review(sufficient=True,supports=[Support(source_id=e['id'],quote=e['content'],needed_for='answer')],missing='')
    class Scorer:
        def score(self,q,docs): return np.ones(len(docs))
    planner=Direct()
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_MULTIHOP_MODE='llm',AML_AUTH_MODE='bearer',AML_API_KEY='test')
    store=VanillaMemory(cfg,Embedder(),reranker=Scorer(),multihop_planner=planner)
    with TestClient(create_app(settings=cfg,backend=store)) as c:
        c.headers['Authorization']='Bearer test'
        for user,text in [('u','Alice lives in Paris.'),('other','Hidden sensitive message.')]:
            assert c.post('/add',json=dict(user_id=user,request_id='r',session_id='s',messages=[dict(role='user',content=text)])).status_code==200
        assert not planner.seen
        response=c.post('/search',json=request('Where does Alice live?',1).model_dump())
        assert response.status_code==200
        assert set(response.json())=={'data'}
        assert all('Hidden' not in e['content'] for e in planner.seen)
        assert set(response.json()['data'][0])=={'id','content','score'}
    store.multihop.mode='off'
    assert store.search(request())==store._search_direct(request())


def test_planner_uses_passed_file_settings_and_structured_output(monkeypatch):
    captured={}
    class Client:
        def __init__(self,**kw): captured['client']=kw
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def post(self,url,**kw):
            captured['request']=kw
            return httpx.Response(200,request=httpx.Request('POST',url),json={'choices':[{'message':{'content':'{"strategy":"direct","queries":[]}'}}]})
    monkeypatch.setenv('LLM_API_KEY','environment-secret-must-not-be-read')
    monkeypatch.setattr(httpx,'Client',Client)
    p=Planner({'LLM_MODEL':'gpt-4o-mini','LLM_API_KEY':'file-value','LLM_PROXY':'http://127.0.0.1:7897'})
    assert p.route('question',None,2).strategy=='direct'
    assert captured['client']['trust_env'] is False
    assert captured['request']['headers']['Authorization']=='Bearer file-value'
    assert captured['request']['json']['response_format']['json_schema']['strict']
    with pytest.raises(ValueError):Planner({'LLM_MODEL':'different-model'})


def test_round_budget_and_truncated_evidence_never_admit_unseen_quotes():
    class ReviewTail:
        def route(self,*args): return Route(strategy='direct')
        def review(self,q,options,strategy,evidence,history,timeout):
            assert 'SecretTail' not in evidence[0]['content']
            return Review(sufficient=True,supports=[Support(source_id='a',quote='SecretTail',needed_for='answer')],
                missing='',queries=[Query(query='Alice employer?',source_id='__question__',bridge='Alice')])
    trace={}
    runner=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_ROUNDS':'1','RAG_MULTIHOP_EVIDENCE_CHARS':'200'},ReviewTail())
    runner.run(request(),lambda p:{'data':[hit('a','Alice '+('x'*300)+' SecretTail')]},lambda q,d:[1]*len(d),trace)
    assert trace['rejected_supports']==1 and trace['stop']=='round_budget'


def test_topk_cannot_falsely_claim_full_chain_fits():
    planner=ChainPlanner();trace={}
    def retrieve(p):
        return {'data':[hit('spouse','Alice is married to Bob.'),hit('employer','Bob works at Atlas.')]}
    out=MultiHop({'RAG_MULTIHOP_MODE':'llm'},planner).run(request(k=1),retrieve,lambda q,d:[1]*len(d),trace)
    assert len(out['data'])==1 and not trace['supports_fit']


def test_options_and_scope_are_preserved_for_every_subquery():
    calls=[];trace={}
    p=request().model_copy(update={'options':['Atlas','Harbor']})
    def retrieve(q):
        calls.append(q)
        return {'data':[hit('spouse','Alice is married to Bob.')]}
    MultiHop({'RAG_MULTIHOP_MODE':'llm'},ChainPlanner()).run(p,retrieve,lambda q,d:[1]*len(d),trace)
    assert all(q.user_id==p.user_id and q.options==p.options for q in calls)
    assert len({normalized_q.query.casefold() for normalized_q in calls})==len(calls)


def test_time_budget_stops_planning_but_preserves_baseline():
    import time
    class Slow:
        def route(self,*args): return Route(strategy='direct')
        def review(self,*args): raise AssertionError('No review after budget')
    runner=MultiHop({'RAG_MULTIHOP_MODE':'llm'},Slow())
    runner.seconds=0.01
    def retrieve(p):
        time.sleep(0.02)
        return {'data':[hit('a','Alice lives in Paris.')]}
    trace={}
    out=runner.run(request(),retrieve,lambda q,d:[1]*len(d),trace)
    assert trace['stop']=='time_budget' and out['data'][0]['content']=='Alice lives in Paris.'


def test_long_bridge_can_only_shorten_to_a_name_already_in_the_cited_phrase():
    class LongBridge(ChainPlanner):
        def route(self,*args):
            return Route(strategy='chain',queries=[Query(query='Who is Alice married to?',source_id='__question__',bridge='spouse of Alice')])
    trace={}
    MultiHop({'RAG_MULTIHOP_MODE':'llm'},LongBridge()).run(request(),
        lambda p:{'data':[hit('spouse','Alice is married to Bob.')]},lambda q,d:[1]*len(d),trace)
    assert trace['shortened_bridges']==1
    assert trace['rounds'][0]['queries'][0]['bridge']=='Alice'
