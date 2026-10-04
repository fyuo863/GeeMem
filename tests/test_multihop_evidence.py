import numpy as np
import pytest

from memory.aml_api import AMLSearch, create_app
from memory.multihop import MultiHop, Route
from memory.multihop_evidence import EvidencePlanner, BoundReview, BoundRequiredReview
from memory.llm import LLMError
from memory.vanilla import VanillaMemory
from fastapi.testclient import TestClient
from test_vanilla import Embedder


def request(k=3):
    return AMLSearch(user_id='u',query='Who is the CEO of the company where the spouse of Nadia works?',top_k=k)


def hit(id, content):
    return dict(id=id,content=content,score=1.)


def support(id, quote):
    return dict(source_id=id,quote=quote,needed_for='required relation')


def trace():
    return dict(repairs=0,repair_errors=[],validation_errors=[],rejected_queries=0,rejected_supports=0)


class Chain:
    def __init__(self): self.inputs=[]
    def complete(self, instruction, payload, schema, timeout):
        self.inputs.append((schema.__name__,payload))
        if schema.__name__=='RequiredRoute':
            return schema.model_validate(dict(strategy='chain',queries=[dict(query='Who is Nadia married to?',source_id='__question__',bridge='Nadia')],
                requirements=[dict(description=d,kind='relation',depends_on=[] if i==0 else [i-1])
                              for i,d in enumerate(('Identify Nadia spouse','Find spouse employer','Find employer CEO'))]))
        ids={e['content']:e['id'] for e in payload['evidence']}
        texts=['Nadia is married to Elias.','Elias works at Solstice Analytics.','The CEO of Solstice Analytics is Ruth Chen.']
        states=[dict(need_id=f'N{i+1}',status='supported' if t in ids else 'missing',
                     supports=[support(ids[t],t)] if t in ids else [],reason='' if t in ids else 'Missing link') for i,t in enumerate(texts)]
        next_index=next((i for i,t in enumerate(texts) if t not in ids),None)
        queries=[]
        if next_index in (1,2):
            previous=texts[next_index-1]
            anchor='Elias' if next_index==1 else 'Solstice Analytics'
            template='What company does {target} work for?' if next_index==1 else 'Who is the CEO of {target}?'
            source=ids[previous]
            if next_index==1 and 'validation_errors' not in payload:
                source='Q0'  # Reproduce the live failure: right entity, wrong origin.
            queries=[dict(need_id=f'N{next_index+1}',source_ref=source,quote=previous,anchor=anchor,template=template)]
        return schema.model_validate(dict(states=states,queries=queries))


def test_repaired_three_link_chain_binds_positions_keeps_scope_and_stays_in_budget():
    calls=[]
    def retrieve(p):
        calls.append(p)
        data={'Who is Nadia married to?':hit('spouse','Nadia is married to Elias.'),
              'What company does Elias work for?':hit('employer','Elias works at Solstice Analytics.'),
              'Who is the CEO of Solstice Analytics?':hit('ceo','The CEO of Solstice Analytics is Ruth Chen.')}
        return {'data':[data.get(p.query,hit('noise','Nadia likes music.'))]}
    cfg=dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on')
    t={};p=Chain()
    result=MultiHop(cfg,p).run(request(),retrieve,lambda q,ds:np.ones(len(ds)),t)
    assert {x['id'] for x in result['data']}=={'spouse','employer','ceo'}
    assert t['stop']=='sufficient' and t['llm_calls']==5 and t['repairs']==1
    assert t['search_calls']==4 and len(t['rounds'])==3
    assert all(x.user_id=='u' for x in calls)
    assert [b['anchor'] for b in t['binding_registry']]==['Elias','Solstice Analytics']
    assert t['binding_registry'][0]['source_id']=='spouse'
    assert 'Nadia is married to Elias.'[t['binding_registry'][0]['start']:t['binding_registry'][0]['end']]=='Elias'
    assert p.inputs[-1][1]['prior_citations'][0]['supports'][0]['source_id'].startswith('E')
    assert all(set(c)=={'need_id','supports'} for c in p.inputs[-1][1]['prior_citations'])


@pytest.mark.parametrize('source,quote,anchor', [('Q0','Nadia is married to Elias.','Elias'),
    ('E99','Nadia is married to Elias.','Elias'), ('E1','Elias','Elias')])
def test_wrong_source_unknown_reference_and_ambiguous_quote_never_bind(source,quote,anchor):
    a=EvidencePlanner(None,True,False,trace(),lambda:False,lambda:1)
    raw=BoundReview.model_validate(dict(sufficient=False,supports=[],missing='employer',queries=[
        dict(source_ref=source,quote=quote,anchor=anchor,template='Where does {target} work?')]))
    sources={'Q0':dict(id='__question__',content=request().query),
             'E1':dict(id='spouse',content='Elias called Elias. Nadia is married to Elias.')}
    result,errors,_=a._validate(raw,sources,[], 'chain',True)
    assert not result.queries and errors and not a.registry


def test_missing_dependency_or_omitted_need_cannot_be_sufficient():
    a=EvidencePlanner(None,True,True,trace(),lambda:False,lambda:1)
    a.requirements=[dict(id='N1',depends_on=[]),dict(id='N2',depends_on=['N1'])]
    raw=BoundRequiredReview.model_validate(dict(states=[dict(need_id='N2',status='supported',
        supports=[support('E1','Elias works at Solstice Analytics.')],reason='company')],queries=[]))
    result,errors,state=a._validate(raw,{'E1':dict(id='employer',content='Elias works at Solstice Analytics.')},[], 'chain',True)
    assert not result.sufficient and errors
    assert all(s['status']=='missing' for s in state)


def test_arithmetic_operands_stop_without_requesting_a_precomputed_answer():
    a=EvidencePlanner(None,True,True,trace(),lambda:False,lambda:1)
    a.requirements=[dict(id='N1',depends_on=[]),dict(id='N2',depends_on=[])]
    sources={'E1':dict(id='a',content='The bike cost $240.'),'E2':dict(id='b',content='The tent cost $180.')}
    raw=BoundRequiredReview.model_validate(dict(states=[dict(need_id=f'N{i}',status='supported',
        supports=[support(f'E{i}',e['content'])],reason='operand known') for i,e in enumerate(sources.values(),1)],queries=[]))
    result,errors,_=a._validate(raw,sources,[], 'split',True)
    assert result.sufficient and not errors and not result.queries


def test_one_repair_only_and_llm_limit_prevents_extra_retrieval():
    calls=[]
    cfg=dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on',RAG_MULTIHOP_LLM_CALLS='3')
    def retrieve(p):
        calls.append(p.query)
        return {'data':[hit('spouse','Nadia is married to Elias.')]}
    t={}
    MultiHop(cfg,Chain()).run(request(),retrieve,lambda q,ds:np.ones(len(ds)),t)
    assert t['repairs']==1 and t['llm_calls']==3 and t['stop']=='llm_budget'
    assert len(calls)==2  # No employer retrieval after the last allowed LLM call.


def test_repair_failure_returns_exact_original_result():
    class Failed(Chain):
        def complete(self, instruction, payload, schema, timeout):
            if 'validation_errors' in payload: raise LLMError('failed')
            return super().complete(instruction,payload,schema,timeout)
    base=[hit('spouse','Nadia is married to Elias.')]
    t={}
    out=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on'),Failed()).run(
        request(),lambda p:dict(data=base),lambda q,ds:np.ones(len(ds)),t)
    assert out==dict(data=base) and t['fallback']


@pytest.mark.parametrize('status',['conflict','time_unknown'])
def test_unresolved_conflict_or_time_cannot_be_declared_complete(status):
    a=EvidencePlanner(None,True,True,trace(),lambda:False,lambda:1)
    a.requirements=[dict(id='N1',depends_on=[])]
    raw=BoundRequiredReview.model_validate(dict(states=[dict(need_id='N1',status=status,
        supports=[support('E1','The visit was last Friday.')],reason='Reference date unknown')],queries=[]))
    result,_,_=a._validate(raw,{'E1':dict(id='event',content='The visit was last Friday.')},[], 'direct',True)
    assert not result.sufficient and result.supports and not result.queries


def test_second_invalid_next_hop_does_not_get_a_second_repair():
    class Twice(Chain):
        def complete(self,instruction,payload,schema,timeout):
            result=super().complete(instruction,payload,schema,timeout)
            if schema.__name__!='RequiredRoute' and result.queries and result.queries[0].need_id=='N3':
                result.queries[0].source_ref='Q0'
            return result
    def retrieve(p):
        if 'Elias work' in p.query:
            return dict(data=[hit('employer','Elias works at Solstice Analytics.')])
        return dict(data=[hit('spouse','Nadia is married to Elias.')])
    t={}
    MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on'),Twice()).run(
        request(),retrieve,lambda q,ds:np.ones(len(ds)),t)
    assert t['repairs']==1 and t['llm_calls']==4 and t['search_calls']==3
    assert t['stop']=='no_grounded_new_query' and t['rejected_queries']==1


def test_binding_only_stops_when_new_queries_make_no_evidence_progress():
    class Unproductive:
        def route(self,*args):return Route(strategy='direct')
        def complete(self,prompt,payload,schema,timeout):
            return schema.model_validate(dict(sufficient=False,supports=[support('E1','Nadia likes music.')],missing='employer',
                queries=[dict(source_ref='E1',quote='Nadia likes music.',anchor='Nadia',
                    template='Where does {target} work?' if len(payload['tried_queries'])==1 else 'What is the employer of {target}?')]))
    t={}
    MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='on'),Unproductive()).run(
        request(),lambda p:dict(data=[hit('a','Nadia likes music.')]),lambda q,ds:np.ones(len(ds)),t)
    assert t['stop']=='no_progress' and t['search_calls']==2 and t['llm_calls']==3


def test_api_add_never_uses_enhanced_planner_and_search_retains_raw_contract(tmp_path):
    class Ready:
        def complete(self,prompt,payload,schema,timeout):
            if schema.__name__=='RequiredRoute':
                return schema.model_validate(dict(strategy='direct',queries=[],requirements=[dict(description='User home city',kind='fact',depends_on=[])]))
            e=payload['evidence'][0]
            assert 'secret' not in e['content']
            return schema.model_validate(dict(states=[dict(need_id='N1',status='supported',supports=[support(e['id'],e['content'])],reason='explicit')],queries=[]))
    class Scorer:
        def score(self,q,docs):return np.ones(len(docs))
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_MULTIHOP_MODE='llm',
             RAG_MULTIHOP_BINDINGS='on',RAG_MULTIHOP_NEEDS='on',AML_AUTH_MODE='bearer',AML_API_KEY='test')
    store=VanillaMemory(cfg,Embedder(),reranker=Scorer(),multihop_planner=Ready())
    with TestClient(create_app(settings=cfg,backend=store)) as client:
        client.headers['Authorization']='Bearer test'
        for uid,text in [('u','Nadia lives in Paris.'),('other','secret')]:
            assert client.post('/add',json=dict(user_id=uid,request_id='r',session_id='s',messages=[dict(role='user',content=text)])).status_code==200
        r=client.post('/search',json=dict(user_id='u',query='Where does Nadia live?',top_k=3))
        assert r.status_code==200
        assert set(r.json())=={'data'} and r.json()['data'][0]['content']=='Nadia lives in Paris.'
        assert set(r.json()['data'][0])=={'id','content','score'}
