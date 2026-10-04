import numpy as np
import pytest
from memory.aml_api import AMLSearch
from memory.multihop import MultiHop, ROUTE_PROMPT
from memory.multihop_evidence import NEED_PLAN, REVIEW_BASE, BINDING_RULES, NEED_RULES
from memory.compact_prompts import effective_instruction
from memory.focused_multihop import StepRead, StepQuery, validate_read, bind_query
from memory.llm import LLMError


def test_short_prompts_keep_legacy_exact_and_cut_instruction_size():
    for text in [ROUTE_PROMPT+NEED_PLAN,REVIEW_BASE+BINDING_RULES+NEED_RULES]:
        assert effective_instruction(text,'long')==text
        assert len(effective_instruction(text,'short')) < len(text)*.4


def test_reader_rejects_invented_value_even_with_valid_quote():
    raw=StepRead(status='supported',reason='',facts=[dict(source_ref='E1',quote='A teaches violin.',value='piano')])
    status,facts,errors=validate_read(raw,{'E1':dict(id='a',content='A teaches violin.')})
    assert status=='missing' and not facts and errors


@pytest.mark.parametrize('anchor,ref',[('Alice','E1'),('Bob','Q0'),('Bob','E99')])
def test_dependent_query_cannot_change_bound_subject(anchor,ref):
    q=StepQuery(query=dict(source_ref=ref,quote='Alice married Bob.',anchor=anchor,template='Where does {target} work?'))
    sources={'E1':dict(id='s',content='Alice married Bob.'),'Q0':dict(id='__question__',content='Alice married Bob.')}
    assert bind_query(q,sources,{'Bob'},True,[]) is None


class Chain:
    def __init__(self): self.calls=[]
    def complete(self,prompt,payload,schema,timeout):
        self.calls.append((schema.__name__,payload))
        if schema.__name__=='RequiredRoute':
            return schema.model_validate(dict(strategy='chain',queries=[],requirements=[
                dict(description='Nadia spouse X',kind='relation',depends_on=[]),
                dict(description='X employer Y',kind='relation',depends_on=[0]),
                dict(description='Y CEO',kind='fact',depends_on=[1])]))
        n=payload['current_need']['id']
        facts={'N1':('Nadia married Elias.','Elias'), 'N2':('Elias works at Solstice.','Solstice'),
               'N3':('Solstice CEO is Ruth.','Ruth')}
        if schema.__name__=='StepRead':
            text,value=facts[n]
            e=next((e for e in payload['evidence'] if e['content']==text),None)
            return schema.model_validate(dict(status='supported' if e else 'missing',reason='',
                facts=[dict(source_ref=e['id'],quote=text,value=value)] if e else []))
        parent=payload['resolved_predecessors'][0]['facts'][0]
        return schema.model_validate(dict(query=dict(source_ref=parent['source_ref'],quote=parent['quote'],
            anchor=parent['value'],template='Where does {target} work?' if n=='N2' else 'Who is CEO of {target}?')))


def test_focused_chain_executes_only_ready_needs_binds_sources_and_returns_raw():
    calls=[]
    facts={'root':'Nadia married Elias.','Where does Elias work?':'Elias works at Solstice.',
           'Who is CEO of Solstice?':'Solstice CEO is Ruth.'}
    def retrieve(p):
        calls.append(p)
        text=facts.get(p.query,facts['root'])
        return {'data':[dict(id=text,content=text,score=1)]}
    planner=Chain();trace={}
    runner=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_PROMPT_STYLE='focused',RAG_MULTIHOP_LLM_CALLS='8'),planner)
    out=runner.run(AMLSearch(user_id='u',query='Who is CEO of the employer of Nadia spouse?',top_k=3),retrieve,lambda q,d:np.ones(len(d)),trace)
    assert trace['stop']=='sufficient' and trace['llm_calls']==8 and trace['search_calls']==3
    assert all(p.user_id=='u' for p in calls)
    assert {h['content'] for h in out['data']}==set(facts.values())
    assert [p.query for p in calls[1:]]==['Where does Elias work?','Who is CEO of Solstice?']
    assert all(len(p['evidence'])<=5 for kind,p in planner.calls if kind=='StepRead')
    assert len(trace['binding_registry'])==2


def test_focused_read_failure_returns_exact_direct_result():
    class Fail(Chain):
        def complete(self,prompt,payload,schema,timeout):
            if schema.__name__!='RequiredRoute': raise LLMError('failed')
            return super().complete(prompt,payload,schema,timeout)
    base={'data':[dict(id='a',content='Nadia married Elias.',score=4)]}; trace={}
    out=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_PROMPT_STYLE='focused'),Fail()).run(
        AMLSearch(user_id='u',query='Nadia spouse employer',top_k=3),lambda p:base,lambda q,d:[1]*len(d),trace)
    assert out==base and trace['fallback'] and trace['error_phase']=='read'


def test_small_budget_stops_before_an_extra_llm_call():
    trace={};p=Chain()
    MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_PROMPT_STYLE='focused',RAG_MULTIHOP_LLM_CALLS='2'),p).run(
        AMLSearch(user_id='u',query='Nadia spouse employer',top_k=3),
        lambda p:{'data':[dict(id='s',content='Nadia married Elias.',score=1)]},lambda q,d:[1]*len(d),trace)
    assert trace['stop']=='llm_budget' and len(p.calls)==2
