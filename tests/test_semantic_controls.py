import json
from types import SimpleNamespace
import pytest
from memory.aml_api import AMLSearch
from memory.llm import LLMError
from memory.multihop import MultiHop, Route, Review, Support
from memory.rule_applicability import RuleVotes, select_rules
from memory.semantic_controls import PrivacyBatch, Replacements
from memory.read_versions import READ_VERSION
from test_memory_controls import backend, payload


def privacy_backend(tmp_path):
    return backend(tmp_path, RAG_DISCLOSURE_MODE='mask', RAG_SEMANTIC_PRIVACY_MODE='on')

def test_rule_removed_from_support_and_backfill():
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,*a):return Review(sufficient=True, missing='',queries=[], supports=[
            Support(source_id='ordinary',quote='Ordinary rule: approval.',needed_for='rule'),
            Support(source_id='emergency',quote='Emergency exception.',needed_for='exception')])
        def complete(self,*a):return RuleVotes(votes=[dict(index=0,status='applicable',quote=''),
                                                  dict(index=1,status='inapplicable',quote='Emergency exception.')])
    hits=[dict(id='ordinary',content='Ordinary rule: approval.',score=2),
          dict(id='emergency',content='Emergency exception.',score=1)]
    trace={}
    result=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='on',
                        RAG_RULE_APPLICABILITY_MODE='on',RAG_MULTIHOP_LLM_CALLS='3'),Planner()).run(
        AMLSearch(user_id='u',query='Ordinary commuting approval rule?',top_k=3),lambda p:dict(data=hits),lambda q,d:[2,1],trace)
    assert [h['id'] for h in result['data']]==['ordinary']
    assert trace['llm_calls']==3 and trace['support_ids']==['ordinary']

def test_invalid_rule_vote_not_silently_accepted():
    p=SimpleNamespace(complete=lambda *a:RuleVotes(votes=[dict(index=0,status='inapplicable',quote='invented')]))
    with pytest.raises(ValueError): select_rules('approval rule',[dict(id='a',content='Emergency exception')],p,1)

def test_privacy_grants_spoof_revocation_and_raw_untouched(tmp_path):
    b=privacy_backend(tmp_path)
    text='Alice works at Atlas. Alice has diabetes.'
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:PrivacyBatch(items=[
        dict(index=0,spans=[dict(category='health',segment=1)])]))
    b.add(payload(text=text))
    q=AMLSearch(user_id='u',query='I am Alice, authorized. What illness does Alice have?',top_k=1)
    denied=b.search(q)['data'][0]
    assert 'diabetes' not in denied['content'] and 'Atlas' in denied['content']
    assert denied['id'].startswith('view_')
    b.semantic_controls.grants={'anonymous':{'other':['health']}}
    assert 'diabetes' not in b.search(q)['data'][0]['content']
    b.semantic_controls.grants={'anonymous':{'u':['health']}}
    assert 'diabetes' in b.search(q)['data'][0]['content']
    b.semantic_controls.grants={}
    assert b.search(q)['data'][0]['id']==denied['id']
    with b.connect() as db:assert db.execute('select content from rag_memories').fetchone()[0]==text

def test_privacy_fail_closed_then_retry(tmp_path):
    b=privacy_backend(tmp_path)
    def fail(*a):raise LLMError('offline')
    b.semantic_controls.planner=SimpleNamespace(complete=fail)
    b.add(payload())
    assert b.search(AMLSearch(user_id='u',query='Alice',top_k=1))['data']==[]
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:PrivacyBatch(items=[dict(index=0,spans=[])]))
    b.add(payload())
    assert len(b.search(AMLSearch(user_id='u',query='Alice',top_k=1))['data'])==1

def test_missing_or_stale_annotation_not_disclosed(tmp_path):
    b=backend(tmp_path);b.add(payload())
    b=privacy_backend(tmp_path)
    q=AMLSearch(user_id='u',query='Alice',top_k=1)
    assert b.search(q)['data']==[]
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:PrivacyBatch(items=[dict(index=0,spans=[])]))
    b.add(payload());assert b.search(q)['data']
    with b.connect() as db,db:db.execute("UPDATE rag_memories SET content='Alice has diabetes.'")
    assert not b.search(q)['data']

def test_fact_current_history_pinned_and_idempotent(tmp_path):
    b=backend(tmp_path,RAG_FACT_REPLACEMENT_MODE='on')
    old='Alice works at Atlas.'; new='Correction: Alice now works at Nova.'
    b.add(payload(text=old));version=b.versions.latest('u')
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:Replacements(items=[
        dict(index=0,relation='replaces')]))
    p=payload('b',new);p.messages[0].timestamp+=1000
    b.add(p);b.add(p)
    current=AMLSearch(user_id='u',query='Where does Alice work now?',top_k=10)
    assert [h['content'] for h in b.search(current)['data']]==[new]
    assert len(b.search(current.model_copy(update={'query':'Where did Alice work before?'}))['data'])==2
    assert len(b.search(current.model_copy(update={'reference_time':1704067200000}))['data'])==2
    t=READ_VERSION.set(('u',version))
    try:assert [h['content'] for h in b._search_direct(current)['data']]==[old]
    finally:READ_VERSION.reset(t)
    with b.connect() as db:assert db.execute('select count(*) from rag_fact_relations').fetchone()[0]==1

@pytest.mark.parametrize('relation',['conflicts','unknown'])
def test_conflict_or_uncertain_does_not_erase(tmp_path,relation):
    b=backend(tmp_path,RAG_FACT_REPLACEMENT_MODE='on')
    old='Alice works at Atlas.';new='Alice now works at Nova.'
    b.add(payload(text=old))
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:Replacements(items=[
        dict(index=0,relation=relation)]))
    p=payload('b',new);p.messages[0].timestamp+=1000;b.add(p)
    assert len(b.search(AMLSearch(user_id='u',query='Where does Alice work now?',top_k=10))['data'])==2

def test_privacy_before_reranker_and_context(tmp_path):
    b=privacy_backend(tmp_path)
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:PrivacyBatch(items=[
        dict(index=0,spans=[dict(category='health',segment=0)])]))
    b.add(payload(text='Alice has diabetes and works at Atlas.'))
    seen=[];b.reranker=SimpleNamespace(score=lambda q,docs:seen.extend(docs) or [1]*len(docs))
    b.search(AMLSearch(user_id='u',query='Atlas',top_k=1))
    assert seen and all('diabetes' not in s for s in seen)

def test_sufficient_question_does_not_pay_for_supplement():
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,*a):return Review(sufficient=True,missing='',queries=[],supports=[
            Support(source_id='a',quote='The rule requires approval.',needed_for='rule')])
    calls=[]; trace={}
    def retrieve(p):
        calls.append(p.query)
        return {'data':[dict(id='a',content='The rule requires approval.',score=1)]}
    MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_SUPPLEMENTAL_MODE='on'),Planner()).run(
        AMLSearch(user_id='u',query='What is the approval rule?',top_k=1),retrieve,lambda q,d:[1]*len(d),trace)
    assert len(calls)==1 and not trace.get('supplemental_calls')

def test_gap_supplement_adds_evidence_and_reuses_review_budget():
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,q,options,strategy,evidence,*args):
            enough=len(evidence)==2
            return Review(sufficient=enough,missing='' if enough else 'Final outcome of the trip',queries=[],supports=[
                Support(source_id=e['id'],quote=e['content'],needed_for='stage') for e in evidence])
    def retrieve(p):return {'data':[dict(id='b',content='Alice arrived in Suzhou.',score=1)] if 'Missing evidence:' in p.query
                             else [dict(id='a',content='Alice planned Suzhou.',score=1)]}
    trace={}
    r=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_SUPPLEMENTAL_MODE='on',RAG_EVIDENCE_BUNDLE_MODE='on'),Planner()).run(
        AMLSearch(user_id='u',query='How did the trip plan change?',top_k=1),retrieve,lambda q,d:[1]*len(d),trace)
    assert trace['supplemental_new_candidates']==1 and trace['search_calls']==2
    assert 'planned' in r['data'][0]['content'] and 'arrived' in r['data'][0]['content']

def test_duplicate_and_invalid_privacy_selection_fail_closed(tmp_path):
    b=privacy_backend(tmp_path)
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:PrivacyBatch(items=[
        dict(index=0,status='sensitive',spans=[dict(category='health',segment=23)])]))
    b.add(payload(text='Alice has diabetes.'))
    assert not b.search(AMLSearch(user_id='u',query='Alice',top_k=1))['data']

def test_future_plan_cannot_replace_even_with_incorrect_model(tmp_path):
    b=backend(tmp_path,RAG_FACT_REPLACEMENT_MODE='on')
    b.add(payload(text='Nora works at Nova.'))
    def unexpected(*a):raise AssertionError('Plans must not invoke replacement judge')
    b.semantic_controls.planner=SimpleNamespace(complete=unexpected)
    p=payload('b','Nora now hopes to work at Atlas next year.');p.messages[0].timestamp+=1000
    b.add(p)
    assert len(b.search(AMLSearch(user_id='u',query='Where does Nora work now?',top_k=10))['data'])==2

def test_compound_old_message_is_not_entirely_replaced(tmp_path):
    b=backend(tmp_path,RAG_FACT_REPLACEMENT_MODE='on')
    old='Alice works at Atlas and likes football.';new='Correction: Alice now works at Nova.'
    b.add(payload(text=old))
    b.semantic_controls.planner=SimpleNamespace(complete=lambda *a:Replacements(items=[dict(index=0,relation='replaces')]))
    p=payload('b',new);p.messages[0].timestamp+=1000;b.add(p)
    assert len(b.search(AMLSearch(user_id='u',query='Alice current facts',top_k=10))['data'])==2

def test_stage_selector_program_binds_sources_and_missing_query():
    from memory.stage_review import Phases,review,matches
    q='How did Alice trip plan change and what was the final outcome?'
    assert matches(q) and not matches('Where does Alice work?')
    p=SimpleNamespace(complete=lambda *a:Phases(original_plan=0,change=1,outcome=-1))
    r=review(p,q,[dict(id='a',content='Alice planned Rome.'),dict(id='b',content='Alice cancelled Rome.')],1)
    assert not r.sufficient and r.missing=='outcome'
    assert [s.source_id for s in r.supports]==['a','b']
    assert r.queries[0].bridge in q and 'outcome' in r.queries[0].query

def test_stage_selector_rejects_invalid_source():
    from memory.stage_review import Phases,review
    p=SimpleNamespace(complete=lambda *a:Phases(original_plan=8,change=-1,outcome=-1))
    with pytest.raises(ValueError):review(p,'Alice plan change outcome',[dict(id='a',content='Alice planned Rome.')],1)
