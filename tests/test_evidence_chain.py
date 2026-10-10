from types import SimpleNamespace
from memory.evidence_chain import build_chain
from memory.evidence_bundle import EvidenceBundler
from memory.multihop import MultiHop, Route, Review, Support, EvidenceLink
from memory.aml_api import AMLSearch


def support(source_id, quote, need):
    return SimpleNamespace(source_id=source_id, quote=quote, needed_for=need)


def test_chain_binds_review_order_and_parent():
    s={'a': 'A fact', 'b': 'A fact B fact'}
    nodes,audit=build_chain({'a':support('a','A fact','first'),
                             'b':support('b','A fact B fact','second')},s,'chain',True,[EvidenceLink(parent=0,child=1,bridge='A fact')])
    assert audit['status']=='complete'
    assert nodes[0].derived_from == [] and nodes[1].derived_from==['a']


def test_chain_rejects_unreferenced_quote_and_incomplete():
    _,audit=build_chain({'a':support('a','not present','first')},{'a':'A fact'},'chain',True)
    assert audit['status']=='invalid' and audit['errors'][0]['error']=='quote_not_grounded'


def test_bundle_follows_chain_and_exposes_provenance():
    cfg={'RAG_EVIDENCE_BUNDLE_MODE':'on'}
    trace={'evidence_chain':{'status':'complete','nodes':[
        {'hop':1,'source_id':'a','derived_from':None},
        {'hop':2,'source_id':'b','derived_from':'a'}]}}
    supports={'a':support('a','A fact','first'),'b':support('b','B fact','second')}
    result=EvidenceBundler(cfg).assemble('u',
        [dict(id='b',content='B fact',score=2),dict(id='a',content='A fact',score=1)],
        supports,True,2,trace)
    text=result['data'][0]['content']
    assert text.index('A fact') < text.index('B fact')
    assert 'hop=2' in text and 'derived_from="a"' in text


def test_multihop_returns_partial_chain_when_middle_link_is_missing():
    class Planner:
        def route(self,*args): return Route(strategy='chain',queries=[])
        def review(self,q,options,strategy,evidence,*args):
            return Review(sufficient=False,missing='missing middle link',queries=[],supports=[
                Support(source_id='a',quote='Veda has brother Niko.',needed_for='brother'),
                Support(source_id='c',quote='Selene teaches clarinet.',needed_for='instrument')])
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_EVIDENCE':'8','RAG_EVIDENCE_CHAIN_MODE':'on',
                     'RAG_MULTIHOP_ROUNDS':'1','RAG_MULTIHOP_QUERIES':'2'},Planner()).run(
        AMLSearch(user_id='u',query='What instrument does Veda brother teacher teach?',top_k=2),
        lambda p:{'data':[dict(id='a',content='Veda has brother Niko.',score=2),
                          dict(id='c',content='Selene teaches clarinet.',score=1)]},
        lambda q,d:[2,1],trace)
    assert trace['evidence_chain']['status']=='partial'
    assert trace['evidence_chain']['nodes'][1]['derived_from']==[]
    assert result['data']


def test_two_unrelated_sources_never_become_chain_by_order():
    ss=[support('a','Veda knows Niko','person'),support('b','Selene teaches clarinet','instrument')]
    _,audit=build_chain(ss,{'a':ss[0].quote,'b':ss[1].quote},'chain',True)
    assert audit['status']=='partial'
    assert audit['nodes'][1]['derived_from']==[]


def test_split_has_independent_sources_and_single_source_chain_is_allowed():
    ss=[support('a','Susie was adopted in May','date'),support('b','Seraphim was adopted in June','date')]
    _,audit=build_chain(ss,{'a':ss[0].quote,'b':ss[1].quote},'split',True)
    assert audit['status']=='complete' and audit['links']==[]
    _,audit=build_chain(ss[:1],{'a':ss[0].quote},'chain',True)
    assert audit['status']=='complete'  # Semantic sufficiency is a model judgment.


def test_nonexistent_cyclic_and_ungrounded_links_rejected():
    ss=[support('a','Veda knows Niko','person'),support('b','Selene teaches clarinet','instrument')]
    for link in [EvidenceLink(parent=1,child=0,bridge='Niko'), EvidenceLink(parent=0,child=1,bridge='Niko'),EvidenceLink(parent=0,child=7,bridge='Niko')]:
        _,audit=build_chain(ss,{'a':ss[0].quote,'b':ss[1].quote},'chain',True,[link])
        assert audit['status']=='invalid'

def test_root_anchor_rejects_other_person_connected_chain():
    ss=[support('a','Lena knows Pavel','person'),support('b','Pavel teaches flute','instrument')]
    _,audit=build_chain(ss,{'a':ss[0].quote,'b':ss[1].quote},'chain',True,
        [EvidenceLink(parent=0,child=1,bridge='Pavel')],root_anchor='Veda')
    assert audit['status']=='partial'
    assert 'missing_question_anchor' in audit['missing_hops']


def test_partial_bundle_exposes_status_and_delivery_limits():
    supports={'a':support('a','A fact','first'),'b':support('b','B fact','second')}
    nodes,audit=build_chain(supports,{'a':'A fact','b':'B fact'},'chain',False)
    ranked=[dict(id='a',content='A fact',score=2),dict(id='b',content='B fact',score=1)]
    trace={'evidence_chain':audit}
    bundler=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'on'})
    result=bundler.assemble('u',ranked,supports,False,1,trace)
    assert 'status: partial' in result['data'][0]['content']
    assert audit['delivery_complete']
    bundler.max_chars=1
    result=bundler.assemble('u',ranked,supports,False,1,trace)
    assert not audit['delivery_complete'] and audit['omitted_source_ids']==['b']

def test_chain_feedback_repairs_once_within_call_budget():
    from memory.multihop import ChainReview
    class Planner:
        calls=0
        def route(self,*args): return Route(strategy='chain',queries=[])
        def review(self,q,options,strategy,evidence,*args):
            self.calls+=1
            links=[]
            if self.calls==2:
                assert evidence[0]['chain_feedback']['missing_hops']==['unverified_dependency']
                links=[EvidenceLink(parent=0,child=1,bridge='Niko')]
            return ChainReview(sufficient=True,missing='',queries=[],links=links,supports=[
                Support(source_id='a',quote='Veda knows Niko.',needed_for='relationship'),
                Support(source_id='b',quote='Niko teaches flute.',needed_for='instrument')])
    planner=Planner();trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_CHAIN_MODE':'on',
        'RAG_EVIDENCE_BUNDLE_MODE':'on','RAG_MULTIHOP_ROUNDS':'2','RAG_MULTIHOP_LLM_CALLS':'3'},planner).run(
        AMLSearch(user_id='u',query='What instrument does the person Veda knows teach?',top_k=1),
        lambda p:{'data':[dict(id='a',content='Veda knows Niko.',score=2),dict(id='b',content='Niko teaches flute.',score=1)]},
        lambda q,d:[2,1],trace)
    assert planner.calls==2 and trace['llm_calls']==3
    assert trace['chain_repair_requested'] and trace['evidence_chain']['status']=='complete'
    assert trace['evidence_chain']['delivery_complete']
    assert 'Niko teaches flute.' in result['data'][0]['content']

def test_bundle_outside_topk_not_counted_as_delivered():
    supports={'a':support('a','A fact','first'),'b':support('b','B fact','second')}
    _,audit=build_chain(supports,{'a':'A fact','b':'B fact'},'split',True)
    trace={'evidence_chain':audit}
    result=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'on'}).assemble('u',[
        dict(id='x',content='other',score=3),dict(id='a',content='A fact',score=2),
        dict(id='b',content='B fact',score=1)],supports,True,1,trace)
    assert result['data'][0]['id']=='x'
    assert audit['omitted_source_ids']==['a','b'] and not audit['delivery_complete']
    assert not result['_bundles']

def test_chain_rejects_incompatible_review_adapters():
    import pytest
    for override in [{'RAG_MULTIHOP_PROMPT_STYLE':'focused'}, {'RAG_MULTIHOP_BINDINGS':'on'}, {'RAG_MULTIHOP_NEEDS':'on'}]:
        with pytest.raises(ValueError,match='standard review'):
            MultiHop(dict(RAG_EVIDENCE_CHAIN_MODE='on',**override))
