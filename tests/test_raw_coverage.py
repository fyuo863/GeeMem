from types import SimpleNamespace
from memory.multihop import MultiHop, Route, Review, Support, CollectionReview, Query
from memory.evidence_bundle import EvidenceBundler
from memory.raw_evidence import select
from memory.aml_api import AMLSearch


def test_unseen_page_review_without_extra_retrieval():
    rows=[dict(id=str(i),content=f'Alice source {i}.',score=20-i) for i in range(20)]
    packets=[]; queries=[]
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,q,o,s,e,h,t):
            packets.append([v['id'] for v in e])
            return Review(sufficient=True,missing='',queries=[],supports=[
                Support(source_id=v['id'],quote=v['content'],needed_for='source')
                for v in e if v['id'] in ('0','18')])
    def retrieve(p):queries.append(p);return dict(data=rows[:p.top_k])
    trace={}
    result=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw',
        RAG_RAW_COVERAGE_MODE='on'),Planner()).run(
        AMLSearch(user_id='u',query='Alice history?',top_k=1),retrieve,lambda q,d:[1]*len(d),trace)
    assert '18' not in packets[0] and '18' in packets[1]
    assert all(len(p)<=12 for p in packets)
    assert len(queries)==1 and trace['llm_calls']==3
    assert 'Alice source 0.' in result['data'][0]['content']
    assert 'Alice source 18.' in result['data'][0]['content']


def test_more_than_eight_selected_sources_are_delivered_before_noise():
    rows=[dict(id=str(i),content=f'Original {i}',score=30-i) for i in range(13)]
    supports={r['id']:SimpleNamespace(quote=r['content']) for r in rows[1:]}
    trace={}
    result=EvidenceBundler(dict(RAG_EVIDENCE_BUNDLE_MODE='raw',RAG_RAW_COVERAGE_MODE='on')).assemble(
        'u',rows,supports,False,3,trace)
    assert len(result['_bundles'])==3
    assert sum(len(b['source_ids']) for b in result['_bundles'])==12
    assert not trace['evidence_bundle']['omitted_source_ids']


def test_gap_is_bound_to_source_and_constructed_by_program():
    def complete(prompt,payload,schema,timeout):
        return schema(selected=[],stop_search=True,missing='',queries=[],
                      gaps=[dict(kind='update',source_index=0)])
    result=select(SimpleNamespace(complete=complete,coverage=True),'Where does Alice work now?',[],
                  'direct',[dict(id='old',content='Alice works at Atlas.')],[],20)
    assert isinstance(result,CollectionReview) and not result.sufficient
    assert result.gap_queries[0].source_id=='old'
    assert result.gap_queries[0].bridge in result.gap_queries[0].query
    assert result.gap_kinds==['update']
    assert result.supports[0].source_id=='old'


def test_companion_queries_share_budget_and_do_not_repeat():
    class Planner:
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,q,o,s,e,h,t):
            return CollectionReview(sufficient=False,missing='update',queries=[],
                supports=[Support(source_id='a',quote='Alice works at Atlas.',needed_for='source')],
                gap_queries=[Query(query='Alice works at Atlas. Find updates.',source_id='a',bridge='Alice')],
                gap_kinds=['update'])
    calls=[];trace={}
    def retrieve(p):calls.append(p.query);return dict(data=[dict(id='a',content='Alice works at Atlas.',score=1)])
    MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw',RAG_RAW_COVERAGE_MODE='on'),Planner()).run(
        AMLSearch(user_id='u',query='Where does Alice work now?',top_k=1),retrieve,lambda q,d:[1],trace)
    assert len(calls)==2 and len(trace['companion_queries'])==1
    assert trace['llm_calls']<=5 and trace['search_calls']<=6


def test_later_review_failure_preserves_selected_evidence():
    from memory.llm import LLMError
    class Planner:
        calls=0
        def route(self,*a):return Route(strategy='direct',queries=[])
        def review(self,q,o,s,e,h,t):
            self.calls+=1
            if self.calls>1: raise LLMError('temporary failure')
            return Review(sufficient=False,missing='next',queries=[],supports=[
                Support(source_id='1',quote='Alice source 1.',needed_for='source'),
                Support(source_id='2',quote='Alice source 2.',needed_for='source')])
    rows=[dict(id=str(i),content=f'Alice source {i}.',score=20-i) for i in range(20)]
    trace={}
    result=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw',RAG_RAW_COVERAGE_MODE='on'),Planner()).run(
        AMLSearch(user_id='u',query='Alice history?',top_k=1),lambda p:dict(data=rows),lambda q,d:[1]*len(d),trace)
    assert trace['fallback'] and trace['fallback_preserved_sources']==['1','2']
    assert all(f'Alice source {i}.' in result['data'][0]['content'] for i in (1,2))
