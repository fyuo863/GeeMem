import json
from types import SimpleNamespace
from memory.source_paths import candidate_paths
from memory.evidence_bundle import EvidenceBundler
from memory.multihop import MultiHop, Route
from memory.raw_evidence import select
from memory.aml_api import AMLSearch


def edge(i,s,r,o,**extra):
    return dict(id=str(i),content=json.dumps(dict(subject=s,relation=r,object=o,**extra)),score=10-i)


def rows():
    return [edge(0,'Atlas','creator','Studio'),edge(1,'Studio','director','Ada'),
            edge(2,'Ada','country','France'),edge(3,'France','language','French'),
            edge(4,'Studio','director','Ben'),edge(5,'Ben','country','Japan'),
            edge(6,'Japan','language','Japanese'),edge(7,'Stranger','language','Spanish')]


QUESTION='What language is used in the country of the director of the creator of Atlas?'


def test_explicit_alternatives_have_complete_paths_and_no_unrelated_endpoint():
    assert candidate_paths(QUESTION,rows())==[[0,1,2,3],[0,4,5,6]]
    assert candidate_paths('What language is used in the country of the director of the creator of Atlantis?',rows())==[]


def test_ids_override_names_and_conflicts_never_join():
    data=[edge(0,'Atlas','creator','Studio',object_id='org1'),
          edge(1,'Studio','language','French',subject_id='org2')]
    assert candidate_paths('What language does the creator of Atlas use?',data)==[]
    data[1]=edge(1,'Different spelling','language','French',subject_id='org1')
    assert candidate_paths('What language does the creator of Atlas use?',data)==[[0,1]]


def test_reverse_or_natural_language_or_missing_terminal_is_not_invented():
    # A path to an inner relation can exist when the outer relation is absent.
    # It remains partial evidence, never a fabricated terminal or sufficient flag.
    assert candidate_paths(QUESTION,rows()[:3])==[[0,1,2]]
    assert candidate_paths(QUESTION,[dict(id='a',content='Atlas was created by Studio.')])==[]
    assert candidate_paths(QUESTION,[edge(0,'Studio','creator','Atlas'),edge(1,'Studio','language','French')])==[]
    data=rows();data[0]['content']='Example, not a fact: '+data[0]['content']
    assert candidate_paths(QUESTION,data)==[]
    data=rows();data[0]=edge(0,'Atlas','creator','Studio',negated=True)
    assert candidate_paths(QUESTION,data)==[]


def test_paths_are_bounded_under_branches_and_cycles():
    data=[edge(0,'Atlas','creator','Atlas')]+[edge(i+1,'Atlas','language',str(i)) for i in range(99)]
    paths=candidate_paths('What language does the creator of Atlas use?',data)
    assert len(paths)<=8 and all(len(p)<=6 and len(p)==len(set(p)) for p in paths)


def test_model_omission_cannot_remove_explicit_candidate_path_originals():
    def complete(prompt,payload,schema,timeout):
        assert payload['candidate_paths']==[[0,1,2,3],[0,4,5,6]]
        assert 'single sentence' in prompt
        return schema(selected=[0,1,2],stop_search=True,missing='',queries=[])
    planner=SimpleNamespace(raw_selection='per_source',raw_chain_selection='joint',complete=complete)
    review=select(planner,QUESTION,None,'chain',rows(),[],20)
    assert {s.source_id for s in review.supports}=={str(i) for i in range(7)}


def test_path_packages_share_root_and_keep_order_and_topk_omissions():
    data=rows();trace={'raw_candidate_paths':[['0','1','2','3'],['0','4','5','6']]}
    bundler=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'raw'})
    result=bundler.assemble('u',data,{},False,2,trace)
    assert [b['source_ids'] for b in result['_bundles']]==trace['raw_candidate_paths']
    assert len(result['data'])==2
    for h,b in zip(result['data'],result['_bundles']):
        offsets=[h['content'].index(next(v['content'] for v in data if v['id']==i)) for i in b['source_ids']]
        assert offsets==sorted(offsets)
    trace={'raw_candidate_paths':[['0','1','2','3'],['0','4','5','6']]}
    limited=bundler.assemble('u',data,{},False,1,trace)
    assert len(limited['data'])==1
    assert trace['evidence_bundle']['omitted_path_source_ids']==['4','5','6']


def test_oversize_or_invalid_paths_preserve_raw_fallback():
    data=rows();data[0]['content']='x'*6001
    result=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'raw'}).assemble('u',data,{},False,10,
        {'raw_candidate_paths':[['0','1','2','3'],['missing','4']]})
    assert result['data']==data


def test_full_orchestrator_delivers_both_paths_in_two_slots():
    class Planner:
        raw_selection='per_source';raw_chain_selection='joint'
        def route(self,*a):return Route(strategy='chain',queries=[])
        def complete(self,prompt,payload,schema,timeout):
            return schema(selected=[0,1,2,3],stop_search=True,missing='',queries=[])
        def review(self,*args):return select(self,*args)
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_BUNDLE_MODE':'raw'},Planner()).run(
        AMLSearch(user_id='u',query=QUESTION,top_k=2),lambda p:dict(data=rows()),lambda q,d:[1]*len(d),trace)
    assert not trace['fallback'] and trace['llm_calls']==2 and trace['search_calls']==1
    assert {tuple(b['source_ids']) for b in result['_bundles']}=={('0','1','2','3'),('0','4','5','6')}


def test_review_failure_still_delivers_explicit_paths():
    class Planner:
        raw_chain_selection='joint'
        def route(self,*a):return Route(strategy='chain',queries=[])
        def review(self,*args):raise TimeoutError('review deadline')
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_BUNDLE_MODE':'raw'},Planner()).run(
        AMLSearch(user_id='u',query=QUESTION,top_k=2),lambda p:dict(data=rows()),lambda q,d:[1]*len(d),trace)
    assert trace['fallback']
    assert {tuple(b['source_ids']) for b in result['_bundles']}=={('0','1','2','3'),('0','4','5','6')}
