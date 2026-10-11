from types import SimpleNamespace
import pytest
from memory.raw_evidence import select, compact_evidence
from memory.evidence_bundle import EvidenceBundler


def sources(n=12):
    return [dict(id=str(i),content=f'Alice state {i}.',created_at=f'2024-01-{i+1:02d}T00:00:00Z',score=float(n-i)) for i in range(n)]


def test_compaction_preserves_indices_text_and_unknown_times():
    rows=sources(3)[::-1]
    rows[0]['context']='Target message: on date: '+rows[0]['content']+'\nPrevious context: Alice old state\nNext context: Alice next state'
    rows[1]['created_at']=None
    compact=compact_evidence(rows)
    assert [r['index'] for r in compact]==[2,0,1]
    assert compact[1]['content']==rows[0]['content']
    assert 'Target message' not in compact[1]['context']
    assert rows[0]['content'] not in compact[1]['context']
    assert 'Alice old state' in compact[1]['context']
    assert 'Target message' in rows[0]['context']  # never mutate audit/raw data


def test_all_votes_bind_and_more_than_eight_survive_packaging():
    rows=sources()
    def complete(prompt,payload,schema,timeout):
        return schema(votes=[dict(index=i,relevant=True) for i in reversed(range(12))],stop_search=True,missing='',queries=[])
    result=select(SimpleNamespace(raw_selection='per_source',complete=complete), 'Alice history?',None,'direct',rows,[],20)
    assert len(result.supports)==12
    trace={}
    out=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'raw'}).assemble('u',rows,{s.source_id:s for s in result.supports},True,3,trace)
    assert sum(len(b['source_ids']) for b in out['_bundles'])==12
    assert not trace['evidence_bundle']['omitted_source_ids']
    assert out['_bundles'][0]['source_ids']==['0','1','2','3']


@pytest.mark.parametrize('indices',[[0],[0,0],[0,2]])
def test_missing_duplicate_or_unknown_vote_is_rejected(indices):
    def complete(prompt,payload,schema,timeout):
        return schema(votes=[dict(index=i,relevant=True) for i in indices],stop_search=True,missing='',queries=[])
    with pytest.raises(ValueError,match='source relevance votes'):
        select(SimpleNamespace(raw_selection='per_source',complete=complete),'q',None,'direct',sources(2),[],20)


def test_negative_vote_does_not_inherit_neighbor_fact():
    def complete(prompt,payload,schema,timeout):
        return schema(votes=[dict(index=0,relevant=False),dict(index=1,relevant=True)],stop_search=True,missing='',queries=[])
    result=select(SimpleNamespace(raw_selection='per_source',complete=complete),'q',None,'direct',sources(2),[],20)
    assert [s.source_id for s in result.supports]==['1']


@pytest.mark.parametrize('strategy',['chain','split'])
def test_multihop_keeps_original_collective_selection(strategy):
    def complete(prompt,payload,schema,timeout):
        assert 'selected' in schema.model_fields and 'votes' not in schema.model_fields
        return schema(selected=[1,0],stop_search=True,missing='',queries=[])
    result=select(SimpleNamespace(raw_selection='per_source',complete=complete),'q',None,strategy,sources(2),[],20)
    assert [s.source_id for s in result.supports]==['1','0']


def test_chain_joint_selection_preserves_bridge_terminal_update_and_source_ids():
    rows=sources(5)
    def complete(prompt,payload,schema,timeout):
        assert 'jointly, not independently' in prompt
        assert 'Judge each independently' not in prompt
        assert 'votes' not in schema.model_fields
        return schema(selected=[0,1,2,3],stop_search=True,missing='',queries=[])
    result=select(SimpleNamespace(raw_selection='per_source',raw_chain_selection='joint',complete=complete),
                  'q',None,'chain',rows,[],20)
    assert [s.source_id for s in result.supports]==['0','1','2','3']
    assert all(s.quote==rows[int(s.source_id)]['content'] for s in result.supports)


def test_chain_gap_keeps_partial_path_and_bound_query():
    from memory.multihop import CollectionReview
    def complete(prompt,payload,schema,timeout):
        return schema(selected=[],stop_search=False,missing='employer',queries=[],
                      gaps=[dict(kind='intermediate',source_index=0)])
    result=select(SimpleNamespace(raw_selection='per_source',raw_chain_selection='joint',coverage=True,complete=complete),
                  'Where does her brother work?',None,'chain',sources(1),[],20)
    assert isinstance(result,CollectionReview)
    assert result.supports[0].source_id=='0'
    assert result.gap_queries[0].source_id=='0'
    assert not result.sufficient


@pytest.mark.parametrize('indices',[[-1],[2]])
def test_chain_joint_unknown_index_is_rejected(indices):
    def complete(prompt,payload,schema,timeout):
        return schema(selected=indices,stop_search=True,missing='',queries=[])
    with pytest.raises(ValueError,match='outside source set'):
        select(SimpleNamespace(raw_selection='per_source',raw_chain_selection='joint',complete=complete),
               'q',None,'chain',sources(2),[],20)


def test_chain_joint_survives_orchestration_and_final_packaging():
    from memory.multihop import MultiHop, Route
    from memory.aml_api import AMLSearch
    rows=sources(10)
    class Planner:
        raw_selection='per_source'
        raw_chain_selection='joint'
        coverage=False
        def route(self,*args): return Route(strategy='chain',queries=[])
        def complete(self,prompt,payload,schema,timeout):
            return schema(selected=list(range(10))+[0],stop_search=True,missing='',queries=[])
        def review(self,*args):return select(self,*args)
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_BUNDLE_MODE':'raw'},Planner()).run(
        AMLSearch(user_id='u',query='Alice chain?',top_k=3),lambda p:dict(data=rows),
        lambda q,docs:[1.0]*len(docs),trace)
    assert not trace['fallback']
    assert trace['llm_calls']==2  # route + one batch review
    assert len(result['_bundles'])==3
    assert sum(len(b['source_ids']) for b in result['_bundles'])==10


def test_explicit_legacy_disables_joint_even_with_joint_chain_setting():
    def complete(prompt,payload,schema,timeout):
        assert 'jointly, not independently' not in prompt
        return schema(selected=[0],stop_search=True,missing='',queries=[])
    select(SimpleNamespace(raw_selection='legacy',raw_chain_selection='joint',complete=complete),
           'q',None,'chain',sources(1),[],20)


def test_production_default_uses_joint_and_rejects_unknown_mode():
    from memory.multihop import Planner
    planner=Planner({'RAG_EVIDENCE_BUNDLE_MODE':'raw'})
    assert planner.raw_selection=='per_source'
    assert planner.raw_chain_selection=='joint'
    def complete(prompt,payload,schema,timeout):
        assert 'votes' not in schema.model_fields
        assert 'Judge each independently' not in prompt
        return schema(selected=[0,1],stop_search=True,missing='',queries=[])
    planner.complete=complete
    result=select(planner,'q',None,'chain',sources(2),[],20)
    assert len(result.supports)==2
    with pytest.raises(ValueError,match='Invalid raw chain selection mode'):
        Planner({'RAG_RAW_CHAIN_SELECTION_MODE':'roles'})
