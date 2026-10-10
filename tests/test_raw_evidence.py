from types import SimpleNamespace
import pytest
from memory.evidence_bundle import EvidenceBundler
from memory.multihop import Planner, MultiHop
from memory.raw_evidence import select
from test_memory_controls import backend


def hits(n):
    return [dict(id=str(i), content=f'Original source number {i}.', score=10-i,
                 created_at=f'2024-01-0{i+1}T00:00:00Z') for i in range(n)]


def package(rows, k=10):
    trace={}
    result=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'raw'}).assemble('u', rows,
        {h['id']:SimpleNamespace(quote=h['content']) for h in rows}, False, k, trace)
    return result,trace


def test_partial_sources_pack_without_semantic_verdict_and_keep_times():
    rows=hits(3)[::-1]; result,trace=package(rows)
    content=result['data'][0]['content']
    assert content.index('number 0') < content.index('number 2')
    assert all(h['content'] in content and h['created_at'] in content for h in rows)
    assert 'not adjudicated' in content and 'derived_from' not in content
    assert trace['evidence_bundle']['semantic_status']=='not_adjudicated'


def test_over_four_sources_split_and_topk_omission_is_recorded():
    result,trace=package(hits(7))
    assert len(result['_bundles'])==2
    assert sum(len(b['source_ids']) for b in result['_bundles'])==7
    result,trace=package(hits(7),1)
    assert len(result['_bundles'])==1
    assert len(trace['evidence_bundle']['omitted_source_ids'])==3


def test_oversized_source_is_not_truncated_or_lost():
    rows=hits(3); rows[0]['content']='x'*6100
    result,_=package(rows)
    assert result['data'][0]['content']==rows[0]['content']
    assert len(result['_bundles'])==1


def test_length_split_dedup_and_invalid_reference_preserve_originals():
    rows=hits(3)
    for h in rows: h['content']+='x'*2700
    result,_=package(rows+[rows[0]])
    assert len(result['_bundles'])==1
    assert len(result['data'][0]['content'])<=6000
    assert result['data'][1]['content']==rows[2]['content']
    trace={}
    result=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'raw'}).assemble('u',hits(2),
        {'0':SimpleNamespace(quote='invented'),'missing':SimpleNamespace(quote='other')},False,2,trace)
    assert result['data']==hits(2) and result['_bundles']==[]


def test_mode_overrides_old_adjudication_switches(tmp_path):
    cfg=dict(RAG_EVIDENCE_BUNDLE_MODE='raw',RAG_FACT_REPLACEMENT_MODE='on',
        RAG_STATE_EVIDENCE_MODE='on',RAG_EVIDENCE_CHAIN_MODE='on',RAG_MULTIHOP_NEEDS='on')
    b=backend(tmp_path,**cfg)
    assert b.semantic_controls.facts=='off' and b.state_evidence.mode=='off'
    assert b.multihop.chain_mode=='off' and b.multihop.needs=='off'
    assert Planner(cfg).raw_evidence


def test_indices_bind_to_program_sources_without_generated_quotes():
    def complete(prompt,payload,schema,timeout):
        assert 'do not decide which' in prompt
        return schema(selected=[1,0,1],stop_search=False,missing='outcome',queries=[])
    result=select(SimpleNamespace(complete=complete),'question',[], 'chain', hits(2),[],20)
    assert [s.source_id for s in result.supports]==['1','0']
    assert result.supports[0].quote==hits(2)[1]['content']
    assert not result.sufficient


def test_later_hop_cannot_erase_previously_selected_original():
    from memory.multihop import Route, Review, Support, Query
    from memory.aml_api import AMLSearch
    class Steps:
        calls=0
        def route(self,*args): return Route(strategy='chain',queries=[])
        def review(self,*args):
            self.calls+=1
            first=self.calls==1
            return Review(sufficient=not first,missing='employer' if first else '',
                supports=[Support(source_id='a' if first else 'b',
                    quote='Alice knows Bob.' if first else 'Bob works at Atlas.',needed_for='source')],
                queries=[Query(query='Where does Bob work?',source_id='a',bridge='Bob')] if first else [])
    rows=[dict(id='a',content='Alice knows Bob.',score=2),dict(id='b',content='Bob works at Atlas.',score=1)]
    trace={}
    result=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw'),Steps()).run(
        AMLSearch(user_id='u',query='Where does the person Alice knows work?',top_k=1),
        lambda p:dict(data=rows),lambda q,d:[1]*len(d),trace)
    assert all(h['content'] in result['data'][0]['content'] for h in rows)


@pytest.mark.parametrize('index',[-1,3])
def test_out_of_range_selection_cannot_bind(index):
    def complete(prompt,payload,schema,timeout):
        return schema(selected=[index],stop_search=True,missing='',queries=[])
    with pytest.raises(ValueError):
        select(SimpleNamespace(complete=complete),'question',[],'direct',hits(2),[],20)
