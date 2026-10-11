import pytest

from memory.supplemental import retrieval_gap, supplemental_query
from memory.multihop import MultiHop, Route, Review, Support, Query
from memory.aml_api import AMLSearch


@pytest.mark.parametrize('text', [
    '', 'None', 'N/A', 'No missing links found.',
    'No relevant intermediate or earlier sources are missing.',
    'No relevant intermediate or earlier sources are missing. No additional evidence is needed.',
    'All required evidence is available.', '没有缺失来源。', '无缺口',
    '所有必要证据均已提供。',
])
def test_no_gap_cannot_become_a_search_query(text):
    assert retrieval_gap(text)==''
    assert supplemental_query('How did the plan change?',text) is None


@pytest.mark.parametrize('text', [
    'No source gives the final outcome.', 'The outcome is not known.',
    'No missing intermediate links, but the outcome is missing.',
    'No missing links found. Need the completion date.',
    '没有缺失中间来源，但缺少最终结果。', '未找到新老师的乐器信息。',
    'Missing evidence about None, the project codename.',
    'No evidence found.', 'No relevant sources.',
    '没有相关证据。', '无来源。',
])
def test_real_or_mixed_gap_is_not_silently_removed(text):
    assert retrieval_gap(text)==text
    assert supplemental_query('How did the plan change?',text)[1].endswith(text)


def run_review(missing, *, neighbors=False, query=False):
    rows=[dict(id='a',content='Alice planned a trip.',score=1)]
    calls=[]; expansions=[]; trace={}
    class Planner:
        def route(self,*args): return Route(strategy='chain',queries=[])
        def review(self,*args):
            return Review(sufficient=False, missing=missing,
                supports=[Support(source_id='a',quote=rows[0]['content'],needed_for='original')],
                queries=[Query(query='Alice trip outcome',source_id='a',bridge='Alice')] if query else [])
    class Retriever:
        def __call__(self,p):
            calls.append(p.query)
            return dict(data=rows)
        def expand_neighbors(self,rows,gap):
            expansions.append(gap)
            return []
    result=MultiHop(dict(RAG_MULTIHOP_MODE='llm',RAG_EVIDENCE_BUNDLE_MODE='raw',
        RAG_SUPPLEMENTAL_MODE='on',RAG_EVIDENCE_GROUP_MODE='adjacent' if neighbors else 'off'),Planner()).run(
        AMLSearch(user_id='gap-test',query='How did Alice trip plan change?',top_k=2),
        Retriever(),lambda q,d:[1]*len(d),trace)
    return result,trace,calls,expansions


@pytest.mark.parametrize('neighbors',[False,True])
def test_explicit_no_gap_blocks_automatic_retrieval_without_losing_sources(neighbors):
    text='No relevant intermediate or earlier sources are missing.'
    result,trace,calls,expansions=run_review(text,neighbors=neighbors)
    assert len(calls)==1
    assert not expansions and not trace.get('supplemental_calls')
    assert trace['llm_calls']==2
    assert trace['rounds'][0]['prose_gap_status']=='explicit_none'
    assert trace['rounds'][0]['review']['missing']==text
    assert 'Alice planned a trip.' in result['data'][0]['content']


def test_actual_gap_still_triggers_one_bounded_supplement():
    _,trace,calls,_=run_review('The actual outcome is missing.')
    assert len(calls)==2 and trace['supplemental_calls']==1
    assert 'Missing evidence: The actual outcome is missing.' in calls[1]


def test_no_gap_prose_does_not_cancel_a_grounded_model_query():
    _,trace,calls,_=run_review('No missing links found.',query=True)
    assert 'Alice trip outcome' in calls
    assert not trace.get('supplemental_calls')
