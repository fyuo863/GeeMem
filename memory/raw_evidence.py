"""Select relevant original sources; leave factual adjudication to the caller."""
from pydantic import Field
from typing import Literal
from .models import StrictModel


class SourceSelection(StrictModel):
    selected: list[int] = Field(max_length=8)
    stop_search: bool
    missing: str = Field(max_length=500)
    queries: list = Field(default_factory=list, max_length=3)


class RetrievalGap(StrictModel):
    kind: Literal['intermediate', 'cause', 'earlier', 'update', 'plan', 'change', 'outcome', 'rule_context']
    source_index: int = Field(ge=0, strict=True)


class GapSelection(SourceSelection):
    gaps: list[RetrievalGap] = Field(default_factory=list, max_length=2)


GAP_HINTS = {
    'intermediate': 'Find the intermediate relationship or connecting passage. 查找中间关系原文。',
    'cause': 'Find an explicit explanation of the cause or reason. 查找明确解释原因的原文。',
    'earlier': 'Find earlier statements of this same subject and attribute. 查找同主体同属性的旧记录。',
    'update': 'Find corrections or subsequent updates of this same subject and attribute. 查找同主体同属性的更正和后续更新。',
    'plan': 'Find the original plan for this same event. 查找同一事件的原计划。',
    'change': 'Find changes or cancellation of this same event. 查找同一事件的变更或取消。',
    'outcome': 'Find the actual outcome of this same event. 查找同一事件的实际结果。',
    'rule_context': 'Find conditions, exceptions or revisions of this same rule. 查找同一规则的条件、例外或修订。',
}


GAP_PROMPT = ''' In gaps select at most two retrieval gap kinds and the index of a
relevant supplied source to anchor each lookup. Only request genuinely absent
companion sources needed for this question: intermediate, cause, earlier, update,
plan, change, outcome, rule_context. Never request an already present phase or an
unrelated attribute. For current-value questions inspect whether a relevant earlier
or corrective statement is absent; do not request all history for ordinary facts.
These are retrieval needs, not factual verdicts. Use gaps=[] when no gap remains.
For a selected gap leave queries=[]; the program constructs its grounded lookup.'''


PROMPT = '''Select relevant source indices for an upstream answering model.
Preserve intermediate relationship passages AND terminal passages. For changing
facts, keep earlier and later statements about the SAME subject and attribute,
including corrections, conflicting claims and relevant plans; do not decide which
value wins. For event history keep the original plan, changes and actual outcome
when available, including standalone earlier sources. Do not infer causal links,
declare a chain valid, calculate dates, or generate an answer. Exclude unrelated
people, attributes and events. Each selected index refers only to that source's
content, not its context. Select at most 8 distinct indices.
stop_search means no useful further lookup is needed, NOT proof of correctness
or completeness. If relevant intermediate/earlier/stage sources are missing,
describe the retrieval gap and propose a short query using only known entities.
Each query uses source_id from evidence or __question__, and a short bridge
copied literally from that source/question and present in the query. Never repeat
a tried query. Return queries=[] when stopping or no grounded query is available.
Source timestamps are message times, not necessarily event times.'''


def select(planner, question, options, strategy, evidence, history, timeout):
    from pydantic import create_model
    from .multihop import Query, Review, Support, CollectionReview
    coverage = getattr(planner, 'coverage', False)
    schema = create_model('RawSourceSelection', __base__=GapSelection if coverage else SourceSelection,
                          queries=(list[Query], Field(default_factory=list, max_length=3)))
    result = planner.complete(PROMPT+(GAP_PROMPT if coverage else ''), dict(question=question, options=options,
        strategy=strategy, evidence=[dict(e, index=i) for i,e in enumerate(evidence)],
        tried_queries=history), schema, timeout)
    supports = []
    # A source selected as the basis of a companion lookup is itself part of
    # the retrieval trail, even if the model omits it from `selected`.
    indices=([g.source_index for g in result.gaps]+result.selected) if coverage else result.selected
    for index in list(dict.fromkeys(indices))[:8]:
        if index < 0 or index >= len(evidence):
            raise ValueError('Raw evidence selection outside source set')
        source = evidence[index]
        # This substring is a program-bound provenance check, not generated proof.
        supports.append(Support(source_id=source['id'], quote=source['content'][:600],
                                needed_for='related original source'))
    if coverage:
        queries=[]; kinds=[]
        for gap in result.gaps:
            if gap.source_index >= len(evidence):
                raise ValueError('Gap anchor outside source set')
            source=evidence[gap.source_index]
            anchor=source['content'][:80]
            if len(anchor)<2: continue
            query=Query(query=question[:180]+'\nSource: '+anchor+'\n'+GAP_HINTS[gap.kind],
                        source_id=source['id'],bridge=anchor)
            if query.query not in [q.query for q in queries]:
                queries.append(query); kinds.append(gap.kind)
        return CollectionReview(sufficient=result.stop_search and not queries, supports=supports,
            missing=result.missing, queries=[] if result.stop_search else result.queries,
            gap_queries=queries, gap_kinds=kinds)
    return Review(sufficient=result.stop_search, supports=supports,
                  missing=result.missing, queries=[] if result.stop_search else result.queries)


def raw_settings(cfg):
    """A single mode overrides semantic adjudicators, even with old config keys."""
    if cfg.get('RAG_EVIDENCE_BUNDLE_MODE') != 'raw':
        return cfg
    return dict(cfg, RAG_EVIDENCE_CHAIN_MODE='off', RAG_FACT_REPLACEMENT_MODE='off',
                RAG_STATE_EVIDENCE_MODE='off', RAG_MULTIHOP_BINDINGS='off',
                RAG_MULTIHOP_NEEDS='off', RAG_MULTIHOP_PROMPT_STYLE='long')
