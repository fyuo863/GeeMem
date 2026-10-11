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


class SourceVote(StrictModel):
    index: int = Field(ge=0, strict=True)
    relevant: bool = Field(strict=True)


class PerSourceSelection(StrictModel):
    votes: list[SourceVote] = Field(max_length=24)
    stop_search: bool
    missing: str = Field(max_length=500)


class PerSourceGaps(PerSourceSelection):
    gaps: list[RetrievalGap] = Field(default_factory=list, max_length=2)


class JointSourceSelection(SourceSelection):
    selected: list[int] = Field(max_length=24)


class JointSourceGaps(JointSourceSelection):
    gaps: list[RetrievalGap] = Field(default_factory=list, max_length=2)


VOTE_PROMPT = '''For EVERY supplied index return one relevant boolean; do not rank or pick a fixed number.
Relevant means its own content supplies the requested subject's attribute, a change
to that attribute, or an intermediate relationship needed by the question.
For an attribute-value question, require an explicit asserted value or an explicit
change in the target content. A question asking for that value, a conditional
prediction, or advice about that topic alone is false. The surrounding context
must not supply a missing value on behalf of the target. Judge each independently.
Keep earlier AND later values, corrections and explicit updates of the same subject
and attribute. Keep historical facts even for current questions; do not decide which
value wins. Exclude generic advice, hypothetical conditions, unrelated attributes
and repeated commentary without a useful fact. Context only resolves references;
it does not make an unrelated target content relevant. Source times are message
times, not necessarily event times. Do not answer, calculate dates or rewrite facts.
stop_search means no useful further lookup is needed. If no concrete retrieval gap
remains, set stop_search=true, missing="", queries=[]. Otherwise describe only the
absent fact and propose at most 3 grounded short queries, never a tried query.
Each query needs source_id from evidence or __question__, and bridge copied exactly
from that source/question AND included in the query.'''


def compact_evidence(evidence):
    """Keep stable original indices while presenting a chronological source list."""
    items = []
    for index, source in enumerate(evidence):
        item = dict(source, index=index)
        context = item.pop('context', '')
        # The retriever's context repeats Target message before adjacent messages.
        if context.startswith('Target message: '):
            pos = context.find('\nPrevious context:')
            context = context[pos+1:] if pos >= 0 else ''
        if context:
            context = context.replace(source['content'], '') if source['content'] else context
            item['context'] = context[:400]
        items.append(item)
    return sorted(items, key=lambda e: (e.get('created_at') is None,
                                       e.get('created_at') or '', e['index']))


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


JOINT_PROMPT = PROMPT.replace('Select at most 8 distinct indices.', '''Select up to 24 distinct indices jointly, not independently.
First locate the starting entity in the question, then follow each requested relationship using ALL supplied original sources. Select every bridge as well as endpoints. A terminal fact about an unconnected entity is irrelevant.
For each selected relationship that has an explicit correction, also follow the replacement entity through the remaining relationships and preserve BOTH paths, including their starting link and available endpoints. Earlier facts are useful evidence even if replaced. Do not return just the winning/current path.
If only a partial path exists, keep it and search for its missing link, not an unrelated endpoint.''')

LINKED_SOURCE_PROMPT = ''' A relationship chain may be supported by DIFFERENT sources joined
through the same intermediate entities. Do not require a single sentence explicitly
restating the entire nested question or the endpoint "in the context" of the start.
Before reporting a missing link, check ALL supplied sources for that entity and
relation, including sources not selected yet. If each requested link is present,
set stop_search=true, missing="", queries=[]. Do not ask again for an endpoint
already supplied. Preserve old and updated routes without choosing a winning truth.
candidate_paths, when supplied, are program-linked indices from explicit source
triples, not inferred answers. Read their sources together; they may connect an
endpoint which does not mention the question's starting entity. They are candidates,
not proof that all question conditions are satisfied. Paths may stop at an inner
relation when the final attribute is absent; a path alone never proves sufficiency.'''


def select(planner, question, options, strategy, evidence, history, timeout):
    from pydantic import create_model
    from .multihop import Query, Review, RawReview, Support, CollectionReview
    coverage = getattr(planner, 'coverage', False)
    joint = (strategy == 'chain' and getattr(planner, 'raw_chain_selection', 'legacy') == 'joint'
             and getattr(planner, 'raw_selection', 'legacy') == 'per_source')
    per_source = getattr(planner, 'raw_selection', 'legacy') == 'per_source' and strategy == 'direct'
    base = ((PerSourceGaps if coverage else PerSourceSelection) if per_source else
            (JointSourceGaps if coverage else JointSourceSelection) if joint else
            (GapSelection if coverage else SourceSelection))
    schema = create_model('RawSourceVotes' if per_source else 'RawSourceSelection', __base__=base,
                          queries=(list[Query], Field(default_factory=list, max_length=3)))
    prompt = JOINT_PROMPT if joint else VOTE_PROMPT if per_source else PROMPT
    from .source_paths import candidate_paths
    paths = candidate_paths(question, evidence) if joint else []
    if paths: prompt += LINKED_SOURCE_PROMPT
    payload = dict(question=question, options=options,
        strategy=strategy, evidence=compact_evidence(evidence) if per_source else [dict(e, index=i) for i,e in enumerate(evidence)],
        tried_queries=history)
    if paths: payload['candidate_paths'] = paths
    result = planner.complete(prompt+(GAP_PROMPT if coverage else ''), payload, schema, timeout)
    if per_source:
        indices = [v.index for v in result.votes]
        if len(indices) != len(set(indices)) or set(indices) != set(range(len(evidence))):
            raise ValueError('Incomplete or duplicate source relevance votes')
        selected = [v.index for v in result.votes if v.relevant]
    else:
        selected = result.selected
    # Retain explicit candidate-path originals even if a model omits an endpoint.
    # This does not override its sufficiency or generate facts.
    selected = list(dict.fromkeys(selected + [i for path in paths for i in path]))
    supports = []
    # A source selected as the basis of a companion lookup is itself part of
    # the retrieval trail, even if the model omits it from `selected`.
    indices=([g.source_index for g in result.gaps]+selected) if coverage else selected
    for index in list(dict.fromkeys(indices))[:24 if per_source or joint else 8]:
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
    return (RawReview if per_source or joint else Review)(sufficient=result.stop_search, supports=supports,
                  missing=result.missing, queries=[] if result.stop_search else result.queries)


def raw_settings(cfg):
    """A single mode overrides semantic adjudicators, even with old config keys."""
    if cfg.get('RAG_EVIDENCE_BUNDLE_MODE') != 'raw':
        return cfg
    return dict(cfg, RAG_EVIDENCE_CHAIN_MODE='off', RAG_FACT_REPLACEMENT_MODE='off',
                RAG_STATE_EVIDENCE_MODE='off', RAG_MULTIHOP_BINDINGS='off',
                RAG_MULTIHOP_NEEDS='off', RAG_MULTIHOP_PROMPT_STYLE='long')
