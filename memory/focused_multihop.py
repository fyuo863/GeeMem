"""One evidence need per LLM read; program-owned dependencies and stop control."""
import time
from typing import Literal
import httpx
import numpy as np
from pydantic import Field
from .models import StrictModel
from .llm import LLMError
from .multihop import normalized, ROUTE_PROMPT
from .multihop_evidence import RequiredRoute, BoundQuery, NEED_PLAN


READ_PROMPT = '''Check ONLY current_need using the passages and resolved predecessors.
Match the subject, exact relation and object. Do not transfer facts between people;
advice/plans are not completed events. Preserve time/scope. Message time is not
event time. Return supported only with exact source quotes proving this need.
Extract short literal values (names, amounts, dates or facts) from those quotes.
Otherwise return missing, conflict or time_unknown. For a list, partial examples
do not prove completeness. Do not answer the original question or propose queries.'''

QUERY_PROMPT = '''Generate ONE lookup for current_need, preserving its person/time
scope. Use a short exact anchor from an allowed source and an exact unique quote;
template has exactly one {target}. For a dependent need the anchor MUST equal a
resolved predecessor value. Q0 is allowed only without dependencies. Do not
invent entities elsewhere in the template. Do not repeat a tried query. If no
useful grounded lookup exists, return query=null. Do not judge sufficiency.'''


class Fact(StrictModel):
    source_ref: str = Field(min_length=1, max_length=128)
    quote: str = Field(min_length=3, max_length=600)
    value: str = Field(min_length=1, max_length=160)


class StepRead(StrictModel):
    status: Literal['supported', 'missing', 'conflict', 'time_unknown']
    facts: list[Fact] = Field(default_factory=list, max_length=6)
    reason: str = Field(max_length=240)


class StepQuery(StrictModel):
    query: BoundQuery | None


def requirements_from(plan):
    requirements = []
    for i, need in enumerate(plan.requirements):
        if any(type(d) is not int or d < 0 or d >= i for d in need.depends_on):
            raise ValueError('Dependencies must reference earlier needs')
        requirements.append(dict(id=f'N{i+1}', description=need.description,
            kind=need.kind, depends_on=[f'N{d+1}' for d in need.depends_on]))
    return requirements


def validate_read(raw, sources):
    """Exact provenance is necessary, not a relation-entailment proof."""
    valid, errors = [], []
    for fact in raw.facts:
        source = sources.get(fact.source_ref)
        if (source is None or source['id']=='__question__'
                or fact.quote not in source['content'] or fact.value not in fact.quote):
            errors.append('Invalid source, quote or literal value')
            continue
        offset = source['content'].index(fact.quote)
        valid.append(dict(source_id=source['id'], source_ref=fact.source_ref,
                          quote=fact.quote, value=fact.value, quote_start=offset))
    status = raw.status
    if status=='supported' and (not valid or errors):
        status = 'missing'
    return status, valid, errors


def bind_query(raw, sources, predecessor_values, dependent, history):
    if raw.query is None:
        return None
    q = raw.query
    source = sources.get(q.source_ref)
    if (source is None or source['content'].count(q.quote)!=1
            or q.quote.count(q.anchor)!=1 or q.template.count('{target}')!=1):
        return None
    if dependent and (q.source_ref=='Q0' or q.anchor not in predecessor_values):
        return None
    text = q.template.replace('{target}', q.anchor)
    if not 2 <= len(text) <= 500 or normalized(text) in {normalized(x) for x in history}:
        return None
    offset = source['content'].index(q.quote)+q.quote.index(q.anchor)
    return dict(query=text, source_id=source['id'], anchor=q.anchor,
                start=offset, end=offset+len(q.anchor), quote=q.quote)


def run_focused(runner, payload, retrieve, rerank, trace):
    started = time.perf_counter()
    trace.update(mode='llm', prompt_style='focused', strategy=None, llm_calls=0,
                 search_calls=0, rounds=[], rejected_queries=0, rejected_supports=0,
                 binding_registry=[], fallback=False)
    pool, refs, history, routes, states = {}, {}, [], [], {}
    baseline = None
    phase = 'plan'

    def remaining():
        return runner.seconds-(time.perf_counter()-started)

    def call(prompt, data, schema):
        trace['llm_calls'] += 1
        return schema.model_validate(runner.planner.complete(prompt, data, schema,
            min(runner.timeout, max(.01, remaining()))))

    def collect(query, count):
        trace['search_calls'] += 1
        hits = retrieve(payload.model_copy(update={'query':query, 'top_k':count}))['data']
        history.append(query)
        routes.append([h['id'] for h in hits])
        for hit in hits:
            if not np.isfinite(hit['score']):
                raise ValueError('Non-finite retrieval score')
            pool.setdefault(hit['id'], dict(hit))
            refs.setdefault(hit['id'], f'E{len(refs)+1}')
        return hits

    try:
        plan = call(ROUTE_PROMPT+NEED_PLAN, dict(question=payload.query, options=payload.options), RequiredRoute)
        needs = requirements_from(plan)
        trace.update(strategy=plan.strategy, initial_plan=plan.model_dump(), requirements=needs)
        states = {n['id']:dict(need_id=n['id'], status='missing', facts=[]) for n in needs}
        phase = 'original_retrieval'
        baseline = collect(payload.query, payload.top_k)
        blocked = set()
        while remaining()>0 and trace['llm_calls']<runner.llm_limit:
            ready = [n for n in needs if states[n['id']]['status']!='supported'
                     and n['id'] not in blocked
                     and all(states[d]['status']=='supported' for d in n['depends_on'])]
            if not ready:
                break
            need = ready[0]
            parents = [states[d] for d in need['depends_on']]
            values = {f['value'] for s in parents for f in s['facts']}
            query = need['description']+'\nResolved values: '+', '.join(sorted(values))
            ids = list(pool)
            if ids:
                phase = 'packet_rerank'
                scores = np.asarray(rerank(query, [pool[i]['content'] for i in ids]), dtype=float).reshape(-1)
                if len(scores)!=len(ids) or not np.isfinite(scores).all():
                    raise ValueError('Invalid focus reranking scores')
                selected = sorted(range(len(ids)), key=lambda j:(-scores[j],j))[:5]
                packet = [dict(id=refs[ids[j]],content=pool[ids[j]]['content'][:900],
                               created_at=pool[ids[j]].get('created_at')) for j in selected]
            else:
                packet = []
            sources = {refs[i]:dict(id=i,content=pool[i]['content'][:900]) for i in ids if refs[i] in {p['id'] for p in packet}}
            phase = 'read'
            read_payload = dict(question=payload.query, current_need=need,
                                resolved_predecessors=parents, evidence=packet)
            raw = call(READ_PROMPT, read_payload, StepRead)
            status, facts, errors = validate_read(raw, sources)
            trace['rejected_supports'] += len(errors)
            states[need['id']] = dict(need_id=need['id'],status=status,facts=facts,reason=raw.reason)
            step = dict(need_id=need['id'], evidence=packet, read=raw.model_dump(), errors=errors,
                        status=status, query=None)
            trace['rounds'].append(step)
            if status=='supported':
                continue
            if trace['llm_calls']>=runner.llm_limit or trace['search_calls']>=runner.query_limit or remaining()<=0:
                break
            # Only the question or validated predecessor facts can supply a new entity.
            allowed = {}
            if not parents:
                allowed['Q0'] = dict(id='__question__',content=payload.query)
            else:
                for parent in parents:
                    for fact in parent['facts']:
                        allowed[fact['source_ref']] = dict(id=fact['source_id'],content=pool[fact['source_id']]['content'])
            phase = 'query'
            proposal = call(QUERY_PROMPT, dict(question=payload.query,current_need=need,
                resolved_predecessors=parents,allowed_sources=[dict(id=k,content=v['content'][:900]) for k,v in allowed.items()],
                tried_queries=history), StepQuery)
            # Validate only against what the model was allowed to see.
            visible = {k:dict(v,content=v['content'][:900]) for k,v in allowed.items()}
            bound = bind_query(proposal, visible, values, bool(parents), history)
            step['query'] = proposal.model_dump()
            step['accepted_query'] = bound
            if bound is None:
                trace['rejected_queries'] += int(proposal.query is not None)
                blocked.add(need['id'])
                continue
            trace['binding_registry'].append(dict(bound, need_id=need['id']))
            phase = 'subquery_retrieval'
            collect(bound['query'], runner.candidates)
        supports = list(dict.fromkeys(f['source_id'] for state in states.values() for f in state['facts']))
        sufficient = all(s['status']=='supported' for s in states.values()) and bool(states)
        trace.update(need_states=list(states.values()), history=history, support_ids=supports[:8],
                     support_overflow=len(supports)>8, supports_fit=len(supports)<=min(8,payload.top_k))
        trace['stop'] = ('support_overflow' if sufficient and len(supports)>8 else 'sufficient' if sufficient else
            'time_budget' if remaining()<=0 else 'llm_budget' if trace['llm_calls']>=runner.llm_limit else
            'query_budget' if trace['search_calls']>=runner.query_limit else 'unresolved')
        if not pool:
            return {'data':[]}
        phase = 'fusion'
        ids = list(pool)
        scores = np.asarray(rerank(payload.query,[pool[i]['content'] for i in ids]),dtype=float).reshape(-1)
        if len(scores)!=len(ids) or not np.isfinite(scores).all():
            raise ValueError('Invalid fusion scores')
        ranking = sorted(range(len(ids)),key=lambda j:(-scores[j],j))
        fused = {ids[j]:1/(60+rank) for rank,j in enumerate(ranking,1)}
        for route in routes:
            for rank,i in enumerate(route,1):
                fused[i] += .5/(60+rank)
        supported = set(supports[:8])
        ordered = sorted(ids,key=lambda i:(i not in supported,-fused[i],ids.index(i)))[:payload.top_k]
        return {'data':[dict(pool[i],score=fused[i]+float(i in supported)) for i in ordered]}
    except (LLMError,ValueError,TypeError,TimeoutError,httpx.HTTPError) as exc:
        trace.update(fallback=True, stop='fallback',error_phase=phase,
                     error_type=type(exc).__name__,cause_type=type(exc.__cause__ or exc).__name__)
        if baseline is None:
            baseline = collect(payload.query,payload.top_k)
        return {'data':baseline}
    finally:
        trace['seconds']=time.perf_counter()-started
