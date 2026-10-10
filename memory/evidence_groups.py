"""Bounded source-only adjacency expansion through the existing reranker."""
from contextlib import closing
import math
from datetime import datetime, timezone


def expand_groups(backend, payload, anchors, missing):
    from .retrieval_scope import RetrievalScope
    scope = RetrievalScope(backend, payload)
    with closing(backend.connect()) as db:
        rows = [dict(r) for r in db.execute('''SELECT m.*,s.speaker,s.source_id,s.source_index
            FROM rag_memories m LEFT JOIN rag_sources s ON s.memory_id=m.id
            WHERE m.user_id=? ORDER BY m.rowid''', (payload.user_id,))]
    if backend.versions is not None: rows = backend.versions.visible(payload.user_id, rows)
    rows = backend.semantic_controls.rows(payload.user_id,payload.query,rows,getattr(payload,'reference_time',None))
    rows = scope.session_rows(backend.disclosure.rows(rows))
    eligible = scope.eligible(rows)
    sessions = {}
    for i,r in enumerate(rows):
        if i in eligible: sessions.setdefault(r['session_id'], []).append(r)
    locations = {r['id']:(session,i) for session,items in sessions.items() for i,r in enumerate(items)}
    groups=[];seen=set()
    for anchor in anchors[:12]:
        location=locations.get(anchor['id'])
        if location is None: continue
        session,i=location
        window=sessions[session][max(0,i-2):i+5]
        key=tuple(r['id'] for r in window)
        if key in seen or len(window)<2: continue
        seen.add(key)
        if any(len(r['content'])>1200 for r in window) or sum(len(r['content']) for r in window)>3600: continue
        hits=[]
        for r in window:
            h=dict(id=r['id'],content=r['content'],score=0.,_group_anchor=anchor['id'])
            if r['timestamp'] is not None:
                h['created_at']=datetime.fromtimestamp(r['timestamp']/1000,timezone.utc).isoformat().replace('+00:00','Z')
            hits.append(h)
        # Disclosure result also owns masked-view IDs; raw IDs never bypass it.
        hits=backend.disclosure.result(payload.user_id,{'data':hits})['data']
        groups.append(hits)
    if not groups: return []
    docs=['\n'.join(h['content'] for h in g) for g in groups]
    scores=backend._rerank_for_multihop(payload.query+'\nMissing evidence: '+missing[:500],docs)
    if len(scores)!=len(groups) or not all(math.isfinite(float(s)) for s in scores):
        raise ValueError('Invalid group rerank scores')
    order=sorted(range(len(groups)),key=lambda i:-scores[i])
    selected=[];covered=set()
    for i in order:
        ids={h['id'] for h in groups[i]}
        if ids <= covered: continue
        # Avoid nearly identical windows consuming both slots.
        if covered and len(ids & covered)/len(ids) > .5: continue
        selected.append(groups[i]);covered.update(ids)
        if len(selected)==2:break
    return selected


def select_packet_ids(support_ids, groups, routes, seen, limit):
    """Keep whole groups together; preserve at most four existing supports."""
    result=list(dict.fromkeys(support_ids))[:min(4,limit)]
    omitted=[]
    for group in groups:
        new=[i for i in dict.fromkeys(group) if i not in result]
        if len(result)+len(new)<=limit: result.extend(new)
        else: omitted.append(list(group))
    order=[]
    for rank in range(max((len(r) for r in routes),default=0)):
        for route in routes:
            if rank<len(route) and route[rank] not in order: order.append(route[rank])
    blocked={i for g in omitted for i in g if i not in result}
    for ids in ([i for i in order if i not in seen],order):
        for i in ids:
            if len(result)>=limit:break
            if i not in result and i not in blocked:result.append(i)
    return result, omitted
