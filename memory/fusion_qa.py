"""Bounded context/target fusion and source-local answer support (no LLM)."""
import re
import numpy as np


def fuse_scores(rows, candidates, context, target):
    context, target = np.asarray(context, dtype=float), np.asarray(target, dtype=float)
    if context.shape != target.shape or not np.isfinite(target).all():
        raise ValueError('Invalid target scores')
    # Short elliptical replies need their context; complete facts may benefit
    # more from their independent score than the original +/-0.5 adjustment.
    floors = []
    for i in candidates:
        text = rows[i]['content'].strip()
        elliptical = len(text.split()) <= 35 and re.match(
            r"^(yeah|yes|no|thanks|thank you|it|they|that|those|sure|super|great|wow|i['’]m|we['’]re)\b", text, re.I)
        floors.append(0.0 if elliptical else -1.0)
    return context + 0.35 * np.minimum(np.maximum(target-context, floors), 4.0)


def answer_support(query, rows, order, scores):
    """Credit only a direct reply to a relevant question, once, within a session."""
    values = {i: float(scores[i]) for i in order}
    if not order or re.search(r'\b(ask|asked|question|questions)\b', query, re.I):
        return order, values
    terms = set(re.findall(r'\w+', query.lower())) - set(
        'what when where who why how which did does do is are was were a an the to of in on and or both his her their'.split())
    original = dict(values)
    for i in order[:20]:
        question = rows[i]['content'].strip()
        j = i+1
        if not question.endswith('?') or j >= len(rows):
            continue
        q, a = dict(rows[i]), dict(rows[j])
        if q['session_id'] != a['session_id'] or q.get('source_id') == a.get('source_id'):
            continue
        if q.get('source_index') is None or a.get('source_index') != q['source_index']+1:
            continue
        speaker_changed = q.get('speaker') and a.get('speaker') and q['speaker'] != a['speaker']
        role_changed = q.get('role') and a.get('role') and q['role'] != a['role']
        if not speaker_changed and not role_changed:
            continue
        reply = a['content'].strip()
        if reply.endswith('?') or len(reply.split()) < 4:
            continue
        if not terms.intersection(re.findall(r'\w+', question.lower())):
            continue
        if original[i] < original[order[0]]-3.0:
            continue
        values[j] = max(values.get(j, -float('inf')), original[i]-0.75)
    result = sorted(values, key=lambda i: (-values[i], -original.get(i, -float('inf')), i))
    return result, values
