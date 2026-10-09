"""Bounded time-window candidate selection; timestamps are not event dates."""
from datetime import date
import math


def window_candidates(rows, order, target, budget, ratio=0.7):
    """Reserve a global floor, then fill with interval-overlapping candidates.

    Input order already respects user/session/type scope and semantic ranking.
    Unknown dates remain eligible through the global lane. A mention is only
    a candidate-selection hint, not proof that every fact has that date.
    """
    if not 0 < ratio < 1 or budget < 1:
        raise ValueError('Invalid temporal window budget')
    order = list(dict.fromkeys(order))
    if target is None:
        return order[:budget], {'triggered': False}
    matched = []
    for i in order:
        for mention in dict(rows[i]).get('time_mentions', []):
            if mention.get('confidence', 1.0) < 0.8:
                continue
            try:
                start = date.fromisoformat(mention['start'])
                end = date.fromisoformat(mention['end'])
            except (KeyError, TypeError, ValueError):
                continue
            if start <= end and start <= target.end and end >= target.start:
                matched.append(i)
                break
    global_count = max(1, budget - math.floor(budget * ratio))
    global_lane = order[:global_count]
    global_set = set(global_lane)
    window_lane = [i for i in matched if i not in global_set][:budget-global_count]
    selected = set(global_lane + window_lane)
    for i in order:
        if len(selected) >= budget:
            break
        selected.add(i)
    # Preserve comparable semantic order, including when reranking is disabled.
    result = [i for i in order if i in selected]
    return result, dict(triggered=True, matched_count=len(matched),
                        window_indices=window_lane, global_indices=global_lane,
                        promoted_indices=[i for i in result if i not in set(order[:budget])])


def _overlap(mention, target):
    try:
        start = date.fromisoformat(mention['start'])
        end = date.fromisoformat(mention['end'])
    except (KeyError, TypeError, ValueError):
        return 0.0
    if start > end or start > target.end or end < target.start:
        return 0.0
    confidence = float(mention.get('confidence', 1.0))
    return max(0.0, min(1.0, confidence))


def temporal_scores(rows, order, target, context=1):
    """Return bounded temporal/context scores aligned with *order*.

    A neighboring message receives only a weak score; it cannot outrank a
    semantically relevant dated message by itself.
    """
    if target is None:
        return {i: 0.0 for i in order}
    values = {}
    for i in order:
        mentions = dict(rows[i]).get('time_mentions', [])
        values[i] = max((_overlap(m, target) for m in mentions), default=0.0)
    if context > 0:
        for i in order:
            if values[i]:
                continue
            row = rows[i]
            for distance in range(1, context + 1):
                for j in (i - distance, i + distance):
                    if 0 <= j < len(rows) and rows[j]['session_id'] == row['session_id']:
                        if any(_overlap(m, target) for m in dict(rows[j]).get('time_mentions', [])):
                            values[i] = 0.3
                            break
                if values[i]:
                    break
    return values


def apply_temporal_fusion(rows, order, scores, target, weight=0.1, context=1):
    """Add a small normalized temporal signal without replacing semantic rank."""
    if target is None or not order or weight <= 0:
        return order, scores, {'matched_count': 0, 'context_count': 0}
    temporal = temporal_scores(rows, order, target, context)
    positive = [i for i, value in temporal.items() if value > 0]
    context_count = sum(temporal[i] == 0.3 for i in positive)
    values = [float(scores[i]) for i in order]
    span = max(values) - min(values) if values else 0.0
    # A fixed fraction of the observed score span avoids changing units across
    # embedding/reranker implementations. Context receives 30% of a match.
    scale = span * float(weight)
    fused = scores.copy()
    for i in positive:
        fused[i] += scale * temporal[i]
    result = sorted(order, key=lambda i: (-float(fused[i]), -float(temporal[i]), i))
    return result, fused, {'matched_count': len(positive) - context_count,
                           'context_count': context_count, 'weight': weight}


def soft_quota(rows, order, target, top_k, ratio=0.5):
    """Prefer a bounded number of window hits in final TopK, then backfill."""
    if target is None or top_k < 1:
        return order, {'applied': False, 'window_selected': 0}
    temporal = temporal_scores(rows, order, target, context=0)
    matches = [i for i in order if temporal[i] >= 0.8]
    quota = min(len(matches), max(1, math.ceil(top_k * ratio)))
    if not matches or quota <= 0:
        return order, {'applied': False, 'window_selected': 0}
    preferred = matches[:quota]
    selected = set(preferred)
    result = preferred + [i for i in order if i not in selected]
    return result, {'applied': True, 'window_available': len(matches),
                    'window_selected': len(preferred), 'ratio': ratio}
