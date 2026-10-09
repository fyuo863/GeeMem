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
