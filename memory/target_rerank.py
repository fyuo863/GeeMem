"""Soft target attribution; neither a hard keyword gate nor a session quota."""
import re
import numpy as np


def combine_target_scores(context, target, weight=0.25):
    context, target = np.asarray(context, dtype=float), np.asarray(target, dtype=float)
    if context.shape != target.shape or not np.isfinite(target).all():
        raise ValueError('Invalid target scores')
    # A short elliptical reply must not lose all of its contextual relevance.
    return context + weight * np.clip(target-context, -2.0, 2.0)


def target_penalty(query, row):
    text = row['content'].strip()
    words = re.findall(r"[\w']+", text)
    penalty = 0.0
    asks_about_question = bool(re.search(r'\b(ask|asked|question|questions|wonder|wondered)\b', query, re.I))
    if len(words) <= 35 and text.endswith('?') and not asks_about_question:
        penalty += 0.6
    # A full-string pattern, not a substring filter: "Thanks, I moved to Paris"
    # must retain its potentially useful fact.
    if len(words) <= 16 and re.fullmatch(
        r"(?:thanks|thank you|wow|great|awesome|good luck|you got this|see you|bye)[\s!.,]*(?:[A-Z][a-z]+[!.,]*)?", text):
        penalty += 0.6
    return penalty


def select_targets(query, rows, order, scores):
    adjusted = {i: float(scores[i])-target_penalty(query, rows[i]) for i in order}
    # Stable ties retain the input order. There is no diversity constraint.
    return sorted(order, key=lambda i: -adjusted[i]), adjusted
