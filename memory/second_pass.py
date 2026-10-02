"""One bounded evidence-guided retrieval pass over caller-scoped source rows."""
from collections import Counter
import re
import numpy as np
from .tags import RuleTagger


def additional_candidates(query, rows, candidates, scores, lexical_score, limit=40,
                          top_threshold=4.0, gap_threshold=2.0):
    if not candidates or limit <= 0:
        return []
    ranked = sorted(range(len(candidates)), key=lambda j: -scores[j])
    top = float(scores[ranked[0]])
    tail = float(scores[ranked[min(9, len(ranked)-1)]])
    multiple_or_time = re.search(r'\b(when|before|after|activities|events|hobbies|countries|cities|list|both|all)\b', query, re.I)
    if top >= top_threshold and top-tail >= gap_threshold and not multiple_or_time:
        return []
    # Clues must occur in at least two highly ranked source messages. This is
    # source corroboration, not a claim of factual/semantic verification.
    tagger = RuleTagger()
    seeds = [candidates[j] for j in ranked[:3] if scores[j] >= 4]
    tags = tagger.extract([r['content'] for r in rows])
    counts = Counter(t for i in seeds for t in tags[i])
    frequency = Counter(t for row_tags in tags for t in row_tags)
    query_tags = set(tagger.extract([query])[0])
    clues = sorted((t for t, n in counts.items() if n >= 2 and t not in query_tags),
                   key=lambda t: (frequency[t], t))[:3]
    windows = [' '.join(rows[j]['content'] for j in range(max(0,i-2), min(len(rows),i+3))
                       if rows[j]['session_id'] == row['session_id']) for i,row in enumerate(rows)]
    # Keep a global original-query route, so a wrong seed cannot fence off the search.
    original = np.asarray(lexical_score(windows, query), dtype=float)
    expanded = np.asarray(lexical_score(windows, query+' '+' '.join(clues)), dtype=float) if clues else original
    votes = np.zeros(len(rows))
    for values in (original, expanded):
        for rank, i in enumerate(sorted((i for i in range(len(rows)) if values[i] > 0), key=lambda i: (-values[i],i)), 1):
            votes[i] += 1/(60+rank)
    present = set(candidates)
    return sorted((i for i in range(len(rows)) if i not in present and votes[i] > 0),
                  key=lambda i: (-votes[i],i))[:limit]
