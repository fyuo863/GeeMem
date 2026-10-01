# Hit@10 optimization from actual retrieval failures — 2026-10-01

## Findings before editing ranking

Current comparator is contextual Cross-Encoder reranking (400 candidates, +/-1 context,
10 original messages), not the original no-reranker RAG. Development: 241 questions in
first 3 conversations. 32 questions had zero hits: 1 had no gold in candidates, 31
were ranking failures. In 8 zero-hit questions, a gold message was immediately adjacent
to a selected message. Among misses with gold candidates, earliest gold rank had
median 65 (quartiles 40.5/155, range 12–331). Therefore expanding only rank 11 or
increasing candidate count cannot solve most misses.

## Development experiments (cached real-model scores)

All candidate text/vectors and query scores came from frozen, gold-independent prior
retrieval. No model calls or gold-based query rules were added. Only development labels
were used to select a general selection rule. Full results: data/hit10-experiments/dev-all-report.json.

| Selection | Development Hit@10 |
|---|---:|
| Existing contextual reranker | 86.72% |
| Hybrid-rank fusion (weight 0.25 / 0.5 / 1) | 83.82 / 77.18 / 74.27% |
| MMR (lambda 0.5 / 0.7 / 0.9) | 73.86 / 78.01 / 81.33% |
| Session cap (2 / 3) | 71.78 / 78.84% |
| Reserve original-retrieval slots (2 / 4) | 84.23 / 82.16% |
| Keep 9 reranked, fill one neighbor | 87.55% |
| Single-hop context support, score penalty 2 | **87.55%** |

Context support won the tie on macro evidence recall: 83.44% versus 83.09%.
Penalties 0.25, 0.5 and 1 scored 84.23%, 85.06%, 86.31% Hit, respectively; smaller
penalties overpromoted neighbors. Strong diversity/caps hurt because good evidence
frequently clusters in the same conversation episode. Do not equate adjacent with useless.

## Implemented rule

For each reranked message i, assign its immediately adjacent same-session message j:
`final(j) = max(original(j), original(i) - 2)`.
Only original scores propagate, once; no chaining or cross-session propagation.
Neighbors may enter from outside the top-400 scored candidates if they provided context.
Missing original score is negative infinity. Ties use original score then insertion
position. Sort resulting scores and return at most ten unchanged source messages.
No extra context is concatenated into returned content, no user boundary is crossed.
Penalty 2 is model/logit specific, not a calibrated probability threshold.

## Fixed validation results

Seven remaining conversations, 618 text questions; penalty frozen before evaluation.
Same split as earlier public-data tests, not an independent unseen external test.
The code does not inspect evidence IDs or answers when selecting messages.

| Scope | Variant | Hit@10 | Recall@10 | Micro Recall@10 | All-evidence Hit@10 |
|---|---|---:|---:|---:|---:|
| development | context | 86.72% | 82.47% | 76.08% | 78.42% |
| development | propagate_2 | 87.55% | 83.44% | 77.41% | 79.67% |
| validation | context | 78.32% | 74.47% | 70.12% | 70.87% |
| validation | propagate_2 | 80.42% | 77.02% | 72.70% | 73.79% |
| aggregate | context | 80.68% | 76.72% | 71.79% | 72.99% |
| aggregate | propagate_2 | 82.42% | 78.82% | 74.02% | 75.44% |

Validation: 15 newly hit questions, 2 lost, net +13 (484 -> 497 of 618), gain 2.10
percentage points. Aggregate Hit@10: 693 -> 708 of 859, 80.68% -> 82.42%.
The aggregate includes development selection and is not an independent test score.
90% Hit/Recall is still NOT reached. No further parameter selection used validation.

## Concrete evidence example

Question: What does Joanna do after receiving a rejection from a production company?

- D24:13 discusses the rejection and encourages Joanna to keep trying.
- D24:14: "Yeah.. Thanks, Nate. It's hard, but I won't let it slow me down.
  I'm gonna keep grinding and moving ahead."
- D24:15 praises her resilience.

Gold answer-bearing evidence is D24:14: it answers what Joanna does, but omits the
word rejection. A context-scored neighboring message may outrank this reply. The
new selection recovers D24:14, turning a complete miss into a hit. Other recovered
questions concern John's mentoring younger players and Audrey's pets' personalities.

Regressions remain: two questions about which country James booked tickets for in
July 2022 / visited in 2021 lost their previously retrieved evidence (D16:9 / D6:12).
Context support can still displace precise time/location evidence with nearby messages.

## Validation, cost and configuration

87 tests passed, including single-hop-only transfer, session boundaries, exact original
content, score validation and prevention of double result-window expansion. Eight actual
API queries (new hits, regressions and ordinary queries) matched the cached experiment
IDs exactly. Real-model synthetic Add/Search smoke also passed. No LLM calls.

This experiment reuses previously computed cross-encoder scores for full-corpus
comparison; it is NOT a new end-to-end latency benchmark. Extra selection-only timing:
P50 0.606 ms, P95 0.881 ms.
No extra model forward pass; Add and embeddings are unchanged. CPU/GPU inference cost
remains that of the preceding contextual reranker. Production was not switched.

Opt-in root .env configuration (all previously documented reranker settings still apply):

```dotenv
RAG_RERANK_MODE=local
RAG_RERANK_CANDIDATES=400
RAG_RERANK_CONTEXT=1
RAG_RERANK_SELECTION=context_support
RAG_RERANK_NEIGHBOR_PENALTY=2
RAG_RESULT_WINDOW=0
RAG_TAG_MODE=off
```

Default selection remains direct. The new mode rejects missing reranker, context !=1
or nonzero result window rather than silently combining two neighbor expansion rules.

Reproduce cached selection: python scripts/benchmark_hit10.py --split dev;
then python scripts/benchmark_hit10.py --split heldout --method propagate_2.
All reports/cases/API verification are in ignored local data/hit10-experiments/.
The shared scoring helper is used by the experiment AND production search path.

Remaining work: the majority of zero-hit cases are still not repaired by local context
attribution. Improving target-message relevance or a stronger conversation-aware
reranker is more promising than adding more diversity or more low-ranked candidates.
A new external evaluation split is needed before claiming broader generalization.
