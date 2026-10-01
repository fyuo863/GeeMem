# Recall@10 target 90% — first optimization round (2026-10-01)

**The 90% target is NOT reached.** Contextual cross-encoder reranking gives a real
improvement, but does not justify claiming success or automatically changing production.

## Fixed metric and split

Macro evidence Recall@10 is the current target; also report micro recall so "90% of
all evidence" is not confused with an equal-weight average over questions. Results
return at most 10 original messages, never 10 artificially enlarged bundles. No LLM
calls, generated answers, gold-aware filtering, benchmark-specific aliases or evidence
injection into model inputs. Text-only 859 questions, same canonical evidence indexing.

Development: first 3 conversations, 241 questions. Fixed validation: remaining 7,
618 questions. These were previously observed in aggregate full-public-set reports,
so validation is held aside from this round's parameter selection, not an unseen
external test. Combined results include development and are descriptive, not an
independent generalization score.

## Development experiments

| Recipe | Macro Recall@10 |
|---|---:|
| Original hybrid retrieval with result window | 63.59% |
| Disable result window, no reranker | 51.63% |
| Plain-message cross-encoder, 200 candidates | 48.48% |
| Contextual cross-encoder, 200 candidates, context +/-1 | 80.19% |
| Contextual cross-encoder, 400 candidates, context +/-2 | 76.73% |
| Contextual cross-encoder, 400 candidates, context +/-1 | **82.47%** |

Selected ONLY using these development results: 400 candidates, context +/-1, no
post-ranking result-window insertion. Model: cross-encoder/ms-marco-MiniLM-L-6-v2,
revision 233902d25c440f23af6f7d6e94d2946bac0bee0a, max length 512, batch size 32.
Query and target message are scored with previous/next same-session message text.
Context is not included in returned content. Selection was frozen before validation.

## Frozen validation and combined results

| Scope | Variant | Recall@10 | Hit@10 | Micro Recall@10 | All evidence hit@10 |
|---|---|---:|---:|---:|---:|
| Development (241) | Original | 63.59% | 67.22% | 57.48% | 60.17% |
| Development (241) | Contextual reranker | 82.47% | 86.72% | 76.08% | 78.42% |
| Validation (618) | Original | 65.64% | 68.45% | 62.35% | 62.78% |
| Validation (618) | Contextual reranker | 74.47% | 78.32% | 70.12% | 70.87% |
| Combined (859) | Original | 65.07% | 68.10% | 60.99% | 62.05% |
| Combined (859) | Contextual reranker | 76.72% | 80.68% | 71.79% | 72.99% |

Validation: 98 questions improved, 39 worsened, 481 tied. Macro Recall@10 gained
8.83 percentage points but remains 15.53 points below the target. Combined gain is
11.65 points, from 65.07% to 76.72%. Combined Micro Recall@10 is 71.79% (771/1074).
If the target means 90% of all annotated evidence rather than macro recall, that target
also remains unmet and requires at least 967 evidence hits under this denominator.

## Bottleneck diagnosis

Combined candidate-400 evidence coverage is 98.17% macro, 1049/1074 micro. Only 25
evidence occurrences are absent before reranking, while 278 of the retrieved evidence
occurrences fall outside the final top 10. Coverage@400 is a diagnostic upper bound,
NOT Recall@10 and NOT a measured oracle ranking system. The main remaining loss is
ranking/selection. Context often helps resolve replies, but also makes adjacent
non-evidence messages seem relevant; longer context hurt development results.

Next experiments should focus on conversation-aware scoring, attribution of support
to the target message (rather than its neighbors), and redundancy/diversity within ten
results. A stronger or in-domain trained reranker is a hypothesis to validate, not an
assumed solution. Further tuning must stay off this validation split, or establish a
new external held-out dataset before making independent accuracy claims.

## Runtime and verification

Validation Search P50: about 190 ms with CPU BGE retrieval + RTX 4070 cross-encoder,
versus historical original P50 about 36 ms. Different runs; not a strict concurrent
throughput comparison. No additional Add work or re-embedding is required. Queries
are more expensive because up to 400 query/message pairs are scored. CPU-only latency
has not been measured for this reranker.

84 unit/API tests passed. Actual memory.aml_api Search returned exactly the benchmark
IDs on 3 sampled validation queries; a real-model synthetic Add/Search API smoke also
passed (user isolation, original text, retries, options, timestamps and top_k). Models
load exclusively from verified local files; downloads use root .env explicit proxy.
Default RAG_RERANK_MODE remains off and the live service was not changed.

## Artifacts and reproduction

Implementation checkpoint d30cacd, with subsequent benchmark parameter flags recorded
in this report's commit. Local ignored data/rerank-benchmarks runs:

- 20261001T021202Z: development plain/context, 200 candidates.
- 20261001T021335Z: development context +/-2, 400 candidates.
- 20261001T021521Z: selected development context +/-1, 400 candidates.
- 20261001T021642Z: frozen validation, plus api-smoke.json.
- recall10-summary.json: aggregate metrics.

Each run has report.json, cases.jsonl and copied memory.sqlite3. The validation database
also contains isolated synthetic smoke records added AFTER the benchmark; original
public-user rows were not changed. Current benchmark does not generate final answers.

See docs/v1-vanilla-rag.md for opt-in configuration and commands. The script diagnoses
internal candidate lists >100 via a direct store call; public AML top_k remains <=100.
