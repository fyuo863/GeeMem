# Local keyword semantics: paired experiment, 2026-09-30

Checkpoint before implementation: ae45b04. Implementation tested: 10aa91c.
No LLM calls; rule keyword lists are encoded with local BGE. Fixed pre-test settings:
RRF keyword weight 0.5, hard cosine threshold 0.5, candidate pool 400, window 1,
window seeds 20. No tuning after observing outcomes. Default remains off.

## Protocol

Same public LoCoMo-Refined conv-26: 419 messages, 19 sessions, same 30 evenly spaced
eligible text questions at K=10/100. Same frozen BGE snapshot and original retrieval
settings. Separate databases per arm, shared model, one warmup excluded from timing.
Write and search arm order rotates. Query keywords AND their embeddings are generated
inside each timed Search, without caching them across K. Gold used only for scoring.
Timing measures local backend calls, not public-network latency; no answer generation.

| Metric | Baseline | Semantic hard filter | Semantic soft RRF |
|---|---:|---:|---:|
| Evidence Recall@10 | 56.67% | 56.67% | 53.33% |
| Evidence Recall@100 | 88.33% | 88.33% | 90.00% |
| Add all messages | 4.209 s | 5.986 s | 6.191 s |
| Search P50 K=100 | 29.28 ms | 44.03 ms | 44.33 ms |
| Search P95 K=100 | 35.31 ms | 48.65 ms | 48.54 ms |
| Mean results K=100 | 100 | 100 | 100 |

At K=100 soft RRF wins on 2 questions, loses on 1, ties on 27. Hard filter ties all
30 questions in evidence recall (not necessarily identical lists). At K=10 soft RRF
loses on 1 question and ties on 29. Soft keyword semantics costs about 47% more Add
time and 51% more median Search time in this run. This is not a stable throughput
benchmark, nor evidence of statistically reliable improvement across conversations.

## Examples

- Melanie's artists/bands question: both semantic arms retain D11:3 (Matt Patterson)
  and D15:16 at K=100, matching original v1. Prior literal filtering lost D11:3.
  Neither semantic arm retrieves these gold passages at K=10 in this test.
- Caroline's relationship status: soft ranking improves evidence recall 0 -> 0.5,
  recovering D2:14.
- Would Melanie enjoy Vivaldi: soft ranking improves 0 -> 1, recovering D15:28.
- What Caroline learned from Becoming Nicole: soft ranking regresses 1 -> 0 at K=100,
  losing D7:13 from the final limited results.
- Poetry-reading subject: soft ranking regresses 1 -> 0 at K=10, losing D17:18.

Soft ranking never discards candidates before window expansion, but changed rankings
still move relevant evidence outside a finite top_k. Keywords lose context and partly
duplicate the original full-text semantic signal. A larger held-out multi-conversation
comparison is needed before deciding to enable or tune this feature.

## Validation and artifacts

79 unit/API tests passed. Includes controlled semantic promotion with explicit test
weight 2, exact RRF arithmetic, zero-weight equality with baseline, semantic no-match
fallback, old keyword stores without vectors, persistence/isolation and atomic failure.
An initial test incorrectly expected weight 0.5 to overturn a stronger existing rank;
its fixture was corrected to weight 2. Benchmark/production defaults stayed 0.5.

Wheel build succeeded: data/builds/csig_memory-0.1.0-py3-none-any.whl,
SHA256 c822596f478c930606f0d2303c9021cd3c8947f1c04d115f7c4317a7c713f344.
Raw reports and all three databases are local ignored artifacts under
`data/semantic-tag-benchmarks/20260930T121049Z/`. report.json records exact source
hash and question indices; cases.json includes paired evidence IDs and timings.

Enable locally with root .env RAG_TAG_MODE=semantic_rank (or semantic_filter).
Use a fresh RAG_MEMORY_DB and re-add source messages to populate keyword vectors;
existing committed requests are not silently backfilled. Production remains unchanged.
Reproduce: python scripts/benchmark_semantic_tags.py.
