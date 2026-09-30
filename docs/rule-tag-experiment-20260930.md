# Rule-keyword experiment — 2026-09-30

Implementation commit: c1fb4a6, committed before tests/build/benchmark.
Replaces semantic LLM tags with deterministic local rules. Same unchanged hard
candidate filter and session-window logic. RAG_TAG_MODE=filter enables the rules;
default remains off because this experiment did not improve evidence recall.
No LLM calls occur in the vanilla Add/Search path, including keyword extraction.
The separate legacy graph backend still has its original LLM implementation.

Rules: NFKC/case normalization, English word extraction, general stop words,
possessive/common plural normalization, Chinese bigrams, independent per-text
extraction and deduplication. No benchmark-specific dictionary, synonym model,
answer access or cross-message tag transfer. Old semantic tag identities are unknown
and remain eligible; fresh databases were used to avoid mixing old tags.

## Validation

74 tests passed, including no-LLM-call assertions, deterministic batch independence,
Chinese tokens, legacy tag compatibility, atomic writes, retries and filtering.
Wheel built successfully with python -m pip wheel --no-deps --no-build-isolation.
Artifact: data/builds/csig_memory-0.1.0-py3-none-any.whl
SHA256: 1499c121b6c9e20d97ea4932e8b14747342f205541f434c432c67317e9028d20

Same public LoCoMo-Refined conv-26 and 30 deterministic text questions as the previous
LLM-tag experiment: 419 messages, 19 sessions, K=10/100. Real local BGE embeddings.
Gold evidence only used for scoring. No official evaluation or answer generation.
Both variants run on the same host; search order alternates by question and write
order alternates by accumulated message count. Timing excludes model startup.

| Metric | Original v1 (same run) | Rule tags |
|---|---:|---:|
| Evidence Recall@10 | 56.67% | 56.67% |
| Evidence Recall@100 | 88.33% | 86.67% |
| Add total | 4.486 s | 4.500 s |
| Search P50 K=100 | 35.32 ms | 38.34 ms |
| Search P95 K=100 | 40.84 ms | 47.69 ms |
| Mean returned chunks K=100 | 100 | 98.63 |

K=10: 0 wins, 0 losses, 30 ties. K=100: 0 wins, 1 loss, 29 ties.
The lost question was "What musical artists/bands has Melanie seen?" Query keywords
artist/band/melanie/musical. Baseline recovered D11:3 and D15:16; rule filtering
retained only D15:16. Literal overlap cannot reliably cover indirect/named-artist
evidence. Recall for this question fell from 1 to 0.5.

Historical LLM-tag run: Recall@10 53.33%, Recall@100 81.67%, Add 127.86 s,
Search P50 K=100 1.634 s. Rules remove this model-call overhead and improve on that
experiment, but still do not beat original v1. Historical timings are separate runs,
not a simultaneous three-way latency benchmark. Millisecond differences should not
be treated as stable throughput estimates; this is one small paired run.

Reports and databases (ignored local artifacts):
data/tag-benchmarks/20260930T115204Z/report.json and cases.json,
baseline.sqlite3 and tagged.sqlite3. Source hash and exact question indices are in
report.json. Reproduce with python scripts/benchmark_tags.py. No remote deployment.
