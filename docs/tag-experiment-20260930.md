# v1 semantic-tag experiment — 2026-09-30

Implementation tested: `30414e1` (initial feature commit `d4360c0`); baseline behavior
is `ddd5531` with RAG_TAG_MODE=off. Code was committed before running tests.

## Protocol

Public LoCoMo-Refined, sample conv-26, first complete conversation: 19 sessions,
419 messages/chunks. 30 deterministic evenly spaced questions among 85 eligible
text-only questions with available evidence. Source SHA256:
`1aef6da702087d72515d1b9224f0956a2fbab415c11936253bf7d967d3cf8c17`.

Both arms use identical raw messages, BGE weights, chunking, BM25/RRF, options
handling and result windows. Two separate databases on the same machine. Baseline
and tagged retrieval execution order alternates by question. Model startup excluded
from Add timing. Each question's real gpt-4o-mini tag extraction is measured once
and charged to both K measurements; query tags are reused across K=10 and K=100.
No benchmark answers/evidence are supplied to tag extraction. Evidence Recall is
macro-average fraction of gold dialogue IDs retrieved per question, not answer accuracy.

## Results

| Metric | Original v1 | Tag filter |
|---|---:|---:|
| Recall@10 | 56.67% | 53.33% |
| Recall@100 | 88.33% | 81.67% |
| Add total, 419 messages | 5.48 s | 127.86 s |
| Search P50, K=100 | 0.0299 s | 1.6340 s |
| Search P95, K=100 | 0.0392 s | 3.8376 s |
| Mean results, K=100 | 100 | 99.6 |

At K=10: 0 wins, 2 losses, 28 ties. At K=100: 1 win, 3 losses, 26 ties.
72 unit/API tests passed. Real model execution completed. The initial run rejected
a wrong-size LLM tag batch without committing that request; the subsequent commit
constrained the response schema to exactly the input count. Results above are from
the complete rerun, not the failed run.

## Failure analysis

1. Question about whether Caroline would pursue writing: query tags career/caroline/
   writing. Gold evidence D7:5 (counseling/mental-health jobs) and D7:9 (reading) had
   mental health/support and books/reading tags. Recall dropped from 1 to 0 because
   the relevant evidence provides an indirect/negative answer, not a literal topic match.
2. Question about Melanie being an ally to the transgender community: query tags
   ally/community/transgender. Evidence used lgbtq community and trans community.
   Exact normalized tag intersection misses these related phrases. Recall 1 to 0.
3. Childhood activity with Caroline's dad: source D13:7 describes horseback riding
   with her father, but the extracted tags were art/horses/painting. Query tags were
   activity/caroline/dad/family. Recall 1 to 0. Tags contain unsupported topics;
   batch cross-item contamination or semantic extraction failure is plausible. Exact
   output count does not establish semantic correctness or correct item alignment.
4. One improvement (Vivaldi preference) recovered a gold passage with empty tags:
   it survived the unknown-tag fallback while other candidates were filtered. This
   is not evidence that its tags accurately captured musical preference.

## Decision and boundaries

Retain the feature behind `RAG_TAG_MODE=filter`, but keep `off` as the default and do
not enable it in the local production configuration or deploy it remotely. This test
shows a regression in both evidence recall and latency. It does not establish results
on the other conversations or official evaluation. No answer generation/judge was run.

Future experiments should use tags as a soft ranking feature rather than a hard gate,
canonicalize related labels, and improve independently attributable tag extraction.
They require a new paired test; this report does not claim those changes improve results.

Raw artifacts (ignored local files):
- data/tag-benchmarks/20260930T114400Z/report.json
- data/tag-benchmarks/20260930T114400Z/cases.json
- data/tag-benchmarks/20260930T114400Z/baseline.sqlite3
- data/tag-benchmarks/20260930T114400Z/tagged.sqlite3

Reproduce: `python scripts/benchmark_tags.py` with the same local public dataset,
BGE snapshot and root .env credentials/proxy. LLM outputs and network latency can vary.
