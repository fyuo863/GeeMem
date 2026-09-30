# Evidence retrieval metrics

Both scripts/benchmark_tags.py and scripts/benchmark_semantic_tags.py now execute
actual Search requests at K=5,10,100 and report the same metrics at every K. K=5 is
not approximated by truncating an earlier K=100 run: candidate/window behavior can
depend on K. All measurements refer to the final returned source chunks.

Let G_q be the set of gold dialogue IDs for question q, and R_q the set of dialogue
IDs covered by returned chunks. Chunk-to-dialogue mapping is program-generated.
Repeated chunks/IDs count once per question; the same evidence used by two questions
counts separately in each question. Non-gold returned evidence earns no credit.

- Recall@K (`recall`): mean over questions of |G_q intersect R_q| / |G_q|.
  This is macro recall: every question has equal weight.
- Hit@K (`hit_rate`): fraction of questions with at least one gold evidence ID returned.
- Micro Recall@K (`micro_recall`): sum of matched gold counts / sum of gold counts.
  Questions with more gold evidence have more influence.
- All-evidence Hit@K / 全证据命中@K (`all_evidence_hit_rate`): fraction of questions
  for which every gold evidence ID is returned.

Raw numerator/denominator counts are also saved: evaluated_questions, hit_questions,
all_evidence_hit_questions, matched_evidence, gold_evidence. No-gold questions are
excluded with skipped_no_evidence; an empty evaluation yields null, not perfect or
zero accuracy. The public benchmark currently prefilters to nonempty available gold.

These metrics measure evidence retrieval, not final answer accuracy or official score.
A partial chunk counts as covering its parent dialogue ID, matching the existing
benchmark convention; this does not prove it contains the entire answer-bearing span.

## Fixed public scenario results (2026-09-30)

419 messages, 30 questions, 39 gold evidence occurrences; no LLM calls.

| Mode | Recall@5 | Hit@5 | Micro Recall@5 | All-evidence Hit@5 |
|---|---:|---:|---:|---:|
| baseline | 49.17% | 50.00% | 43.59% | 46.67% |
| semantic_filter | 49.17% | 50.00% | 43.59% | 46.67% |
| semantic_rank | 33.33% | 33.33% | 33.33% | 33.33% |

Report: data/semantic-tag-benchmarks/20260930T121649Z/report.json.
Single-scenario evidence retrieval, not official evaluation or answer accuracy.
Soft semantic ranking wins on 1 question and loses on 5 at K=5; default remains off.
