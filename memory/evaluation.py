"""Evidence metrics over dialogue IDs, independent of retrieval implementation."""


def evidence_metrics(cases, variant):
    """Cases must already be restricted to one K; duplicate IDs count only once.

    Gold evidence is counted per question (the same ID across questions is counted
    in each question). Questions without gold evidence are excluded, not successes.
    """
    recalls = []
    hit_questions = full_questions = total_gold = total_hits = skipped = 0
    for case in cases:
        gold = set(case['evidence'])
        if not gold:
            skipped += 1
            continue
        matched = gold.intersection(case[variant]['hit_evidence'])
        n = len(matched)
        recalls.append(n / len(gold))
        hit_questions += n > 0
        full_questions += n == len(gold)
        total_gold += len(gold)
        total_hits += n
    count = len(recalls)
    return dict(recall=sum(recalls)/count if count else None,
                hit_rate=hit_questions/count if count else None,
                micro_recall=total_hits/total_gold if total_gold else None,
                all_evidence_hit_rate=full_questions/count if count else None,
                evaluated_questions=count, skipped_no_evidence=skipped,
                hit_questions=hit_questions, all_evidence_hit_questions=full_questions,
                matched_evidence=total_hits, gold_evidence=total_gold)
