"""Audit frozen Hit10 results against source provenance; never tune retrieval here.

Run from any directory with Python and the project's installed dependencies.
Gold labels are used only for evaluation. No network or model inference is needed.
"""
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT
from memory.evaluation import canonical_evidence, evidence_metrics
from memory.rerank import context_support_scores
from memory.vanilla import chunks


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def summarize(cases):
    metrics = evidence_metrics(cases, 'selected')
    misses = [c for c in cases if not c['selected']['hit_evidence']]
    ranks = sorted(c['best_final_gold_rank'] for c in misses if c['best_final_gold_rank'])
    return dict(metrics=metrics, zero_hit=len(misses),
                partial_hit=sum(0 < len(c['selected']['hit_evidence']) < len(c['evidence']) for c in cases),
                candidate_hit_questions=sum(bool(c['candidate_hits']) for c in cases),
                missed_without_candidate=sum(not c['candidate_hits'] for c in misses),
                missed_with_candidate=sum(bool(c['candidate_hits']) for c in misses),
                miss_best_rank_buckets={label: sum(low <= r <= high for r in ranks)
                    for label, low, high in [('11-20', 11, 20), ('21-50', 21, 50), ('51-100', 51, 100), ('101+', 101, 1000000)]},
                misses_gold_outside_final_pool=sum(c['best_final_gold_rank'] is None for c in misses),
                misses_neighbor_distance={str(k): sum(c['nearest_selected_distance'] == k for c in misses) for k in [1, 2, 3]},
                misses_same_session=sum(c['nearest_selected_distance'] is not None for c in misses))


def main():
    source_path = PROJECT_ROOT / 'data/locomo-refined/data/raw/locomo_refined.json'
    source = read_json(source_path)
    samples = {s['sample_id']: s for s in source}
    cases = []
    db_audits = {}
    input_hashes = {str(source_path.relative_to(PROJECT_ROOT)): hashlib.sha256(source_path.read_bytes()).hexdigest()}
    for split, run, filename in [('dev', '20261001T021521Z', 'dev-all-cases.json'),
                                 ('heldout', '20261001T021642Z', 'heldout-propagate_2-cases.json')]:
        root = PROJECT_ROOT / 'data/rerank-benchmarks' / run
        frozen_path = PROJECT_ROOT / 'data/hit10-experiments' / filename
        frozen = {(c['sample_id'], c['question_index']): c for c in read_json(frozen_path)}
        for p in [root / 'cases.jsonl', frozen_path]:
            input_hashes[str(p.relative_to(PROJECT_ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        original = [json.loads(line) for line in (root / 'cases.jsonl').read_text(encoding='utf-8').splitlines()]
        rows_by_sample = {}
        provenance = {}
        with sqlite3.connect(f'{(root / "memory.sqlite3").as_uri()}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            db_audits[split] = dict(columns=[r['name'] for r in db.execute('PRAGMA table_info(rag_memories)')],
                public_chunks=db.execute("SELECT COUNT(*) FROM rag_memories WHERE user_id LIKE 'public-full:%'").fetchone()[0],
                public_timestamped=db.execute("SELECT COUNT(*) FROM rag_memories WHERE user_id LIKE 'public-full:%' AND timestamp IS NOT NULL").fetchone()[0])
            for sid in sorted({c['sample_id'] for c in original}):
                user = 'public-full:' + sid
                rows_by_sample[sid] = [dict(r) for r in db.execute('SELECT * FROM rag_memories WHERE user_id=? ORDER BY rowid', (user,))]
                conv = samples[sid]['conversation']
                for session, messages in conv.items():
                    if not session.startswith('session_') or not isinstance(messages, list):
                        continue
                    for start in range(0, len(messages), 40):
                        for i, m in enumerate(messages[start:start+40]):
                            for j, text in enumerate(chunks(m['text'])):
                                mid = hashlib.sha256(json.dumps([user, session + ':' + str(start), i, j]).encode()).hexdigest()
                                provenance[mid] = dict(evidence_id=canonical_evidence([m['dia_id']])[0],
                                    speaker=m['speaker'], session=session, turn=start+i,
                                    date=conv.get(session + '_date_time'), content=text)
        for c in original:
            sid, qi = c['sample_id'], c['question_index']
            qa = samples[sid]['qa'][qi]
            assert not qa.get('is_multi_modality') and c['evidence']
            rows = rows_by_sample[sid]
            assert all(r['content'] == provenance[r['id']]['content'] for r in rows)
            positions = {r['id']: i for i, r in enumerate(rows)}
            ids = c['ceiling_400']['ids']
            assert len(ids) == len(c['context']['candidate_scores'])
            candidate_positions = [positions[mid] for mid in ids]
            order, _ = context_support_scores(rows, candidate_positions, c['context']['candidate_scores'], 2)
            final_ids = [rows[i]['id'] for i in order]
            selected = frozen[(sid, qi)]['propagate_2']
            assert final_ids[:10] == selected['ids'], 'Frozen selection no longer reproducible'
            gold = set(c['evidence'])
            found = sorted({provenance[mid]['evidence_id'] for mid in selected['ids']} & gold)
            assert found == sorted(selected['hit_evidence'])
            assert len(selected['ids']) <= 10 and len(set(selected['ids'])) == len(selected['ids'])
            gold_sources = [dict(id=mid, **provenance[mid]) for mid in positions if provenance[mid]['evidence_id'] in gold]
            assert {g['evidence_id'] for g in gold_sources} == gold
            selected_sources = [dict(id=mid, **provenance[mid]) for mid in selected['ids']]
            candidate_hits = sorted(gold & {provenance[mid]['evidence_id'] for mid in ids})
            distances = [abs(g['turn'] - s['turn']) for g in gold_sources for s in selected_sources if g['session'] == s['session']]
            best_rank = next((i+1 for i, mid in enumerate(final_ids) if provenance[mid]['evidence_id'] in gold), None)
            cases.append(dict(sample_id=sid, question_index=qi, split=split, question=c['question'],
                category=qa.get('category'), evidence=c['evidence'], selected=dict(hit_evidence=found),
                candidate_hits=candidate_hits, best_final_gold_rank=best_rank,
                nearest_selected_distance=min(distances) if distances else None,
                temporal_cue=bool(re.search(r'\b(when|before|after|year|month|week|day|since|last|first|recent|20\d\d)\b', c['question'], re.I)),
                gold_sources=gold_sources, selected_sources=selected_sources))
    report = dict(input_sha256=input_hashes, database=db_audits, aggregate=summarize(cases),
        by_split={s: summarize([c for c in cases if c['split'] == s]) for s in ['dev', 'heldout']},
        by_conversation={s: summarize([c for c in cases if c['sample_id'] == s]) for s in samples},
        by_category={str(k): summarize([c for c in cases if c['category'] == k]) for k in sorted({c['category'] for c in cases})},
        by_evidence_count={name: summarize([c for c in cases if (len(c['evidence']) > 1) == multi]) for name, multi in [('single', False), ('multiple', True)]},
        temporal_heuristic={str(v): summarize([c for c in cases if c['temporal_cue'] == v]) for v in [True, False]},
        notes='Frozen cached scores, no new inference or tuning. Candidate ceiling and diagnostic gold ranks are oracle audit statistics, not achievable retrieval scores. Temporal flag is a question-text heuristic, not a truth label. Old heldout data are now diagnostic data; use a fresh external test for future tuning.')
    out = PROJECT_ROOT / 'data/evidence-coverage-audit-20261001'
    out.mkdir(exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'cases.json').write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
