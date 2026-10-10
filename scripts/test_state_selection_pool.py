"""Real selector on deliberately adverse, fixed candidate rankings (not retrieval metrics)."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.state_evidence import StateEvidenceSelector
from test_state_evidence_real import CASES


def main():
    cfg = load_settings()
    cfg['RAG_STATE_EVIDENCE_MODE'] = 'on'
    selector = StateEvidenceSelector(cfg)
    complete = selector.model.complete
    captured = {}
    def record(*args, **kwargs):
        result = complete(*args, **kwargs)
        captured['output'] = result.model_dump()
        return result
    selector.model.complete = record
    rows = []
    for name, question, texts, gold in CASES[:8]:
        # Fixed adverse order isolates selection from upstream model randomness.
        distractors = [f'Unrelated archived record {i}: the cafeteria opens at noon.' for i in range(12)]
        ordered = [t for i, t in enumerate(texts) if i not in gold] + distractors + [texts[i] for i in gold]
        ordered = ordered[:16]
        hits = [dict(id=str(i), content=t, score=1-i*.01, created_at='2025-01-01T00:00:00Z') for i,t in enumerate(ordered)]
        trace = {}
        result = selector.select(SimpleNamespace(query=question, reference_time=1738368000000), hits, trace)
        target = {texts[i] for i in gold}
        row = dict(name=name, baseline_all_at_gold_k=target <= {h['content'] for h in hits[:len(gold)]},
                   selected_all_at_gold_k=target <= {h['content'] for h in result[:len(gold)]},
                   original_top=[h['content'] for h in hits[:len(gold)]],
                   selected_top=[h['content'] for h in result[:len(gold)]], diagnostics=trace,
                   model_output=captured.get('output'))
        rows.append(row)
        print(name, row['selected_all_at_gold_k'], trace, flush=True)
    out = PROJECT_ROOT/'data/state-evidence/fixed-pool.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    print('RESULT', sum(r['selected_all_at_gold_k'] for r in rows), '/', len(rows), flush=True)


if __name__ == '__main__':
    main()
