"""Synthetic bilingual cases, real providers, isolated local HTTP API test."""
import json
from pathlib import Path
import sys
import time
from datetime import datetime, timezone
from statistics import mean
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from memory.config import PROJECT_ROOT, load_settings
from memory.aml_api import create_app
from memory.vanilla import VanillaMemory

# Gold references are source positions, not generated answer strings.
CASES = [
 ('current_en', 'Where does Mei live now?', ['Mei lives in Paris.', 'Correction: Mei moved from Paris to Rome and now lives in Rome.', 'Mei may move to Berlin next year.'], [1]),
 ('current_zh', '小林目前在哪家公司工作？', ['小林在甲公司工作。', '更正：小林已经离开甲公司，目前在乙公司工作。', '小王目前在丙公司工作。'], [1]),
 ('future', 'Where does Omar currently work?', ['Omar currently works at Cedar Labs.', 'Omar hopes to join Maple Labs next year but has not accepted an offer.'], [0]),
 ('reported_history', 'Where does Nina live now?', ['Nina now lives in Oslo after leaving Lima.', 'Looking back at 2020, Nina lived in Lima then.'], [0]),
 ('conflict', 'What is the current delivery address for Alex?', ['Alex says the current delivery address is 12 Oak Street.', 'Alex also says the current delivery address is 90 Pine Street. Neither statement is marked as a correction.'], [0, 1]),
 ('person', '李明目前住在哪里？', ['李明住在北京。', '李华现在住在上海。'], [0]),
 ('rule_en', 'What is the current expense rule for an emergency taxi?', ['The expense rule requires manager approval for taxi rides.', 'Exception to the taxi rule: emergency hospital trips need no prior approval, but require a receipt.'], [0, 1]),
 ('rule_zh', '当前周末故障值班规则有什么例外？', ['周末故障由当周值班员处理。', '上述规则的例外：涉及支付故障时，必须联系支付组，值班员不能单独处理。'], [0, 1]),
 ('historical_control', 'Where did Mei live in 2020?', ['Mei lived in Paris in 2020.', 'Mei moved to Rome in 2025.'], [0]),
 ('time_control', 'Which happened first, graduation or relocation?', ['Tara graduated in June 2020.', 'Tara relocated in September 2021.'], [0, 1]),
 ('chain_control', "What instrument does Veda's brother's teacher teach?", ['Veda has a brother named Arun.', 'Arun takes lessons from teacher Mira.', 'Mira teaches the violin.'], [0, 1, 2]),
 ('direct_control', 'What is Leo allergic to?', ['Leo is allergic to peanuts.', 'Maya is allergic to shellfish.'], [0]),
]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--reuse', type=Path, help='Reuse this isolated synthetic test directory')
    args = parser.parse_args()
    out = args.reuse or PROJECT_ROOT/'data/state-evidence'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True, exist_ok=True)
    report_path = out/('report-final.json' if args.reuse else 'report.json')
    cfg = load_settings()
    cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'), AML_API_KEY='isolated-state-test',
               RAG_STATE_EVIDENCE_MODE='on')
    store = VanillaMemory(cfg)
    report = {'kind': 'synthetic cases / real configured providers', 'cases': [], 'add': []}
    with TestClient(create_app(settings=cfg, backend=store)) as client:
        client.headers['Authorization'] = 'Bearer isolated-state-test'
        assert client.get('/health').status_code == 200
        for name, question, texts, gold_indices in CASES:
            r = client.post('/add', json=dict(user_id=name, session_id=name, request_id=name,
                messages=[dict(role='user', content=text, timestamp=1735689600000+i*86400000) for i, text in enumerate(texts)]))
            report['add'].append(dict(case=name, status=r.status_code))
            if r.status_code != 200:
                print('ADD FAILED', name, r.status_code, flush=True)
                continue
            item = dict(name=name, question=question, sources=texts, gold_indices=gold_indices)
            for mode in ['off', 'on']:
                store.state_evidence.mode = mode
                trace = {}
                original = store.search
                def traced(payload, **kwargs):
                    return original(payload, trace=trace)
                store.search = traced
                start = time.perf_counter()
                try:
                    response = client.post('/search', json=dict(user_id=name, query=question, top_k=10,
                                                               reference_time=1738368000000))
                finally:
                    store.search = original
                row = dict(status=response.status_code, seconds=time.perf_counter()-start,
                           selector=trace.get('state_evidence'), pipeline=trace.get('pipeline'))
                if response.status_code == 200:
                    hits = response.json()['data']
                    assert all(h['content'] in texts for h in hits)
                    ranked = [texts.index(h['content']) for h in hits]
                    gold = set(gold_indices)
                    row.update(ranked=ranked, recall10=len(gold & set(ranked))/len(gold),
                               precision1=int(bool(ranked) and ranked[0] in gold),
                               all_at_gold_k=int(gold <= set(ranked[:len(gold)])))
                item[mode] = row
                print(name, mode, row.get('ranked'), round(row['seconds'], 2), row.get('selector'), flush=True)
            report['cases'].append(item)
            report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
        r = client.post('/search', json=dict(user_id='empty-user', query='Where does Mei live now?', top_k=10))
        report['isolation_passed'] = r.status_code == 200 and r.json()['data'] == []
    for mode in ['off', 'on']:
        rows = [x[mode] for x in report['cases'] if x[mode]['status'] == 200]
        report[mode] = {key: mean(r[key] for r in rows) for key in ['recall10', 'precision1', 'all_at_gold_k', 'seconds']} if rows else {}
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')
    print('REPORT', out, report.get('off'), report.get('on'), flush=True)


if __name__ == '__main__':
    main()
