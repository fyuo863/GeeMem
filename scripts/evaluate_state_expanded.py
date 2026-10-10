"""Frozen-selector evaluation: public replay, synthetic order stress, pooled API."""
import argparse
import json
import random
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.state_evidence import StateEvidenceSelector
from test_state_evidence_real import CASES

EXTRA = [
 ('correction_email', 'What is the current contact email for Hana?', ['Hana uses hana@old.example.', 'Hana: My old email is disabled. Use hana@new.example from now on.'], [1]),
 ('correct_number', '更正后，仓库当前库存是多少？', ['仓库库存是五十件。', '刚才数错了，更正为四十五件，这是当前确认的库存。'], [1]),
 ('cancelled_change', 'Where does Jules currently work?', ['Jules works at Bay Labs.', 'Jules planned to leave Bay Labs for Hill Labs.', 'Jules cancelled the move and remains at Bay Labs.'], [2]),
 ('temporary_location', 'Where does Priya currently live?', ['Priya lives in Delhi.', 'Priya is visiting Tokyo for a week; her home remains in Delhi.'], [1]),
 ('third_party', 'What is the current phone number for Ken?', ['Ken: My number is 555-1100.', 'Ken: My colleague Kai just changed his number to 555-2200.'], [0]),
 ('conditional_rule', 'What is the current refund rule for an opened faulty device?', ['Refunds require unopened packaging.', 'Exception: faulty devices qualify for refunds even with opened packaging.', 'Cosmetic damage from misuse does not qualify as a fault.'], [0, 1, 2]),
 ('not_applicable_exception', 'What is the current approval rule for a normal taxi to the office?', ['Normal office taxi rides require approval.', 'Exception: emergency hospital trips need no approval.'], [0]),
 ('conflict_zh', '目前两份记录对小周的合同期限有什么冲突？', ['记录甲写小周合同在六月到期。', '记录乙写小周合同在八月到期，目前没有确认哪份正确。'], [0, 1]),
 ('history_explicit', 'Before the latest correction, what limit was stated for uploads?', ['The upload limit was 5 MB.', 'Correction: the upload limit is 20 MB now.'], [0]),
 ('negated', 'Does Rowan currently have a dog?', ['Rowan used to have a dog.', 'Rowan no longer has a dog; the dog was adopted by a friend.'], [1]),
 ('future_effective', 'What is the current membership fee on 2025-02-01?', ['The membership fee is 10 dollars per month.', 'Starting 2025-03-01, the membership fee will become 15 dollars per month.'], [0]),
 ('changed_back', '小许现在喜欢哪支球队？', ['小许最喜欢蓝队。', '小许后来改为支持红队。', '小许：我又改回来了，现在最喜欢蓝队。'], [2]),
]
SCENARIOS = CASES[:8] + EXTRA


def metrics(ranked, gold):
    gold = set(gold)
    result = {'rr': next((1/(i+1) for i,x in enumerate(ranked) if x in gold), 0),
              'top1': int(bool(ranked) and ranked[0] in gold)}
    for k in [1, 5, 10]:
        n = len(set(ranked[:k]) & gold)
        result.update({f'recall{k}': n/len(gold), f'hit{k}': int(n > 0),
                       f'all{k}': int(n == len(gold)), f'found{k}': n})
    result['gold_count'] = len(gold)
    return result


def aggregate(rows):
    summary = {'n':len(rows)}
    for mode in ['off', 'on']:
        summary[mode] = {key: mean(r[mode][key] for r in rows) for key in
                           ['rr','top1','recall5','recall10','hit5','hit10','all5','all10']}
        for k in [5,10]:
            summary[mode][f'micro_recall{k}'] = sum(r[mode][f'found{k}'] for r in rows)/sum(r[mode]['gold_count'] for r in rows)
    summary['recall10_improved'] = sum(r['on']['recall10'] > r['off']['recall10'] for r in rows)
    summary['recall10_regressed'] = sum(r['on']['recall10'] < r['off']['recall10'] for r in rows)
    summary['top1_improved'] = sum(r['on']['top1'] > r['off']['top1'] for r in rows)
    summary['top1_regressed'] = sum(r['on']['top1'] < r['off']['top1'] for r in rows)
    return summary


def record_selector(cfg):
    selector = StateEvidenceSelector(cfg)
    captured = {}
    complete = selector.model.complete
    def record(*args, **kwargs):
        captured.clear()
        result = complete(*args, **kwargs)
        captured.update(result.model_dump())
        return result
    selector.model.complete = record
    return selector, captured


def run_public(cfg, out):
    selector, captured = record_selector(cfg)
    rows = []
    for file in sorted((PROJECT_ROOT/'data/final-text-validation/20261009-qwen-full').glob('*/cases.jsonl')):
        with sqlite3.connect(file.parent.joinpath('memory.sqlite3').as_uri()+'?mode=ro', uri=True) as db:
            sources = {r[0]:r[1:] for r in db.execute('select id,content,timestamp from rag_memories')}
        for line in file.read_text(encoding='utf8').splitlines():
            case = json.loads(line)
            if not case['gold']:
                continue
            base = case['current']
            assert base['status'] == 200
            ids, ranked = base['ids'], base['ranked']
            mapped = dict(zip(ids, ranked))
            assert len(ids) == len(ranked)
            trace = {}; captured.clear()
            hits = [dict(id=x, content=sources[x][0], score=1-i*.001,
                         created_at=datetime.fromtimestamp(sources[x][1]/1000,timezone.utc).isoformat()) for i,x in enumerate(ids)]
            result = selector.select(SimpleNamespace(query=case['question'], reference_time=None),hits,trace)
            row = dict(name=f"{case['sample_id']}:{case['question_index']}", question=case['question'],
                       category=case['category'], gold=case['gold'], off=metrics(ranked,case['gold']),
                       on=metrics([mapped[h['id']] for h in result],case['gold']), trace=trace)
            if trace['state_evidence']['calls']:
                row.update(model_output=dict(captured), baseline=ranked[:16], after=[mapped[h['id']] for h in result[:16]])
                print('PUBLIC',row['name'],row['off']['recall10'],row['on']['recall10'],flush=True)
            rows.append(row)
    data=dict(kind='public frozen candidate replay, real selector only', rows=rows,summary=aggregate(rows))
    data['triggered'] = sum(r['trace']['state_evidence']['calls'] for r in rows)
    (out/'public.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('PUBLIC SUMMARY',data['summary'], 'calls',data['triggered'],flush=True)


def run_stress(cfg, out):
    selector,captured = record_selector(cfg)
    rows=[]
    for n,(name,question,texts,gold) in enumerate(SCENARIOS):
        # Exactly 16 sources, the same texts in three input permutations.
        alltexts=texts+[f'Archive {j}: unrelated cafeteria opening notice.' for j in range(16-len(texts))]
        for order in ['gold_first','gold_last','shuffled']:
            indices=list(range(16))
            if order == 'gold_first': indices=gold+[i for i in indices if i not in gold]
            elif order == 'gold_last': indices=[i for i in indices if i not in gold]+gold
            else: random.Random(1000+n).shuffle(indices)
            hits=[dict(id=str(i),content=alltexts[i],score=1-j*.01,created_at='2025-01-01T00:00:00Z') for j,i in enumerate(indices)]
            trace={}; captured.clear()
            result=selector.select(SimpleNamespace(query=question,reference_time=1738368000000),hits,trace)
            gold_ids={str(i) for i in gold}
            selected={hits[c['index']]['id'] for c in captured.get('selected',[]) if c['role']=='primary'}
            row=dict(name=name,order=order,question=question,sources=alltexts,gold=gold,baseline=indices,
                     after=[int(h['id']) for h in result],off=metrics([h['id'] for h in hits],gold_ids),
                     on=metrics([h['id'] for h in result],gold_ids),trace=trace,model_output=dict(captured),
                     primary_precision=len(selected&gold_ids)/len(selected) if selected else 0,
                     primary_recall=len(selected&gold_ids)/len(gold_ids),primary_exact=int(selected==gold_ids))
            rows.append(row)
            (out/'stress.json').write_text(json.dumps(dict(rows=rows),ensure_ascii=False,indent=2),encoding='utf8')
            print('STRESS',len(rows),name,order,row['on']['top1'],row['primary_exact'],flush=True)
    data=dict(kind='20 authored scenarios x3 orders, not 60 independent questions',rows=rows,summary=aggregate(rows),
              by_order={o:aggregate([r for r in rows if r['order']==o]) for o in ['gold_first','gold_last','shuffled']},
              primary={k:mean(r[k] for r in rows) for k in ['primary_precision','primary_recall','primary_exact']})
    excluded=['cancelled_change','temporary_location']
    unambiguous=[r for r in rows if r['name'] not in excluded]
    data['sensitivity_excluding_ambiguous_cases']=dict(excluded=excluded,summary=aggregate(unambiguous),
        primary_exact=mean(r['primary_exact'] for r in unambiguous))
    (out/'stress.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('STRESS SUMMARY',data['summary'],data['primary'],flush=True)


def run_api(cfg,out):
    from fastapi.testclient import TestClient
    from memory.aml_api import create_app
    from memory.vanilla import VanillaMemory
    cfg=dict(cfg,RAG_MEMORY_DB=str(out/'memory.sqlite3'),AML_API_KEY='isolated-expanded-test')
    store=VanillaMemory(cfg)
    texts=list(dict.fromkeys(t for _,_,ss,_ in SCENARIOS for t in ss))
    rows=[]; adds=[]
    with TestClient(create_app(settings=cfg,backend=store)) as client:
        client.headers['Authorization']='Bearer isolated-expanded-test'
        assert client.get('/health').status_code==200
        for i in range(0,len(texts),8):
            r=client.post('/add',json=dict(user_id='pooled',session_id=f's{i}',request_id=f'a{i}',
                messages=[dict(role='user',content=t,timestamp=1735689600000+j*60000) for j,t in enumerate(texts[i:i+8])]))
            adds.append(r.status_code)
            print('ADD',i,r.status_code,flush=True)
            if r.status_code !=200: raise RuntimeError('Isolated Add failed; no incomplete corpus comparison')
        for name,question,sources,gold in SCENARIOS:
            row=dict(name=name,question=question,gold=[sources[i] for i in gold])
            for mode in ['off','on']:
                store.state_evidence.mode=mode
                trace={}; original=store.search
                def wrapped(payload,**kwargs): return original(payload,trace=trace)
                store.search=wrapped; start=time.perf_counter()
                try:r=client.post('/search',json=dict(user_id='pooled',query=question,top_k=10,reference_time=1738368000000))
                finally:store.search=original
                seconds=time.perf_counter()-start
                if r.status_code!=200: raise RuntimeError(f'Search failed {r.status_code}')
                hits=r.json()['data']; assert all(h['content'] in texts for h in hits)
                row[mode]=metrics([h['content'] for h in hits],row['gold'])
                row[mode].update(seconds=seconds,ranked=[h['content'] for h in hits],selector=trace.get('state_evidence'))
                print('API',name,mode,row[mode]['recall10'],row[mode]['top1'],round(seconds,2),flush=True)
            rows.append(row)
            (out/'api.json').write_text(json.dumps(dict(rows=rows,adds=adds),ensure_ascii=False,indent=2),encoding='utf8')
        r=client.post('/search',json=dict(user_id='empty',query='Where does Mei live now?',top_k=10))
        isolation=r.status_code==200 and r.json()['data']==[]
    data=dict(kind='real providers, pooled synthetic corpus, paired HTTP runs',sources=len(texts),adds=adds,rows=rows,
              summary=aggregate(rows),isolation=isolation,
              latency={m:mean(r[m]['seconds'] for r in rows) for m in ['off','on']})
    (out/'api.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('API SUMMARY',data['summary'],data['latency'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['public','stress','api']);p.add_argument('--output',required=True)
    args=p.parse_args();out=PROJECT_ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    cfg=load_settings();cfg['RAG_STATE_EVIDENCE_MODE']='on'
    {'public':run_public,'stress':run_stress,'api':run_api}[args.mode](cfg,out)
