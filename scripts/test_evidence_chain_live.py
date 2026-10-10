"""Opt-in chain trial: real providers, frozen dataset DB + synthetic adversarial cases."""
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import AMLAdd, create_app
from fastapi.testclient import TestClient


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--cases', nargs='*')
    parser.add_argument('--only-on', action='store_true')
    args=parser.parse_args()
    source = PROJECT_ROOT/'data/final-text-validation/20261010-semantic-full'
    out = PROJECT_ROOT/'data/evidence-chain'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    with sqlite3.connect((source/'conv-48/memory.sqlite3').as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(out/'memory.sqlite3') as dst:
        src.backup(dst)
    cfg=load_settings()
    cfg.update(json.loads((source/'manifest.json').read_text())['settings'])
    cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'), AML_API_KEY='isolated-chain-test',
        RAG_WRITE_GATE_MODE='off', RAG_BUILD_MODE='off', RAG_TIME_ANNOTATION_MODE='off',
        RAG_FACT_REPLACEMENT_MODE='off', RAG_SEMANTIC_PRIVACY_MODE='off',
        RAG_MULTIHOP_BINDINGS='off', RAG_MULTIHOP_NEEDS='off',
        RAG_MULTIHOP_PROMPT_STYLE='long', RAG_EVIDENCE_CHAIN_MODE='on')
    store=VanillaMemory(cfg)
    synthetic=[
        ('three_hop', "What instrument does Veda's brother's teacher teach?",
         ['Veda has a brother named Niko.', 'Niko takes lessons from teacher Selene.', 'Selene teaches the clarinet.', 'Veda plays the piano.'], [0,1,2], True),
        ('missing_middle', "What instrument does Veda's brother's teacher teach?",
         ['Veda has a brother named Niko.', 'Selene teaches the clarinet.', 'Veda plays the piano.'], [0], False),
        ('wrong_person', "What instrument does Veda's brother's teacher teach?",
         ['Veda has a brother named Niko.', 'Lena has a brother named Pavel.', 'Pavel takes lessons from teacher Selene.', 'Selene teaches the clarinet.'], [0], False),
        ('independent_dates', 'Which happened first, graduation or relocation?',
         ['Tara graduated in June 2020.', 'Tara relocated in September 2021.', 'Tara visited Oslo in 2019.'], [0,1], True),
        ('single_source', "What instrument does Veda's brother's teacher teach?",
         ['Veda has a brother named Niko, whose teacher Selene teaches the clarinet.'], [0], True),
    ]
    report={'kind':'real providers; frozen dataset search and synthetic held-out chain controls',
            'output':str(out),'cases':[], 'settings_overrides':{k:cfg[k] for k in ['RAG_FACT_REPLACEMENT_MODE','RAG_MULTIHOP_BINDINGS','RAG_MULTIHOP_NEEDS','RAG_MULTIHOP_PROMPT_STYLE']}}
    def save(): (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    cases=[]
    for name,q,texts,gold,answerable in synthetic:
        store.add(AMLAdd(user_id='chain:'+name,request_id=name,session_id=name,messages=[dict(role='user',content=t,timestamp=1735689600000) for t in texts]))
        cases.append(dict(name=name,user_id='chain:'+name,question=q,gold=[texts[i] for i in gold],answerable=answerable,kind='synthetic'))
    repeats=json.loads((source/'diagnostic_repeats.json').read_text())
    unique={r['question']:r for r in repeats}
    for q,old in unique.items():
        cases.append(dict(name='dataset_'+str(old['question_index']),user_id='release-text:conv-48',question=q,gold=old['gold'],kind='dataset'))
    samples=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))
    sample=next(s for s in samples if s['sample_id']=='conv-48')
    with store.connect() as db:
        id_to_dia={r['id']:sample['conversation'][r['session_id']][int(r['request_id'].rsplit(':',1)[1])+r['message_index']]['dia_id']
            for r in db.execute("select * from rag_memories where user_id='release-text:conv-48'")}
    report['source_mapping']=id_to_dia
    with TestClient(create_app(settings=cfg,backend=store)) as client:
        client.headers['Authorization']='Bearer isolated-chain-test'
        report['health_status']=client.get('/health').status_code
        for case in cases:
            if args.cases and case['name'] not in args.cases: continue
            row=dict(case)
            for mode in (['on'] if args.only_on else ['off','on']):
                store.multihop.chain_mode=mode
                store.multihop.planner.chain_mode=mode=='on'
                trace={}; original=store.search
                def wrapped(payload,**kwargs): return original(payload,trace=trace)
                store.search=wrapped; start=time.perf_counter()
                try:
                    r=client.post('/search',json=dict(user_id=case['user_id'],query=case['question'],top_k=1 if case['kind']=='synthetic' else 10))
                finally: store.search=original
                response=r.json(); hits=response.get('data',[])
                text='\n'.join(h['content'] for h in hits)
                if case['kind']=='synthetic':
                    found=[g for g in case['gold'] if g in text]
                else:
                    mids={h['id'] for h in hits}
                    with store.connect() as db:
                        for h in hits:
                            b=db.execute('select source_ids from rag_evidence_bundles where user_id=? and bundle_id=?',(case['user_id'],h['id'])).fetchone()
                            if b: mids.update(json.loads(b[0]))
                    found=sorted(set(case['gold']) & {id_to_dia.get(i) for i in mids})
                row[mode]=dict(status=r.status_code,seconds=time.perf_counter()-start,found=found,
                    all_required_sources_returned=len(found)==len(case['gold']),response=response,trace=trace)
                print(case['name'],mode,'sources',len(found),'/',len(case['gold']),'chain',trace.get('evidence_chain',{}).get('status'),'stop',trace.get('stop'),flush=True)
                save()
            report['cases'].append(row);save()
    print('DONE',out,flush=True)


if __name__=='__main__': main()
