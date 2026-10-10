"""Real providers, isolated copied corpus; source delivery, not answer accuracy."""
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from memory.config import PROJECT_ROOT, load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import AMLAdd, create_app


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--modes',nargs='+',choices=['on','raw','coverage'],default=['on','raw'])
    parser.add_argument('--dataset-extra',type=int,default=0)
    parser.add_argument('--multi-evidence',action='store_true')
    args=parser.parse_args()
    out=PROJECT_ROOT/'data/raw-bundles'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    source=PROJECT_ROOT/'data/final-text-validation/20261010-semantic-full'
    dbpath=out/'memory.sqlite3'
    with sqlite3.connect((source/'conv-48/memory.sqlite3').as_uri()+'?mode=ro',uri=True) as src, sqlite3.connect(dbpath) as dst:
        src.backup(dst)
    cfg=load_settings()
    cfg.update(RAG_MEMORY_DB=str(dbpath),AML_API_KEY='isolated-raw-test',RAG_POSITION_LOG_PATH='',
        RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off',RAG_TIME_ANNOTATION_MODE='off',
        RAG_MULTIHOP_MODE='llm',RAG_MULTIHOP_BINDINGS='off',RAG_MULTIHOP_NEEDS='off',
        RAG_MULTIHOP_PROMPT_STYLE='long',RAG_EVIDENCE_CHAIN_MODE='off',
        RAG_FACT_REPLACEMENT_MODE='off',RAG_STATE_EVIDENCE_MODE='off',
        RAG_SEMANTIC_PRIVACY_MODE='off',RAG_EVIDENCE_GROUP_MODE='off',
        RAG_EVIDENCE_BUNDLE_MODE='on')
    cases=[]
    synthetic=[
      ('relationship',"What instrument does Veda's brother's teacher teach?",
       ['Veda has a brother named Niko.','Niko takes lessons from teacher Selene.','Selene teaches the clarinet.','Veda plays the piano.'],[0,1,2]),
      ('wrong_person',"What instrument does Veda's brother's teacher teach?",
       ['Veda has a brother named Niko.','Lena has a brother named Pavel.','Pavel takes lessons from teacher Selene.','Selene teaches the clarinet.'],[0]),
      ('current','Where does Mei currently live?',
       ['Mei lives in Paris.','Correction: Mei moved from Paris to Rome and now lives in Rome.','Lena lives in Berlin.'],[0,1]),
      ('future','Where does Omar currently work?',
       ['Omar works at Cedar Labs.','Omar hopes to work at Maple Labs next year but has not accepted any offer.','Lena works at Atlas.'],[0,1]),
      ('stages',"How did Tara's travel plan change and what was the final outcome?",
       ['Tara planned to visit Oslo in June.','Tara cancelled Oslo and booked Bergen instead.','Tara actually visited Bergen in July.','Lena visited Oslo in May.'],[0,1,2]),
      ('missing_stage',"How did Tara's travel plan change and what was the final outcome?",
       ['Tara planned to visit Oslo in June.','Tara cancelled Oslo and booked Bergen instead.','Lena visited Oslo in May.'],[0,1]),
    ]
    if args.dataset_extra: synthetic=[]
    store=VanillaMemory(cfg)
    with TestClient(create_app(settings=cfg,backend=store)) as client:
        client.headers['Authorization']='Bearer isolated-raw-test'
        for name,q,texts,gold in synthetic:
            user='raw:'+name
            for i,text in enumerate(texts):
                response=client.post('/add',json=dict(user_id=user,session_id=name,request_id=f'{name}:{i}',
                    messages=[dict(role='user',content=text,timestamp=1704067200000+i*86400000)]))
                response.raise_for_status()
            cases.append(dict(name=name,user_id=user,question=q,gold=[texts[i] for i in gold],kind='synthetic'))
    repeats=json.loads((source/'diagnostic_repeats.json').read_text(encoding='utf8'))
    for old in {r['question']:r for r in repeats}.values():
        cases.append(dict(name='dataset_'+str(old['question_index']),user_id='release-text:conv-48',
                          question=old['question'],gold=old['gold'],kind='dataset'))
    samples=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf8'))
    sample=next(s for s in samples if s['sample_id']=='conv-48')
    if args.dataset_extra:
        import hashlib
        eligible=[(i,q) for i,q in enumerate(sample['qa']) if i not in (51,75)
                  and q.get('category') in (2,3) and not q.get('is_multi_modality') and q.get('evidence')]
        if args.multi_evidence:
            eligible=[(i,q) for i,q in eligible if len(set(q['evidence']))>1]
        eligible.sort(key=lambda v:hashlib.sha256(('raw-coverage-check:'+str(v[0])).encode()).hexdigest())
        cases=[dict(name='extra_'+str(i),user_id='release-text:conv-48',question=q['question'],
                    gold=q['evidence'],kind='dataset') for i,q in eligible[:args.dataset_extra]]
    with store.connect() as db:
        mapping={r['id']:sample['conversation'][r['session_id']][int(r['request_id'].rsplit(':',1)[1])+r['message_index']]['dia_id']
                 for r in db.execute("select * from rag_memories where user_id='release-text:conv-48'")}
    report=dict(kind='real providers; deterministic dataset subset' if args.dataset_extra else
        'real providers; six synthetic controls and two dataset cases; no upstream answering',cases=cases)
    for mode in args.modes:
        settings=dict(cfg,RAG_EVIDENCE_BUNDLE_MODE='raw' if mode=='coverage' else mode,
                      RAG_RAW_COVERAGE_MODE='on' if mode=='coverage' else 'off')
        store=VanillaMemory(settings)
        with TestClient(create_app(settings=settings,backend=store)) as client:
            client.headers['Authorization']='Bearer isolated-raw-test'
            assert client.get('/health').status_code==200
            for case in cases:
                trace={}; original=store.search
                def wrapped(payload,**kwargs): return original(payload,trace=trace)
                store.search=wrapped; start=time.perf_counter()
                try:
                    response=client.post('/search',json=dict(user_id=case['user_id'],query=case['question'],
                        top_k=1 if case['kind']=='synthetic' else 10))
                finally: store.search=original
                hits=response.json().get('data',[]); text='\n'.join(h['content'] for h in hits)
                ids={h['id'] for h in hits}
                ids.update(i for values in trace.get('bundle_sources',{}).values() for i in values)
                found=([g for g in case['gold'] if g in text] if case['kind']=='synthetic'
                       else sorted(set(case['gold']) & {mapping.get(i) for i in ids}))
                case[mode]=dict(status=response.status_code,seconds=time.perf_counter()-start,
                    found=found,response=response.json(),trace=trace)
                (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
                print(mode,case['name'],response.status_code,len(found),'/',len(case['gold']),
                      'fallback',trace.get('fallback'),flush=True)
    print('DONE',out,flush=True)


if __name__=='__main__': main()
