"""Real /search on a backup of the isolated synthetic source test database."""
import json
import sqlite3
import sys
import time
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fastapi.testclient import TestClient
from memory.config import PROJECT_ROOT,load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app
from test_state_evidence_real import CASES


def main():
    out=PROJECT_ROOT/'data/evidence-bundle'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    source=PROJECT_ROOT/'data/state-evidence/20261010T024712Z/memory.sqlite3'
    with sqlite3.connect(source.as_uri()+'?mode=ro',uri=True) as src,sqlite3.connect(out/'memory.sqlite3') as dst:
        src.backup(dst)
    cfg=load_settings();cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'),AML_API_KEY='bundle-local',
                                   RAG_EVIDENCE_BUNDLE_MODE='on',RAG_STATE_EVIDENCE_MODE='off')
    store=VanillaMemory(cfg);rows=[]
    with TestClient(create_app(settings=cfg,backend=store)) as c:
        c.headers['Authorization']='Bearer bundle-local'
        for name,q,texts,gold in CASES:
            if name not in ['chain_control','rule_en','current_en','conflict']:continue
            row=dict(name=name,question=q,gold=[texts[i] for i in gold])
            for mode in ['off','on']:
                store.multihop.bundler.mode=mode;trace={};original=store.search
                def wrapped(p,**kw):return original(p,trace=trace)
                store.search=wrapped;start=time.perf_counter()
                try:r=c.post('/search',json=dict(user_id=name,query=q,top_k=1))
                finally:store.search=original
                assert r.status_code==200
                hits=r.json()['data'];assert len(hits)<=1
                text='\n'.join(h['content'] for h in hits)
                row[mode]=dict(result=r.json(),seconds=time.perf_counter()-start,
                               source_recall=sum(t in text for t in row['gold'])/len(gold),
                               chars=len(text),llm_calls=trace.get('llm_calls'),bundle=trace.get('evidence_bundle'),
                               stop=trace.get('stop'))
                print(name,mode,row[mode]['source_recall'],row[mode]['bundle'],flush=True)
            rows.append(row)
            (out/'report.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    with sqlite3.connect(out/'memory.sqlite3') as db:
        maps=db.execute('select user_id,bundle_id,source_ids from rag_evidence_bundles').fetchall()
        for user,bundle_id,ids in maps:
            for sid in json.loads(ids):
                assert db.execute('select 1 from rag_memories where user_id=? and id=?',(user,sid)).fetchone()
    print('DONE',out,'verified mappings',len(maps),flush=True)


if __name__=='__main__':main()
