"""24 source-only public dataset writes; save all sources and persisted records."""
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
import json
import time
import argparse
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app

OUT=Path('data/add-dataset-audit-20261008')

def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default=str(OUT))
    args=parser.parse_args()
    OUT=Path(args.output)
    OUT.mkdir(parents=True,exist_ok=True)
    public=Path('data/locomo-refined/data/public')
    conversations={c['sample_id']:c for c in map(json.loads,(public/'conversations.jsonl').read_text(encoding='utf8').splitlines())}
    questions=list(map(json.loads,(public/'questions.jsonl').read_text(encoding='utf8').splitlines()))
    selected=[]
    for category in ('1','2','3','4'):
        pool=[q for q in questions if q['category']==category and not q['is_multi_modality'] and q['evidence_messages']]
        # Deterministic spread across conversations rather than first six in one conversation.
        pool.sort(key=lambda q:(q['qa_index'],q['sample_id']))
        selected.extend(pool[:6])
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((OUT/'memory.sqlite3').resolve()),RAG_WRITE_GATE_MODE='on',
        RAG_BUILD_MODE='on',RAG_MULTIHOP_MODE='off',RAG_RERANK_MODE='off',RAG_TAG_MODE='off',
        RAG_TARGET_MODE='off',RAG_FUSION_QA='off',RAG_RERANK_SELECTION='direct')
    store=VanillaMemory(cfg)
    tables=['profile_facts','relationship_assertions','rule_records','event_records']
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'audit'},backend=store)) as client:
        for number,q in enumerate(selected,1):
            target=OUT/f'{number:02}.json'
            if target.exists():continue
            evidence=set(q['evidence']); sessions=[]
            for session in conversations[q['sample_id']]['sessions']:
                indices={i for i,m in enumerate(session['messages']) if m['dia_id'] in evidence}
                visible=sorted({j for i in indices for j in range(max(0,i-2),i+1)})
                if visible:sessions.append((session,visible))
            uid=f'audit-{number}'; requests=[]; elapsed=0
            for session,indices in sessions:
                stamp=int(datetime.strptime(session['date_time'],'%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp()*1000)
                source=[session['messages'][i] for i in indices]
                payload=dict(user_id=uid,request_id=f"s{session['session_index']}",session_id=f"s{session['session_index']}",
                    messages=[dict(role=m['role'],speaker=m['speaker'],content=m['text'],timestamp=stamp) for m in source])
                start=time.perf_counter()
                response=client.post('/add',json=payload,headers={'Authorization':'Bearer audit'})
                seconds=time.perf_counter()-start; elapsed+=seconds
                requests.append(dict(source=source,payload=payload,status=response.status_code,seconds=seconds))
            with closing(store.connect()) as db:
                routes=[dict(r) for r in db.execute('SELECT * FROM rag_memory_routes WHERE user_id=?',(uid,))]
                existing={r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                records={t:[dict(r) for r in db.execute(f'SELECT * FROM {t} WHERE user_id=?',(uid,))] if t in existing else [] for t in tables}
                if 'event_sources' in existing:
                    records['event_sources']=[dict(r) for r in db.execute('SELECT * FROM event_sources WHERE user_id=?',(uid,))]
                entities=[dict(r) for r in db.execute('SELECT * FROM relationship_entities WHERE user_id=?',(uid,))]
                links={}
                for table,owner,parent in [('profile_evidence','fact_id','profile_facts'),('relationship_evidence','relationship_id','relationship_assertions'),('rule_evidence','rule_id','rule_records'),('event_evidence','event_id','event_records')]:
                    if table not in existing:
                        links[table]=[]
                        continue
                    links[table]=[dict(r) for r in db.execute(f'SELECT e.* FROM {table} e JOIN {parent} p ON p.id=e.{owner} WHERE p.user_id=?',(uid,))]
            result=dict(number=number,question=q,requests=requests,elapsed_s=elapsed,routes=routes,records=records,entities=entities,evidence=links)
            target.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
            print(number,q['category'],q['qa_id'],round(elapsed,2),[r['status'] for r in routes],flush=True)

if __name__=='__main__':main()
