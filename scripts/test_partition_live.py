"""Small authored fixture; live add classifier, route selector, embedding and reranker."""
from datetime import datetime, timezone
from pathlib import Path
import json
import time
from fastapi.testclient import TestClient
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app


def main():
    out=Path('data/partition-live')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    cfg=dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve()),
        RAG_WRITE_GATE_MODE='on',RAG_BUILD_MODE='on',RAG_PARTITION_MODE='dual',
        RAG_MULTIHOP_MODE='off',RAG_RESULT_WINDOW='0',RAG_RERANK_CANDIDATES='8',
        RAG_TAG_MODE='off')
    b=VanillaMemory(cfg)
    facts=[
        'Lin prefers Arsenal to any other football team.',
        'Lin is allergic to peanuts.',
        'Veda is Lin\'s sister. Arun is Veda\'s teacher.',
        'Arun works at Riverside School.',
        'When writing reports for Lin, put the conclusion first and use bullet points.',
        'For urgent incident reports only, omit the usual introduction.',
        'Lin ran the Hangzhou Marathon in October 2023.',
        'Lin moved from Hangzhou to Shanghai in March 2024 and lives in Shanghai now.',
        'Wang supports Manchester United.',
        'Lin watched a Chelsea match yesterday, without supporting Chelsea.',
        'Lin bought a Tottenham shirt as a gift for Wang.',
        'Lin used to live in Hangzhou before moving away.',
        'Veda went to a concert in Beijing last Friday.',
        'Wang works at Mountain School.',
        'For Wang, reports should use paragraphs rather than bullet points.',
        'Lin likes almond cookies but must avoid peanuts.',
    ]
    cases=[('Which football team does Lin prefer?',[0]),
           ('What food is Lin allergic to?',[1]),
           ('Who is Veda\'s teacher?',[2]),
           ('Where does Arun work?',[3]),
           ('How should reports for Lin be formatted?',[4]),
           ('When should the usual report introduction be omitted?',[5]),
           ('Which marathon did Lin run in October 2023?',[6]),
           ('Where does Lin live after moving in March 2024?',[7])]
    results=[]
    headers={'Authorization':'Bearer test'}
    with TestClient(create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'test'},backend=b)) as c:
        response=c.post('/add',headers=headers,json=dict(user_id='u',request_id='r',session_id='s',
            messages=[dict(role='user',content=t) for t in facts]))
        if response.status_code!=200: raise RuntimeError('Add failed: '+str(response.status_code))
        for query,gold in cases:
            row={'query':query,'gold':gold,'modes':{}}
            for mode in ['off','strict','dual']:
                b.partition_search.mode=mode
                start=time.perf_counter()
                response=c.post('/search',headers=headers,json=dict(user_id='u',query=query,top_k=5))
                elapsed=time.perf_counter()-start
                hits=response.json().get('data',[])
                found={i for i in gold if any(h['content']==facts[i] for h in hits)}
                row['modes'][mode]=dict(status=response.status_code,seconds=elapsed,
                    hit5=int(bool(found)),recall5=len(found)/len(gold),hits=hits)
            results.append(row)
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            print('Completed',len(results),'/',len(cases),flush=True)
    summary={mode:{'hit5':sum(r['modes'][mode]['hit5'] for r in results)/len(results),
                   'recall5':sum(r['modes'][mode]['recall5'] for r in results)/len(results),
                   'mean_seconds':sum(r['modes'][mode]['seconds'] for r in results)/len(results)}
             for mode in ['off','strict','dual']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(out, json.dumps(summary),flush=True)


if __name__=='__main__':main()
