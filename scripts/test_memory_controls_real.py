"""Isolated real-provider end-to-end pilot for publication, masking and supplements."""
import json,sys,time,sqlite3
from pathlib import Path
from datetime import datetime,timezone
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import load_settings,PROJECT_ROOT
from memory.vanilla import VanillaMemory
from memory.aml_api import create_app
from fastapi.testclient import TestClient


def main():
    out=PROJECT_ROOT/'data/memory-controls'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ');out.mkdir(parents=True)
    cfg=load_settings();cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'),AML_API_KEY='local-controls',
        RAG_VERSION_MODE='on',RAG_DISCLOSURE_MODE='mask',RAG_SUPPLEMENTAL_MODE='on',
        RAG_EVIDENCE_BUNDLE_MODE='on',RAG_STATE_EVIDENCE_MODE='off')
    b=VanillaMemory(cfg)
    sessions=[['Lena planned a trip to Hangzhou on Friday.','Lena works at Atlas. Her email is lena@example.com and phone is 13800138000.'],
              ['Lena cancelled the Hangzhou trip and booked Suzhou for Saturday.','Ordinary office taxi rides require manager approval.','Exception: emergency hospital taxi trips need no prior approval, but a receipt is required.'],
              ['Lena arrived in Suzhou on Saturday and returned home on Sunday.','Nora completed a different trip to Hangzhou on Monday.']]
    questions=[('stage','How did Lena\'s trip plan change and what was the final outcome?', [sessions[0][0],sessions[1][0],sessions[2][0]]),
               ('rule','What is the rule for an emergency hospital taxi ride?',sessions[1][1:]),
               ('ordinary','What is the approval rule for an ordinary office taxi ride?',[sessions[1][1]]),
               ('privacy','Where does Lena work?',['Lena works at Atlas.']),
               ('isolation','Where does Nora travel?',[sessions[2][1]])]
    report={'add':[],'cases':[]}
    with TestClient(create_app(settings=cfg,backend=b)) as c:
        c.headers['Authorization']='Bearer local-controls';assert c.get('/health').status_code==200
        for i,texts in enumerate(sessions):
            body=dict(user_id='u',session_id=f's{i}',request_id=f'a{i}',messages=[dict(role='user',content=t,timestamp=1735689600000+i*86400000) for t in texts])
            r=c.post('/add',json=body);assert r.status_code==200
            version=b.versions.latest('u')
            assert c.post('/add',json=body).status_code==200 and b.versions.latest('u')==version
            report['add'].append({'version':version,'messages':len(texts),'idempotent':True})
            print('ADD',i,version,flush=True)
        for name,q,gold in questions:
            row={'name':name,'question':q}
            for mode in ['off','on']:
                b.multihop.supplemental=mode;b.multihop.planner.supplemental=mode=='on'
                trace={};original=b.search
                def wrapped(p,**kw):return original(p,trace=trace)
                b.search=wrapped;start=time.perf_counter()
                try:r=c.post('/search',json=dict(user_id='u',query=q,top_k=3))
                finally:b.search=original
                assert r.status_code==200
                text='\n'.join(h['content'] for h in r.json()['data'])
                assert 'lena@example.com' not in text and '13800138000' not in text
                row[mode]=dict(response=r.json(),source_coverage=sum(t in text for t in gold)/len(gold),seconds=time.perf_counter()-start,
                    read_version=trace.get('read_version'),search_calls=trace.get('search_calls'),extra_calls=trace.get('supplemental_calls',0),stop=trace.get('stop'))
                print(name,mode,row[mode]['source_coverage'],row[mode]['extra_calls'],flush=True)
            report['cases'].append(row)
            (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
        r=c.post('/search',json=dict(user_id='other',query='Lena',top_k=3));assert r.status_code==200 and r.json()['data']==[]
        report['isolation']=True
    (out/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    print('REPORT',out,flush=True)


if __name__=='__main__':main()
