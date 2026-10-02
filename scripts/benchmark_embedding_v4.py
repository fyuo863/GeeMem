"""Real HTTP Add/Search on the public text corpus; compare frozen BGE+FM ranks."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import load_settings
from memory.evaluation import canonical_evidence, evidence_metrics
from memory.vanilla import chunks
from smoke_aml import auth_headers


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['source','reference','out']:
        p.add_argument('--'+name,required=True)
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=False)
    cfg=load_settings();source=Path(args.source)
    assert cfg['RAG_EMBEDDING_API_MODEL']=='text-embedding-v4'
    assert all(cfg.get(k)=='on' for k in ['RAG_FUSION_QA','RAG_SOFT_RECALL','RAG_METADATA_MODE','RAG_TARGET_MODE','RAG_SECOND_PASS'])
    assert cfg.get('RAG_MULTI_QUERY')=='off'
    samples=json.loads(source.read_text())
    refs={(c['sample_id'],c['question_index']):c for c in map(json.loads,Path(args.reference).read_text().splitlines())}
    prefix='v4-public:'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    manifest=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),prefix=prefix,
        settings={k:v for k,v in cfg.items() if k.startswith('RAG_') and not any(x in k for x in ['KEY','PROXY'])},
        notes='Public development data only; full memory and text-only questions; caller metadata matches prior BGE+FM test. Fresh HTTP top100; no Answer model. BGE historical latency is not directly comparable.')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    mapping={};add_times=[];cases=[];messages=0
    with httpx.Client(base_url=cfg['AML_BASE_URL'],headers=auth_headers(cfg),trust_env=False,timeout=1800) as client:
        assert client.get('/health').status_code==200
        def post(path,body):
            for attempt in range(4):
                r=client.post(path,json=body)
                if r.status_code not in (408,429,500,502,503,504) or attempt==3:break
                time.sleep(2**attempt)
            if r.status_code!=200:raise RuntimeError('HTTP '+str(r.status_code)+' at '+path)
            return r.json()
        for sample in samples:
            sid=sample['sample_id'];uid=prefix+':'+sid;conv=sample['conversation']
            for session,arr in conv.items():
                if not session.startswith('session_') or not isinstance(arr,list):continue
                stamp=int(datetime.strptime(conv[session+'_date_time'],'%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp()*1000)
                for start in range(0,len(arr),40):
                    batch=arr[start:start+40];rid=prefix+':'+sid+':'+session+':'+str(start)
                    body=dict(request_id=rid,user_id=uid,session_id=session,session_timestamp=stamp,
                        messages=[dict(role='user' if m['speaker']==conv['speaker_a'] else 'assistant',speaker=m['speaker'],content=m['text']) for m in batch])
                    before=time.perf_counter();result=post('/add',body);add_times.append(time.perf_counter()-before)
                    assert result==dict(success=True,request_id=rid,user_id=uid,session_id=session)
                    messages+=len(batch)
                    for i,m in enumerate(batch):
                        for j,text in enumerate(chunks(m['text'],int(cfg['RAG_CHUNK_TOKENS']),int(cfg['RAG_CHUNK_OVERLAP']))):
                            mid=hashlib.sha256(json.dumps([uid,rid,i,j]).encode()).hexdigest()
                            mapping[mid]=(sid,canonical_evidence([m['dia_id']])[0],text)
            print('ADDED',sid,messages,flush=True)
        (out/'ingestion.json').write_text(json.dumps(dict(messages=messages,chunks=len(mapping),batches=len(add_times),seconds=sum(add_times)),indent=2))
        with (out/'cases.jsonl').open('w') as stream:
            for sample in samples:
                sid=sample['sample_id'];known={v[1] for v in mapping.values() if v[0]==sid}
                for qi,q in enumerate(sample['qa']):
                    if q.get('is_multi_modality'):continue
                    ref=refs[(sid,qi)];assert ref['question']==q['question']
                    before=time.perf_counter()
                    hits=post('/search',dict(user_id=prefix+':'+sid,query=q['question'],top_k=100))['data']
                    seconds=time.perf_counter()-before
                    assert len(hits)<=100 and len({h['id'] for h in hits})==len(hits)
                    assert all(mapping[h['id']][0]==sid and mapping[h['id']][2]==h['content'] for h in hits)
                    assert all(a['score']>=b['score'] for a,b in zip(hits,hits[1:]))
                    gold=canonical_evidence(q.get('evidence'));assert gold==ref['evidence']
                    case=dict(sample_id=sid,question_index=qi,evidence=gold,missing_evidence=sorted(set(gold)-known),
                        v4=dict(ids=[h['id'] for h in hits],ranked_evidence=[mapping[h['id']][1] for h in hits],seconds=seconds),BGE=ref['FM'])
                    cases.append(case);stream.write(json.dumps(case)+'\n');stream.flush()
                    if len(cases)%20==0:print('SEARCHED',len(cases),'/',len(refs),flush=True)
    assert len(cases)==len(refs)
    metrics={}
    for k in [5,10,100]:
        metrics[str(k)]={}
        for variant in (['v4','BGE'] if k<=10 else ['v4']):
            valid=[dict(c,**{variant:dict(hit_evidence=c[variant]['ranked_evidence'][:k])}) for c in cases if not c['missing_evidence']]
            metrics[str(k)][variant]=evidence_metrics(valid,variant)
    valid=[c for c in cases if c['evidence'] and not c['missing_evidence']]
    def hit(c,n):return bool(set(c['evidence'])&set(c[n]['ranked_evidence'][:10]))
    changes={kind:[dict(sample_id=c['sample_id'],question_index=c['question_index']) for c in valid if (hit(c,'v4') and not hit(c,'BGE') if kind=='wins' else hit(c,'BGE') and not hit(c,'v4'))] for kind in ['wins','losses']}
    times=[c['v4']['seconds'] for c in cases]
    report=dict(questions=len(cases),messages=messages,metrics=metrics,changes=changes,
        http_search_latency=dict(mean=float(np.mean(times)),p50=float(np.median(times)),p95=float(np.percentile(times,95))),
        add_seconds=sum(add_times),official_evaluation=False,errors=0)
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',json.dumps(report),flush=True)


if __name__=='__main__':main()
