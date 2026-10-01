"""Fresh full-corpus ABC vs pre-ABC baseline: real paired inference and write costs."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import psutil
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.aml_api import AMLAdd,AMLSearch
from memory.config import PROJECT_ROOT,load_settings
from memory.vanilla import VanillaMemory,LocalEmbedder,chunks
from memory.rerank import LocalReranker
from memory.evaluation import canonical_evidence

BASE_REF='47d9fab'


def git(*args):
    return subprocess.check_output(['git','-c','safe.directory='+str(PROJECT_ROOT),*args],cwd=PROJECT_ROOT,text=True,encoding='utf-8').strip()


def legacy_module(out,name,path):
    code=git('show',BASE_REF+':'+path)+'\n'
    filename=out/(name+'.py');filename.write_text(code,encoding='utf-8')
    spec=importlib.util.spec_from_file_location('memory.'+name,filename)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


class RecordedEmbedder:
    def __init__(self,model):self.model=model;self.identity=model.identity;self.replay=False;self.values={}
    def documents(self,texts):return self.model.documents(texts)
    def queries(self,texts):
        key=tuple(texts)
        if self.replay:return self.values[key]
        value=self.model.queries(texts);self.values[key]=value;return value


class RecordedReranker:
    def __init__(self,model):self.model=model;self.reset()
    def reset(self):self.replay=False;self.values={};self.calls=0;self.documents=0;self.seconds=0
    def score(self,query,documents):
        key=(query,tuple(documents))
        if self.replay:return self.values[key]
        before=time.perf_counter();value=self.model.score(query,documents);self.seconds+=time.perf_counter()-before
        self.calls+=1;self.documents+=len(documents);self.values[key]=value
        return value


def percentiles(values):
    return {k:float(v) for k,v in zip(['p50','p95','p99'],np.percentile(values,[50,95,99]))}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out');parser.add_argument('--text-only',action='store_true');args=parser.parse_args()
    out=PROJECT_ROOT/(args.out or 'data/abc-full-benchmarks/'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    out.mkdir(parents=True,exist_ok=False)
    print('OUTPUT',out,flush=True)
    cfg=load_settings()
    assert all(cfg.get(k)=='on' for k in ['RAG_METADATA_MODE','RAG_TARGET_MODE','RAG_SECOND_PASS'])
    assert cfg.get('RAG_RESULT_WINDOW')=='0' and int(cfg.get('RAG_RERANK_CANDIDATES','0'))>=100
    raw_path=PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json'
    samples=json.loads(raw_path.read_text(encoding='utf-8'))
    if args.text_only:
        # Keep the complete memory corpus for Add, but exclude multimodal-marked
        # questions from Search so this run measures text-only evaluation.
        samples=[dict(s,qa=[q for q in s['qa'] if not q.get('is_multi_modality')]) for s in samples]
    old=legacy_module(out,'_legacy_vanilla','memory/vanilla.py')
    old_api=legacy_module(out,'_legacy_api','memory/aml_api.py')
    before=time.perf_counter();embedder=LocalEmbedder(cfg);embed_load=time.perf_counter()-before
    before=time.perf_counter();reranker=LocalReranker(cfg);rerank_load=time.perf_counter()-before
    embed=RecordedEmbedder(embedder);rank=RecordedReranker(reranker)
    common=dict(cfg,RAG_TAG_MODE='off')
    stores={
        'baseline':old.VanillaMemory(dict(common,RAG_MEMORY_DB=str(out/'baseline.sqlite3')),embed,reranker=rank),
        'ABC':VanillaMemory(dict(common,RAG_MEMORY_DB=str(out/'ABC.sqlite3')),embed,reranker=rank)}
    manifest=dict(commit=git('rev-parse','HEAD'),baseline_commit=git('rev-parse',BASE_REF),
        source_sha256=hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        code_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (PROJECT_ROOT/'memory').glob('*.py')},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        settings={k:v for k,v in cfg.items() if k.startswith('RAG_') and not any(word in k for word in ['PROXY','KEY'])},
        embedder_identity=embedder.identity,reranker_identity=reranker.identity,
        device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        model_load_seconds=dict(embedding=embed_load,reranker=rerank_load),
        notes=('Fresh separate DBs; paired alternating Add/Search order. '
               + ('Text-only questions; multimodal-marked questions excluded from Search. ' if args.text_only else 'Full text inputs, including multimodal-labeled questions without images. ')
               + 'Timed top10 always fresh model inference. Top100 replays only that same fresh query call sequence, verified same top10; prefixes yield K5/10/20/50/100. No historic scores. Models shared, so RSS/GPU are process observations, not isolated service memory. No HTTP transport or answer generation.')
    )
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    embedder.documents(['Warm up the embedding model.']);reranker.score('warm up',['Warm up the reranker.'])
    mapping={};by_sample={};add_times={n:[] for n in stores};batch_count=0;messages=0
    for sample in samples:
        user='full-abc:'+sample['sample_id'];by_sample[sample['sample_id']]={}
        conv=sample['conversation']
        for session,batch_all in conv.items():
            if not session.startswith('session_') or not isinstance(batch_all,list):continue
            stamp=int(datetime.strptime(conv[session+'_date_time'],'%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp()*1000)
            for start in range(0,len(batch_all),40):
                batch=batch_all[start:start+40];rid=session+':'+str(start)
                plain=[dict(role='user' if m['speaker']==conv['speaker_a'] else 'assistant',content=m['text']) for m in batch]
                requests={'baseline':old_api.AMLAdd(user_id=user,session_id=session,request_id=rid,messages=plain),
                    'ABC':AMLAdd(user_id=user,session_id=session,request_id=rid,session_timestamp=stamp,
                        messages=[dict(p,speaker=m['speaker']) for p,m in zip(plain,batch)])}
                for name in (['baseline','ABC'] if batch_count%2==0 else ['ABC','baseline']):
                    before=time.perf_counter();stores[name].add(requests[name]);add_times[name].append(time.perf_counter()-before)
                batch_count+=1;messages+=len(batch)
                for i,m in enumerate(batch):
                    for j,text in enumerate(chunks(m['text'],int(cfg.get('RAG_CHUNK_TOKENS','320')),int(cfg.get('RAG_CHUNK_OVERLAP','40')))):
                        mid=hashlib.sha256(json.dumps([user,rid,i,j]).encode()).hexdigest()
                        mapping[mid]=dict(sample_id=sample['sample_id'],evidence_id=canonical_evidence([m['dia_id']])[0],content=text,speaker=m['speaker'],session=session,date=conv[session+'_date_time'])
                        by_sample[sample['sample_id']][mapping[mid]['evidence_id']]=m['text']
        print('ADDED',sample['sample_id'],messages,flush=True)
    ingestion={}
    for name,store in stores.items():
        with store.connect() as db:
            db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
            count=db.execute('SELECT COUNT(*) FROM rag_memories').fetchone()[0]
            table_stats={r[0]:r[1] for r in db.execute('SELECT name,COUNT(*) FROM sqlite_master WHERE type="table" GROUP BY name')}
        ingestion[name]=dict(messages=messages,chunks=count,batches=batch_count,total_seconds=sum(add_times[name]),
            messages_per_second=messages/sum(add_times[name]),batch_seconds=percentiles(add_times[name]),
            database_bytes=store.path.stat().st_size,tables=list(table_stats))
    (out/'ingestion.json').write_text(json.dumps(ingestion,indent=2),encoding='utf-8')
    (out/'source-map.json').write_text(json.dumps(mapping,ensure_ascii=False),encoding='utf-8')
    cases=[];process=psutil.Process()
    with (out/'cases.jsonl').open('w',encoding='utf-8') as stream:
        for sample in samples:
            for qi,q in enumerate(sample['qa']):
                gold=canonical_evidence(q.get('evidence'))
                case=dict(sample_id=sample['sample_id'],question_index=qi,question=q['question'],category=q.get('category'),
                    multimodal=bool(q.get('is_multi_modality')),evidence=gold,missing_evidence=sorted(set(gold)-set(by_sample[sample['sample_id']])) )
                for name in (['baseline','ABC'] if len(cases)%2==0 else ['ABC','baseline']):
                    rank.reset();embed.replay=False;embed.values.clear()
                    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
                    request=AMLSearch(user_id='full-abc:'+sample['sample_id'],query=q['question'],top_k=10)
                    before=time.perf_counter();hits=stores[name].search(request)['data'];elapsed=time.perf_counter()-before
                    timed=dict(seconds=elapsed,rerank_calls=rank.calls,rerank_documents=rank.documents,rerank_seconds=rank.seconds,
                        rss_bytes=process.memory_info().rss,gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None)
                    # Replay only this exact freshly executed query, never an earlier run.
                    rank.replay=True;embed.replay=True
                    top100=stores[name].search(request.model_copy(update={'top_k':100}))['data']
                    assert [h['id'] for h in hits]==[h['id'] for h in top100[:10]]
                    assert len(top100)<=100 and len(set(h['id'] for h in top100))==len(top100)
                    assert all(mapping[h['id']]['content']==h['content'] and mapping[h['id']]['sample_id']==sample['sample_id'] for h in top100)
                    ranked=[mapping[h['id']]['evidence_id'] for h in top100]
                    case[name]=dict(timed,ids=[h['id'] for h in top100],ranked_evidence=ranked,
                        hit_evidence=sorted(set(ranked[:10])&set(gold)))
                cases.append(case);stream.write(json.dumps(case,ensure_ascii=False)+'\n');stream.flush()
                if len(cases)%20==0:print('SEARCHED',len(cases),'/',sum(len(s['qa']) for s in samples),sample['sample_id'],flush=True)
    assert len(cases)==sum(len(s['qa']) for s in samples)
    (out/'complete.json').write_text(json.dumps(dict(queries=len(cases),searches=len(cases)*2,errors=0,source_preservation=True,top10_prefix_checks=len(cases)*2),indent=2),encoding='utf-8')
    print('COMPLETE',out,flush=True)


if __name__=='__main__':main()
