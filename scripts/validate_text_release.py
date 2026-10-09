"""Public LoCoMo full textual HTTP contract test with resumable isolated stores.

No reference_time, custom speaker field, session_timestamp or gold is sent to API.
Inline mode preserves dataset-supplied speaker labels inside the raw content.
Gold is used only by the reporting step. Defaults to project .env for providers.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import threading

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.aml_api import create_app
from memory.embeddings import HTTPEmbedder
from memory.rerank import LocalReranker, HTTPReranker
from memory.vanilla import VanillaMemory, chunks
from memory.evaluation import canonical_evidence


class LockedReranker:
    def __init__(self, delegate):
        self.delegate, self.lock = delegate, threading.Lock()
        self.identity = delegate.identity
        self.relative_scores = getattr(delegate, 'relative_scores', False)

    def score(self, query, documents):
        with self.lock:
            return self.delegate.score(query, documents)


class TracedMemory(VanillaMemory):
    def search(self, payload, **kwargs):
        trace = {}
        result = super().search(payload, trace=trace)
        self.last_trace = trace
        return result


def load_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf8').splitlines()] if path.exists() else []


def append(path, data):
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(data, ensure_ascii=False)+'\n')


def metrics(cases, arm, k):
    valid = [c for c in cases if c['gold']]
    ns = [len(set(c['gold']) & set(c[arm]['ranked'][:k])) for c in valid]
    totals = [len(set(c['gold'])) for c in valid]
    return dict(hit=float(np.mean([n>0 for n in ns])),
                recall=float(np.mean([n/t for n,t in zip(ns, totals)])),
                micro_recall=sum(ns)/sum(totals),
                all_evidence=float(np.mean([n==t for n,t in zip(ns,totals)])),
                evaluated=len(valid), excluded_no_gold=len(cases)-len(valid))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--phase', choices=['all','add','search','report'], default='all')
    parser.add_argument('--speaker-mode',choices=['role_only','inline'],default='inline',
                        help='Preserve supplied dataset identity inside standard content; role_only reproduces the earlier lossy adapter.')
    parser.add_argument('--reranker',choices=['configured','minilm'],default='configured')
    parser.add_argument('--sample',action='append',help='Resume only selected sample IDs; reports still aggregate the complete run.')
    args = parser.parse_args()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    source = PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json'
    samples = json.loads(source.read_text(encoding='utf8'))
    if args.sample and not set(args.sample)<={s['sample_id'] for s in samples}:
        raise ValueError('Unknown sample ID')
    cfg = load_settings()
    assert cfg['LLM_MODEL']=='gpt-4o-mini'
    assert cfg['RAG_EMBEDDING_API_MODEL']=='text-embedding-v4'
    # Keep actual root .env retrieval/write settings; isolate only authentication
    # and database location. Never silently enable experiments for a release run.
    cfg.update(AML_AUTH_MODE='bearer',AML_API_KEY='isolated-public-test')
    if args.reranker=='minilm':
        cfg.pop('RAG_RERANK_API_URL',None)
        cfg.update(RAG_RERANK_MODE='local',RAG_RERANK_PATH='data/models/ms-marco-MiniLM-L-6-v2')
    safe = {k:v for k,v in cfg.items() if (k.startswith('RAG_') or k=='LLM_MODEL')
            and not any(word in k for word in ['KEY','URL','PROXY'])}
    fingerprint=hashlib.sha256(json.dumps(dict(settings=safe,speaker_mode=args.speaker_mode,reranker=args.reranker),sort_keys=True).encode()).hexdigest()
    manifest=out/'manifest.json'
    if manifest.exists():
        assert json.loads(manifest.read_text(encoding='utf8'))['settings_fingerprint']==fingerprint
    elif args.phase!='report':
        hashes={str(p.relative_to(PROJECT_ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (PROJECT_ROOT/'memory').glob('*.py')}
        manifest.write_text(json.dumps(dict(settings=safe,settings_fingerprint=fingerprint,
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_hashes=hashes,
            commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            request_contract='Standard textual fields; top_k=100; no query reference time',
            speaker_mode=args.speaker_mode,reranker=args.reranker,
            comparator='Same code/model/corpus, direct atomic retrieval (planning/partition/temporal off)',
            official_evaluation=False),indent=2),encoding='utf8')
    if args.phase!='report':
        reranker=LockedReranker(HTTPReranker(cfg) if cfg.get('RAG_RERANK_API_URL') else LocalReranker(cfg))
        def run(sample):
            sid=sample['sample_id']; folder=out/sid;folder.mkdir(exist_ok=True)
            local=dict(cfg,RAG_MEMORY_DB=str(folder/'memory.sqlite3'))
            backend=TracedMemory(local,reranker=reranker)
            # Avoid loading a separate GPU model for each conversation.
            backend.reranker=reranker
            from memory.retrieval import CallbackReranker
            backend.search_service.reranker=CallbackReranker(backend._rerank_for_multihop)
            comparator=TracedMemory(dict(local,RAG_MULTIHOP_MODE='off',RAG_PARTITION_MODE='off',
                RAG_TEMPORAL_MODE='off',RAG_TEMPORAL_EXPERIMENT='off',RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off'),
                backend.embedder,reranker=reranker)
            uid='release-text:'+sid; conv=sample['conversation']; mapping={}; batches=[]
            for session,messages in conv.items():
                if not session.startswith('session_') or not isinstance(messages,list):continue
                stamp=int(datetime.strptime(conv[session+'_date_time'],'%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp()*1000)
                for start in range(0,len(messages),20):
                    batch=messages[start:start+20];rid=sid+':'+session+':'+str(start)
                    def source_text(m):
                        return ('[speaker: '+m['speaker']+'] '+m['text']) if args.speaker_mode=='inline' else m['text']
                    body=dict(user_id=uid,request_id=rid,session_id=session,messages=[dict(
                        role='user' if m['speaker']==conv['speaker_a'] else 'assistant',
                        content=source_text(m),timestamp=stamp) for m in batch])
                    batches.append(body)
                    for i,m in enumerate(batch):
                        for j,text in enumerate(chunks(source_text(m),int(cfg.get('RAG_CHUNK_TOKENS','320')),int(cfg.get('RAG_CHUNK_OVERLAP','40')))):
                            mid=hashlib.sha256(json.dumps([uid,rid,i,j]).encode()).hexdigest()
                            mapping[mid]=(canonical_evidence([m['dia_id']])[0],text)
            with ExitStack() as stack:
                clients={name:stack.enter_context(TestClient(create_app(settings=local,backend=b)))
                         for name,b in [('current',backend),('direct',comparator)]}
                for client in clients.values(): client.headers['Authorization']='Bearer isolated-public-test'
                done={r['request_id'] for r in load_lines(folder/'add.jsonl') if r['status']==200}
                if args.phase in ('all','add'):
                    for body in batches:
                        if body['request_id'] in done:continue
                        for attempt in range(2):
                            t=time.perf_counter()
                            r=clients['current'].post('/add',json=body)
                            append(folder/'add.jsonl',dict(request_id=body['request_id'],status=r.status_code,
                                seconds=time.perf_counter()-t,messages=len(body['messages']),attempt=attempt))
                            if r.status_code==200:break
                        if r.status_code!=200: raise RuntimeError('Add failed '+sid+' HTTP '+str(r.status_code))
                        assert r.json()==dict(success=True,request_id=body['request_id'],user_id=uid,session_id=body['session_id'])
                        print('ADD',sid,len(done)+1,'/',len(batches),flush=True);done.add(body['request_id'])
                assert len(done)==len(batches),'Incomplete ingestion'
                if args.phase=='add':return
                done_q={r['question_index'] for r in load_lines(folder/'cases.jsonl')}
                for qi,q in enumerate(sample['qa']):
                    if q.get('is_multi_modality') or qi in done_q:continue
                    gold=canonical_evidence(q.get('evidence'))
                    assert set(gold)<=set(v[0] for v in mapping.values())
                    case=dict(sample_id=sid,question_index=qi,question=q['question'],category=q.get('category'),gold=gold)
                    for name in (['direct','current'] if qi%2==0 else ['current','direct']):
                        t=time.perf_counter()
                        r=clients[name].post('/search',json=dict(user_id=uid,query=q['question'],top_k=100))
                        record=dict(status=r.status_code,seconds=time.perf_counter()-t,ranked=[],ids=[])
                        if r.status_code==200:
                            hits=r.json()['data'];assert len(hits)<=100 and len({h['id'] for h in hits})==len(hits)
                            assert all(h['id'] in mapping and h['content']==mapping[h['id']][1] for h in hits)
                            assert all(a['score']>=b['score'] for a,b in zip(hits,hits[1:]))
                            record.update(ranked=[mapping[h['id']][0] for h in hits],ids=[h['id'] for h in hits])
                        trace=(backend if name=='current' else comparator).last_trace
                        record.update(fallback=trace.get('fallback',False),strategy=trace.get('strategy'),
                            llm_calls=trace.get('llm_calls',0),stop=trace.get('stop'),
                            routing_errors=sum('routing_error' in p for p in trace.get('partition_queries',[])),
                            error_type=trace.get('error_type'))
                        case[name]=record
                    append(folder/'cases.jsonl',case)
                    print('SEARCH',sid,len(done_q)+1,flush=True);done_q.add(qi)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures={pool.submit(run,s):s['sample_id'] for s in samples
                     if not args.sample or s['sample_id'] in args.sample}
            errors=[]
            for future in as_completed(futures):
                try:future.result()
                except Exception as e:
                    errors.append(dict(sample_id=futures[future],type=type(e).__name__,message=str(e)[:200]))
                    print('FAILED',futures[future],type(e).__name__,flush=True)
            (out/'errors.json').write_text(json.dumps(errors,indent=2),encoding='utf8')
    cases=[c for s in samples for c in load_lines(out/s['sample_id']/'cases.jsonl')]
    expected=sum(not q.get('is_multi_modality',False) for s in samples for q in s['qa'])
    report=dict(questions=len(cases),expected_questions=expected,complete=len(cases)==expected,
        metrics={str(k):{arm:metrics(cases,arm,k) for arm in ('direct','current')} for k in (5,10,100)} if cases else {})
    if cases:
        report['diagnostics']={arm:dict(errors=sum(c[arm]['status']!=200 for c in cases),
            fallbacks=sum(c[arm]['fallback'] for c in cases),routing_errors=sum(c[arm]['routing_errors'] for c in cases),
            p50=float(np.median([c[arm]['seconds'] for c in cases])),p95=float(np.percentile([c[arm]['seconds'] for c in cases],95)))
            for arm in ('direct','current')}
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('REPORT',json.dumps(report),flush=True)


if __name__=='__main__':main()
