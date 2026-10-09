"""Small real Add/Search comparison. Reports planner fallback separately from success."""
import argparse
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.aml_api import create_app
from memory.config import PROJECT_ROOT, load_settings
from memory.rerank import LocalReranker, HTTPReranker
from memory.vanilla import VanillaMemory, HTTPEmbedder, LocalEmbedder, chunks


class TracedMemory(VanillaMemory):
    def search(self,payload,**kwargs):
        self.last_trace={}
        return super().search(payload,trace=self.last_trace)

    def _search_direct(self, payload):
        from types import SimpleNamespace
        internal = SimpleNamespace(**(payload.model_dump() if hasattr(payload, 'model_dump') else vars(payload)))
        internal.retrieval_trace = self.last_trace
        return super()._search_direct(internal)


def synthetic():
    definitions=[
        ('direct','Where does Mira live?', ['Mira lives in Tallinn.']),
        ('split','What is the total cost of my bicycle and tent?',
         ['I paid $240 for my bicycle.','My tent cost $180.']),
        ('chain','Where does the spouse of Leona work?',
         ['Leona is married to Tomas.','Tomas works at Cedar Labs.']),
        ('chain','Who is the CEO of the company where the spouse of Nadia works?',
         ['Nadia is married to Elias.','Elias works at Solstice Analytics.','The CEO of Solstice Analytics is Ruth Chen.'])]
    out=[]
    for i,(strategy,question,evidence) in enumerate(definitions):
        texts=['We discussed a garden project and an upcoming trip.']*6+evidence+[
            'Mira recommended a book about Paris.', 'Leona works at Harbor Studio.',
            'Nadia visited Cedar Labs last year.', 'A bicycle repair cost $30.',
            'The rental tent cost $20 per day.', 'The CEO of Harbor Studio is Alan.']
        # Distinct sessions prevent adjacency alone from supplying the whole chain.
        messages=[dict(role='user',content=t,timestamp=1704067200000+j*60000) for j,t in enumerate(texts)]
        out.append(dict(id=f'synthetic-{i}',category=strategy,question=question,
            sessions=[(str(j),[m]) for j,m in enumerate(messages)],gold=[f'{j}:0' for j in range(6,6+len(evidence))]))
    return out


def public_samples(limit=10, kinds=('temporal-reasoning',)):
    source=PROJECT_ROOT/'data/longmemeval/longmemeval_s_cleaned.json'
    data=json.loads(source.read_text(encoding='utf-8'))
    result=[]
    for kind in kinds:
        selected=sorted((s for s in data if s['question_type']==kind and not s['question_id'].endswith('_abs')),
            key=lambda s:hashlib.sha256(('evidence-pilot-v1:'+s['question_id']).encode()).hexdigest())[:limit]
        for sample in selected:
            sessions=[];gold=[]
            for sid,date,messages in zip(sample['haystack_session_ids'],sample['haystack_dates'],sample['haystack_sessions'],strict=True):
                stamp=int(datetime.strptime(date[:10]+' '+date[-5:],'%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc).timestamp()*1000)
                sessions.append((sid,[dict(role=m['role'],content=m['content'],timestamp=stamp) for m in messages]))
                gold.extend(f'{sid}:{i}' for i,m in enumerate(messages) if m.get('has_answer'))
            reference=sample['question_date']
            reference_stamp=int(datetime.strptime(reference[:10]+' '+reference[-5:],'%Y/%m/%d %H:%M').replace(tzinfo=timezone.utc).timestamp()*1000)
            result.append(dict(id=sample['question_id'],category=kind,question=sample['question'],sessions=sessions,gold=gold,reference_time=reference_stamp))
    return result


def batches(sample,cfg):
    uid=('multihop:' if sample['id'].startswith('synthetic') else 'lme-pilot:')+sample['id']
    requests=[];mapping={}
    for sid,messages in sample['sessions']:
        expanded=[]
        for i,m in enumerate(messages):
            expanded.extend((dict(m,content=m['content'][j:j+30000]),i) for j in range(0,len(m['content']),30000)
                            if m['content'][j:j+30000].strip())
        for start in range(0,len(expanded),20):
            part=expanded[start:start+20];rid=f'{sample["id"]}:{sid}:{start}'
            requests.append(dict(user_id=uid,request_id=rid,session_id=sid,messages=[m for m,_ in part]))
            for i,(m,source_i) in enumerate(part):
                for j,text in enumerate(chunks(m['content'],int(cfg.get('RAG_CHUNK_TOKENS','320')),int(cfg.get('RAG_CHUNK_OVERLAP','40')))):
                    key=hashlib.sha256(json.dumps([uid,rid,i,j]).encode()).hexdigest()
                    mapping[key]=dict(message=f'{sid}:{source_i}',content=text)
    return requests,mapping


def metrics(cases,k):
    recalls=[len(set(c['gold'])&set(c['ranked_messages'][:k]))/len(set(c['gold'])) for c in cases]
    return dict(hit=float(np.mean([r>0 for r in recalls])),recall=float(np.mean(recalls)),
        micro_recall=sum(len(set(c['gold'])&set(c['ranked_messages'][:k])) for c in cases)/sum(len(set(c['gold'])) for c in cases),
        all_evidence=float(np.mean([r==1 for r in recalls])))


def main():
    parser=argparse.ArgumentParser(description='Paired temporal retrieval ablation on cached real LongMemEval histories.')
    parser.add_argument('--out',required=True)
    parser.add_argument('--public-cache',type=Path,required=True)
    parser.add_argument('--count',type=int,default=10)
    parser.add_argument('--controls-per-kind',type=int,default=0,help='Cached non-abstention controls per other dataset category.')
    parser.add_argument('--experiment',choices=['off','soft_score','absolute_query','window'],default='off')
    parser.add_argument('--baseline-experiment',choices=['off','absolute_query'],default='off')
    parser.add_argument('--resolved-only',action='store_true',help='Diagnostic replay of resolvable questions from the selected set.')
    parser.add_argument('--rerank-mode',choices=['off','onnx','local'],default='onnx')
    parser.add_argument('--english-minilm',action='store_true',help='Use the verified local English MiniLM reranker for both arms.')
    parser.add_argument('--live-annotations',type=int,default=4,help='Maximum real annotation sources across the run.')
    args=parser.parse_args()
    out=Path(args.out).resolve();out.mkdir(parents=True,exist_ok=False)
    cfg=load_settings()
    cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'),RAG_RESULT_WINDOW='0',RAG_TEMPORAL_MODE='off',
               RAG_RERANK_MODE=args.rerank_mode,RAG_MULTIHOP_MODE='off',RAG_PARTITION_MODE='off',
               RAG_WRITE_GATE_MODE='off',RAG_BUILD_MODE='off',RAG_TIME_ANNOTATION_MODE='off',
               AML_AUTH_MODE='bearer',AML_API_KEY='local-test')
    cfg['RAG_TEMPORAL_EXPERIMENT']=args.baseline_experiment
    if args.experiment!='off': cfg['RAG_TEMPORAL_MODE']='on'
    if args.english_minilm:
        cfg.update(RAG_RERANK_MODE='local',RAG_RERANK_PATH='data/models/ms-marco-MiniLM-L-6-v2')
        cfg.pop('RAG_RERANK_API_URL',None)
    with closing(sqlite3.connect(args.public_cache.resolve().as_uri()+'?mode=ro',uri=True)) as src:
        available={r[0] for r in src.execute('select distinct user_id from rag_memories')}
        samples=[s for s in public_samples(1000) if 'lme-pilot:'+s['id'] in available][:args.count]
        if len(samples)!=args.count: raise ValueError('Not enough cached temporal questions')
        for kind in ('single-session-user','single-session-assistant','single-session-preference','multi-session','knowledge-update'):
            controls=[s for s in public_samples(1000,(kind,)) if 'lme-pilot:'+s['id'] in available][:args.controls_per_kind]
            if len(controls)!=args.controls_per_kind: raise ValueError('Not enough cached control questions')
            samples.extend(controls)
        if args.resolved_only:
            from memory.query_time import resolve_query_time
            samples = [s for s in samples if resolve_query_time(s['question'], s['reference_time']) is not None]
        # Freeze selections and verify complete histories before any search.
        inventory=[]
        for sample in samples:
            _, mapping=batches(sample,cfg)
            cached=dict(src.execute('SELECT id,content FROM rag_memories WHERE user_id=?',('lme-pilot:'+sample['id'],)))
            if cached!={k:v['content'] for k,v in mapping.items()}:
                raise ValueError('Cache differs from full dataset history: '+sample['id'])
            if not sample['gold']: raise ValueError('Evidence recall requires nonempty gold')
            inventory.append(dict(id=sample['id'],category=sample['category'],chunks=len(mapping),gold_count=len(set(sample['gold']))))
        (out/'selection.json').write_text(json.dumps(inventory,indent=2),encoding='utf8')
        with closing(sqlite3.connect(cfg['RAG_MEMORY_DB'])) as dst: src.backup(dst)
    embedder=HTTPEmbedder(cfg) if cfg.get('RAG_EMBEDDING_API_URL') else LocalEmbedder(cfg)
    # Query vectors are memoized so the two arms consume byte-identical vectors.
    class CachedQueries:
        identity=embedder.identity
        cache={}
        def queries(self,texts):
            key=tuple(texts)
            if key not in self.cache: self.cache[key]=embedder.queries(texts)
            return self.cache[key]
        def documents(self,texts): return embedder.documents(texts)
    off=TracedMemory(cfg,CachedQueries())
    on=TracedMemory(dict(cfg,RAG_TEMPORAL_MODE='on',RAG_TEMPORAL_EXPERIMENT=args.experiment),off.embedder,reranker=off.reranker)
    from memory.aml_api import AMLAdd
    from memory.temporal_index import TemporalIndex
    from memory.temporal import TemporalNormalizer
    from memory.annotator import TimeAnnotator
    from memory.llm import LLM
    from memory.provenance import source_id
    class RecordingAnnotator:
        def __init__(self): self.delegate=TimeAnnotator(LLM(max_attempts=2));self.calls=0
        def annotate(self,sources):
            self.calls+=1
            result=self.delegate.annotate(sources)
            with (out/'annotations.jsonl').open('a',encoding='utf8') as f:
                f.write(json.dumps(dict(inputs=sources,outputs=result),ensure_ascii=False)+'\n')
            return result
    recorder=RecordingAnnotator()
    live=TemporalIndex(on.connect,annotator=recorder)
    cases=[]; changed=0
    for sample_number,sample in enumerate(samples):
        requests,mapping=batches(sample,cfg)
        # Backfill side index only. Never rewrite cached source payloads or
        # re-embed them; source IDs and full histories remain unchanged.
        for body in requests:
            on.temporal_index.write(AMLAdd.model_validate(body))
            if recorder.calls < args.live_annotations:
                for i,m in enumerate(body['messages']):
                    if TemporalNormalizer.mentions(m['content'],m['timestamp']) and len(m['content'])<=6000:
                        # Preserve actual message position when selecting one
                        # bounded annotation call from this Add payload.
                        from types import SimpleNamespace
                        class OneSource:
                            def annotate(self,sources):
                                if recorder.calls>=args.live_annotations: raise ValueError('budget')
                                return recorder.annotate(sources)
                        live.annotator=OneSource()
                        live.write(AMLAdd.model_validate(body))
                        break
        print('indexed',sample['id'],flush=True)
        query_started=time.perf_counter()
        off.embedder.queries([sample['question']])
        if args.experiment in ('absolute_query','window') or args.baseline_experiment=='absolute_query':
            from memory.query_time import resolve_query_time, expanded_query
            off.embedder.queries([expanded_query(sample['question'],resolve_query_time(sample['question'],sample['reference_time']))])
        query_seconds=time.perf_counter()-query_started
        arms=[('off',off),('on',on)]
        if sample_number%2: arms.reverse()
        for name,store in arms:
            with TestClient(create_app(settings=cfg,backend=store)) as client:
                client.headers['Authorization']='Bearer local-test'
                before=time.perf_counter()
                r=client.post('/search',json=dict(user_id='lme-pilot:'+sample['id'],query=sample['question'],top_k=10,
                                                 reference_time=sample['reference_time'],reference_timezone='UTC'))
                seconds=time.perf_counter()-before
                assert r.status_code==200,r.status_code
                hits=r.json()['data']
                assert all(h['id'] in mapping and h['content']==mapping[h['id']]['content'] for h in hits)
                assert len(hits)<=10 and len({h['id'] for h in hits})==len(hits)
                assert all(a['score']>=b['score'] for a,b in zip(hits,hits[1:]))
                from memory.query_time import resolve_query_time
                resolved=resolve_query_time(sample['question'],sample['reference_time'])
                c=dict(variant=name,id=sample['id'],category=sample['category'],question=sample['question'],gold=sample['gold'],
                       reference_time=sample['reference_time'],resolved_time=resolved.description() if resolved else None,
                       temporal_triggered=bool(__import__('memory.temporal',fromlist=['TemporalRanker']).TemporalRanker.QUERY.search(sample['question'])),
                       query_embedding_seconds=query_seconds,trace=store.last_trace,
                       candidate_messages=[mapping[mid]['message'] for mid in store.last_trace.get('candidate_ids',[])],
                       ranked_messages=[mapping[h['id']]['message'] for h in hits],hits=hits,seconds=seconds)
                cases.append(c)
                with (out/'cases.jsonl').open('a',encoding='utf8') as f:f.write(json.dumps(c,ensure_ascii=False)+'\n')
                print(name,sample['id'],metrics([c],10),flush=True)
        changed+=cases[-1]['ranked_messages']!=cases[-2]['ranked_messages']
    report={name:dict(metrics={str(k):metrics([c for c in cases if c['variant']==name],k) for k in (5,10)},
             mean_seconds=float(np.mean([c['seconds'] for c in cases if c['variant']==name]))) for name in ('off','on')}
    report.update(count=len(samples),changed_rankings=changed,annotation_calls=recorder.calls,experiment=args.experiment,baseline_experiment=args.baseline_experiment,
                  protocol='Real /search through in-process HTTP, cached full-history vectors, side-index backfill; planning/partition disabled in both arms; no answer generation.',
                  reranker='MiniLM' if args.english_minilm else args.rerank_mode,query_timing_caveat='Query vectors prefetched, including rewritten query when used; alternating arm order; times exclude remote embedding.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    manifest=dict(ids=[s['id'] for s in samples],experiment=args.experiment,revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                  source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),PROJECT_ROOT/'memory/temporal.py',PROJECT_ROOT/'memory/temporal_index.py',PROJECT_ROOT/'memory/vanilla.py',PROJECT_ROOT/'memory/query_time.py',PROJECT_ROOT/'memory/aml_api.py']},
                  dataset_sha256=hashlib.sha256((PROJECT_ROOT/'data/longmemeval/longmemeval_s_cleaned.json').read_bytes()).hexdigest(),
                  settings={k:v for k,v in cfg.items() if (k.startswith('RAG_') or k=='LLM_MODEL') and not any(s in k for s in ('KEY','URL','PROXY'))})
    manifest['baseline_experiment'] = args.baseline_experiment
    manifest['resolved_only'] = args.resolved_only
    window_source = PROJECT_ROOT/'memory/temporal_window.py'
    if window_source.exists():
        manifest['source_hashes'][str(window_source)] = hashlib.sha256(window_source.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('COMPLETE',out,flush=True)


if __name__=='__main__': main()
