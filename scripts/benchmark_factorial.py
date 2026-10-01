"""Eight-way source-grounded ablation through the production search implementation.

Accuracy run shares exact query/document scores, not timing measurements.
Use --latency for uncached, counterbalanced warmed endpoint-engine timings.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import gzip
import itertools
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memory.aml_api import AMLAdd, AMLSearch
from memory.config import PROJECT_ROOT, load_settings
from memory.evaluation import canonical_evidence, evidence_metrics
from memory.rerank import LocalReranker
from memory.vanilla import LocalEmbedder, VanillaMemory, chunks

VARIANTS = {'base': (0,0,0), 'A': (1,0,0), 'B': (0,1,0), 'C': (0,0,1),
            'AB': (1,1,0), 'AC': (1,0,1), 'BC': (0,1,1), 'ABC': (1,1,1)}


class ScoreCache:
    def __init__(self, model):
        self.model, self.values = model, {}
        self.inferred = 0

    def score(self, query, documents):
        keys = {d: hashlib.sha256(json.dumps([query,d],ensure_ascii=False).encode()).hexdigest() for d in documents}
        missing = list(dict.fromkeys(d for d in documents if keys[d] not in self.values))
        if missing:
            scores = self.model.score(query, missing)
            assert len(scores) == len(missing) and np.isfinite(scores).all()
            self.values.update({keys[d]: float(v) for d,v in zip(missing,scores)})
            self.inferred += len(missing)
        return np.array([self.values[keys[d]] for d in documents])


class QueryCache:
    def __init__(self, model):
        self.model, self.identity, self.cache = model, model.identity, {}

    def documents(self, texts):
        return self.model.documents(texts)

    def queries(self, texts):
        key = tuple(texts)
        if key not in self.cache:
            self.cache[key] = self.model.queries(texts)
        return self.cache[key]


def config(out):
    return dict(load_settings(), RAG_MEMORY_DB=str(out/'memory.sqlite3'), RAG_TAG_MODE='off',
        RAG_MODEL_PATH='data/models/bge-small-en-v1.5', RAG_DEVICE='cuda',
        RAG_RERANK_PATH='data/models/ms-marco-MiniLM-L-6-v2', RAG_RERANK_DEVICE='cuda',
        RAG_RERANK_BATCH_SIZE='32', RAG_RERANK_MAX_LENGTH='512', RAG_RERANK_MODE='local',
        RAG_RERANK_CONTEXT='1', RAG_RERANK_CANDIDATES='400', RAG_RERANK_SELECTION='context_support',
        RAG_RERANK_NEIGHBOR_PENALTY='2', RAG_RESULT_WINDOW='0', RAG_RETRIEVAL_MODE='hybrid',
        RAG_RRF_K='60', RAG_LEXICAL_WEIGHT='0.5', RAG_CHUNK_TOKENS='320', RAG_CHUNK_OVERLAP='40')


def stores_for(cfg, embedder, ranker):
    return {name: VanillaMemory(dict(cfg,RAG_METADATA_MODE='on' if a else 'off',
        RAG_TARGET_MODE='on' if b else 'off', RAG_SECOND_PASS='on' if c else 'off'),embedder,reranker=ranker)
        for name,(a,b,c) in VARIANTS.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/factorial-benchmarks/20261001')
    parser.add_argument('--latency', action='store_true')
    args = parser.parse_args()
    out = (PROJECT_ROOT/args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    cfg = config(out)
    source = PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json'
    samples = json.loads(source.read_text(encoding='utf-8'))
    identity = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        implementation={name:hashlib.sha256((PROJECT_ROOT/name).read_bytes()).hexdigest() for name in
            ['memory/vanilla.py','memory/provenance.py','memory/target_rerank.py','memory/second_pass.py','memory/rerank.py',
             'memory/tags.py','scripts/benchmark_factorial.py']},
        config={k:v for k,v in cfg.items() if k.startswith('RAG_') and not any(t in k for t in ['PROXY','KEY','TOKEN'])})
    identity_path=out/'identity.json'
    if identity_path.exists():
        assert json.loads(identity_path.read_text(encoding='utf-8'))==identity, 'Use a new output directory after implementation/config changes'
    else:identity_path.write_text(json.dumps(identity,indent=2),encoding='utf-8')
    score_dir=out/'scores';score_dir.mkdir(exist_ok=True)
    embedder, ranker = LocalEmbedder(cfg), LocalReranker(cfg)
    cached_embedder, cached_ranker = QueryCache(embedder), ScoreCache(ranker)
    stores = stores_for(cfg, cached_embedder, cached_ranker)
    ingestion_path = out/'ingestion.json'
    before = time.perf_counter()
    mapping = {}
    for sample in samples:
        user = 'factorial:'+sample['sample_id']
        for session,messages in sample['conversation'].items():
            if not session.startswith('session_') or not isinstance(messages,list): continue
            date_text = sample['conversation'][session+'_date_time']
            # Dataset gives no timezone. UTC is an explicit date-preserving benchmark convention.
            stamp = int(datetime.strptime(date_text, '%I:%M %p on %d %B, %Y').replace(tzinfo=timezone.utc).timestamp()*1000)
            for start in range(0,len(messages),40):
                batch=messages[start:start+40]; request_id=session+':'+str(start)
                request=AMLAdd(user_id=user,request_id=request_id,session_id=session,session_timestamp=stamp,
                    messages=[dict(role='user' if m['speaker']==sample['conversation']['speaker_a'] else 'assistant',
                        speaker=m['speaker'],content=m['text']) for m in batch])
                stores['base'].add(request)
                for i,m in enumerate(batch):
                    for j,text in enumerate(chunks(m['text'])):
                        mid=hashlib.sha256(json.dumps([user,request_id,i,j]).encode()).hexdigest()
                        mapping[mid]=(sample['sample_id'],canonical_evidence([m['dia_id']])[0],text)
        print('INGESTED',sample['sample_id'],flush=True)
    if not ingestion_path.exists():
        ingestion_path.write_text(json.dumps(dict(seconds=time.perf_counter()-before,chunks=len(mapping),
            source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),session_timezone_assumption='UTC',
            models=[embedder.identity,ranker.identity]),indent=2),encoding='utf-8')
    questions = [(s,qi,q) for s in samples for qi,q in enumerate(s['qa'])
                 if not q.get('is_multi_modality') and canonical_evidence(q.get('evidence'))]
    def checked_search(store,sample,q,top_k=10):
        hits=store.search(AMLSearch(user_id='factorial:'+sample['sample_id'],query=q['question'],top_k=top_k))['data']
        assert len(hits)<=top_k and len({h['id'] for h in hits})==len(hits)
        assert all(mapping[h['id']][0]==sample['sample_id'] and mapping[h['id']][2]==h['content'] for h in hits)
        return hits
    if args.latency:
        # Fixed source-order sample independent of observed wins/losses, covering all conversations.
        subset=[]
        for s in samples:
            subset.extend([row for row in questions if row[0]['sample_id']==s['sample_id']][:3])
        fresh=stores_for(cfg,embedder,ranker)
        checked_search(fresh['ABC'],* (subset[0][i] for i in [0,2]))
        times={name:[] for name in VARIANTS}
        checks=[]
        for index,(s,qi,q) in enumerate(subset):
            names=list(VARIANTS); shift=index%len(names)
            for name in names[shift:]+names[:shift]:
                before=time.perf_counter();hits=checked_search(fresh[name],s,q);elapsed=time.perf_counter()-before
                times[name].append(elapsed)
                checks.append(dict(sample_id=s['sample_id'],question_index=qi,variant=name,ids=[h['id'] for h in hits]))
            print('LATENCY',index+1,'/',len(subset),flush=True)
        result={n:dict(n=len(t),p50_seconds=float(np.median(t)),p95_seconds=float(np.percentile(t,95))) for n,t in times.items()}
        (out/'latency.json').write_text(json.dumps(dict(metrics=result,cases=checks,notes='Uncached warm engine search including SQLite, query embedding, retrieval, all rerank calls and selection; excludes HTTP transport. Counterbalanced order, 30 questions.'),indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2));return
    casepath=out/'cases.jsonl'
    cases=[json.loads(l) for l in casepath.read_text(encoding='utf-8').splitlines()] if casepath.exists() else []
    complete={(c['sample_id'],c['question_index']) for c in cases}
    with casepath.open('a',encoding='utf-8') as stream:
        for s,qi,q in questions:
            if (s['sample_id'],qi) in complete:continue
            cached_ranker.values.clear();cached_embedder.cache.clear()
            score_path=score_dir/(s['sample_id']+'-'+str(qi)+'.json.gz')
            if score_path.exists():
                with gzip.open(score_path,'rt',encoding='utf-8') as handle:cached_ranker.values=json.load(handle)
            c=dict(sample_id=s['sample_id'],question_index=qi,question=q['question'],evidence=canonical_evidence(q['evidence']),
                   split='dev' if s in samples[:3] else 'validation')
            for name,store in stores.items():
                hits=checked_search(store,s,q)
                gold=set(c['evidence']); found={mapping[h['id']][1] for h in hits}&gold
                c[name]=dict(ids=[h['id'] for h in hits],hit_evidence=sorted(found),recall=len(found)/len(gold))
            with gzip.open(score_path.with_suffix('.tmp'),'wt',encoding='utf-8') as handle:
                json.dump(cached_ranker.values,handle)
            score_path.with_suffix('.tmp').replace(score_path)
            cases.append(c);stream.write(json.dumps(c,ensure_ascii=False)+'\n');stream.flush()
            if len(cases)%10==0:print('SEARCHED',len(cases),'/',len(questions), 'pairs',cached_ranker.inferred,flush=True)
    metrics={}
    for scope in ['all','dev','validation']:
        group=[c for c in cases if scope=='all' or c['split']==scope]
        metrics[scope]={}
        for name in VARIANTS:
            m=evidence_metrics(group,name)
            m.update(hit_wins=sum(bool(c[name]['hit_evidence']) and not c['base']['hit_evidence'] for c in group),
                     hit_losses=sum(not c[name]['hit_evidence'] and bool(c['base']['hit_evidence']) for c in group))
            metrics[scope][name]=m
    report=dict(metrics=metrics,variants=VARIANTS,questions=len(cases),top_k=10,llm_calls=0,
        commit=subprocess.check_output(['git','-c','safe.directory='+str(PROJECT_ROOT),'rev-parse','HEAD'],text=True,cwd=PROJECT_ROOT).strip(),
        notes='Full text-only public corpus; previously inspected diagnostic data, not unseen external evaluation. A uses caller-supplied source metadata. Exact model scores shared for accuracy only. No gold in retrieval. Rank by Hit10, ties by micro then macro evidence recall; report latency separately.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(metrics['all'],indent=2),flush=True)


if __name__=='__main__':main()
