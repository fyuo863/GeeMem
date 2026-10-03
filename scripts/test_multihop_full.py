"""Resumable full or stratified LongMemEval-S comparison; no generated answers."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import time

from test_multihop_live import (
    PROJECT_ROOT, HTTPEmbedder, LocalReranker, HTTPReranker, TracedMemory,
    TestClient, create_app, load_settings, batches, metrics, np, public_samples, synthetic,
)


def samples_from(path):
    for sample in json.loads(path.read_text(encoding='utf-8')):
        sessions, gold = [], []
        for sid, date, messages in zip(sample['haystack_session_ids'], sample['haystack_dates'],
                                       sample['haystack_sessions'], strict=True):
            stamp = int(datetime.strptime(date[:10]+' '+date[-5:], '%Y/%m/%d %H:%M')
                        .replace(tzinfo=timezone.utc).timestamp()*1000)
            sessions.append((sid, [dict(role=m['role'], content=m['content'], timestamp=stamp) for m in messages]))
            gold.extend(f'{sid}:{i}' for i, m in enumerate(messages) if m.get('has_answer'))
        yield dict(id=sample['question_id'], category=sample['question_type'], question=sample['question'],
                   sessions=sessions, gold=gold, abstention=sample['question_id'].endswith('_abs'))


class DocumentCache:
    """Only accelerates ingestion; query encoding always calls the real endpoint."""
    def __init__(self, inner, path):
        self.inner, self.identity = inner, inner.identity
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute('CREATE TABLE IF NOT EXISTS meta(identity TEXT)')
        old = self.db.execute('SELECT identity FROM meta').fetchone()
        if old and old[0] != self.identity:
            raise ValueError('Document cache model identity mismatch')
        if not old:
            self.db.execute('INSERT INTO meta VALUES (?)', (self.identity,))
        self.db.execute('CREATE TABLE IF NOT EXISTS vectors(key TEXT PRIMARY KEY, vector BLOB)')
        self.db.commit()

    @staticmethod
    def key(text):
        return hashlib.sha256(text.encode()).hexdigest()

    def documents(self, texts):
        rows = [self.db.execute('SELECT vector FROM vectors WHERE key=?', (self.key(t),)).fetchone() for t in texts]
        if any(r is None for r in rows):
            raise ValueError('Ingestion cache not prefetched')
        return np.stack([np.frombuffer(r[0], dtype='<f4') for r in rows])

    def queries(self, texts):
        return self.inner.queries(texts)

    def prefetch(self, texts, workers):
        missing = [t for t in dict.fromkeys(texts) if not self.db.execute(
            'SELECT 1 FROM vectors WHERE key=?', (self.key(t),)).fetchone()]
        def fetch(part):
            # Retries apply to benchmark ingestion only, never to measured Search.
            for attempt in range(3):
                try:
                    return self.inner.documents(part)
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(2 ** attempt)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fetch, missing[i:i+10]): missing[i:i+10]
                       for i in range(0, len(missing), 10)}
            for future in as_completed(futures):
                part = futures[future]
                vectors = future.result()
                self.db.executemany('INSERT OR IGNORE INTO vectors VALUES (?,?)',
                    [(self.key(t), v.astype('<f4').tobytes()) for t, v in zip(part, vectors, strict=True)])
                self.db.commit()
        return len(missing)


def summarize(cases, expected):
    result = dict(expected_questions=expected, completed_searches=len(cases), variants={})
    names = list(dict.fromkeys(c['variant'] for c in cases)) or ['baseline','multihop']
    for name in names:
        arm = [c for c in cases if c['variant'] == name]
        scored = [c for c in arm if c['gold'] and not c['abstention']]
        ks = [k for k in (3,5,10) if k<=min((c.get('top_k',10) for c in arm),default=10)]
        result['variants'][name] = dict(
            completed=len(arm), scored=len(scored), abstention=sum(c['abstention'] for c in arm),
            unannotated=sum(not c['gold'] and not c['abstention'] for c in arm),
            metrics={str(k): metrics(scored, k) for k in ks} if scored else {},
            categories={kind: {str(k): metrics([c for c in scored if c['category']==kind], k)
                               for k in ks} for kind in sorted({c['category'] for c in scored})},
            mean_seconds=float(np.mean([c['seconds'] for c in arm])) if arm else None,
            p95_seconds=float(np.percentile([c['seconds'] for c in arm], 95)) if arm else None,
            fallback_count=sum(c['trace'].get('fallback', False) for c in arm),
            strategies=dict(Counter(c['trace'].get('strategy', 'off') for c in arm)),
            stops=dict(Counter(c['trace'].get('stop', 'off') for c in arm)),
            llm_calls=sum(c['trace'].get('llm_calls', 0) for c in arm),
            search_calls=sum(c['trace'].get('search_calls', 1) for c in arm),
            abstention_nonempty=sum(bool(c['hits']) for c in arm if c['abstention']))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--embedding-workers', type=int, choices=range(1, 9), default=4)
    parser.add_argument('--per-category', type=int, choices=range(1, 31),
                        help='Deterministic answerable sample per category, excluding the prior eight pilot questions.')
    parser.add_argument('--cache-from', type=Path, help='Read-only copy of a compatible document cache.')
    parser.add_argument('--memory-from', type=Path, help='Read-only copy of a compatible ingested memory DB.')
    parser.add_argument('--ablation', action='store_true', help='Baseline, original multihop, binding-only, needs-only, combined.')
    parser.add_argument('--question-ids', nargs='+', help='Explicit diagnostic subset; not a held-out evaluation.')
    parser.add_argument('--prompt-baseline', type=Path, help='Frozen prompt JSON for paired previous/combined prompt-only comparison.')
    parser.add_argument('--synthetic', action='store_true', help='Four existing functional scenes plus two new dependency chains.')
    parser.add_argument('--top-k', type=int, choices=[3,5,10], default=10)
    parser.add_argument('--subquery-candidates',type=int,choices=range(1,101))
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=args.resume)
    source = PROJECT_ROOT/'data/longmemeval/longmemeval_s_cleaned.json'
    samples = list(samples_from(source))
    if args.synthetic:
        if args.per_category:
            parser.error('--synthetic cannot be combined with --per-category')
        samples = [dict(s,abstention=False) for s in synthetic()]
        definitions=[
            ('Which city hosts the headquarters of the company that employs Iris\'s mentor?',
             ['Iris is mentored by Dario.','Dario works for Aurora Systems.','Aurora Systems has its headquarters in Oslo.'],
             ['Iris works for Cobalt Media.','Cobalt Media has its headquarters in London.','Iris lives in Rome.']),
            ('What instrument does the instructor of Veda\'s brother teach?',
             ['Veda has a brother named Niko.','Niko takes lessons from Selene.','Selene teaches the clarinet.'],
             ['Veda takes lessons from Pascal.','Pascal teaches the violin.','Niko enjoys listening to the cello.'])]
        for i,(question,facts,distractors) in enumerate(definitions,4):
            texts=facts+distractors+['We discussed a garden and travel plans.']*6
            samples.append(dict(id=f'synthetic-{i}',category='chain',question=question,abstention=False,
                sessions=[(str(j),[dict(role='user',content=t,timestamp=1704067200000+j*60000)]) for j,t in enumerate(texts)],
                gold=['0:0','1:0','2:0']))
    if args.per_category:
        pilot_ids = {s['id'] for s in public_samples()}
        samples = [s for kind in sorted({s['category'] for s in samples})
                   for s in sorted((s for s in samples if s['category']==kind and s['gold']
                                    and not s['abstention'] and s['id'] not in pilot_ids),
                       key=lambda s: hashlib.sha256(('multihop-stratified-v1:'+s['id']).encode()).hexdigest())[:args.per_category]]
    if args.question_ids:
        selected=set(args.question_ids)
        available={s['id'] for s in samples}
        if selected-available:
            parser.error('Unknown question IDs: '+','.join(sorted(selected-available)))
        samples=[s for s in samples if s['id'] in selected]
    cfg = load_settings()
    cfg.update(RAG_MEMORY_DB=str(out/'memory.sqlite3'), RAG_RESULT_WINDOW='0',
               AML_AUTH_MODE='bearer', AML_API_KEY='local-full-test')
    if args.subquery_candidates is not None:
        cfg['RAG_MULTIHOP_CANDIDATES']=str(args.subquery_candidates)
    variants = ({'baseline':('off','off','off'), 'original':('llm','off','off'),
                 'bindings':('llm','on','off'), 'needs':('llm','off','on'), 'combined':('llm','on','on')}
                if args.ablation else {'baseline':('off','off','off'),'multihop':('llm','off','off')})
    old_prompts = None
    if args.prompt_baseline:
        if args.ablation:
            parser.error('--prompt-baseline cannot be combined with --ablation')
        old_prompts=json.loads(args.prompt_baseline.read_text(encoding='utf-8'))
        variants={'baseline':('off','off','off'),'previous':('llm','on','on'),'combined':('llm','on','on')}
    manifest = dict(dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        count=len(samples), top_k=args.top_k, sample_ids=[s['id'] for s in samples], per_category=args.per_category,
        synthetic=args.synthetic,
        variants={k:list(v) for k,v in variants.items()},
        source_sha256={p: hashlib.sha256((PROJECT_ROOT/p).read_bytes()).hexdigest()
                       for p in ['memory/multihop.py', 'memory/multihop_evidence.py', 'memory/vanilla.py',
                                 'scripts/test_multihop_live.py', 'scripts/test_multihop_full.py']},
        settings={k:v for k,v in cfg.items() if (k.startswith('RAG_') or k.startswith('LLM_'))
                  and not any(s in k for s in ('KEY', 'URL', 'PROXY'))},
        protocol='Full history for every selected question. Real in-process Add/Search. Document embedding cache only; uncached queries. No gold to retrieval/LLM. Abstention and unannotated cases excluded from evidence metrics.')
    if old_prompts is not None:
        manifest['prompt_baseline']=old_prompts
    manifest_path = out/'manifest.json'
    if args.resume:
        if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Resume requires identical dataset, code and settings')
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    cases_path = out/'cases.jsonl'
    cases = [json.loads(line) for line in cases_path.read_text(encoding='utf-8').split('\n') if line] if cases_path.exists() else []
    done = {(c['variant'], c['id']) for c in cases}
    if args.memory_from and not (out/'memory.sqlite3').exists():
        with closing(sqlite3.connect(args.memory_from.resolve().as_uri()+'?mode=ro', uri=True)) as src, closing(sqlite3.connect(out/'memory.sqlite3')) as dst:
            src.backup(dst)
    if args.cache_from and not (out/'document-cache.sqlite3').exists():
        with closing(sqlite3.connect(args.cache_from.resolve().as_uri()+'?mode=ro', uri=True)) as src, closing(sqlite3.connect(out/'document-cache.sqlite3')) as dst:
            src.backup(dst)
    embedding = DocumentCache(HTTPEmbedder(cfg), out/'document-cache.sqlite3')
    reranker = HTTPReranker(cfg) if cfg.get('RAG_RERANK_API_URL') else LocalReranker(cfg)
    stores = {name: TracedMemory(dict(cfg, RAG_MULTIHOP_MODE=mode, RAG_MULTIHOP_BINDINGS=binding,
                                    RAG_MULTIHOP_NEEDS=needs), embedding, reranker=reranker)
              for name, (mode,binding,needs) in variants.items()}
    if old_prompts is not None:
        # Per-instance replacement: identical code, schemas, budgets and retrieval;
        # only the instruction strings differ. No global monkeypatch or API keys.
        from memory import multihop, multihop_evidence
        replacements=[(getattr(module,key),old_prompts[key])
            for module,keys in [(multihop,['ROUTE_PROMPT','REVIEW_PROMPT']),
                (multihop_evidence,['NEED_PLAN','REVIEW_BASE','BINDING_RULES','NEED_RULES','LEGACY_QUERY_RULES'])]
            for key in keys]
        planner=stores['previous'].multihop.planner
        complete=planner.complete
        def previous_complete(instruction, payload, schema, timeout):
            for current, previous in replacements:
                instruction=instruction.replace(current,previous)
            return complete(instruction,payload,schema,timeout)
        planner.complete=previous_complete
    for index, sample in enumerate(samples, 1):
        if all((name, sample['id']) in done for name in stores):
            continue
        print('START', index, len(samples), sample['id'], flush=True)
        phase = 'prefetch'
        try:
            requests, mapping = batches(sample, cfg)
            before = time.perf_counter()
            fresh = embedding.prefetch([m['content'] for m in mapping.values()], args.embedding_workers)
            print('EMBEDDED', index, fresh, round(time.perf_counter()-before, 2), flush=True)
            # Rotate measured variants to distribute warm-up and transient network effects.
            names=list(stores)
            names=names[(index-1)%len(names):]+names[:(index-1)%len(names)]
            for name in names:
                store=stores[name]
                if (name, sample['id']) in done:
                    continue
                phase = name+'-add'
                with TestClient(create_app(settings=cfg, backend=store)) as client:
                    client.headers['Authorization'] = 'Bearer local-full-test'
                    for body in requests:
                        response = client.post('/add', json=body)
                        assert response.status_code == 200, response.status_code
                        assert response.json() == dict(success=True, **{k:body[k] for k in ('user_id','session_id','request_id')})
                    phase = name+'-search'
                    before = time.perf_counter()
                    response = client.post('/search', json=dict(user_id=requests[0]['user_id'], query=sample['question'], top_k=args.top_k))
                    seconds = time.perf_counter()-before
                    assert response.status_code == 200, response.status_code
                    hits = response.json()['data']
                    assert len(hits)<=args.top_k and len({h['id'] for h in hits})==len(hits)
                    assert all(h['id'] in mapping and h['content']==mapping[h['id']]['content'] for h in hits)
                    assert all(a['score']>=b['score'] for a,b in zip(hits,hits[1:]))
                    case = dict(variant=name, id=sample['id'], category=sample['category'],
                        question=sample['question'], gold=sample['gold'], abstention=sample['abstention'],
                        ranked_messages=[mapping[h['id']]['message'] for h in hits], hits=hits,
                        trace=store.last_trace, seconds=seconds, top_k=args.top_k)
                    with cases_path.open('a', encoding='utf-8') as f:
                        f.write(json.dumps(case, ensure_ascii=False)+'\n')
                    cases.append(case)
                    done.add((name, sample['id']))
                    (out/'report.json').write_text(json.dumps(summarize(cases, len(samples)), indent=2), encoding='utf-8')
                    print('RESULT', index, name, round(seconds, 2), store.last_trace.get('stop'), flush=True)
        except Exception as error:
            # Do not emit HTTP headers, keys or unbounded exception messages.
            failure = dict(id=sample['id'], phase=phase, error_type=type(error).__name__)
            with (out/'errors.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(failure)+'\n')
            print('ERROR', failure, flush=True)
    print('FINISHED', len(cases), '/', len(samples)*len(stores), flush=True)
    if len(cases) != len(samples)*len(stores):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
