"""Frozen eight-way F/Q/M experiment on an existing ABC corpus snapshot.

Only exact query/document model results are shared in accuracy evaluation.
Latency is uncached; no gold enters retrieval. No tuning is performed here.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import time
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT, load_settings
from memory.aml_api import AMLSearch
from memory.evaluation import canonical_evidence, evidence_metrics
from memory.vanilla import VanillaMemory, LocalEmbedder, chunks
from memory.rerank import LocalReranker
from benchmark_factorial import ScoreCache, QueryCache

VARIANTS={'ABC':(0,0,0),'F':(1,0,0),'Q':(0,1,0),'M':(0,0,1),
          'FQ':(1,1,0),'FM':(1,0,1),'QM':(0,1,1),'FQM':(1,1,1)}


def settings(cfg, name):
    return dict(cfg,**{k:'on' if v else 'off' for k,v in zip(
        ['RAG_FUSION_QA','RAG_MULTI_QUERY','RAG_SOFT_RECALL'],VARIANTS[name])})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True);p.add_argument('--snapshot',required=True)
    p.add_argument('--reference-cases',required=True);p.add_argument('--prefix',required=True)
    p.add_argument('--out',required=True);p.add_argument('--latency',action='store_true')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    cfg=load_settings();cfg['RAG_MEMORY_DB']=str(out/'memory.sqlite3')
    assert all(cfg.get(k)=='on' for k in ['RAG_METADATA_MODE','RAG_TARGET_MODE','RAG_SECOND_PASS'])
    assert cfg.get('RAG_RESULT_WINDOW')=='0' and cfg.get('RAG_TAG_MODE','off')=='off'
    source=Path(args.source);samples=json.loads(source.read_text())
    reference={(c['sample_id'],c['question_index']):c for c in map(json.loads,Path(args.reference_cases).read_text().splitlines())}
    mapping={};questions=[]
    for sample in samples:
        sid=sample['sample_id'];uid=args.prefix+':'+sid
        for session,arr in sample['conversation'].items():
            if not session.startswith('session_') or not isinstance(arr,list):continue
            for start in range(0,len(arr),40):
                rid=args.prefix+':'+sid+':'+session+':'+str(start)
                for i,m in enumerate(arr[start:start+40]):
                    for j,content in enumerate(chunks(m['text'])):
                        mid=hashlib.sha256(json.dumps([uid,rid,i,j]).encode()).hexdigest()
                        mapping[mid]=(sid,m['dia_id'],content)
        for qi,q in enumerate(sample['qa']):
            if not q.get('is_multi_modality'):questions.append((sid,qi,q))
    identity=dict(code={str(f.relative_to(PROJECT_ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in (PROJECT_ROOT/'memory').glob('*.py')},
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),prefix=args.prefix,
                  settings={k:v for k,v in cfg.items() if k.startswith('RAG_') and not any(w in k for w in ['KEY','PROXY'])},variants=VARIANTS,
                  selection_rule='Max full-corpus Hit@10; ties by Micro Recall@10, then macro Recall@10, then latency. Public development corpus; no unseen generalization claim.')
    if (out/'identity.json').exists():assert json.loads((out/'identity.json').read_text())==json.loads(json.dumps(identity)), 'Code/config changed; use a new output directory'
    else:(out/'identity.json').write_text(json.dumps(identity,indent=2))
    if not Path(cfg['RAG_MEMORY_DB']).exists():
        with sqlite3.connect('file:'+str(Path(args.snapshot).resolve())+'?mode=ro',uri=True) as src,sqlite3.connect(cfg['RAG_MEMORY_DB']) as dst:src.backup(dst)
    embed,rank=LocalEmbedder(cfg),LocalReranker(cfg)
    ec,rc=QueryCache(embed),ScoreCache(rank)
    stores={n:VanillaMemory(settings(cfg,n),embed if args.latency else ec,reranker=rank if args.latency else rc) for n in VARIANTS}
    def search(name,sid,q):
        hits=stores[name].search(AMLSearch(user_id=args.prefix+':'+sid,query=q['question'],top_k=10))['data']
        assert len(hits)<=10 and len(set(h['id'] for h in hits))==len(hits)
        assert all(mapping[h['id']][0]==sid and mapping[h['id']][2]==h['content'] for h in hits)
        return hits
    if args.latency:
        cases={(c['sample_id'],c['question_index']):c for c in map(json.loads,(out/'cases.jsonl').read_text().splitlines())}
        subset=[]
        for sample in samples:
            rows=[r for r in questions if r[0]==sample['sample_id']]
            subset.extend(rows[i] for i in np.linspace(0,len(rows)-1,6,dtype=int))
        search('FQM',subset[0][0],subset[0][2])
        elapsed={n:[] for n in VARIANTS};checks=[]
        for j,(sid,qi,q) in enumerate(subset):
            names=list(VARIANTS);shift=j%len(names)
            for name in names[shift:]+names[:shift]:
                begin=time.perf_counter();hits=search(name,sid,q);elapsed[name].append(time.perf_counter()-begin)
                ids=[h['id'] for h in hits]
                checks.append(dict(sample_id=sid,question_index=qi,variant=name,exact=ids==cases[(sid,qi)][name]['ids']))
            print('LATENCY',j+1,'/',len(subset),flush=True)
        result=dict(metrics={n:dict(count=len(v),mean=float(np.mean(v)),p50=float(np.median(v)),p95=float(np.percentile(v,95))) for n,v in elapsed.items()},checks=checks,
                    note='60 questions selected by fixed evenly spaced indices before results; uncached, alternating warm engine measurements; excludes HTTP and model startup.')
        (out/'latency.json').write_text(json.dumps(result,indent=2));print(json.dumps(result['metrics'],indent=2));return
    path=out/'cases.jsonl';cases=[json.loads(l) for l in path.read_text().splitlines()] if path.exists() else []
    done={(c['sample_id'],c['question_index']) for c in cases}
    with path.open('a') as stream:
        for sid,qi,q in questions:
            if (sid,qi) in done:continue
            ec.cache.clear();rc.values.clear()
            c=dict(sample_id=sid,question_index=qi,question=q['question'],category=q.get('category'),evidence=canonical_evidence(q.get('evidence')))
            for name in VARIANTS:
                hits=search(name,sid,q);ranked=[mapping[h['id']][1] for h in hits]
                c[name]=dict(ids=[h['id'] for h in hits],ranked_evidence=ranked,hit_evidence=sorted(set(ranked)&set(c['evidence'])))
            c['original_abc_parity']=c['ABC']['ids']==[h['id'] for h in reference[(sid,qi)]['ABC']['hits']]
            cases.append(c);stream.write(json.dumps(c)+'\n');stream.flush()
            if len(cases)%20==0:print('SEARCHED',len(cases),'/',len(questions),'model_pairs',rc.inferred,flush=True)
    assert len(cases)==len(questions)==861
    metrics={}
    for k in (5,10):
        subset=[]
        for c in cases:
            row=dict(c)
            for name in VARIANTS:row[name]=dict(hit_evidence=sorted(set(c[name]['ranked_evidence'][:k])&set(c['evidence'])))
            subset.append(row)
        metrics[str(k)]={n:evidence_metrics(subset,n) for n in VARIANTS}
    changes={n:dict(wins=sum(bool(c[n]['hit_evidence']) and not c['ABC']['hit_evidence'] for c in cases),
                    losses=sum(not c[n]['hit_evidence'] and bool(c['ABC']['hit_evidence']) for c in cases)) for n in VARIANTS}
    ranked=sorted(VARIANTS,key=lambda n:(-metrics['10'][n]['hit_rate'],-metrics['10'][n]['micro_recall'],-metrics['10'][n]['recall']))
    report=dict(questions=len(cases),variants=VARIANTS,metrics=metrics,changes=changes,ranked=ranked,winner=ranked[0],
                abc_parity=sum(c['original_abc_parity'] for c in cases),official_evaluation=False)
    (out/'report.json').write_text(json.dumps(report,indent=2));(out/'complete.json').write_text(json.dumps(dict(questions=861,searches=6888,errors=0)))
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
