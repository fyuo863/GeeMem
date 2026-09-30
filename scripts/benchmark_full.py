"""Full public LoCoMo-Refined text-input comparison; no answer/judge or private data."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import subprocess
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT,load_settings
from memory.vanilla import VanillaMemory,LocalEmbedder,chunks
from memory.aml_api import AMLAdd,AMLSearch
from memory.evaluation import evidence_metrics,canonical_evidence

NAMES=['baseline','semantic_filter','semantic_rank']
KS=[5,10,100]


def summarize(cases):
    result={}
    for scope in ['all_text_input','text_only','multimodal_text_input']:
        scoped=[c for c in cases if scope=='all_text_input' or (c['multimodal'] == (scope=='multimodal_text_input'))]
        result[scope]={}
        for k in KS:
            subset=[c for c in scoped if c['top_k']==k]
            valid=[c for c in subset if c['evidence'] and not c['missing_evidence']]
            result[scope][str(k)]={}
            for name in NAMES:
                metrics=evidence_metrics(valid,name)
                metrics.update(total_questions=len(subset),excluded_no_gold=sum(not c['evidence'] for c in subset),
                    excluded_missing_gold=sum(bool(c['missing_evidence']) for c in subset),
                    p50_seconds=float(np.median([c[name]['seconds'] for c in subset])) if subset else None,
                    p95_seconds=float(np.percentile([c[name]['seconds'] for c in subset],95)) if subset else None,
                    mean_results=float(np.mean([c[name]['count'] for c in subset])) if subset else None)
                if name!='baseline':
                    metrics.update(wins=sum(c[name]['recall']>c['baseline']['recall'] for c in valid),
                                   losses=sum(c[name]['recall']<c['baseline']['recall'] for c in valid))
                result[scope][str(k)][name]=metrics
    return result


def main():
    source=PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json'
    samples=json.loads(source.read_text(encoding='utf-8'))
    out=PROJECT_ROOT/'data/full-semantic-benchmarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    cfg=load_settings();model=LocalEmbedder(cfg)
    common=dict(cfg,RAG_RESULT_WINDOW='1',RAG_RESULT_WINDOW_SEED_K='20',RAG_TAG_CANDIDATES='400',
                RAG_TAG_WEIGHT='0.5',RAG_TAG_THRESHOLD='0.5',RAG_RETRIEVAL_MODE='hybrid',RAG_RRF_K='60',
                RAG_LEXICAL_WEIGHT='0.5',RAG_CHUNK_TOKENS='320',RAG_CHUNK_OVERLAP='40')
    stores={name:VanillaMemory(dict(common,RAG_MEMORY_DB=str(out/(name+'.sqlite3')),
            RAG_TAG_MODE='off' if name=='baseline' else name),model) for name in NAMES}
    model.documents(['Warm up retrieval embedding.'])
    cases=[];sample_reports=[];times={n:0. for n in NAMES};batch_index=0;question_index=0
    print('OUTPUT',out,flush=True)
    with (out/'cases.jsonl').open('w',encoding='utf-8') as stream:
        for sample in samples:
            sid=sample['sample_id'];conv=sample['conversation'];user='public-full:'+sid
            mapping={};message_count=0;local_times={n:0. for n in NAMES}
            sessions=sorted([k for k,v in conv.items() if k.startswith('session_') and isinstance(v,list)],key=lambda k:int(k.split('_')[1]))
            for session in sessions:
                for start in range(0,len(conv[session]),40):
                    batch=conv[session][start:start+40];rid=session+':'+str(start)
                    payload=AMLAdd(user_id=user,session_id=session,request_id=rid,
                        messages=[dict(role='user' if m['speaker']==conv['speaker_a'] else 'assistant',content=m['text']) for m in batch])
                    for i,m in enumerate(batch):
                        for j,_ in enumerate(chunks(m['text'],320,40)):
                            mid=hashlib.sha256(json.dumps([user,rid,i,j]).encode()).hexdigest()
                            mapping[mid]=canonical_evidence([m['dia_id']])[0]
                    offset=batch_index%3
                    for name in NAMES[offset:]+NAMES[:offset]:
                        before=time.perf_counter();stores[name].add(payload);elapsed=time.perf_counter()-before
                        times[name]+=elapsed;local_times[name]+=elapsed
                    batch_index+=1;message_count+=len(batch)
            print('ADDED',sid,message_count,'messages;',len(sample['qa']),'questions',flush=True)
            sample_cases=[]
            for qi,q in enumerate(sample['qa']):
                gold=set(canonical_evidence(q.get('evidence')));missing=sorted(gold-set(mapping.values()))
                for k in KS:
                    request=AMLSearch(user_id=user,query=q['question'],top_k=k)
                    c=dict(sample_id=sid,question_index=qi,question=q['question'],category=q.get('category'),
                           multimodal=bool(q.get('is_multi_modality',False)),evidence=sorted(gold),
                           missing_evidence=missing,top_k=k)
                    offset=(question_index+KS.index(k))%3
                    for name in NAMES[offset:]+NAMES[:offset]:
                        before=time.perf_counter();hits=stores[name].search(request)['data'];elapsed=time.perf_counter()-before
                        # A foreign-user hit causes KeyError and fails the run.
                        found={mapping[h['id']] for h in hits};matched=sorted(gold & found)
                        c[name]=dict(recall=len(matched)/len(gold) if gold else None,hit_evidence=matched,
                            ids=[h['id'] for h in hits],count=len(hits),seconds=elapsed)
                    cases.append(c);sample_cases.append(c)
                    stream.write(json.dumps(c,ensure_ascii=False)+'\n');stream.flush()
                question_index+=1
                if (qi+1)%25==0 or qi+1==len(sample['qa']):
                    print('SEARCHED',sid,qi+1,'/',len(sample['qa']),'total',question_index,flush=True)
            sr=dict(sample_id=sid,messages=message_count,chunks=len(mapping),add_seconds=local_times,metrics=summarize(sample_cases))
            sample_reports.append(sr)
            (out/(sid+'.json')).write_text(json.dumps(sr,indent=2),encoding='utf-8')
    report=dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),questions=question_index,
        conversations=len(samples),messages=sum(s['messages'] for s in sample_reports),
        chunks=sum(s['chunks'] for s in sample_reports),add_seconds=times,metrics=summarize(cases),samples=sample_reports,
        model='BAAI/bge-small-en-v1.5',tag_weight=0.5,tag_threshold=0.5,llm_calls=0,official_evaluation=False,
        commit=subprocess.check_output(['git','-c','safe.directory='+str(PROJECT_ROOT),'rev-parse','HEAD'],cwd=PROJECT_ROOT,text=True).strip(),
        notes='All questions searched, text input only. Macro over questions, micro over per-question evidence IDs. Empty or missing gold excluded from evidence scores. No answer generation. Frozen settings; no tuning.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('COMPLETE',out/'report.json',flush=True)


if __name__=='__main__':main()
