"""Paired public-data tag ablation; never reads official evaluation payloads."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT,load_settings
from memory.vanilla import VanillaMemory,LocalEmbedder,chunks
from memory.tags import RuleTagger
from memory.aml_api import AMLAdd,AMLSearch


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--questions',type=int,default=30)
    args=parser.parse_args()
    if args.questions < 1:
        parser.error('--questions must be positive')
    source=PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json'
    sample=json.loads(source.read_text(encoding='utf-8'))[0]
    conversation=sample['conversation']
    sessions=sorted([k for k,v in conversation.items() if k.startswith('session_') and isinstance(v,list)],key=lambda k:int(k.split('_')[1]))
    out=PROJECT_ROOT/'data/tag-benchmarks'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    cfg=load_settings();model=LocalEmbedder(cfg);tagger=RuleTagger()
    common=dict(cfg,RAG_RESULT_WINDOW='1',RAG_RESULT_WINDOW_SEED_K='20',RAG_TAG_CANDIDATES='400')
    baseline=VanillaMemory(dict(common,RAG_MEMORY_DB=str(out/'baseline.sqlite3'),RAG_TAG_MODE='off'),model)
    tagged=VanillaMemory(dict(common,RAG_MEMORY_DB=str(out/'tagged.sqlite3'),RAG_TAG_MODE='filter'),model,tagger)
    timings={'baseline':0.,'tagged':0.};mapping={};message_count=0
    for session in sessions:
        messages=conversation[session]
        for start in range(0,len(messages),40):
            batch=messages[start:start+40];rid=session+':'+str(start)
            payload=AMLAdd(user_id='public-first-conversation',session_id=session,request_id=rid,
                messages=[dict(role='user' if m['speaker']==conversation['speaker_a'] else 'assistant',content=m['text']) for m in batch])
            for i,m in enumerate(batch):
                for j,_ in enumerate(chunks(m['text'],baseline.size,baseline.overlap)):
                    mid=hashlib.sha256(json.dumps([payload.user_id,rid,i,j]).encode()).hexdigest()
                    mapping[mid]=m['dia_id']
            write_order=[('baseline',baseline),('tagged',tagged)]
            if message_count%2:write_order.reverse()
            for name,store in write_order:
                before=time.perf_counter();store.add(payload);timings[name]+=time.perf_counter()-before
            message_count+=len(batch)
            print('Added',session,start,'messages',message_count,flush=True)
    eligible=[q for q in sample['qa'] if not q.get('is_multi_modality',False) and q.get('evidence') and set(q['evidence']) <= set(mapping.values())]
    indices=np.linspace(0,len(eligible)-1,min(args.questions,len(eligible)),dtype=int).tolist()
    cases=[]
    # Query tags are extracted once, charged to each tag-search measurement.
    for number,index in enumerate(indices):
        q=eligible[index]
        before=time.perf_counter();query_tags=tagger.extract([q['question']],query=True);tag_seconds=time.perf_counter()-before
        class Cached:
            identity=tagger.identity
            def extract(self,texts,query=False):return query_tags
        tagged.tagger=Cached()
        for k in [10,100]:
            request=AMLSearch(user_id='public-first-conversation',query=q['question'],top_k=k)
            result={}
            # Alternate execution order to reduce warm-cache bias.
            stores=[('baseline',baseline),('tagged',tagged)]
            if number%2:stores.reverse()
            for name,store in stores:
                before=time.perf_counter();hits=store.search(request)['data'];elapsed=time.perf_counter()-before
                found={mapping[h['id']] for h in hits};gold=set(q['evidence'])
                result[name]=dict(recall=len(gold & found)/len(gold),seconds=elapsed+(tag_seconds if name=='tagged' else 0),
                    retrieval_seconds=elapsed,count=len(hits),hit_evidence=sorted(gold & found),ids=[h['id'] for h in hits])
            cases.append(dict(question=q['question'],evidence=q['evidence'],query_tags=query_tags[0],top_k=k,**result))
        print('Searched',number+1,'/',len(indices),flush=True)
        (out/'cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    summary={}
    for k in [10,100]:
        subset=[c for c in cases if c['top_k']==k]
        summary[str(k)]={name:dict(recall=float(np.mean([c[name]['recall'] for c in subset])),
            p50_seconds=float(np.median([c[name]['seconds'] for c in subset])),
            p95_seconds=float(np.percentile([c[name]['seconds'] for c in subset],95)),
            mean_results=float(np.mean([c[name]['count'] for c in subset]))) for name in ['baseline','tagged']}
        summary[str(k)].update(wins=sum(c['tagged']['recall']>c['baseline']['recall'] for c in subset),
            losses=sum(c['tagged']['recall']<c['baseline']['recall'] for c in subset))
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),sample_id=sample['sample_id'],
        selected_question_indices=indices,eligible_questions=len(eligible),messages=message_count,chunks=len(mapping),
        add_seconds=timings,metrics=summary,official_evaluation=False,tagger=tagger.identity,model='BAAI/bge-small-en-v1.5',llm_calls=0,
        note='Single full public conversation; deterministic evenly spaced questions. Add excludes model loading. Tag extraction counted in Search. No answer generation.')
    (out/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2));print('REPORT',out/'report.json',flush=True)


if __name__=='__main__':main()
