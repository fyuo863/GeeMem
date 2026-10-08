"""Replay fixed original candidates, changing only reranker and context."""
from contextlib import closing
from datetime import datetime,timezone
from pathlib import Path
import json
import time
from memory.config import load_settings
from memory.rerank import ONNXReranker,LocalReranker
from memory.vanilla import VanillaMemory


def main():
    cfg=load_settings()
    out=Path('data/bge-onnx')/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out.mkdir(parents=True)
    bge_cfg=dict(cfg,RAG_RERANK_PATH='D:/Program Data/Github Program/.ts/ts-learn/.fyuobot/models/transformers/Xenova/bge-reranker-base',
                 RAG_RERANK_MANIFEST='data/models/bge-reranker-base-onnx-manifest.json',RAG_RERANK_BATCH_SIZE='8',RAG_RERANK_THREADS='4')
    models={'minilm':LocalReranker(cfg),'bge':ONNXReranker(bge_cfg)}
    results=[]
    for lang,folder in [('zh','20261008T114331Z'),('en','20261008T120213Z')]:
        source=Path('data/partition-types')/folder
        cases=json.loads((source/'results.json').read_text(encoding='utf8'))
        b=VanillaMemory(dict(cfg,RAG_MEMORY_DB=str((source/'memory.sqlite3').resolve()),RAG_PARTITION_MODE='off'),reranker=models['minilm'])
        with closing(b.connect()) as db:
            rows=db.execute('''SELECT m.*,s.role,s.speaker,s.session_timestamp,s.source_id
                FROM rag_memories m LEFT JOIN rag_sources s ON m.id=s.memory_id
                WHERE m.user_id='types-test' ORDER BY m.rowid''').fetchall()
        positions={r['id']:i for i,r in enumerate(rows)}
        # Warm up each model outside reported ranking time.
        for model in models.values():model.score(cases[0]['query'],[rows[0]['content']])
        for case in cases:
            candidates=[positions[mid] for mid in case['dual']['trace']['candidate_ids']]
            entry=dict(language=lang,category=case['category'],query=case['query'],gold=case['gold'],candidate_ids=case['dual']['trace']['candidate_ids'],modes={})
            for mode,model,context in [('minilm_context','minilm',1),('bge_context','bge',1),('bge_target','bge',0)]:
                b.reranker=models[model];b.rerank_context=context
                start=time.perf_counter();scores=b.score_candidates(case['query'],rows,candidates);elapsed=time.perf_counter()-start
                order=sorted(range(len(candidates)),key=lambda j:-scores[j])
                hits=[dict(id=rows[candidates[j]]['id'],content=rows[candidates[j]]['content'],score=float(scores[j])) for j in order]
                gold={g['id'] for g in case['gold']};found={h['id'] for h in hits[:10]}
                entry['modes'][mode]=dict(seconds=elapsed,hits=hits,hit10=int(bool(gold&found)),recall10=len(gold&found)/len(gold),
                    evidence_found=len(gold&found),evidence_total=len(gold),all10=int(gold<=found),hit1=int(hits[0]['id'] in gold))
            results.append(entry)
            (out/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
            print(lang,len(results),'/',40,flush=True)
    summary={}
    for lang in ['zh','en']:
        group=[r for r in results if r['language']==lang]
        summary[lang]={}
        for mode in models_for_report():
            summary[lang][mode]={k:sum(r['modes'][mode][k] for r in group)/len(group) for k in ['hit10','recall10','all10','hit1','seconds']}
            summary[lang][mode]['micro_recall10']=sum(r['modes'][mode]['evidence_found'] for r in group)/sum(r['modes'][mode]['evidence_total'] for r in group)
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(out,json.dumps(summary),flush=True)


def models_for_report():return ['minilm_context','bge_context','bge_target']


if __name__=='__main__':main()
