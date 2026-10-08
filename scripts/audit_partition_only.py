"""Replay recorded routing choices with strict retrieval; no second routing call."""
from pathlib import Path
import json
import sys
from memory.config import load_settings
from memory.vanilla import VanillaMemory
from memory.atomic_retriever import AtomicQuery


def main():
    out=Path(sys.argv[1])
    rows=json.loads((out/'results.json').read_text(encoding='utf8'))
    b=VanillaMemory(dict(load_settings(),RAG_MEMORY_DB=str((out/'memory.sqlite3').resolve())))
    for row in rows:
        types=tuple(row['dual']['trace'].get('partitions',[]))
        hits=b.atomic_retriever.query(AtomicQuery('types-test',row['query'],top_k=10,
            memory_types=types,fallback=False))['data'] if types else []
        gold={g['id'] for g in row['gold']};found={h['id'] for h in hits}
        row['partition_only']=dict(partitions=types,hits=hits,hit10=int(bool(gold&found)),
            recall10=len(gold&found)/len(gold),all10=int(gold<=found))
    from types import SimpleNamespace
    reranker=b.reranker
    print('Reranker:', type(reranker).__name__, 'path:', load_settings().get('RAG_RERANK_PATH','data/models/ms-marco-MiniLM-L-6-v2'),flush=True)
    for mode in ['no_reranker', 'no_rerank_context']:
        b.reranker=None if mode=='no_reranker' else reranker
        if mode=='no_rerank_context': b.rerank_context=0
        for row in rows:
            types=tuple(row['dual']['trace'].get('partitions',[]))
            payload=SimpleNamespace(user_id='types-test',query=row['query'],top_k=10,options=None,
                memory_types=types,fallback=True,dual_channel=bool(types))
            hits=b.atomic_retriever.retrieve(payload)['data']
            gold={g['id'] for g in row['gold']};found={h['id'] for h in hits}
            row[mode]=dict(hits=hits,hit10=int(bool(gold&found)),recall10=len(gold&found)/len(gold),all10=int(gold<=found))
    (out/'audited.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    summary={}
    for category in ['profile','relationship','rule','event','cross','all']:
        group=[r for r in rows if category=='all' or r['category']==category]
        summary[category]={'questions':len(group),'gold':sum(len(r['gold']) for r in group)}
        for mode in ['off','dual','partition_only','no_reranker','no_rerank_context']:
            summary[category][mode]={k:sum(r[mode][k] for r in group)/len(group)
                                    for k in ['hit10','recall10','all10']}
        summary[category]['covered']=sum(r['dual']['gold_in_selected_partition'] for r in group)
        summary[category]['reserved']=sum(r['dual']['gold_in_partition_channel'] for r in group)
    (out/'audit-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    for row in rows:
        if row['partition_only']['all10']!=1 or row['dual']['all10']!=1:
            print(json.dumps(dict(query=row['query'],partitions=row['partition_only']['partitions'],
                gold=[dict(content=g['content'],types=g['types']) for g in row['gold']],
                strict_recall=row['partition_only']['recall10'],dual_recall=row['dual']['recall10']),ensure_ascii=False))


if __name__=='__main__':main()
