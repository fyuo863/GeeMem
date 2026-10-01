"""Cached-score, gold-independent selection experiments for Hit@10."""
import argparse,hashlib,json,sqlite3,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from memory.config import PROJECT_ROOT
from memory.evaluation import canonical_evidence,evidence_metrics
from memory.vanilla import chunks


def choose(ids,scores,rows,method):
    order=np.argsort(-np.asarray(scores),kind='stable').tolist()
    if method=='context':return [ids[i] for i in order[:10]]
    if method.startswith('propagate_'):
        from memory.rerank import context_support_scores
        penalty=float(method.split('_')[1])
        ordered=sorted(rows,key=lambda mid:rows[mid]['position'])
        positions={mid:i for i,mid in enumerate(ordered)}
        order,_=context_support_scores([rows[mid] for mid in ordered],[positions[mid] for mid in ids],scores,penalty)
        return [ordered[i] for i in order[:10]]
    if method.startswith('blend_'):
        weight=float(method.split('_')[1]);ranks=np.empty(len(ids));ranks[order]=np.arange(1,len(ids)+1)
        fusion=1/(60+ranks)+weight/(60+np.arange(1,len(ids)+1))
        return [ids[i] for i in np.argsort(-fusion,kind='stable')[:10]]
    if method.startswith('mmr_'):
        weight=float(method.split('_')[1]);pool=order[:100]
        vectors=np.stack([np.frombuffer(rows[ids[i]]['vector'],dtype='<f4') for i in pool])
        similarity=vectors@vectors.T
        rel=1/(1+np.exp(-np.clip(np.asarray(scores)[pool],-30,30)))
        selected=[];remaining=list(range(len(pool)))
        for _ in range(min(10,len(pool))):
            best=max(remaining,key=lambda i:weight*rel[i]-(1-weight)*(max(similarity[i,j] for j in selected) if selected else 0))
            selected.append(best);remaining.remove(best)
        return [ids[pool[i]] for i in selected]
    if method.startswith('session_'):
        cap=int(method.split('_')[1]);selected=[];counts={}
        for i in order:
            sid=rows[ids[i]]['session_id']
            if counts.get(sid,0)>=cap:continue
            selected.append(ids[i]);counts[sid]=counts.get(sid,0)+1
            if len(selected)==10:return selected
        return (selected+[ids[i] for i in order if ids[i] not in selected])[:10]
    if method.startswith('reserve_'):
        count=int(method.split('_')[1]);selected=[ids[i] for i in order[:10-count]]
        for mid in ids:
            if mid not in selected:selected.append(mid)
            if len(selected)==10:break
        return selected
    if method.startswith('neighbors_'):
        seeds=int(method.split('_')[1]);selected=[ids[i] for i in order[:seeds]]
        by_position={(row['session_id'],row['position']):mid for mid,row in rows.items()}
        for mid in selected[:]:
            for distance in [-1,1]:
                neighbor=by_position.get((rows[mid]['session_id'],rows[mid]['position']+distance))
                if neighbor and neighbor not in selected:selected.append(neighbor)
                if len(selected)==10:return selected
        return (selected+[ids[i] for i in order if ids[i] not in selected])[:10]
    raise ValueError(method)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--split',choices=['dev','heldout'],default='dev');parser.add_argument('--method',default='all');args=parser.parse_args()
    root=PROJECT_ROOT/'data/rerank-benchmarks'/('20261001T021521Z' if args.split=='dev' else '20261001T021642Z')
    cases=[json.loads(x) for x in (root/'cases.jsonl').read_text(encoding='utf-8').splitlines()]
    source=json.loads((PROJECT_ROOT/'data/locomo-refined/data/raw/locomo_refined.json').read_text(encoding='utf-8'))
    methods=['context','blend_0.25','blend_0.5','blend_1','mmr_0.5','mmr_0.7','mmr_0.9','session_2','session_3','reserve_2','reserve_4','neighbors_4','neighbors_6','neighbors_8','neighbors_9','propagate_0.25','propagate_0.5','propagate_1','propagate_2'] if args.method=='all' else ['context',args.method]
    mapping={};rowsets={}
    with sqlite3.connect(root/'memory.sqlite3') as db:
        db.row_factory=sqlite3.Row
        for sample in source:
            user='public-full:'+sample['sample_id'];rs=db.execute('SELECT * FROM rag_memories WHERE user_id=? ORDER BY rowid',(user,)).fetchall()
            rowsets[sample['sample_id']]={r['id']:dict(r,position=i) for i,r in enumerate(rs)}
            for session,messages in sample['conversation'].items():
                if not session.startswith('session_') or not isinstance(messages,list):continue
                for start in range(0,len(messages),40):
                    for i,m in enumerate(messages[start:start+40]):
                        for j,_ in enumerate(chunks(m['text'])):
                            mid=hashlib.sha256(json.dumps([user,session+':'+str(start),i,j]).encode()).hexdigest()
                            mapping[mid]=canonical_evidence([m['dia_id']])[0]
    output=[]
    for c in cases:
        result=dict(sample_id=c['sample_id'],question_index=c['question_index'],question=c['question'],evidence=c['evidence'])
        for method in methods:
            before=time.perf_counter();ids=choose(c['ceiling_400']['ids'],c['context']['candidate_scores'],rowsets[c['sample_id']],method);elapsed=time.perf_counter()-before
            found=set(mapping[i] for i in ids)&set(c['evidence'])
            result[method]=dict(ids=ids,hit_evidence=sorted(found),recall=len(found)/len(set(c['evidence'])),selection_seconds=elapsed)
        output.append(result)
    report={m:evidence_metrics(output,m) for m in methods}
    for m in methods:
        report[m].update(hit_wins=sum(bool(c[m]['hit_evidence']) and not c['context']['hit_evidence'] for c in output),hit_losses=sum(not c[m]['hit_evidence'] and bool(c['context']['hit_evidence']) for c in output))
    out=PROJECT_ROOT/'data/hit10-experiments';out.mkdir(exist_ok=True)
    (out/(args.split+'-'+args.method+'-cases.json')).write_text(json.dumps(output,indent=2),encoding='utf-8')
    (out/(args.split+'-'+args.method+'-report.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    for m,r in report.items():print(m,'hit',round(r['hit_rate']*100,2),'recall',round(r['recall']*100,2),'wins/losses',r['hit_wins'],r['hit_losses'])


if __name__=='__main__':main()
