"""Rule-only person-specific queries and bounded rank fusion."""
import re
import numpy as np


def subqueries(query, rows):
    names=sorted({dict(r).get('speaker') for r in rows if dict(r).get('speaker')}, key=lambda n:(-len(n),n))
    mentioned=[n for n in names if re.search(r'(?<!\w)'+re.escape(n)+r'(?!\w)', query, re.I)]
    if len(mentioned)!=2 or not re.search(r'\b(both|common|share|shared|and)\b',query,re.I):
        return []
    a,b=mentioned
    pair=r'(?:both\s+)?(?:'+re.escape(a)+r'\s+(?:and|&)\s+'+re.escape(b)+'|'+re.escape(b)+r'\s+(?:and|&)\s+'+re.escape(a)+r')'
    if not re.search(pair,query,re.I):return []
    result=[]
    for name in sorted(mentioned):
        q=re.sub(pair,lambda _:name,query,flags=re.I)
        q=re.sub(r'\bboth\s*','',q,flags=re.I)
        q=re.sub(r'\bin common\b','',q,flags=re.I)
        if re.search(r'\bwhat do .+ have\s*\?',q,re.I):
            q=f'What interests, activities and experiences does {name} have?'
        result.append((name,re.sub(r'\s+',' ',q).strip()))
    return result


def candidate_routes(query, rows, embedder, matrix, lexical_score, rrf=60, weight=0.5, limit=60):
    queries=subqueries(query,rows)
    if not queries:return []
    vectors=np.asarray(embedder.queries([q for _,q in queries]),dtype=float)
    if vectors.shape != (len(queries),matrix.shape[1]) or not np.isfinite(vectors).all():
        raise ValueError('Invalid subquery embedding')
    norms=np.linalg.norm(vectors,axis=1,keepdims=True)
    if np.any(norms==0):raise ValueError('Zero subquery vector')
    vectors=vectors/norms
    routes=[]
    for (name,q),vector in zip(queries,vectors):
        dense=matrix@vector
        lexical=lexical_score([r['content'] for r in rows],q)
        votes=np.zeros(len(rows))
        for rank,i in enumerate(sorted(range(len(rows)),key=lambda i:(-dense[i],i)),1):votes[i]+=1/(rrf+rank)
        for rank,i in enumerate(sorted((i for i in range(len(rows)) if lexical[i]>0),key=lambda i:(-lexical[i],i)),1):votes[i]+=weight/(rrf+rank)
        order=sorted(range(len(rows)),key=lambda i:(-votes[i],-dense[i],i))[:limit]
        routes.append((q,order))
    return routes


def merge_routes(order, routes):
    # The original query always retains a vote. Do not compare uncalibrated
    # cross-encoder logits from different queries as if they shared a scale.
    votes={i:1/(20+rank) for rank,i in enumerate(order,1)}
    prior={i:rank for rank,i in enumerate(order)}
    for indices in routes:
        for rank,i in enumerate(indices,1):votes[i]=votes.get(i,0)+0.75/(20+rank)
    result=sorted(votes,key=lambda i:(-votes[i],prior.get(i,len(order)),i))
    return result,votes
