"""Bounded candidate paths from explicit source triples, never inferred facts."""
import json
import re


def _position(text, phrase):
    match = re.search(r'(?<!\w)' + re.escape(phrase) + r'(?!\w)', text, re.I)
    return match.start() if match else None


def candidate_paths(question, evidence):
    """Use only literal question anchors and explicit directed JSON relationships.

    The first literally named relation is a candidate outer/terminal relation.
    No aliases, inverse relations, natural-language fact extraction or truth
    adjudication. The caller must label the output as candidate source paths.
    Bounds: 100 sources, 6 edges/path, 512 expansions, 8 paths.
    """
    edges=[]
    for index, source in enumerate(evidence[:100]):
        text=source['content']; start=text.find('{')
        if start<0: continue
        # Accept a JSON source, optionally with the corpus transport prefix.
        # Arbitrary prose containing a quoted JSON example is not a fact record.
        prefix=text[:start].strip()
        if prefix and prefix not in ('Corpus:', 'UPDATE:',
            'Corpus: UPDATE: replace the prior value of this subject and relation with the following fact.'):
            continue
        try: fact=json.loads(text[start:])
        except (ValueError, TypeError): continue
        if not isinstance(fact,dict) or not all(isinstance(fact.get(k),str) and fact[k].strip()
                for k in ('subject','relation','object')): continue
        # Do not discard qualifiers such as negation, uncertainty or validity.
        if set(fact)-{'subject','relation','object','subject_id','relation_id','object_id'}: continue
        if any(k in fact and (not isinstance(fact[k],str) or not fact[k])
               for k in ('subject_id','object_id')): continue
        edges.append(dict(fact,index=index))
    terminal=[(_position(question,e['relation']),e['relation'].casefold()) for e in edges]
    terminal=[(p,r) for p,r in terminal if p is not None]
    if not terminal: return []
    first=min(p for p,_ in terminal)
    endings={r for p,r in terminal if p==first}
    if len(endings)!=1: return []
    ending=next(iter(endings))
    # A name with conflicting explicit IDs is never a name-only join.
    identities={}
    for e in edges:
        for side in ('subject','object'):
            if e.get(side+'_id'):
                identities.setdefault(e[side].casefold(),set()).add(e[side+'_id'])
    def joins(a,b):
        if a.get('object_id') and b.get('subject_id'):
            return a['object_id']==b['subject_id']
        name=a['object'].casefold()
        return name==b['subject'].casefold() and len(identities.get(name,()))<=1
    paths=[]; expansions=0
    def walk(path):
        nonlocal expansions
        if expansions>=512 or len(paths)>=8: return
        expansions+=1
        tail=path[-1]
        if tail['relation'].casefold()==ending:
            if len(path)>=2: paths.append([e['index'] for e in path])
            return
        if len(path)>=6:return
        for edge in edges:
            if all(edge['index']!=e['index'] for e in path) and joins(tail,edge):
                walk(path+[edge])
    for edge in edges:
        if _position(question,edge['subject']) is not None:
            walk([edge])
    return paths
