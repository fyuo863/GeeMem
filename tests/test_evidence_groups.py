from types import SimpleNamespace
import numpy as np
from memory.evidence_groups import select_packet_ids, expand_groups
from memory.vanilla import VanillaMemory
from memory.aml_api import AMLAdd, AMLSearch
from memory.multihop import MultiHop, Route, Review, Query


def test_packet_reserves_whole_group_and_unseen_positions():
    ids, omitted=select_packet_ids(['old1','old2'],[['topic','question','answer']],
        [['old1','old2','noise','fresh']],{'old1','old2','noise'},6)
    assert ids==['old1','old2','topic','question','answer','fresh'] and not omitted
    ids, omitted=select_packet_ids(['old1','old2'],[['topic','question','answer']],
        [['topic','question','answer','fresh']],set(),4)
    assert 'answer' not in ids and omitted==[['topic','question','answer']]


def test_expansion_keeps_user_session_and_masking(tmp_path):
    class Embedder:
        identity='groups-test'
        def documents(self,texts):return np.ones((len(texts),3))
        queries=documents
    class Ranker:
        def score(self,q,docs):return [1.]*len(docs)
    b=VanillaMemory({'RAG_MEMORY_DB':str(tmp_path/'db'),'RAG_DISCLOSURE_MODE':'mask'},Embedder(),reranker=Ranker())
    for user,session,texts in [('u','s',['Topic pets','password=abc123','Next month']),('u','other',['Other session']),('other','s',['Other user'])]:
        b.add(AMLAdd(user_id=user,session_id=session,request_id=user+session,
            messages=[dict(role='user',content=t) for t in texts]))
    with b.connect() as db: anchor=dict(db.execute("select id,content from rag_memories where user_id='u' and content='Topic pets'").fetchone())
    groups=expand_groups(b,AMLSearch(user_id='u',query='When?',top_k=3),[anchor],'date')
    contents=[h['content'] for g in groups for h in g]
    assert len(groups)==1 and len(contents)==3
    assert not any('abc123' in s or 'Other' in s for s in contents)
    assert any('[REDACTED:' in s for s in contents)


def test_recovery_does_not_recover_its_own_variant():
    calls=[]
    class Planner:
        def route(self,*a):return Route(strategy='chain',queries=[Query(query='Who knows Alice?',source_id='__question__',bridge='Alice')])
        def review(self,*a):return Review(sufficient=False,supports=[],missing='person',queries=[])
    def retrieve(p):
        calls.append(p.query)
        return {'data':[dict(id='a',content='Alice likes tea.',score=1)]}
    trace={}
    MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_MULTIHOP_RECOVERY':'on','RAG_MULTIHOP_ROUNDS':'4'},Planner()).run(
        AMLSearch(user_id='u',query='Where does Alice friend work?',top_k=2),retrieve,lambda q,d:[1]*len(d),trace)
    assert max(q.count('Context question:') for q in calls)==1
    assert trace['recovery_queries']==1
