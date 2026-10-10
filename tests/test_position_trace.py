from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from memory.aml_api import AMLAdd, AMLSearch
from memory.vanilla import VanillaMemory
from memory.search_service import SearchService
from memory.position_trace import record_returned


def test_positions_follow_real_rerank_cutoff_and_public_result(tmp_path):
    class Embedder:
        identity='position-test'
        def documents(self,texts): return np.ones((len(texts),3))
        queries=documents
    class Ranker:
        def score(self,q,docs): return [float({'alpha':1,'beta':2,'gamma':3}[d]) for d in docs]
    store=VanillaMemory({'RAG_MEMORY_DB':str(tmp_path/'test.db'),'RAG_RESULT_WINDOW':'0'},Embedder(),reranker=Ranker())
    for i,text in enumerate(['alpha','beta','gamma']):
        store.add(AMLAdd(user_id='u',session_id='s',request_id=str(i),messages=[{'role':'user','content':text}]))
    trace={}
    response=store.search(AMLSearch(user_id='u',query='unrelated',top_k=1),trace=trace)
    assert response['data'][0]['content']=='gamma'
    sid=response['data'][0]['id']
    stages=trace['retrieval_queries'][0]['positions']
    assert stages['candidate'][sid]==3
    assert stages['rerank'][sid]==1 and stages['selected']=={sid:1}
    assert stages['returned']=={sid:1}
    assert trace['positions']['final_results']=={sid:1}
    assert trace['final_source_positions'][sid][0]['result_rank']==1
    assert store.search(AMLSearch(user_id='u',query='unrelated',top_k=1))==response


def test_parallel_retrieval_position_traces_are_separate():
    class Planner:
        mode='llm'
        def run(self,p,retrieve,rerank,trace):
            with ThreadPoolExecutor(max_workers=2) as pool:
                results=list(pool.map(retrieve,[SimpleNamespace(query=q,user_id='u',top_k=1) for q in ['a','b']]))
            return {'data':[h for r in results for h in r['data']]}
    def direct(p):
        p.retrieval_trace['positions']={'candidate':{p.query:1}}
        return {'data':[dict(id=p.query,content=p.query,score=1)]}
    trace={}
    SearchService(direct,Planner()).search(SimpleNamespace(query='q',user_id='u',top_k=2),trace)
    calls=trace['retrieval_queries']
    assert len({c['call_id'] for c in calls})==2
    for c in calls:
        assert c['positions']['candidate']==c['positions']['returned']=={c['query']:1}


def test_final_bundle_positions_reflect_post_filter_response():
    trace={'bundle_sources':{'bundle_a':['a','b'],'hidden':['c']}}
    record_returned(trace,[{'id':'bundle_a'},{'id':'x'}])
    assert trace['final_source_positions']['b']==[{'result_rank':1,'member_position':2,'result_id':'bundle_a'}]
    assert 'c' not in trace['final_source_positions']
    assert trace['final_source_positions']['x'][0]['result_rank']==2

def test_position_log_contains_positions_not_query_or_source_text(tmp_path):
    import json
    from memory.position_trace import PositionLog
    path=tmp_path/'positions.jsonl'
    sink=PositionLog({'RAG_POSITION_LOG_PATH':str(path)})
    trace={'retrieval_queries':[{'call_id':1,'query':'PRIVATE QUERY','positions':{'candidate':{'a':1}}}],
           'rounds':[{'round':0,'review':{'quote':'PRIVATE CONTENT'},'positions':{'review_input':{'a':1}}}]}
    sink.write(trace)
    sink.handler.close()
    text=path.read_text(encoding='utf8');event=__import__('json').loads(text)
    assert 'PRIVATE' not in text
    assert event['queries'][0]['positions']['candidate']=={'a':1}
    assert event['trace_id']==trace['position_trace_id']
