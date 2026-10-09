import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import AMLAdd, AMLSearch, create_app
from memory.query_time import resolve_query_time
from memory.rerank import HTTPReranker
from memory.vanilla import VanillaMemory
from memory.multihop import Route, Review, Support
from test_vanilla import Embedder


def test_old_temporal_index_refreshes_rules_without_llm(tmp_path):
    import sqlite3
    from memory.temporal_index import TemporalIndex
    def connect():
        db=sqlite3.connect(tmp_path/'old.db');db.row_factory=sqlite3.Row;return db
    index=TemporalIndex(connect)
    with connect() as db:
        db.execute('INSERT INTO rag_time_mentions VALUES (?,?,?,?,?)',('u','s','time-mentions-v1-rules','[]','rules'))
    text='On 31 July, 2023 I travelled.'
    hit=index.enrich('u',[dict(source_id='s',content=text,char_start=10,char_end=10+len(text))])[0]
    assert hit['time_mentions']
    m=hit['time_mentions'][0]
    assert text[m['char_start']-10:m['char_end']-10]=='31 July, 2023'


@pytest.mark.parametrize('date,start,end',[
    ('31 July, 2023','2023-07-31','2023-07-31'),
    ('July 31st, 2023','2023-07-31','2023-07-31'),
    ('31 Jul 2023','2023-07-31','2023-07-31'),
    ('November 2022','2022-11-01','2022-11-30'),
    ('Feb 2024','2024-02-01','2024-02-29')])
def test_english_dates_without_invented_clock(date,start,end):
    target=resolve_query_time('What happened on '+date+'?')
    assert target and target.start.isoformat()==start and target.end.isoformat()==end


@pytest.mark.parametrize('q',['on 31 April 2023','on 29 February 2023','on July 31',
    'between July 31, 2023 and August 2, 2023','before November 2022'])
def test_ambiguous_or_invalid_time_is_not_window(q):
    assert resolve_query_time(q) is None


def test_dashscope_one_pool_auth_indices_and_no_environment(monkeypatch):
    observed=[]
    def reply(request):
        import json
        payload=json.loads(request.content);observed.append(payload)
        assert request.headers['authorization']=='Bearer test-key'
        assert payload['parameters']['top_n']==2
        assert payload['input']['documents']==['a','b']
        return httpx.Response(200,json={'output':{'results':[{'index':1,'relevance_score':.2},{'index':0,'relevance_score':.9}]}})
    ranker=HTTPReranker(dict(RAG_RERANK_API_URL='https://example.test/rerank',
        RAG_RERANK_API_PROTOCOL='dashscope',RAG_RERANK_API_MODEL='qwen3.7-text-rerank',RAG_RERANK_API_KEY='test-key'))
    ranker.client.close()
    ranker.client=httpx.Client(transport=httpx.MockTransport(reply),headers={'Authorization':'Bearer test-key'},trust_env=False)
    assert ranker.score('q',['a','b']).tolist()==[.9,.2]
    with pytest.raises(ValueError):ranker.score('q',['x']*501)
    assert len(observed)==1


@pytest.mark.parametrize('rows',[
    [{'index':0,'relevance_score':.5}]*2,
    [{'index':1,'relevance_score':.5}],
    [{'index':0,'relevance_score':float('nan')},{'index':1,'relevance_score':.2}]])
def test_bad_api_scores_fail_closed(rows):
    r=HTTPReranker({'RAG_RERANK_API_URL':'https://example.test/rerank'})
    r.client.close();r.client=httpx.Client(transport=httpx.MockTransport(lambda req:httpx.Response(200,json={'results':rows})) )
    with pytest.raises(ValueError):r.score('q',['a','b'])


def test_identity_context_reaches_review_and_final_rerank_but_not_public_response(tmp_path):
    reviews=[];documents=[]
    class Planner:
        def route(self,*a):return Route(strategy='direct')
        def review(self,q,options,strategy,evidence,history,timeout):
            reviews.extend(evidence)
            target=next(e for e in evidence if 'Piano' in e['content'])
            return Review(sufficient=True,supports=[Support(source_id=target['id'],quote=target['content'],needed_for='instrument')],missing='')
    class Ranker:
        def score(self,q,docs):documents.append(list(docs));return np.ones(len(docs))
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_MULTIHOP_MODE='llm',RAG_RERANK_CONTEXT='1',
        RAG_METADATA_MODE='on',RAG_RESULT_WINDOW='0',AML_AUTH_MODE='bearer',AML_API_KEY='test')
    store=VanillaMemory(cfg,Embedder(),reranker=Ranker(),multihop_planner=Planner())
    texts=['[speaker: Alice] Which instrument do you play?','[speaker: Bob] Piano.']
    with TestClient(create_app(settings=cfg,backend=store)) as c:
        c.headers['Authorization']='Bearer test'
        assert c.post('/add',json=dict(user_id='u',session_id='s',request_id='r',messages=[dict(role='user',content=t) for t in texts])).status_code==200
        store.add(AMLAdd(user_id='secret',session_id='s',request_id='r',messages=[dict(role='user',content='PRIVATE_NEIGHBOR')]))
        response=c.post('/search',json=dict(user_id='u',query='What does Bob play?',top_k=2)).json()
    assert all(h['content'] in texts and set(h)<= {'id','content','score','created_at'} for h in response['data'])
    target=next(e for e in reviews if e['content']==texts[1])
    assert 'Bob said' in target['context'] and 'Which instrument' in target['context']
    assert len(target['content'])+len(target['context'])<=store.multihop.chars
    assert any('Bob said' in d and 'Which instrument' in d for d in documents[-1])
    assert 'PRIVATE_NEIGHBOR' not in str(documents)+str(reviews)
