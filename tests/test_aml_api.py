from concurrent.futures import ThreadPoolExecutor
from threading import Event
import pytest
from fastapi.testclient import TestClient
from memory.aml_api import create_app
from memory.models import Graph, Node
from memory.store import Store
from memory.llm import LLMError


class Fake:
    def __init__(self):self.calls=0;self.queries=[]
    def extract(self,request):
        self.calls+=1
        return Graph(nodes=[Node(key='memory:'+str(i),name='Hangzhou',kind='place',message_indices=[i])
                            for i,_ in enumerate(request.messages)],edges=[])
    def keywords(self,query):self.queries.append(query);return ['Hangzhou']


def payload(user='u',session='s',rid='r'):
    return dict(user_id=user,session_id=session,request_id=rid,
                messages=[dict(role='user',content='I live in Hangzhou.')])


def setup(tmp_path,**cfg):
    store=Store(tmp_path/'db');llm=Fake()
    app=create_app(store,llm,settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test-secret',**cfg))
    return app,store,llm


HEAD={'Authorization':'Bearer test-secret'}


def test_long_official_textual_query_reaches_backend_unchanged(tmp_path):
    class Backend:
        def search(self,payload):
            self.query=payload.query
            return {'data':[]}
    backend=Backend()
    app=create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test-secret'),backend=backend)
    query=('Background context about past events. '*400)+'Which event happened first?'
    with TestClient(app) as client:
        response=client.post('/search',headers=HEAD,json=dict(user_id='long-query-test',query=query,top_k=100))
    assert response.status_code==200
    assert backend.query==query


def test_validation_logs_field_codes_without_sensitive_input(tmp_path,caplog):
    app,_,_=setup(tmp_path)
    with TestClient(app) as c:
        r=c.post('/search',headers=HEAD,json=dict(user_id='private-user',query='private-question',
            top_k='bad-private-value',**{'private-extra-name':'private-extra-value'}))
    assert r.status_code==422
    assert 'api_validation_failed endpoint=/search' in caplog.text
    assert 'top_k' in caplog.text and 'int_type' in caplog.text and 'extra_forbidden' in caplog.text
    assert 'private-' not in caplog.text and 'test-secret' not in caplog.text


def test_official_sync_contract_retry_and_optional_timestamp(tmp_path):
    app,store,llm=setup(tmp_path)
    with TestClient(app) as c:
        p=payload();r=c.post('/add',json=p,headers=HEAD)
        assert r.status_code==200 and r.json()==dict(success=True,user_id='u',session_id='s',request_id='r')
        s=c.post('/search',headers=HEAD,json=dict(user_id='u',query='Where?',top_k=100)).json()
        assert s['data'][0]['content']==p['messages'][0]['content']
        assert 'created_at' not in s['data'][0]
        assert c.post('/add',json=p,headers=HEAD).json()==r.json() and llm.calls==1
        again=c.post('/search',headers=HEAD,json=dict(user_id='u',query='Where?',top_k=1)).json()
        assert again['data'][0]['id']==s['data'][0]['id']
        p['messages'][0]['content']='Changed'
        assert c.post('/add',json=p,headers=HEAD).status_code==409


def test_multiple_sessions_options_order_and_user_isolation(tmp_path):
    app,store,llm=setup(tmp_path)
    with TestClient(app) as c:
        for user,session,rid,stamp in [('u','s1','r1',1000),('u','s2','r2',2000),('other','s1','r1',3000)]:
            p=payload(user,session,rid);p['messages'][0]['timestamp']=stamp
            assert c.post('/add',json=p,headers=HEAD).status_code==200
        q=dict(user_id='u',query='Which city?',options=['Hangzhou','Tokyo'],top_k=100)
        hits=c.post('/search',json=q,headers=HEAD).json()['data']
        assert len(hits)==2 and hits[0]['created_at']=='1970-01-01T00:00:02Z'
        assert 'Tokyo' in llm.queries[-1]
        q['top_k']=1
        assert len(c.post('/search',json=q,headers=HEAD).json()['data'])==1
        q['user_id']='missing'
        assert c.post('/search',json=q,headers=HEAD).json()=={'data':[]}


@pytest.mark.parametrize('mode,headers',[
    ('bearer',{'Authorization':'Bearer test-secret'}),('token',{'Authorization':'Token test-secret'}),
    ('x-api-key',{'X-Api-Key':'test-secret'})])
def test_authentication_and_public_health(tmp_path,mode,headers):
    app=create_app(Store(tmp_path/'db'),Fake(),dict(AML_AUTH_MODE=mode,AML_API_KEY='test-secret'))
    with TestClient(app) as c:
        assert c.get('/health').status_code==200
        assert c.post('/add',json=payload()).status_code==401
        assert c.post('/add',json=payload(),headers={'Authorization':'Bearer wrong'}).status_code==401
        assert c.post('/add',json=payload(),headers=headers).status_code==200


@pytest.mark.parametrize('cfg',[{},dict(AML_AUTH_MODE='none'),dict(AML_AUTH_MODE='unknown')])
def test_missing_auth_configuration_fails_startup(tmp_path,cfg):
    with pytest.raises(ValueError):
        with TestClient(create_app(Store(tmp_path/'db'),Fake(),cfg)):pass


@pytest.mark.parametrize('change',[
    {'top_k':101},{'top_k':0},{'top_k':True},{'top_k':'100'},
    {'session_id':'s'},{'query':['multimodal']},{'options':['']},{'user_id':' u'}])
def test_unsupported_payloads_are_explicit_422(tmp_path,change):
    app,_,_=setup(tmp_path)
    with TestClient(app) as c:
        q=dict(user_id='u',query='private-query',top_k=100);q.update(change)
        r=c.post('/search',json=q,headers=HEAD)
        assert r.status_code==422
        assert 'private-query' not in r.text


def test_invalid_timestamp_no_truncation_and_failure_atomicity(tmp_path):
    app,store,llm=setup(tmp_path)
    with TestClient(app) as c:
        p=payload();p['messages'][0]['timestamp']='1700000000000'
        assert c.post('/add',json=p,headers=HEAD).status_code==422
        p=payload();p['messages'][0]['content']='x'*32001
        assert c.post('/add',json=p,headers=HEAD).status_code==422
        def fail(_):raise LLMError('upstream')
        llm.extract=fail
        assert c.post('/add',json=payload(),headers=HEAD).status_code==502
        with store.connect() as db:assert db.execute('select count(*) from messages').fetchone()[0]==0


def test_capacity_returns_retry_after_and_releases_slot(tmp_path):
    app,store,llm=setup(tmp_path,AML_ADD_CONCURRENCY='1')
    entered=Event();release=Event();original=llm.extract
    def block(r):entered.set();assert release.wait(5);return original(r)
    llm.extract=block
    with TestClient(app) as c, ThreadPoolExecutor() as pool:
        first=pool.submit(c.post,'/add',json=payload(),headers=HEAD)
        try:
            assert entered.wait(3)
            second=c.post('/add',json=payload(rid='other'),headers=HEAD)
            assert second.status_code==429 and second.headers['Retry-After']=='5'
        finally:release.set()
        assert first.result().status_code==200
        assert c.post('/add',json=payload(rid='third'),headers=HEAD).status_code==200


def test_model_rule(tmp_path,monkeypatch):
    from memory import config
    from memory.aml_api import EvaluationLLM
    monkeypatch.setattr(config,'PROJECT_ROOT',tmp_path)
    (tmp_path/'.env').write_text('LLM_MODEL=another-model',encoding='utf-8')
    with pytest.raises(ValueError,match='gpt-4o-mini'):EvaluationLLM()


@pytest.mark.parametrize('endpoint',['/add','/search'])
def test_unlimited_admission_accepts_overlapping_requests(endpoint):
    from threading import Barrier
    barrier=Barrier(2)
    class Backend:
        def add(self,p):barrier.wait(timeout=5)
        def search(self,p):barrier.wait(timeout=5);return {'data':[]}
    app=create_app(settings=dict(AML_AUTH_MODE='bearer',AML_API_KEY='test-secret',
        AML_ADD_CONCURRENCY='0',AML_SEARCH_CONCURRENCY='0'),backend=Backend())
    body=payload() if endpoint=='/add' else dict(user_id='u',query='q',top_k=100)
    with TestClient(app) as client,ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(client.post,endpoint,json=body,headers=HEAD) for _ in range(2)]
        assert all(f.result().status_code==200 for f in futures)


def test_concurrency_has_no_arbitrary_upper_bound(tmp_path):
    app,_,_=setup(tmp_path,AML_ADD_CONCURRENCY='64',AML_SEARCH_CONCURRENCY='128')
    with TestClient(app) as c:assert c.get('/health').status_code==200


def test_negative_concurrency_is_rejected(tmp_path):
    app,_,_=setup(tmp_path,AML_ADD_CONCURRENCY='-1')
    with pytest.raises(ValueError,match='nonnegative'):
        with TestClient(app):pass
