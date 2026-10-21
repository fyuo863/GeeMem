import json
import httpx
import numpy as np
import pytest
from memory.vanilla import HTTPEmbedder
from memory.embeddings import EmbeddingUnavailable


def config(**kwargs):
    return dict(RAG_EMBEDDING_API_URL='https://embedding.test/v1/embeddings',
        RAG_EMBEDDING_API_MODEL='text-embedding-v4', RAG_EMBEDDING_API_KEY='test-key', **kwargs)


@pytest.mark.parametrize('failure', ['timeout', 'disconnect', 429, 502, 503])
def test_transient_failures_retry_only_failed_batch(monkeypatch, failure):
    calls=[]
    original=httpx.Client
    monkeypatch.setattr('memory.embeddings.time.sleep',lambda _:None)
    def respond(req):
        batch=json.loads(req.content)['input'];calls.append(batch)
        if len(calls)==2:
            if failure=='timeout':raise httpx.ConnectTimeout('private-provider-details')
            if failure=='disconnect':raise httpx.RemoteProtocolError('private-provider-details')
            return httpx.Response(failure,headers={'Retry-After':'2'})
        return httpx.Response(200,json={'data':[{'index':i,'embedding':[1,2]} for i in range(len(batch))]})
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    e=HTTPEmbedder(config(RAG_EMBEDDING_BATCH_SIZE='1'))
    assert e.documents(['a','b']).shape==(2,2)
    assert calls==[['a'],['b'],['b']]
    e.client.close()


@pytest.mark.parametrize('status,expected',[(401,1),(400,1),(502,4)])
def test_attempt_limit_and_safe_error(monkeypatch,status,expected):
    calls=[];original=httpx.Client
    monkeypatch.setattr('memory.embeddings.time.sleep',lambda _:None)
    def respond(req):
        calls.append(req)
        return httpx.Response(status,text='private-secret-provider-body')
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    e=HTTPEmbedder(config())
    with pytest.raises(EmbeddingUnavailable) as error:e.documents(['a'])
    assert len(calls)==expected
    assert 'private' not in str(error.value)
    e.client.close()


def test_embedding_concurrency_cap():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Lock
    import time
    e=HTTPEmbedder(config(RAG_EMBEDDING_CONCURRENCY='2'));e.client.close()
    active=0;peak=0;lock=Lock()
    def respond(req):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.02)
        with lock:active-=1
        return httpx.Response(200,json={'data':[{'index':0,'embedding':[1,2]}]})
    e.client=httpx.Client(transport=httpx.MockTransport(respond))
    with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(lambda _:e.documents(['a']),range(16)))
    assert peak==2
    e.client.close()


def test_retry_budget_stops_before_backoff(monkeypatch):
    original=httpx.Client;calls=[]
    def respond(req):
        calls.append(req)
        raise httpx.ReadTimeout('secret')
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(respond),**kw))
    monkeypatch.setattr('memory.embeddings.time.sleep',lambda _:pytest.fail('budget must stop retry'))
    e=HTTPEmbedder(config(RAG_EMBEDDING_RETRY_BUDGET='0.001'))
    with pytest.raises(EmbeddingUnavailable):e.documents(['a'])
    assert len(calls)<=1
    e.client.close()


def test_v4_auth_batching_and_response_order(monkeypatch):
    calls=[]
    original=httpx.Client
    def respond(request):
        assert request.headers['Authorization']=='Bearer test-key'
        body=json.loads(request.content);calls.append(body)
        assert body['encoding_format']=='float' and body['model']=='text-embedding-v4'
        return httpx.Response(200,json={'data':[
            dict(index=i,embedding=[float(t),1.]) for i,t in reversed(list(enumerate(body['input']))) ]})
    def client(**kwargs):
        assert kwargs['trust_env'] is False and kwargs['proxy'] is None
        return original(transport=httpx.MockTransport(respond),**kwargs)
    monkeypatch.setattr(httpx,'Client',client)
    e=HTTPEmbedder(config());v=e.documents([str(i) for i in range(23)])
    assert [len(c['input']) for c in calls]==[10,10,3]
    expected=np.array([[i,1.] for i in range(23)])
    np.testing.assert_allclose(v,expected/np.linalg.norm(expected,axis=1,keepdims=True),rtol=1e-6)
    assert 'test-key' not in e.identity


@pytest.mark.parametrize('data',[
    [{'index':0,'embedding':[1,0]},{'index':0,'embedding':[0,1]}],
    [{'index':1,'embedding':[1,0]},{'index':2,'embedding':[0,1]}],
    [{'embedding':[1,0]},{'embedding':[0,1]}],
])
def test_reject_ambiguous_vector_alignment(monkeypatch,data):
    original=httpx.Client
    monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(
        lambda request:httpx.Response(200,json={'data':data})),**kw))
    with pytest.raises(ValueError,match='indexes'):
        HTTPEmbedder(config()).documents(['first','second'])


def test_v4_requires_explicit_credentials_not_environment(monkeypatch):
    monkeypatch.setenv('RAG_EMBEDDING_API_KEY','environment-secret')
    cfg=config();cfg.pop('RAG_EMBEDDING_API_KEY')
    with pytest.raises(ValueError,match='root .env'):HTTPEmbedder(cfg)
    with pytest.raises(ValueError,match='batch size'):
        HTTPEmbedder(config(RAG_EMBEDDING_BATCH_SIZE='11'))


def test_dimension_identity_and_legacy_endpoint(monkeypatch):
    a=HTTPEmbedder(config(RAG_EMBEDDING_DIMENSIONS='1024'))
    b=HTTPEmbedder(config(RAG_EMBEDDING_DIMENSIONS='512'))
    assert a.identity!=b.identity
    legacy=HTTPEmbedder({'RAG_EMBEDDING_API_URL':'http://localhost:8004/v1/embeddings'})
    assert 'authorization' not in legacy.client.headers
    for e in (a,b,legacy):e.client.close()
