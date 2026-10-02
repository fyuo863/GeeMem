import json
import httpx
import numpy as np
import pytest
from memory.vanilla import HTTPEmbedder


def config(**kwargs):
    return dict(RAG_EMBEDDING_API_URL='https://embedding.test/v1/embeddings',
        RAG_EMBEDDING_API_MODEL='text-embedding-v4', RAG_EMBEDDING_API_KEY='test-key', **kwargs)


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
