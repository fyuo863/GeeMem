from types import SimpleNamespace
import pytest
from memory.evidence_bundle import EvidenceBundler
from memory.multihop import MultiHop
from test_multihop import ChainPlanner, request, hit


def assemble(user='u', text='Bob works at Atlas.', sufficient=True, mode='on'):
    ranked=[hit('end',text,3),hit('link','Alice is married to Bob.',2),hit('noise','Other text',1)]
    supports={h['id']:SimpleNamespace(quote=h['content']) for h in ranked[:2]}
    trace={}
    result=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':mode}).assemble(user,ranked,supports,sufficient,2,trace)
    return result,trace


def test_exact_sources_stable_id_dedup_backfill_and_scope():
    a,t=assemble();b,_=assemble();c,_=assemble(user='v');d,_=assemble(text='Bob works at Other.')
    assert a==b and a['data'][0]['id']!=c['data'][0]['id']!=d['data'][0]['id']
    assert a['data'][0]['id'].startswith('bundle_')
    assert a['data'][1]['id']=='noise'
    assert 'Bob works at Atlas.' in a['data'][0]['content']
    assert 'Alice is married to Bob.' in a['data'][0]['content']
    assert 'created_at' not in a['data'][0]
    assert a['_bundles'][0]['source_ids']==['end','link']


@pytest.mark.parametrize('kwargs',[{'sufficient':False},{'mode':'off'},{'text':'x'*6001}])
def test_no_incomplete_or_oversized_bundle(kwargs):
    result,trace=assemble(**kwargs)
    assert result['data'][0]['id']=='end'
    assert '_bundles' not in result


def test_untrusted_quote_rejected():
    trace={};r=EvidenceBundler({'RAG_EVIDENCE_BUNDLE_MODE':'on'}).assemble('u',
        [hit('a','original'),hit('b','other')],
        {'a':SimpleNamespace(quote='invented'),'b':SimpleNamespace(quote='other')},True,1,trace)
    assert r['data'][0]['id']=='a' and trace['evidence_bundle']['status']=='invalid_quote'


def test_chain_bundles_before_top1_without_extra_model_calls():
    def retrieve(p):
        if p.query=='Who is Alice married to?': return {'data':[hit('spouse','Alice is married to Bob.')]}
        if p.query=='Where does Bob work?': return {'data':[hit('employer','Bob works at Atlas.')]}
        return {'data':[hit('noise','Alice likes hiking.')]}
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_BUNDLE_MODE':'on'},ChainPlanner()).run(
        request(k=1),retrieve,lambda q,ds:[1]*len(ds),trace)
    assert len(result['data'])==1
    assert all(s in result['data'][0]['content'] for s in ['Alice is married to Bob.','Bob works at Atlas.'])
    assert trace['llm_calls']==3 and trace['search_calls']==3


def test_backend_persists_mapping_without_private_api_fields(tmp_path):
    import sqlite3
    from memory.vanilla import VanillaMemory
    from test_vanilla import Embedder
    store=VanillaMemory({'RAG_MEMORY_DB':str(tmp_path/'memory.db')},embedder=Embedder())
    store.search_service=SimpleNamespace(search=lambda *a,**kw:assemble()[0])
    result=store.search(request(k=2))
    assert set(result)=={'data'}
    with sqlite3.connect(tmp_path/'memory.db') as db:
        assert db.execute('select user_id,source_ids from rag_evidence_bundles').fetchone()==('u','["end", "link"]')


def test_expanded_review_pool_still_limits_failure_response():
    from memory.llm import LLMError
    calls=[]
    class FailedReview(ChainPlanner):
        def review(self,*args): raise LLMError('temporary failure')
    def retrieve(p):
        calls.append(p.top_k)
        return {'data':[hit(str(i),f'Alice evidence {i}.') for i in range(p.top_k)]}
    trace={}
    result=MultiHop({'RAG_MULTIHOP_MODE':'llm','RAG_EVIDENCE_BUNDLE_MODE':'on'},FailedReview()).run(
        request(k=1),retrieve,lambda q,ds:[1]*len(ds),trace)
    assert calls[0]==12 and trace['fallback']
    assert len(result['data'])==1 and '_bundles' not in result
