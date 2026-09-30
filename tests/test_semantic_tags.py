import sqlite3
import numpy as np
import pytest
from memory.aml_api import AMLSearch
from memory.vanilla import VanillaMemory
from test_vanilla import add


class Model:
    identity='semantic-test-v1'
    def documents(self,texts):
        return np.array([[0.,1.] if text=='song' else [1.,0.] for text in texts])
    def queries(self,texts):
        return np.array([[0.,1.] if text=='musician' else [1.,0.] for text in texts])


class Tags:
    identity='fixed-test-tags-v1'
    def extract(self,texts,query=False):
        return [['musician'] if query else ['song'] if 'song' in t else ['other'] for t in texts]


def setup(tmp_path,mode,**extra):
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE=mode,RAG_RESULT_WINDOW='0',**extra)
    store=VanillaMemory(cfg,Model(),Tags())
    store.add(add(content='source unrelated'))
    store.add(add(rid='two',content='source song'))
    return store,cfg


def query():return AMLSearch(user_id='u',query='find performer',top_k=10)


def test_soft_semantics_promotes_indirect_evidence_without_deleting(tmp_path):
    s,cfg=setup(tmp_path,'semantic_rank')
    result=s.search(query())
    assert [h['content'] for h in result['data']]==['source song','source unrelated']
    assert result['data'][0]['score']==pytest.approx(1/62+0.5/61)
    assert VanillaMemory(cfg,Model(),Tags()).search(query())==result
    assert s.search(AMLSearch(user_id='other',query='find performer',top_k=10))=={'data':[]}


def test_hard_semantics_and_no_match_fallback(tmp_path):
    s,cfg=setup(tmp_path,'semantic_filter')
    assert [h['content'] for h in s.search(query())['data']]==['source song']
    s.embedder.queries=lambda texts: np.array([[-1.,-1.] for t in texts])
    assert len(s.search(query())['data'])==2


def test_zero_weight_preserves_original_order(tmp_path):
    s,cfg=setup(tmp_path,'semantic_rank',RAG_TAG_WEIGHT='0')
    baseline=VanillaMemory(dict(cfg,RAG_TAG_MODE='off'),Model())
    assert s.search(query())==baseline.search(query())


def test_legacy_keywords_have_no_vector_and_are_preserved(tmp_path):
    s,cfg=setup(tmp_path,'filter')
    upgraded=VanillaMemory(dict(cfg,RAG_TAG_MODE='semantic_filter'),Model(),Tags())
    before=upgraded.search(query())
    assert len(before['data'])==2
    upgraded.add(add(rid='new',content='source song'))
    assert len(upgraded.search(query())['data'])==3


def test_failed_keyword_embedding_writes_nothing(tmp_path):
    model=Model()
    original=model.documents
    def invalid(texts):
        return np.full((len(texts),2),np.nan) if texts==['song'] else original(texts)
    model.documents=invalid
    s=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE='semantic_rank'),model,Tags())
    with pytest.raises(ValueError):s.add(add(content='source song'))
    with sqlite3.connect(s.path) as db:
        assert db.execute('SELECT count(*) FROM rag_requests').fetchone()[0]==0
        assert db.execute('SELECT count(*) FROM rag_tag_vectors').fetchone()[0]==0
