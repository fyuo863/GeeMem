import json
import sqlite3
import pytest
from memory.aml_api import AMLSearch
from memory.vanilla import VanillaMemory
from memory.tags import normalize_tags, RuleTagger
from memory.llm import LLMError
from test_vanilla import Embedder, add


class Tags:
    identity = 'fake-tags-v1'
    def __init__(self): self.calls = 0
    def extract(self, texts, query=False):
        self.calls += 1
        return [['sport'] if 'marathon' in text else ['music'] for text in texts]


def test_filter_and_no_match_fallback(tmp_path):
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_TAG_MODE='filter')
    t=Tags();s=VanillaMemory(cfg,Embedder(),t)
    s.add(add(content='marathon'));s.add(add(rid='two',content='piano'))
    hits=s.search(AMLSearch(user_id='u',query='marathon',top_k=10))['data']
    assert [h['content'] for h in hits]==['marathon']
    t.extract=lambda texts,query=False:[['unknown'] for _ in texts]
    assert len(s.search(AMLSearch(user_id='u',query='unknown',top_k=10))['data'])==2


def test_untagged_legacy_and_other_versions_survive(tmp_path):
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0')
    old=VanillaMemory(cfg,Embedder());old.add(add(content='legacy'))
    s=VanillaMemory(dict(cfg,RAG_TAG_MODE='filter'),Embedder(),Tags())
    s.add(add(rid='new',content='marathon'))
    assert len(s.search(AMLSearch(user_id='u',query='marathon',top_k=10))['data'])==2
    assert s.search(AMLSearch(user_id='other',query='marathon',top_k=10))=={'data':[]}


def test_tags_atomic_and_retry(tmp_path):
    t=Tags();s=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE='filter'),Embedder(),t)
    s.add(add());s.add(add());assert t.calls==1
    def fail(*args,**kwargs):raise LLMError('invalid')
    t.extract=fail
    with pytest.raises(LLMError):s.add(add(rid='fail'))
    with sqlite3.connect(s.path) as db:
        assert db.execute('SELECT count(*) FROM rag_requests').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM rag_tags').fetchone()[0]==1


def test_rule_keywords_are_independent_and_deterministic():
    tagger=RuleTagger()
    assert normalize_tags([' Sport ', 'SPORT','', 'x'*81])==['sport']
    assert tagger.extract(['Hello! How are you?']) == [[]]
    texts=["Alice’s horses and books", 'Painting in Paris', '杭州马拉松']
    batch=tagger.extract(texts)
    assert batch[0]==['alice','book','horse']
    assert 'painting' not in batch[0]
    assert '杭州' in batch[2] and '马拉' in batch[2]
    assert batch==[tagger.extract([text])[0] for text in texts]
    assert batch==tagger.extract(texts,query=True)


def test_rule_backend_never_calls_llm(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise AssertionError('Unexpected LLM call')
    monkeypatch.setattr('memory.llm.LLM.complete',fail)
    store=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE='filter',RAG_RESULT_WINDOW='0'),Embedder())
    store.add(add(content='marathon training'))
    store.add(add(rid='two',content='piano concert'))
    hits=store.search(AMLSearch(user_id='u',query='marathon',top_k=10))['data']
    assert [h['content'] for h in hits]==['marathon training']


def test_old_semantic_tags_remain_unknown(tmp_path):
    cfg=dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_TAG_MODE='filter',RAG_RESULT_WINDOW='0')
    store=VanillaMemory(cfg,Embedder())
    store.add(add(content='marathon'))
    store.add(add(rid='old',content='piano'))
    with sqlite3.connect(store.path) as db:
        db.execute("UPDATE rag_tags SET identity='semantic-tags-v1:gpt-4o-mini' WHERE memory_id IN (SELECT id FROM rag_memories WHERE content='piano')")
    assert len(store.search(AMLSearch(user_id='u',query='marathon',top_k=10))['data'])==2
