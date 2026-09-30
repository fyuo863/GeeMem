import json
import sqlite3
import pytest
from memory.aml_api import AMLSearch
from memory.vanilla import VanillaMemory
from memory.tags import normalize_tags, SemanticTagger, TagBatch
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


def test_normalize_and_response_alignment():
    assert normalize_tags([' Sport ', 'SPORT','', 'x'*81])==['sport']
    t=SemanticTagger.__new__(SemanticTagger)
    class Fake:
        def complete(self,*args):return TagBatch(items=[dict(tags=['sport'])])
    t.llm=Fake()
    with pytest.raises(LLMError):t.extract(['one','two'])
