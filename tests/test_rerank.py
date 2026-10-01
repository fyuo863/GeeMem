import numpy as np
import pytest
from memory.vanilla import VanillaMemory
from memory.aml_api import AMLSearch
from test_vanilla import Embedder,add


class Reranker:
 def __init__(self):self.docs=[]
 def score(self,query,documents):
  self.docs=documents
  return [1 if 'piano' in d else 0 for d in documents]


def test_rerank_source_preservation_and_isolation(tmp_path):
 rank=Reranker();s=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0'),Embedder(),reranker=rank)
 s.add(add(content='marathon'));s.add(add(rid='two',content='piano'));s.add(add(user='other',content='private'))
 q=AMLSearch(user_id='u',query='marathon',top_k=1)
 hits=s.search(q)['data']
 assert hits[0]['content']=='piano' and 'private' not in rank.docs
 rank.score=lambda q,docs:np.full(len(docs),np.nan)
 with pytest.raises(ValueError):s.search(q)


def test_context_is_scoped_and_not_returned(tmp_path):
 rank=Reranker();s=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_RERANK_CONTEXT='1'),Embedder(),reranker=rank)
 s.add(add(content='marathon'));s.add(add(rid='two',session='separate',content='piano'))
 hits=s.search(AMLSearch(user_id='u',query='marathon',top_k=2))['data']
 assert {h['content'] for h in hits}=={'marathon','piano'}
 assert all(not ('piano' in d and 'marathon' in d) for d in rank.docs)
