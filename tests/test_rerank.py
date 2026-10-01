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


def test_context_support_is_single_step_and_session_scoped():
 from memory.rerank import context_support_scores
 rows=[dict(session_id='a') for _ in range(4)]+[dict(session_id='b')]
 order,scores=context_support_scores(rows,[0,3],[10.,7.],2.)
 assert order==[0,1,3,2]
 assert scores[1]==8 and scores[2]==5
 assert 4 not in scores


def test_support_promotes_neighbor_without_changing_original_text(tmp_path):
 class Fixed:
  def score(self,query,docs):return [8 if 'Target message: marathon' in d else 0 for d in docs]
 s=VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RESULT_WINDOW='0',RAG_RERANK_CONTEXT='1',RAG_RERANK_SELECTION='context_support',RAG_RERANK_CANDIDATES='1'),Embedder(),reranker=Fixed())
 s.add(add(content='marathon'));s.add(add(rid='two',content='piano'))
 hits=s.search(AMLSearch(user_id='u',query='marathon',top_k=2))['data']
 assert [h['content'] for h in hits]==['marathon','piano']
 assert hits[1]['score']==6


def test_support_rejects_double_window(tmp_path):
 with pytest.raises(ValueError,match='result window=0'):
  VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_RERANK_CONTEXT='1',RAG_RERANK_SELECTION='context_support'),Embedder(),reranker=Reranker())
