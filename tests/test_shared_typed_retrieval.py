from contextlib import closing
import numpy as np
import pytest

from memory.aml_api import AMLAdd
from memory.atomic_retriever import AtomicQuery
from memory.vanilla import VanillaMemory


class Embedder:
    identity = 'shared-test'
    def documents(self, texts): return np.ones((len(texts), 3))
    def queries(self, texts): return np.ones((len(texts), 3))


class Reranker:
    def __init__(self): self.calls = []
    def score(self, query, documents):
        self.calls.append((query, documents))
        return [10 if 'Arsenal' in d else 1 for d in documents]


def setup(tmp_path):
    reranker = Reranker()
    b = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'db'),RAG_BUILD_MODE='on',
                           RAG_RESULT_WINDOW='0'), Embedder(), reranker=reranker)
    p = AMLAdd(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='user',content='I like football.'),
        dict(role='user',content='Arsenal is my favorite team.'),
        dict(role='user',content='My sister likes football.')])
    b.add(p)
    b.route_writer.builders['profile'].write(p,[0,1])
    b.route_writer.builders['relationship'].write(p,[2])
    return b, reranker


def test_typed_adapter_uses_same_ranker_and_results(tmp_path):
    b, reranker = setup(tmp_path)
    scoped = b.profile_retriever.search('u','football',fallback=False)
    direct = b.atomic_retriever.query(AtomicQuery('u','football',memory_types=('profile',),
                                      fallback=False,include_evidence=True))['data']
    assert scoped == direct
    assert scoped[0]['content']=='Arsenal is my favorite team.'
    assert len(reranker.calls)==2
    assert all(len(docs)==2 for _,docs in reranker.calls)
    assert len(scoped)==2 and all(h['memory_type']=='profile' for h in scoped)


def test_multiple_types_union_and_session_boundary(tmp_path):
    b,_ = setup(tmp_path)
    result = b.atomic_retriever.query(AtomicQuery('u','football',memory_types=('profile','relationship'),
                                     fallback=False,include_evidence=True))['data']
    assert len(result)==3 and len({h['id'] for h in result})==3
    assert b.atomic_retriever.query(AtomicQuery('u','football',session_id='other'))=={'data':[]}
    assert b.atomic_retriever.query(AtomicQuery('other','football'))=={'data':[]}


def test_strict_type_does_not_leak_neighbor_and_fallback_uses_same_engine(tmp_path):
    b,_ = setup(tmp_path)
    b.window=1
    strict=b.relationship_retriever.search('u','football',fallback=False)
    assert len(strict)==1 and strict[0]['content']=='My sister likes football.'
    assert len(b.relationship_retriever.search('u','football',fallback=True))==3
    assert b.rule_retriever.search('u','football',fallback=False)==[]


def test_missing_type_table_is_empty_in_strict_mode(tmp_path):
    b,_=setup(tmp_path)
    with closing(b.connect()) as db,db: db.execute('DROP TABLE profile_sources')
    assert b.profile_retriever.search('u','football',fallback=False)==[]
    assert len(b.profile_retriever.search('u','football',fallback=True))==3


def test_unknown_type_rejected():
    with pytest.raises(ValueError): AtomicQuery('u','q',memory_types=('invented',))
