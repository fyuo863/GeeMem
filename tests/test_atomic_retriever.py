from types import SimpleNamespace

import pytest

from memory.atomic_retriever import AtomicQuery, AtomicRetriever


class Backend:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def _search_direct(self, payload):
        self.calls.append(payload)
        return self.result


def test_atomic_retriever_is_a_single_query_model_free_boundary():
    backend = Backend({'data': [{'id': 'm1', 'content': 'Alice lives in Paris.', 'score': 0.8}]})
    retriever = AtomicRetriever(backend)
    result = retriever.query(AtomicQuery('u', 'Where does Alice live?', top_k=3))
    assert result['data'][0]['id'] == 'm1'
    assert backend.calls[0].query == 'Where does Alice live?'
    assert backend.calls[0].user_id == 'u'
    assert backend.calls[0].top_k == 3


def test_atomic_retriever_deduplicates_and_rejects_invalid_evidence():
    backend = Backend({'data': [
        {'id': 'm1', 'content': 'first', 'score': 1},
        {'id': 'm1', 'content': 'duplicate', 'score': 0.5},
    ]})
    assert len(AtomicRetriever(backend).retrieve(SimpleNamespace(query='q'))['data']) == 1
    with pytest.raises(ValueError, match='non-finite'):
        AtomicRetriever(Backend({'data': [{'id': 'm1', 'content': 'x', 'score': float('nan')}]})).retrieve(
            SimpleNamespace(query='q'))
