from types import SimpleNamespace

from memory.retrieval import CallbackRetriever, CallbackReranker, RetrievalQuery, chunks
from memory.search_service import SearchService


def test_retrieval_contracts_are_small_and_composable():
    payload = SimpleNamespace(query='memory')
    retriever = CallbackRetriever(lambda value: {'data': [{'content': value.query}]})
    assert retriever.retrieve(payload)['data'][0]['content'] == 'memory'

    reranker = CallbackReranker(lambda query, docs: [len(query) + len(doc) for doc in docs])
    assert list(reranker.score('q', ['abc'])) == [4]
    assert RetrievalQuery('u', 'q', 3).top_k == 3
    assert chunks('alpha beta', size=10, overlap=0) == ['alpha beta']


def test_search_service_dispatches_only_atomic_retrieval():
    payload = SimpleNamespace(query='q')
    direct = lambda value: {'data': [{'content': 'direct'}]}
    trace = {}
    service = SearchService(direct)
    assert service.search(payload, trace=trace)['data'][0]['content'] == 'direct'
    assert trace['mode'] == 'atomic'
    assert trace['retrieval_queries'][0]['query'] == 'q'
