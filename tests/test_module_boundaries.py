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


def test_search_service_switches_between_direct_and_multihop():
    payload = SimpleNamespace(query='q')
    direct = lambda value: {'data': [{'content': 'direct'}]}
    assert SearchService(direct).search(payload)['data'][0]['content'] == 'direct'

    class Planner:
        mode = 'llm'

        def run(self, value, retrieve, rerank, trace):
            assert retrieve(value)['data'][0]['content'] == 'direct'
            assert rerank('q', ['doc']) == [3]
            return {'data': [{'content': 'planned'}]}

    service = SearchService(direct, Planner(), lambda query, docs: [3])
    assert service.search(payload)['data'][0]['content'] == 'planned'
