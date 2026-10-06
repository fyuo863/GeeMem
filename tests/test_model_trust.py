from memory.multihop import MultiHop, Query, Review, Route, Support
from test_multihop import hit, request


def test_model_trust_executes_query_without_program_bridge_rejection():
    class Planner:
        def route(self, *args):
            return Route(strategy='chain', queries=[
                Query(query='Who is Alice married to?', source_id='missing-source', bridge='invented-bridge')
            ])

        def review(self, *args):
            return Review(sufficient=True, supports=[
                Support(source_id='memory', quote='model selected this', needed_for='answer')
            ], missing='')

    calls = []

    def retrieve(payload):
        calls.append(payload.query)
        return {'data': [hit('memory', 'Alice is married to Bob.')]} 

    trace = {}
    result = MultiHop({'RAG_MULTIHOP_MODE': 'llm', 'RAG_MULTIHOP_VALIDATION': 'trust'}, Planner()).run(
        request(k=1), retrieve, lambda query, docs: [1] * len(docs), trace)
    assert calls == [request().query, 'Who is Alice married to?']
    assert result['data'][0]['id'] == 'memory'
    assert trace['rejected_queries'] == 0
    assert trace['rejected_supports'] == 0


def test_strict_mode_keeps_bridge_rejection_for_comparison():
    class Planner:
        def route(self, *args):
            return Route(strategy='chain', queries=[
                Query(query='unusable', source_id='missing-source', bridge='invented-bridge')
            ])

        def review(self, *args):
            return Review(sufficient=True, supports=[], missing='')

    trace = {}
    MultiHop({'RAG_MULTIHOP_MODE': 'llm', 'RAG_MULTIHOP_VALIDATION': 'strict'}, Planner()).run(
        request(k=1), lambda payload: {'data': [hit('memory', 'Alice lives in Paris.')]},
        lambda query, docs: [1] * len(docs), trace)
    assert trace['rejected_queries'] == 1
