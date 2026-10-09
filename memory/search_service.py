"""One search pipeline: plan, select scope, retrieve, review."""
from threading import Lock
from .retrieval import CallbackRetriever


class SearchService:
    def __init__(self, direct_retriever, multihop=None, reranker=None, partition_search=None):
        self.direct_retriever = (direct_retriever if hasattr(direct_retriever, 'retrieve')
                                 else CallbackRetriever(direct_retriever))
        self.multihop = multihop
        self.reranker = reranker
        self.partition_search = partition_search

    def search(self, payload, trace=None):
        trace = trace if trace is not None else {}
        partitioned = self.partition_search is not None and self.partition_search.mode != 'off'
        planned = self.multihop is not None and self.multihop.mode != 'off'
        trace_lock = Lock()

        def retrieve_query(query):
            # Each concurrent branch owns its scope and diagnostics. Never let a
            # subquery's partition trace overwrite the planner's request trace.
            if not partitioned:
                return self.direct_retriever.retrieve(query)
            local = {'query': query.query}
            try:
                return self.partition_search.search(query, self.direct_retriever, local)
            finally:
                with trace_lock:
                    trace.setdefault('partition_queries', []).append(local)

        if planned:
            trace['pipeline'] = 'plan_scope_retrieve_review'
            return self.multihop.run(payload, retrieve_query,
                                    self.reranker.score if self.reranker else None, trace)
        # Disabling planning reduces the same pipeline to one concrete query.
        if partitioned:
            return self.partition_search.search(payload, self.direct_retriever, trace)
        trace.update(mode='atomic')
        return self.direct_retriever.retrieve(payload)
