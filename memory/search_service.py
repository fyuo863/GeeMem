"""Application-level orchestration for the atomic retrieval backend."""

from .retrieval import CallbackRetriever


class SearchService:
    def __init__(self, direct_retriever, multihop=None, reranker=None, partition_search=None):
        self.direct_retriever = (direct_retriever if hasattr(direct_retriever, 'retrieve')
                                 else CallbackRetriever(direct_retriever))
        self.multihop = multihop
        self.reranker = reranker
        self.partition_search = partition_search

    def search(self, payload, trace=None):
        if self.multihop is not None and self.multihop.mode != 'off':
            result = self.multihop.run(payload, self.direct_retriever.retrieve,
                                       self.reranker.score if self.reranker else None, trace or {})
            return result
        if self.partition_search is not None and self.partition_search.mode != 'off':
            return self.partition_search.search(payload, self.direct_retriever, trace)
        if trace is not None:
            trace.update(mode='atomic')
        return self.direct_retriever.retrieve(payload)
