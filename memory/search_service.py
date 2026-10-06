"""Application-level search orchestration.

This layer decides whether a request uses direct retrieval or the multihop
executor. It deliberately knows nothing about SQLite, embeddings, or HTTP.
"""

from .retrieval import CallbackRetriever, CallbackReranker


class SearchService:
    def __init__(self, direct_retriever, multihop=None, reranker=None):
        self.direct_retriever = (direct_retriever if hasattr(direct_retriever, 'retrieve')
                                 else CallbackRetriever(direct_retriever))
        self.multihop = multihop
        self.reranker = (reranker if hasattr(reranker, 'score') else
                         (CallbackReranker(reranker) if reranker is not None else None))

    def search(self, payload, trace=None):
        if self.multihop is None or self.multihop.mode == 'off':
            if trace is not None:
                trace.update(mode='off')
            return self.direct_retriever.retrieve(payload)
        if self.reranker is None:
            raise ValueError('Multihop search requires a reranker')
        return self.multihop.run(payload, self.direct_retriever.retrieve,
                                 self.reranker.score, trace if trace is not None else {})
