"""Application-level orchestration for the atomic retrieval backend."""

from .retrieval import CallbackRetriever


class SearchService:
    def __init__(self, direct_retriever):
        self.direct_retriever = (direct_retriever if hasattr(direct_retriever, 'retrieve')
                                 else CallbackRetriever(direct_retriever))

    def search(self, payload, trace=None):
        if trace is not None:
            trace.update(mode='atomic')
        return self.direct_retriever.retrieve(payload)
