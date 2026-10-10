"""One search pipeline: plan, select scope, retrieve, review."""
from threading import Lock
from contextvars import copy_context
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
        request_context = copy_context()

        def retrieve_query(query):
            return request_context.copy().run(retrieve_in_context, query)

        def retrieve_in_context(query):
            # Planner-generated concrete subqueries inherit the request clock.
            from types import SimpleNamespace
            query = SimpleNamespace(**(query.model_dump() if hasattr(query,'model_dump') else vars(query)))
            query.reference_time = getattr(payload,'reference_time',None)
            query.reference_timezone = getattr(payload,'reference_timezone','UTC')
            query._include_retrieval_context = planned
            # Each concurrent branch owns its scope and diagnostics. Never let a
            # subquery's partition trace overwrite the planner's request trace.
            local = {'query': query.query, 'top_k': getattr(query, 'top_k', None)}
            with trace_lock:
                local['call_id'] = len(trace.setdefault('retrieval_queries', [])) + 1
                trace['retrieval_queries'].append(local)
                if partitioned:
                    trace.setdefault('partition_queries', []).append(local)
            try:
                if partitioned:
                    result = self.partition_search.search(query, self.direct_retriever, local)
                else:
                    query.retrieval_trace = local
                    result = self.direct_retriever.retrieve(query)
                from .position_trace import record_positions
                if all(isinstance(h, dict) and 'id' in h for h in result['data']):
                    record_positions(local, 'returned', [h['id'] for h in result['data']])
                if not planned:
                    trace.update({k:(dict(v) if k == 'positions' else v)
                                  for k,v in local.items() if k not in ('query', 'call_id')})
                return result
            except Exception as exc:
                from .search_audit import error_metadata
                local.update(error_metadata(exc))
                raise

        if hasattr(self.direct_retriever, 'expand_neighbors'):
            retrieve_query.expand_neighbors = lambda anchors, missing: request_context.copy().run(
                self.direct_retriever.expand_neighbors, payload, anchors, missing)

        if planned:
            trace['pipeline'] = 'plan_scope_retrieve_review'
            result = self.multihop.run(payload, retrieve_query,
                                      self.reranker.score if self.reranker else None, trace)
            return dict(result, data=[{k:v for k,v in hit.items() if not k.startswith('_')}
                                      if isinstance(hit,dict) else hit for hit in result['data']])
        # Disabling planning reduces the same pipeline to one concrete query.
        trace.update(mode='atomic' if not partitioned else 'partition')
        return retrieve_query(payload)
