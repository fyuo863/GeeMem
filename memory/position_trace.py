"""Request-local, 1-based source positions. No raw text or provider credentials."""


def record_positions(trace, stage, ids):
    if trace is not None:
        trace.setdefault('positions', {})[stage] = {
            source_id: rank for rank, source_id in enumerate(ids, 1)
        }


def record_returned(trace, hits):
    record_positions(trace, 'final_results', [h['id'] for h in hits])
    members = trace.get('bundle_sources', {})
    trace['final_source_positions'] = {}
    for rank, hit in enumerate(hits, 1):
        for offset, sid in enumerate(members.get(hit['id'], [hit['id']]), 1):
            trace['final_source_positions'].setdefault(sid, []).append(
                {'result_rank': rank, 'member_position': offset, 'result_id': hit['id']})

class PositionLog:
    """Optional rotating JSONL sink; only IDs/positions, never query or source text."""
    def __init__(self, cfg):
        from logging.handlers import RotatingFileHandler
        from pathlib import Path
        from .config import PROJECT_ROOT
        self.handler = None
        path = cfg.get('RAG_POSITION_LOG_PATH', '').strip()
        if path:
            target = Path(path)
            if not target.is_absolute(): target = PROJECT_ROOT / target
            target.parent.mkdir(parents=True, exist_ok=True)
            self.handler = RotatingFileHandler(target, maxBytes=10*1024*1024,
                                               backupCount=3, encoding='utf8', delay=True)

    def write(self, trace):
        if self.handler is None: return
        import json
        import logging
        from datetime import datetime, timezone
        from uuid import uuid4
        trace.setdefault('position_trace_id', uuid4().hex)
        event = dict(trace_id=trace['position_trace_id'], at=datetime.now(timezone.utc).isoformat(),
            positions=trace.get('positions', {}), final_source_positions=trace.get('final_source_positions', {}),
            queries=[{k:q[k] for k in ('call_id','top_k','positions','reranker_applied','error_type') if k in q}
                     for q in trace.get('retrieval_queries', [])],
            rounds=[{k:r[k] for k in ('round','positions') if k in r} for r in trace.get('rounds', [])],
            adjacency_groups=trace.get('adjacency_groups', []),
            omitted_review_groups=trace.get('omitted_review_groups', []),
            stop=trace.get('stop'), fallback=trace.get('fallback', False))
        self.handler.handle(logging.LogRecord(__name__, logging.INFO, '', 0,
            json.dumps(event, ensure_ascii=False, separators=(',',':')), (), None))
