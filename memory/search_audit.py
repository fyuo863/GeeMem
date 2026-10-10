"""Opt-in local search audit. Never serialize configuration or exception text."""
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path


class AuditHandler(RotatingFileHandler):
    def handleError(self, record):
        # logging's default error handler can print the entire sensitive record.
        logging.getLogger(__name__).warning('search_audit_write_failed')


class SearchAudit:
    def __init__(self, cfg):
        self.handler = None
        path = cfg.get('RAG_SEARCH_AUDIT_PATH', '').strip()
        if path:
            from .config import PROJECT_ROOT
            target = Path(path)
            if not target.is_absolute():
                target = PROJECT_ROOT / target
            target.parent.mkdir(parents=True, exist_ok=True)
            self.handler = AuditHandler(
                target, maxBytes=int(cfg.get('RAG_SEARCH_AUDIT_MAX_BYTES', 20*1024*1024)),
                backupCount=int(cfg.get('RAG_SEARCH_AUDIT_BACKUPS', 5)),
                encoding='utf8', delay=True)
            if self.handler.maxBytes <= 0 or self.handler.backupCount < 1:
                raise ValueError('Search audit requires positive size and backup count')

    @property
    def enabled(self):
        return self.handler is not None

    def write(self, payload, trace, result, error, started_at, elapsed):
        if not self.enabled:
            return
        try:
            request = {k: getattr(payload, k) for k in (
                'user_id', 'query', 'options', 'top_k', 'session_id',
                'reference_time', 'reference_timezone') if hasattr(payload, k)}
            fields = ('pipeline', 'mode', 'strategy', 'initial_plan', 'read_version',
                      'rounds', 'audit_review_inputs', 'retrieval_queries', 'positions', 'final_source_positions',
                      'bundle_sources', 'support_ids', 'stop', 'fallback', 'error_type',
                      'error_phase', 'cause_type', 'http_status', 'search_calls', 'llm_calls',
                      'rule_applicability', 'rejected_queries', 'rejected_supports')
            event = dict(schema_version=1, trace_id=trace['position_trace_id'],
                         started_at=started_at, finished_at=datetime.now(timezone.utc).isoformat(),
                         elapsed_seconds=elapsed, status='error' if error else 'success',
                         request=request, trace={k: trace[k] for k in fields if k in trace},
                         response=result, error=error)
            self.handler.handle(logging.LogRecord(__name__, logging.INFO, '', 0,
                json.dumps(event, ensure_ascii=False, separators=(',', ':')), (), None))
        except Exception:
            # An audit failure must never change the search result or leak a payload.
            logging.getLogger(__name__).warning('search_audit_write_failed')


def error_metadata(exc):
    cause = exc.__cause__ or exc
    result = {'error_type': type(exc).__name__, 'cause_type': type(cause).__name__}
    response = getattr(cause, 'response', None)
    if response is not None:
        result['http_status'] = response.status_code
    return result
