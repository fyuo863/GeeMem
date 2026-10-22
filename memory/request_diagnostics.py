"""Content-free request/stage diagnostics; never log exception strings or bodies."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import logging
from pathlib import Path
import sqlite3
import time
from .provider_errors import EmbeddingUnavailable

_current = ContextVar('request_diagnostic', default=None)
_log = logging.getLogger(__name__)
_reasons = {
    'Multi-label judge must return every label exactly once': 'classifier_labels',
    'Select at least one label': 'classifier_empty',
    'Exclusive label cannot coexist with other labels': 'classifier_exclusivity',
    'Only selected labels must have nonempty source indices': 'classifier_selection_indices',
    'Invalid source indices': 'classifier_indices',
    'Invalid route source indices': 'route_indices',
    'No nonblank content to embed': 'empty_chunks',
    'Invalid embedding response indexes': 'embedding_indexes',
    'Embedding dimension mismatch': 'embedding_dimension',
    'Embedding dimension changed': 'embedding_dimension',
    'Invalid embedding API response': 'embedding_vectors',
    'Zero embedding vector': 'embedding_zero_vector',
}


@contextmanager
def diagnostic_stage(name):
    ctx = _current.get()
    old = ctx['stage'] if ctx else None
    started = time.monotonic()
    if ctx is not None:
        ctx['stage'] = name
    try:
        yield
    except Exception:
        # Retain the innermost failed stage until the request boundary logs it.
        if ctx is not None and 'stage_elapsed_ms' not in ctx:
            ctx['stage_elapsed_ms'] = round((time.monotonic()-started)*1000)
        raise
    else:
        if ctx is not None:
            ctx['stage'] = old


def observe_request(endpoint):
    def decorate(func):
        @wraps(func)
        def wrapped(payload):
            identity = [endpoint, payload.user_id, getattr(payload, 'request_id', '')]
            ctx = {'endpoint': endpoint, 'request_hash': hashlib.sha256(
                json.dumps(identity, ensure_ascii=True).encode()).hexdigest()[:24], 'stage': 'backend'}
            token = _current.set(ctx)
            started = time.monotonic()
            status = 200
            try:
                return func(payload)
            except Exception as exc:
                status = (503 if isinstance(exc, (EmbeddingUnavailable, sqlite3.OperationalError))
                          else getattr(exc, 'status_code', 500))
                causes = []
                seen = set()
                item = exc
                while item is not None and id(item) not in seen and len(causes) < 8:
                    seen.add(id(item))
                    row = {'type': type(item).__name__}
                    response = getattr(item, 'response', None)
                    if response is not None:
                        row['upstream_status'] = response.status_code
                    tb = item.__traceback__
                    while tb is not None:
                        if tb.tb_frame.f_globals.get('__name__', '').startswith('memory.'):
                            row['location'] = Path(tb.tb_frame.f_code.co_filename).name + ':' + str(tb.tb_lineno)
                        tb = tb.tb_next
                    # Only exact, program-owned validation strings may become codes.
                    if type(item) is ValueError and str(item) in _reasons:
                        row['code'] = _reasons[str(item)]
                    causes.append(row)
                    item = item.__cause__ or item.__context__
                _log.warning('request_failed %s', json.dumps(dict(ctx, status=status, causes=causes,
                    elapsed_ms=round((time.monotonic()-started)*1000)), ensure_ascii=True))
                raise
            finally:
                # This logger can be enabled without enabling HTTP access logs.
                _log.warning('request_finished %s', json.dumps(dict(ctx, status=status,
                    elapsed_ms=round((time.monotonic()-started)*1000)), ensure_ascii=True))
                _current.reset(token)
        return wrapped
    return decorate
