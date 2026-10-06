"""Single-fact retrieval with no planning or language-model calls.

``AtomicRetriever`` is the lowest retrieval layer. It accepts one concrete
query, delegates dense/lexical/tag/reranker work to the configured vector
backend, and returns only original memory evidence. Planning and multi-hop
controllers can call it repeatedly.
"""

from dataclasses import dataclass
from math import isfinite
from types import SimpleNamespace
from typing import Any


@dataclass(frozen=True)
class AtomicQuery:
    user_id: str
    query: str
    top_k: int = 10
    session_id: str | None = None
    options: tuple[str, ...] = ()


class AtomicRetriever:
    """Adapter exposing the backend's model-free direct retrieval path."""

    def __init__(self, backend):
        if not hasattr(backend, '_search_direct'):
            raise TypeError('AtomicRetriever requires a direct-retrieval backend')
        self.backend = backend

    def retrieve(self, payload: Any) -> dict:
        """Retrieve evidence for exactly one query; never invoke a planner."""
        result = self.backend._search_direct(payload)
        return self._normalize(result)

    def query(self, request: AtomicQuery) -> dict:
        """Convenience entry point for callers outside the HTTP layer."""
        payload = SimpleNamespace(user_id=request.user_id, query=request.query,
                                  top_k=request.top_k, session_id=request.session_id,
                                  options=list(request.options))
        return self.retrieve(payload)

    @staticmethod
    def _normalize(result: dict) -> dict:
        if not isinstance(result, dict) or not isinstance(result.get('data'), list):
            raise ValueError('Atomic retrieval must return a data list')
        normalized = []
        seen = set()
        for hit in result['data']:
            if not isinstance(hit, dict) or not hit.get('id') or not isinstance(hit.get('content'), str):
                raise ValueError('Atomic retrieval returned malformed evidence')
            if hit['id'] in seen:
                continue
            score = hit.get('score', 0.0)
            if not isinstance(score, (int, float)) or not isfinite(float(score)):
                raise ValueError('Atomic retrieval returned a non-finite score')
            seen.add(hit['id'])
            normalized.append(dict(hit, score=float(score)))
        return {'data': normalized}
