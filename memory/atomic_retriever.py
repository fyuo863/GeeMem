"""Single-fact retrieval with no planning or language-model calls.

``AtomicRetriever`` is the lowest retrieval layer. It accepts one concrete
query, delegates dense/lexical/tag/reranker work to the configured vector
backend, and returns only original memory evidence. Planning and multi-hop
controllers can call it repeatedly.
"""

from dataclasses import dataclass, asdict
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
    memory_types: tuple[str, ...] = ()
    fallback: bool = True
    include_evidence: bool = False
    reference_time: int | None = None
    reference_timezone: str = 'UTC'

    def __post_init__(self):
        from .typed_sources import TYPES
        if not self.user_id.strip() or not self.query.strip():
            raise ValueError('User and query must be nonempty')
        if type(self.top_k) is not int or not 1 <= self.top_k <= 100:
            raise ValueError('top_k must be 1..100')
        if isinstance(self.memory_types, str) or any(t not in TYPES for t in self.memory_types):
            raise ValueError('Unsupported memory type')
        if type(self.fallback) is not bool or type(self.include_evidence) is not bool:
            raise ValueError('Expected boolean retrieval options')


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
        payload = SimpleNamespace(**asdict(request))
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
