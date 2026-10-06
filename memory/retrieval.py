"""Reusable retrieval primitives and dependency boundaries.

The concrete backend still owns its database access, while these interfaces
make query planning, candidate retrieval, and ranking replaceable components.
"""

from collections import Counter
from dataclasses import dataclass, field
import math
import re
from typing import Any, Callable, Protocol, Sequence

import numpy as np


TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]")


def chunks(text, size=320, overlap=40):
    """Split a message into stable, overlapping retrieval passages."""
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    spans = list(TOKEN.finditer(text))
    if len(spans) <= size:
        return [text] if text.strip() else []
    out = []
    for start in range(0, len(spans), size - overlap):
        end = min(start + size, len(spans))
        out.append(text[spans[start].start():spans[end - 1].end()])
        if end == len(spans):
            break
    return out


def bm25(documents, query):
    """Compute the lexical component of the hybrid candidate score."""
    def tokens(text):
        return [t.casefold() for t in TOKEN.findall(text)
                if any(c.isalnum() or c == '_' for c in t)]
    docs = [Counter(tokens(d)) for d in documents]
    terms = set(tokens(query))
    n = len(docs)
    average = sum(sum(d.values()) for d in docs) / max(n, 1) or 1
    frequency = {t: sum(t in d for d in docs) for t in terms}
    scores = np.zeros(n)
    for i, d in enumerate(docs):
        norm = 1.5 * (0.25 + 0.75 * sum(d.values()) / average)
        for t in terms.intersection(d):
            idf = math.log(1 + (n - frequency[t] + 0.5) / (frequency[t] + 0.5))
            scores[i] += idf * d[t] * 2.5 / (d[t] + norm)
    return scores


class Retriever(Protocol):
    def retrieve(self, payload: Any) -> dict:
        """Return a response containing a ``data`` list of memory hits."""


class Reranker(Protocol):
    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        """Score documents for a query in the same order as the input."""


@dataclass(frozen=True)
class RetrievalQuery:
    user_id: str
    query: str
    top_k: int
    options: tuple[str, ...] = ()


@dataclass
class CandidateSet:
    """A named candidate collection passed between retrieval stages."""

    hits: list[dict] = field(default_factory=list)
    routes: list[list[str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class CallbackRetriever:
    """Adapter for legacy backends while they are migrated to ``Retriever``."""

    def __init__(self, callback: Callable[[Any], dict]):
        self._callback = callback

    def retrieve(self, payload: Any) -> dict:
        return self._callback(payload)


class CallbackReranker:
    """Adapter that gives planners a stable reranking dependency."""

    def __init__(self, callback: Callable[[str, Sequence[str]], Sequence[float]]):
        self._callback = callback

    def score(self, query: str, documents: Sequence[str]) -> Sequence[float]:
        return self._callback(query, documents)
