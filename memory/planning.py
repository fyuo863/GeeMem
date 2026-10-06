"""Contracts for query planning implementations.

The current implementation lives in :mod:`memory.multihop`; this module keeps
the planner boundary independent from the HTTP client and retrieval backend.
"""

from typing import Any, Protocol


class QueryPlanner(Protocol):
    def route(self, question: str, options: list[str] | None, timeout: float) -> Any:
        """Choose direct, independent, or chained retrieval."""

    def review(self, question: str, options: list[str] | None, strategy: str,
               evidence: list[dict], history: list[str], timeout: float) -> Any:
        """Review evidence and propose grounded follow-up queries."""
