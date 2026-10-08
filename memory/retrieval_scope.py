"""Scope and provenance for the shared retrieval engine; no separate ranker."""
from contextlib import closing
import json

from .typed_sources import TYPES


class RetrievalScope:
    def __init__(self, backend, payload):
        self.backend = backend
        self.user_id = payload.user_id
        self.session_id = getattr(payload, 'session_id', None)
        self.types = tuple(dict.fromkeys(getattr(payload, 'memory_types', ())))
        if any(t not in TYPES for t in self.types):
            raise ValueError('Unsupported memory type')
        self.fallback = getattr(payload, 'fallback', True)
        self.include_evidence = getattr(payload, 'include_evidence', False)
        self.sources = {}
        with closing(backend.connect()) as db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            for kind in self.types:
                if kind + '_sources' not in tables:
                    continue
                for row in db.execute(f'SELECT * FROM {kind}_sources WHERE user_id=?', (self.user_id,)):
                    self.sources.setdefault(row['source_id'], {})[kind] = dict(row)

    def session_rows(self, rows):
        return [r for r in rows if self.session_id is None or r['session_id'] == self.session_id]

    def eligible(self, rows):
        return {i for i, r in enumerate(rows)
                if not self.types or self.fallback or r['source_id'] in self.sources}

    def rank_candidates(self, rows, order, scores, eligible, rrf):
        order = [i for i in order if i in eligible]
        if self.types and self.fallback:
            # A single type vote, even if several requested labels match.
            rank = 0
            for i in order:
                if rows[i]['source_id'] in self.sources:
                    rank += 1
                    scores[i] += .25 / (rrf + rank)
            order.sort(key=lambda i: -scores[i])
        return order

    def enrich(self, hit, row):
        if not self.include_evidence:
            return
        matches = self.sources.get(row['source_id'], {})
        context_ids = list(dict.fromkeys(sid for item in matches.values()
                                        for sid in json.loads(item['context_source_ids'])))
        context = []
        with closing(self.backend.connect()) as db:
            for sid in context_ids:
                context.extend(dict(r) for r in db.execute('''
                    SELECT m.id,m.content,s.source_id,s.speaker,m.timestamp
                    FROM rag_memories m JOIN rag_sources s ON s.memory_id=m.id
                    WHERE m.user_id=? AND m.session_id=? AND s.source_id=? ORDER BY m.chunk_index
                ''', (self.user_id, row['session_id'], sid)))
        scores = [r['classification_score'] for r in matches.values() if r['classification_score'] is not None]
        hit.update(source_id=row['source_id'], speaker=row['speaker'], role=row['role'],
                   session_id=row['session_id'], message_timestamp=row['timestamp'],
                   memory_types=list(matches), memory_type=next(iter(matches), 'fallback'),
                   classification_score=max(scores) if scores else None, context=context)
