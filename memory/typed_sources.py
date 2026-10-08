"""Typed source indices. Reuse original text and vectors without fact generation."""
from contextlib import closing
import json
import sqlite3
from .provenance import source_id


TYPES = frozenset({"profile", "relationship", "rule", "event"})


class SourceBuilder:
    def __init__(self, db_path, memory_type):
        if memory_type not in TYPES: raise ValueError("Unsupported memory type")
        self.memory_type = memory_type
        self.table = memory_type + "_sources"
        self.db_path = str(db_path)
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute(f'''CREATE TABLE IF NOT EXISTS {self.table}(
                user_id TEXT NOT NULL, source_id TEXT NOT NULL,
                request_id TEXT NOT NULL, session_id TEXT NOT NULL,
                message_index INTEGER NOT NULL, context_source_ids TEXT NOT NULL,
                classification_score REAL,
                PRIMARY KEY(user_id,source_id))''')
            db.execute(f'CREATE INDEX IF NOT EXISTS {self.memory_type}_source_request ON {self.table}(user_id,request_id)')

    def write(self, payload, selected_indices, score=None):
        indices = sorted(set(selected_indices))
        if not indices or any(type(i) is not int or not 0 <= i < len(payload.messages) for i in indices):
            raise ValueError('Invalid typed source indices')
        if score is not None and not 0 <= score <= 1:
            raise ValueError('Invalid classification score')
        ids=[]
        with closing(sqlite3.connect(self.db_path)) as db, db:
            db.execute('BEGIN IMMEDIATE')
            for i in indices:
                sid=source_id(payload.user_id,payload.request_id,i)
                exists=db.execute('''SELECT 1 FROM rag_sources s JOIN rag_memories m ON m.id=s.memory_id
                    WHERE m.user_id=? AND m.request_id=? AND s.source_id=?''',
                    (payload.user_id,payload.request_id,sid)).fetchone()
                if exists is None:
                    raise ValueError('Typed source must be persisted before indexing')
                context=[source_id(payload.user_id,payload.request_id,j) for j in range(max(0,i-2),i)]
                db.execute(f'INSERT OR IGNORE INTO {self.table} VALUES (?,?,?,?,?,?,?)',
                    (payload.user_id,sid,payload.request_id,payload.session_id,i,json.dumps(context),score))
                ids.append(sid)
        return ids


class SourceRetriever:
    """Type selection adapter; all ranking belongs to AtomicRetriever."""
    def __init__(self, atomic_retriever, memory_type):
        if memory_type not in TYPES: raise ValueError('Unsupported memory type')
        if not hasattr(atomic_retriever, 'query'):
            raise TypeError('SourceRetriever requires an AtomicRetriever, not a database path')
        self.atomic_retriever = atomic_retriever
        self.memory_type = memory_type

    def search(self, user_id, query, *, limit=10, session_id=None, fallback=True):
        from .atomic_retriever import AtomicQuery
        return self.atomic_retriever.query(AtomicQuery(
            user_id=user_id, query=query, top_k=limit, session_id=session_id,
            memory_types=(self.memory_type,), fallback=fallback, include_evidence=True))['data']
