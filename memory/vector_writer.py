"""Vector ingestion only: chunking, embeddings, provenance and idempotent storage.

prepare performs external work without a database transaction; persist joins a
caller-owned transaction and never commits. write provides standalone ingestion.
No judgement, tags, queue or retrieval dependencies.
"""
from contextlib import closing
from dataclasses import dataclass
import hashlib
import json
import numpy as np
from .provenance import payload_digest, store_sources
from .retrieval import TOKEN, chunks
from .store import Conflict


@dataclass
class PreparedWrite:
    payload: object
    digest: str
    chunks: list
    vectors: np.ndarray


class VectorWriter:
    def __init__(self, embedder, connect, lock, *, size=320, overlap=40):
        if not 0 <= overlap < size:
            raise ValueError('Invalid chunk configuration')
        self.embedder, self.connect, self.lock = embedder, connect, lock
        self.size, self.overlap = size, overlap

    def initialize(self):
        """Initialize compatible vector/source tables without upper-level modules."""
        identity = json.dumps([self.embedder.identity, self.size, self.overlap])
        with self.lock, closing(self.connect()) as db, db:
            tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables and 'rag_meta' not in tables:
                raise ValueError('RAG requires a separate database')
            db.execute('PRAGMA journal_mode=WAL')
            db.executescript("""
                CREATE TABLE IF NOT EXISTS rag_meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rag_requests(user_id TEXT, request_id TEXT, digest TEXT NOT NULL,
                    PRIMARY KEY(user_id,request_id));
                CREATE TABLE IF NOT EXISTS rag_memories(id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL, request_id TEXT NOT NULL, message_index INTEGER NOT NULL,
                    chunk_index INTEGER NOT NULL, content TEXT NOT NULL, timestamp INTEGER,
                    vector BLOB NOT NULL, dimension INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS rag_user ON rag_memories(user_id);
            """)
            previous = db.execute("SELECT value FROM rag_meta WHERE key='identity'").fetchone()
            if previous and previous[0] != identity:
                raise ValueError('Embedding/chunk identity changed; use a new RAG_MEMORY_DB')
            from .provenance import initialize
            initialize(db)
            db.execute("INSERT OR IGNORE INTO rag_meta VALUES ('identity', ?)", (identity,))

    @staticmethod
    def vectors(value, count):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != count or arr.shape[1] == 0 or not np.isfinite(arr).all():
            raise ValueError('Invalid embedding matrix')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Zero embedding vector')
        return arr / norms

    def existing(self, db, payload):
        row = db.execute('SELECT digest FROM rag_requests WHERE user_id=? AND request_id=?',
                         (payload.user_id, payload.request_id)).fetchone()
        if row is not None and row[0] != payload_digest(payload):
            raise Conflict('request_id already used with a different payload')
        return row is not None

    def prepare(self, payload):
        payload = payload.model_copy(deep=True)
        pending = [(i, j, text, message.timestamp) for i, message in enumerate(payload.messages)
                   for j, text in enumerate(chunks(message.content, self.size, self.overlap))]
        if not pending:
            raise ValueError('No nonblank content to embed')
        with self.lock:
            matrix = self.vectors(self.embedder.documents([p[2] for p in pending]), len(pending))
        return PreparedWrite(payload, payload_digest(payload), pending, matrix)

    def persist(self, db, prepared):
        if not db.in_transaction:
            raise ValueError('VectorWriter.persist requires a caller-owned transaction')
        payload = prepared.payload
        if payload_digest(payload) != prepared.digest:
            raise ValueError('Prepared payload changed')
        if self.existing(db, payload):
            ids = [row[0] for row in db.execute(
                'SELECT id FROM rag_memories WHERE user_id=? AND request_id=? ORDER BY message_index,chunk_index',
                (payload.user_id, payload.request_id))]
            return dict(memory_ids=ids, deduplicated=True)
        matrix = self.vectors(prepared.vectors, len(prepared.chunks))
        dim = db.execute("SELECT value FROM rag_meta WHERE key='dimension'").fetchone()
        if dim and int(dim[0]) != matrix.shape[1]:
            raise ValueError('Embedding dimension changed')
        db.execute("INSERT OR IGNORE INTO rag_meta VALUES ('dimension', ?)", (str(matrix.shape[1]),))
        db.execute('INSERT INTO rag_requests VALUES (?,?,?)',
                   (payload.user_id, payload.request_id, prepared.digest))
        ids = []
        for (i, j, text, stamp), vector in zip(prepared.chunks, matrix):
            mid = hashlib.sha256(json.dumps([payload.user_id, payload.request_id, i, j]).encode()).hexdigest()
            ids.append(mid)
            db.execute('INSERT INTO rag_memories VALUES (?,?,?,?,?,?,?,?,?,?)',
                       (mid, payload.user_id, payload.session_id, payload.request_id, i, j, text, stamp,
                        vector.astype('<f4').tobytes(), len(vector)))
        store_sources(db, payload, chunks, TOKEN, self.size, self.overlap)
        return dict(memory_ids=ids, deduplicated=False)

    def write(self, payload):
        """Write to an initialized vector store without any upper-level processing."""
        with self.lock, closing(self.connect()) as db:
            if self.existing(db, payload):
                ids = [row[0] for row in db.execute(
                    'SELECT id FROM rag_memories WHERE user_id=? AND request_id=? ORDER BY message_index,chunk_index',
                    (payload.user_id, payload.request_id))]
                return dict(memory_ids=ids, deduplicated=True)
            prepared = self.prepare(payload)
            with db:
                db.execute('BEGIN IMMEDIATE')
                return self.persist(db, prepared)
