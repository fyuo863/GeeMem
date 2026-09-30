"""Independent implementation of the reference v0.6 hybrid retrieval recipe."""
from collections import Counter
from contextlib import closing
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from threading import RLock
import numpy as np
from .config import PROJECT_ROOT
from .store import Conflict

TOKEN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9_]+|[^\w\s]")


def chunks(text, size=320, overlap=40):
    if not 0 <= overlap < size:
        raise ValueError("Require 0 <= overlap < chunk size")
    spans = list(TOKEN.finditer(text))
    if len(spans) <= size:
        return [text] if text.strip() else []
    out = []
    for start in range(0, len(spans), size - overlap):
        end = min(start + size, len(spans))
        out.append(text[spans[start].start():spans[end-1].end()])
        if end == len(spans):
            break
    return out


def bm25(documents, query):
    def tokens(text):
        return [t.casefold() for t in TOKEN.findall(text) if any(c.isalnum() or c == '_' for c in t)]
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


class LocalEmbedder:
    def __init__(self, cfg):
        from sentence_transformers import SentenceTransformer
        path = Path(cfg.get('RAG_MODEL_PATH', 'data/models/bge-small-en-v1.5'))
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not (path / 'manifest.json').is_file():
            raise ValueError('Download the embedding model with scripts/download_rag_model.py first')
        manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        for name, expected in manifest['sha256'].items():
            file = (path / name).resolve()
            if not file.is_relative_to(path.resolve()):
                raise ValueError('Invalid model manifest path')
            with file.open('rb') as handle:
                if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                    raise ValueError('Model snapshot checksum mismatch')
        prefix = cfg.get('RAG_QUERY_PREFIX', 'Represent this sentence for searching relevant passages: ')
        self.identity = json.dumps([manifest, prefix], sort_keys=True)
        self.prefix = prefix
        self.model = SentenceTransformer(str(path), device=cfg.get('RAG_DEVICE', 'cpu'), local_files_only=True, trust_remote_code=False)

    def documents(self, texts):
        return self.model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    def queries(self, texts):
        return self.documents([self.prefix + t for t in texts])


class VanillaMemory:
    def __init__(self, cfg, embedder=None):
        self.embedder = embedder if embedder is not None else LocalEmbedder(cfg)
        self.size = int(cfg.get('RAG_CHUNK_TOKENS', '320'))
        self.overlap = int(cfg.get('RAG_CHUNK_OVERLAP', '40'))
        self.mode = cfg.get('RAG_RETRIEVAL_MODE', 'hybrid')
        self.rrf = int(cfg.get('RAG_RRF_K', '60'))
        self.weight = float(cfg.get('RAG_LEXICAL_WEIGHT', '0.5'))
        self.window = int(cfg.get('RAG_RESULT_WINDOW', '1'))
        self.seeds = int(cfg.get('RAG_RESULT_WINDOW_SEED_K', '20'))
        if not (0 <= self.overlap < self.size and self.rrf > 0 and math.isfinite(self.weight) and self.weight >= 0
                and self.window >= 0 and self.seeds > 0 and self.mode in ('hybrid', 'dense')):
            raise ValueError('Invalid RAG configuration')
        self.path = Path(cfg.get('RAG_MEMORY_DB', 'data/aml/vanilla.sqlite3'))
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        identity = json.dumps([self.embedder.identity, self.size, self.overlap])
        with closing(self.connect()) as db, db:
            # Never silently mix graph stores or incompatible vector stores.
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
            db.execute("INSERT OR IGNORE INTO rag_meta VALUES ('identity', ?)", (identity,))
            if db.execute("SELECT value FROM rag_meta WHERE key='identity'").fetchone()[0] != identity:
                raise ValueError('Embedding/chunk identity changed; use a new RAG_MEMORY_DB')

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    @staticmethod
    def vectors(value, count):
        arr = np.asarray(value, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] != count or arr.shape[1] == 0 or not np.isfinite(arr).all():
            raise ValueError('Invalid embedding matrix')
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError('Zero embedding vector')
        return arr / norms

    def add(self, payload):
        digest = hashlib.sha256(payload.model_dump_json().encode()).hexdigest()
        def exists(db):
            row = db.execute('SELECT digest FROM rag_requests WHERE user_id=? AND request_id=?',
                             (payload.user_id, payload.request_id)).fetchone()
            if row is not None and row[0] != digest:
                raise Conflict('request_id already used with a different payload')
            return row is not None
        with self.lock, closing(self.connect()) as db:
            if exists(db):
                return
            pending = [(i, j, text, message.timestamp) for i, message in enumerate(payload.messages)
                       for j, text in enumerate(chunks(message.content, self.size, self.overlap))]
            matrix = self.vectors(self.embedder.documents([p[2] for p in pending]), len(pending))
            with db:
                db.execute('BEGIN IMMEDIATE')
                if exists(db):
                    return
                dim = db.execute("SELECT value FROM rag_meta WHERE key='dimension'").fetchone()
                if dim and int(dim[0]) != matrix.shape[1]:
                    raise ValueError('Embedding dimension changed')
                db.execute("INSERT OR IGNORE INTO rag_meta VALUES ('dimension', ?)", (str(matrix.shape[1]),))
                db.execute('INSERT INTO rag_requests VALUES (?,?,?)', (payload.user_id, payload.request_id, digest))
                for (i, j, text, stamp), vector in zip(pending, matrix):
                    key = json.dumps([payload.user_id, payload.request_id, i, j])
                    mid = hashlib.sha256(key.encode()).hexdigest()
                    db.execute('INSERT INTO rag_memories VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (mid, payload.user_id, payload.session_id, payload.request_id, i, j, text, stamp,
                         vector.astype('<f4').tobytes(), len(vector)))

    def search(self, payload):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT * FROM rag_memories WHERE user_id=? ORDER BY rowid', (payload.user_id,)).fetchall()
        if not rows:
            return {'data': []}
        query = payload.query
        if payload.options:
            query += '\nOptions:\n' + '\n'.join(payload.options)
        with self.lock:
            vector = self.vectors(self.embedder.queries([query]), 1)[0]
        if any(row['dimension'] != len(vector) for row in rows):
            raise ValueError('Query dimension mismatch')
        matrix = np.stack([np.frombuffer(r['vector'], dtype='<f4') for r in rows])
        dense = matrix @ vector
        order = sorted(range(len(rows)), key=lambda i: (-float(dense[i]), i))
        scores = dense.astype(float)
        if self.mode == 'hybrid' and self.weight > 0:
            scores = np.zeros(len(rows))
            for rank, i in enumerate(order, 1):
                scores[i] = 1 / (self.rrf + rank)
            lexical = bm25([r['content'] for r in rows], query)
            for rank, i in enumerate(sorted((i for i in range(len(rows)) if lexical[i] > 0), key=lambda i: (-lexical[i], i)), 1):
                scores[i] += self.weight / (self.rrf + rank)
            order.sort(key=lambda i: (-scores[i], -float(dense[i]), i))
        selected = []
        seen = set()
        def include(i):
            if i not in seen and len(selected) < payload.top_k:
                selected.append(i)
                seen.add(i)
        if self.window:
            for i in order[:self.seeds]:
                include(i)
                for distance in range(1, self.window + 1):
                    for neighbor in (i-distance, i+distance):
                        if 0 <= neighbor < len(rows) and rows[neighbor]['session_id'] == rows[i]['session_id']:
                            include(neighbor)
        for i in order:
            include(i)
        hits = []
        for i in selected:
            row = rows[i]
            hit = dict(id=row['id'], content=row['content'], score=float(scores[i]))
            if row['timestamp'] is not None:
                hit['created_at'] = datetime.fromtimestamp(row['timestamp']/1000, timezone.utc).isoformat().replace('+00:00', 'Z')
            hits.append(hit)
        return {'data': hits}
