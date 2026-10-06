"""Independent implementation of the reference v0.6 hybrid retrieval recipe."""
from contextlib import closing
from datetime import datetime, timezone
import json
import math
import sqlite3
from threading import RLock
import numpy as np
from pathlib import Path
from .config import PROJECT_ROOT
from .embeddings import HTTPEmbedder, LocalEmbedder
from .retrieval import bm25, chunks
from .vector_writer import VectorWriter
from .store import Conflict


class VanillaMemory:
    def __init__(self, cfg, embedder=None, tagger=None, reranker=None, multihop_planner=None, write_selector=None):
        from .atomic_retriever import AtomicRetriever
        if multihop_planner is not None or cfg.get('RAG_MULTIHOP_MODE', 'off') != 'off':
            raise ValueError('Atomic base does not include a multihop planner')
        self.metadata_mode = cfg.get('RAG_METADATA_MODE', 'off')
        if self.metadata_mode not in ('off', 'on'):
            raise ValueError('Invalid metadata mode')
        if embedder is None:
            embedder = HTTPEmbedder(cfg) if cfg.get('RAG_EMBEDDING_API_URL') else LocalEmbedder(cfg)
        self.embedder = embedder
        gate_mode = cfg.get('RAG_WRITE_GATE_MODE', 'off')
        if gate_mode not in ('off', 'on'):
            raise ValueError('Invalid write gate mode')
        self.write_selector = write_selector
        if gate_mode == 'on' and self.write_selector is None:
            from .write_gate import MemoryValueSelector
            self.write_selector = MemoryValueSelector()
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
        self.tag_mode = cfg.get('RAG_TAG_MODE', 'off')
        self.tag_candidates = int(cfg.get('RAG_TAG_CANDIDATES', '400'))
        self.tag_weight = float(cfg.get('RAG_TAG_WEIGHT', '0.5'))
        self.tag_threshold = float(cfg.get('RAG_TAG_THRESHOLD', '0.5'))
        self.semantic_tags = self.tag_mode in ('semantic_filter', 'semantic_rank')
        if (self.tag_mode not in ('off', 'filter', 'semantic_filter', 'semantic_rank') or self.tag_candidates < 1
                or not math.isfinite(self.tag_weight) or self.tag_weight < 0
                or not math.isfinite(self.tag_threshold) or not -1 <= self.tag_threshold <= 1
                or (self.tag_mode == 'semantic_rank' and (self.mode != 'hybrid' or self.weight <= 0))):
            raise ValueError('Invalid tag configuration')
        self.tagger = tagger
        if self.tag_mode != 'off' and self.tagger is None:
            from .tags import RuleTagger
            self.tagger = RuleTagger()
        self.tag_vector_identity = json.dumps([getattr(self.tagger, 'identity', None), self.embedder.identity, 'sorted-space-join-v1'])
        self.reranker = reranker
        rerank_mode = cfg.get('RAG_RERANK_MODE', 'off')
        self.rerank_candidates = int(cfg.get('RAG_RERANK_CANDIDATES', '200'))
        self.rerank_context = int(cfg.get('RAG_RERANK_CONTEXT', '0'))
        if rerank_mode not in ('off', 'local') or self.rerank_candidates < 1 or not 0 <= self.rerank_context <= 2:
            raise ValueError('Invalid reranking configuration')
        if self.reranker is None and rerank_mode == 'local':
            from .rerank import HTTPReranker, LocalReranker
            self.reranker = HTTPReranker(cfg) if cfg.get('RAG_RERANK_API_URL') else LocalReranker(cfg)
        self.path = Path(cfg.get('RAG_MEMORY_DB', 'data/aml/vanilla.sqlite3'))
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.vector_writer = VectorWriter(self.embedder, self.connect, self.lock,
                                          size=self.size, overlap=self.overlap)
        self.vector_writer.initialize()
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS rag_tags(memory_id TEXT PRIMARY KEY, identity TEXT NOT NULL,
                    tags TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS rag_tag_vectors(memory_id TEXT PRIMARY KEY,
                    identity TEXT NOT NULL, vector BLOB NOT NULL, dimension INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS rag_write_decisions(
                    user_id TEXT NOT NULL, request_id TEXT NOT NULL, decision TEXT NOT NULL,
                    result TEXT NOT NULL, PRIMARY KEY(user_id,request_id));
                CREATE TABLE IF NOT EXISTS rag_memory_queue(
                    user_id TEXT NOT NULL, request_id TEXT NOT NULL, payload TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', PRIMARY KEY(user_id,request_id));
            """)
        from .search_service import SearchService
        self.atomic_retriever = AtomicRetriever(self)
        self.search_service = SearchService(self.atomic_retriever)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    vectors = staticmethod(VectorWriter.vectors)

    def add(self, payload):
        with self.lock, closing(self.connect()) as db:
            if self.vector_writer.existing(db, payload):
                return
            decision = self.write_selector.select(payload) if self.write_selector else None
            if decision is not None and decision.label not in ('valuable', 'vector_only'):
                raise ValueError('Unknown write route')
            enrich = decision is None or decision.label == 'valuable'
            prepared = self.vector_writer.prepare(payload)
            pending, matrix = prepared.chunks, prepared.vectors
            tags = self.tagger.extract([p[2] for p in pending]) if enrich and self.tag_mode != 'off' else None
            if tags is not None and len(tags) != len(pending):
                raise ValueError('Tag count mismatch')
            tag_vectors = {}
            if self.semantic_tags and tags is not None:
                from .tags import normalize_tags
                tagged_indices = [i for i, values in enumerate(tags) if normalize_tags(values)]
                if tagged_indices:
                    values = self.vectors(self.embedder.documents(
                        [' '.join(normalize_tags(tags[i])) for i in tagged_indices]), len(tagged_indices))
                    if values.shape[1] != matrix.shape[1]:
                        raise ValueError('Tag embedding dimension mismatch')
                    tag_vectors = dict(zip(tagged_indices, values))
            with db:
                db.execute('BEGIN IMMEDIATE')
                if self.vector_writer.existing(db, payload):
                    return
                written = self.vector_writer.persist(db, prepared)
                if decision is not None:
                    db.execute('INSERT INTO rag_write_decisions VALUES (?,?,?,?)',
                               (payload.user_id, payload.request_id, decision.label, decision.model_dump_json()))
                    if decision.label == 'valuable':
                        db.execute('INSERT INTO rag_memory_queue(user_id,request_id,payload) VALUES (?,?,?)',
                                   (payload.user_id, payload.request_id, payload.model_dump_json()))
                for index, mid in enumerate(written['memory_ids']):
                    if tags is not None:
                        from .tags import normalize_tags
                        db.execute('INSERT INTO rag_tags VALUES (?,?,?)',
                                   (mid, self.tagger.identity, json.dumps(normalize_tags(tags[index]))))
                    if index in tag_vectors:
                        tag_vector = tag_vectors[index]
                        db.execute('INSERT INTO rag_tag_vectors VALUES (?,?,?,?)',
                                   (mid, self.tag_vector_identity, tag_vector.astype('<f4').tobytes(), len(tag_vector)))


    def retrieval_text(self, row):
        if self.metadata_mode == 'on':
            from .provenance import metadata_text
            return metadata_text(row)
        return row['content']

    def score_candidates(self, query, rows, candidates):
        documents = []
        for i in candidates:
            text = self.retrieval_text(rows[i])
            if self.rerank_context:
                before = [self.retrieval_text(rows[j]) for j in range(max(0,i-self.rerank_context),i)
                          if rows[j]['session_id'] == rows[i]['session_id']]
                after = [self.retrieval_text(rows[j]) for j in range(i+1,min(len(rows),i+1+self.rerank_context))
                         if rows[j]['session_id'] == rows[i]['session_id']]
                text = 'Target message: ' + text + '\nPrevious context: ' + ' '.join(before) + '\nNext context: ' + ' '.join(after)
            documents.append(text)
        with self.lock:
            reranked = np.asarray(self.reranker.score(query, documents),dtype=float).reshape(-1)
        if len(reranked) != len(candidates) or not np.isfinite(reranked).all():
            raise ValueError('Invalid reranker scores')
        if self.target_mode == 'on':
            from .target_rerank import combine_target_scores
            with self.lock:
                target = self.reranker.score(query, [self.retrieval_text(rows[i]) for i in candidates])
            if self.fusion_qa == 'on':
                from .fusion_qa import fuse_scores
                reranked = fuse_scores(rows, candidates, reranked, target)
            else:
                reranked = combine_target_scores(reranked, target)
        return reranked

    def search(self, payload, *, trace=None):
        # SearchService owns orchestration; this backend owns retrieval details.
        return self.search_service.search(payload, trace=trace)

    def _search_direct(self, payload):
        with closing(self.connect()) as db:
            rows = db.execute('SELECT m.*, s.role, s.speaker, s.session_timestamp, s.source_id, s.source_index, s.char_start, s.char_end, t.tags, t.identity AS tag_identity, v.vector AS tag_vector, v.dimension AS tag_dimension, v.identity AS tag_vector_identity FROM rag_memories m '
                              'LEFT JOIN rag_sources s ON s.memory_id=m.id LEFT JOIN rag_tags t ON t.memory_id=m.id LEFT JOIN rag_tag_vectors v ON v.memory_id=m.id WHERE m.user_id=? ORDER BY m.rowid',
                              (payload.user_id,)).fetchall()
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
        if self.tag_mode == 'filter':
            from .tags import normalize_tags
            # Candidate options affect original retrieval only, not tag gating.
            query_tags = set(normalize_tags(self.tagger.extract([payload.query], query=True)[0]))
            pool = order[:max(payload.top_k, self.tag_candidates)]
            matching = [i for i in pool if rows[i]['tag_identity'] == self.tagger.identity
                        and query_tags.intersection(json.loads(rows[i]['tags']))]
            if matching:
                # Unknown/empty/old-version tags are never negative evidence.
                allowed = set(matching) | {i for i in pool if rows[i]['tag_identity'] != self.tagger.identity
                                            or not json.loads(rows[i]['tags'])}
                order = [i for i in pool if i in allowed]
            # No matches: original ranking is retained, preventing an empty result.
        if self.semantic_tags:
            from .tags import normalize_tags
            query_tags = normalize_tags(self.tagger.extract([payload.query], query=True)[0])
            pool = order[:max(payload.top_k, self.tag_candidates)]
            known = [i for i in pool if rows[i]['tag_vector_identity'] == self.tag_vector_identity]
            if query_tags and known:
                with self.lock:
                    query_vector = self.vectors(self.embedder.queries([' '.join(query_tags)]), 1)[0]
                if any(rows[i]['tag_dimension'] != len(query_vector) for i in known):
                    raise ValueError('Query tag dimension mismatch')
                keyword_matrix = np.stack([np.frombuffer(rows[i]['tag_vector'], dtype='<f4') for i in known])
                if not np.isfinite(keyword_matrix).all():
                    raise ValueError('Invalid stored tag vectors')
                similarity = dict(zip(known, keyword_matrix @ query_vector))
                if self.tag_mode == 'semantic_filter':
                    matching = {i for i in known if similarity[i] >= self.tag_threshold}
                    if matching:
                        unknown = set(pool) - set(known)
                        order = [i for i in pool if i in matching or i in unknown]
                elif self.tag_weight > 0:
                    # Only add a third RRF vote; never discard a candidate.
                    for rank, i in enumerate(sorted(known, key=lambda i: (-float(similarity[i]), i)), 1):
                        scores[i] += self.tag_weight / (self.rrf + rank)
                    order.sort(key=lambda i: (-scores[i], -float(dense[i]), i))
        if self.reranker is not None:
            candidates = order[:max(payload.top_k, self.rerank_candidates)]
            reranked = self.score_candidates(payload.query, rows, candidates)
            # Stable ties preserve the existing retrieval order.
            ranking = sorted(range(len(candidates)), key=lambda j: -reranked[j])
            scores = scores.copy()
            for j,i in enumerate(candidates):scores[i] = reranked[j]
            order = [candidates[j] for j in ranking]
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
