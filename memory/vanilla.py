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
    def __init__(self, cfg, embedder=None, tagger=None, reranker=None, multihop_planner=None, write_selector=None, builders=None, partition_selector=None):
        from .atomic_retriever import AtomicRetriever
        from .multihop import MultiHop
        self.multihop = MultiHop(cfg, multihop_planner)
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
            from .write_gate import MemoryTypeSelector
            self.write_selector = MemoryTypeSelector()
        self.size = int(cfg.get('RAG_CHUNK_TOKENS', '320'))
        self.overlap = int(cfg.get('RAG_CHUNK_OVERLAP', '40'))
        self.mode = cfg.get('RAG_RETRIEVAL_MODE', 'hybrid')
        self.rrf = int(cfg.get('RAG_RRF_K', '60'))
        self.weight = float(cfg.get('RAG_LEXICAL_WEIGHT', '0.5'))
        self.window = int(cfg.get('RAG_RESULT_WINDOW', '1'))
        self.temporal_mode = cfg.get('RAG_TEMPORAL_MODE', 'off')
        self.temporal_experiment = cfg.get('RAG_TEMPORAL_EXPERIMENT','off')
        if self.temporal_experiment not in ('off','absolute_query','window'):
            raise ValueError('Invalid temporal experiment')
        if self.temporal_mode not in ('off', 'on'):
            raise ValueError('Invalid temporal mode')
        self.temporal_fusion_weight = float(cfg.get('RAG_TEMPORAL_FUSION_WEIGHT', '0.10'))
        self.temporal_window_ratio = float(cfg.get('RAG_TEMPORAL_WINDOW_RATIO', '0.50'))
        self.temporal_context = int(cfg.get('RAG_TEMPORAL_CONTEXT', '1'))
        self.seeds = int(cfg.get('RAG_RESULT_WINDOW_SEED_K', '20'))
        if not (0 <= self.overlap < self.size and self.rrf > 0 and math.isfinite(self.weight) and self.weight >= 0
                and self.window >= 0 and self.seeds > 0 and self.mode in ('hybrid', 'dense')
                and 0 <= self.temporal_fusion_weight <= 1
                and 0 < self.temporal_window_ratio <= 1
                and 0 <= self.temporal_context <= 2):
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
        self.target_mode = cfg.get('RAG_TARGET_MODE', 'off')
        self.fusion_qa = cfg.get('RAG_FUSION_QA', 'off')
        self.rerank_selection = cfg.get('RAG_RERANK_SELECTION', 'direct')
        self.neighbor_penalty = float(cfg.get('RAG_RERANK_NEIGHBOR_PENALTY', '2'))
        if self.target_mode not in ('off', 'on') or self.fusion_qa not in ('off', 'on'):
            raise ValueError('Invalid target/fusion mode')
        if self.rerank_selection not in ('direct', 'context_support'):
            raise ValueError('Invalid rerank selection')
        rerank_mode = cfg.get('RAG_RERANK_MODE', 'off')
        self.rerank_candidates = int(cfg.get('RAG_RERANK_CANDIDATES', '200'))
        self.rerank_context = int(cfg.get('RAG_RERANK_CONTEXT', '0'))
        if rerank_mode not in ('off', 'local', 'onnx') or self.rerank_candidates < 1 or not 0 <= self.rerank_context <= 2:
            raise ValueError('Invalid reranking configuration')
        if self.reranker is None and rerank_mode == 'local':
            from .rerank import HTTPReranker, LocalReranker
            self.reranker = HTTPReranker(cfg) if cfg.get('RAG_RERANK_API_URL') else LocalReranker(cfg)
        if self.reranker is None and rerank_mode == 'onnx':
            from .rerank import ONNXReranker
            self.reranker = ONNXReranker(cfg)
        if self.rerank_selection == 'context_support' and (self.rerank_context != 1 or self.window != 0):
            raise ValueError('Context support requires reranker, context=1 and result window=0')
        self.path = Path(cfg.get('RAG_MEMORY_DB', 'data/aml/vanilla.sqlite3'))
        if not self.path.is_absolute():
            self.path = PROJECT_ROOT / self.path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        self.vector_writer = VectorWriter(self.embedder, self.connect, self.lock,
                                          size=self.size, overlap=self.overlap)
        self.vector_writer.initialize()
        self.temporal_index = None
        if self.temporal_mode == 'on':
            from .temporal_index import TemporalIndex
            self.temporal_index = TemporalIndex(self.connect, annotate=cfg.get('RAG_TIME_ANNOTATION_MODE','off') == 'on')
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
                CREATE TABLE IF NOT EXISTS rag_memory_routes(
                    user_id TEXT NOT NULL, request_id TEXT NOT NULL, memory_type TEXT NOT NULL,
                    message_indices TEXT NOT NULL, source_ids TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY(user_id,request_id,memory_type));
            """)
            # Additive migration: existing route records remain readable.
            db.execute('BEGIN IMMEDIATE')
            columns = {row[1] for row in db.execute('PRAGMA table_info(rag_memory_routes)')}
            for column in ('context_indices', 'context_source_ids', 'builder_messages'):
                if column not in columns:
                    db.execute(f"ALTER TABLE rag_memory_routes ADD COLUMN {column} TEXT NOT NULL DEFAULT '[]'")
        from .search_service import SearchService
        self.route_writer = None
        build_mode = cfg.get('RAG_BUILD_MODE', 'on' if gate_mode == 'on' else 'off')
        if build_mode not in ('on', 'off'):
            raise ValueError('Invalid build mode')
        if build_mode == 'on' or builders is not None:
            from .route_writer import RouteWriter
            self.route_writer = RouteWriter(self, builders)
        self.atomic_retriever = AtomicRetriever(self)
        if self.route_writer is not None:
            from .event import EventRetriever
            self.event_retriever = EventRetriever(self.atomic_retriever)
            from .typed_sources import SourceRetriever
            for memory_type in ('profile', 'relationship', 'rule'):
                setattr(self, memory_type + '_retriever', SourceRetriever(self.atomic_retriever, memory_type))
        from .retrieval import CallbackReranker
        from .partition_search import PartitionSearch
        self.partition_search = PartitionSearch(cfg, partition_selector)
        self.search_service = SearchService(
            self.atomic_retriever,
            partition_search=self.partition_search,
            multihop=self.multihop,
            reranker=CallbackReranker(self._rerank_for_multihop) if self.reranker is not None else None)

    def _rerank_for_multihop(self, query, documents):
        with self.lock:
            return self.reranker.score(query, documents)

    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        return db

    vectors = staticmethod(VectorWriter.vectors)

    def add(self, payload):
        # Retry committed requests too: failed routes must not be skipped by
        # the vector writer's request deduplication.
        with self.lock:
            self._add_sources(payload)
            if self.temporal_index is not None:
                self.temporal_index.write(payload)
            if self.route_writer is not None:
                self.route_writer.process(payload.user_id, payload.request_id)

    def _add_sources(self, payload):
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
                        for route in getattr(decision, 'routes', []):
                            db.execute('INSERT INTO rag_memory_routes '
                                       '(user_id,request_id,memory_type,message_indices,source_ids,'
                                       'context_indices,context_source_ids,builder_messages) VALUES (?,?,?,?,?,?,?,?)',
                                       (payload.user_id, payload.request_id, route.memory_type,
                                        json.dumps(route.message_indices), json.dumps(route.source_ids),
                                        json.dumps(route.context_indices), json.dumps(route.context_source_ids),
                                        json.dumps(route.builder_messages, ensure_ascii=False)))
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
        text = row['content']
        if self.metadata_mode == 'on':
            from .provenance import metadata_text
            text = metadata_text(row)
        mentions = dict(row).get('time_mentions', [])
        references = []
        for mention in mentions:
            if mention['start']:
                references.append(mention['text'] + ' = ' + mention['start'] + ' to ' + mention['end'])
        if references:
            text += '\nSource time mentions (not inferred event dates): ' + '; '.join(references)
        return text

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
        if self.rerank_selection == 'context_support':
            from .rerank import context_support_scores
            _, supported = context_support_scores(rows, candidates, reranked, self.neighbor_penalty)
            reranked = np.asarray([supported[i] for i in candidates], dtype=float)
        return reranked

    def search(self, payload, *, trace=None):
        # SearchService owns orchestration; this backend owns retrieval details.
        return self.search_service.search(payload, trace=trace)

    def _search_direct(self, payload):
        target = None
        if self.temporal_mode=='on' and self.temporal_experiment in ('absolute_query','window'):
            from .query_time import resolve_query_time, expanded_query
            from types import SimpleNamespace
            target=resolve_query_time(payload.query,getattr(payload,'reference_time',None),
                                      getattr(payload,'reference_timezone','UTC'))
            if target is not None:
                payload=SimpleNamespace(**(payload.model_dump() if hasattr(payload,'model_dump') else vars(payload)))
                payload.query=expanded_query(payload.query,target)
        from .retrieval_scope import RetrievalScope
        scope = RetrievalScope(self, payload)
        with closing(self.connect()) as db:
            rows = db.execute('SELECT m.*, s.role, s.speaker, s.session_timestamp, s.source_id, s.source_index, s.char_start, s.char_end, t.tags, t.identity AS tag_identity, v.vector AS tag_vector, v.dimension AS tag_dimension, v.identity AS tag_vector_identity FROM rag_memories m '
                              'LEFT JOIN rag_sources s ON s.memory_id=m.id LEFT JOIN rag_tags t ON t.memory_id=m.id LEFT JOIN rag_tag_vectors v ON v.memory_id=m.id WHERE m.user_id=? ORDER BY m.rowid',
                              (payload.user_id,)).fetchall()
        rows = scope.session_rows(rows)
        from .temporal import TemporalRanker
        if self.temporal_index is not None and (target is not None or TemporalRanker.QUERY.search(payload.query)):
            rows = self.temporal_index.enrich(payload.user_id, rows)
        eligible = scope.eligible(rows)
        if not eligible:
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
            lexical = bm25([self.retrieval_text(r) if dict(r).get('time_mentions') else r['content'] for r in rows], query)
            for rank, i in enumerate(sorted((i for i in range(len(rows)) if lexical[i] > 0), key=lambda i: (-lexical[i], i)), 1):
                scores[i] += self.weight / (self.rrf + rank)
            order.sort(key=lambda i: (-scores[i], -float(dense[i]), i))
        order = scope.rank_candidates(rows, order, scores, eligible, self.rrf)
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
        budget = max(payload.top_k, self.rerank_candidates)
        if self.temporal_experiment == 'window' and target is not None:
            from .temporal_window import window_candidates
            window_pool, window_trace = window_candidates(rows, order, target, budget)
            # Retain the tail for the existing partition reservation, if enabled.
            chosen = set(window_pool)
            order = window_pool + [i for i in order if i not in chosen]
            retrieval_trace = getattr(payload, 'retrieval_trace', None)
            if retrieval_trace is not None:
                retrieval_trace['temporal_window'] = window_trace
        if getattr(payload, 'dual_channel', False):
            from .partition_search import merge_candidates
            matched = {i for i in order if rows[i]['source_id'] in scope.sources}
            retrieval_trace = getattr(payload, 'retrieval_trace', None)
            if retrieval_trace is not None:
                typed = [i for i in order if i in matched][:budget//2]
                retrieval_trace['partition_channel_ids'] = [rows[i]['id'] for i in typed]
                retrieval_trace['global_channel_ids'] = [rows[i]['id'] for i in order if i not in set(typed)][:budget-len(typed)]
            order = merge_candidates(order, matched, budget)
        candidates = order[:budget]
        retrieval_trace = getattr(payload, 'retrieval_trace', None)
        if retrieval_trace is not None:
            retrieval_trace['candidate_ids'] = [rows[i]['id'] for i in candidates]
            retrieval_trace['candidate_count'] = len(candidates)
            retrieval_trace['typed_candidate_count'] = sum(rows[i]['source_id'] in scope.sources for i in candidates)
        if self.reranker is not None:
            reranked = self.score_candidates(payload.query, rows, candidates)
            # Stable ties preserve the existing retrieval order.
            ranking = sorted(range(len(candidates)), key=lambda j: -reranked[j])
            scores = scores.copy()
            for j,i in enumerate(candidates):scores[i] = reranked[j]
            order = [candidates[j] for j in ranking]
        if self.temporal_experiment == 'window' and target is not None:
            from .temporal_window import apply_temporal_fusion
            order, scores, fusion_trace = apply_temporal_fusion(
                rows, order, scores, target, self.temporal_fusion_weight,
                self.temporal_context)
            retrieval_trace = getattr(payload, 'retrieval_trace', None)
            if retrieval_trace is not None:
                retrieval_trace['temporal_fusion'] = fusion_trace
        if self.temporal_mode == 'on':
            from .temporal import TemporalRanker
            previous = list(order)
            order = TemporalRanker.order(payload.query, rows, order, scores)
            if order != previous:
                # Reuse sorted score slots; never replace scores with arbitrary
                # integers or feed temporal ranks into semantic fusion.
                scores = scores.copy()
                slots = sorted((float(scores[i]) for i in previous), reverse=True)
                for i, value in zip(order, slots):
                    scores[i] = value
        if self.temporal_experiment == 'window' and target is not None:
            from .temporal_window import soft_quota
            order, quota_trace = soft_quota(rows, order, target, payload.top_k,
                                            self.temporal_window_ratio)
            # Quota is a deterministic selection policy. Keep the public
            # scores monotonic after moving a bounded window candidate.
            slots = sorted((float(scores[i]) for i in order), reverse=True)
            scores = scores.copy()
            for i, value in zip(order, slots):
                scores[i] = value
            retrieval_trace = getattr(payload, 'retrieval_trace', None)
            if retrieval_trace is not None:
                retrieval_trace['temporal_quota'] = quota_trace
        selected = []
        seen = set()
        def include(i):
            if i in eligible and i not in seen and len(selected) < payload.top_k:
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
            scope.enrich(hit, row)
            hits.append(hit)
        return {'data': hits}
