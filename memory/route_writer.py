"""Synchronous durable route processing after source persistence.

Extraction is checkpointed before writing, so retries reuse identical candidates.
Builder writes are idempotent; completed routes never call the model again.
The backend lock serializes requests in this single-process deployment.
"""
from contextlib import closing
import json
import time

from .llm import LLMError


class RouteWriter:
    def __init__(self, backend, builders=None):
        self.backend = backend
        if builders is None:
            from .profile import ProfileBuilder, ProfileExtraction
            from .relationship import RelationshipBuilder, RelationshipExtraction
            from .rule import RuleBuilder, RuleExtraction
            from .event import EventBuilder, EventExtraction
            builders = {name: (builder(str(backend.path)), schema) for name,builder,schema in [
                ('profile',ProfileBuilder,ProfileExtraction),
                ('relationship',RelationshipBuilder,RelationshipExtraction),
                ('rule',RuleBuilder,RuleExtraction),('event',EventBuilder,EventExtraction)]}
        self.builders = builders
        with closing(backend.connect()) as db, db:
            cols = {r[1] for r in db.execute('PRAGMA table_info(rag_memory_routes)')}
            for name,typ in [('extraction','TEXT'),('record_ids',"TEXT NOT NULL DEFAULT '[]'"),
                             ('attempts','INTEGER NOT NULL DEFAULT 0'),('error','TEXT'),
                             ('elapsed_ms','REAL')]:
                if name not in cols:
                    db.execute(f'ALTER TABLE rag_memory_routes ADD COLUMN {name} {typ}')

    def process(self, user_id, request_id):
        with closing(self.backend.connect()) as db:
            routes = db.execute('SELECT * FROM rag_memory_routes WHERE user_id=? AND request_id=?',
                                (user_id,request_id)).fetchall()
        failures = []
        for route in routes:
            if route['status'] in ('completed','stored_only'):
                continue
            key = (user_id,request_id,route['memory_type'])
            started = time.perf_counter()
            with closing(self.backend.connect()) as db, db:
                db.execute("UPDATE rag_memory_routes SET status='processing',attempts=attempts+1,error=NULL WHERE user_id=? AND request_id=? AND memory_type=?",key)
            try:
                if route['memory_type'] == 'other_memory':
                    status, ids = 'stored_only', []
                else:
                    builder,schema = self.builders[route['memory_type']]
                    messages = json.loads(route['builder_messages'])
                    if not messages:
                        raise ValueError('Route has no source messages')
                    if route['extraction']:
                        extraction = schema.model_validate_json(route['extraction'])
                    else:
                        extraction = builder.extract(messages)
                        with closing(self.backend.connect()) as db, db:
                            db.execute('UPDATE rag_memory_routes SET extraction=? WHERE user_id=? AND request_id=? AND memory_type=?',
                                       (extraction.model_dump_json(),*key))
                    ids = builder.write(user_id,extraction,messages)
                    status = 'completed'
                with closing(self.backend.connect()) as db, db:
                    db.execute('UPDATE rag_memory_routes SET status=?,record_ids=?,elapsed_ms=? WHERE user_id=? AND request_id=? AND memory_type=?',
                               (status,json.dumps(ids),(time.perf_counter()-started)*1000,*key))
            except Exception as exc:
                failures.append(type(exc).__name__)
                with closing(self.backend.connect()) as db, db:
                    db.execute("UPDATE rag_memory_routes SET status='failed',error=?,elapsed_ms=? WHERE user_id=? AND request_id=? AND memory_type=?",
                               (type(exc).__name__,(time.perf_counter()-started)*1000,*key))
        with closing(self.backend.connect()) as db, db:
            db.execute('UPDATE rag_memory_queue SET status=? WHERE user_id=? AND request_id=?',
                       ('failed' if failures else 'completed',user_id,request_id))
        if failures:
            raise LLMError('Memory routes failed; retry the same request to resume')
