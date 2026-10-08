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
            from .build_audit import AuditedLLM
            from .profile import ProfileBuilder, ProfileExtraction
            from .relationship import RelationshipBuilder, RelationshipExtraction
            from .rule import RuleBuilder, RuleExtraction
            from .event import EventBuilder, EventExtraction
            builders = {name: (builder(str(backend.path), llm=AuditedLLM()), schema) for name,builder,schema in [
                ('profile',ProfileBuilder,ProfileExtraction),
                ('relationship',RelationshipBuilder,RelationshipExtraction),
                ('rule',RuleBuilder,RuleExtraction),('event',EventBuilder,EventExtraction)]}
        self.builders = builders
        with closing(backend.connect()) as db, db:
            cols = {r[1] for r in db.execute('PRAGMA table_info(rag_memory_routes)')}
            for name,typ in [('extraction','TEXT'),('record_ids',"TEXT NOT NULL DEFAULT '[]'"),
                             ('attempts','INTEGER NOT NULL DEFAULT 0'),('error','TEXT'),
                             ('elapsed_ms','REAL'), ('audit','TEXT'), ('failure_stage','TEXT')]:
                if name not in cols:
                    db.execute(f'ALTER TABLE rag_memory_routes ADD COLUMN {name} {typ}')

    def process(self, user_id, request_id):
        with closing(self.backend.connect()) as db:
            routes = db.execute('SELECT * FROM rag_memory_routes WHERE user_id=? AND request_id=?',
                                (user_id,request_id)).fetchall()
            queued = db.execute('SELECT payload FROM rag_memory_queue WHERE user_id=? AND request_id=?', (user_id,request_id)).fetchone()
        failures = []
        for route in routes:
            if route['status'] in ('completed','stored_only'):
                continue
            key = (user_id,request_id,route['memory_type'])
            started = time.perf_counter()
            stage = 'prepare'
            audit = []
            with closing(self.backend.connect()) as db, db:
                db.execute("UPDATE rag_memory_routes SET status='processing',attempts=attempts+1,error=NULL WHERE user_id=? AND request_id=? AND memory_type=?",key)
            try:
                if route['memory_type'] == 'other_memory':
                    status, ids = 'stored_only', []
                else:
                    builder,schema = self.builders[route['memory_type']]
                    messages = json.loads(route['builder_messages'])
                    if queued:
                        from .provenance import source_id
                        payload = json.loads(queued['payload'])
                        messages = [dict(m, message_index=i, source_id=source_id(user_id,request_id,i),
                                         source_kind='evidence') for i,m in enumerate(payload['messages'])]
                    if not messages:
                        raise ValueError('Route has no source messages')
                    if route['extraction']:
                        extraction = schema.model_validate_json(route['extraction'])
                    else:
                        stage = 'extract'
                        field = next(iter(schema.model_fields))
                        items = []
                        if hasattr(builder.llm, 'audit'): builder.llm.audit = []
                        for start in range(0,len(messages),20):
                            window = [dict(m,source_kind='context' if m['message_index']<start else 'evidence')
                                      for m in messages[max(0,start-2):start+20]]
                            part = builder.extract(window)
                            items.extend(x for x in getattr(part,field) if any(i>=start for i in x.message_indices))
                        extraction = schema.model_validate({field:items})
                        audit = getattr(builder.llm,'audit',[])
                        with closing(self.backend.connect()) as db, db:
                            db.execute('UPDATE rag_memory_routes SET extraction=? WHERE user_id=? AND request_id=? AND memory_type=?',
                                       (extraction.model_dump_json(),*key))
                    stage = 'write'
                    ids = builder.write(user_id,extraction,messages)
                    status = 'completed'
                with closing(self.backend.connect()) as db, db:
                    db.execute('UPDATE rag_memory_routes SET status=?,record_ids=?,elapsed_ms=?,audit=?,failure_stage=NULL WHERE user_id=? AND request_id=? AND memory_type=?',
                               (status,json.dumps(ids),(time.perf_counter()-started)*1000,json.dumps(audit),*key))
            except Exception as exc:
                if route['memory_type'] in self.builders:
                    audit = getattr(self.builders[route['memory_type']][0].llm,'audit',audit)
                failures.append(type(exc).__name__)
                with closing(self.backend.connect()) as db, db:
                    # Do not retain arbitrary exception strings that may contain credentials.
                    detail = type(exc).__name__ + (': '+str(exc)[:300] if type(exc) is ValueError else '')
                    db.execute("UPDATE rag_memory_routes SET status='failed',error=?,elapsed_ms=?,failure_stage=?,audit=? WHERE user_id=? AND request_id=? AND memory_type=?",
                               (detail,(time.perf_counter()-started)*1000,stage,json.dumps(audit),*key))
        with closing(self.backend.connect()) as db, db:
            db.execute('UPDATE rag_memory_queue SET status=? WHERE user_id=? AND request_id=?',
                       ('failed' if failures else 'completed',user_id,request_id))
        if failures:
            raise LLMError('Memory routes failed; retry the same request to resume')
