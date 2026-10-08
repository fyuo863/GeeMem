"""Synchronous durable route processing after source persistence.

Selected source references are indexed without model-generated facts.
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
        from .event import EventBuilder
        self.event_builder = EventBuilder(backend.path)
        from .typed_sources import SourceBuilder, TYPES
        self.builders = {name: SourceBuilder(backend.path, name) for name in TYPES}
        self.builders['event'] = self.event_builder
        if builders is not None:
            self.builders.update(builders)
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
                if route['memory_type'] in self.builders:
                    from .aml_api import AMLAdd
                    if not queued: raise ValueError('Typed route requires original request')
                    payload=AMLAdd.model_validate_json(queued['payload'])
                    with closing(self.backend.connect()) as db:
                        decision=db.execute('SELECT result FROM rag_write_decisions WHERE user_id=? AND request_id=?',
                                            (user_id,request_id)).fetchone()
                    assessments=json.loads(decision[0]).get('classification',{}).get('assessments',[]) if decision else []
                    score=next((a['score'] for a in assessments if a['label']==route['memory_type']),None)
                    stage='index_sources'
                    ids=self.builders[route['memory_type']].write(payload,json.loads(route['message_indices']),score)
                    status='completed'
                    with closing(self.backend.connect()) as db,db:
                        db.execute('UPDATE rag_memory_routes SET extraction=NULL WHERE user_id=? AND request_id=? AND memory_type=?',key)
                    audit=[{'mode':'source_only','model_calls':0}]
                elif route['memory_type'] == 'other_memory':
                    status, ids = 'stored_only', []
                else:
                    raise ValueError('Unsupported memory route')
                with closing(self.backend.connect()) as db, db:
                    db.execute('UPDATE rag_memory_routes SET status=?,record_ids=?,elapsed_ms=?,audit=?,failure_stage=NULL WHERE user_id=? AND request_id=? AND memory_type=?',
                               (status,json.dumps(ids),(time.perf_counter()-started)*1000,json.dumps(audit),*key))
            except Exception as exc:
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
