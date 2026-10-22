import json
import logging
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from memory.aml_api import AMLAdd, create_app
from memory.request_diagnostics import diagnostic_stage, observe_request


def payload(rid='private-request'):
    return AMLAdd(user_id='private-user', request_id=rid, session_id='private-session',
                  messages=[{'role':'user','content':'private-original-text'}])


def test_safe_cause_status_and_stage_without_response_changes(caplog):
    class Backend:
        def add(self, p):
            with diagnostic_stage('write_classifier'):
                try:
                    r=httpx.Response(502, text='private-response-body', request=httpx.Request(
                        'POST','https://provider.test/private-key',headers={'Authorization':'private-token'}))
                    r.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    from memory.llm import LLMError
                    raise LLMError('private-exception-detail') from exc
    app=create_app(settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'private-token'},backend=Backend())
    with TestClient(app) as c:
        r=c.post('/add',json=payload().model_dump(),headers={'Authorization':'Bearer private-token'})
    assert r.status_code==502 and r.json()=={'detail':'Memory extraction failed'}
    assert 'write_classifier' in caplog.text and '"upstream_status": 502' in caplog.text
    assert 'private-' not in caplog.text and 'provider.test' not in caplog.text
    assert 'request_failed' in caplog.text and 'request_finished' in caplog.text


def test_validation_codes_and_concurrent_context_isolation(caplog):
    @observe_request('/add')
    def operation(p):
        if p.request_id.endswith('fail'):
            with diagnostic_stage('write_classifier'):
                try:raise ValueError('Invalid source indices')
                except ValueError as exc:raise HTTPException(502,'Memory write failed') from exc
        return p.request_id
    def run(i):
        p=payload(str(i)+('fail' if i%2 else 'ok'))
        try:return operation(p)
        except HTTPException:return None
    with ThreadPoolExecutor(max_workers=8) as pool:values=list(pool.map(run,range(16)))
    assert sum(v is not None for v in values)==8
    rows=[json.loads(r.message.split(' ',1)[1]) for r in caplog.records if r.message.startswith('request_finished ')]
    assert len(rows)==16 and len({r['request_hash'] for r in rows})==16
    assert all(r['stage']==('backend' if r['status']==200 else 'write_classifier') for r in rows)
    assert 'classifier_indices' in caplog.text and 'private-' not in caplog.text


def test_failed_write_resumes_after_backend_restart(tmp_path,caplog):
    import numpy as np
    from memory.vanilla import VanillaMemory
    from memory.write_gate import MemoryTypeSelector
    from memory.typed_sources import SourceBuilder
    from test_multilabel_judge import Fake, output

    class Embedder:
        identity='diagnostic-resume-test'
        calls=0
        def documents(self,texts):
            self.calls+=1
            return np.ones((len(texts),3))
    class FailingBuilder(SourceBuilder):
        def write(self,*args):raise ValueError('private-builder-error')
    path=tmp_path/'db';cfg={'RAG_MEMORY_DB':str(path)}
    e=Embedder()
    def backend(builders=None):
        return VanillaMemory(cfg,e,write_selector=MemoryTypeSelector(Fake(output({'rule':[0]}))),builders=builders)
    store=backend()
    from memory.route_writer import RouteWriter
    store.route_writer=RouteWriter(store,{'rule':FailingBuilder(path,'rule')})
    settings={'AML_AUTH_MODE':'bearer','AML_API_KEY':'private-token'}
    headers={'Authorization':'Bearer private-token'}
    with TestClient(create_app(settings=settings,backend=store)) as c:
        assert c.post('/add',headers=headers,json=payload().model_dump()).status_code==502
    store=backend()
    store.route_writer=RouteWriter(store)
    with TestClient(create_app(settings=settings,backend=store)) as c:
        for _ in range(2):assert c.post('/add',headers=headers,json=payload().model_dump()).status_code==200
    assert e.calls==1
    with store.connect() as db:
        assert db.execute('select count(*) from rag_memories').fetchone()[0]==1
        assert db.execute('select count(*) from rule_sources').fetchone()[0]==1
        assert db.execute('select status from rag_memory_queue').fetchone()[0]=='completed'
    assert 'typed_routes' in caplog.text and 'private-' not in caplog.text
