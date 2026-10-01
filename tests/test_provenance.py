import hashlib
import json
import sqlite3
import pytest
from memory.aml_api import AMLAdd, AMLSearch
from memory.provenance import metadata_text, payload_digest
from memory.vanilla import VanillaMemory
from test_vanilla import Embedder


def test_sources_offsets_idempotency_and_legacy_digest(tmp_path):
    payload = AMLAdd(user_id='u', session_id='s', request_id='r', messages=[dict(role='user', content='x ' * 20)])
    old = dict(request_id='r')  # Construct old serialized payload in its original field order.
    old = payload.model_dump(exclude={'session_timestamp', 'messages'})
    old['messages'] = [dict(role='user', content='x ' * 20, timestamp=None)]
    assert payload_digest(payload) == hashlib.sha256(json.dumps(old, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    cfg = dict(RAG_MEMORY_DB=str(tmp_path/'memory.db'), RAG_CHUNK_TOKENS='5', RAG_CHUNK_OVERLAP='2')
    store = VanillaMemory(cfg, Embedder())
    store.add(payload); store.add(payload)
    with store.connect() as db:
        rows = db.execute('SELECT s.*,m.content FROM rag_sources s JOIN rag_memories m ON s.memory_id=m.id ORDER BY m.rowid').fetchall()
    assert [r['char_start'] for r in rows] == [0,6,12,18,24,30]
    assert len({r['source_id'] for r in rows}) == 1
    assert all(payload.messages[0].content[r['char_start']:r['char_end']] == r['content'] for r in rows)
    assert all(r['speaker'] is None for r in rows)
    # Simulate an old DB with no provenance table and reopen it safely.
    with store.connect() as db:
        db.execute('DROP TABLE rag_sources'); db.commit()
    reopened = VanillaMemory(cfg, Embedder()); reopened.add(payload)
    with reopened.connect() as db:
        legacy = db.execute('SELECT * FROM rag_sources').fetchall()
    assert len(legacy) == len(rows)
    assert all(r['char_start'] is None and r['speaker'] is None for r in legacy)


def test_metadata_is_index_only_and_conflicts_are_detected(tmp_path):
    from memory.store import Conflict
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'memory.db'), RAG_METADATA_MODE='on'), Embedder())
    p = AMLAdd(user_id='u', session_id='s', request_id='r', session_timestamp=1681776000000,
               messages=[dict(role='user', speaker='John', content='We attended a convention last month.')])
    store.add(p)
    with store.connect() as db:
        row = db.execute('SELECT m.*,s.speaker,s.session_timestamp FROM rag_memories m JOIN rag_sources s ON m.id=s.memory_id').fetchone()
    view = metadata_text(row)
    assert 'John said' in view and 'March 2023' in view
    result = store.search(AMLSearch(user_id='u', query='convention', top_k=10))['data']
    assert result[0]['content'] == p.messages[0].content
    assert store.search(AMLSearch(user_id='other', query='convention', top_k=10)) == {'data': []}
    with pytest.raises(Conflict):
        store.add(p.model_copy(update={'session_timestamp': 1681862400000}))
    assert metadata_text(dict(content='unknown', timestamp=None)) == 'unknown'


def test_source_indices_continue_across_requests(tmp_path):
    store = VanillaMemory(dict(RAG_MEMORY_DB=str(tmp_path/'memory.db')), Embedder())
    for rid, speaker in [('1','Alice'),('2','Bob')]:
        store.add(AMLAdd(user_id='u',session_id='s',request_id=rid,messages=[dict(role='user',speaker=speaker,content='same')]))
    with store.connect() as db:
        rows = db.execute('SELECT * FROM rag_sources ORDER BY source_index').fetchall()
    assert [r['source_index'] for r in rows] == [0,1]
    assert [r['speaker'] for r in rows] == ['Alice','Bob']
    assert rows[0]['source_id'] != rows[1]['source_id']
