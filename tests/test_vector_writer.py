from contextlib import closing
from concurrent.futures import ThreadPoolExecutor
from threading import RLock
import sqlite3
import numpy as np
import pytest
from memory.aml_api import AMLAdd
from memory.vector_writer import VectorWriter
from memory.store import Conflict

class Embedder:
    identity='writer-test'
    calls=0
    def documents(self,texts):
        self.calls+=1
        return np.tile([3.,4.],(len(texts),1))

def setup_writer(path):
    def connect():
        db=sqlite3.connect(path,timeout=30); db.row_factory=sqlite3.Row; return db
    writer=VectorWriter(Embedder(),connect,RLock(),size=4,overlap=1)
    writer.initialize()
    return writer

def request():
    return AMLAdd(user_id='u',request_id='r',session_id='s',messages=[
        dict(role='user',speaker='Alice',content='one two three four five six seven',timestamp=1704067200000)])

def test_standalone_writer_source_positions_and_retry(tmp_path):
    writer=setup_writer(tmp_path/'db');payload=request()
    result=writer.write(payload)
    assert len(result['memory_ids'])==2 and not result['deduplicated']
    again=writer.write(payload)
    assert again==dict(memory_ids=result['memory_ids'],deduplicated=True)
    assert writer.embedder.calls==1
    with closing(writer.connect()) as db:
        rows=db.execute('SELECT m.*,s.* FROM rag_memories m JOIN rag_sources s ON s.memory_id=m.id ORDER BY chunk_index').fetchall()
        for row in rows:
            assert payload.messages[0].content[row['char_start']:row['char_end']]==row['content']
            assert row['speaker']=='Alice' and row['role']=='user'
            assert np.linalg.norm(np.frombuffer(row['vector'],dtype='<f4'))==pytest.approx(1)
    with pytest.raises(Conflict): writer.write(payload.model_copy(update={'session_id':'other'}))

def test_persist_never_commits_callers_transaction(tmp_path):
    writer=setup_writer(tmp_path/'db');prepared=writer.prepare(request())
    with closing(writer.connect()) as db:
        with pytest.raises(ValueError): writer.persist(db,prepared)
        with pytest.raises(RuntimeError), db:
            db.execute('BEGIN IMMEDIATE')
            writer.persist(db,prepared)
            raise RuntimeError('next stage failed')
        for table in ['rag_requests','rag_sources','rag_memories']:
            assert db.execute('SELECT count(*) FROM '+table).fetchone()[0]==0
    assert not writer.write(request())['deduplicated']

def test_concurrent_writer_instances_do_not_duplicate(tmp_path):
    path=tmp_path/'db'; a=setup_writer(path); b=setup_writer(path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda w:w.write(request()),[a,b]))
    assert sorted(r['deduplicated'] for r in results)==[False,True]
    assert results[0]['memory_ids']==results[1]['memory_ids']

def test_dimension_mismatch_rolls_back(tmp_path):
    writer=setup_writer(tmp_path/'db');writer.write(request())
    writer.embedder.documents=lambda texts:np.ones((len(texts),3))
    with pytest.raises(ValueError,match='dimension'):
        writer.write(request().model_copy(update={'request_id':'new'}))
    with closing(writer.connect()) as db:
        assert db.execute('SELECT count(*) FROM rag_requests').fetchone()[0]==1
