import json
import numpy as np
import pytest
from memory.vanilla import VanillaMemory
from memory.write_gate import MemoryTypeSelector
from memory.typed_sources import SourceBuilder
from memory.llm import LLMError
from memory.aml_api import AMLAdd
from test_multilabel_judge import Fake, output


def test_failed_route_resume_does_not_repeat_extraction_or_sources(tmp_path):
    class Embedder:
        identity='test'
        def documents(self,texts):return np.ones((len(texts),3))
    class Builder(SourceBuilder):
        calls=0
        fail=True
        def write(self,*args):
            self.calls+=1
            if self.fail: raise ValueError('temporary failure')
            return super().write(*args)
    path=tmp_path/'db'
    store=VanillaMemory({'RAG_MEMORY_DB':str(path)},Embedder(),write_selector=MemoryTypeSelector(Fake(output({'rule':[0]}))))
    b=Builder(path, 'rule')
    from memory.route_writer import RouteWriter
    store.route_writer=RouteWriter(store,{'rule':b})
    payload=AMLAdd(user_id='u',request_id='r',session_id='s',messages=[dict(role='user',content='Reports need conclusions first')])
    with pytest.raises(LLMError):store.add(payload)
    b.fail=False
    store.add(payload); store.add(payload)
    assert b.calls==2
    with store.connect() as db:
        route=db.execute('SELECT * FROM rag_memory_routes').fetchone()
        assert route['status']=='completed' and route['attempts']==2
        assert len(json.loads(route['record_ids']))==1
        assert db.execute('SELECT count(*) FROM rag_memories').fetchone()[0]==1
        assert db.execute('SELECT count(*) FROM rule_sources').fetchone()[0]==1
