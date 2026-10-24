import pytest
from memory.selector import MultiLabelJudge, indexed_result_schema
from memory.write_gate import MemoryTypeSelector, MEMORY_TYPE_CONFIG
from memory.aml_api import AMLAdd
from memory.provenance import source_id
from memory.llm import strict_json_schema
from test_multilabel_judge import Fake, output


@pytest.mark.parametrize('count',[1,2,10,200])
def test_schema_constrains_original_indices_and_strict_types(count):
    schema=indexed_result_schema(count)
    raw=output({'profile':[count-1]}).model_dump()
    assert schema.model_validate(raw).assessments[0].message_indices==[count-1]
    for invalid in [-1,count,True,'0',0.5]:
        raw['assessments'][0]['message_indices']=[invalid]
        with pytest.raises(ValueError):schema.model_validate(raw)
    wire=strict_json_schema(schema.model_json_schema())
    item=wire['$defs']['IndexedLabelAssessment']['properties']['message_indices']['items']
    assert item['enum']==list(range(count)) and item['type']=='integer'
    assert wire['additionalProperties'] is False
    assert indexed_result_schema(count) is schema


def test_duplicate_indices_normalize_without_another_call(caplog):
    llm=Fake(output({'profile':[1,0,1,0]}))
    p=AMLAdd(user_id='u',session_id='s',request_id='r',messages=[
        {'role':'user','content':'first'},{'role':'user','content':'second'}])
    r=MemoryTypeSelector(llm).select(p)
    assert llm.calls==1
    assert r.routes[0].message_indices==[0,1]
    assert r.routes[0].source_ids==[source_id('u','r',i) for i in (0,1)]
    assert r.version=='memory-types-v3'
    assert 'reason=duplicates' in caplog.text
    assert 'first' not in caplog.text and 'second' not in caplog.text


def test_out_of_range_is_not_clamped_or_dropped(monkeypatch,caplog):
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    llm=Fake(output({'profile':[0,99]}))
    with pytest.raises(ValueError,match='Invalid source indices'):
        MultiLabelJudge(MEMORY_TYPE_CONFIG,llm).judge([{'content':'private-original'}])
    assert llm.calls==3
    assert 'reason=out_of_range' in caplog.text and 'private-original' not in caplog.text


def test_repair_prompt_and_schema_are_request_local(monkeypatch):
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    class Repair:
        calls=0
        def complete(self,instruction,payload,schema):
            self.calls+=1
            assert '0 through 1' in instruction
            assert schema is indexed_result_schema(2)
            if self.calls==1:return output({'profile':[2]})
            assert 'Previous attempt failed' in instruction
            return output({'profile':[1,1]})
    llm=Repair()
    r=MultiLabelJudge(MEMORY_TYPE_CONFIG,llm).judge([{'content':'one'},{'content':'two'}])
    assert llm.calls==2 and r.assessments[0].message_indices==[1]


def test_existing_receipt_skips_new_classifier_after_reopen(tmp_path):
    import numpy as np
    from memory.vanilla import VanillaMemory
    class Embedder:
        identity='resume-index-repair'
        def documents(self,texts):return np.ones((len(texts),3))
    class MustNotClassify:
        def select(self,p):pytest.fail('Completed receipt must bypass reclassification')
    cfg={'RAG_MEMORY_DB':str(tmp_path/'db')}
    p=AMLAdd(user_id='u',request_id='r',session_id='s',messages=[{'role':'user','content':'synthetic source'}])
    old=VanillaMemory(cfg,Embedder(),write_selector=MemoryTypeSelector(Fake(output({'profile':[0]}))))
    old.add(p)
    with old.connect() as db:
        receipt=tuple(db.execute('select * from rag_requests').fetchone())
        ids=[r[0] for r in db.execute('select id from rag_memories')]
        decision=db.execute('select result from rag_write_decisions').fetchone()[0]
    reopened=VanillaMemory(cfg,Embedder(),write_selector=MustNotClassify())
    reopened.add(p)
    with reopened.connect() as db:
        assert tuple(db.execute('select * from rag_requests').fetchone())==receipt
        assert [r[0] for r in db.execute('select id from rag_memories')]==ids
        assert db.execute('select result from rag_write_decisions').fetchone()[0]==decision
