import json
import numpy as np
import pytest
from memory.relationship import (RelationshipBuilder, RelationshipRetriever,
    RelationshipCandidate, RelationshipExtraction, entity_key)

class FakeLLM:
    def __init__(self, result): self.result=result; self.payloads=[]
    def complete(self, instruction,payload,schema): self.payloads.append(payload); return self.result

def messages():
    return [
      dict(message_index=0,source_id='src-0',source_kind='context',role='assistant',content='小王是谁？'),
      dict(message_index=1,source_id='src-1',source_kind='evidence',role='user',content='小王是我的朋友。'),
    ]

def test_builder_extracts_explicit_relation_and_rejects_unsupported_indices(tmp_path):
    good=RelationshipExtraction(relationships=[RelationshipCandidate(
        subject_name='小林',relation='朋友',object_name='小王',message_indices=[1],confidence=.95)])
    fake=FakeLLM(good);builder=RelationshipBuilder(str(tmp_path/'db'),fake)
    result=builder.extract(messages())
    assert result.relationships[0].relation=='friend'
    assert fake.payloads[0]['messages'][1]['message_index']==1
    bad=RelationshipExtraction(relationships=[RelationshipCandidate(
        subject_name='小林',relation='朋友',object_name='小王',message_indices=[9],confidence=.9)])
    with pytest.raises(ValueError): RelationshipBuilder(str(tmp_path/'bad'),FakeLLM(bad)).extract(messages())

def test_builder_deduplicates_and_preserves_evidence(tmp_path):
    extraction=RelationshipExtraction(relationships=[RelationshipCandidate(
      subject_name='小林',relation='朋友',object_name='小王',message_indices=[1],confidence=.8)])
    builder=RelationshipBuilder(str(tmp_path/'db'),FakeLLM(extraction))
    ids=builder.write('u',extraction,messages()); assert len(ids)==1
    extraction2=extraction.model_copy(update={'relationships':[extraction.relationships[0].model_copy(update={'confidence':.95})]})
    assert builder.write('u',extraction2,messages())==ids
    rows=RelationshipRetriever(str(tmp_path/'db')).find('u',subject='小林',relation='朋友')
    assert len(rows)==1 and rows[0].object_name=='小王'
    assert rows[0].confidence==pytest.approx(.95)
    assert rows[0].evidence[0]['source_id']=='src-1'

def test_entities_are_user_scoped_and_same_name_is_not_cross_user(tmp_path):
    extraction=RelationshipExtraction(relationships=[RelationshipCandidate(
      subject_name='小林',relation='导师',object_name='小王',message_indices=[1],confidence=.9)])
    builder=RelationshipBuilder(str(tmp_path/'db'),FakeLLM(extraction))
    builder.write('u1',extraction,messages());builder.write('u2',extraction,messages())
    with builder.connect() as db:
        assert db.execute('select count(*) from relationship_entities').fetchone()[0]==4

def test_retriever_supports_reverse_lookup_and_bounded_expansion(tmp_path):
    extraction=RelationshipExtraction(relationships=[
      RelationshipCandidate(subject_name='小林',relation='朋友',object_name='小王',message_indices=[1],confidence=.9),
      RelationshipCandidate(subject_name='小王',relation='导师',object_name='小李',message_indices=[1],confidence=.8)])
    builder=RelationshipBuilder(str(tmp_path/'db'),FakeLLM(extraction));builder.write('u',extraction,messages())
    retriever=RelationshipRetriever(str(tmp_path/'db'))
    assert retriever.find('u',object='小王')[0].subject_name=='小林'
    expanded=retriever.expand('u','小林',max_hops=2)
    assert {r.relation for r in expanded}=={'朋友','导师'}

def test_joint_event_does_not_become_relationship(tmp_path):
    extraction=RelationshipExtraction(relationships=[])
    builder=RelationshipBuilder(str(tmp_path/'db'),FakeLLM(extraction))
    assert builder.extract([dict(message_index=0,source_id='s',source_kind='evidence',content='我和小王一起看电影。')]).relationships==[]


