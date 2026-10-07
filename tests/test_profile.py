import pytest
from memory.profile import ProfileBuilder,ProfileRetriever,ProfileCandidate,ProfileExtraction
from memory.relationship import RelationshipBuilder,RelationshipCandidate,RelationshipExtraction
class Fake:
 def __init__(self,r):self.r=r
 def complete(self,*a):return self.r
def msgs():return [dict(message_index=0,source_id='s0',source_kind='evidence',content='我的朋友小王是医生。')]
def test_profile_uses_shared_entity_with_relationship(tmp_path):
 p=str(tmp_path/'db')
 rel=RelationshipBuilder(p,Fake(RelationshipExtraction(relationships=[RelationshipCandidate(subject_name='我',relation='friend',object_name='小王',message_indices=[0],confidence=.9)])))
 rel.write('u',rel.extract(msgs()),msgs())
 prof=ProfileBuilder(p,Fake(ProfileExtraction(facts=[ProfileCandidate(subject_name='小王',attribute='occupation',value='医生',message_indices=[0],confidence=.95)])))
 prof.write('u',prof.extract(msgs()),msgs())
 with prof.connect() as db:
  assert db.execute('select count(*) from relationship_entities where user_id=?',('u',)).fetchone()[0]==2
 row=ProfileRetriever(p).find('u',subject='小王',attribute='occupation')[0]
 assert row.value=='医生' and row.evidence[0]['source_id']=='s0'
 with rel.connect() as db:
  shared=db.execute("select id from relationship_entities where user_id=? and name=?",('u','小王')).fetchone()[0]
 assert row.entity_id==shared

def test_profile_deduplicates_and_keeps_uncertainty(tmp_path):
 p=str(tmp_path/'db'); b=ProfileBuilder(p)
 ex=ProfileExtraction(facts=[ProfileCandidate(subject_name='我',attribute='居住地',value='上海',certainty='planned',message_indices=[0],confidence=.7)])
 m=[dict(message_index=0,source_id='s',source_kind='evidence',content='我可能搬到上海。')]
 assert len(b.write('u',ex,m))==1; assert len(b.write('u',ex,m))==1
 row=ProfileRetriever(p).find('u',subject='我')[0]
 assert row.certainty=='planned' and row.status=='active'

def test_profile_rejects_invalid_source_index(tmp_path):
 b=ProfileBuilder(str(tmp_path/'db'),Fake(ProfileExtraction(facts=[ProfileCandidate(subject_name='我',attribute='喜好',value='跑步',message_indices=[4],confidence=.8)])))
 with pytest.raises(ValueError): b.extract(msgs())

