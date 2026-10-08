import tempfile
from memory.event import *
def test_relative_time_and_evidence():
 p=tempfile.mktemp(); m=[dict(message_index=0,source_id='s',source_kind='evidence',content='小林去年参加杭州马拉松',timestamp=1704067200000)]
 e=EventCandidate(subject_name='小林',event_type='比赛',description='杭州马拉松',time_expression='去年',reference_timestamp=1704067200000,time_precision='relative',message_indices=[0],confidence=.9)
 b=EventBuilder(p); b.write('u',EventExtraction(events=[e]),m); row=EventRetriever(p).find('u',subject='小林')[0]
 assert row['event_time_start']=='2023' and row['time_precision']=='year' and row['evidence'][0]['source_timestamp']==1704067200000
def test_revision_keeps_old():
 p=tempfile.mktemp(); m=[dict(message_index=0,source_id='s',source_kind='evidence',content='x')]; b=EventBuilder(p)
 e=EventCandidate(subject_name='u',event_type='x',description='planned',status='planned',message_indices=[0],confidence=.8); old=b.write('u',EventExtraction(events=[e]),m)[0]
 b.write('u',EventExtraction(events=[e.model_copy(update={'status':'cancelled'})]),m,supersedes=old)
 assert len(EventRetriever(p).find('u',status='superseded'))==1 and len(EventRetriever(p).find('u'))==1
