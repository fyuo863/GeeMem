from memory.build_audit import AuditedLLM
from memory.rule import RuleExtraction
from memory.event import EventBuilder, EventCandidate, EventExtraction, EventRetriever
from memory.event_time import parse_time
from memory.entity_resolver import EntityResolver


def test_coverage_repairs_unsupported_extracted_claim():
    class Fake:
        calls=0
        def complete(self,prompt,payload,schema):
            self.calls+=1
            return schema.model_validate(dict(rules=[],coverage=[dict(message_index=0,
                disposition='extracted' if self.calls==1 else 'not_applicable',reason='greeting')]))
    llm=AuditedLLM(Fake())
    assert llm.complete('rules',{'messages':[dict(message_index=0,content='hello')]},RuleExtraction).rules==[]
    assert [a['valid'] for a in llm.audit]==[False,True]


def test_event_uses_source_anchor_not_model_date(tmp_path):
    builder=EventBuilder(tmp_path/'db')
    source=[dict(message_index=0,source_id='s',content='Yesterday I lost my job.',timestamp=1674172800000)]
    candidate=EventCandidate(subject_name='Jon',event_type='job loss',description='lost job',
        event_time_start='2023-01-16',event_time_end='2023-01-16',time_expression='Yesterday',
        reference_timestamp=0,time_message_index=0,message_indices=[0],confidence=1)
    builder.write('u',EventExtraction(events=[candidate]),source)
    row=EventRetriever(tmp_path/'db').find('u')[0]
    assert row['event_time_start']=='2023-01-19'
    assert row['reference_timestamp']==source[0]['timestamp']


def test_ambiguous_time_anchor_remains_unknown(tmp_path):
    b=EventBuilder(tmp_path/'db')
    c=EventCandidate(subject_name='A',event_type='visit',description='visit',time_expression='yesterday',
                     message_indices=[0,1],confidence=1)
    messages=[dict(message_index=i,source_id=str(i),content='Visited yesterday',timestamp=1704067200000+i*86400000) for i in range(2)]
    b.write('u',EventExtraction(events=[c]),messages)
    assert EventRetriever(tmp_path/'db').find('u')[0]['event_time_start'] is None


def test_calendar_boundaries_and_nulls():
    assert parse_time('last month',1704067200000)==('2023-12','2023-12','month')
    assert parse_time('two weeks ago',1704067200000)==('2023-12-18','2023-12-18','day')
    assert parse_time('2024-02-30',None)[:2]==(None,None)
    assert parse_time('yesterday',None)[:2]==(None,None)
    c=EventCandidate(subject_name='A',event_type='x',description='x',location='null',
                     reference_timestamp='null',message_indices=[0],confidence=1)
    assert c.location is None and c.reference_timestamp is None


def test_unknown_people_do_not_merge_across_mentions(tmp_path):
    from memory.relationship import RelationshipBuilder
    from contextlib import closing
    b=RelationshipBuilder(tmp_path/'db')
    with closing(b.connect()) as db,db:
        resolver=EntityResolver()
        a=resolver.resolve(db,'u','unknown',source_id='one')
        b=resolver.resolve(db,'u','unknown',source_id='two')
        assert a['id']!=b['id']
