import pytest
from memory.rule import RuleBuilder, RuleRetriever, RuleCandidate, RuleExtraction


def message(text='报告先给结论', source='s', kind='evidence'):
    return [dict(message_index=0, source_id=source, content=text, source_kind=kind)]


def extraction(action='先给结论', scope='report'):
    return RuleExtraction(rules=[RuleCandidate(name='报告格式', condition='写报告时',
        actions=[action], exceptions=['紧急情况先说明风险'], scope=scope,
        message_indices=[0], confidence=.9)])


def test_roundtrip_and_idempotence(tmp_path):
    path = tmp_path/'r.db'; b = RuleBuilder(path)
    first = b.write('u', extraction(), message())
    assert b.write('u', extraction(), message(source='s2')) == first
    row = RuleRetriever(path).find('u')[0]
    assert row['actions'] == ['先给结论']
    assert row['exceptions'] == ['紧急情况先说明风险']
    assert {e['source_id'] for e in row['evidence']} == {'s','s2'}
    assert RuleRetriever(path).find('other') == []
    assert RuleRetriever(path).find_applicable('u','报告')[0]['id'] == first[0]
    assert RuleRetriever(path).find_applicable('u','报告',scope='other') == []
    assert RuleRetriever(path).find_applicable('u','zzzzqqqq') == []


def test_version_and_retry(tmp_path):
    path=tmp_path/'r.db'; b=RuleBuilder(path); r=RuleRetriever(path)
    old=b.write('u',extraction(),message())[0]
    new=b.write('u',extraction('先给数据'),message(source='new'),supersedes=old)
    assert b.write('u',extraction('先给数据'),message(source='new'),supersedes=old)==new
    assert r.find('u')[0]['version']==2
    assert r.find('u',status='superseded')[0]['id']==old
    with pytest.raises(ValueError):
        b.write('other',extraction(),message(),supersedes=old)
    with pytest.raises(ValueError):
        b.write('u',extraction('其他'),message(),supersedes=old)


@pytest.mark.parametrize('messages', [message(kind='context'),
    [dict(message_index=1,source_id='s',content='内容')],message()+message()])
def test_invalid_sources_leave_no_writes(tmp_path,messages):
    path=tmp_path/'r.db'; b=RuleBuilder(path)
    with pytest.raises(ValueError): b.write('u',extraction(),messages)
    assert RuleRetriever(path).find('u')==[]


def test_model_extract_and_empty(tmp_path):
    class Fake:
        def complete(self,prompt,payload,schema):
            assert 'message_index' in prompt
            return extraction()
    b=RuleBuilder(tmp_path/'r.db',Fake())
    assert b.extract(message()).rules[0].scope=='report'
    assert b.write('u',RuleExtraction(rules=[]),message())==[]
