import pytest

from memory.selector import (ConfigurableJudge, JudgeResult, OptionSelector, ProfileGate,
                             ProfileGateResult, SelectionResult)


class FakeLLM:
    def __init__(self):
        self.calls = []

    def complete(self, instruction, payload, schema):
        self.calls.append((instruction, payload, schema))
        return SelectionResult(scores=[
            {'option': payload['options'][0], 'score': 0.8},
            {'option': payload['options'][1], 'score': 0.2},
        ])


def test_selector_scores_supplied_options_without_retrieval():
    llm = FakeLLM()
    result = OptionSelector(llm).score('Where does Alice live?', ['Paris', 'Tokyo'])
    assert [item.option for item in result.scores] == ['Paris', 'Tokyo']
    assert [item.score for item in result.scores] == [0.8, 0.2]
    assert llm.calls[0][1] == {'question': 'Where does Alice live?', 'options': ['Paris', 'Tokyo']}


@pytest.mark.parametrize('question,options', [
    ('', ['Paris']),
    ('question', []),
    ('question', ['Paris', 'Paris']),
    ('question', ['']),
])
def test_selector_rejects_invalid_inputs(question, options):
    with pytest.raises(ValueError):
        OptionSelector(FakeLLM()).score(question, options)


class FakeProfileLLM:
    def complete(self, instruction, payload, schema):
        return ProfileGateResult(decision='profile_candidate', profile_type='residence',
                                 stability='stable', confidence=0.9,
                                 evidence='住在杭州', reason='用户明确陈述居住地')


def test_profile_gate_requires_grounded_evidence_and_profile_type():
    result = ProfileGate(FakeProfileLLM()).classify('我现在住在杭州。')
    assert result.decision == 'profile_candidate'
    assert result.profile_type == 'residence'
    with pytest.raises(ValueError):
        ProfileGate(FakeProfileLLM()).classify('我现在住在杭州。', role='moderator')


class FakeJudgeLLM:
    def complete(self, instruction, payload, schema):
        return JudgeResult(label='keep', scores=[
            {'option': 'keep', 'score': 0.8},
            {'option': 'discard', 'score': 0.2},
        ], confidence=0.9, evidence=payload['text'], reason='matches configured criteria')


def test_configurable_judge_can_be_reused_for_another_subject():
    config = {
        'name': 'character-memory-gate',
        'subject': '角色 A',
        'criteria': '保留角色 A 的稳定属性，丢弃一次性噪声。',
        'labels': [
            {'name': 'keep', 'description': '进入角色画像'},
            {'name': 'discard', 'description': '不进入角色画像'},
        ],
    }
    result = ConfigurableJudge(config, FakeJudgeLLM()).judge('角色 A 喜欢下棋。')
    assert result.label == 'keep'
    assert [item.option for item in result.scores] == ['keep', 'discard']


@pytest.mark.parametrize('evidence,expected', [
    ('"我同事小王住在北京。"', '我同事小王住在北京。'),
    (' “我同事小王住在北京。” ', '我同事小王住在北京。'),
    ('「小王住在北京」', '小王住在北京'),
    ('小王住在北京', '小王住在北京'),
])
def test_judge_accepts_only_source_quotes_with_optional_wrappers(evidence, expected):
    from memory.selector import _ground_quote
    assert _ground_quote(evidence, '我同事小王住在北京。') == expected


@pytest.mark.parametrize('evidence', ['', '   ', '""', '“ ”',
    '"我同事小王住在上海。"', '我同事小王住在北京.', '“小王住在北京"'])
def test_judge_rejects_empty_rewritten_and_mismatched_quotes(evidence):
    from memory.selector import _ground_quote
    with pytest.raises(ValueError):
        _ground_quote(evidence, '我同事小王住在北京。')


def test_judge_quote_repair_integration_and_optional_evidence():
    config = dict(name='test', subject='user', criteria='classify', labels=[
        dict(name='keep', description='keep'), dict(name='discard', description='discard')])
    class WrappedLLM(FakeJudgeLLM):
        def complete(self, instruction, payload, schema):
            result = super().complete(instruction, payload, schema)
            self.raw = result.model_copy(update={'evidence': '“' + payload['text'] + '”'
                if payload['require_evidence'] else ''})
            return self.raw
    llm = WrappedLLM()
    result = ConfigurableJudge(config, llm).judge('我同事小王住在北京。')
    assert result.evidence == '我同事小王住在北京。'
    assert llm.raw.evidence == '“我同事小王住在北京。”'
    config['require_evidence'] = False
    assert ConfigurableJudge(config, llm).judge('text').evidence == ''


def test_source_quotation_marks_are_preserved():
    from memory.selector import _ground_quote
    assert _ground_quote('“你好”', '他说“你好”。') == '“你好”'

@pytest.mark.parametrize('failure', ['request', 'evidence', 'label', 'schema'])
def test_judge_retries_failed_model_output_then_succeeds(monkeypatch, failure):
    from memory.llm import LLMError
    monkeypatch.setattr('memory.selector.time.sleep', lambda _: None)
    config=dict(name='test',subject='user',criteria='test',labels=[
        dict(name='keep',description='keep'),dict(name='discard',description='discard')])
    class Flaky(FakeJudgeLLM):
        calls=0
        def complete(self,instruction,payload,schema):
            self.calls+=1
            if self.calls==1:
                if failure=='request': raise LLMError('temporary network failure')
                if failure=='schema': return JudgeResult.model_validate({})
                result=super().complete(instruction,payload,schema)
                return result.model_copy(update={'evidence':'invented'} if failure=='evidence' else {'label':'unknown'})
            assert 'Previous attempt failed' in instruction
            return super().complete(instruction,payload,schema)
    llm=Flaky()
    assert ConfigurableJudge(config,llm).judge('hello').label=='keep'
    assert llm.calls==2


def test_judge_retry_limit_and_invalid_input(monkeypatch):
    from memory.llm import LLMError
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    config=dict(name='test',subject='user',criteria='test',max_attempts=2,labels=[
        dict(name='keep',description='keep'),dict(name='discard',description='discard')])
    class Broken:
        calls=0
        def complete(self,*args):
            self.calls+=1
            raise LLMError('failure')
    llm=Broken();judge=ConfigurableJudge(config,llm)
    with pytest.raises(ValueError): judge.judge(' ')
    assert llm.calls==0
    with pytest.raises(LLMError): judge.judge('hello')
    assert llm.calls==2


@pytest.mark.parametrize('status,attempts',[(401,1),(403,1),(429,3),(503,3)])
def test_judge_http_retry_policy(monkeypatch,status,attempts):
    import httpx
    from memory.llm import LLMError
    monkeypatch.setattr('memory.selector.time.sleep',lambda _:None)
    config=dict(name='test',subject='user',criteria='test',labels=[
        dict(name='keep',description='keep'),dict(name='discard',description='discard')])
    class Broken:
        calls=0
        def complete(self,*args):
            self.calls+=1
            response=httpx.Response(status,request=httpx.Request('POST','https://example.test'))
            try: response.raise_for_status()
            except httpx.HTTPStatusError as exc: raise LLMError('failure') from exc
    llm=Broken()
    with pytest.raises(LLMError): ConfigurableJudge(config,llm).judge('hello')
    assert llm.calls==attempts
