import pytest
from memory.conversation import TurnDraft
from memory.llm import LLM, LLMError
from memory.models import AddRequest


def request():
    return AddRequest(request_id="r",user_id="u",session_id="s",messages=[
        dict(role="user",content="Hello.",timestamp=0)])


def test_validation_feedback_and_bounded_repair():
    calls=[]
    class Fixed(LLM):
        def complete(self,instruction,payload,schema):
            calls.append(payload)
            return TurnDraft(entities=[],relations=[dict(source='A',target='invented' if len(calls)==1 else 'B',relation='询问')])
    assert len(Fixed().extract(request()).nodes)==2
    assert len(calls)==2
    assert 'Unknown relation endpoint' in calls[1]['validation_error']


def test_failed_repair_is_bounded():
    calls=[]
    class Broken(LLM):
        def complete(self,instruction,payload,schema):
            calls.append(payload)
            return TurnDraft(entities=[],relations=[dict(source='A',target='invented',relation='询问')])
    with pytest.raises(LLMError,match="after one repair"):
        Broken().extract(request())
    assert len(calls)==2
