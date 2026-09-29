import pytest
from memory.grounding import GroundedGraph, admit_graph
from memory.llm import LLM, LLMError
from memory.models import AddRequest


def request():
    return AddRequest(request_id="r", user_id="u", session_id="s", messages=[
        dict(role="user", content="My family supports me.", timestamp=0)])


def candidate(owner=None):
    return GroundedGraph.model_validate(dict(nodes=[
        dict(key="person:a", name="A", kind="person", owner_key=owner, message_indices=[0]),
        dict(key="family", name="Family", kind="group", owner_key="person:a", message_indices=[0])],
        edges=[dict(source="person:a", target="family", relation="支持", message_indices=[0],
                    evidence=[dict(message_index=0, text="My family supports me.")])]))


def test_null_normalization_is_narrow_and_non_mutating():
    g = candidate("null")
    assert admit_graph(g, request()).nodes[0].owner_key is None
    assert g.nodes[0].owner_key == "null"
    with pytest.raises(ValueError, match="Unknown node owner"):
        admit_graph(candidate("unknown"), request())
    g.nodes.append(g.nodes[0].model_copy(update={"key":"null", "owner_key":None}))
    assert admit_graph(g, request()).nodes[0].owner_key == "null"


def test_validation_feedback_repairs_once_and_keeps_ownership():
    calls = []
    class RepairLLM(LLM):
        def complete(self, instruction, payload, schema):
            calls.append(payload)
            return candidate("missing") if len(calls)==1 else candidate()
    result = RepairLLM().extract(request())
    assert len(calls)==2
    assert "missing" in calls[1]["validation_error"]
    assert calls[1]["original_request"]["messages"][0]["content"]==request().messages[0].content
    assert result.nodes[1].owner_key=="person:a"


def test_repair_failure_is_bounded():
    calls=[]
    class Broken(LLM):
        def complete(self, instruction, payload, schema):
            calls.append(payload)
            return candidate("missing")
    with pytest.raises(LLMError, match="after one repair"):
        Broken().extract(request())
    assert len(calls)==2


def test_null_string_does_not_require_second_call():
    calls=[]
    class NullLLM(LLM):
        def complete(self, instruction, payload, schema):
            calls.append(payload)
            return candidate("null")
    assert NullLLM().extract(request()).nodes[0].owner_key is None
    assert len(calls)==1
