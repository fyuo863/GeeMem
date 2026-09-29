import pytest
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.grounding import GroundedGraph, admit_graph, merge_graphs
from memory.llm import LLM
from memory.models import AddRequest
from memory.store import Store


def request(count=17):
    return AddRequest(request_id="r", user_id="u", session_id="s", messages=[
        dict(role="user", content=f"Exact original {i}.", timestamp=i) for i in range(count)])


def graph(indices, req):
    return GroundedGraph.model_validate(dict(nodes=[
        dict(key=k, name=k, kind="person", message_indices=indices) for k in ("a", "b")],
        edges=[dict(source="a", target="b", relation="询问", message_indices=indices,
                    evidence=[dict(message_index=i, text=req.messages[i].content) for i in indices])]))


def test_global_indices_context_and_evidence_union():
    req = request()
    calls = []
    class ChunkLLM(LLM):
        def complete(self, instruction, payload, schema):
            calls.append(payload)
            return graph(payload["focus_message_indices"], req)
    result = ChunkLLM().extract(req)
    assert len(calls) == 3
    assert [m["message_index"] for m in calls[2]["messages"]] == [0, 1, 14, 15, 16]
    assert calls[1]["known_entities"]
    assert len(result.nodes) == 2 and len(result.edges) == 1
    assert result.edges[0].message_indices == list(range(17))
    assert len(result.edges[0].evidence) == 17


def test_later_chunk_failure_writes_nothing(tmp_path):
    req = request(9)
    class BadChunk(LLM):
        def complete(self, instruction, payload, schema):
            original = payload.get("original_request", payload)
            g = graph(original["focus_message_indices"], req)
            if 8 in original["focus_message_indices"]:
                g.edges[0].target = "missing"
            return g
    store = Store(tmp_path / "db")
    with TestClient(create_app(store, BadChunk())) as client:
        assert client.post("/add", json=req.model_dump()).status_code == 502
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_unique_exact_quote_relocation_preserves_original_candidate():
    req = request(3)
    g = graph([0], req)
    g.edges[0].evidence[0].text = req.messages[2].content
    result = admit_graph(g, req)
    assert result.edges[0].message_indices == [2]
    assert result.edges[0].evidence[0].message_index == 2
    assert g.edges[0].message_indices == [0]
    with pytest.raises(ValueError):
        admit_graph(g, req, allowed_indices={0, 1})


def test_ambiguous_quotes_not_relocated_and_errors_aggregated():
    req = request(3)
    req.messages[1].content = req.messages[2].content = "Repeated original."
    g = graph([0], req)
    g.nodes[0].owner_key = "unknown-owner"
    g.edges[0].target = "missing"
    g.edges[0].evidence[0].text = "Repeated original."
    with pytest.raises(ValueError) as error:
        admit_graph(g, req)
    assert "unknown-owner" in str(error.value)
    assert "missing" in str(error.value)
    assert "exact_matches=[1, 2]" in str(error.value)


def test_merge_refuses_different_owners_for_same_key():
    req = request(2)
    first = admit_graph(graph([0], req), req)
    second = first.model_copy(deep=True)
    second.nodes[0].owner_key = "b"
    with pytest.raises(ValueError, match="Conflicting"):
        merge_graphs([first, second], req)
