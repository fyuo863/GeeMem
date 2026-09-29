import pytest
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.grounding import GroundedGraph, GraphPatch, admit_graph, apply_patch
from memory.llm import LLM
from memory.models import AddRequest
from memory.store import Store


def source():
    return AddRequest(request_id="write", user_id="u", session_id="s", messages=[
        dict(role="assistant", content="My family supports me.", timestamp=0),
        dict(role="user", content="How do they help you?", timestamp=1),
        dict(role="user", content="What gives you hope?", timestamp=2),
    ])


def node(key, kind="person"):
    return dict(key=key, name=key, kind=kind, message_indices=[0])


def edge(a, relation, b, indices):
    return dict(source=a, relation=relation, target=b, message_indices=indices,
                evidence=[dict(message_index=i, text=source().messages[i].content) for i in indices])


def candidate():
    return GroundedGraph.model_validate(dict(nodes=[node("speaker"), node("listener")],
        edges=[edge("speaker", "支持", "listener", [0]), edge("listener", "询问", "speaker", [1])]))


def repair():
    return GraphPatch.model_validate(dict(remove_edge_indices=[0, 1], upsert_nodes=[node("family", "group")],
        add_edges=[edge("family", "支持", "speaker", [0]), edge("listener", "询问", "speaker", [1, 2])]))


def test_local_repair_corrects_third_party_and_merges_question_evidence():
    original = candidate()
    fixed = admit_graph(apply_patch(original, repair()), source())
    assert {(e.source, e.relation, e.target) for e in fixed.edges} == {
        ("family", "支持", "speaker"), ("listener", "询问", "speaker")}
    assert fixed.edges[1].message_indices == [1, 2]
    assert original.edges[0].source == "speaker"  # no mutation of audit evidence


@pytest.mark.parametrize("fault", ["invented_quote", "missing_quote", "wrong_index", "causes_person", "inverse", "unknown_edge"])
def test_invalid_graph_or_patch_is_rejected(fault):
    graph = candidate()
    patch = GraphPatch()
    if fault == "invented_quote":
        graph.edges[0].evidence[0].text = "An invented statement."
    elif fault == "missing_quote":
        graph.edges[1].message_indices.append(2)
    elif fault == "wrong_index":
        graph.edges[0].evidence[0].message_index = 1
    elif fault == "causes_person":
        graph.edges[0].relation = "导致"
    elif fault == "inverse":
        graph.edges[0].relation = "experienced"
    else:
        patch.remove_edge_indices = [999]
    with pytest.raises(ValueError):
        admit_graph(apply_patch(graph, patch), source())


def test_duplicates_keep_all_exact_quotes():
    graph = candidate()
    graph.edges.append(type(graph.edges[1]).model_validate(edge("listener", "询问", "speaker", [2])))
    fixed = admit_graph(graph, source())
    assert fixed.edges[1].message_indices == [1, 2]
    assert len(fixed.edges[1].evidence) == 2


def test_semantic_review_patch_and_quotes_persist(tmp_path):
    class ReviewedLLM(LLM):
        def complete(self, instruction, payload, schema):
            if schema is GroundedGraph:
                return candidate()
            assert schema is GraphPatch
            assert payload["source"]["messages"][0]["role"] == "assistant"
            return repair()
    store = Store(tmp_path / "db.sqlite")
    with TestClient(create_app(store, ReviewedLLM())) as client:
        response = client.post("/add", json=source().model_dump())
        assert response.status_code == 200
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM edge_quotes").fetchone()[0] == 3
        rows = db.execute("SELECT s.name,e.relation,t.name FROM edges e JOIN nodes s ON s.id=e.source JOIN nodes t ON t.id=e.target").fetchall()
        assert ("family", "支持", "speaker") in [tuple(r) for r in rows]


def test_failed_audit_does_not_write_partial_graph(tmp_path):
    class BadAudit(LLM):
        def complete(self, instruction, payload, schema):
            if schema is GroundedGraph:
                graph = candidate()
                graph.edges[0].evidence[0].text = "Unsupported quote"
                return graph
            return GraphPatch()
    store = Store(tmp_path / "db.sqlite")
    with TestClient(create_app(store, BadAudit())) as client:
        assert client.post("/add", json=source().model_dump()).status_code == 502
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_strict_schema_requires_default_fields_without_mutating_original():
    from memory.llm import strict_json_schema
    schema = GraphPatch.model_json_schema()
    schema["properties"]["findings"]["default"] = []
    converted = strict_json_schema(schema)
    assert set(converted["required"]) == set(converted["properties"])
    assert converted["additionalProperties"] is False
    assert "default" not in converted["properties"]["findings"]
    assert "default" in schema["properties"]["findings"]
