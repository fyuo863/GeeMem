import pytest
from fastapi.testclient import TestClient
from memory.api import create_app
from memory.grounding import GroundedGraph, admit_graph
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


def test_reversed_duplicates_keep_all_exact_quotes():
    graph = candidate()
    graph.edges.append(type(graph.edges[1]).model_validate(edge("speaker", "询问", "listener", [2])))
    fixed = admit_graph(graph, source())
    assert fixed.directed is False
    assert len(fixed.edges) == 2
    question = next(e for e in fixed.edges if e.relation == "询问")
    assert question.message_indices == [1, 2]
    assert len(question.evidence) == 2
    assert (question.source, question.target) == tuple(sorted(("speaker", "listener")))


@pytest.mark.parametrize("fault", ["invented_quote", "missing_quote", "wrong_index", "unknown_endpoint"])
def test_invalid_evidence_is_rejected(fault):
    graph = candidate()
    if fault == "invented_quote":
        graph.edges[0].evidence[0].text = "An invented statement."
    elif fault == "missing_quote":
        graph.edges[1].message_indices.append(2)
    elif fault == "wrong_index":
        graph.edges[0].evidence[0].message_index = 1
    else:
        graph.edges[0].target = "unknown"
    with pytest.raises(ValueError):
        admit_graph(graph, source())


def test_single_extraction_and_quote_persistence(tmp_path):
    calls = []
    class OneCallLLM(LLM):
        def complete(self, instruction, payload, schema):
            calls.append(schema)
            assert schema is GroundedGraph
            return candidate()
    store = Store(tmp_path / "db.sqlite")
    with TestClient(create_app(store, OneCallLLM())) as client:
        response = client.post("/add", json=source().model_dump())
        assert response.status_code == 200
    assert len(calls) == 1
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM edge_quotes").fetchone()[0] == 2


def test_failed_admission_does_not_write_partial_graph(tmp_path):
    class BadEvidence(LLM):
        def complete(self, instruction, payload, schema):
            graph = candidate()
            graph.edges[0].evidence[0].text = "Unsupported quote"
            return graph
    store = Store(tmp_path / "db.sqlite")
    with TestClient(create_app(store, BadEvidence())) as client:
        assert client.post("/add", json=source().model_dump()).status_code == 502
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_strict_schema_requires_default_fields_without_mutating_original():
    from memory.llm import strict_json_schema
    from memory.models import Graph
    schema = Graph.model_json_schema()
    converted = strict_json_schema(schema)
    assert set(converted["required"]) == set(converted["properties"])
    assert converted["additionalProperties"] is False
    assert "default" not in converted["properties"]["directed"]
    assert schema["properties"]["directed"]["default"] is False


def test_legacy_migration_preserves_reverse_edge_evidence_and_quotes(tmp_path):
    path = tmp_path / "db.sqlite"
    store = Store(path)
    with store.connect() as db:
        db.execute("DELETE FROM schema_meta WHERE name='undirected_v1'")
        for mid in ('m1', 'm2'):
            db.execute("INSERT INTO messages VALUES (?,?,?,?,?,?)", (mid,'u',mid,'user',mid,0))
        for nid in ('a', 'b'):
            db.execute("INSERT INTO nodes (id,user_id,node_key,name,kind,aliases) VALUES (?,?,?,?,?,?)", (nid,'u',nid,nid,'person','[]'))
        db.execute("INSERT INTO edges VALUES ('edge1','u','b','a','询问')")
        db.execute("INSERT INTO edges VALUES ('edge2','u','a','b','询问')")
        for eid,mid in [('edge1','m1'),('edge2','m2')]:
            db.execute("INSERT INTO edge_evidence VALUES (?,?)",(eid,mid))
            db.execute("INSERT INTO edge_quotes VALUES (?,?,?)",(eid,mid,mid))
    migrated = Store(path)
    Store(path)  # migration is idempotent
    with migrated.connect() as db:
        assert tuple(db.execute("SELECT source,target FROM edges").fetchone()) == ('a','b')
        assert db.execute("SELECT count(*) FROM edges").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM edge_evidence").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM edge_quotes").fetchone()[0] == 2
        assert not db.execute("PRAGMA foreign_key_check").fetchall()


def test_reverse_writes_across_sessions_merge_and_remain_scoped(tmp_path):
    from memory.models import SearchRequest
    store = Store(tmp_path / "db.sqlite")
    first = source()
    original = admit_graph(candidate(), first)
    store.add(first, original)
    second = source().model_copy(update={"request_id":"write2", "session_id":"other"})
    reversed_graph = original.model_copy(deep=True)
    for e in reversed_graph.edges:
        e.source, e.target = e.target, e.source
    saved = store.add(second, reversed_graph)
    third = source().model_copy(update={"user_id":"another"})
    store.add(third, original)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM edges WHERE user_id='u'").fetchone()[0] == 2
        assert db.execute("SELECT count(*) FROM edges WHERE user_id='another'").fetchone()[0] == 2
    result = store.search(SearchRequest(user_id='u',session_id='other',query='询问'),['询问'])
    assert result['data']
    assert {h['id'] for h in result['data']} <= set(saved['message_ids'])
