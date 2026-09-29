import sqlite3
from concurrent.futures import ThreadPoolExecutor
import httpx
import pytest
from fastapi.testclient import TestClient
from memory import config
from memory.api import create_app
from memory.llm import LLM, LLMError
from memory.models import AddRequest, Graph, SearchRequest
from memory.store import Store, Conflict


def request(user="u", session="s", rid="r"):
    return AddRequest(request_id=rid, user_id=user, session_id=session, messages=[
        {"role": "user", "content": "用户A向用户B询问比赛地点。", "timestamp": 1704067200000},
        {"role": "assistant", "content": "比赛在杭州举行。", "timestamp": 1704067320123},
        {"role": "user", "content": "我每周训练三次。", "timestamp": 1704067380000},
    ])


def graph():
    return Graph.model_validate({"nodes": [
        {"key": "a", "name": "用户A", "kind": "person", "message_indices": [0]},
        {"key": "b", "name": "用户B", "kind": "person", "message_indices": [0]},
        {"key": "city", "name": "杭州", "kind": "place", "message_indices": [1]},
        {"key": "training", "name": "训练", "kind": "activity", "message_indices": [2]},
    ], "edges": [
        {"source": "a", "target": "b", "relation": "询问", "message_indices": [0]},
        {"source": "b", "target": "city", "relation": "回答地点", "message_indices": [1]},
        {"source": "city", "target": "training", "relation": "训练地", "message_indices": [2]},
    ]})

class FakeLLM:
    def __init__(self):
        self.calls = 0

    def extract(self, payload):
        self.calls += 1
        return graph()

    def keywords(self, query):
        return [query]

@pytest.fixture
def store(tmp_path):
    return Store(tmp_path / "memory.db")


def test_api_contract_and_idempotency(store):
    llm = FakeLLM()
    with TestClient(create_app(store, llm)) as client:
        payload = request().model_dump()
        first = client.post("/add", json=payload)
        assert first.status_code == 200
        repeated = client.post("/add", json=payload).json()
        assert repeated["message_ids"] == first.json()["message_ids"]
        assert repeated["deduplicated"] and llm.calls == 1
        response = client.post("/search", json={"user_id": "u", "query": "杭州", "max_hops": 0})
        assert response.status_code == 200
        hit = response.json()["data"][0]
        assert set(hit) == {"id", "content", "score", "created_at"}
        assert hit["content"] == payload["messages"][1]["content"]
        assert hit["created_at"] == "2024-01-01T00:02:00.123000Z"
        payload["messages"][0]["content"] = "changed"
        assert client.post("/add", json=payload).status_code == 409
        assert client.post("/search", json={"user_id": "u", "query": "x", "max_hops": 5}).status_code == 422


def test_undirected_and_multihop(store):
    saved = store.add(request(), graph())
    with store.connect() as db:
        edge = db.execute("SELECT s.node_key, t.node_key FROM edges e JOIN nodes s ON s.id=e.source JOIN nodes t ON t.id=e.target WHERE e.relation='询问'").fetchone()
        assert set(edge) == {"a", "b"}
    def search(hops):
        return store.search(SearchRequest(user_id="u", query="用户A", max_hops=hops), ["用户A"])["data"]
    assert len(search(0)) == 1
    assert saved["message_ids"][1] not in {r["id"] for r in search(1)}
    assert saved["message_ids"][1] in {r["id"] for r in search(2)}
    assert search(2)[0]["score"] > search(2)[1]["score"]


def test_scoping_and_persistence(store):
    a = store.add(request(), graph())
    b = store.add(request(session="other", rid="r2"), graph())
    c = store.add(request(user="other"), graph())
    restarted = Store(store.path)
    hits = restarted.search(SearchRequest(user_id="u", session_id="s", query="杭州"), ["杭州"])["data"]
    assert hits
    assert {r["id"] for r in hits} <= set(a["message_ids"])
    assert not {r["id"] for r in hits} & set(b["message_ids"] + c["message_ids"])
    assert restarted.search(SearchRequest(user_id="missing", query="杭州"), ["杭州"]) == {"data": []}
    assert restarted.search(SearchRequest(user_id="u", query="不存在"), ["不存在"]) == {"data": []}


def test_concurrent_duplicate_writes(store):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: store.add(request(), graph()), range(4)))
    assert sum(not r["deduplicated"] for r in results) == 1
    assert len({tuple(r["message_ids"]) for r in results}) == 1


def test_invalid_graph_no_partial_write(store):
    invalid = graph()
    invalid.edges[0].target = "missing"
    with pytest.raises(ValueError):
        store.add(request(), invalid)
    assert store.existing(request()) is None
    invalid = graph()
    invalid.nodes[0].message_indices = [99]
    with pytest.raises(ValueError):
        store.add(request(), invalid)
    with store.connect() as db:
        assert db.execute("SELECT count(*) FROM messages").fetchone()[0] == 0


def test_failure_is_explicit_and_retryable(store):
    class Broken(FakeLLM):
        def extract(self, payload):
            raise LLMError("Unavailable")
    with TestClient(create_app(store, Broken())) as client:
        assert client.post("/add", json=request().model_dump()).status_code == 502
    assert store.existing(request()) is None
    with TestClient(create_app(store, FakeLLM())) as client:
        assert client.post("/add", json=request().model_dump()).status_code == 200


def test_llm_json_adapter(monkeypatch, tmp_path):
    original_client = httpx.Client
    seen = []
    def respond(req):
        import json
        payload = json.loads(req.content)
        seen.append(payload)
        candidate = graph().model_dump()
        for edge in candidate["edges"]:
            edge.pop("evidence", None)
        content = json.dumps(candidate) if len(seen) == 1 else "{}"
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=test-key\n", encoding="utf-8")
    monkeypatch.setattr(httpx, "Client", lambda **kw: original_client(transport=httpx.MockTransport(respond), **kw))
    result = LLM().extract(request())
    assert result.edges[0].relation == "询问"
    assert len(seen) == 1
    assert seen[0]["response_format"]["type"] == "json_schema"
    assert seen[0]["response_format"]["json_schema"]["strict"] is True
    assert result.edges[0].evidence[0].text == request().messages[0].content
    assert len(seen[0]["messages"]) == 2
    assert "用户A" in seen[0]["messages"][1]["content"]
    import json
    indexed = json.loads(seen[0]["messages"][1]["content"])["messages"]
    assert [m["message_index"] for m in indexed] == [0, 1, 2]
    assert [m["content"] for m in indexed] == [m.content for m in request().messages]


def test_llm_malformed_response(monkeypatch, tmp_path):
    original_client = httpx.Client
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=test-key\n", encoding="utf-8")
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"choices": []}))
    monkeypatch.setattr(httpx, "Client", lambda **kw: original_client(transport=transport, **kw))
    with pytest.raises(LLMError):
        LLM().extract(request())


def test_connection_retry_is_bounded(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".env").write_text("LLM_API_KEY=test-key\n", encoding="utf-8")
    original = httpx.Client
    attempts = []
    def fail(req):
        attempts.append(req)
        raise httpx.ConnectError("Connection unavailable", request=req)
    monkeypatch.setattr(httpx, "Client", lambda **kw: original(transport=httpx.MockTransport(fail), **kw))
    with pytest.raises(LLMError):
        LLM().keywords("test")
    assert len(attempts) == 3
