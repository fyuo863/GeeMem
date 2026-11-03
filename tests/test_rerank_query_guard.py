import httpx
import numpy as np
import pytest
from memory.rerank import HTTPReranker, bounded_rerank_query

@pytest.mark.parametrize("size", [1, 49999, 50000, 50001, 101118, 185627])
def test_query_boundary(size):
    query = "前" * (size - 1) + "？"
    sent = bounded_rerank_query(query)
    if size <= 50000:
        assert sent == query
    else:
        assert len(sent) == 8000
        assert sent.startswith(query[:2000])
        assert sent.endswith(query[-5000:])
    assert len(query) == size

@pytest.mark.parametrize("protocol", ["dashscope", "vllm"])
def test_payload_and_scores_unchanged_except_oversized_query(protocol):
    reranker = HTTPReranker({"RAG_RERANK_API_URL": "https://example.test/rerank", "RAG_RERANK_API_PROTOCOL": protocol})
    query = "context " * 10000 + "What is the answer?"
    documents = ["source A", "source B"]
    def respond(request):
        import json
        payload = json.loads(request.content)
        values = payload["input"] if protocol == "dashscope" else payload
        assert values["documents"] == documents
        assert values["query"] == (bounded_rerank_query(query) if protocol == "dashscope" else query)
        rows = {"results": [{"index": 1, "relevance_score": .8}, {"index": 0, "relevance_score": .2}]}
        return httpx.Response(200, json={"output": rows} if protocol == "dashscope" else rows)
    reranker.client.close()
    reranker.client = httpx.Client(transport=httpx.MockTransport(respond))
    try:
        assert np.allclose(reranker.score(query, documents), [.2, .8])
        assert documents == ["source A", "source B"]
    finally:
        reranker.client.close()


def test_error_logs_are_redacted(caplog):
    reranker = HTTPReranker({"RAG_RERANK_API_URL": "https://example.test/rerank", "RAG_RERANK_API_PROTOCOL": "dashscope"})
    reranker.client.close()
    reranker.client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(400, json={"code": "InvalidParameter", "message": "secret source text"})))
    try:
        with pytest.raises(httpx.HTTPStatusError):
            reranker.score("private query", ["private evidence"])
        assert "code=InvalidParameter" in caplog.text
        assert "private query" not in caplog.text
        assert "private evidence" not in caplog.text
        assert "secret source text" not in caplog.text
    finally:
        reranker.client.close()
