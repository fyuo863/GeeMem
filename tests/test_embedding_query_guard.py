import pytest
from memory.embeddings import HTTPEmbedder, bounded_embedding_query

@pytest.mark.parametrize("text", ["a"*29999, "a"*30000, "a"*30001, "中"*50000, "😀"*20000], ids=["below", "boundary", "above", "chinese", "emoji"])
def test_byte_budget_preserves_unicode_and_ends(text):
    sent = bounded_embedding_query(text)
    if len(text.encode("utf8")) <= 30000:
        assert sent == text
    else:
        assert len(sent.encode("utf8")) <= 30000
        assert sent.startswith(text[:1000])
        assert sent.endswith(text[-1000:])
        assert "�" not in sent

@pytest.mark.parametrize("model", ["text-embedding-v4", "other-model"])
def test_only_queries_change_not_identity_or_documents(model):
    e = HTTPEmbedder({"RAG_EMBEDDING_API_URL":"https://example.test", "RAG_EMBEDDING_API_KEY":"test", "RAG_EMBEDDING_API_MODEL":model, "RAG_QUERY_PREFIX":"PREFIX", "RAG_QUERY_INSTRUCTION":"INSTRUCTION"})
    identity = e.identity
    seen = []
    e.documents = lambda texts: seen.extend(texts) or texts
    text = "中"*50000 + "What is the answer?"
    try:
        e.queries([text, "short query"])
        assert seen[1] == "PREFIXINSTRUCTION\nQuery: short query"
        assert seen[0].endswith("What is the answer?")
        assert len(seen[0].encode("utf8")) <= 30000 if model == "text-embedding-v4" else seen[0] == "PREFIXINSTRUCTION\nQuery: " + text
        assert identity == e.identity
    finally:
        e.client.close()
