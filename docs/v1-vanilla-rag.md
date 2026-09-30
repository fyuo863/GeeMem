# v1: Vanilla RAG implementation

## Branches and reference

`test_base` is the renamed evaluation/deployment infrastructure branch. `v1` starts at
`266a355` and implements an independent retrieval backend. The existing baseline tags,
`main`, and `graph` are unchanged. The production server still runs `4766046`.

Reference: https://github.com/wenxiaof345-ctrl/vanilla-rag-memory
Reference commit: `31ab7bf9cfa3ee3c4f986e82f6e7a00b134ba8ca` (README version 0.6.0).
The reference does not declare a license at that revision. No upstream source files
are vendored; this is an independent implementation of its default retrieval recipe,
not a claim of authorship over the reference method or a byte-for-byte fork.

## Implemented default recipe

- One message is split into source-text chunks: 320 lexical tokens, overlap 40.
  English words, individual CJK characters and punctuation define chunk boundaries.
- Local `BAAI/bge-small-en-v1.5`, normalized document/query embeddings; English BGE
  query prefix. Batch size 64. Model snapshot revision and file hashes are recorded.
- SQLite persists chunks/vectors and request digests. Search is scoped by user_id,
  across sessions. No LLM calls, graph extraction or answer generation.
- Hybrid retrieval combines dense rank and positive BM25 rank using weighted RRF:
  `1/(60+dense_rank) + 0.5/(60+lexical_rank)` (ranks start at 1).
  BM25 uses k1=1.5, b=0.75. Dense mode is also available.
- Append multiple-choice options to the retrieval query.
- Expand the first 20 ranked seeds with the preceding/following stored chunk in the
  same session (window 1); deduplicate then fill from the base ranking, up to top_k.
  Like upstream, adjacency uses insertion order, and the expanded output is not
  necessarily sorted by score. Scores remain the underlying retrieval scores.
- Configuration is read exclusively from root `.env`. No environment overrides.
  Model downloads use an explicit HTTP client with trust_env=False and the proxy
  from `.env`; inference loads only the downloaded local files.

## Intentional differences

Search returns exact source substrings, without upstream role/date prefixes or inferred
date annotations, to preserve GeeMem's original-text contract. Unknown timestamps are
omitted, not replaced with ingestion time. The Unix epoch is preserved when explicitly
provided. Request IDs are scoped by user. Existing AML auth, size limits and concurrency
limits remain. Inference/model and chunk configuration are locked to each database;
model checksum mismatches, nonfinite/zero vectors, or changed dimensions fail explicitly.

The reference's optional enhanced context/option-view mode, cross-encoder reranker,
temporal enrichment, offline FAISS/Answer/Eval pipeline are outside this default v0.6
recipe. Its published accuracy numbers do not apply to this implementation.

## Run locally

Recommended Python 3.11/3.12, at least 8 GB RAM. First prepare dependencies and model:

```powershell
python -m pip install -e ".[rag,test]"
python scripts/download_rag_model.py
```

Keep existing AML authentication entries in root `.env` and add as needed:

```dotenv
MEMORY_BACKEND=vanilla
RAG_MEMORY_DB=data/aml/vanilla.sqlite3
RAG_MODEL_PATH=data/models/bge-small-en-v1.5
RAG_MODEL_REVISION=5c38ec7c405ec4b44b94cc5a9bb96e735b38267a
RAG_DEVICE=cpu
RAG_RETRIEVAL_MODE=hybrid
RAG_CHUNK_TOKENS=320
RAG_CHUNK_OVERLAP=40
RAG_RRF_K=60
RAG_LEXICAL_WEIGHT=0.5
RAG_RESULT_WINDOW=1
RAG_RESULT_WINDOW_SEED_K=20
```

Optional download proxy: `RAG_DOWNLOAD_PROXY`; falls back to `.env`'s `LLM_PROXY`.
The model download is approximately 130 MB. Model files, databases, reference checkout
and test reports live in ignored `data/`; they are not pushed to GitHub.

```powershell
python -m pytest -q -p no:cacheprovider
python scripts/smoke_aml.py
python -m uvicorn memory.aml_api:app --host 127.0.0.1 --port 8000 --workers 1
```

`memory.aml_api:app` defaults to the vanilla backend on v1. `MEMORY_BACKEND=graph`
selects the inherited graph implementation; `memory.api:app` remains the legacy graph
research interface. LLM_MODEL remains gpt-4o-mini for that graph implementation.

## Deployment boundary

The existing server updater now tracks `test_base`, not v1. Do not use it to deploy v1:
its verified shared runtime does not include the new embedding dependencies. A v1
rollout requires an isolated Python runtime (or Dockerfile.aml), downloaded model files,
a separate vector database and explicit v1 source revision. Mount the root `.env` and
model/data directories when using Docker; the image does not embed keys or weights.
No production service was switched to v1 and no official evaluation was launched.

## Validation (2026-09-30)

- 68 unit/API tests passed, including ranking math, exact source chunks, session
  windows, restart persistence, user isolation, duplicate/conflicting requests,
  concurrent retries and failed embedding transaction atomicity.
- Real BGE model + local FastAPI smoke passed in 11.958 seconds (includes loading).
  Report: `data/aml-smoke/20260930T090910178148Z/report.json` (ignored local artifact).
- Ran the pinned reference and this implementation with the same real BGE model,
  12 synthetic memories across 4 sessions, and 4 queries including options. Top-5
  source sequence matched in all four cases; maximum score difference < 1.4e-9.
  Report: `data/vanilla-reference-parity.json` (ignored local artifact).
- These are functional/parity checks, not a benchmark accuracy measurement or an
  exhaustive equivalence proof. No official evaluation data was used.

## Rule-keyword tag experiment

`RAG_TAG_MODE=filter` enables local rule keywords at Add and Search; `off` preserves
original v1. No LLM or network calls occur in keyword extraction. BGE embeddings
remain local. Rules: Unicode NFKC and case normalization, English word extraction,
general stop-word removal, possessive/common plural normalization, Chinese overlapping
bigrams, deduplication. Every chunk is independent and all remaining distinct keywords
are retained, with identical rules for queries. No benchmark-specific entity dictionary,
gold answer lookup, learned synonym expansion, or cross-message inference is used.

Tags are written transactionally with vectors. Current identity is
`rule-keywords-v1:nfkc-stopwords-cjk-bigrams`. Old semantic tags are treated as unknown,
not as rule keywords. Existing committed requests remain idempotent and are not
retagged automatically; use a fresh database and re-add source messages for comparisons.

Search filtering is unchanged: intersect keywords with candidates from the original
hybrid ranking, preserve unknown/empty/version-mismatched tags, fall back to original
ranking when no candidate matches, then expand session neighbors. This remains a hard
filter, so indirect evidence or synonyms can still be lost. It does not accelerate the
full vector/BM25 scan. Enable with root `.env` RAG_TAG_MODE=filter; default remains off
until evaluation supports a change. Run `python scripts/benchmark_tags.py` for the same
419-message, 30-question paired public-data experiment at K=10/100.

Previous LLM experiment (historical): [report](tag-experiment-20260930.md).
