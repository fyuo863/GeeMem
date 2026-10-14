# Temporal-window deployment — 2026-10-14

Release tag: `smoke-temporal-window-20261014`.
Runtime code: main `e93b12d` (same runtime code as `605faab`; later changes were documentation/assets). This report adds no runtime changes.

## Production configuration

Only the following settings changed from the preceding deployment:

```dotenv
RAG_TEMPORAL_MODE=on
RAG_TIME_ANNOTATION_MODE=on
RAG_TEMPORAL_EXPERIMENT=window
RAG_TEMPORAL_FUSION_WEIGHT=0.10
RAG_TEMPORAL_WINDOW_RATIO=0.50
RAG_TEMPORAL_CONTEXT=1
```

Retained: gpt-4o-mini, text-embedding-v4, qwen3.7-text-rerank; dual partition retrieval; LLM multihop/recovery; raw evidence bundles; per-source direct selection and joint chain selection; version protection, credential masking, rule applicability, supplemental retrieval and audit logs. Add/Search concurrency settings remain 0 (no fixed application semaphore limit).

Fact replacement, state evidence adjudication, semantic privacy, raw coverage, old evidence chain/group modes remain off. The raw evidence route is retained.

A Git tag records source, not private `.env`. Reproduce this deployment using these explicit overrides and the existing private production configuration. No credentials are included here.

## Validation

- Local targeted regression: 47 passed (temporal, window, raw evidence and HTTP adapter tests).
- Production HTTP synthetic smoke: 13/13 checks passed, including health, Add, idempotence, conflict rejection, date query, ordinary query, multihop evidence presence, response format/order, user isolation, authentication and persisted time annotations.
- Six-message Add: 12.68 seconds. Date Search: 7.77 seconds; ordinary Search: 6.93 seconds; multihop Search: 26.08 seconds. These are single samples, not latency estimates.
- Two dated messages received model annotations; four other messages used rules.
- A direct backend diagnostic with production settings confirmed window triggered, one matching date, temporal fusion and final soft quota applied. This small pool had no out-of-pool promotion; it proves activation, not a recall gain.
- Public `/health`: HTTP 200. Service active/running, zero restarts after deployment.
- Reports: `data/deploy-temporal-20261014/report.json`, `window-trace.json` on server and local workspace (not committed).

These checks use synthetic messages and real model APIs, not an official evaluation or full accuracy benchmark. Ordinary/date query evidence presence checks do not measure precision. The official score change remains to be measured.

## Limits

Historical messages are not batch-reannotated by the LLM or re-embedded; read-time rule enrichment remains available. New Add requests have bounded annotation calls. Relative query dates still require a usable reference time; unknown dates retain global recall. Message timestamps must not be treated as event dates. Later partition fusion and upper retrieval selection can affect the final evidence order.

## Deployment and rollback

Service: `csig-aml-v1.service`; repository: `/GeeAI/AIAgent/CSIG`; public base URL: `http://210.16.160.209:18092`.

Previous source tag: `smoke-evidence-fixes-20261011` (`605faab`). Private pre-change backup: `/home/agent/csig-deploy/rollback-temporal-20261014T105231Z`, containing source SHA, private `.env` and consistent SQLite backup.

Rollback should restore the previous source/config and restart the service while retaining the current database. Do not replace the database with its backup and discard subsequent writes without a separate recovery decision. No official evaluation was submitted by this deployment.
