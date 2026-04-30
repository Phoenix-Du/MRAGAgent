# Agents.md

## Project Overview

This repository is a FastAPI-based multimodal RAG orchestration service. It accepts chat-style user queries, detects whether the request is normal QA or image-oriented search, prepares source documents or image candidates, sends content into a RAG layer, and returns grounded answers with supporting context.

The project is built as an orchestrator around several local or remote services:

- FastAPI orchestrator: main API and web entry point.
- RAGAnything-compatible bridge: document ingestion and retrieval/generation.
- Image pipeline service: multimodal image search and image metadata extraction.
- Rasa parse service: optional intent parsing.
- Redis/MySQL: optional memory and preference storage.
- Local fallback clients: keep core flows testable when external services are missing.

## Main Runtime Flow

1. `app/main.py` creates the FastAPI application, serves the web UI, exposes health/metrics endpoints, and mounts the `/v1` chat router.
2. `app/api/chat.py` handles `/v1/chat/query`.
   - Normalizes the request.
   - Uses request-provided `intent` when present.
   - Otherwise calls the unified LLM query planner first.
   - Falls back to Rasa, heuristic intent routing, and legacy scene parsers only when planner is unavailable or low-confidence.
   - Handles clarification when input is under-specified.
   - Calls `MinAdapter` for orchestration.
3. `app/adapters/min_adapter.py` is the central adapter.
   - `normalize_input`: canonicalizes incoming request data.
   - `ingest_to_rag`: dispatches to document/image preparation and sends prepared sources to RAG.
   - `query_with_context`: retrieves/generates the final answer and attaches context.
4. `app/services/dispatcher.py` prepares route-specific material.
   - `general_qa`: uses source documents, URLs, or search results, then can rerank/crawl/filter.
   - `image_search`: invokes the image pipeline and turns image results into source documents.
5. `app/services/rag_client.py` talks to the configured RAG endpoint first and falls back to a local in-memory implementation for tests/dev.
6. `app/services/memory_client.py` stores conversation memory and preferences through in-memory, Redis, MySQL, or hybrid modes.

## Current Query Flow

The current `/v1/chat/query` flow is:

1. Validate `QueryRequest` with Pydantic.
   - `uid`, `query`, limits, confidence, and URL shape are bounded.
   - URL inputs must be public `http`/`https`; localhost/private/link-local targets are rejected.
2. Reset request runtime flags and initialize request metrics.
3. Determine intent and retrieval plan.
   - If the client explicitly passes `intent`, use it.
   - Otherwise call `app/services/query_planner.py`.
   - The planner makes one LLM call and returns `intent`, `confidence`, `search_rewrite`, `entities`, and either `general_constraints` or `image_constraints`.
   - On planner success, `chat.py` applies those constraints directly and does not call the older scene-specific parser again.
   - If planner fails, returns bad JSON, or is low-confidence, the code falls back to Rasa and then heuristics.
4. Resolve pending clarification state from memory.
   - User replies to prior clarification questions are merged back into the original query.
   - Weather clarification avoids duplicating city names already present in the query.
5. Apply route-specific constraints.
   - `general_qa`: `general_constraints.search_rewrite` becomes the retrieval query.
   - `image_search`: `image_constraints.search_rewrite` becomes `image_search_query`, while the original user query is preserved for answering context.
6. Run clarification checks.
   - Missing weather city or overly generic image requests can return a clarification response before retrieval.
7. `MinAdapter.normalize_input()` calls `TaskDispatcher.prepare_documents()`.
   - `general_qa` uses provided docs, a direct URL, or search -> rerank -> safe URL filtering -> concurrent crawl -> optional body rerank.
   - `image_search` uses supplied images or calls the image pipeline, then wraps image results into a source document.
8. `MinAdapter.ingest_to_rag()` ingests normalized documents.
   - Image-search ingest can be skipped through config.
   - RAG client uses remote RAGAnything-compatible endpoint first, then bounded local fallback if allowed.
9. `MinAdapter.query_with_context()` gets user memory and returns the final response.
   - `general_qa` queries RAG.
   - `image_search` builds a VLM-backed image answer.
10. The response includes `answer`, `evidence`, `images`, `trace_id`, `latency_ms`, `route`, and `runtime_flags`.

## Planner And Parser Architecture

- `app/services/query_planner.py`: primary LLM planner for first-pass intent routing and retrieval planning. It is designed to avoid two LLM calls by merging intent detection and scene-specific query planning.
- `app/services/image_query_parser.py`: legacy/fallback parser for image and general constraints when the planner is unavailable. It still contains heuristic and LLM parsing logic.
- `app/services/llm_json_client.py`: shared OpenAI-compatible JSON client utilities used by planner/parser code.
- `app/services/parser_cache.py`: bounded TTL cache used by parser paths.
- `app/core/url_safety.py`: shared public URL safety checks used by schema validation and dispatcher filtering.

## Important Modules

- `app/api/chat.py`: public chat API, clarification behavior, image proxy SSRF hardening.
- `app/adapters/min_adapter.py`: top-level orchestration contract used by the API.
- `app/services/connectors.py`: search, crawl, BGE rerank, and image-pipeline clients.
- `app/services/dispatcher.py`: converts user intent into prepared RAG source documents.
- `app/services/rag_client.py`: remote RAG client plus local fallback.
- `app/services/memory_client.py`: memory and user preference storage.
- `app/services/image_query_parser.py`: heuristic/LLM parsing for image and general constraints.
- `app/services/qwen_vlm_images.py`: OpenAI-compatible/Qwen VLM helpers for image understanding.
- `app/integrations/`: local bridge services and adapters for RAGAnything, image pipeline, Rasa, and RankLLM.
- `app/models/schemas.py`: Pydantic request/response contracts.
- `web/`: simple browser UI for manual interaction.
- `rasa_project/`: Rasa configuration, domain, and training data.
- `scripts/`: PowerShell helpers for service startup and health checks.
- `docs/architecture/system-technical-solution.md`: technical architecture notes.

## External Services And Ports

Common local service layout:

- Orchestrator API: `http://127.0.0.1:8000`
- Image pipeline: `http://127.0.0.1:9010`
- Rasa parse: `http://127.0.0.1:5005`
- RAGAnything bridge: `http://127.0.0.1:9002`
- Redis/MySQL: usually started through Docker Compose when needed.

Important endpoints:

- `GET /healthz`
- `GET /metrics`
- `POST /v1/chat/query`
- `GET /v1/chat/image-proxy`

## Configuration

Configuration is primarily handled through:

- `app/core/settings.py`
- `app/integrations/bridge_settings.py`
- `.env` for local secrets/runtime settings
- `.env.example` as the tracked example template

Do not commit `.env`. Keep `.env.example` tracked.

## Testing

The base test command is:

```powershell
pytest -q
```

`pyproject.toml` configures:

- `pythonpath = ["."]`
- `testpaths = ["tests"]`
- `norecursedirs = ["third_party", "models", "raganything_storage", "tmp"]`

At the time of the last cleanup, the suite passed with 9 tests.

After the first optimization pass, the baseline test suite passed with 12 tests.
After the second optimization pass, the baseline test suite passed with 14 tests:
After the third optimization pass, the baseline test suite passed with 17 tests:
After the fourth optimization pass, the baseline test suite passed with 18 tests:
After the first major-refactor pass, the baseline test suite passed with 21 tests:
After the second major-refactor pass, the baseline test suite passed with 22 tests:
After the query planner fusion pass, the baseline test suite passed with 24 tests:

```powershell
pytest -q
```

## Git And Repository Hygiene

The repository previously had no ignore file, which caused large/generated/local files to enter commit history or staging. The intended tracked content is source code, tests, scripts, docs, config templates, and small project metadata.

The current `.gitignore` should keep these out of Git:

- `.env`
- `.venv/`
- Python caches and test caches
- logs
- `/tmp/`
- `/raganything_storage/`
- `/rasa_project/models/`
- root-level `/models/`
- root-level `/third_party/`

Important detail: ignore root-level `/models/`, not plain `models/`, because `app/models/` contains tracked Pydantic schemas and must remain versioned.

Clean remote upload status:

- Backup branch was created: `codex/backup-before-git-cleanup`
- Clean branch was created: `codex/clean-upload`
- Clean commit: `14ed9af84b499be35aeede92fdc810d2313b1b1b`
- That clean branch was pushed to remote `main`.

Local `main` may still contain noisy staged cleanup from the earlier oversized state. Do not run destructive reset/checkout commands unless the user explicitly asks. If more Git cleanup is needed, inspect status carefully and preserve user changes.

## Recent Optimization Pass

The first low-risk project optimization pass made these changes:

- Added request validation boundaries for `QueryRequest`: uid/query length, intent confidence range, max image/doc/candidate limits.
- Made CORS origins configurable through `CORS_ALLOW_ORIGINS`.
- Added `WEB_CRAWL_CONCURRENCY` and changed general QA page crawling to limited concurrency.
- Added `ALLOW_PLACEHOLDER_FALLBACK` so production can disable placeholder search/crawl/RAG fallback behavior.
- Restored high-impact Chinese text in the web UI, local RAG fallback answer, intent keywords, and image-query heuristics.
- Added regression tests for request limit validation, image-proxy loopback rejection, and real Chinese spatial image constraints.
- Verified with `pytest -q` and `python -m compileall app`.

The second low-risk optimization pass made these changes:

- Added `http`/`https` validation for `QueryRequest.url`; non-web schemes now fail request validation.
- Added `PARSER_CACHE_MAX_ENTRIES` through bridge settings and bounded parser caches with oldest-entry eviction.
- Added regression tests for non-http URL rejection and parser cache bounds.
- Verified with `pytest -q` and `python -m compileall app`.

The third low-risk optimization pass made these changes:

- Changed image proxy remote fetching to follow redirects manually and validate every redirect target against SSRF rules.
- Added `IMAGE_PROXY_MAX_REDIRECTS` for redirect bounds.
- Added `LOCAL_RAG_STORE_MAX_DOCS` and bounded the local RAG fallback store with oldest-entry eviction.
- Deduplicated selected web URLs before crawling in the general QA dispatcher branch.
- Added regression tests for blocked redirect targets, bounded local RAG storage, and dispatcher URL dedupe.
- Verified with `pytest -q` and `python -m compileall app`.

The fourth low-risk optimization pass made these changes:

- Added clean runtime definitions for chat intent heuristics and Chinese count parsing using Unicode escapes, so real Chinese queries are handled without relying on mojibake strings.
- Added regression coverage for real Chinese image intent, general QA intent, and Chinese count parsing.
- Verified with `pytest -q` and `python -m compileall app`.

The first major-refactor pass made these changes:

- Added `app/core/url_safety.py` as a shared URL safety module.
- Reused URL safety rules in request schema validation and dispatcher crawl URL filtering.
- Dispatcher now skips unsafe search result URLs before crawl and records `unsafe_crawl_url_skipped`.
- MySQL memory preference values are decoded from JSON strings when needed.
- Local RAG fallback answer is normalized through a helper to avoid future mojibake pollution.
- Added regression tests for localhost URL rejection, unsafe dispatcher URL filtering, and preference JSON decoding.
- Verified with `pytest -q` and `python -m compileall app`.

The second major-refactor pass made these changes:

- Added `app/services/parser_cache.py` with a generic `ParserCache` for TTL and bounded-size parser caches.
- Replaced inline parser cache dictionaries and `_cache_put` in `image_query_parser.py` with `ParserCache` instances.
- Added regression coverage for cache capacity eviction and TTL expiry.
- Verified with `pytest -q` and `python -m compileall app`.

The query planner fusion pass made these changes:

- Added `app/services/query_planner.py` so one LLM call can produce intent, retrieval rewrite, entities, and either general or image constraints.
- `chat.py` now uses planner output directly when available; it skips the older scene-specific parser call on planner success.
- Old `parse_general_query_constraints` and `parse_image_search_constraints` remain as fallback paths when the planner is unavailable or low-confidence.
- Fixed weather clarification rewriting so a city already present in the query is not duplicated.
- Added regression tests proving planner image/general paths avoid second parser calls and preserve planner rewrites.
- Verified with `pytest -q` and `python -m compileall app`.

The LLM client deduplication pass made these changes:

- Added `app/services/llm_json_client.py` for shared OpenAI-compatible JSON extraction and POST behavior.
- `query_planner.py` now uses the shared LLM JSON client.
- `image_query_parser.py` delegates its compatibility wrappers to the shared LLM JSON client.
- Planner prompt text was restored to readable Chinese using Unicode escapes to avoid mojibake in model instructions.
- Verified with `pytest -q` and `python -m compileall app`.

## Known Issues

The largest correctness issue is garbled Chinese text in several files. The code is syntactically valid, but semantic quality is degraded in prompts, UI strings, heuristics, and tests. This needs careful restoration from intent/context, not blind replacement.

High-impact files affected by garbled Chinese:

- `app/api/chat.py`
- `app/services/image_query_parser.py`
- `web/`
- README/examples/tests containing Chinese prompts or expected messages

Other likely improvement areas:

- Reduce duplicated intent/constraint parsing logic between API and parser modules.
- Make connector fallback behavior explicit in logs and tests.
- Add focused tests for clarification behavior, image constraints, and SSRF rejection.
- Keep generated model weights and local RAG storage outside Git.
- Review startup scripts after any port/config changes.

## Working Rules For Future Agents

- Read existing code patterns before changing architecture.
- Treat this `Agents.md` as the persistent project handoff file. Update it after meaningful architecture discoveries, Git cleanup decisions, behavior changes, new known issues, or verification results.
- Keep this file concise and current; remove stale statements when they stop being true.
- Prefer narrow edits that preserve current public API behavior.
- Do not commit secrets, model weights, runtime storage, crawled artifacts, or generated Rasa models.
- Do not overwrite `.env`; update `.env.example` when adding new configuration keys.
- Do not revert unrelated local changes.
- Be careful with Chinese mojibake: restore meaning deliberately and verify tests/user-visible behavior.
- Use `pytest -q` for the baseline verification.
- When touching frontend UI, verify text does not overflow and keep the app as a usable tool, not a landing page.
- When touching image proxy or URL fetching code, preserve SSRF protections.
