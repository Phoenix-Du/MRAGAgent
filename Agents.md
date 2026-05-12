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
   - `QueryRequest` is now the external API request model only. Internal execution fields such as `original_query`, `image_search_query`, `image_constraints`, and `general_constraints` are rejected at the API boundary.
   - `uid`, `query`, limits, confidence, and URL shape are bounded.
   - URL inputs must be public `http`/`https`; localhost/private/link-local targets are rejected.
2. Reset request runtime flags and initialize request metrics.
3. Resolve pending clarification state from memory before planning.
   - User replies to prior clarification questions are merged back into the original query before Query Planner sees the request.
   - New pending state is lightweight: `type`, `route`, `original_query`, `question`, `missing`, and `created_at`.
   - User context should prefer `preferences.profile.location` and `preferences.response.style`; old `default_city` and `answer_style` are read only for backward compatibility.
4. Determine intent and retrieval plan.
   - If the client explicitly passes `intent`, use it.
   - Otherwise call `app/services/query_planner.py`.
   - The planner makes one LLM call and returns `intent`, `confidence`, `search_rewrite`, `entities`, and either `general_constraints` or `image_constraints`.
   - The planner prompt can receive generic user context hints from `profile`, `response`, and `retrieval` preferences.
   - On planner success, `chat.py` applies those constraints directly and does not call the older scene-specific parser again.
   - If planner fails, returns bad JSON, or is low-confidence, the code falls back to Rasa and then heuristics.
5. Apply route-specific constraints.
   - `chat.py` creates a `QueryExecutionContext` after intent is known. This internal model carries `original_query`, route-specific query rewrites, and constraints.
   - `general_qa`: LLM-derived `general_constraints.search_rewrite` is the primary retrieval query. `TaskDispatcher` only uses local `optimize_web_query()` as a heuristic fallback when there is no LLM rewrite or constraints came from the heuristic parser.
   - `image_search`: `image_constraints.search_rewrite` becomes `image_search_query`, while the original user query is preserved for answering context. When structured image constraints exist, `chat.py` now consumes `search_rewrite` directly and does not run a second image-query optimizer.
6. Run clarification checks.
   - `app/services/clarification.py` uses LLM planner/parser `needs_clarification` as the primary signal.
   - Local clarification logic now only manages unified pending state and deterministic fallback for missing weather city or overly generic image requests.
   - Clarification responses complete progress state and return before retrieval.
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
- `app/services/clarification.py`: unified clarification state machine. LLM planner/parser results are the primary clarification signal; local logic only manages lightweight pending state, generic profile location resolution, and deterministic fallback.
- `app/core/url_safety.py`: shared public URL safety checks used by schema validation and dispatcher filtering.

## Current Architecture Snapshot

The system should be understood as a multimodal RAG product prototype with two first-class user routes, not as a thin wrapper around models.

Core request lifecycle:

1. Frontend or API client sends `QueryRequest` to `POST /v1/chat/query`.
2. `chat.py` creates a `request_id`, starts progress tracking, resets runtime flags, and keeps `QueryRequest` as external input only.
3. `clarification.py` resolves any existing pending clarification reply before intent planning, so the planner receives a complete merged query.
4. `query_planner.py` is the preferred path when no explicit intent is provided. One LLM JSON call should produce both intent and retrieval plan.
5. `clarification.py` can pause execution based on planner/parser clarification fields, with deterministic fallback only for missing weather city or overly generic image requests; pending clarification state is stored in memory preferences.
6. `QueryExecutionContext` is created after intent resolution and becomes the internal execution carrier for rewritten queries and constraints.
7. `MinAdapter` is the public internal boundary:
   - normalize request into `NormalizedPayload`
   - ingest prepared docs when needed
   - query RAG or build image VLM answer
8. `TaskDispatcher` prepares route material:
   - `general_qa`: direct docs/URL or planned retrieval query -> search hits -> BGE snippet rerank -> safe URL filtering -> concurrent Crawl4AI crawl -> optional body rerank.
   - `image_search`: supplied images or image pipeline retrieval -> wraps results into one image-search `SourceDoc`. The image pipeline now stops after search/source fallback, URL dedupe, reachability/cache, and Chinese-CLIP filtering; VLM ranking is reserved for final image answering.
9. Final response must include `answer`, `evidence`, `images`, `trace_id`, `latency_ms`, `route`, and `runtime_flags`.

Key architectural decisions:

- `query` and `image_search_query` are intentionally separate. Preserve the user's original semantic query for answering, and use `image_search_query` only for retrieval.
- `general_qa` retrieval query ownership belongs to the Query Planner/parser layer. The dispatcher consumes the planned `search_rewrite`; its local optimizer is a fallback, not a second normal rewrite pass.
- `image_search` retrieval query ownership also belongs to the Query Planner/parser layer. `TaskDispatcher` only consumes `image_search_query`; local image query optimization is limited to the old no-constraints entity fallback path.
- `image_search` avoids duplicate VLM ranking. The pipeline no longer runs a CLIP-after VLM rerank stage because `build_image_search_vlm_response()` already performs VLM selection, strict spatial filtering, and answer generation.
- `image_search` defaults to skipping normal RAG ingest (`image_search_ingest_enabled=false`) because the current final answer is generated directly from image evidence by VLM.
- `general_qa` is evidence-first: search/crawl/rerank material is sent into the RAG layer before answer generation.
- Runtime flags are part of the debugging contract. When adding fallback, filtering, or special routing behavior, add a clear flag and, if applicable, a metric.
- `progress_event()` is the frontend-visible execution trace. Long or user-visible stages should emit progress events with concise Chinese messages.

## Thesis Framing

For current thesis writing, frame Crawl4AI and RAGAnything as open-source systems that this project understands, integrates, adapts, and orchestrates. Do not describe them as fully self-developed project modules unless the user explicitly changes this again.

Current framing: Crawl4AI is an open-source webpage crawling/collection system integrated by this project; RAGAnything is an open-source multimodal RAG system integrated through the project-owned RAGAnything Bridge. The project's own thesis contribution is orchestration, data adaptation, bridge conversion, reliability handling, image pipeline integration, safety hardening, and end-to-end productization.

When writing or revising thesis content:

- Explain the open-source system's own architecture and algorithms first.
  - Crawl4AI: `AsyncWebCrawler`, `BrowserConfig`, `CrawlerRunConfig`, browser rendering, cache validation, HTML cleanup, Markdown generation, link citations, media extraction, table scoring, BM25 content filtering, pruning filter, LLM extraction, memory-adaptive dispatcher, and deep crawling strategies.
  - RAGAnything: `RAGAnythingConfig`, `insert_content_list()`, text/multimodal separation, context-aware modal processors, image/table/equation/generic processors, chunk/entity/vector/knowledge-graph insertion, LightRAG hybrid query, multimodal query, and VLM-enhanced query.
- Then describe this project's own integration work: `CrawlClient` SDK/HTTP wrapping and `SourceDoc` mapping, `crawl4ai_full` preservation, `raganything_bridge.py` `content_list` conversion, remote image materialization, table conversion, weak fallback evidence, image pipeline orchestration, progress/metrics/runtime flags, and security hardening.
- Avoid wording like "the system simply calls Crawl4AI/RAGAnything". Preferred wording: "本项目基于 Crawl4AI 的网页采集能力进行了统一封装和结构化转换..." and "本项目通过 RAGAnything Bridge 将内部证据文档适配为 RAGAnything 多模态入库格式...".

Current thesis-related source of truth:

- Final thesis target: `docs/architecture/graduation-template-outline.md`
- Expanded design/implementation draft: `docs/architecture/graduation-system-design-and-implementation.md`
- Preferred rewritten design/implementation draft: `docs/architecture/graduation-system-design-and-implementation-v3.md`
- Deep implementation-only draft: `docs/architecture/graduation-system-implementation-deep-v4.md`
- Current-code understanding: `docs/architecture/thesis-current-code-understanding.md`
- Web collection deep design: `docs/architecture/thesis-crawl4ai-deep-design.md`
- Multimodal RAG engine deep design: `docs/architecture/thesis-raganything-deep-design.md`
- Writing plan: `docs/architecture/thesis-writing-plan.md`

The older `docs/architecture/thesis-project-reference.md` and other architecture notes may be useful historical material, but current thesis writing should be grounded in the actual current code plus the four thesis docs above.

For implementation-section rewrites where the user asks for module-internal logic chains, fallback strategies, and concrete algorithm/data-structure detail, prefer `docs/architecture/graduation-system-implementation-deep-v4.md` over the lighter implementation section in v3.

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
After the clarification state-machine pass, the baseline test suite passed with 27 tests:
After the request/context model split pass, the baseline test suite passed with 28 tests:
After the generalized constraints pass, the baseline test suite passed with 28 tests:
After the retrieval-query ownership pass, the baseline test suite passed with 30 tests:
After the image retrieval-query ownership pass, the baseline test suite passed with 32 tests:
After the image pipeline rerank simplification pass, the baseline test suite passed with 33 tests:
After the general QA RAG connectivity pass, the baseline test suite passed with 34 tests:

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

The clarification state-machine pass made these changes:

- Refactored `app/services/clarification.py` so LLM planner/parser `needs_clarification` is the primary signal.
- Local clarification logic now manages unified pending state, default-city/preference resolution, backward-compatible pending scenario names, and deterministic fallback only.
- `chat.py` now resolves pending clarification replies before planner/Rasa/heuristic routing, so the planner sees the merged full query.
- Removed duplicated clarification return blocks from `chat.py`; clarification responses now go through one helper and call `progress_complete`.
- Added regression tests for planner-driven clarification, pending-before-planner query merge, and deterministic generic-image clarification fallback.
- Verified with `pytest -q` and `python -m compileall app`.

The request/context model split pass made these changes:

- Simplified `QueryRequest` into an external API input model and forbids internal execution fields at the API boundary.
- Added `QueryExecutionContext` for backend-only state: `original_query`, `image_search_query`, route-specific constraints, and execution limits.
- Updated `chat.py`, `MinAdapter`, and `TaskDispatcher` to pass the internal context after intent resolution instead of mutating `QueryRequest`.
- Simplified persisted `pending_clarification` to `type`, `route`, `original_query`, `question`, `missing`, and `created_at`; old pending keys are still readable for compatibility.
- Replaced `default_city` as the preferred model with generic `preferences.profile.location`; `answer_style` is now `preferences.response.style`, with legacy reads retained.
- Query Planner can receive generic user context hints from `profile`, `response`, and `retrieval` preferences.
- Added regression coverage proving internal execution fields are rejected in `QueryRequest`.
- Verified with `pytest -q` and `python -m compileall app`.

The generalized constraints pass made these changes:

- Replaced `GeneralQueryConstraints.city` as a primary field with generic `entities.location`; `.city` remains a compatibility property.
- Reworked `ImageSearchConstraints` around canonical `entities`, `visual_attributes`, `relations`, and `negative_constraints`.
- Added `ConstraintRelation` with `type`, `relation`, `subject`, and `object` so spatial/action/object relations share one structure.
- Kept compatibility properties such as `subjects`, `landmark`, `spatial_relations`, `action_relations`, and `exclude_terms` for existing downstream ranking and VLM code.
- Updated Query Planner and fallback parsers to produce the generalized constraint structures.
- Added regression assertions for `entities.location`, image `entities.subjects`, and unified `relations`.
- Verified with `pytest -q` and `python -m compileall app`.

The retrieval-query ownership pass made these changes:

- `TaskDispatcher` now consumes LLM planner/parser `general_constraints.search_rewrite` directly for `general_qa` retrieval.
- Local `optimize_web_query()` is now only used as a heuristic fallback when no LLM rewrite exists or constraints came from the heuristic parser.
- Added regression tests proving LLM rewrites are not optimized a second time and heuristic constraints still use fallback optimization.
- Verified with `pytest -q` and `python -m compileall app`.

The image retrieval-query ownership pass made these changes:

- `chat.py` now applies `ImageSearchConstraints.search_rewrite` directly into `QueryExecutionContext.image_search_query` when image constraints exist.
- Structured image-search constraints no longer call the local image query optimizer as a second rewrite step.
- Local `optimize_image_query()` remains only for the no-constraints entity fallback path.
- Added regression tests proving `_apply_image_constraints()` does not call the image optimizer when constraints are present.
- Verified with `pytest -q` and `python -m compileall app`.

The image pipeline rerank simplification pass made these changes:

- Removed the image pipeline's CLIP-after VLM rerank stage from `search_rank()`.
- Removed the now-unused `vlm_rank_clip_pool()` helper; VLM image ranking is consolidated in final answer generation.
- Removed `IMAGE_VLM_RANK_ENABLED` and `IMAGE_VLM_RANK_POOL` from bridge settings and `.env.example`.
- `image_pipeline_bridge.py` now returns Chinese-CLIP filtered, reachable images directly; final VLM ranking/filtering remains in `build_image_search_vlm_response()`.
- Added regression coverage proving the image pipeline does not call `vlm_rank_clip_pool()` after CLIP filtering.
- Verified with `pytest -q` and `python -m compileall app`.

The general QA RAG connectivity pass made these changes:

- Installed `raganything` into the active Python runtime and verified RAGAnything Bridge health reports `raganything_ready=true`.
- Verified the configured OpenAI-compatible LLM endpoint can return valid JSON for planner/parser calls.
- Tightened general QA clarification: LLM/parser `needs_clarification=true` only blocks general QA when it maps to a deterministic missing slot such as weather city; generic non-weather QA is allowed through.
- Updated `raganything_bridge.py` so queries with current uid-scoped ingested documents first generate from those documents via the configured LLM, avoiding polluted answers from old global RAGAnything storage.
- Added regression coverage for ignoring non-weather general QA clarification signals.
- Verified a full `general_qa` request now returns evidence-grounded Chinese content without `rag_ingest_fallback` or `rag_query_fallback`.
- Verified with `pytest -q` and `python -m compileall app`.

The general QA evidence and latency pass made these changes:

- `CrawlClient` now has an HTTP HTML/text fallback before placeholder crawl output. It follows safe public redirects manually, extracts title/body/images from real HTML, and labels evidence with `source=http_fallback_crawl`.
- `SearchClient` now has a no-key DuckDuckGo HTML search fallback before placeholder search results. This keeps automatic web QA useful when SerpAPI or the configured search endpoint is unavailable.
- `TaskDispatcher` now overselects crawl candidates, crawls a small candidate pool, and drops placeholder crawl docs when any real crawled document is available.
- `raganything_bridge.py` now defaults to `RAGANYTHING_FULL_INGEST_ENABLED=false`; interactive requests cache uid-scoped evidence immediately and can still enable full RAGAnything insertion through config.
- The bridge now falls back to an extractive answer from uid-scoped evidence if the configured generation model is unavailable, instead of returning a fixed generic fallback string.
- `chat.py` skips general LLM parsing for explicit `general_qa` requests with provided `source_docs` or `url`, because retrieval rewrite is not needed when evidence is already supplied.
- If the unified Query Planner was attempted and failed or timed out, legacy parsers are forced into heuristic mode to avoid a second LLM parser call on the fallback path.
- Verified with `pytest -q` and `python -m compileall app`; baseline is now 41 tests.
- Manual checks after restarting local services: source-doc QA returned in about 2.5s with `general_query_context_direct`; no-intent weather clarification returned in about 0.5s; no-intent FastAPI web QA returned real crawled evidence and a grounded answer in about 27s.

The automatic web QA answer-quality pass made these changes:

- Added search-candidate quality scoring in `TaskDispatcher`: official/product domains, documentation paths, title/domain/query-term matches, and HTTPS are promoted; low-quality blog/social/Q&A domains are demoted.
- Search candidate selection now merges BGE snippet rerank with deterministic source-quality ranking, instead of relying on one ranking signal.
- When LLM `search_rewrite` differs from the original user question, `TaskDispatcher` can run a small original-query supplement search and rank evidence against the combined retrieval/original query. This protects answer quality when a rewrite becomes too narrow.
- For common technical topics, the dispatcher can add official-site supplemental search and curated official seed URLs when search results miss official docs. Current seeds cover Python list comprehensions, Docker, Redis, Kubernetes, OAuth 2.0, and FastAPI.
- Overview-style questions get an extra ranking preference for overview/introduction/get-started paths and a mild demotion for very deep documentation paths, so broad “what is / used for” questions do not overfit to one narrow subpage.
- Crawl failures for high-quality search hits can now fall back to `search_result_evidence`, so an official search snippet can still beat a low-quality blog body when the official page is temporarily unreachable.
- Body rerank no longer truncates candidates before evidence-quality ranking; final evidence selection happens after source-quality scoring.
- Crawled/derived evidence now receives `evidence_quality_score`, and the RAG bridge sorts uid-scoped context documents by that score.
- RAG bridge prompts no longer expose raw `doc_id`, `source_type`, `quality_score`, or URL fields to the model. Evidence blocks are now ordered by quality and presented in natural form to reduce metadata leakage in final answers.
- Generated answers are sanitized for internal metadata when the user did not ask for sources.
- Extractive fallback answers were changed from debug-style "model unavailable" output to user-facing evidence-based summaries, with localized handling for common technical snippets such as FastAPI, Docker, Redis, OAuth, Kubernetes, RAG, and Python list comprehensions.
- Extractive fallback now selects query-relevant evidence sentences and filters common navigation/sidebar noise before forming the answer.
- `CrawlClient` now bounds embedded Crawl4AI SDK calls with `CRAWL4AI_LOCAL_TIMEOUT_SECONDS` before falling back to HTTP extraction, so slow browser rendering does not stall the whole QA chain.
- Shared LLM JSON calls now use the normal request timeout as the read upper bound, preventing parser calls from blocking a request for very long periods before deterministic fallback can run.
- Non-technical general QA was tested with sleep quality, tomato-and-egg cooking, photosynthesis, sea-water salinity, used-car buying, Hangzhou travel, self-study, and household budgeting. This exposed weak evidence behavior for lifestyle/advice queries.
- DuckDuckGo HTML fallback now filters DuckDuckGo self links such as feedback/help pages.
- Placeholder crawl detection now treats any `Fetched content from ...` text as placeholder evidence, regardless of connector source label.
- Search-result snippet evidence no longer includes raw URLs in its text body, reducing answer contamination.
- General-domain authority scoring now promotes trusted government, education, health, science, and consumer-finance sources such as `.gov`, `.edu`, `gov.cn`, `edu.cn`, `cdc.gov`, `nih.gov`, `who.int`, `sleepfoundation.org`, `consumerfinance.gov`, `britannica.com`, and similar domains.
- Advice-style questions can trigger supplemental retrieval using checklist/steps/risk terms. Used-car queries add focused terms such as accident car, flood car, transfer, contract, and vehicle inspection.
- Used-car buying now has a consumer-protection official seed source and query-specific scoring that promotes before-buying inspection/contract evidence while demoting after-purchase maintenance and insurance-only pages.
- Extractive fallback has a structured used-car advice template, so if generation is unavailable it returns a purchase checklist rather than raw webpage excerpts.
- HTTP text cleanup now removes common unsupported-browser boilerplate before evidence snippets are shown.
- Verified with `pytest -q` and `python -m compileall app`; baseline is now 69 tests.
- Manual FastAPI web-QA check now selects FastAPI official documentation evidence and returns a grounded answer about FastAPI being a Python type-hint-based framework for building APIs.
- Manual multi-query checks after this pass covered Python list comprehensions, Docker, Redis, OAuth 2.0, RAG, and Kubernetes. Redis and Kubernetes now return grounded Chinese answers without internal metadata leakage; Kubernetes completed in about 20s after timeout tightening. Python list comprehension uses official-site supplement/seed behavior and returns a structured Chinese answer.
- Manual non-technical checks after this pass showed sleep quality and household budgeting now return useful answers without placeholder/DuckDuckGo self evidence. Used-car buying now returns a purchase-before-signing checklist with accident/history, independent inspection, test drive, contract, warranty, add-on fees, and insurance-history cautions.

## Known Issues

Current likely improvement areas:

- Some historical docs are stale. Prefer current code and the newer `thesis-*.md` docs for thesis/project understanding.
- Chinese mojibake was largely repaired in core user-visible paths, but old examples/docs may still contain garbled text. Restore meaning deliberately if touched.
- Reduce duplicated intent/constraint parsing logic between API and parser modules.
- Make connector fallback behavior explicit in logs and tests.
- Add focused tests for more image constraint edge cases and SSRF rejection variants.
- Full RAGAnything insertion is now opt-in for interactive bridge calls; if enabling it, expect first-call latency and verify background/async behavior.
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
