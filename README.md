This project is a production-oriented starter for your full solution:

- FastAPI orchestration service
- Minimal Adapter with 3 core functions:
  - `normalize_input()`
  - `ingest_to_rag()`
  - `query_with_context()`
- Memory service (in-memory now, pluggable to Redis/MySQL)
- RAG service client with remote-first + fallback strategy
- Unified request/response contracts
- Retry wrapper for external calls
- Request-level observability logs (`trace_id`, latency, route, fallback events)
- Task dispatcher with two routes:
  - `general_qa`: URL/search(n) -> BGE(snippet/title) rerank -> select m URLs -> crawl pages -> RAG
  - `image_search`: image pipeline branch

## Quick Start

```bash
cd multimodal-rag-agent
python -m venv .venv
.venv\Scripts\activate
pip install -e .
uvicorn app.main:app --reload --port 8000
```

Optional: copy `.env.example` to `.env` and fill real connector endpoints and memory backend.

## End-to-end tests

Run lightweight API regression tests:

```bash
pytest tests/test_chat_api.py -q
```

The tests cover:

- service health endpoint (`/healthz`)
- chat query contract (`/v1/chat/query`) with runtime flags present

## Docker Compose (template)

This repo now includes a full-stack template at `docker-compose.yml`:

- `orchestrator` (`:8000`)
- `image-pipeline` (`:9010`)
- `rasa-parse` (`:5005`)
- `raganything-bridge` (`:9002`)
- `redis` (`:6379`)
- `mysql` (`:3306`)

Start:

```bash
docker compose up -d --build
```

Check:

```bash
docker compose ps
```

### Integrate cloned open-source services

Repositories are cloned under `third_party/`:

- `third_party/crawl4ai`
- `third_party/rasa_main`
- `third_party/rank_llm`
- `third_party/RAG-ANYTHING`

Quick start helpers:

- Start Rasa parse API: `powershell -ExecutionPolicy Bypass -File scripts/start_rasa_parse.ps1`
- Start RAGAnything bridge API: `powershell -ExecutionPolicy Bypass -File scripts/start_raganything_bridge.ps1`
- Start image pipeline API: `powershell -ExecutionPolicy Bypass -File scripts/start_image_pipeline.ps1`
- Start all dependencies: `powershell -ExecutionPolicy Bypass -File scripts/start_all_stack.ps1`
- Check full stack health: `powershell -ExecutionPolicy Bypass -File scripts/check_stack_health.ps1`

Then set:

- `CRAWL4AI_LOCAL_ENABLED=true` (default, use crawl4ai as in-process SDK)
- Optional remote fallback: `CRAWL4AI_ENDPOINT=http://127.0.0.1:11235/crawl`
- `RASA_ENDPOINT=http://127.0.0.1:5005/model/parse`
- `WEB_SEARCH_CANDIDATES_N=12` (n: search candidates before rerank)
- `WEB_URL_SELECT_M=5` (m: URLs selected after rerank)
- `BGE_RERANKER_MODEL=BAAI/bge-reranker-base`
- `BGE_RERANKER_LOCAL_FILES_ONLY=false`
- `RAG_ANYTHING_ENDPOINT=http://127.0.0.1:9002`
- `IMAGE_PIPELINE_ENDPOINT=http://127.0.0.1:9010/search-rank`

Bridge mode notes (current default for local stability):

- Upstream project is SDK-first (no official fixed `/ingest` + `/query` HTTP contract).
- This repo provides `app/integrations/raganything_bridge.py` as a compatibility bridge for current orchestrator contracts.
- `scripts/start_rasa_parse.ps1` runs `app/integrations/rasa_parse_bridge.py` (Rasa parse-compatible API contract).
- Web rerank is handled by an in-process BGE cross-encoder reranker (`BAAI/bge-reranker-base` by default),
  and it scores search hit summaries/snippets (not full crawled page body).
- This keeps orchestration contracts stable while you iterate locally; you can later swap to real upstream services by replacing endpoints.

### Memory backend

- `MEMORY_BACKEND=memory` (default): in-process memory for local dev
- `MEMORY_BACKEND=redis`: context and preferences in Redis
- `MEMORY_BACKEND=mysql`: context and preferences in MySQL
- `MEMORY_BACKEND=hybrid`: read/write Redis first, fallback to MySQL

When MySQL is enabled, tables are auto-created on first request:

- `user_memory_history`
- `user_preferences`

### RAG Anything endpoint contract

Set `RAG_ANYTHING_ENDPOINT` to your RAG service base URL.

- Ingest call: `POST {RAG_ANYTHING_ENDPOINT}/ingest`
- Query call: `POST {RAG_ANYTHING_ENDPOINT}/query`

Expected request shape:

- ingest: `{"documents":[...], "tags": {...}}`
- query: `{"query":"...", "uid":"...", "trace_id":"..."}`

Accepted response variants:

- ingest: `indexed_doc_ids` or `doc_ids`
- query: standard (`answer/evidence/images`) or compatible (`response/sources/modal_elements`)

### Image pipeline endpoint contract

Set `IMAGE_PIPELINE_ENDPOINT` to your image retrieval/rerank service.

- Call: `POST {IMAGE_PIPELINE_ENDPOINT}`
- Request: `{"query":"...", "top_k":5}`
- Response supports:
  - `{"images":[{"url":"...", "desc":"..."}]}`
  - `{"results":[{"image_url":"...", "caption":"..."}]}`
  - `{"modal_elements":[{"type":"image","url":"...","desc":"..."}]}`

Default local implementation:

- `app/integrations/image_pipeline_bridge.py` provides a real retrieval + ranking chain:
  - Retrieval: SerpAPI Google Images (`SERPAPI_API_KEYS` or `SERPAPI_API_KEY`)
  - Initial filter: Chinese-CLIP image-text similarity (`CHINESE_CLIP_MODEL`)
  - Rerank: direct Qwen cloud API call (`QWEN_API_KEY`, OpenAI-compatible endpoint)
  - Degradation fallback: Unsplash Source query URLs when SerpAPI is unavailable
  - Tunable knobs:
    - `IMAGE_TOP_K_DEFAULT` (default return count when request omits top_k, default `5`)
    - `IMAGE_RETRIEVAL_K` (retrieval count before filtering, default `top_k*5`, cap `50`)
    - `IMAGE_CLIP_EVAL` (max images scored by CLIP, default `top_k*4`)
    - `IMAGE_CLIP_KEEP` (max candidates kept after CLIP, default `top_k*2`)
    - `IMAGE_CLIP_MIN_SCORE` (CLIP minimum score threshold, default `0.18`)

## API

- `GET /healthz`
- `POST /v1/chat/query`

## Observability

- Set `LOG_LEVEL` in `.env` (default `INFO`).
- `chat_query` emits per-request summary logs including:
  - `uid`, `trace_id`, `route`
  - `total_latency_ms`, `rag_latency_ms`
  - `intent_source`, `intent_fallback`
  - `success=true/false`
- Connector and RAG clients emit fallback events when remote calls degrade to local strategies.
- `GET /metrics` exposes lightweight Prometheus-style counters:
  - `mmrag_requests_total`, `mmrag_requests_success_total`, `mmrag_requests_failed_total`
  - `mmrag_intent_fallback_total`, `mmrag_rag_ingest_fallback_total`, `mmrag_rag_query_fallback_total`
  - `mmrag_search_fallback_total`, `mmrag_crawl_fallback_total`, `mmrag_bge_rerank_fallback_total`, `mmrag_image_pipeline_fallback_total`
  - `mmrag_total_latency_ms_sum`, `mmrag_rag_latency_ms_sum`
- `POST /v1/chat/query` response includes `runtime_flags` for request-level degradation traces
  (for example: `intent_fallback`, `search_fallback`, `rag_query_fallback`).

`POST /v1/chat/query` intent behavior:

- If `intent` is provided, it takes precedence.
- If `intent` is omitted and `use_rasa_intent=true`, the service queries `RASA_ENDPOINT`.
- If Rasa is unavailable or confidence is below `intent_confidence_threshold`, it falls back to `general_qa`.

## Example Request

```json
{
  "uid": "u-001",
  "intent": "general_qa",
  "query": "总结这个网页的核心观点",
  "url": "https://example.com/article",
  "source_docs": [],
  "images": []
}
```

## Notes for next steps

1. Replace bridge services with real upstream deployments when environment is stable:
   - Rasa intent service
   - BGE-Reranker service/model
   - RAG Anything production endpoint
2. Tune memory retention and cleanup policies for Redis/MySQL.
3. Add real upstream connectors:
   - Crawl4AI fetch/parser
   - CLIP + QwenVLM image branch

## Current flow

1. Receive `/v1/chat/query`
2. Dispatcher prepares docs by intent route
3. Adapter normalizes docs
4. Ingests into RAG client
5. Queries with user memory context
6. Returns unified response (`answer/evidence/images/trace_id/latency_ms/route`)

