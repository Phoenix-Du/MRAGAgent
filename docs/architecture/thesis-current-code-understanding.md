# 毕业论文工作文档：当前代码理解

本文档基于当前仓库代码重新梳理系统，不依赖旧版技术参考稿。其用途是为 `graduation-template-outline.md` 后续章节提供事实依据。

## 1. 项目定位

当前项目是一个面向“通用问答”和“文搜图”两类场景的多模态 RAG 原型系统。系统并不是单一模型调用脚本，而是围绕用户自然语言请求建立统一编排层，将意图规划、网页搜索、网页解析、文本重排、多模态图像检索、视觉语言模型回答、RAGAnything 接入、会话记忆、运行进度和前端展示纳入同一个服务。

主服务使用 FastAPI，入口位于 `app/main.py`。核心 API 为 `POST /v1/chat/query`，前端页面由 `web/index.html` 与 `web/static/*` 提供。后端通过 `/v1/chat/progress` 暴露阶段化进度，通过 `/metrics` 暴露 Prometheus 风格指标，通过 `/v1/chat/image-proxy` 为图片展示提供本地或远程代理能力。

## 2. 主运行链路

请求首先进入 `app/api/chat.py` 的 `chat_query()`。该函数完成请求级初始化，包括生成 `request_id`、重置 runtime flags、创建进度任务、统计 metrics。随后进入意图规划：

1. 若请求显式传入 `intent`，直接使用。
2. 若未传入，调用 `app/services/query_planner.py` 的 `plan_query()`，通过一次 LLM JSON 调用同时输出 intent、confidence、search_rewrite、entities，以及 general/image 约束。
3. 若 planner 失败或低置信度，回退到 `RasaClient`。
4. 若 Rasa 仍不可用，则使用 `_heuristic_image_search_intent()` 与 `_heuristic_general_qa_intent()` 兜底。

该设计的重点是“融合式 query planner”：通用问答与文搜图不再先做一次 intent LLM 调用、再做一次场景解析 LLM 调用，而是由 planner 在一次 JSON 输出中完成路由和检索规划。成功时 `chat.py` 会直接复用 planner 生成的 `ImageSearchConstraints` 或 `GeneralQueryConstraints`，跳过旧 parser。

## 3. 数据结构

核心数据结构在 `app/models/schemas.py`：

- `QueryRequest`：统一请求，包括 `uid`、`query`、可选 `intent`、`url`、`source_docs`、`images`、`image_search_query`、`max_images`、`max_web_docs` 等。URL 字段通过 `is_safe_public_http_url()` 做 SSRF 风险控制。
- `ImageSearchConstraints`：文搜图结构化约束，包括 subjects、attributes、subject_synonyms、style_terms、exclude_terms、count、spatial_relations、action_relations、object_relations、clarification 等。
- `GeneralQueryConstraints`：通用问答约束，包括 search_rewrite、city、attributes、compare_targets、clarification 等。
- `SourceDoc`：从搜索、爬虫或用户输入来的原始证据文档。
- `NormalizedDocument` / `NormalizedPayload`：适配层内部统一文档格式。
- `QueryResponse`：统一返回 answer、evidence、images、trace_id、latency_ms、route、runtime_flags。

`image_search_query` 是当前代码中的关键字段。它将“用于检索的改写 query”和“用户原始语义 query”分离，避免文搜图为了召回而把用户原句过度改写后影响 VLM 回答。

## 4. 适配层

`app/adapters/min_adapter.py` 定义 `MinAdapter`，是 API 层和服务层之间的稳定边界。它有三个核心方法：

- `normalize_input()`：调用 `TaskDispatcher.prepare_documents()`，将不同来源的文档和图片统一转换为 `NormalizedPayload`。
- `ingest_to_rag()`：将归一化文档交给 RAG。对于 `image_search`，默认通过 `image_search_ingest_enabled=false` 跳过普通 RAG ingest，因为文搜图最终由 VLM 基于图片直接回答。
- `query_with_context()`：加载用户上下文，对 `general_qa` 调用 `RagClient.query()`，对 `image_search` 调用 `build_image_search_vlm_response()`。

适配层让上层 API 不关心搜索、爬虫、RAG 或 VLM 的具体细节，也让测试可以通过替换 adapter 方法验证接口契约。

## 5. 通用问答链路

通用问答由 `TaskDispatcher._general_qa_branch()` 实现，执行路径如下：

1. 如果用户传入 `source_docs`，直接作为证据源。
2. 如果用户传入 `url`，直接调用 `CrawlClient.crawl()` 抓取指定网页。
3. 若没有直接证据，则调用 `optimize_web_query()` 生成网页检索 query。
4. `SearchClient.search_web_hits()` 搜索候选网页，优先使用配置的搜索接口或 SerpAPI，失败后可按配置降级。
5. `BGERerankClient.rerank()` 对搜索结果标题和摘要做摘要级重排，选出待抓取 URL。
6. URL 去重并通过 `is_safe_public_http_url()` 过滤 localhost、私有 IP 等不安全目标。
7. `_crawl_urls()` 按 `WEB_CRAWL_CONCURRENCY` 并发抓取网页。
8. 如果 `general_qa_body_rerank_enabled=true`，抓取正文后再执行正文级 BGE 重排，减少摘要相关但正文不相关的噪声。
9. 文档进入 `RagClient.ingest_documents()` 和 `RagClient.query()`。

`CrawlClient` 当前优先使用本地 Crawl4AI SDK：`from crawl4ai import AsyncWebCrawler`，抓取结果被转换为 `SourceDoc`。转换时会保留 Markdown、title、media、tables、links 以及完整 `crawl4ai_full` 快照，后续 RAGAnything Bridge 会利用这些结构信息。

## 6. 文搜图链路

文搜图由 `TaskDispatcher._image_search_branch()` 和 `app/integrations/image_pipeline_bridge.py` 协作完成。

主服务阶段：

1. `chat.py` 解析 `ImageSearchConstraints`。
2. `_apply_image_constraints()` 设置 `image_search_query`、`max_images` 和结构化约束。
3. `TaskDispatcher` 调用 `ImagePipelineClient.search_and_rank_images_with_debug()`。
4. 返回图片被包装为 `SourceDoc`，其 `structure.type=image_search_result`。
5. `MinAdapter.query_with_context()` 调用 `build_image_search_vlm_response()`，直接基于图片候选生成答案。

图像 pipeline 阶段：

1. `_search_serpapi_with_debug()` 调用 SerpAPI Google Images 召回候选。
2. `_filter_accessible_candidates()` 对图片 URL 并发探测，下载成功的图片写入 `image_cache_dir`，并回填 `local_path`。
3. `_chinese_clip_filter()` 使用 Chinese-CLIP 对文本 query 和候选图进行粗排；模型不可用时降级为标题/描述词项相关性。
4. `vlm_rank_clip_pool()` 使用 Qwen/OpenAI 兼容 VLM 对 CLIP 后候选进行精排。
5. `_ensure_accessible_topk()` 在最终返回前再次保证结果可访问。
6. 主服务的 `build_image_search_vlm_response()` 再执行 VLM 排序+回答，并在左右空间约束场景下调用 `vlm_filter_strict_match_indices()` 做严格过滤，必要时重答。

该链路的工程重点是“语义相关 + 可展示”。系统不是只返回搜索引擎图片链接，而是把可访问性、本地缓存、CLIP 粗排、VLM 精排、空间约束验证和前端代理展示串成完整流程。

## 7. RAG 客户端与降级

`app/services/rag_client.py` 采用远程优先、本地兜底策略。若配置了 `RAG_ANYTHING_ENDPOINT`，ingest/query 分别调用 `{endpoint}/ingest` 和 `{endpoint}/query`。远程失败时打 runtime flag，并使用本地 `_store` 保存最近文档。`local_rag_store_max_docs` 限制本地兜底文档数量，防止长期运行时内存无限增长。

RAGAnything Bridge 位于 `app/integrations/raganything_bridge.py`，提供固定 HTTP 契约 `/ingest` 与 `/query`，内部调用 RAGAnything SDK 的 `insert_content_list()` 和 `aquery()`。它还维护 `_fallback_docs`，即使 SDK 初始化失败，也能返回弱证据和图片列表，保证系统演示与测试阶段可运行。

## 8. 记忆、澄清与可观测性

`MemoryClient` 支持 memory、redis、mysql、hybrid 四种后端。它保存历史问答和用户偏好，当前也用于保存 `pending_clarification`。`clarification.py` 针对天气城市缺失、泛化图像请求等场景生成追问，并在用户下一轮回复时合并原始问题。

可观测性由三部分组成：

- `runtime_flags`：请求级路径标记，如 `query_planner_llm`、`intent_fallback`、`general_qa_body_rerank`、`image_search_vlm_spatial_filter_applied`。
- `metrics`：累计请求数、成功失败数、fallback 次数、延迟总和。
- `progress`：前端可轮询查看阶段化进度，覆盖 intent planning、search、crawl、rerank、image retrieval、answering 等阶段。

## 9. 安全与鲁棒性

当前代码中比较明确的安全和鲁棒性设计包括：

- 请求 URL 校验：拒绝非 http/https、localhost、私有地址、保留地址等。
- 图片代理 SSRF 防护：远程图片代理每次请求和每次重定向后都重新校验 host/IP。
- 图片本地路径白名单：只允许访问 image cache 和 RAGAnything remote_images 目录。
- 请求字段边界：`uid`、`query`、`max_images`、`max_web_docs`、`max_web_candidates` 均有 Pydantic 限制。
- 外部 HTTP 调用使用 `trust_env=False`，减少环境代理污染。
- 外部服务失败后通过 runtime flags 标记降级路径。

## 10. 可作为论文贡献的工程点

1. 融合式 LLM query planner：一次输出 intent、检索改写和场景约束。
2. 双路线统一编排：`general_qa` 和 `image_search` 共用统一 API、adapter、response。
3. Crawl4AI 到 RAGAnything 的结构化桥接：保留网页 Markdown、HTML、表格、图片和结构快照。
4. 可访问性感知文搜图：搜索结果先验证和缓存，再进入 CLIP/VLM。
5. 多阶段图像排序：SerpAPI 召回、Chinese-CLIP 粗排、VLM 精排、空间约束严格过滤。
6. 运行可解释性：runtime flags、progress 和 metrics 共同支撑调试与实验分析。
