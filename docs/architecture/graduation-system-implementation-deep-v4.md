# 系统实现深度版（版本四）

> 本文档只重写“系统实现”部分，重点是模块内部完整逻辑链路、核心数据结构、算法流程、兜底策略和模块之间的实现衔接。本文不再以源码粘贴为主要表达方式，而是用论文正文、伪代码、流程表和策略表说明系统如何真正运行。

# 4 系统实现

本章在系统设计的基础上，对多模态 RAG 原型系统的工程实现进行详细说明。系统实现的重点不是展示大量源代码，而是说明设计章中的模块边界如何在代码中落地：一次用户请求如何经过主编排、意图规划、澄清、证据准备、证据归一、RAG 入库、图片排序、视觉回答、记忆写回和运行观测，最终形成统一响应。

从代码结构看，系统实现集中在 `app/`、`web/`、`scripts/` 和 `tests/` 四部分。`app/main.py` 创建 FastAPI 主应用；`app/api/chat.py` 是请求编排核心；`app/models/schemas.py` 定义模块间数据契约；`app/services/` 实现搜索、爬虫、RAG、记忆、Query Planner、图片解析和 VLM 图片回答；`app/integrations/` 提供 RAGAnything Bridge、Image Pipeline Bridge、Rasa Bridge 和 RankLLM Bridge；`web/` 提供前端交互；`tests/` 覆盖了请求校验、SSRF 防护、parser 缓存、planner 快路径、dispatcher URL 去重和 RAG fallback 等关键分支。

## 4.1 主应用启动与运行环境实现

系统主应用位于 `app/main.py`。该文件负责完成服务启动的基础装配：创建 FastAPI 实例、配置 CORS、挂载 `/v1` 聊天路由、托管静态前端、提供健康检查和指标接口。

主应用启动逻辑可以概括为：

```text
读取 AppSettings
配置 logging
创建 FastAPI app
注册 lifespan：关闭时释放 MemoryClient 后端连接
添加 CORS 中间件
挂载 chat_router 到 /v1
挂载 web/static
提供 / 首页、/healthz、/metrics
```

配置读取由 `app/core/settings.py` 和 `app/integrations/bridge_settings.py` 共同完成。前者负责主编排服务配置，例如搜索、爬虫、RAG、图像 pipeline、记忆后端、请求超时、RAG 入库超时、图片代理重定向次数等；后者负责桥接服务和模型配置，例如 Qwen/OpenAI-compatible API、RAGAnything 工作目录、解析器、Chinese-CLIP 模型、图片缓存目录、图片召回数量、VLM 精排开关等。

这两个配置类的实现特点是：所有配置都从 `.env` 读取，并允许缺省运行。也就是说，在没有完整外部服务和模型 API 的情况下，系统仍可通过 fallback 路径完成本地测试；当配置齐全时，系统会优先调用真实搜索、Crawl4AI、RAGAnything 和 VLM 服务。

脚本目录提供了配套启动能力。`start_all_stack.ps1` 会依次启动 Rasa、RAGAnything Bridge、图像 pipeline 和主编排服务，并调用健康检查；`start_raganything_bridge.ps1` 会创建虚拟环境、安装 `raganything`、`docling`、FastAPI 等依赖并启动 9002 端口；`start_image_pipeline.ps1` 启动 9010 端口图像 pipeline；`check_stack_health.ps1` 检查主服务、Rasa、RAGAnything Bridge 和图像 pipeline 的健康状态。这些脚本使系统能够从单个 FastAPI 原型扩展为多服务运行栈。

## 4.2 核心数据契约实现

系统内部大量模块都通过 Pydantic 模型通信，因此数据契约是实现层的基础。关键模型并不是“数据库实体”，而是请求生命周期中的阶段性载体。

### 4.2.1 QueryRequest：入口请求契约

`QueryRequest` 是 `/v1/chat/query` 的请求体。它承载用户输入、任务控制参数和场景约束。实现中对它进行了多类边界限制：

| 字段 | 实现约束 | 作用 |
|---|---|---|
| `uid` | 1 到 128 字符 | 绑定会话记忆和偏好 |
| `request_id` | 可选，6 到 128 字符 | 绑定进度事件 |
| `query` | 1 到 4000 字符 | 防止空请求和超长输入 |
| `intent` | `general_qa` 或 `image_search` | 允许调用方强制指定路由 |
| `url` | 最大 2048 字符，必须公开 http/https | 指定网页问答并防止危险 URL |
| `max_images` | 1 到 12 | 控制文搜图规模 |
| `max_web_docs` | 1 到 10 | 控制网页抓取和 RAG 入库规模 |
| `max_web_candidates` | 1 到 50 | 控制搜索召回候选 |

其中 URL 校验直接调用 `is_safe_public_http_url()`，在入口层拒绝非 http/https、localhost 和私有 IP 等地址。这样可以避免危险 URL 进入搜索、爬虫或图片代理。

### 4.2.2 SourceDoc 与 ModalElement：证据契约

`SourceDoc` 是任务分发模块输出的统一证据文档。它可以表示搜索摘要、网页采集结果、用户直传文档，也可以表示图片检索结果。

核心结构如下：

```text
SourceDoc:
    doc_id          文档唯一 ID
    text_content    文本正文或摘要
    modal_elements  图片、表格、公式、通用元素
    structure       表格、链接、网页结构、图片结果类型
    metadata        来源、URL、标题、crawl4ai_full 等
```

`ModalElement` 是多模态元素契约：

```text
ModalElement:
    type        image / table / equation / generic
    url         远程资源 URL
    desc        模态描述
    local_path  本地缓存路径
```

实现中，CrawlClient 会把网页图片转为 `ModalElement(type="image")`，图像 pipeline 会把搜索结果转为图片 ModalElement，RAGAnything Bridge 则根据 ModalElement 再生成 RAGAnything 的 image/table/equation item。

### 4.2.3 NormalizedPayload：执行载荷契约

Adapter 输出 `NormalizedPayload`，这是进入 RAG 或 VLM 回答前的统一载荷。它保留 `query`、`original_query`、`image_search_query` 三个不同语义字段。

这三个字段的实现意义不同：

| 字段 | 用途 |
|---|---|
| `query` | 当前执行 query，通用问答中可能被改写 |
| `original_query` | 用户原始表达，用于最终回答语义保真 |
| `image_search_query` | 文搜图召回 query，只用于图片搜索 |

文搜图中，系统有意不覆盖 `query`，而是把检索改写放入 `image_search_query`。测试中也专门验证了这一点：图片搜索使用 dedicated search query，但用户原始 query 仍保留给最终 VLM 回答。

## 4.3 聊天主编排链路实现

`app/api/chat.py` 中的 `chat_query()` 是系统最核心的实现。它不是普通 controller，而是整个系统的运行状态机。其内部逻辑可以分成九个阶段。

### 4.3.1 请求初始化阶段

请求进入后，系统首先执行：

```text
记录 started 时间
reset_runtime_flags()
requests_total + 1
生成或复用 request_id
progress_start(request_id)
初始化 intent_source、intent_fallback、effective_intent
```

这里 `runtime_flags` 使用 `ContextVar` 保存，保证并发请求之间的 flags 不互相污染。`progress_start()` 会在内存进度表中创建任务项，前端可通过 request_id 轮询。

### 4.3.2 意图规划阶段

如果请求中没有显式 `intent`，系统优先调用 `plan_query(req)`。Planner 成功并且置信度达到阈值时，系统直接采信结果，并保存：

```text
effective_intent
parsed_entities
planned_image_constraints
planned_general_constraints
planner flags
```

如果 Planner 不可用或置信度不足，且允许 Rasa，则进入 Rasa 阶段。Rasa 返回 intent 后，系统还有一层 guardrail：如果 Rasa 判断为 `image_search`，但本地 heuristic 强烈认为是通用问答，则拒绝 Rasa 的 image intent，并记录 `intent_rasa_image_rejected`。这个保护是为了避免天气、对比类 query 被误判为图片搜索。

如果 Rasa 也不可用，系统进入 heuristic。启发式逻辑不是简单关键词判断，而是使用 `_score_intent()` 给两个 intent 加分：

```text
image_search:
    强图片搜索词 * 4
    普通图片词 * 2
    数量 + 张/图片模式 * 3

general_qa:
    分析、解释、总结、对比等强问答词 * 3
    天气、温度、为什么、如何等普通问答词 * 2
    “这张图/图里”等图片理解引用词 * 2
    有 URL/source_docs/images 时额外加分
```

当图片分高于通用问答一定阈值时才走 `image_search`，否则默认 `general_qa`。这种默认策略更稳妥，因为不确定请求走通用问答更容易返回解释或澄清，而误入图片搜索会导致召回方向完全错误。

### 4.3.3 记忆与 pending 澄清恢复阶段

确定初步 intent 后，系统读取 `MemoryClient.get_context(uid)`，取出 preferences。随后调用 `maybe_resolve_pending()` 判断用户是否正在回复上一轮澄清。

处理逻辑为：

```text
if pending exists and current query can resolve it:
    merged_query = original_query + user supplement
    clear pending_clarification
    maybe override effective_intent
else if pending remains unresolved:
    keep pending
```

例如天气澄清中，用户回复城市名后会合并回原始天气问题；图片澄清中，用户回复主体或地点后会追加到原始图片请求。

### 4.3.4 场景约束应用阶段

当最终 intent 为 `image_search` 时，系统使用 Planner 已有的 `planned_image_constraints`，否则调用 `parse_image_search_constraints()`。这体现了“Planner 成功时避免第二次 LLM 调用”的优化。约束解析后，`_apply_image_constraints()` 会：

1. 根据 constraints.count 调整 `max_images`，并限制在 1 到 12。
2. 使用 `search_rewrite` 或结构化约束生成 `image_search_query`。
3. 保持原始 `query` 不变。
4. 把 constraints 写回请求。

当最终 intent 为 `general_qa` 时，系统使用 Planner 已有的 `planned_general_constraints`，否则调用 `parse_general_query_constraints()`。`_apply_general_constraints()` 会将 `search_rewrite` 写入 `query`，并保存 general constraints。

### 4.3.5 约束级澄清阶段

Parser 或 Planner 可能直接返回 `needs_clarification=true`。例如模型判断图片请求缺少主体，或天气请求缺少城市。此时系统不会继续执行证据准备，而是：

```text
add_runtime_flag("clarification_needed")
clarification_needed_total + 1
memory.set_preference(uid, "pending_clarification", {...})
return QueryResponse(answer=clarification_question)
```

这类澄清来自结构化 parser。除此之外，系统还会调用 `should_clarify()` 进行规则级澄清，作为另一层保护。

### 4.3.6 证据准备与归一化阶段

澄清通过后，系统调用：

```text
normalized = adapter.normalize_input(req)
```

该步骤内部会触发 TaskDispatcher 准备通用问答或文搜图证据，并把结果转为 `NormalizedPayload`。完成后，系统统计归一化文档数和图片数，写入 progress。

### 4.3.7 RAG 入库与回答阶段

系统先调用 `adapter.ingest_to_rag(normalized)`。通用问答会执行 RAG 入库，文搜图默认跳过普通 RAG 入库并记录 `image_search_ingest_skipped`。

随后调用 `adapter.query_with_context(normalized)`。通用问答走 RagClient 查询，文搜图走 VLM 图片答案生成。

### 4.3.8 指标与收尾阶段

成功时，系统执行：

```text
requests_success_total + 1
metrics.add_latency(total_latency_ms, rag_latency_ms)
progress_complete(request_id, result summary)
return result
```

失败时，系统记录 `requests_failed_total`，调用 `progress_error()`，并重新抛出异常，让 FastAPI 返回错误响应。

## 4.4 Query Planner 与 Parser 实现

系统的 query 理解分为主路径 Planner 和 fallback parser 两层。Planner 位于请求关键路径，目标是一次 LLM 调用完成意图和检索规划；parser 是 Planner 不可用或低置信度时的后备结构化解析。

### 4.4.1 LLM JSON 客户端

`llm_json_client.py` 封装了 OpenAI-compatible JSON 调用，包括：

1. 从 `bridge_settings` 读取 API key、base URL 和 parser model。
2. 发送 `/chat/completions` 请求。
3. 从 message content 中抽取第一个 JSON object。
4. 去除 markdown code fence 和 JSON 尾逗号。
5. 支持有限重试。

该模块使 Query Planner 和 image/general parser 复用同一套 LLM JSON 抽取逻辑，避免重复实现。

### 4.4.2 Query Planner 规划逻辑

Planner 通过 `_build_prompt()` 构造 JSON-only prompt。Prompt 中包含固定 schema、请求上下文和场景规则。它要求模型输出：

```text
intent
confidence
search_rewrite
entities
general_constraints
image_constraints
```

模型返回后，`plan_from_obj()` 执行结构转换：

```text
normalize intent
clamp confidence
stringify entities
if intent == image_search:
    build ImageSearchConstraints
    flags = ["query_planner_llm", "image_query_rewritten"]
else:
    build GeneralQueryConstraints
    flags = ["query_planner_llm", "general_query_rewritten"]
```

图片约束转换时，系统会解析 spatial_relations、action_relations、object_relations、subject_synonyms、count、landmark 和 time_of_day。count 会被限制到 1 到 12。通用问答约束转换时，系统解析 city、attributes、compare_targets 和 clarification 字段。

### 4.4.3 ParserCache 缓存逻辑

图片 parser 和通用 parser 都使用 `ParserCache`。缓存项保存 `(timestamp, value)`。读取时如果超过 TTL 就删除并返回 None；写入时如果超过最大容量，就按时间戳删除最旧项。

缓存逻辑为：

```text
get(key, now):
    if key missing: return None
    if now - timestamp > ttl:
        delete key
        return None
    return value

put(key, value, now):
    set key = (now, value)
    if size > max_entries:
        delete oldest overflow items
```

测试中专门覆盖了容量淘汰和 TTL 过期，说明 parser 缓存是系统稳定性的一部分。

### 4.4.4 图片约束 parser

图片 parser 的实现是 LLM + heuristic 双路径。

首先根据 `query + entities` 计算 MD5 作为缓存 key。如果命中缓存，直接返回。然后生成 heuristic 结果作为兜底。若没有 API key，则直接缓存并返回 heuristic。

当 LLM 可用时，系统要求模型输出 subjects、subject_synonyms、attributes、count、spatial_relations、action_relations、exclude_terms、must_have_all_subjects、needs_clarification、clarification_question 和 search_rewrite。Prompt 中包含 few-shot 示例，强调把左右关系转成可检索表达，同时保留结构化空间关系供后续 VLM 严格过滤。

如果 LLM 返回坏 JSON 或请求异常，系统返回 heuristic。

heuristic 图片解析包含以下逻辑：

1. 解析数量：支持阿拉伯数字和中文数量词，“几张”默认 5。
2. 解析左右、前后、上下等空间关键词。
3. 用正则识别“左边是 A 右边是 B”这类 pair pattern。
4. 用正则识别简单动作关系。
5. 从 entities 中补 landmark、time_of_day、style。
6. 如果没有主体，则去除填充词后取前几个 token 作为主体。
7. 从主体前缀中提取颜色或状态属性。
8. 构造 search_rewrite：时间、地点、主体、同义词、属性、风格、动作词、“同框/合影/互动”和“照片”。

该 parser 的特点是即使没有 LLM，也能对常见中文图片请求给出可用结构，尤其是空间关系和数量。

### 4.4.5 通用问答 parser

通用问答 parser 也使用缓存和 LLM/heuristic 双路径。heuristic 主要识别城市和比较对象。

城市提取策略包括：

1. 匹配带“市/区/县/州”等后缀的中文地点。
2. 匹配常见热门城市，如北京、上海、广州、深圳、杭州、武汉、成都、重庆、南京、西安、苏州、天津等。

比较对象识别主要根据“对比、区别、哪个好、差异”等关键词，将 query 按“和、与、vs”等分隔，取前两个对象。

LLM parser 则要求输出 search_rewrite、city、attributes、compare_targets、needs_clarification 和 clarification_question。如果 LLM 失败，则回到 heuristic。

## 4.5 澄清与记忆模块实现

### 4.5.1 澄清规则实现

`clarification.py` 中的澄清逻辑分为 weather 和 image_search 两类。

天气类判断：

```text
if intent == general_qa and query contains weather keywords:
    city = entities.city or extract_city(query)
    default_city = preferences.default_city
    if city exists:
        return no ask, maybe rewrite query
    if default_city exists:
        return no ask, rewrite query with default_city
    return ask city
```

图片类判断：

```text
if intent == image_search:
    if no landmark and (generic pattern or query too short):
        ask target subject/location
```

这里的澄清不是对所有不确定请求都追问，而只处理影响执行质量的关键缺槽。

### 4.5.2 pending 状态恢复

`maybe_resolve_pending()` 根据 pending scenario 合并用户回复。

weather scenario：

```text
city = extract_city(current_query) or preferences.default_city
if city:
    merged = original_query if city already exists else city + original_query
    return resolved, merged, intent general_qa
```

image_search scenario：

```text
subject = current_query.strip()
if subject:
    merged = original_query + subject
    return resolved, merged, intent image_search
```

这种实现用较轻量的数据结构实现了多轮交互。

### 4.5.3 MemoryClient 后端实现

MemoryClient 支持四种模式。

memory 模式使用：

```text
_history: dict[uid, deque(maxlen=max_turns)]
_prefs: dict[uid, dict]
```

Redis 模式使用：

```text
history key = {prefix}:history:{uid}
prefs key   = {prefix}:prefs:{uid}

write history:
    LPUSH
    LTRIM 0 max_turns-1

write prefs:
    HSET prefs_key key JSON(value)
```

MySQL 模式启动时自动创建：

```text
user_memory_history:
    id, uid, query_text, answer_text, intent, created_at

user_preferences:
    id, uid, pref_key, pref_value(JSON), updated_at
    unique(uid, pref_key)
```

读取 MySQL 偏好时，系统会对 JSON 字符串执行解码；若不是合法 JSON，则保留原字符串。测试覆盖了这一行为。

hybrid 模式优先 Redis，Redis 没有结果再读 MySQL；写入时 Redis 和 MySQL 都写，同时也写内存 fallback。

## 4.6 Adapter 与任务分发实现

### 4.6.1 MinAdapter 归一化逻辑

MinAdapter 是 API 层和回答层之间的工程边界。它的 `normalize_input()` 首先调用 TaskDispatcher 获取证据，然后执行三层 fallback：

```text
source_docs, prepared_images = dispatcher.prepare_documents(request)
documents = convert each SourceDoc to NormalizedDocument

if documents empty and prepared_images exists:
    create image branch document

if documents still empty:
    create direct query document

return NormalizedPayload
```

这里的 direct query document 保证任何请求即使没有搜索结果或图片结果，也能进入后续链路，不会因为 documents 为空导致流程中断。

### 4.6.2 入库策略

`ingest_to_rag()` 根据 intent 决定是否执行 RAG 入库。文搜图默认跳过普通 RAG 入库，因为当前文搜图最终回答由 VLM 直接基于图片证据生成。跳过时记录 `image_search_ingest_skipped`，并写 progress。通用问答则通过 RagClient 入库，并使用 `with_retry()` 做三次指数退避重试。

### 4.6.3 回答策略

`query_with_context()` 会先读取用户上下文。如果用户偏好中有 `answer_style`，系统把偏好拼接到 query 中。对于 general_qa，系统额外加入中文回答约束；如果判断为天气问题，还加入天气聚焦约束，要求回答只围绕天气现象、温度、降雨、风力、湿度、空气质量等，不扩展城市历史和旅游信息。

回答分支：

```text
if intent == image_search:
    build_image_search_vlm_response()
else:
    rag_client.query()
```

回答完成后，Adapter 更新记忆，并把当前请求的 runtime flags 写入 QueryResponse。

### 4.6.4 TaskDispatcher 通用问答分支

通用问答分支有三条路径：

1. `source_docs` 路径：直接使用用户传入文档。
2. `url` 路径：直接抓取指定 URL。
3. 搜索路径：搜索、摘要重排、URL 过滤、并发抓取、正文重排。

搜索路径完整逻辑：

```text
n_candidates = max(max_web_docs, configured_candidates)
optimized_query = optimize_web_query(query)
search_hits = SearchClient.search_web_hits(optimized_query)
selected_hits = BGERerankClient.rerank(optimized_query, search_hits)

for hit in selected_hits:
    url = hit.metadata.url
    skip if empty or duplicate
    skip if unsafe public URL check fails
    selected_urls.append(url)

crawled = crawl selected_urls with semaphore
if body rerank enabled and crawled > 1:
    crawled = BGE rerank over full body

return crawled[:max_web_docs]
```

Dispatcher 层同时承担 URL 去重和安全过滤责任。测试覆盖了重复 URL 只抓一次，以及 localhost URL 被跳过。

### 4.6.5 TaskDispatcher 文搜图分支

文搜图分支首先确定 `image_query`：

```text
image_query = request.image_search_query or request.query
```

如果请求没有携带图片，则调用 ImagePipelineClient 检索和排序图片，并接收 debug 信息。debug 中包含 provider、fallback_used、query_variants、serpapi key 尝试情况等，会写入 progress。

最后，Dispatcher 将图片结果封装为一个 `SourceDoc`：

```text
doc_id = image_branch::md5(image_query)
text_content = original_query or query
modal_elements = images
structure = {"type": "image_search_result"}
metadata = {"source": "image_pipeline", "image_search_query": image_query}
```

这样图片搜索结果也能通过 Adapter 的统一文档机制继续流转。

## 4.7 搜索、重排与网页采集实现

### 4.7.1 SearchClient 实现

SearchClient 负责网页搜索召回。它支持三层搜索路径：

```text
if serpapi_endpoint configured:
    POST endpoint with query/top_k
    map response
    if fail -> search_fallback

if direct SerpAPI keys exist:
    for each key:
        GET serpapi google search
        skip key on 401/403/429/error
        map first non-empty response
    if all fail -> search_fallback

if placeholder fallback disabled:
    return []
else:
    return example.com placeholder SourceDoc
```

搜索结果映射支持多种返回格式：`hits`、`results`、`organic_results` 或 `urls`。映射后每个结果都是 `SourceDoc`，`text_content` 由 title 和 snippet 组成，metadata 保存 URL、title、snippet、query 和 source。

### 4.7.2 BGERerankClient 实现

BGE 重排使用 `AutoTokenizer` 和 `AutoModelForSequenceClassification` 加载 reranker 模型。模型采用类级缓存，避免每次请求重复加载。

评分流程：

```text
ensure model/tokenizer loaded
for doc in docs:
    text = doc.text_content or metadata.url
    pair = (query, text[:1200])
tokenize pairs with max_length 512
model forward
logits -> scores
sort docs by score desc
return top_k
```

如果模型加载或推理失败，系统记录 `bge_rerank_fallback` 和指标，返回原始前 top_k 文档。这种设计保证重排能力是增强项，而不是单点故障。

### 4.7.3 CrawlClient 实现

CrawlClient 实现了本地 SDK、远程服务和占位 fallback 三层网页采集。

本地 SDK 路径：

```text
try import AsyncWebCrawler
async with AsyncWebCrawler:
    result = crawler.arun(url)
snapshot = serialize CrawlResult excluding pdf bytes
markdown = extract preferred markdown
media = result.media or snapshot.media
tables = result.tables or snapshot.tables
links = result.links or snapshot.links
return crawl4ai-like dict
```

远程路径先尝试 Crawl4AI Docker API schema：

```text
POST endpoint {"urls": [url], "browser_config": {}, "crawler_config": {}}
```

失败后尝试简单 API：

```text
POST endpoint {"url": url}
```

映射为 SourceDoc 时，系统执行：

1. 兼容 Docker 风格 `{"results": [...]}`。
2. 生成稳定 doc_id：`crawl::md5(url)`。
3. 从多种字段提取正文：`text_content`、`content`、`markdown.fit_markdown`、`raw_markdown`、`cleaned_text`。
4. 从 `modal_elements`、`images`、`media.images/videos/audios` 中提取模态元素。
5. 将未知 modal type 降级为 `generic`。
6. 将 title、tables、links 写入 structure。
7. 将 source、url、title 和 `crawl4ai_full` 写入 metadata。

`crawl4ai_full` 是后续 RAGAnything Bridge 混合转换的关键字段。

## 4.8 RAG 客户端与 RAGAnything Bridge 实现

### 4.8.1 RagClient 主服务客户端

RagClient 的实现策略是远程优先、本地兜底。

入库流程：

```text
if rag_anything_endpoint exists:
    POST /ingest with documents and tags
    if success: return indexed ids
    if fail: add rag_ingest_fallback

for each document:
    key = uid::doc_id
    store[key] = doc
trim store to local_rag_store_max_docs
return doc ids
```

查询流程：

```text
if rag_anything_endpoint exists:
    POST /query with query, uid, trace_id
    if success: map QueryResponse
    if fail: add rag_query_fallback

if placeholder fallback disabled:
    raise rag unavailable

candidates = first 3 docs for uid from local store
evidence = snippets from candidates
images = image modal elements from candidates
return local fallback answer
```

本地 store 使用插入顺序进行容量裁剪，超过 `local_rag_store_max_docs` 时删除最旧文档。测试覆盖了该边界。

### 4.8.2 RAGAnything Bridge 初始化

Bridge 启动时尝试导入 LightRAG 和 RAGAnything。如果不可用，则 `_RAGANYTHING_AVAILABLE=False`。真实 RAG 实例只有在 SDK 可用且配置了 API key 时才创建。

初始化时还做两个环境处理：

1. `_ensure_parser_scripts_dir_on_path()`：将 Python scripts 目录加入 PATH，保证 mineru 等 parser CLI 可被子进程找到。
2. `_disable_env_proxies()`：清理 HTTP_PROXY/HTTPS_PROXY 等代理变量，避免本地服务和 DashScope 调用被代理污染。

RAGAnything 实例需要三类函数：

| 函数 | 作用 |
|---|---|
| `llm_model_func` | 文本生成、实体抽取、表格分析 |
| `vision_model_func` | 图片理解和 VLM-enhanced query |
| `embedding_func` | 文本向量化 |

这些函数都通过 OpenAI-compatible API 配置。

### 4.8.3 RAGAnything Bridge 入库流程

Bridge 的 `/ingest` 首先将文档写入 `_fallback_docs[uid]`，作为弱证据缓存。随后如果真实 RAG 实例可用，逐个处理文档。

处理优先级如下：

```text
for doc in documents:
    if hybrid crawl content list exists:
        rag.insert_content_list(hybrid_list)
        continue

    if doc should route html to docling:
        fetch or use html_body
        parse html with docling
        if parsed:
            rag.insert_content_list(parsed)
            continue

    build generic content_list from text and modal_elements
    if content_list:
        rag.insert_content_list(content_list)
```

### 4.8.4 Crawl4AI 混合转换算法

`_build_hybrid_crawl_content_list()` 是 Bridge 中最复杂的实现。它只在 `raganything_crawl_hybrid` 开启，且文档 metadata 中存在 `crawl4ai_full` 时执行。

算法步骤：

```text
full = metadata.crawl4ai_full
structure = metadata.crawl_structure or {}
items = []

html_body = pick fit_html, then cleaned_html, then raw html
if html_body:
    truncate to raganything_max_html_chars
    write to working_dir/html_inputs/{doc_id}_crawl_hybrid.html
    parsed = docling parse html
    if parsed: items.extend(parsed)

if items empty and doc.text:
    add text item from doc.text

md_extra = pick fit_markdown/raw_markdown/markdown_with_citations/extracted_content
if md_extra complements doc.text:
    add markdown supplement text item

tables = structure.tables or full.tables
for each table:
    convert headers/rows/caption to markdown table
    add table item

collect image URLs from doc.modal_elements and full.media.images
for each unique image:
    materialize remote image
    if success:
        add image item with img_path/caption
    else:
        add text fallback

return items or None
```

该算法解决了 Crawl4AI 输出与 RAGAnything 输入不一致的问题。它不依赖单一字段，而是组合 HTML、Markdown、表格和图片，尽量保留网页多模态结构。

### 4.8.5 远程图片 materialize

Bridge 的图片本地化流程为：

```text
reject non-http/https URL
create working_dir/remote_images
GET image with browser User-Agent
require content-type starts with image/
guess extension from content-type or URL path
filename = doc_id + index + md5(url)
write bytes
return local path
```

如果图片下载失败，Bridge 不直接丢弃，而是将其降级为文本 item：`[image] desc url`。这样至少保留图片作为证据信息。

### 4.8.6 Bridge 查询流程

查询时如果真实 RAG 可用，Bridge 调用：

```text
rag.aquery(req.query, mode=raganything_query_mode)
```

证据和图片目前从 `_fallback_docs` 中构造弱 evidence/images，以保证响应结构完整。如果 RAG 查询失败或 RAG 未初始化，Bridge 返回 fallback answer 和弱证据。

## 4.9 图像 Pipeline 与文搜图实现

文搜图链路由三层组成：主服务 ImagePipelineClient、独立 Image Pipeline Bridge、VLM 图片回答模块。

### 4.9.1 ImagePipelineClient

主服务中的 ImagePipelineClient 负责调用 `IMAGE_PIPELINE_ENDPOINT`。它支持多种响应格式：

1. `{"images":[{"url","desc"}]}`
2. `{"results":[{"image_url","caption"}]}`
3. `{"modal_elements":[...]}`

映射时统一生成 `ModalElement(type="image", url, desc, local_path)`。如果远程 pipeline 失败，记录 `image_pipeline_fallback`，返回 placeholder 图片结果。

### 4.9.2 Image Pipeline Bridge 配置与候选模型

Image Pipeline Bridge 使用 `ImageCandidate` 表示候选：

```text
ImageCandidate:
    url
    title
    desc
    source
    score
    local_path
```

它对 top_k、retrieval_k、source_min_accessible、max_check、concurrency、timeout、clip_eval、clip_keep、vlm_rank_pool 等参数都做了最小值和最大值裁剪，防止配置异常导致过大开销。

### 4.9.3 图片搜索召回

Bridge 根据 `image_search_provider` 选择召回供应商。当前主要实现 SerpAPI Google Images。

SerpAPI 调用逻辑：

```text
load SERPAPI_API_KEYS and SERPAPI_API_KEY
dedupe keys preserving order
for each query variant:
    for each key:
        GET serpapi google_images
        if 401/403/429/error: try next key
        map images_results to ImageCandidate
        run accessibility filter
        if accessible candidates exist:
            return candidates and debug
```

debug 信息记录 key 的 mask、index、usable、status_code 和 reason。这样前端 progress 可以展示搜索阶段是否用了 fallback、哪些 query variant 被尝试。

如果 SerpAPI 没有候选，Bridge 使用 Unsplash source URL 作为无 key fallback，并设置 `fallback_used=true`。

### 4.9.4 Query variants 构造

`_build_query_variants()` 会对图片 query 做轻量扩展：

1. 保留基础 query。
2. 如果启用 multi-query，添加“实拍 场景”变体。
3. 对部分中文犬种和空间关系进行英文替换，如 Golden Retriever、Border Collie、together photo、left、right。
4. 去重并限制最大变体数。

这样可以提高开放域图片召回率，尤其是中文 query 在英文图片搜索结果中召回不足时。

### 4.9.5 可达性验证与缓存

图片 pipeline 将可达性验证放在排序前。核心逻辑：

```text
dedupe candidates by url
probe first max_check candidates
for each candidate concurrently:
    if cached file exists:
        touch and return local_path
    else:
        download url
        require content-type image/*
        write to image_cache_dir/md5(url).ext
        candidate.local_path = path
return accessible candidates
```

缓存清理通过 TTL 和 cleanup interval 控制。超过 TTL 的缓存文件会在清理周期内删除。这个机制保证图片可展示，同时避免缓存无限增长。

### 4.9.6 Chinese-CLIP 粗排

Chinese-CLIP 加载时优先尝试本地模型目录 `models/chinese-clip-vit-base-patch16`，再尝试配置的模型名。若 `local_files_only=true` 但本地不存在，会再尝试远程加载。

粗排流程：

```text
load model and processor
choose eval_pool
concurrently open/download images
processor(images)
processor(text=[query])
image_features = model.get_image_features
text_features = text_projection(CLS hidden state)
normalize features
similarity = image_features @ text_features.T
write score to candidates
sort by score desc
filter by min_score; if empty take ranked top_k
return top keep_n
```

如果模型不可用或推理失败，Bridge 使用 `_clip_like_filter()`，即标题/描述词项重合度作为 lexical score。

### 4.9.7 VLM 精排

CLIP 粗排后，Bridge 取 `image_vlm_rank_pool` 大小的 subset 交给 VLM。优先调用 `vlm_rank_clip_pool()`，它会把候选说明和图片内容组织为多模态消息，要求模型输出 JSON：

```text
{"order": [indices by relevance], "irrelevant": [indices]}
```

解析后，系统补全缺失索引，移除 irrelevant，再按顺序取 top_k。如果 VLM 无凭证、调用异常或输出不符合要求，则退回 `_qwen_rerank()`，后者基于候选元数据做文本 JSON 排序；如果仍失败，则使用原排序。

### 4.9.8 最终可达 top_k 保证

VLM 排序后，系统再次调用 `_ensure_accessible_topk()`，把 preferred 和 fallback_pool 合并去重，再进行可达性检测。它优先保留 VLM 选中的可达图片，不足 top_k 时从 CLIP 过滤池补足。

最终返回结构包括 images 和 debug。每张图片包含 url、desc/title、score、source、local_path。

## 4.10 VLM 图片回答实现

最终图片回答由 `image_search_vlm_answer.py` 和 `qwen_vlm_images.py` 完成。

### 4.10.1 图片行收集与可达性过滤

`_collect_image_rows()` 从 NormalizedDocument 的 modal_elements 中收集图片，按 URL 去重，形成：

```text
(url, desc, local_path)
```

随后 `filter_reachable_image_rows()` 确保每张图有可读本地文件。如果 local_path 无法读取，则尝试下载并写入 image_cache_dir。没有本地文件的图片会被过滤掉。

### 4.10.2 VLM 输入构造

`qwen_vlm_images.py` 会把本地图片或远程图片读取为 bytes，然后转换为 data URL。多模态消息由文本段和 image_url 段交替组成。

图片读取优先级：

```text
if local_path readable:
    read bytes from local_path
else:
    fetch image bytes from remote URL
```

这种策略减少远程请求，并利用 image pipeline 已经完成的缓存。

### 4.10.3 联合排序与回答

`vlm_rank_and_answer_from_image_urls()` 用一次 VLM 调用同时完成排序、选择和回答。它要求模型输出：

```text
{
    "order": [按相关性排序的编号],
    "selected": [用于回答的编号],
    "answer": "最终回答文本"
}
```

如果 query 中包含左右等空间关系，prompt 会加入严格提示：只有真正满足空间关系的图片才能进入 selected，不允许在答案中把不满足约束的图片描述为满足。

返回后，系统解析 order 和 selected。order 用于候选排序，selected 用于最终图片集合。如果 JSON 解析失败，则把模型原始文本作为 answer fallback。

### 4.10.4 空间关系严格过滤

联合排序回答之后，系统还会针对空间关系执行二次严格过滤。触发条件是 query 或约束文本中包含 left/right 或中文左右关键词。

严格过滤流程：

```text
strict_pool = selected or ranked top pool
strict_indices = vlm_filter_strict_match_indices(query, strict_pool)
if strict_indices is not None:
    top = strict_pool[strict_indices]
    add image_search_vlm_spatial_filter_applied
    if top empty:
        answer = "候选图片中没有严格满足左右位置约束的结果"
    if top count changed:
        regenerate answer from top
        add image_search_vlm_answer_regenerated_after_spatial_filter
```

该阶段是文搜图准确性的关键，尤其适合“左边是 A、右边是 B”的需求。

### 4.10.5 图片回答降级

如果 VLM 无法生成答案，系统会返回降级说明，并保留图片链接供人工查看，同时记录 `image_search_vlm_answer_degraded`。如果完全没有可用图片，则返回“未找到可用于回答的图片结果”，并记录 `image_search_vlm_no_images`。

## 4.11 图片代理、安全与 SSRF 防护实现

系统有两套 URL 安全逻辑：通用 URL 安全和图片代理安全。

### 4.11.1 通用 URL 安全

`url_safety.py` 实现：

```text
is_safe_public_http_url(url):
    require scheme in http/https
    require netloc
    reject localhost
    if hostname is IP:
        reject private/loopback/link-local/multicast/reserved/unspecified
```

该函数用于 QueryRequest.url 校验和 dispatcher 搜索 URL 过滤。

### 4.11.2 图片代理本地路径安全

本地图片代理允许两个根目录：

1. 图片 pipeline 缓存目录。
2. RAGAnything 工作目录下的 `remote_images`。

本地路径处理：

```text
path must be file
resolved path must be under allowed roots
MIME type must start with image/
read bytes and return
```

这样防止用户通过 `local_path` 读取任意本地文件。

### 4.11.3 图片代理远程 URL 安全

远程代理不直接 `follow_redirects=True`，而是手动处理重定向：

```text
current_url = raw_url
for redirect_count in 0..max_redirects:
    validate current_url scheme and host
    GET current_url without following redirects
    if status not redirect:
        return response
    location = response.location
    current_url = urljoin(current_url, location)
```

host 校验会解析 DNS，并拒绝任何解析到私有、回环、link-local、reserved、unspecified 等 IP 的地址。测试覆盖了公开 URL 重定向到 127.0.0.1 被拒绝的场景。

## 4.12 可观测实现

### 4.12.1 Progress

Progress 使用全局内存字典保存最近请求。每个任务结构包含：

```text
request_id
status
created_at_ms
updated_at_ms
seq
query
intent
events[]
```

事件包含 ts_ms、elapsed_ms、seq、stage、message 和 data。系统最多保留 300 个任务；超过时删除最旧的 50 个。单任务最多保留最近 120 个事件。

关键 stage 包括：

```text
intent.planning
intent.rasa
intent.heuristic
intent.finalized
general_qa.parse_constraints
general_qa.search_hits
general_qa.snippet_rerank
general_qa.urls_selected
general_qa.crawled
image_search.query_ready
image_search.pipeline_done
image_search.answering
pipeline.normalized
pipeline.ingest
pipeline.query
pipeline.answer_done
```

### 4.12.2 Metrics

MetricsStore 使用 dataclass 和 Lock 保存计数器，包括请求数、成功失败数、各类 fallback 次数、澄清次数和延迟累计。`render_prometheus()` 输出 Prometheus 风格文本，便于后续接入监控。

### 4.12.3 Runtime Flags

Runtime flags 使用 ContextVar 保存每个请求的 flags。模块内部通过 `add_runtime_flag()` 增加标记，Adapter 在最终响应中读取 `get_runtime_flags()`。

这套机制比全局变量更适合异步 Web 服务，因为并发请求之间不会共享状态。

## 4.13 前端实现

前端位于 `web/static/app.js`。其实现逻辑围绕请求发送、进度轮询、答案渲染和图片代理渲染。

请求发送时，前端生成 `request_id`，构造 body：

```text
uid
request_id
query
use_rasa_intent
intent_confidence_threshold
max_images
max_web_docs
optional intent
optional url
```

提交后，前端立即创建 thinking panel，并轮询 `/v1/chat/progress?request_id=...`。如果 progress 接口返回运行中，前端渲染阶段事件；如果事件长时间没有变化，会追加 heartbeat 提示；如果 progress 接口不可用，则显示降级提示。

答案返回后，前端渲染：

1. answer 文本。
2. trace_id、route、latency。
3. runtime flags。
4. evidence 片段。
5. images 图片行。

图片渲染优先使用 local_path：

```text
if image.local_path:
    src = /v1/chat/image-proxy?local_path=...
else:
    src = /v1/chat/image-proxy?url=...
```

这与后端图片缓存和代理安全策略配合，提升图片展示稳定性。

## 4.14 辅助 Bridge 与测试实现

### 4.14.1 Rasa Parse Bridge

`rasa_parse_bridge.py` 是轻量 Rasa parse 兼容服务。它提供 `/version` 和 `/model/parse`。在真实 Rasa 服务未启动时，该 bridge 可用关键词推断 image_search 或 general_qa，并返回 Rasa 风格 JSON。正式运行时，`RasaClient` 也可以对接真实 Rasa `/model/parse`。

### 4.14.2 RankLLM Bridge

`rankllm_bridge.py` 提供一个轻量 `/v1/rerank` 接口。当前实现使用词项重合度作为排序分数，输出 RankLLM 风格 artifacts。它主要用于占位或本地联调，不是主通用问答链路的 BGE reranker。

### 4.14.3 自动化测试覆盖

`tests/test_chat_api.py` 覆盖了系统关键行为：

| 测试方向 | 覆盖内容 |
|---|---|
| 基础接口 | `/healthz`、`/v1/chat/query` 响应结构 |
| 请求校验 | max_images 越界、非 http URL、localhost URL |
| 图片代理安全 | loopback 拒绝、重定向到 blocked host 拒绝 |
| Parser | 图片空间约束、中文数量、general city 提取 |
| Cache | ParserCache 容量淘汰和 TTL 过期 |
| RagClient | 本地 store 容量上限 |
| Dispatcher | URL 去重和 unsafe URL 跳过 |
| Memory | MySQL JSON preference 解码 |
| Planner 快路径 | planner image/general 成功时不再二次调用 parser |
| Query 语义保护 | image_search_query 不覆盖用户原始 query |
| Rasa guardrail | 天气对比类 query 不被 Rasa 误导为 image_search |

这些测试说明系统实现重点不仅是“能跑”，还覆盖了安全、缓存、兜底、数据语义和 planner 架构等关键风险点。

## 4.15 两条主链路的端到端实现总结

### 4.15.1 通用问答端到端链路

完整通用问答流程为：

```text
QueryRequest
  -> chat_query 初始化
  -> LLM Planner / Rasa / heuristic 得到 general_qa
  -> parse/apply GeneralQueryConstraints
  -> weather clarification check
  -> TaskDispatcher general_qa_branch
      -> source_docs or url or search path
      -> optimize_web_query
      -> SearchClient search_web_hits
      -> BGERerank snippet rerank
      -> URL safety + dedupe
      -> concurrent CrawlClient.crawl
      -> optional BGE body rerank
  -> MinAdapter normalize
  -> RagClient ingest
      -> RAGAnything Bridge ingest
      -> hybrid content_list conversion
      -> RAGAnything insert_content_list
  -> RagClient query
      -> RAGAnything aquery
      -> QueryResponse
  -> memory update
  -> progress complete
```

该链路中的核心实现点是证据质量递进：搜索保证覆盖，BGE 保证相关，Crawl4AI 保证内容完整，RAGAnything Bridge 保证多模态入库，RAG 查询保证回答基于证据。

### 4.15.2 文搜图端到端链路

完整文搜图流程为：

```text
QueryRequest
  -> chat_query 初始化
  -> LLM Planner / Rasa / heuristic 得到 image_search
  -> parse/apply ImageSearchConstraints
      -> set image_search_query
      -> preserve original query
      -> adjust max_images by count
  -> generic image clarification check
  -> TaskDispatcher image_search_branch
      -> ImagePipelineClient search_and_rank_images
      -> Image Pipeline Bridge
          -> query variants
          -> SerpAPI key rotation recall
          -> accessible filtering + cache
          -> Chinese-CLIP coarse rank
          -> VLM rank or fallback text rerank
          -> ensure accessible top_k
      -> wrap images as image_search SourceDoc
  -> MinAdapter normalize
  -> skip normal RAG ingest by config
  -> build_image_search_vlm_response
      -> collect image rows
      -> ensure reachable local images
      -> constraints_to_prompt_text
      -> VLM rank_and_answer
      -> strict spatial filter
      -> regenerate answer if needed
  -> memory update
  -> progress complete
```

该链路中的核心实现点是图片可靠性递进：先召回，再验证可达，再缓存，再粗排，再精排，再做硬约束过滤，最后生成图片证据回答。

## 4.16 本章小结

本章对系统实现进行了深度梳理。系统实现的核心不是简单地把 FastAPI、Crawl4AI、RAGAnything、Chinese-CLIP 和 VLM 拼接起来，而是在每个连接处设计数据结构、转换逻辑和兜底路径。主编排模块把一次请求组织成可观测的状态机；Query Planner 和 parser 把自然语言转为结构化执行计划；TaskDispatcher 把不同任务转换为统一证据；CrawlClient 把开源网页采集结果映射为系统证据；RAGAnything Bridge 把网页、表格和图片转换为多模态 RAG 入库格式；图像 pipeline 把开放域图片搜索变成可访问、可排序、可解释的图片证据；VLM 图片回答模块进一步处理复杂语义和空间关系；Memory、Progress、Metrics、Runtime Flags 和图片代理则支撑多轮交互、安全访问和调试分析。

通过这些实现，系统形成了一条从用户自然语言请求到多模态证据回答的完整工程链路，并在外部服务不可用、模型输出异常、图片 URL 失效、网页 URL 不安全等情况下提供多级兜底能力。这也是本项目区别于简单模型调用 Demo 的主要实现价值。
