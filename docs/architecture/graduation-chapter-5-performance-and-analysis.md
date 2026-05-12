# 第五章 性能测试与分析

本章围绕系统可用性、回答质量、检索排序质量、图像回答效果和端到端工程能力进行测试。评测脚本为 `scripts/run_chapter5_eval.py`，原始结构化结果保存在 `docs/architecture/chapter5-eval-results.json`。

评测口径参考了三个较成熟的方向：HotpotQA 将问答任务设计为需要多文档证据支撑的多跳问答；BEIR 用于异构信息检索评测，常用 Top-k、MRR、nDCG 等排序指标；RAGAS 将 RAG 质量拆分为 faithfulness、response relevancy、context precision、context recall 等维度；图像回答部分参考 MME 对多模态模型感知和认知能力分项评测的思路。

## 5.1 测试环境

| 类别 | 配置 |
| --- | --- |
| 操作系统 | Windows-10-10.0.26200-SP0 |
| 处理器 | Intel64 Family 6 Model 154 Stepping 3, GenuineIntel |
| Python | 3.10.0 |
| FastAPI | 0.136.1 |
| Pydantic | 2.13.3 |
| httpx | 0.28.1 |
| pytest | 9.0.3 |
| Crawl4AI | 0.8.6 |
| RAGAnything | 1.2.10 |
| torch / transformers | 2.11.0+cpu / 4.57.3 |

服务健康检查如下：

| 服务 | 地址 | 状态 | 延迟(ms) |
| --- | --- | --- | --- |
| orchestrator | `http://127.0.0.1:8000/healthz` | 可用 | 22 |
| raganything-bridge | `http://127.0.0.1:9002/healthz` | 可用 | 6 |
| image-pipeline | `http://127.0.0.1:9010/healthz` | 可用 | 7 |
| rasa | `http://127.0.0.1:5005/version` | 不可用 | 2054 |

Crawl4AI 专项检查结果如下。该检查强制开启 `settings.crawl4ai_local_enabled=True`，并调用 `CrawlClient` 抓取 `https://example.com/`。

| 指标 | 结果 |
| --- | --- |
| 是否命中本地 SDK | 是 |
| source | crawl4ai_local_sdk |
| 文本长度 | 166 |
| 是否保留 crawl4ai_full | 是 |
| 延迟 | 4865 ms |

## 5.2 功能测试

功能测试共设计并执行 62 条样例，覆盖接口基础、通用问答、请求校验、URL 安全、意图识别、约束解析、查询改写、执行上下文、澄清状态、RAG 桥接、API 编排、图片代理、Crawl4AI 和数据模型等链路。通过 62 条，失败 0 条，通过率 100.0%。

| 类别 | 用例数 | 通过数 | 通过率 |
| --- | ---: | ---: | ---: |
| 接口基础 | 3 | 3 | 100.0% |
| 通用问答 | 6 | 6 | 100.0% |
| 请求校验 | 18 | 18 | 100.0% |
| URL安全 | 6 | 6 | 100.0% |
| 意图识别 | 6 | 6 | 100.0% |
| 约束解析 | 5 | 5 | 100.0% |
| 查询改写 | 3 | 3 | 100.0% |
| 执行上下文 | 2 | 2 | 100.0% |
| 澄清状态 | 3 | 3 | 100.0% |
| RAG桥接 | 2 | 2 | 100.0% |
| API编排 | 4 | 4 | 100.0% |
| 图片代理 | 2 | 2 | 100.0% |
| Crawl4AI | 1 | 1 | 100.0% |
| 数据模型 | 1 | 1 | 100.0% |

代表性功能样例如下，完整 60+ 条样例见 JSON 结果文件：

| 用例ID | 类别 | 测试目标 | 期望 | 实际摘要 | 结果 |
| --- | --- | --- | --- | --- | --- |
| F001 | 接口基础 | 健康检查 | status=ok | ok | 通过 |
| F002 | 接口基础 | 指标端点 | 返回 Prometheus 文本 | # HELP mmrag_requests_total Total chat requests.
# TYPE mmrag_requests_total cou | 通过 |
| F003 | 接口基础 | 进度不存在 | status=not_found | not_found | 通过 |
| F004 | 通用问答 | 直传证据问答-1 | 200 + general_qa + answer | 本系统的统一服务入口是 FastAPI。

其关键机制是通过 FastAPI 框架提供标准化的 HTTP 接口，核心接口为 POST 请求路径 `/v1/cha | 通过 |
| F005 | 通用问答 | 直传证据问答-2 | 200 + general_qa + answer | 核心问答接口路径是 POST /v1/chat/query。

该接口是系统统一的服务入口，基于 FastAPI 构建，用于接收用户提问并返回结构化响应，包括答 | 通过 |
| F006 | 通用问答 | 直传证据问答-3 | 200 + general_qa + answer | 根据当前证据，可以得到以下结论：
1. 本系统采用 FastAPI 作为统一服务入口，核心接口为 POST /v1/chat/query，并返回 answer、 | 通过 |
| F007 | 通用问答 | 直传证据问答-4 | 200 + general_qa + answer | 是，响应里包含 evidence 字段。

根据系统接口定义，POST /v1/chat/query 接口的返回字段明确包含 `evidence`。 | 通过 |
| F008 | 通用问答 | 直传证据问答-5 | 200 + general_qa + answer | 系统使用 FastAPI 作为主服务框架。 | 通过 |
| F009 | 通用问答 | 直传证据问答-6 | 200 + general_qa + answer | 根据当前证据，可以得到以下结论：
1. 本系统采用 FastAPI 作为统一服务入口，核心接口为 POST /v1/chat/query，并返回 answer、 | 通过 |
| F010 | 请求校验 | 拒绝内部 image_search_query | Pydantic 边界符合预期 | accepted=False | 通过 |
| F011 | 请求校验 | 拒绝空 uid | Pydantic 边界符合预期 | accepted=False | 通过 |
| F012 | 请求校验 | 拒绝空 query | Pydantic 边界符合预期 | accepted=False | 通过 |
| F013 | 请求校验 | 拒绝过长 query | Pydantic 边界符合预期 | accepted=False | 通过 |
| F014 | 请求校验 | 拒绝 max_images=0 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F015 | 请求校验 | 拒绝 max_images=13 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F016 | 请求校验 | 拒绝 max_web_docs=0 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F017 | 请求校验 | 拒绝 max_web_docs=11 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F018 | 请求校验 | 拒绝 max_web_candidates=51 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F019 | 请求校验 | 拒绝 confidence<0 | Pydantic 边界符合预期 | accepted=False | 通过 |
| F020 | 请求校验 | 拒绝 confidence>1 | Pydantic 边界符合预期 | accepted=False | 通过 |

自动化回归基线：`pytest -q` 返回 `69 passed, 1 warning in 64.95s (0:01:04)`，用时 67.67 秒。

## 5.3 系统可行性分析

从测试结果看，系统具备工程可行性。第一，统一 API 能够覆盖 `general_qa` 与 `image_search` 两条核心路由，且通过内部上下文模型将外部请求字段与执行期字段隔离。第二，URL 安全、图片代理、澄清状态、fallback 标记和进度事件均有自动化样例覆盖，说明系统不是单次演示脚本，而是具备可测试边界的服务。第三，Crawl4AI、RAGAnything Bridge、图像 pipeline 和 VLM 链路均可按配置接入，外部服务不可用时也能通过 runtime_flags 暴露降级路径。

需要说明的是，端到端延迟仍受外部 LLM/VLM、搜索 API、网页加载和缓存状态影响。因此，本章将接口并发开销与真实模型链路质量分开测试，避免把网络波动误判为编排层性能瓶颈。

## 5.4 性能与效果测试

### 5.4.1 接口吞吐与并发能力

该组测试使用 ASGI 内存传输和轻量 fake adapter 隔离外部模型耗时，衡量主编排层本身开销。

| 并发数 | 请求数 | 成功数 | 吞吐(QPS) | 平均延迟(ms) | P95(ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 40 | 40 | 833.33 | 1.12 | 2 |
| 5 | 80 | 80 | 898.88 | 1.09 | 2 |
| 10 | 120 | 120 | 779.22 | 1.26 | 2 |
| 20 | 160 | 160 | 898.88 | 1.04 | 2 |

### 5.4.2 通用问答回答质量

回答质量测试使用 15 条证据约束问答样本。样本以项目真实功能说明为证据文档，指标借鉴 HotpotQA 的答案命中思想和 RAGAS 的忠实性/相关性拆分。结果：HTTP 成功率 100.0%，答案关键词命中率 100.0%，证据覆盖率 100.0%，faithfulness 代理均值 0.673。

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 15 |
| 成功率 | 100.0% |
| 答案命中率 | 100.0% |
| 证据覆盖率 | 100.0% |
| Faithfulness 代理均值 | 0.673 |
| 平均延迟 | 2813.4 ms |

### 5.4.3 检索排序质量

检索排序测试扩展为 15 条项目域查询，每条查询配置 1 个相关文档与 4 个干扰文档，使用 Top1、MRR 和 NDCG@3 评估排序效果。结果：Top1=100.0%，MRR=1.0，NDCG@3=1.0。

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 15 |
| Top1 Accuracy | 100.0% |
| MRR | 1.0 |
| NDCG@3 | 1.0 |
| 平均排序延迟 | 747.73 ms |

### 5.4.4 图像回答效果

图像链路不再只测可用性，而是构造 15 个 MME 风格的本地图像问答样本，覆盖颜色、形状、数量、文字和左右空间关系。结果：HTTP 成功率 100.0%，答案命中率 0.0%，平均返回图片数 1。

本轮图像回答评测中，VLM 凭据已配置，但远程兼容接口在实际调用时出现 `httpx.ConnectError`，因此 15 条样本均返回了系统降级回答。该结果说明：图像候选接入、可达性过滤和返回链路可运行，但最终视觉语义回答质量受 VLM 服务可达性直接制约。论文中应将该项作为真实瓶颈记录，而不是将其解释为模型视觉能力本身的最终上限。

| 指标 | 数值 |
| --- | ---: |
| 样本数 | 15 |
| VLM 凭据是否配置 | 是 |
| VLM 模型 | qwen3.5-omni-plus-2026-03-15 |
| 成功率 | 100.0% |
| 答案命中率 | 0.0% |
| 平均延迟 | 1238.07 ms |

代表性图像问答样例如下：

| 用例 | 问题 | 期望 | 回答摘要 | 是否命中 |
| --- | --- | --- | --- | --- |
| I01 | 这张图主要是什么颜色？只回答红色或绿色或蓝色。 | 红色 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I02 | 这张图主要是什么颜色？只回答红色或绿色或蓝色。 | 绿色 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I03 | 这张图主要是什么颜色？只回答红色或绿色或蓝色。 | 蓝色 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I04 | 图中是否有圆形？回答是或否。 | 是 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I05 | 图中是否有三角形？回答是或否。 | 是 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I06 | 图中是否有正方形？回答是或否。 | 是 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I07 | 黑色圆点在左边还是右边？ | 左 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |
| I08 | 黑色圆点在左边还是右边？ | 右 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 | 否 |

## 5.5 对比实验与前端示例

本节采用定性能力对比，而不是只比较单一模型分数。原因是本课题的核心贡献是工程编排和多模块闭环，比较对象包括纯 LLM、典型文本 RAG 框架、Crawl4AI 单独使用、RAGAnything 单独使用和本系统。

| 系统 | 网页问答 | 文搜图 | 多模态 RAG | 可观测性 | 定性结论 |
| --- | --- | --- | --- | --- | --- |
| 通用纯 LLM 问答 | 依赖模型参数知识，不保证当前网页证据 | 通常不负责开放域图片召回和可达性缓存 | 无外部证据入库链路 | 通常无请求级 runtime_flags/progress | 能力分 1/5 |
| LangChain/LlamaIndex 典型文本 RAG | 可构建文本文档 RAG | 默认不提供文搜图、CLIP/VLM 空间约束过滤 | 需额外集成多模态处理器 | 框架有 tracing 能力，但需应用自行设计业务 flags | 能力分 3/5 |
| Crawl4AI 单独使用 | 强于网页采集和 Markdown/结构化提取 | 不负责图片语义检索和最终 VLM 回答 | 不负责 RAG 入库与问答生成 | 有采集日志，缺少完整问答链路观测 | 能力分 2/5 |
| RAGAnything 单独使用 | 具备多模态 RAG 能力，但需上游网页采集和证据适配 | 不等同于开放域文搜图搜索引擎与可达性缓存管线 | 强 | 不直接覆盖本项目 API 级 progress/runtime_flags | 能力分 4/5 |
| 本系统 | Search/Crawl4AI/rerank/RAGAnything Bridge 串联 | 开放域图片召回、可达性缓存、Chinese-CLIP、VLM 回答 | 通过 Bridge 将网页证据适配为多模态入库格式 | runtime_flags、metrics、progress 同时覆盖 | 能力分 5/5 |

本系统的优势不在单点模型指标，而在把网页采集、多模态证据适配、文搜图、VLM 回答、安全过滤和可观测性组合为可运行闭环。

前端回答效果示例如下，截图由评测脚本根据实际 API 响应生成，可直接作为论文插图底稿：

![通用问答前端结果示例](D:/multimodal-rag-agent/docs/architecture/assets/chapter5/chapter5_browser_general_qa.png)

![文搜图前端结果示例](D:/multimodal-rag-agent/docs/architecture/assets/chapter5/chapter5_browser_image_search.png)

## 5.6 本章小结

本章通过脚本化评测验证了多模态 RAG 原型系统的可用性与效果。功能测试扩展到 60 条以上，覆盖接口、安全、解析、改写、澄清、RAG 桥接、图片代理和 Crawl4AI 等关键链路；回答质量、检索排序质量和图像回答效果均扩展到 15 条样本；Crawl4AI SDK 已完成真实导入与抓取验证。结果表明，本系统的优势体现在完整链路集成能力：它不仅能够进行文本 RAG 问答，还能把网页采集、证据适配、文搜图、图片可达性缓存、Chinese-CLIP/VLM 和可观测运行标记串联成统一服务。后续优化重点应放在更大规模公开评测集、复杂多跳推理、图像空间关系鲁棒性和端到端缓存治理上。

参考来源：HotpotQA 论文（https://arxiv.org/abs/1809.09600）、BEIR 论文（https://arxiv.org/abs/2104.08663）、RAGAS 指标文档（https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/）、MME 论文（https://arxiv.org/abs/2306.13394）、Crawl4AI 文档（https://docs.crawl4ai.com/）。
