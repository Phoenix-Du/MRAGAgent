# 毕业论文工作文档：正文写作方案

## 1. 论文题目建议

面向通用问答与文搜图场景的多模态 RAG 原型系统设计与实现

## 2. 总体写作策略

第 1 章已完成，后续章节按模板继续写。论文主线不是“提出新模型”，而是“设计并实现一个可运行、可观测、可扩展的多模态 RAG 工程系统”。写作时要突出：

- 统一 API 编排；
- 融合式 LLM query planner；
- 网页搜索、Crawl4AI 解析、BGE 重排、RAGAnything 入库与回答；
- 文搜图的 SerpAPI 召回、图片可达性验证、Chinese-CLIP 粗排、VLM 精排与回答；
- Crawl4AI 与 RAGAnything 被系统深度适配为内部模块；
- runtime flags、progress、metrics 形成可解释运行链路；
- 安全与降级设计保证原型系统稳定运行。

## 3. 第 2 章写作重点

第 2 章为方案论证，重点写“为什么这样做”。

应包含：

1. 系统需求分析：通用问答、文搜图、网页解析、RAG、多模态回答、前端交互、配置与可观测、安全与降级。
2. 可行性分析：技术可行、经济可行、运行可行、维护可行。
3. 开发工具分析：Python、FastAPI、Pydantic、httpx、Crawl4AI、RAGAnything、Chinese-CLIP、Qwen-VL、Rasa、Redis/MySQL、HTML/CSS/JS、pytest。
4. 关键技术分析：RAG、多模态 RAG、网页解析、跨模态检索、LLM 结构化规划、视觉语言模型、SSRF 防护。
5. 基本方案制定：双路线统一编排，总体流程从 query planner 到 adapter，再到 dispatcher 和不同任务链路。

## 4. 第 3 章写作重点

第 3 章为系统设计，重点写“系统由哪些部分组成，数据怎么流动”。

应包含：

1. 功能需求：用户、管理员/开发者两类视角。
2. 总体架构：前端层、API 编排层、服务层、桥接集成层、模型与存储层。
3. 数据模型设计：QueryRequest、QueryResponse、SourceDoc、ModalElement、ImageSearchConstraints、GeneralQueryConstraints。
4. 模块设计：query planner、dispatcher、general_qa、image_search、RAGAnything Bridge、Crawl4AI 采集、memory、progress/metrics。
5. 安全与降级设计：URL safety、image proxy、fallback、runtime flags。

可放 mermaid 图：总体架构图、请求时序图、通用问答流程图、文搜图流程图。

## 5. 第 4 章写作重点

第 4 章为系统实现，重点写“代码如何实现”。

应包含：

1. FastAPI 主服务与前端托管。
2. `chat_query()` 的请求生命周期。
3. LLM planner 与 fallback parser 的实现。
4. 通用问答：search、BGE、safe URL、Crawl4AI、body rerank、RAG client。
5. RAGAnything Bridge：content_list 装配、图片物化、表格转换、Docling 路由。
6. 文搜图：SerpAPI、图片缓存、Chinese-CLIP、VLM 精排、严格空间过滤、VLM 回答。
7. 会话记忆、澄清、前端进度和图片代理。
8. 配置管理、运行脚本和部署。

## 6. 第 5 章写作重点

第 5 章为测试与分析，重点写“如何证明系统可用”。

由于当前仓库已有 pytest 回归测试，可写：

- 测试环境：Windows/Python/FastAPI/pytest，本地服务端口。
- 单元与接口测试：healthz、请求限制、URL 安全、图片代理、parser cache、RAG 本地存储、dispatcher URL 去重、planner 路径等。
- 功能测试用例：通用问答、指定 URL 问答、文搜图、空间约束文搜图、天气澄清、前端进度展示。
- 性能分析指标：端到端延迟、搜索候选数、可达图片数、CLIP 保留数、VLM 精排数、fallback 率。
- 系统界面：前端聊天、意图选择、URL 输入、图片展示、runtime flags 和 progress。

可如实说明：外部 API 与模型依赖会影响真实性能，论文实验可采用本地回归测试与典型人工测试结合。

## 7. 第 6 章写作重点

第 6 章总结系统成果和不足。

成果：

- 完成统一多模态 RAG 编排；
- 实现通用问答和文搜图两条核心链路；
- 实现 Crawl4AI 与 RAGAnything 深度桥接；
- 实现可访问性感知图片检索和 VLM 回答；
- 实现安全校验、降级和可观测性。

不足：

- 依赖外部搜索和模型服务；
- 图像空间关系判断仍受 VLM 能力限制；
- RAG 成功路径 evidence 仍偏弱；
- 评测集规模有限；
- 配置中心仍分为主服务与 bridge 两套。

展望：

- 建立离线评测集；
- 引入统一多模态 embedding；
- 优化 query planner；
- 增强 evidence 和引用；
- 完善缓存治理和 tracing；
- 支持更多文档格式和多轮任务。
