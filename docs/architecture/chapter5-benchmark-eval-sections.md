## 5.4 性能与效果测试

本节按照系统实际贡献重新组织测试内容。网页侧不再采用偏网页截图理解的基准，而是使用 WebSRC 验证网页 HTML、表格、链接、图片元素/资源与布局结构在抓取后是否能被转换为统一证据格式；文搜图侧采用 MSCOCO Karpathy test 的图文检索范式，验证系统能否从候选图片集合中筛选出与文本需求匹配的图片。测试同时加入 Raw HTML、Crawl4AI only、本系统融合方案和本系统 + RAGAnything Bridge 的消融对比，以区分开源组件能力与本项目适配编排能力。

### 5.4.1 接口吞吐与并发能力

该组测试使用 ASGI 内存传输和轻量 fake adapter 隔离外部模型、搜索与网页加载耗时，主要衡量 FastAPI 编排层、请求校验、路由分发和响应封装本身的开销。

| 并发数 | 请求数 | 成功数 | 吞吐(QPS) | 平均延迟(ms) | P95(ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 20 | 958.55 | 0.95 | 1 |
| 5 | 50 | 50 | 1381.14 | 0.66 | 1 |
| 10 | 80 | 80 | 1380.06 | 0.68 | 1 |

结果表明，在外部服务耗时被隔离后，系统编排层具备较低的请求处理开销。真实部署中的性能瓶颈主要来自网页渲染、图片下载、图文模型推理和 RAG 服务，而不是 FastAPI 路由本身。

### 5.4.2 WebSRC 网页多模态解析与结构化适配能力

WebSRC 是面向网页结构阅读理解的数据集，包含 HTML、页面截图、元素 bounding box 和问答标注。本节将其作为网页结构化解析基准，重点评估系统是否能够在保留答案文本的同时，把图片、表格、链接和布局信息转换为项目内部 `SourceDoc`，并进一步转换为 RAGAnything 可消费的 `content_list`。需要说明的是，WebSRC 的图片密度并不高，测试重点是网页结构、多模态资源和布局元数据的保留，而不是网页截图理解。

| 数据集整体统计 | 数值 |
| --- | ---: |
| WebSRC test 可用评测页面数 | 721 |
| 含图片页面比例 | 44.8% |
| 含表格页面比例 | 41.5% |
| 含链接页面比例 | 72.4% |
| 含布局标注页面比例 | 100.0% |
| 平均图片数/页 | 1.251 |
| 平均表格数/页 | 0.4147 |
| 平均链接数/页 | 2.8336 |
| 平均布局框数/页 | 184.7129 |

| 测试指标 | 数值 |
| --- | ---: |
| 分层抽样样本数 | 15 |
| 样本平均图片数 | 2.67 |
| 样本平均表格数 | 0.33 |
| 样本平均链接数 | 2.2 |
| 样本平均布局框数 | 229.87 |
| Raw HTML 答案覆盖率 | 100.0% |
| 本系统融合解析答案覆盖率 | 100.0% |
| SourceDoc 结构化有效率 | 100.0% |
| RAGAnything Bridge 转换有效率 | 100.0% |

| 用例 | 领域 | 页面ID | 问题摘要 | 标准答案 | 图片数 | 表格数 | 链接数 | 布局框数 | Bridge转换 |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| W01 | game | 3800070 | What is this game's launch date? | Oct 05, 2018 | 4 | 0 | 2 | 172 | 是 |
| W02 | game | 3800117 | What is this game's launch date? | Nov 05, 2019 | 4 | 0 | 1 | 172 | 是 |
| W03 | game | 3800039 | What is this game's launch date? | Apr 28, 2020 | 4 | 0 | 1 | 170 | 是 |
| W04 | game | 3800041 | What is this game's launch date? | Sep 30, 2020 | 4 | 0 | 1 | 170 | 是 |
| W05 | game | 3800083 | What is this game's launch date? | Jun 13, 2019 | 4 | 0 | 1 | 170 | 是 |
| W06 | sports | 1600025 | What's the MIN value during 2011-12 in CLE? | 30.5 | 0 | 1 | 4 | 356 | 是 |
| W07 | sports | 1600053 | How many FGM did this player have in PHL when 2016-17? | 6.45 | 0 | 1 | 4 | 356 | 是 |
| W08 | sports | 1600023 | What is the count of GP in SEA when 2007-08? | 80 | 0 | 1 | 4 | 354 | 是 |
| W09 | sports | 1600055 | What's the FGA value during 2007-08 in ATL? | 8.25 | 0 | 1 | 4 | 354 | 是 |
| W10 | sports | 1600072 | What is the count of MIN in MEM when 2008-09? | 30.7 | 0 | 1 | 4 | 352 | 是 |

从测试结果看，Raw HTML 与本系统融合解析在答案文本覆盖率上保持一致，说明结构化转换没有造成关键信息丢失；同时，系统能够将 Crawl4AI 风格输出中的图片、表格、链接和布局信息保存在 `modal_elements`、`structure` 与 `crawl4ai_full` 中，并进一步通过 RAGAnything Bridge 转换为可入库的多模态内容列表。该结果说明本项目的核心价值在于数据适配和链路编排：它不是重新实现 Crawl4AI 或 RAGAnything，而是让网页采集结果能够稳定进入后续多模态 RAG 流程。

### 5.4.3 MSCOCO 文搜图候选筛选能力

文搜图测试采用 MSCOCO Karpathy test shard 构造标准图文检索任务。每个样本以一条人工 caption 作为文本查询，并在 50 张候选图片中检索对应目标图片。评价指标采用图文检索常用的 Recall@1、Recall@5、Recall@10 和 MRR。对比方法包括随机排序、开源 CLIP、开源 BLIP-ITM，以及本系统的“CLIP 粗筛 + Qwen VLM 最终筛选”链路。

| 方法 | 模型/链路 | R@1 | R@5 | R@10 | MRR | 说明 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Random | random permutation | 0.0% | 20.0% | 33.3% | 0.105 | 随机候选顺序基线 |
| CLIP | openai/clip-vit-base-patch32 | 86.7% | 100.0% | 100.0% | 0.922 | 开源图文向量检索基线 |
| BLIP-ITM | Salesforce/blip-itm-base-coco | 86.7% | 100.0% | 100.0% | 0.933 | 开源图文匹配模型基线 |
| System CLIP coarse + Qwen VLM selector | qwen3.5-omni-plus-2026-03-15 | 86.7% | 100.0% | - | 0.911 | 系统文搜图候选筛选链路 |

| 系统链路稳定性指标 | 数值 |
| --- | ---: |
| 返回图片率 | 100.0% |
| VLM 回答降级率 | 66.7% |
| 平均端到端延迟 | 33885.67 ms |
| P95 端到端延迟 | 158921 ms |

| 用例 | 查询描述摘要 | CLIP名次 | BLIP-ITM名次 | 系统返回名次 | 系统返回图片数 | 系统回答摘要 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| I01 | A man with a red helmet on a small moped on a dirt road. | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I02 | A young girl inhales with the intent of blowing out a candle. | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I03 | A man on a bicycle riding next to a train | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I04 | A kitchen is shown with a variety of items on the counters. | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I05 | A wooden ball on top of a wooden stick. | 2 | 1 | 1 | 1 | The selected image displays a collection of wooden kitchen utensils laid out on... |
| I06 | Multiple wooden spoons are shown on a table top. | 1 | 2 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I07 | A bathroom that has a broken wall in the shower. | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I08 | A bathroom with an enclosed shower next to a sink and a toilet. | 1 | 1 | 1 | 1 | The image displays a bathroom featuring an enclosed glass shower stall situated... |
| I09 | people on bicycles ride down a busy street | 1 | 1 | 1 | 5 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I10 | The bathroom is clean and ready to be used. | 1 | 1 | 3 | 3 | The selected images depict bathrooms that appear clean and ready for use. One i... |

结果显示，MSCOCO 候选筛选任务能够更直接对应文搜图功能：系统需要在多个候选多模态资源中选择目标图片，而不是判断单张图片是否正确。CLIP 和 BLIP-ITM 代表开源图文检索/匹配模型的基础能力，本系统在其上增加候选可达性、统一证据封装和 VLM 最终选择，能够给出带解释的图片返回结果。由于最终选择仍依赖外部 VLM，多图推理稳定性会影响最终 Recall@K；因此该实验也暴露了后续优化方向，即增加候选缓存、重试机制和更稳定的粗排/精排协同策略。

### 5.4.4 消融对比实验

| 方案 | 答案文本覆盖率 | 资源保留率 | 布局保留率 | SourceDoc有效率 | Bridge转换有效率 | 结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Raw HTML Text | 100.0% | 0.0% | 0.0% | 0.0% | 0.0% | 只能保留文本，缺少统一多模态结构 |
| Crawl4AI only | 100.0% | 100.0% | 100.0% | 0.0% | 0.0% | 具备采集能力，但输出仍停留在抓取器内部格式 |
| System fusion adapter | 100.0% | 100.0% | 100.0% | 100.0% | 0.0% | 完成项目内部证据结构化，便于后续编排 |
| System + RAGAnything Bridge | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 完成从网页采集到多模态RAG入库格式的闭环 |

消融结果表明，Raw HTML 只能作为文本基线，Crawl4AI only 能提供较完整的采集结果，但仍缺少面向本系统的统一证据对象；本系统融合方案将网页内容转换为 `SourceDoc`、`modal_elements`、`structure` 和 `crawl4ai_full`，解决了下游模块稳定消费的问题；进一步接入 RAGAnything Bridge 后，网页证据能够转换为多模态 RAG 入库格式，形成从网页抓取、结构化适配到多模态问答/文搜图应用的完整链路。

## 5.5 本章小结

本章围绕系统核心贡献重新设计了性能与效果测试。接口吞吐测试表明，在外部模型和网络服务被隔离后，FastAPI 编排层本身开销较低；WebSRC 测试表明，系统能够在保持网页关键答案文本覆盖的同时，将图片、表格、链接和布局信息保存在统一结构化证据中，并通过 RAGAnything Bridge 转换为可入库内容；MSCOCO 图文检索测试则验证了文搜图链路能够在候选图片集合中进行筛选和返回，评价方式更贴近系统真实使用路径。

总体来看，本项目的优势不体现在重新训练通用问答模型、重排模型或单图 VLM，而体现在对开源网页采集、多模态 RAG、图文检索和 VLM 筛选能力的统一封装、适配和编排。实验结果说明，系统已经具备从网页抓取到多模态证据组织，再到问答和文搜图应用的端到端闭环能力。后续工作可继续扩大 WebSRC、WebQA、MSCOCO/Flickr30K 等基准的测试规模，补充图片密集网页的解析评测，并针对外部 VLM 波动引入缓存、重试和更稳定的粗排精排融合策略。
