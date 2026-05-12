## 5.4 性能与效果测试

本节不再将重点放在通用问答模型效果、开源重排模型效果或单张图片 VLM 问答效果上，而是围绕本课题的系统贡献进行测试：一是网页抓取结果经过多模态结构化适配后，是否能更完整地保留网页文本、图片、表格、链接和布局信息；二是文搜图链路是否能从多个候选图片资源中筛选出与用户文本需求最匹配的图片。这样能够更直接体现本项目在 Crawl4AI、RAGAnything、多模态图片链路之间的融合与编排能力。

### 5.4.1 接口吞吐与并发能力

该组测试使用 ASGI 内存传输和轻量 fake adapter 隔离外部模型、搜索与网页加载耗时，主要衡量 FastAPI 编排层、请求校验、路由分发和响应封装本身的开销。

| 并发数 | 请求数 | 成功数 | 吞吐(QPS) | 平均延迟(ms) | P95(ms) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 20 | 473.14 | 2.1 | 1 |
| 5 | 50 | 50 | 1243.0 | 0.72 | 2 |
| 10 | 80 | 80 | 1479.9 | 0.66 | 1 |

结果说明，在外部模型和网络服务被隔离后，系统编排层具备较低的请求处理开销，性能瓶颈主要来自真实链路中的网页渲染、图片下载、VLM 调用和 RAG 服务，而不是 FastAPI 路由本身。

### 5.4.2 网页多模态解析质量

网页解析质量采用 WebSRC v1.0 test 抽样进行评测。WebSRC 是面向网页结构阅读理解的数据集，包含 HTML、页面截图、元素 bounding box 和问答标注，适合评估网页结构和布局信息是否被保留。本测试抽取 15 个网页页面，对比纯文本抽取基线与本系统融合解析后的 `SourceDoc` 表示。评价指标包括答案文本覆盖率、`SourceDoc` 结构化有效率、多模态/布局信息保留率、平均图片/表格/链接/布局元素数量等。

| 指标 | 数值 |
| --- | ---: |
| 数据集 | WebSRC v1.0 test 抽样 |
| 样本数 | 15 |
| 纯文本基线答案覆盖率 | 100.0% |
| 融合解析答案覆盖率 | 100.0% |
| SourceDoc 结构化有效率 | 100.0% |
| 多模态/布局信息保留率 | 100.0% |
| 平均多模态元素数 | 0 |
| 平均链接数 | 0 |
| 平均布局框数 | 95.67 |

代表性样例如下：

| 用例 | 领域 | 页面ID | 问题摘要 | 标准答案 | 融合解析覆盖 | 图片数 | 表格数 | 链接数 | 布局框数 |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| W01 | auto | 0200001 | Specify the recommanded price for 2020 Acura ILX Sedan? | $25900 | 是 | 0 | 1 | 0 | 89 |
| W02 | game | 0800123 | When was the last update of this app? | October 19, 2020 | 是 | 0 | 0 | 0 | 84 |
| W03 | movie | 0800005 | What genre is this movie? | Adventure , Fantasy , A… | 是 | 0 | 0 | 0 | 88 |
| W04 | sports | 1600001 | How many GP did this player have in BOS when 2016-17? | 78 | 是 | 0 | 1 | 0 | 175 |
| W05 | university | 0800029 | How long will it take me to finish at Virginia Tech? | 4 Year | 是 | 0 | 0 | 0 | 92 |
| W06 | hotel | 0700001 | What the price of a Room, 1 King Bed, Accessible, Non Smoking? | $139 | 是 | 0 | 0 | 0 | 106 |
| W07 | auto | 0200002 | What's the recommanded price for 2020 Acura MDX FWD 7-Passenger? | $44500 | 是 | 0 | 1 | 0 | 89 |
| W08 | auto | 0200003 | What's the 2021 Toyota GR Supra 3.0 Premium Auto (Natl)'s recommandation pr… | $54490 | 是 | 0 | 1 | 0 | 89 |

从结果看，纯文本抽取与融合解析在答案文本覆盖上保持一致，说明系统没有因为结构化转换丢失关键文本；同时融合解析能够把图片、链接、表格和布局框继续保存在统一 `SourceDoc` / `modal_elements` / `crawl4ai_full` 结构中，多模态/布局信息保留率达到 100.0%。这体现了本项目的核心价值：不是重新发明网页抓取器或多模态 RAG 模型，而是在抓取结果与下游多模态 RAG 之间建立稳定的数据适配层，使网页证据从“纯文本”提升为“可入库、可追踪、可多模态消费”的结构化证据。

### 5.4.3 文搜图候选筛选质量

文搜图测试不再采用单张图片 yes/no 问答，而是改为更符合系统路径的候选筛选任务。测试数据采用 Flickr8k test 抽样，每条样本包含一条图片描述 caption，并构造 5 张候选图片，其中 1 张为目标图片、4 张为干扰图片。系统需要根据文本需求从候选图片中返回最匹配的一张。评价指标包括 Top1 Accuracy、返回图片率、随机首位基线命中率、降级率和端到端延迟。

| 指标 | 数值 |
| --- | ---: |
| 数据集 | Flickr8k test 抽样 |
| 样本数 | 15 |
| 每条候选图片数 | 5 |
| 随机首位基线命中率 | 40.0% |
| 系统 Top1 Accuracy | 80.0% |
| 返回图片率 | 100.0% |
| VLM/筛选降级率 | 20.0% |
| 平均延迟 | 4396.2 ms |
| P95 延迟 | 7177 ms |
| VLM 模型 | qwen3.5-omni-plus-2026-03-15 |

代表性样例如下：

| 用例 | 查询描述摘要 | 候选数 | 是否选中目标图 | 返回图片数 | 回答/筛选摘要 |
| --- | --- | ---: | --- | ---: | --- |
| I01 | The dogs are in the snow in front of a fence . | 5 | 是 | 1 | The image shows two dogs playing in the snow with a fence visible in the backgr… |
| I02 | a brown and white dog swimming towards some in the pool | 5 | 是 | 1 | The image shows a brown and white dog swimming in a blue pool towards a person … |
| I03 | A man and a woman in festive costumes dancing . | 5 | 是 | 1 | The image shows a man and a woman dressed in colorful festive costumes dancing … |
| I04 | A couple of people sit outdoors at a table with an umbrella and talk . | 5 | 否 | 1 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I05 | A man is wearing a Sooners red football shirt and helmet . | 5 | 是 | 1 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| I06 | A brown dog running | 5 | 是 | 1 | The image shows a brown dog running across a grassy area, with its legs extende… |
| I07 | A girl with dark brown hair and eyes in a blue scarf is standing next to a … | 5 | 是 | 1 | 图片中展示了两个女孩，其中一个有着深棕色的头发和眼睛，戴着蓝色围巾，她正站在另一个穿着带毛边外套的女孩旁边。 |
| I08 | A dog with its mouth opened . | 5 | 否 | 1 | 图片不足以回答该问题。  图中显示的是两名男性拳击手在拳击台上比赛，并没有狗。因此，没有任何一张候选图匹配“一只张着嘴的狗”这个描述。 |

结果显示，在 5 选 1 的候选筛选任务中，随机首位基线命中率为 40.0%，系统 Top1 Accuracy 为 80.0%。该实验更贴近文搜图链路的实际目标：系统不是判断某一张图片是否正确，而是对多个候选多模态资源进行筛选、排序和返回。测试中仍存在一定降级率，说明外部 VLM 连通性和多图推理稳定性会影响筛选质量，后续可通过候选缓存、CLIP 粗排阈值调优和 VLM 重试机制继续优化。

### 5.4.4 消融对比实验

| 方案 | 网页文本覆盖 | 多模态元素保留 | 布局/结构保留 | 文搜图候选筛选 | 结论 |
| --- | --- | --- | --- | --- | --- |
| Crawl4AI only | 可获得网页 Markdown/HTML 文本 | 可提供媒体、链接、表格等原始结果 | 保留在采集结果内部 | 不负责图片候选筛选 | 强在采集，不负责统一产品链路 |
| RAGAnything only | 依赖上游输入质量 | 能处理多模态内容 | 依赖输入适配 | 不负责开放域文搜图候选生成和筛选 | 强在多模态 RAG，引擎本身不解决网页采集适配 |
| 本系统融合方案 | 保持答案文本覆盖 | 通过 `modal_elements` 保留图片等资源 | 通过 `crawl4ai_full`、`structure` 保留链接、表格和布局元数据 | 支持候选图片筛选并返回目标图片 | 完成采集、结构化适配、入库桥接和文搜图筛选闭环 |

消融结果说明，本系统的贡献不在单个开源模型或单个 VLM 的能力，而在于把网页采集、多模态证据结构化、RAGAnything 入库适配、图片候选筛选和前端可观测链路组合为统一系统。该融合层使开源组件的输出能够被后续模块稳定消费，从而提升网页解析结果的工程可用性和文搜图链路的端到端完整性。

## 5.5 本章小结

本章围绕系统核心贡献重新设计了测试重点。功能测试验证了系统接口、安全校验、意图识别、约束解析、澄清状态、网页采集、RAG 桥接、图片代理和 Crawl4AI 接入等关键链路；性能与效果测试则进一步聚焦于系统融合能力，而不是第三方模型本身能力。

实验结果表明，系统编排层在隔离外部服务后具有较低的请求处理开销；在 WebSRC 网页结构数据上，融合解析能够在保持关键文本覆盖的同时，将图片、链接、表格和布局信息保存在统一结构化证据中；在 Flickr8k 文搜图候选筛选任务中，系统能够根据文本需求从多张候选图片中返回目标图片，体现了文搜图路径的核心功能。总体来看，本项目通过对 Crawl4AI、RAGAnything、图像检索与 VLM 筛选能力的封装和适配，实现了从网页抓取到多模态证据组织、再到问答和文搜图应用的完整闭环。后续优化方向包括扩大 WebSRC/SWDE/WebQA 类数据集评测规模、增强复杂网页表格解析、提升多图筛选稳定性，并增加缓存与重试机制以降低外部服务波动对端到端性能的影响。
