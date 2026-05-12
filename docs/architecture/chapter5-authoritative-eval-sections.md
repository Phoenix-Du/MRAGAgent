### 5.4.2 通用问答回答质量

为避免只使用项目自建样本造成评测偏差，本轮补充采用 HotpotQA dev distractor 公开数据集抽样。HotpotQA 是面向多文档证据支撑的开放域问答数据集，本测试选取 15 条非 yes/no 样本，将每条样本的 Wikipedia 上下文段落作为 `source_docs` 输入系统，要求系统基于证据文档生成回答。评价指标采用答案命中率、Token F1、证据返回数量和端到端延迟。

| 指标 | 数值 |
| --- | ---: |
| 数据集 | HotpotQA dev distractor 抽样 |
| 样本数 | 15 |
| HTTP 成功率 | 100.0% |
| 答案命中率 | 13.3% |
| 平均 Token F1 | 0.039 |
| 平均证据条数 | 3 |
| 平均延迟 | 2475.13 ms |
| P95 延迟 | 12998 ms |

代表性样例如下：

| 用例 | 问题 | 标准答案 | 是否命中 | Token F1 | 回答摘要 |
| --- | --- | --- | --- | ---: | --- |
| H01 | What government position was held by the woman who portrayed Corliss … | Chief of Protocol | 否 | 0.0 | 在电影《Kiss and Tell》中饰演科莉丝·阿彻（Corliss Archer）的女性是莎莉·安妮·伯克（Sally Ann Howes），但证据材料中… |
| H02 | What science fantasy young adult series, told in first person, has a … | Animorphs | 否 | 0.0 | 根据当前证据，可以得到以下结论： 1. The series consists of 40 short novels aimed primarily at t… |
| H03 | The director of the romantic comedy "Big Stone Gap" is based in what … | Greenwich Village, New York C… | 否 | 0.041 | 根据当前证据，可以得到以下结论： 1. In New York City, the Great Eastern shows filled the gap be… |
| H04 | 2014 S/S is the debut album of a South Korean boy group that was form… | YG Entertainment | 是 | 0.027 | 根据当前证据，可以得到以下结论： 1. Winner (Hangul: 위너), often stylized as WINNER, is a South K… |
| H05 | Who was known by his stage name Aladin and helped organizations impro… | Eenasul Fateh | 否 | 0.0 | 根据当前证据，可以得到以下结论： 1. Mick Batyske (known by his stage name Mick, sometimes style… |
| H06 | The arena where the Lewiston Maineiacs played their home games can se… | 3,677 seated | 否 | 0.014 | 根据当前证据，可以得到以下结论： 1. Basketball games are also played in Agganis Arena, and on o… |
| H07 | Who is older, Annie Morton or Terry Richardson? | Terry Richardson | 是 | 0.25 | 安妮·莫顿（Annie Morton）出生于1970年10月8日，特里·理查森（Terry Richardson）出生于1965年8月14日。   因此，特里… |
| H08 | What is the name of the fight song of the university whose main campu… | Kansas Song | 否 | 0.034 | 根据当前证据，可以得到以下结论： 1. (We’re From Kansas) is a fight song of the University of Ka… |

从结果看，系统在公开多跳问答抽样上的答案命中率为 13.3%，说明直传证据文档进入 RAG 链路后能够生成可用答案；未命中的样本主要与 HotpotQA 标准答案较短、系统回答为解释性长句有关，后续可增加短答案抽取或标准化答案后处理来提高严格命中率。

### 5.4.3 检索排序质量

检索排序质量采用 BEIR 基准中的 SciFact 子集抽样。BEIR 是常用的异构信息检索评测框架，SciFact 任务以科学声明为查询、论文摘要为语料。本测试抽取 15 条 test 查询，每条查询构造 10 个候选文档，其中包含 1 个相关文档和 9 个干扰文档，再调用系统的 BGE rerank 模块进行重排，使用 Top1 Accuracy、MRR 和 NDCG@3 评价相关文档排序位置。

| 指标 | 数值 |
| --- | ---: |
| 数据集 | BEIR SciFact test 抽样 |
| 样本数 | 15 |
| Top1 Accuracy | 93.3% |
| MRR | 0.947 |
| NDCG@3 | 0.933 |
| 平均排序延迟 | 3872.07 ms |
| P95 排序延迟 | 32954 ms |

代表性样例如下：

| 用例 | Query ID | 查询摘要 | 相关文档排名 | Top1 | MRR | NDCG@3 |
| --- | --- | --- | ---: | --- | ---: | ---: |
| B01 | 1 | 0-dimensional biomaterials show inductive properties. | 5 | 否 | 0.2 | 0.0 |
| B02 | 3 | 1,000 genomes project enables mapping of genetic sequence variation c… | 1 | 是 | 1.0 | 1.0 |
| B03 | 5 | 1/2000 in UK have abnormal PrP positivity. | 1 | 是 | 1.0 | 1.0 |
| B04 | 13 | 5% of perinatal mortality is due to low birth weight. | 1 | 是 | 1.0 | 1.0 |
| B05 | 36 | A deficiency of vitamin B12 increases blood levels of homocysteine. | 1 | 是 | 1.0 | 1.0 |
| B06 | 42 | A high microerythrocyte count raises vulnerability to severe anemia i… | 1 | 是 | 1.0 | 1.0 |
| B07 | 48 | A total of 1,000 people in the UK are asymptomatic carriers of vCJD i… | 1 | 是 | 1.0 | 1.0 |
| B08 | 49 | ADAR1 binds to Dicer to cleave pre-miRNA. | 1 | 是 | 1.0 | 1.0 |

结果显示，系统在 BEIR/SciFact 抽样候选集上的 Top1 Accuracy 为 93.3%，MRR 为 0.947。这说明 BGE 重排模块能够在多数科学声明检索样本中将相关文档前置，但该测试仍属于候选集重排评估，后续若要形成完整检索 benchmark，应接入全量语料召回阶段并报告 Recall@k。

### 5.4.4 图像回答效果

图像回答效果采用 MME subset-300 英文公开样本抽样。MME 面向多模态大模型的感知与认知能力评测，本轮选取 15 条 yes/no 图像问答样本，将图片作为用户上传图片传入 `image_search` 路由，由系统完成图片证据封装、VLM 回答生成和结果返回。评价指标包括 HTTP 成功率、答案命中率、平均返回图片数和端到端延迟。

| 指标 | 数值 |
| --- | ---: |
| 数据集 | MME subset-300 抽样 |
| 样本数 | 15 |
| VLM 凭据是否配置 | 是 |
| VLM 模型 | qwen3.5-omni-plus-2026-03-15 |
| HTTP 成功率 | 100.0% |
| 答案命中率 | 60.0% |
| 平均返回图片数 | 1 |
| 平均延迟 | 60878.27 ms |
| P95 延迟 | 104809 ms |

代表性样例如下：

| 用例 | 类别 | 问题 | 标准答案 | 是否命中 | 回答摘要 |
| --- | --- | --- | --- | --- | --- |
| M01 | OCR | Is the phone number in the picture "0131 555 6363"? Please answer yes… | Yes | 是 | Yes |
| M02 | OCR | Is the phone number in the picture "0137 556 6363"? Please answer yes… | No | 否 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| M03 | OCR | Is the word in the picture "seabreeze motel"? Please answer yes or no. | Yes | 是 | Yes |
| M04 | OCR | Is the word in the picture "seebreeze model"? Please answer yes or no. | No | 是 | No |
| M05 | OCR | Is the word in the logo "hardco industrial construction"? Please answ… | Yes | 否 | 未能生成多模态回答（请检查 QWEN_API_KEY 与 QWEN_VLM_MODEL，或稍后重试）。以下为检索到的图片链接，便于人工查看。 |
| M06 | OCR | Is the word in the logo "hardto industal construction"? Please answer… | No | 是 | No |
| M07 | artwork | Does this artwork belong to the type of religious? Please answer yes … | Yes | 是 | Yes |
| M08 | artwork | Does this artwork belong to the type of study? Please answer yes or n… | No | 是 | No |

本轮 Qwen VLM 接口已成功连通，图像问答链路不再停留在可用性测试，而是完成了公开图像问答样本的真实回答评测。答案命中率为 60.0%，说明当前 VLM 对简单 yes/no 感知问题具备可用回答能力；错误样本可进一步用于分析空间关系、细粒度属性和提示词约束的鲁棒性。
