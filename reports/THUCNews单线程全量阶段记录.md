# THUCNews 词向量学习实验

<a id="thucnews-word-learning"></a>

## 数据与范围

本轮遍历完整 THUCNews 压缩包，清洗后 826,609 篇有效文章中，821,609 篇训练，5,000 篇留出诊断。比较 Gensim Word2Vec 与 FastText，新闻类别不作为模型输入。保留此前 5 万篇结果，沿用其 5,000 篇留出文章，便于比较数据规模变化。此前原论文 90 个全量实验保持停止；本轮不训练监督分类器。

原压缩包实测包含 836,075 篇文章、14 个新闻类别，压缩大小 1,560,309,796 字节。官方简介称约 74 万篇，本次按 ZIP 实际文章文件计数。macOS 元数据排除；中文文件名的 UTF-8 标志缺失已经修复。压缩包记录 SHA-256，所有读取成员检查 CRC。

| 新闻类别 | 原始文章数 | 训练样本 | 留出样本 | 排除文章 |
| --- | --- | --- | --- | --- |
| 体育 | 131604 | 130615 | 787 | 202 |
| 娱乐 | 92632 | 91943 | 554 | 135 |
| 家居 | 32586 | 29644 | 195 | 2747 |
| 彩票 | 7588 | 7263 | 46 | 279 |
| 房产 | 20050 | 19739 | 120 | 191 |
| 教育 | 41936 | 40950 | 251 | 735 |
| 时尚 | 13368 | 13013 | 80 | 275 |
| 时政 | 63086 | 62192 | 377 | 517 |
| 星座 | 3578 | 3536 | 21 | 21 |
| 游戏 | 24373 | 24137 | 146 | 90 |
| 社会 | 50849 | 50503 | 304 | 42 |
| 科技 | 162929 | 160739 | 974 | 1216 |
| 股票 | 154398 | 150952 | 923 | 2523 |
| 财经 | 37098 | 36383 | 222 | 493 |

分词采用固定 jieba 字典，关闭 HMM；统一 Unicode、清除控制字符和链接，不删除停用词。相同清洗文本去重，少于 30 词的文章排除；候选排除计数为 {'short_document': 8232, 'duplicate_clean_text': 1234}。上下文以句子组织，最长 256 词，不跨文章。训练文章按文件名 SHA-256 排序，跨类别交错。未进行近重复分组，留出诊断不视为分类泛化结果。

| 分区 | 文章 | 句子/分块 | 词数 |
| --- | --- | --- | --- |
| train | 821609 | 18576718 | 378641761 |
| holdout | 5000 | 114052 | 2337756 |

数据清单与逐句索引：[9cb3ecd006ae1a9a1f7f/manifest.json](../data/thucnews/prepared/9cb3ecd006ae1a9a1f7f/manifest.json)。

完整遍历 836,075 篇，保留 826,609 篇，排除 9,466 篇，三者可核对。去重优先保留指定留出文章，其重复副本不能进入训练。留出文本与逐句索引的 SHA-256 与 5 万篇实验一致。分词使用 8 个进程，结果按原输入顺序合并；提交检查点支持中断后恢复。

## 模型与训练

两种模型统一使用 CBOW、100 维、窗口 5、最低词频 5、5 轮、种子 42、单线程，初始学习率 0.025，线性降到 0.0001，下采样 0.001。为保存上下文评分，本轮使用层次 softmax：hs=1、negative=0。FastText 增加字符子词 1–3 和 200,000 个桶。训练文本、词典拟合范围与输入顺序一致。Gensim 4.4.0 在 CPU 上训练，未使用 CUDA。

当前已完成 1/2 个正式模型；每种只有一个种子，不做统计显著性或置信区间声明。

| 模型 | 词表大小 | 训练轮次耗时 (秒) | 模型进程峰值 RSS (MiB) | 完整原生模型 (MiB) |
| --- | --- | --- | --- | --- |
| word2vec | 263356 | 1824.8 | 962.6 | 239.7 |

训练轮次耗时包含逐轮固定探针评分，不包含建词典、导出、完整评价；检查点保存另计。峰值 RSS 包含训练、导出、重载和评价，不包括此前的数据准备。进程总耗时另存 summary.json。

## 词向量评价

评价使用已归档中文 WordSim240、WordSim297/296、CA translated 和 CA8 数据。词相似度的 Pearson、Spearman 只在词表内词对上计算；p 值和逐题数值保存在 JSON/CSV。WordSim296 是从 297 词对中移除 OPEC–石油后的口径，两个版本分别保留。类比同时报告已覆盖题目的条件正确率与 OOV 按错计入的全量正确率。FastText 的 OOV 子词向量结果另列，不能与词表内评价混为一谈。

| 模型 | 词对集 | Spearman ρ | Pearson r | 覆盖词对 | 含生成 OOV 的 ρ（探索） |
| --- | --- | --- | --- | --- | --- |
| word2vec | wordsim240 | 0.4301 | 0.4349 | 239/240 | — |
| word2vec | wordsim297 | 0.5095 | 0.5443 | 293/297 | — |
| word2vec | wordsim296 | 0.5127 | 0.5461 | 292/296 | — |

| 模型 | 类比集 | 条件正确率 | 全量正确率 | 覆盖题数 |
| --- | --- | --- | --- | --- |
| word2vec | CA_translated | 0.1931 | 0.1931 | 953/953 |
| word2vec | CA8_morphological | 0.0774 | 0.0585 | 7692/10177 |
| word2vec | CA8_semantic | 0.0556 | 0.0347 | 4763/7636 |

## 与 5 万篇实验比较

词相似度比较限定为 5 万篇模型已覆盖的相同词对，避免新增词表改变评分分母。模型参数、种子和留出集一致；训练数据、词表、下采样统计与 HS 树随规模变化。这是两次单种子规模对照，差值不表示统计显著性。

| 模型 | 词对集 | 共同词对数 | 5 万篇 ρ | 全量 ρ | 差值 |
| --- | --- | --- | --- | --- | --- |
| word2vec | wordsim240 | 233 | 0.4015 | 0.4248 | 0.0233 |
| word2vec | wordsim297 | 280 | 0.5239 | 0.5458 | 0.0219 |
| word2vec | wordsim296 | 279 | 0.5238 | 0.5492 | 0.0254 |

| 模型 | 规模 | 词表 | 同一留出集词表覆盖率 | 训练轮次耗时 (秒) | 原生模型 (MiB) |
| --- | --- | --- | --- | --- | --- |
| word2vec | 5 万篇 | 90819 | 0.9861 | 108.6 | 81.8 |
| word2vec | 全量 | 263356 | 0.9981 | 1824.8 | 239.7 |

| 模型 | 留出词表覆盖率 | 可用向量覆盖率 | OOV 词次 | 可生成 OOV 词次 |
| --- | --- | --- | --- | --- |
| word2vec | 0.9981 | 0.9981 | 4445 | 0 |

覆盖率衡量能否得到向量，不能证明语义正确。每个运行保存近邻、词/词组相似度、乘法类比和异类词探针；这些例子用于定性检查，不独立作为质量指标。

| 模型 | 查询词 | 向量来源 | 前三个近邻（余弦） |
| --- | --- | --- | --- |
| word2vec | 经济 | in_vocabulary | 宏观经济 (0.797)、总体经济 (0.722)、能源需求 (0.722) |
| word2vec | 股票 | in_vocabulary | 次新股 (0.673)、基金净值 (0.672)、债券 (0.669) |
| word2vec | 新能源汽车产业 | no_vector | 无向量 |

## 句子上下文评分

历史 HS 上下文评分保留在原始 CSV。评分忽略 OOV，不是整句生成概率，不计算困惑度；当前逐轮验证和模型选择见[新实验](THUCNews词向量实验.md)。

## 保存与复现

保存加载后检查全部词表内向量逐位相同，固定 256 个留出句子的评分完全一致。文本导出的前 100 个词重新读回，记录最大误差。所有完整资源保存 SHA-256。模型原生格式和 .kv 保留 FastText 子词能力；.vec 只导出词表内词向量，可供原生 fastText 初始化，不传递完整字符子词参数。

word2vec：[原生模型](../artifacts/thucnews/runs/009641f3baed144efd5f6170/model.gensim) · [词向量 KV](../artifacts/thucnews/runs/009641f3baed144efd5f6170/vectors.kv) · [预训练 VEC](../artifacts/thucnews/runs/009641f3baed144efd5f6170/vectors.vec) · [训练日志](../artifacts/thucnews/runs/009641f3baed144efd5f6170/run.log) · [逐轮记录](../artifacts/thucnews/runs/009641f3baed144efd5f6170/epochs.csv) · [句子评分](../artifacts/thucnews/runs/009641f3baed144efd5f6170/heldout_sentence_scores.csv) · [近邻与 API 探针](../artifacts/thucnews/runs/009641f3baed144efd5f6170/queries.json) · [评价明细](../artifacts/thucnews/runs/009641f3baed144efd5f6170/metrics.json) · [重载核查](../artifacts/thucnews/runs/009641f3baed144efd5f6170/save_load_check.json)

项目测试 69 项通过，覆盖全量遍历、去重留出优先、断点恢复和原生检查点恢复。[测试日志](../artifacts/thucnews_full_suite.log)。

```powershell
conda run -n Gensim python -m gensim_lab.thucnews --stage all
conda run -n Gensim python -m gensim_lab.thucnews --stage report
conda run -n Gensim python -m gensim_lab.thucnews_audit
conda run -n Gensim python -m gensim_lab.thucnews --stage query --model fasttext --word 经济
```

## 来源

[清华 THUNLP 数据说明](https://github.com/thunlp/THUCTC)、[原始压缩包](https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip)、[CWE 中文词对数据](https://github.com/Leonard-Xu/CWE)、[Chinese Word Vectors 类比数据](https://github.com/Embedding/Chinese-Word-Vectors)。基准版本及校验值沿用 references/word_benchmarks/manifest.json；配置为 configs/thucnews.yaml。
