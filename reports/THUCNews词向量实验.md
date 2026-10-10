# THUCNews 三种表示学习实验

<a id="thucnews-word-learning"></a>

## 数据与预处理

原始 836,075 篇文章，清洗、短文过滤与完全去重后保留 826,609 篇。标题和正文合并；统一全半角，处理 HTML/实体、网址、控制字符与空白；jieba 关闭 HMM 分词，不删除停用词。少于 30 词的文章排除，每句最多 256 词。固定文章顺序，独立保留 5,000 篇文章做覆盖率诊断。新闻类别不用于本轮词向量训练和轮数选择。本轮没有训练类别预测器。

| 分区 | 文章 | 句子/分块 | 词数 |
| --- | --- | --- | --- |
| train | 821609 | 18576718 | 378641761 |
| holdout | 5000 | 114052 | 2337756 |

| 排除原因 | 文章 |
| --- | --- |
| short_document | 8232 |
| duplicate_clean_text | 1234 |

[数据清单](../data/thucnews/prepared/9cb3ecd006ae1a9a1f7f/manifest.json)。语料已排除完全重复，未按近重复新闻分组。

## 模型与线程

TF-IDF 沿用同一训练集拟合的统计模型：文档频次至少 3、最高占比 0.9、至多 50,000 词，log2(N/DF)，L2 归一化。其结果与训练设置未改变，可直接复用。Word2Vec 与 FastText 分别从头训练原生 Gensim 模型：CBOW、100 维、窗口 5、最低词频 5、HS=1/negative=0、下采样 0.001。FastText 使用字符子词 1–3、200,000 桶。两个模型各使用 20 个 CPU 训练线程，数值库 4 线程；固定种子 42、输入顺序和 Python hash seed=0。多线程更新不保证逐位确定性；每个模型仅一次正式训练（n=1），不报告重复实验误差或显著性。

使用 LineSentence 和 Gensim 有界任务队列；每个任务最多 9,000 原始词，保留句子边界。学习率 0.025→0.0001 按最多 15 轮线性调度；提前停止时尚未走完该调度。因此，本次与历史固定 5 轮训练的差异还包含学习率日程，不能只归因于增加轮数。

[配置](../configs/thucnews_validation.yaml) · [训练入口](../gensim_lab/thucnews_validation.py)。

## 独立验证与测试

评价题目先去重，再固定划分验证与测试。相同无序词对及关系交换/同时反向的等价类比题跨基准共享分区，避免同题泄漏。同一基准的重复题保留第一题，冲突评分单独记录；不同版本的人工评分不合并。WordSim240 237 个唯一词对固定分为 95 验证、142 测试；其他基准按同一题组规则划分。逐轮记录三个词对集全部验证词对的 Pearson、Spearman、覆盖率与 p 值，以及每套类比验证池固定抽取最多 128 题的正确率。测试阶段使用全部测试题。

Word2Vec、FastText 分别按 WordSim240 验证集严格词表内 Spearman 选择最佳检查点。至少 5 轮、最多 15 轮；连续 3 轮未比有效提升参照增加超过 0.001 时停止。任何更高分都更新最佳检查点，但不超过 0.001 的小幅提升不重置耐心计数。Pearson 与类比只作辅助诊断，不参与选择。两个模型独立完成选择并重载核对后，才计算各自最终测试结果。验证曲线只能支持当前语料、配置和评价任务的轮数选择，不能证明全局最优或所有下游任务均已收敛。

| 基准 | 原题数 | 等价重复删除 | 验证池 | 逐轮评价 | 测试 |
| --- | --- | --- | --- | --- | --- |
| wordsim240 | 240 | 3 | 95 | 95 | 142 |
| wordsim297 | 297 | 2 | 109 | 109 | 186 |
| wordsim296 | 296 | 2 | 108 | 108 | 186 |
| CA_translated | 953 | 413 | 221 | 128 | 319 |
| CA8_morphological | 10177 | 33 | 4076 | 128 | 6068 |
| CA8_semantic | 7636 | 1 | 3092 | 128 | 4543 |

[分区清单](../artifacts/thucnews_validation/benchmarks/c1943509f229e0645ba2/manifest.json) · [题组分配](../artifacts/thucnews_validation/benchmarks/c1943509f229e0645ba2/assignments.csv)。分区题组交集为 0。WordSim297/296 的本次测试题完全相同，图中合并标注；两份指标分别保存。该协议不同于历史完整基准直接测试，历史分数不能直接作为本次提升幅度。原完整基准已被历史实验观察过，本次保留集不能声称是未观察过的新基准盲测；当前选轮程序不读取测试分数。

## 逐轮验证与模型检查点

### fasttext

实际训练 5 轮，因验证主指标连续三轮无有效提升结束；选择第 1 轮，验证 Spearman=0.3499。每轮均保存完整原生检查点、配置和逐题评价，最终 model.gensim 为所选检查点，训练最后一轮与部署模型可能不同。

![fasttext 逐轮验证](figures/section11/fasttext_validation.png)

| 轮次 | 验证主指标 ρ | 最佳轮次 | 耐心计数 | 训练秒 | 评价秒 | 保存秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3499 | 1 | 0 | 173.9 | 2.0 | 1.2 |
| 2 | 0.3186 | 1 | 1 | 175.4 | 2.0 | 1.3 |
| 3 | 0.3019 | 1 | 2 | 175.6 | 2.0 | 1.3 |
| 4 | 0.3103 | 1 | 3 | 175.5 | 2.2 | 1.3 |
| 5 | 0.3395 | 1 | 4 | 217.6 | 2.2 | 1.6 |

### word2vec

实际训练 8 轮，因验证主指标连续三轮无有效提升结束；选择第 5 轮，验证 Spearman=0.3655。每轮均保存完整原生检查点、配置和逐题评价，最终 model.gensim 为所选检查点，训练最后一轮与部署模型可能不同。

![word2vec 逐轮验证](figures/section11/word2vec_validation.png)

| 轮次 | 验证主指标 ρ | 最佳轮次 | 耐心计数 | 训练秒 | 评价秒 | 保存秒 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3341 | 1 | 0 | 204.6 | 2.3 | 1.7 |
| 2 | 0.3239 | 1 | 1 | 184.0 | 2.4 | 2.2 |
| 3 | 0.3158 | 1 | 2 | 163.3 | 2.1 | 1.2 |
| 4 | 0.3077 | 1 | 3 | 155.4 | 2.1 | 1.3 |
| 5 | 0.3655 | 5 | 0 | 202.9 | 2.3 | 1.6 |
| 6 | 0.3584 | 5 | 1 | 181.3 | 2.3 | 1.3 |
| 7 | 0.3312 | 5 | 2 | 158.3 | 2.2 | 1.2 |
| 8 | 0.3476 | 5 | 3 | 157.5 | 2.1 | 1.2 |

## 最终测试与覆盖率

词相似度仅对严格词表内词对计算正式相关系数。FastText 子词生成的 OOV 结果另存探索指标。类比同时报告覆盖题正确率与 OOV 按错误计入的全部测试题正确率；TF-IDF 不适用这些词向量评价。

| 方法 | 词对集 | 覆盖/测试 | Spearman ρ | Pearson r | 含生成 OOV 的 ρ（探索） |
| --- | --- | --- | --- | --- | --- |
| fasttext | wordsim240 | 141/142 | 0.4521 | 0.4526 | 0.4480 |
| fasttext | wordsim297 | 182/186 | 0.4785 | 0.4761 | 0.4853 |
| fasttext | wordsim296 | 182/186 | 0.4785 | 0.4761 | 0.4853 |
| word2vec | wordsim240 | 141/142 | 0.4875 | 0.4852 | — |
| word2vec | wordsim297 | 182/186 | 0.5147 | 0.5145 | — |
| word2vec | wordsim296 | 182/186 | 0.5147 | 0.5145 | — |

| 方法 | 类比集 | 覆盖/测试 | 正确题 | 覆盖题正确率 | 全部测试正确率 |
| --- | --- | --- | --- | --- | --- |
| fasttext | CA_translated | 319/319 | 11 | 0.0345 | 0.0345 |
| fasttext | CA8_morphological | 4568/6068 | 1836 | 0.4019 | 0.3026 |
| fasttext | CA8_semantic | 2831/4543 | 188 | 0.0664 | 0.0414 |
| word2vec | CA_translated | 319/319 | 27 | 0.0846 | 0.0846 |
| word2vec | CA8_morphological | 4568/6068 | 273 | 0.0598 | 0.0450 |
| word2vec | CA8_semantic | 2831/4543 | 86 | 0.0304 | 0.0189 |

| 方法 | 词表/特征数 | 训练/拟合秒 | 主进程峰值 RSS MiB | 原生模型 MiB | 留出可用词覆盖率 |
| --- | --- | --- | --- | --- | --- |
| fasttext | 263356 | 918.0 | 1126.3 | 320.9 | 1.0000 |
| tfidf | 50000 | 56.7 | 416.4 | 5.3 | 0.9177 |
| word2vec | 263356 | 1407.4 | 823.1 | 240.7 | 0.9981 |

神经模型训练时间仅累计所有实际训练轮的训练调用，不包含词表构建、验证评价、检查点保存或最终测试；上述开销分别保存。TF-IDF 拟合包含 DF/CF 统计和 IDF 计算，计时范围不同。原生模型体积不含所有检查点总占用；TF-IDF 包含词典。峰值内存为模型主进程 RSS。

## 句子上下文评分与核验

fasttext 的最佳模型保存 114,052 条留出句子上下文评分，[逐句 CSV](../artifacts/thucnews_validation/posthoc/b560d36061b15b8ab5904931/heldout_sentence_scores.csv)。

[定性 API 探针](../artifacts/thucnews_validation/posthoc/b560d36061b15b8ab5904931/qualitative_probes.json)保存近邻、词／词组相似度、乘法类比与异类词结果；不作为训练停止指标。

word2vec 的最佳模型保存 114,052 条留出句子上下文评分，[逐句 CSV](../artifacts/thucnews_validation/posthoc/ea6051ebda1db1bf95921d3d/heldout_sentence_scores.csv)。

[定性 API 探针](../artifacts/thucnews_validation/posthoc/ea6051ebda1db1bf95921d3d/qualitative_probes.json)保存近邻、词／词组相似度、乘法类比与异类词结果；不作为训练停止指标。

HS 上下文 score 是固定窗口中词预测事件的分数，忽略 OOV；FastText 继承 Word2Vec 的评分路径，这里仅作兼容性诊断。它不是整句生成概率，不计算困惑度，也不用于早停。训练过程只使用外部验证指标。

独立核验通过：题组无交叉、每轮主指标复算、原生检查点重载、最佳轮次与耐心规则、训练词数、训练词表频次、逐题测试分母和保存向量均一致。[核验记录](../artifacts/thucnews_validation/verification.json)。

## 资源与复现

项目测试 73 项通过。[测试日志](../artifacts/thucnews_validation/tests.log)。

fasttext：[配置](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/spec.json) · [日志](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/run.log) · [逐轮记录](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/epochs.csv) · [逐轮指标](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/validation_metrics.csv) · [最佳模型](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/model.gensim) · [KV](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/vectors.kv) · [测试指标](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/metrics.json) · [重载核对](../artifacts/thucnews_validation/runs/b560d36061b15b8ab5904931/save_load_check.json)

word2vec：[配置](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/spec.json) · [日志](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/run.log) · [逐轮记录](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/epochs.csv) · [逐轮指标](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/validation_metrics.csv) · [最佳模型](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/model.gensim) · [KV](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/vectors.kv) · [测试指标](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/metrics.json) · [重载核对](../artifacts/thucnews_validation/runs/ea6051ebda1db1bf95921d3d/save_load_check.json)

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_validation --smoke
conda run -n Gensim python -m gensim_lab.thucnews_validation
conda run -n Gensim python -m gensim_lab.thucnews_audit
conda run -n Gensim python -m gensim_lab.thucnews --stage report
```

中断后按配置/数据/代码指纹从最后完整检查点恢复；已封存结果复用缓存。历史固定 5 轮和分类实验资源保留，当前报告采用上述新分区。原论文全量复现仍保持停止。[THUNLP](https://github.com/thunlp/THUCTC) · [CWE 词对](https://github.com/Leonard-Xu/CWE) · [Chinese Word Vectors 类比](https://github.com/Embedding/Chinese-Word-Vectors)。
