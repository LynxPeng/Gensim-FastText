# THUCNews 训练方式消融与复验方案

## 目的与实验矩阵

在完整 THUCNews 训练语料上检查当前 CBOW＋HS 是否合适。Word2Vec、FastText 各运行下列六个训练方式，种子 42 初筛共 12 次；每个模型的基线和最佳新方案再补种子 2026、7，总计 20 次。TF-IDF 沿用已有统计对照，本轮不训练类别预测器。

| 方案 | sg | hs | negative | 对比含义 |
| --- | --- | --- | --- | --- |
| CBOW＋HS | 0 | 1 | 0 | 当前基线从头复跑 |
| CBOW＋NS5 | 0 | 0 | 5 | 替换训练目标 |
| CBOW＋NS10 | 0 | 0 | 10 | 增加负例数 |
| Skip-gram＋HS | 1 | 1 | 0 | 替换上下文预测方式 |
| Skip-gram＋NS5 | 1 | 0 | 5 | 预测方式与目标组合 |
| Skip-gram＋NS10 | 1 | 0 | 10 | 更多负例与预测方式组合 |

## 固定条件

使用已清洗、分词、去重的 821,609 篇训练文章，18,576,718 个句子/分块、378,641,761 词。沿用 5,000 篇留出文章诊断词表覆盖率。原始类别不参与表示训练、验证或选参；本轮不重新采样新闻文章。

100 维、window=5、min_count=5、sample=0.001、20 个训练线程、4 个数值库线程、batch_words=9,000。固定 ns_exponent=0.75、cbow_mean=1、shrink_windows=True、sorted_vocab=1。FastText 的字符子词 1–3、200,000 桶固定。

学习率从 0.025 到 0.0001，按最多 15 轮线性调度，所有组合一致。至少 5 轮、最多 15 轮，连续 3 轮未超过有效提升参照 0.001 时停止；微小更高分仍更新最佳检查点。每轮保存完整原生模型，不记录训练 loss。使用可重新迭代的 LineSentence；核对每轮全部原始词数，避免语料漏读。

## 评价隔离与选择

评价划分种子 `split_seed=42` 固定，训练种子 `model_seed=42/2026/7` 独立。直接锁定已有去重分区清单的路径与 SHA-256，训练种子变化不重新划分题目。相同无序词对、等价类比跨基准共享同一分区，验证/测试题组交集为 0。

WordSim240 的 95 个验证词对中严格词表内 Spearman ρ 是唯一选择指标。逐轮还计算全部词对验证集的 Pearson、Spearman、覆盖率及 p 值；类比使用每套固定 128 题验证探针，全部题正确率与覆盖题正确率仅作诊断。词表外子词生成指标单列，不参与选择。输入和词表相同保证正式主指标的题目覆盖口径相同。

1. 初筛 12 组只计算验证指标；旧训练入口的自动测试行为不复用。
2. 每个模型在五个新方案中选验证分最高者，完全同分按方案名稳定排序；它与基线各增加两个种子。
3. 三种子验证均值选择最终训练方式，报告样本 SD（ddof=1）。同分优先基线。只对这两个方案作复验后的选择，不声称未复验方案更差。
4. 20 次训练全部完成并核验后，保存不可变的搜索快照、冻结决定、运行封存哈希与所选检查点哈希。
5. 统一测试两模型的基线和候选各三个检查点，共 12 个；全部测试题参与评价。测试结果只用于描述，不改变配置或轮数。

每个方案独立选轮，比较包含相同选轮规则的实际训练效果。时间记录区分建词表、累计训练、平均每轮、验证、保存以及最终测试，不能把提前停止导致的成本差异全部归因于算法。

## 解释边界与后续阶段

本轮是探索性选参：完整评价基准已被历史实验观察，固定分区和本轮测试隔离不能恢复真正未观察过的盲测条件。三种子用于描述随机波动，没有显著性结论。多线程不是逐位确定性；中断恢复保留完整模型与随机状态，但线程调度仍可能不同。

本矩阵完成后，独立的下一阶段可在优胜训练方式下比较 alpha=0.01、0.025、0.05，继续固定现有评价划分；这一阶段不混入本次 20 次训练的结论。子词、桶、窗口和维度尚未搜索。TF-IDF 若要调参，须先定义文档任务的验证指标。

## 资源与复现

[配置](../configs/thucnews_ablation.yaml)、[入口](../gensim_lab/thucnews_ablation.py)、[结果报告](../reports/THUCNews训练方式消融实验.md)。使用项目 Gensim 环境。

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_ablation --smoke
conda run -n Gensim python -m gensim_lab.thucnews_ablation
conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage audit
conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage report
conda run -n Gensim python -m gensim_lab.thucnews_ablation_figures --stage preview
# 检查彩色/灰度预览，保存 visual_review.json 后导出
conda run -n Gensim python -m gensim_lab.thucnews_ablation_figures --stage export
# 已启动训练的单次依赖步骤：完成后刷新详细报告和综合报告，不启动训练
conda run -n Gensim python tools/finalize_thucnews_ablation.py --wait
```

`artifacts/thucnews_ablation/latest.json` 定位正式实验；`smoke_latest.json` 单独定位冒烟，冒烟不会覆盖正式报告。每个运行的 `run.log` 保存训练日志，`state.json` 用于最后完整轮次恢复，`epochs.csv` 和 `validation_metrics.csv` 保存逐轮记录，`checkpoints/epoch_NN.gensim` 保存原生模型。正式报告每组完成后自动刷新，提交报告仅新增 1.1.4，保持 1.2 及之后内容。

训练编排持有项目文件锁，同一时间只允许一个训练编排进程。缓存按数据、配置、评价分区、版本和训练代码指纹隔离，逐文件核对 SHA-256；测试资源保存于冻结实验的独立目录。每轮检查点全部保留，启动前要求至少 120 GiB 可用磁盘。

图表先数据剖析再选图，矩阵用数值热力图，逐轮用折线，三种子用全部实测点和均值±SD。机器检查布局、字体、尺寸和 DPI，彩色与灰度预览通过视觉复核后才导出 300 DPI PNG 和 SVG。图和原始数值通过相对路径嵌入 Markdown。

[Gensim Word2Vec 参数文档](https://radimrehurek.com/gensim/models/word2vec.html)、[Gensim FastText 参数文档](https://radimrehurek.com/gensim/models/fasttext.html)说明 sg、hs 和 negative 的含义；实际运行以保存的 Gensim 4.4.0 版本与原生模型参数检查为准。
