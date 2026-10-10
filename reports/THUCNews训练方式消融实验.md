# THUCNews 训练方式消融与对比实验

<a id="thucnews-training-ablation"></a>

## 研究问题与运行状态

比较 Word2Vec 与 FastText 的上下文预测方式（CBOW／Skip-gram）及训练目标（HS／NS5／NS10）。两因素构成 2×3 小矩阵，每个模型 6 组；NS5→NS10 比较负例数的收益与成本。本轮只做词向量表示学习和内在评价，没有训练新闻类别预测器。

更新时间：2026-10-09T22:46:03+08:00。当前阶段：**已暂停（种子 42 初筛）**；已完成 **4/20** 次正式训练。本页只纳入已封存且重载验证通过的结果，未完成项不填充猜测数值。

用户要求暂停训练，训练进程与报告等待进程均已退出。已完成 4/20 组；word2vec/sg_ns5/seed42 保存至第 8 轮，第 9 轮中途停止。完整检查点、验证指标与日志均保留，尚未开展本轮测试。[暂停记录](../artifacts/thucnews_ablation/stopped.json)。

## 固定条件与评价隔离

沿用完整 THUCNews 清洗与去重后的 821,609 篇训练文章、378,641,761 词。5,000 篇文章只用于向量覆盖率诊断。清洗、jieba 分词、分句和文章顺序保持原协议：[数据处理说明](THUCNews词向量实验.md#数据与预处理)。新闻类别标签不参与训练或选参。

100 维、window=5、min_count=5、sample=0.001、20 个训练线程、4 个数值库线程，batch_words=9,000。学习率 0.025→0.0001 按最多 15 轮线性调度；至少训练 5 轮，验证主指标连续 3 轮未超过有效提升参照 0.001 时停止。更高的小幅分数仍更新最佳检查点。FastText 固定 min_n=1、max_n=3、bucket=200,000；不记录训练 loss。

额外显式固定 ns_exponent=0.75、CBOW 上下文均值、动态窗口（shrink_windows=True）和频次排序词表。所有方案从头训练，不沿用已训练基线权重。每轮核对完整原始词数，并保存原生检查点与逐题验证结果。多线程更新不保证逐位确定性，种子重复也包括线程调度差异。

**评价划分种子固定为 42，训练种子独立变化为 42、2026、7。**直接锁定已有去重评价分区的路径和 SHA-256；训练种子不会重新分配题目。WordSim240 使用相同 95 个验证词对，主指标为严格词表内 Spearman ρ。三个词对基准逐轮计算 Pearson／Spearman／覆盖率；类比使用每套固定 128 题验证探针作辅助诊断。正式测试使用全部测试题；辅助指标不参与配置或检查点选择。

[固定分区](../artifacts/thucnews_validation/benchmarks/c1943509f229e0645ba2/manifest.json) · [配置](../configs/thucnews_ablation.yaml) · [代码](../gensim_lab/thucnews_ablation.py)。

## 搜索、复验与冻结规则

1. 种子 42 运行两个模型各 6 组，共 12 次，搜索阶段只计算验证指标。
2. 每个模型在五个新方案中选验证分最高者；与 CBOW＋HS 基线各补 2026、7 两种子，共增加 8 次训练。
3. 对基线和新方案分别计算三个种子的验证均值与样本标准差（ddof=1），按均值选最终训练方式。完全同分时优先基线，初筛新方案同分按配置名称稳定排序。
4. 封存全部选择及检查点哈希后，统一评价两模型的基线和候选各三个种子，共 12 个最佳检查点。比较两方案测试分数用于描述效果，不反过来改选配置。

每次训练独立按验证选择轮数，因此比较的是“训练方式＋相同选轮规则”的效果，各方案实际轮数可能不同；训练时间同时报告每轮与累计值，不能把累计时间差全部归因于算法。TF-IDF 保留已有统计对照；其无词向量，未用 WordSim 主指标搜索过滤参数。

## 种子 42 初筛结果

| 模型 | 训练方式 | 验证 Spearman | 最佳/实际轮数 | 训练秒 | 每轮秒 | 峰值 RSS MiB | 原生模型 MiB |
| --- | --- | --- | --- | --- | --- | --- | --- |
| word2vec | CBOW＋HS | 0.3483 | 1/5 | 749.9 | 150.0 | 823.4 | 240.7 |
| word2vec | CBOW＋NS5 | 0.4408 | 6/9 | 985.3 | 109.5 | 595.6 | 210.0 |
| word2vec | CBOW＋NS10 | 0.4457 | 2/5 | 588.9 | 117.8 | 581.9 | 210.0 |
| word2vec | Skip-gram＋HS | 0.4940 | 1/5 | 1905.7 | 381.1 | 823.6 | 240.7 |

12 组初筛未全部完成，尚未选择新方案。

## 三种子复验

尚未进入复验；初筛单次结果不报告标准差或显著性。

尚未冻结配置，**本轮未计算测试指标**。

## 冻结后的统一测试

等待 20 次训练和配置冻结，不显示历史测试分数作为本轮结果。

## 结果图与解释范围

初筛为每模型 2×3 矩阵，适合带数值标注的热力图；逐轮指标适合折线。三种子比较展示每次实测值和均值±样本 SD；不连接无序训练方式、不做曲线平滑，不用均值柱遮住 n=3。

完整实验图在数据剖析、机器布局检查及彩色／灰度视觉复核后发布；当前数据可从原始 CSV 追溯。

**结论边界：探索性选参。** 这些基准的完整结果曾被历史实验观察，本轮固定分区和测试隔离避免当前搜索再用测试分数选参，但不能恢复未观察过的盲测条件。验证基准较小且尝试多个配置，最高分存在选择偏差；三种子复验不能消除基准偏差。新的确认性结论需要独立评价数据。

学习率敏感性列为本矩阵后的独立后续阶段，暂不混入本次最多 20 次训练的结果。子词范围、桶数、窗口与维度也未在本矩阵中改变。

## 日志、检查点与复现

[搜索汇总 CSV](../artifacts/thucnews_ablation/experiments/f3be9d7811d1f07eb3ea2bfa/search_results.csv) · [编排索引](../artifacts/thucnews_ablation/experiments/f3be9d7811d1f07eb3ea2bfa/index.json) · [进度](../artifacts/thucnews_ablation/progress.json)。

word2vec/cbow_hs/seed42：[日志](../artifacts/thucnews_ablation/runs/2875059751764be4a7d9e37e/run.log) · [逐轮指标](../artifacts/thucnews_ablation/runs/2875059751764be4a7d9e37e/validation_metrics.csv) · [逐轮记录](../artifacts/thucnews_ablation/runs/2875059751764be4a7d9e37e/epochs.csv) · [所选原生模型](../artifacts/thucnews_ablation/runs/2875059751764be4a7d9e37e/checkpoints/epoch_01.gensim) · [保存加载核验](../artifacts/thucnews_ablation/runs/2875059751764be4a7d9e37e/save_load_check.json)。

word2vec/cbow_ns5/seed42：[日志](../artifacts/thucnews_ablation/runs/cd08fa42804016df27c8e772/run.log) · [逐轮指标](../artifacts/thucnews_ablation/runs/cd08fa42804016df27c8e772/validation_metrics.csv) · [逐轮记录](../artifacts/thucnews_ablation/runs/cd08fa42804016df27c8e772/epochs.csv) · [所选原生模型](../artifacts/thucnews_ablation/runs/cd08fa42804016df27c8e772/checkpoints/epoch_06.gensim) · [保存加载核验](../artifacts/thucnews_ablation/runs/cd08fa42804016df27c8e772/save_load_check.json)。

word2vec/cbow_ns10/seed42：[日志](../artifacts/thucnews_ablation/runs/ad4fb1bc495f1dba16dc0ce7/run.log) · [逐轮指标](../artifacts/thucnews_ablation/runs/ad4fb1bc495f1dba16dc0ce7/validation_metrics.csv) · [逐轮记录](../artifacts/thucnews_ablation/runs/ad4fb1bc495f1dba16dc0ce7/epochs.csv) · [所选原生模型](../artifacts/thucnews_ablation/runs/ad4fb1bc495f1dba16dc0ce7/checkpoints/epoch_02.gensim) · [保存加载核验](../artifacts/thucnews_ablation/runs/ad4fb1bc495f1dba16dc0ce7/save_load_check.json)。

word2vec/sg_hs/seed42：[日志](../artifacts/thucnews_ablation/runs/2623ffd65282992889c34fd6/run.log) · [逐轮指标](../artifacts/thucnews_ablation/runs/2623ffd65282992889c34fd6/validation_metrics.csv) · [逐轮记录](../artifacts/thucnews_ablation/runs/2623ffd65282992889c34fd6/epochs.csv) · [所选原生模型](../artifacts/thucnews_ablation/runs/2623ffd65282992889c34fd6/checkpoints/epoch_01.gensim) · [保存加载核验](../artifacts/thucnews_ablation/runs/2623ffd65282992889c34fd6/save_load_check.json)。

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_ablation --smoke
conda run -n Gensim python -m gensim_lab.thucnews_ablation
conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage audit
conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage report
```

同一指纹再次运行时复用封存结果；中断训练从最后完整原生检查点恢复，数据／参数／训练代码改变则创建新运行，不混用旧模型。原论文全量复现保持停止。[Gensim Word2Vec 参数](https://radimrehurek.com/gensim/models/word2vec.html) · [Gensim FastText 参数](https://radimrehurek.com/gensim/models/fasttext.html)。网页显示的文档版本与本机 4.4.0 可能不同，本轮参数以保存的版本和原生模型属性核验为准。
