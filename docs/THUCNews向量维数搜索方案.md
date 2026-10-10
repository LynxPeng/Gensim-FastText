# THUCNews 向量维数搜索方案

学习率搜索完成后，在同一固定 5 万篇训练子集上分别比较 Word2Vec 与 Gensim FastText 的向量维数。维数取 50、100、150、200、300，每档包含训练种子 42、2026、7，共 30 组结果。评价划分种子保持 42，主指标仍为 WordSim240 验证集严格词表内 Spearman。

| 模型 | 固定训练方式 | 固定初始学习率 | 字符子词 |
| --- | --- | --- | --- |
| Word2Vec | Skip-gram＋NS5 | 0.025 | 无 |
| Gensim FastText | Skip-gram＋NS10 | 0.05 | 长度 1–3，200,000 桶 |

固定窗口 5、最低词频 5、下采样 0.001、20 个训练线程、数值库 4 线程。使用相同的 23,080,089 词语料、输入顺序、94,426 词词表及 92/95 主验证词对。逐轮评价、检查点、至少 3 轮／最多 15 轮、0.001 有效增幅与近期趋势停止规则沿用学习率搜索，终值学习率为 0.0001。停止规则以当前运行的实际验证曲线执行，不预设固定最佳轮数。

100 维的六组已完成模型在配置与封存文件全部核验一致后复用，其余 24 组从头训练。每档完成三种子后按验证均值分别选参；精确同分时选择较低维数。该选择仍属于在已选训练方式和学习率下的条件搜索，未穷举维数与学习率的交互。

第 3 轮若当前主指标相对前一轮与前两轮均未提高超过 0.001，即停止；此后若连续 3 轮未形成有效提升，且最近一至两轮也无超过 0.001 的增幅，则停止。任何严格更高的得分均更新最佳检查点，0.001 仅为停止规则的增幅阈值。学习率按 15 轮上限线性降至 0.0001，提前停止不重新缩放日程。提前停止的结果不作为充分收敛的证据。

比较验证均值、样本标准差、全部逐种子得分、辅助词相似度和类比指标、覆盖率、最佳／实际轮数、训练总耗时及每轮耗时、峰值进程内存和原生模型大小。先作数据剖析，再分别绘制两个模型的维数效果图、逐轮曲线、辅助诊断和计算成本图。每档真实 n=3，保留原始点；误差棒为样本 SD，图中不标显著性。维数采用真实数值横轴，轮次不补造早停后的数据。使用 scipilot-figure-skill 的程序检查及彩色／灰度读图流程，保存 PNG 300 DPI 与 SVG。

搜索仅计算验证指标。原学习率阶段索引与封存模型保持原样，不自动启动全量训练或类别预测。保存配置、指纹、日志、逐轮指标、检查点与加载复核；完整网格结束后核验所有运行，更新 Markdown 实验报告及提交报告的超参数搜索章节。

## 运行与进度

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid
conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage stop
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage audit
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage report
```

配置见 [thucnews_dimension_grid.yaml](../configs/thucnews_dimension_grid.yaml)。最新索引为 `artifacts/thucnews_small_search/latest.json`，进度为 `artifacts/thucnews_small_search/progress.json`，每组训练日志保存在对应运行目录的 `run.log`。单个编排进程持有项目锁，各模型内部使用多线程。

## 完成记录

30 组结果已完成，其中 24 组新训练、6 组封存复用。Word2Vec 选中 300 维（验证 Spearman 0.5724±0.0103），Gensim FastText 保留 100 维（0.5949±0.0307）；误差均为三个种子的样本标准差。300 维为当前网格上界，候选仍属于固定学习率与训练预算下的探索性选参。

完整结果、八幅图及资源比较见[维数实验报告](../reports/THUCNews向量维数搜索.md)。[运行核验](../artifacts/thucnews_small_search/experiments/c0654ea5f83982f1f307e526/verification.json)确认完整网格、相同词表与覆盖率、逐轮选模及保存加载一致；[图表与文档核验](../artifacts/thucnews_small_search/experiments/c0654ea5f83982f1f307e526/publication_checks.json)逐项核对绘图统计量、全部逐轮点和文档链接。
