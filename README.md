# Gensim 与 FastText 中文表示学习和主题分类实验

当前报告：[中文新闻文本表示的比较与 FastText 实现分析](reports/Gensim&FastText-终稿.md)，补充结果见[独立附录 A–D](reports/Gensim&FastText-附录.md)。项目公开地址：[LynxPeng/Gensim-FastText](https://github.com/LynxPeng/Gensim-FastText)。报告保留摘要、实验结果与第三部分的关键代码；正文 14 幅图，附录 23 幅图。

## 公开仓库与数据准备

公开版本保留 Python 源码、配置、测试、Markdown 报告、小型统计结果和图表，以及附带原有许可证的上游源码与评价题组。另提供一份语料清洗清单、小型固定评价划分和候选选参记录，以核验选参边界。大型训练语料、训练模型、编译工具、缓存和日志不随仓库上传。历史报告中指向这些运行产物的链接需要完成相应实验后才可访问；当前报告与独立附录的图像已随仓库提供。

安装方法见下文“环境”部分。数据与工具来源同时列在当前报告末尾：

- [THUCNews 说明](https://github.com/thunlp/THUCTC#中文文本分类数据集thucnews)与[语料下载](https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip)。项目的 `thucnews_data` 入口按配置下载并校验语料。
- [人民日报标注语料官方入口](https://opendata.pku.edu.cn/dataset.xhtml?persistentId=doi%3A10.18170%2FDVN%2FSEYRX5)。按提供方条件取得相应版本后，将一月至三月词性标注文本置于 `data/raw/199801.txt`、`199802.txt`、`199803.txt`。主题标签由本项目另行自动生成，并非该语料的人工主题标签。
- 词向量评价题组与固定版本源码见 `references`；可用 `tools/fetch_word_benchmarks.py` 和 `tools/fetch_sources.py` 核验或取得。依赖版本锁定于 `requirements.lock.txt`，工具官网见报告末尾。

主题自动标注从环境变量 `DEEPSEEK_API_KEY` 读取访问凭据。公开仓库不包含凭据、标注请求记录或完整新闻文本。已有自动标签及划分的生成过程见[主题分类复现指南](docs/人民日报主题分类复现指南.md)。

## 公开版本核验

报告与源码测试可在未下载新闻语料、未训练正式模型的情况下运行。模型相关测试使用构造的小样本；测试过程中不调用标注接口。

```powershell
$env:PYTHONHASHSEED = '0'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
conda run -n Gensim python tools/check_gensim_fasttext_report.py
conda run -n Gensim python -m pytest tests -q --basetemp temp/pytest-public
```

`CUBLAS_WORKSPACE_CONFIG` 用于 CUDA 测试中的确定性运算。逐文件发布清单见 [公开版本文件清单](docs/公开版本文件清单.md)；构建公开快照可使用 `tools/export_public_project.py`，每个上传资源文件限制为 5 MiB。

## 项目范围与历史实验

工程包括清华 THUCNews 的 TF-IDF、Word2Vec、FastText 表示学习，以及人民日报八类主题分类。分类包括原生 FastText 的两种初始化路线，以及三种冻结表示接同一 LinearSVC 的对照；主题标签由 DeepSeek 单次自动生成。旧版面分类属于历史实验，不作为主题识别结果。THUCNews 类别不参与表示模型训练和词向量选轮。文档全部使用 Markdown，计算数值为 CSV/JSON，图为 PNG／SVG，模型保留二进制。

主要阅读入口为[综合研究报告](reports/综合研究报告.md)：第一章基于 `paper` 目录中的《Bag of Tricks for Efficient Text Classification》arXiv v3 重写；文献综述源文件为 [FastText 论文解读与综述](docs/FastText论文解读与综述.md)，报告生成时自动纳入。报告按“摘要、FastText 论文解读、FastText 复现、核心代码阅读笔记、Gensim 向量化实验、实现方法对比、参数比较”组织。第二章已接入原论文的完整公开数据，原生监督与 Gensim 两阶段近似路线分开报告；全量进度见[复现记录](reports/论文实验复现.md)。逐运行、逐类、逐关系、逐轮等明细另存 CSV，见[数据总索引](reports/data/csv_index.csv)。生成器同时更新报告和 CSV，并核验章节结构、来源、记录数、精度读回及资源链接。

```powershell
# 从现有资源重新编制单文件报告；不会训练模型
conda run -n Gensim python tools/build_consolidated_report.py
# 仅检查已保存报告的覆盖、来源指纹、结构、内部引用及资源路径
conda run -n Gensim python tools/build_consolidated_report.py --check
```

## 先看结果

当前主题分类使用**人民日报 9801–9803**：一、二月训练，三月按主题和重复组划为30%验证、70%测试。八类为政治、财经、科技、体育、娱乐、社会、军事、国际；版号不进入标注请求、模型特征或划分依据。原生 FastText 比较随机与预训练初始化，六档预算由验证 Macro-F1 选定，再做三种子测试。三种冻结表示复用相同标签及5,333／820／1,910篇划分，统一接 LinearSVC，验证选择 C={0.1,1,10}。TF-IDF、Word2Vec、Gensim FastText 的测试 Macro-F1 分别为80.15%、78.60%、78.01%；配对差值区间均包含零。LinearSVC无epoch接口，另保存1—20次求解预算的验证准确率曲线，不能解释为逐epoch训练。原生实验保存于 `artifacts/pfr_topics`，冻结表示对照保存于 `artifacts/pfr_topic_svm`；结果写入[提交报告](reports/Gensim&FastText.md)1.2与[原生独立报告](reports/人民日报原生FastText主题分类.md)、[LinearSVC对比报告](reports/人民日报三种表示LinearSVC对比.md)。命令和弱标签解释见[主题分类复现指南](docs/人民日报主题分类复现指南.md)。

```powershell
# 现有成功标签会直接复用；正式运行需先完成标注
conda run -n Gensim python -m gensim_lab.pfr_topics prepare
conda run -n Gensim python -m gensim_lab.pfr_topic_native run
# 训练日志
Get-Content artifacts/pfr_topics/native_training.log -Tail 20
# 三种表示接同一LinearSVC；直接复用封存标签，不需要API密钥
conda run -n Gensim python -m gensim_lab.pfr_topic_svm run
conda run -n Gensim python -m gensim_lab.pfr_topic_svm_iterations
# 当前比较结果和求解迭代日志
Get-Content artifacts/pfr_topic_svm/training.log -Tail 20
Get-Content artifacts/pfr_topic_svm/iteration_diagnostic.log -Tail 20
```

旧版面分类的数据、模型与日志保留在 `artifacts/pfr_frozen_classification`，其方案及复现入口仅供历史审查，不沿用结论。

2026-10-10 已按小规模搜索确定的候选参数完成全量重训并通过独立核验：Word2Vec 使用 Skip-gram＋NS5、alpha=0.025、300 维；Gensim FastText 使用 Skip-gram＋NS10、alpha=0.05、100 维。沿用 821,609 篇训练文章、种子 42、20 个训练线程以及原全量实验的 5–15 轮验证选轮规则。Word2Vec 实际训练 5 轮，选中第 2 轮；FastText 实际训练 14 轮，选中第 11 轮。WordSim240 测试 Spearman 分别为 0.5061 和 0.4963，比旧基线提高 0.0186 和 0.0442。两个模型从头训练，正式索引已统一更新，旧 CBOW＋HS 模型继续保留。配置见 [thucnews_selected_full.yaml](configs/thucnews_selected_full.yaml)，完整指标、验证曲线及运行记录见 [候选参数全量重训练](reports/THUCNews候选参数全量重训练.md)。

```powershell
# 候选参数全量训练；中断后恢复，封存完成的模型会校验后复用
conda run -n Gensim python -m gensim_lab.thucnews_selected_full
# 独立核验当前候选参数全量实验，不重新训练
conda run -n Gensim python -m gensim_lab.thucnews_selected_full --stage audit
# 查看两模型进度、最佳轮次及日志位置
powershell -File tools/read_thucnews_selected_full_progress.ps1
```

总进度及进程记录保存在 `artifacts/thucnews_selected_full`；逐轮训练进度位于 `artifacts/thucnews_validation/progress.json`，各运行目录保存 `run.log`、`epochs.csv`、`validation_metrics.csv` 和完整检查点。两个模型均完成并通过独立核验后才统一更新正式模型索引。负采样模型不调用仅适用于 HS 的上下文对数评分。下文的 CBOW＋HS 全量结果属于历史基线，当前提交报告的手工修改继续保留。

当前搜索使用固定 5 万篇，见[小规模搜索方案](docs/THUCNews小规模搜索方案.md)。训练方式阶段已完成 20 组。学习率扩展为 **0.005、0.01、0.025、0.05、0.1** 五档，**每档均做种子 42、2026、7**：Word2Vec 固定 Skip-gram＋NS5，FastText 固定 Skip-gram＋NS10，共 30 组学习率配置，复用 14 组、补充 16 组。与训练方式阶段共享 6 组，总计 44 次不同配置训练。先训练 3 轮，验证曲线仍上升则继续，最多 15 轮；每次独立选轮，五档都完成三种子后按验证均值选参。两个模型分别绘图，同一模型的五档学习率在同图比较，显示原始种子点及均值±样本 SD。本轮只用验证选参，不计算测试指标、不自动启动全量学习。进度见 `artifacts/thucnews_small_search/progress.json`，结果见[小规模对比报告](reports/THUCNews小规模超参数搜索.md)。

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid
conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage stop
conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid --stage audit
conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid --stage report
```

新入口按[五档网格配置](configs/thucnews_alpha_grid.yaml)逐组核验、复用封存结果，补齐全部三种子后统一选参；中断后从最后完整轮次恢复。原工作器协议、数据、三档阶段索引和模型保持不变，[三档历史记录](reports/THUCNews学习率三档历史记录.md)保留先前结果。旧 `thucnews_small_search` 训练入口只对应历史三档方案，不用于五档搜索；其 `--stage stop` 仍可停止当前工作器。单个编排进程持有项目锁。正式资源位于 `artifacts/thucnews_small_search`，既有冒烟记录不能代替正式结果。历史[全量消融实验](reports/THUCNews训练方式消融实验.md)保持暂停，原论文全量复现也保持停止。

学习率阶段完成后，新增[向量维数搜索](docs/THUCNews向量维数搜索方案.md)：50、100、150、200、300 维，每档均做种子 42、2026、7，共 30 组，复用 100 维六组，其余 24 组重新训练。Word2Vec 固定 Skip-gram＋NS5、alpha=0.025；Gensim FastText 固定 Skip-gram＋NS10、alpha=0.05。训练和评价口径保持一致，按三种子验证均值分别选维数，另比较资源成本。运行 `conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid`；停止仍用上面的 `--stage stop`，完成后分别运行该模块的 `--stage audit` 与 `--stage report`。最新进度文件已用于维数阶段，结果见[维数实验报告](reports/THUCNews向量维数搜索.md)。历史学习率结果保存在原实验目录；上面的学习率 `--stage audit/report` 使用最新指针，维数阶段应使用新入口。

维数阶段已完成全部 30 组并通过核验。Word2Vec 的候选维数为 300，WordSim240 验证 Spearman 为 0.5724±0.0103；Gensim FastText 保留 100 维，为 0.5949±0.0307（均为三种子均值±样本 SD）。Word2Vec 的最高均值位于已测维数上界；两类模型的选择均限定于固定学习率、当前子集与训练预算。八幅图已加入两个提交报告的 1.1.3.3。本阶段的复现入口为：

```powershell
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage audit
conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage report
```

当前新增实验见 [THUCNews 词向量实验](reports/THUCNews词向量实验.md)及用户维护的[提交报告](reports/Gensim&FastText.md)。完整压缩包实测 836,075 篇，清洗去重后 821,609 篇训练、5,000 篇留出。TF-IDF 复用同一训练集的统计模型；Word2Vec、FastText 各自从头训练原生 Gensim 模型。两个模型均为 CBOW、100 维、窗口 5、最低词频 5、HS-only、种子 42、20 线程，分别通过逐轮验证选择轮数。本轮仅做表示学习与内在评价，没有新增类别预测器。分词使用 8 个进程。具体口径见[实验方案](docs/THUCNews词向量实验方案.md)。

评价题目去重后固定划分验证／测试，等价题组跨基准共享分区。主指标为 WordSim240 验证 Spearman：至少 5 轮、最多 15 轮、连续 3 轮无超过 0.001 的有效提升时停止。每轮保存原生检查点、Pearson／Spearman、覆盖率和固定类比验证探针；独立选择完成后才评价全部测试题。TF-IDF 是稀疏文档表示，不参加词对和类比评价。历史分类诊断及 5 万篇结果保留，不能与本次新测试子集直接比较。

```powershell
# 下载/核验、全量处理、三种表示与 Markdown 报告
conda run -n Gensim python -m gensim_lab.thucnews --stage all
# 单独准备数据、训练或更新报告
conda run -n Gensim python -m gensim_lab.thucnews --stage prepare
conda run -n Gensim python -m gensim_lab.thucnews --stage train
# 单独冒烟或分别训练两个词向量模型
conda run -n Gensim python -m gensim_lab.thucnews_validation --smoke
conda run -n Gensim python -m gensim_lab.thucnews_validation --model word2vec
conda run -n Gensim python -m gensim_lab.thucnews_validation --model fasttext
conda run -n Gensim python -m gensim_lab.thucnews --stage report
# 查询学习到的词向量
conda run -n Gensim python -m gensim_lab.thucnews --stage query --model fasttext --word 经济
```

本轮运行索引为 `artifacts/thucnews_validation/latest.json`，实时阶段为同目录 `progress.json`；各模型 `runs/<指纹>/run.log` 保存日志，`epochs.csv` 保存轮次和选择状态，`validation_metrics.csv` 保存逐轮指标。`checkpoints/epoch_NN.gensim` 保存每轮完整检查点，`model.gensim` 为验证选定模型，`vectors.kv` 用于查词，`vectors.vec` 可供原生 fastText 初始化词表内向量。独立核查入口为 `conda run -n Gensim python -m gensim_lab.thucnews_audit`，同时保存所选模型的全部留出句子 HS 上下文评分至 `artifacts/thucnews_validation/posthoc/<指纹>/heldout_sentence_scores.csv`；评分不作为整句生成概率或早停指标。图表采用 SciPilot 的数据剖析、预览及审核流程，不设变化幅度门槛。旧统计代码已移除，只保留加载历史 pickle 所需的兼容名称。

绘图和提交报告的更新顺序：先运行 `python -m gensim_lab.submission_figures --stage preview`，检查 `artifacts/submission_section11_figures` 的彩色／灰度 PNG，并把审核状态及预览 SHA-256 写入该目录 `visual_review.json`（现有文件提供字段示例）。随后运行 `--stage export`，导出 300 DPI PNG 和 SVG。运行 `python -m gensim_lab.thucnews_probes` 保存定性 API 探针，再更新 THUCNews 报告及 `python -m gensim_lab.thucnews_submission`；后者只更新提交报告 1.1，保留 1.2 及之后内容。所有 Python 命令均在 Gensim 环境运行。

此前的原论文全量复现已按用户要求停止，保留 17/90 个完成运行，见 [停止记录](artifacts/paper_reproduction/stopped.json)。THUCNews 入口不会启动旧论文实验。

本轮审查修正了 Doc2Vec 文章编号影响推断的问题，并通过补充参数实验改善原生 FastText 的拟合。更新结果、日期迁移的限制与完整核查证据见下表；新增调参结果明确标为探索。

| 内容 | 文档 |
| --- | --- |
| 实验结果、资源与时间泛化 | [最终实验报告](reports/最终实验报告.md) |
| 全部高级要求验收 | [高级要求完成清单](reports/高级要求完成清单.md) |
| 论文目标与周边源码 | [论文阅读与源码分析](reports/论文阅读与源码分析.md)、[固定源码行号](reports/源码位置索引.md) |
| 参数、速度、OOV 与碰撞 | [参数分析](reports/参数分析.md) |
| 词相似度/类比、逐句 score 与补充图 | [全面评价报告](reports/词向量全面评价报告.md)、[数据剖析](reports/词向量评价数据剖析.md)、[评价计划](docs/全面评价与绘图计划.md) |
| CUDA 支持与本机检查 | [CUDA 可用性评估](reports/CUDA可用性评估.md) |
| 深度核查与补充改进 | [数据与实验核查](reports/数据与实验深度核查.md)、[补充实验报告](reports/补充实验与改进报告.md)、[验证计划](docs/审查与补充验证计划.md) |
| 实际探针与独立重训 | [运行核验证据](reports/运行核验证据.md) |
| 数据清洗与损失 | [数据质量](reports/数据质量报告.md)、[各版面清洗损失](reports/各版面清洗损失.md) |
| 逐类 PRF、混淆和错误 | [逐类指标](reports/逐类指标与混淆矩阵.md)、[错误分析](reports/错误分析.md) |
| 配置、日志与数值 | [实验记录](reports/实验记录.md)、[汇总 CSV](reports/实验汇总.csv) |
| 作业材料 | [原文转录](docs/作业原文转录.md)、[需求说明](docs/作业需求说明.md)、[实施计划](docs/实施计划.md) |

## 环境

用户确认本项目使用专用 conda 环境 `Gensim`，这是全局默认 `Codex` 环境规则的例外。当前环境路径 `C:\Users\wangj\.conda\envs\Gensim`，Python 3.12.15。核心依赖 Gensim 4.4.0、fasttext-wheel 0.9.2、NumPy 1.26.4。全部精确版本见[锁定清单](requirements.lock.txt)和[环境快照](artifacts/environment.json)。

从项目根目录，在 PowerShell 中安装或复现：

```powershell
conda env create -f environment.yml
conda run -n Gensim python -m pip install -r requirements.lock.txt
conda run -n Gensim python -m pip install --editable . --no-deps
conda run -n Gensim python -m pip check
```

环境已存在时跳过创建。也可使用 `& 'C:\Users\wangj\.conda\envs\Gensim\python.exe'` 代替 `conda run -n Gensim python`。程序自动设置 `PYTHONHASHSEED=0`；默认训练 20 线程、数值库 4 线程、预处理 8 进程，配置见 `configs/runtime.yaml`。模型工作进程顺序启动，内部并行；多线程不保证逐位确定性。不要同时启动两个 run 进程写同一 artifacts 索引。

当前流程使用操作系统文件锁保护 prepare/run/evidence/report 和补充实验的写入，进程退出会自动释放；保留的 orchestration.lock 文件不代表正在占用。缓存复用核验完整文件、预测 ID/标签与 SHA-256，损坏运行保留到 artifacts/cache_recovery 后重建，旧选参记录会重新核对。

Windows 原生 FastText 使用第三方 `fasttext-wheel` 的 cp312 win_amd64 wheel；版本、文件名、下载 URL、SHA-256 和上游源码阅读版本见[来源清单](references/source_manifest.json)。该分发包提供 `import fasttext`，不是另外安装同名源码包。NumPy 固定 1.26.4，避免 fastText 0.9.2 Python 接口与 NumPy 2 的数组复制行为冲突。

本机有 RTX 5070，Gensim 表示模型、原生 FastText 和 LinearSVC 使用 CPU。第二章的外部 Huffman 标签分类头使用 PyTorch 2.7.1+cu128，已验证 CUDA 张量计算；精确标签树搜索使用 CPU。此前中文实验的检查依据见[CUDA 可用性评估](reports/CUDA可用性评估.md)。

## 第二章：原论文全量复现

八套分类数据及完整 YFCC100M 数据保存在 `data/paper_reproduction/raw`，压缩包与解压文件均记录 SHA-256。YFCC 原训练 91,188,648 行，原验证、测试各 930,497 行；词和标签最低频次 100，有效测试 543,424 行与论文一致。训练保留全量原始数据。原生模型联合监督训练，Gensim 先训练无监督 FastText，再接 LinearSVC 或外部 Huffman 分类头，属于近似对照。

本章使用用户批准的 20 个 CPU 训练线程、三个种子 42/2026/7，多线程不保证逐位确定性；报告给出均值与样本标准差。全项目已取消正式训练的单线程限制；历史结果仍标注实际线程数。完整配置见 [paper_reproduction.yaml](configs/paper_reproduction.yaml)，精确依赖见 [requirements.paper.lock.txt](requirements.paper.lock.txt)。Torch CUDA wheel 来自官方 cu128 索引，不能用普通 CPU wheel 替代。

```powershell
# 已下载的数据会被校验后复用；新机器先安装依赖、下载原始数据
conda run -n Gensim python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
conda run -n Gensim python -m pip install -r requirements.paper.lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
conda run -n Gensim python tools/fetch_paper_data.py --stage all
conda run -n Gensim python tools/probe_yfcc_vocabulary.py
# 冒烟不计入正式结果；全量共 90 个运行，顺序执行
conda run -n Gensim python -m gensim_lab.paper_reproduction --stage all --task classification --datasets ag_news --seeds 42 --smoke
conda run -n Gensim python -m gensim_lab.paper_reproduction --stage all --task all
# 仅更新第二章及两份综合报告，不训练模型
conda run -n Gensim python -m gensim_lab.paper_reproduction --stage report
```

每个运行保存自包含配置、源码与数据指纹、模型、完整预测、逐类指标和全分区重载检查。轮次检查点与同协议的完整结果可复用；旧单线程结果不计入当前汇总。原论文实验保持停止；进度保存在 `artifacts/paper_reproduction/progress.json`，历史训练日志在各运行目录。

## 四个入口

```powershell
# 解析、清洗、去重、近重复分组与一月固定划分
conda run -n Gensim python -m gensim_lab prepare

# 七种方法的小样本验证，快速检查环境；不产生正式测试结果
conda run -n Gensim python -m gensim_lab run --profile smoke

# 完整主矩阵；首次按先单种子、后消融、再三种子复验的顺序执行
conda run -n Gensim python -m gensim_lab run --profile full --seeds 42
conda run -n Gensim python -m gensim_lab run --profile ablations
conda run -n Gensim python -m gensim_lab run --profile full --seeds 42,2026,7

# 时间扩展，完成后统一测试此前主矩阵、选定模型与未来月份
conda run -n Gensim python -m gensim_lab run --profile temporal --seeds 42,2026,7 --finalize

# 补充探索及完整审查：探针/重训、重载、数据核查、报告与测试
conda run -n Gensim python -m gensim_lab run --profile supplemental
conda run -n Gensim python -m gensim_lab.word_evaluation --stage all
conda run -n Gensim python -m gensim_lab.word_precision_matrix
conda run -n Gensim python -m gensim_lab.word_fasttext_score
conda run -n Gensim python -m gensim_lab.word_evaluation --stage report
# 确认生成的预览无裁切/缺字/遮盖并记录审阅后，导出报告图和 SVG
conda run -n Gensim python -m gensim_lab.word_evaluation --stage report --export
conda run -n Gensim python -m gensim_lab.word_checks
conda run -n Gensim python tools/finalize_review.py

# 单独更新核验证据与报告时，先更新深度核查
conda run -n Gensim python -m gensim_lab evidence
conda run -n Gensim python tools/deep_audit.py
conda run -n Gensim python -m gensim_lab report
```

当前绘图不设置极差或变化幅度门槛，所有有限的实测比较均可绘图；按数据类型和样本量选择图型，保留原始 CSV/JSON。旧版采用阈值的图表作为历史记录保留，可从 `artifacts/word_evaluation/presentation_history` 追溯。提交报告 1.1.3 的绘图入口为 `conda run -n Gensim python -m gensim_lab.submission_figures --stage preview`，视觉核验后使用 `--stage export`。

自动按上述次序执行并断点复用缓存：

```powershell
conda run -n Gensim python tools/run_all.py
```

当前进度保存为 `artifacts/pipeline.json`。数据、代码、依赖版本或配置变化会生成新的指纹目录；历史缓存不计入当前报告。prepare 的 `--force` 可重新解析相同数据；训练入口自动校验并重建损坏缓存。evidence 已自动完成七种方法的独立重训，并为 Doc2Vec 调用正确的协议入口。

预测时将 `--model` 指向 `artifacts/runs` 内保存了 bundle.joblib 的运行目录。下列 PowerShell 示例自动选择验证集选定的 Gensim FastText：

```powershell
$selection = Get-Content artifacts/selection.json -Raw | ConvertFrom-Json
$modelPath = Join-Path (Get-Location) ('artifacts/runs/' + $selection.fasttext)
conda run -n Gensim python -m gensim_lab predict --model $modelPath --text '经济 市场 改革 发展' --tokenized
```

去掉 `--tokenized` 会用 jieba 处理未分词文本，其分词与原语料人工标注不同，输入分布可能变化。输出为 JSON，含版面、分数类型和覆盖率；SVM 边距不能当作概率。本文命令从工程根目录执行，`gensim-lab` console script 与 `python -m gensim_lab` 等价。

## 数据与模型目录

| 路径 | 内容 |
| --- | --- |
| data/raw | 六个 UTF-8-sig 月度文本及一致内容的压缩包，实验仅读取文本 |
| data/processed | 逐月 JSONL、原始 SHA、排除清单、异常、近重复对及一月划分 |
| configs/experiments.yaml | 主模型、类别规模、种子与全部单因素取值 |
| artifacts/runs/配置指纹 | spec、summary、训练/评估日志、逐篇预测、指标、模型、诊断、保存加载检查 |
| artifacts/selection.json | 仅基于一月验证 Macro-F1 的配置选择 |
| artifacts/temporal_splits.json | 扩展训练、五月验证、六月测试的文章 ID |
| artifacts/reproducibility | 七种方法独立进程重训模型及字节一致校验 |
| artifacts/evidence | 哈希、OOV、采样、迭代、学习率、数组、原生接口及保存加载探针 |
| references/sources | 固定版本开源代码阅读副本及许可证 |
| reports | 自动汇总和人工分析的 Markdown；figures 中为报告插图 |

每个 spec 内含完整划分 ID；模型只拟合 train，validation 用于 C 与表示参数选择。`metrics_test.json` 在 finalize 后生成。消融独有配置只保留验证结果；与正式主配置重合的缓存可以共用测试结果，但不用于消融选参。frozen 运行通过 source_run 载入一月模型，训练时间为 0。

Gensim 模型另导出 `representation.gensim`、词典 `dictionary.gensim` 和适用的 `idf.gensim`，可由对应 Gensim 类的 load 直接载入；保存加载值核验见 native_export.json。完整分类使用 Bundle；同内容的额外原生导出不重复计入报告的模型大小。FastText 原生监督模型为 native.bin。

Doc2Vec 当前使用 tokens_sha256_v1 推断协议：从词序列生成稳定随机种子，同一文本更换编号或输入顺序不会改变文章向量。协议源码 SHA-256 随配置保存；修正前依赖编号的历史模型保留，但不进入当前结果，predict 入口要求使用修正后模型。训练文档 ID 仍用于追踪，不作为分类输入。

覆盖率表示能否得到单词向量，不代表语义正确率。原生 FastText 的 zero_documents 字段统计原始 token 全部未命中单词表的文章；EOS 和词 bigram 仍可能提供监督特征，不能把该计数解释为监督输入必然为零。Doc2Vec 全部 token 未登录时，修正协议实际输出零向量并计入 zero_documents。

原生 FastText 的 `native_train.txt` 是模型要求的训练输入，属于计算资源；每篇一行 `__label__pNN 词 ...`。原文转录及需求说明的旧 Word 文件归档到内部 .artifact-work/history-docx，Markdown 版本是维护和交付入口。

## 核验

```powershell
conda run -n Gensim python -m pytest --basetemp .artifact-work/pytest-temp -q
conda run -n Gensim python -m gensim_lab evidence
conda run -n Gensim python -m gensim_lab report
```

测试覆盖原分词/实体标注清理、异常段落、近重复分组与分层隔离、训练词典不接触验证、OOV 两条路径、保存加载和 Doc2Vec 推断顺序。evidence 使用真实模型完成手工向量及哈希复算，并独立重训一月 12 类全部七种方法。report 核验模型与种子矩阵、分区文章/组、参数覆盖、有限指标、保存加载及 Markdown 本地链接。

后续深度核查和补充验证使用下列入口。原测试已查看后的新增调参结果明确作为探索，不视为新的盲测；原始主配置保留，Doc2Vec 的编号依赖修正单独记录。

```powershell
conda run -n Gensim python -m gensim_lab run --profile supplemental
conda run -n Gensim python tools/deep_audit.py
conda run -n Gensim python tools/check_saved_predictions.py

# 一次完成补充报告、独立重训、完整重载、深度核查与测试
conda run -n Gensim python tools/finalize_review.py
```

Markdown 表格列数、代码围栏与图片引用结构可用 `conda run -n Gensim python tools/check_markdown.py` 检查，结果保存为 artifacts/markdown_structure_check.json；关键实验报告图已查看。

## 软件与文献

科学结论依据固定论文、上游源码和当前实际运行，完整书目信息见[论文阅读](reports/论文阅读与源码分析.md)。引用元数据核对流程使用 Scientific Agent Skills 的 citation-management 指南，工具方法参考：Timothy Kassis、Vinayak Agarwal、Yuhuan He、Darshil Patel、Aubrey M. Brueckner，2026，[*Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*](https://arxiv.org/abs/2609.00065)。该指南用于参考文献校验，不是模型或实验数据来源。
