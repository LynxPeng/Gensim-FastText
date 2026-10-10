# Gensim FastText：优先分析的 10 段核心代码

日期：2026-10-09。范围：当前项目 Gensim 4.4.0，以及此前核心代码文档的 C01–C19 编号。

建议选择 **C04、C07、C08、C09、C11、C12、C13、C15、C16、C19**。选择依据是它们是否直接决定训练工作量、更新规则、内存访问，或最终交给下游任务的向量。下表是源码机制判断，没有通过全量计时或任务消融建立影响大小的数值排名。

## 选择及理由

| 编号 | 核心机制 | 对时间／资源的影响 | 对模型或运行结果的影响 |
| --- | --- | --- | --- |
| [C04](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:156>) | UTF-8 字符 n-gram 提取 | 字符长度范围决定每词子词项数，从而影响预处理、缓存大小和训练时读取／更新的输入向量数；训练中使用缓存，不能说每个词对都重新提取 | 决定模型能共享哪些字符特征，影响短词、词形变化和 OOV 表示 |
| [C07](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:277>) | 词表过滤、高频降采样、子词缓存绑定 | 删除未知词和被降采样的词，直接减少有效训练 token；复用子词索引避免重复查找 | 删除 token 会压缩序列，改变实际上下文关系；训练时不把词表外 token 自动当成 OOV 子词样本更新 |
| [C08](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:328>) | 窗口和 SG／CBOW、NS／HS 调度 | SG 对窗口里的每个上下文分别更新，CBOW 聚合后更新；窗口和输出目标决定更新次数及输出计算量 | 决定训练样本与目标。两个输出头使用独立 `if`，同时开启会同时训练，不能当成二选一 |
| [C09](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:395>) | SG + NS 更新内核 | 每个词对合成输入，并执行正例／负例打分和向量更新；负样本数量影响输出工作量 | 优先检查 `n` 分母、正负例碰撞后跳过、饱和分数跳过和输入更新规则；这些实际决定训练轨迹 |
| [C11](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:560>) | CBOW + NS 更新内核 | 聚合窗口内全部整词与字符项，再更新输出；子词项数和负样本数共同决定计算量 | 隐藏层按全部特征项平均；`cbow_mean`、梯度缩放、负例跳过及 sigmoid 截断决定数值行为；直接对应论文复现路线的表示训练 |
| [C12](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:653>) | CBOW + HS 更新内核 | 输出工作量随目标词的 Huffman 路径长度变化；是否快于 NS 取决于词表、词频与负例数，不能预先断言 | 使用词表树及路径二元目标；当前 THUCNews 默认 `sg=0, hs=1, negative=0`，应优先检查这一实际运行分支 |
| [C13](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:728>) | 负采样累计分布 | 建表发生在准备阶段；训练内核从累计表抽样。它的重要性主要在分布定义，不在单独建表耗时 | 默认频次指数 0.75 与此前核对的 0.5 不同，改变负样本频率；仅影响 NS 路线 |
| [C15](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:822>) | 稠密参数分配和初始化 | 字符矩阵内存约为 `4 × bucket × dimension` 字节；影响初始化、内存压力和缓存访问，桶数增长不等于训练时间同比增长 | 初始化与桶容量影响参数起点和哈希碰撞；扩大桶数不保证任务效果单调提高 |
| [C16](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:845>) | 导出向量合成、字符桶重建 | 重建词向量需遍历词表及其字符项，训练结束／加载时会产生额外工作和合成矩阵内存 | 决定下游实际使用的向量；IV 按 `n+1` 合成，与 C09 的 SG 训练分母不同。EOS 特判差异也在这里暴露 |
| [C19](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md:939>) | 文件流训练更新与学习率调度 | 与 C03 的入口一起分析文件流和批次执行；可能减少 Python 语料处理开销，但速度增益须用相同语料与线程设置计时 | 批次／轮次进度决定当前 `alpha`，影响每次更新的步长；该摘录展示 SG 文件流分支，CBOW 文件流由对应函数处理 |

## 当前项目的阅读优先级

先读 **C08、C07、C12**：它们解释当前 THUCNews 的训练样本、有效工作量和 CBOW／HS 实际更新。正式配置见 [thucnews.yaml](<C:/Projects/LLM/Gensim/configs/thucnews.yaml:22>)。

随后读 **C09 与 C16**：将训练输入和最终导出向量放在一起比较，才能看清此前确认的 SG 分母不一致。主实验 SG／NS 路线见 [experiments.yaml](<C:/Projects/LLM/Gensim/configs/experiments.yaml:12>)；当前 THUCNews 的 CBOW／HS 分支不触发 SG 分母问题。

再读 **C11、C13**：它们对应 CBOW／NS 表示训练与采样分布；论文复现配置见 [paper_reproduction.yaml](<C:/Projects/LLM/Gensim/configs/paper_reproduction.yaml:3>)。C13 对 HS-only 实验不生效。

分析规模扩大后的资源问题时，结合 **C04、C15、C19**，分别检查子词项数、矩阵体积和输入／调度开销。代码间存在交互，不能把每段的时间影响简单相加。

## 为什么暂不优先选其它编号

- C01 适合用作参数背景，但只是默认值声明；实际运行配置和上述执行代码更能解释耗时与结果。
- C02 对判断是否等价复现监督分类模型很重要，但其限制检查本身不是训练耗时来源；监督范围差异已在工程报告中单独解释。
- C03 是入口，应配合 C19 阅读；C05、C06、C18 主要补充哈希、包装和依赖关系，此次不重复占用重点名额。
- C10 是 SG／HS 分支，当前三条主要默认路线分别是 SG／NS、CBOW／HS、CBOW／NS，因此先覆盖 C09、C12、C11。
- C14 在专门研究 OOV 推理覆盖率和速度时应优先补充；本次以训练及下游向量为主，选择覆盖 IV 合成的 C16。C17 的句向量包装也不直接对应当前项目自己实现的文档池化。

完整对比与已验证数值见 [工程差异报告](<C:/Projects/LLM/Gensim/reports/Gensim_FastText与论文工程差异.md>)、[audit.json](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/audit.json>)。此次只增加筛选说明，未修改模型或实验配置。
