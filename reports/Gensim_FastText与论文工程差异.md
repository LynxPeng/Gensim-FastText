# 当前项目 Gensim FastText 与论文的工程差异

审查日期：2026-10-09。运行环境：`C:/Users/wangj/.conda/envs/Gensim/python.exe`，Python 3.12、Gensim **4.4.0**、`fasttext-wheel` **0.9.2**。本文核对当前文件和实际加载的编译内核，不以旧报告结论代替本次核查。

核心代码另见 [Gensim FastText 核心代码](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md>)：原样摘录 Gensim 的 Python／Cython 训练与向量查询代码，附项目调用代码和原生 C++ 对照。

## 1. 先明确比较的是哪一种 FastText

项目保存了两篇论文，它们描述的训练任务不同：

| 对照对象 | 本文定位 | 项目中的对应路线 |
| --- | --- | --- |
| *Bag of Tricks for Efficient Text Classification* | 监督文本分类；重点核查 §2、§2.1、§2.2 和 §3.1 的工程要求 | 原生 `fasttext.train_supervised`；Gensim 词向量接外部分类器是两阶段近似 |
| *Enriching Word Vectors with Subword Information* | 无监督字符子词词向量；重点核查 §3.1、§3.2、§4.2、§4.3 | `gensim.models.FastText` 的直接算法对照 |

项目分类论文 PDF 是 **arXiv:1607.01759v3，2016-08-09**；子词论文 PDF 为 TACL 排版，方法与 2017 年正式发表版本对应。论文原文均已提取并检查关键页的渲染图；本文不评估全文实验结论。公开核对入口：[分类论文](https://aclanthology.org/E17-2068/)、[子词论文](https://aclanthology.org/Q17-1010/)。

**主要结论：当前项目的 Gensim 路线不能视为分类论文模型的等价实现；即使只比较子词词向量，4.4.0 内核也有真实的归一化和采样差异。** 其中最值得优先记录的是：

1. Gensim 缺少分类论文的标签联合训练和词 bigram；项目用词向量池化接 SVM／外部标签树。
2. Skip-gram 训练用字符子词数 `n` 作分母，而向量查询用 `n+1`；原生 0.9.2 训练也用 `n+1`。这是本次通过编译内核实测确认的内部不一致。
3. 默认负采样指数为 `0.75`，子词论文与原生实现采用 `0.5`。
4. `compute_loss=True` 不会得到有效的 FastText 训练 loss；导入原生模型时，特殊 token `</s>` 也存在合成差异。

这些结论分别属于模型范围、实际内核行为或接口问题，不能全部归因于本项目“调用错误”，也不能据此声称已测定它们对分类准确率的影响。

## 2. 与监督分类论文相比：模型结构没有对齐

分类论文 §2 的比较基准是监督目标直接训练输入与输出矩阵；§2.1 的树组织类别，§2.2 的 n-gram 表示相邻词组合。下表据此检查当前代码。论文定义见 [§2 原文](https://aclanthology.org/E17-2068.pdf)。

| 编号 | 当前实现／调用 | 不一致及工程影响 | 证据 |
| --- | --- | --- | --- |
| A1 | Gensim `FastText` 继承 `Word2Vec`，从上下文训练词表示；项目另拟合分类器 | 输入表示不会收到分类标签的反向梯度；两阶段学习与论文联合优化不是同一训练过程 | 核心代码 C02、C08、P01、P03；[models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:52>) |
| A2 | `word_ngrams` 只接受 `1`，设置 `2` 直接抛 `NotImplementedError` | 缺少分类论文用于局部词序的 word bigram。`min_n/max_n` 调的是字符长度，不能补上词序特征 | C02；运行探针 `word_ngrams_2_error` |
| A3 | `syn1neg` 的行对应词；`syn1` 的树也是词表树 | `hs=1` 不能使 Gensim 自动成为论文中的标签层次 softmax；NS 训练目标也不是类别 softmax | C08–C13；[paper_worker.py](<C:/Projects/LLM/Gensim/gensim_lab/paper_worker.py:304>) |
| A4 | 主路线取词向量平均／TF-IDF 加权，再对文档向量做 L2 单位化；之后拟合 `LinearSVC` | 改变特征几何、目标函数和分数含义；SVM margin 不能直接解释为论文的分类概率 | P02、P03、P04；[models.py](<C:/Projects/LLM/Gensim/gensim_lab/models.py:122>) |
| A5 | 标签任务使用项目自定义 `HuffmanClassifier`，输入为已缓存的文档向量 | 虽然输出层也是标签树，表示仍被冻结；只能说外接标签层次分类头，不能说完成论文端到端训练 | [paper_worker.py](<C:/Projects/LLM/Gensim/gensim_lab/paper_worker.py:311>) |

分类路线的核心对比可以写成：

$$
\text{当前 Gensim 路线：}\quad \text{无监督词／子词训练}\ \rightarrow\ \text{冻结表示并池化}\ \rightarrow\ \text{外部分类器}.
$$

Gensim 输出查询中的 `wv.vectors` 不是监督类别矩阵。它同时包含整词输入向量和字符桶输入向量的合成结果；两层线性映射的形状相似，不足以证明训练等价。

**一个容易误判的细节：** 按词类型累加 `frequency * vector` 再除以 token 总数，与逐 token 求均值在代数上相同。项目用 `Counter` 加速这一点本身没有引入偏离。真正额外的变化是字符表示、TF-IDF 选项、L2 单位化和后续分类目标。

## 3. 与子词词向量论文相比：训练内核的差异

子词论文 §3.2 写出子词向量求和的打分形式；§4.3 给出 SG、5 个负样本和频次平方根采样。基准公式与参数位置可在 [论文第 3–4 页](https://aclanthology.org/Q17-1010.pdf) 核对。

### B1. 论文求和、原生均值、Gensim SG 的分母：三个口径

定义 `n` 为**字符 n-gram 项数**，不含独立整词向量；哈希重复项按次数保留。令

$$
S_w=u_w+\sum_{g\in\mathrm{char}(w)}z_{h(g)}.
$$

| 位置 | 实际使用的输入表示 | 判定 |
| --- | --- | --- |
| 子词论文 §3.2 打分定义 | $S_w$ | 论文表达 |
| 原生 0.9.2 `Model::computeHidden` | $S_w/(n+1)$ | 原生代码已对论文求和形式作了平均化处理 |
| Gensim 4.4.0 SG + NS／HS | $S_w/n$；若 `n=0` 则直接用整词输入向量 | 与原生训练、Gensim 自己的导出分母都不同 |
| Gensim `adjust_vectors`，IV 查询 | $S_w/(n+1)$ | 普通词向量导出与原生一致 |
| Gensim OOV 查询 | 字符桶向量之和除以 `n`，不含整词参数 | 没有整词参数时使用 `n` 合理，不能与 SG 的 IV 问题混为一谈 |

代码证据：C09 的 `norm_factor = ONEF / subwords_len`；C10 相同；C16 的 `len(ngram_buckets) + 1`；N01 的 `input.size()`。原生的 `input` 包含整词 ID，见 N03、N04。原生代码公开入口：[model.cc](https://github.com/facebookresearch/fastText/blob/v0.9.2/src/model.cc)。

因此，这不是仅凭“论文说 sum，代码说 average”推断出的错误。**Gensim SG 与自身导出／原生代码的 `n` 对 `n+1` 不一致**是单独、可验证的问题。若只在推理后给一个词向量乘正数，其自身的余弦方向不变；这里分母用于训练前向，会改变 logit 和更新，不能说“最后用余弦就完全等价”。

本次固定输入总和 `S=6`、输出权重 `0.1`、学习率 `0.01`，调用实际加载的 `train_batch_any`：

| 内核与词 | `n` | 训练隐藏值 | 导出应使用的隐藏值 | 实测输出参数增量 | 与源码计算吻合 |
| --- | --- | --- | --- | --- | --- |
| SG，`a` | 1 | 6 | 3 | 0.021346487 | 是 |
| SG，`ab` | 2 | 3 | 2 | 0.012904420 | 是 |
| CBOW，单一上下文 `a` | 1 | 3 | 3 | 0.012904420 | 是 |

输出增量按实际 sigmoid 查表计算。探针把所有噪声候选设置为真实正例以排除负例干扰，并让另一方向的输入起始为零。它验证的是编译内核行为，不是语料准确率评估。对 `n>0`，SG 训练值相对原生均值多出因子 `(n+1)/n`，短词或子词项少时更明显。这里将其判为**确认的计数不一致、疑似计数缺陷**，没有证据判断上游作者是否有意保留该行为。

### B2. 默认噪声分布不是论文的平方根分布

Gensim 通过 `make_cum_table()` 使用

$$
q(w)\propto f(w)^{\mathrm{ns\_exponent}},\qquad \mathrm{ns\_exponent}_{\mathrm{default}}=0.75.
$$

原生 0.9.2 用 `pow(targetCounts[i], 0.5)`；当前项目三个主要配置均没有显式覆盖 `ns_exponent`，因此使用 `0.75`。代码见 C01、C13、N02；[原生 loss.cc](https://github.com/facebookresearch/fastText/blob/v0.9.2/src/loss.cc)。

实测词频 `[16, 1]`：默认累计表对应概率 `[8/9, 1/9]`；设 `ns_exponent=0.5` 后为 `[4/5, 1/5]`。这会改变高频与低频词作为负例的相对权重，不能通过只对齐 `negative=5` 消除。HS-only 路线不使用负采样，因此此项对它不生效。

### B3. “5 个负样本”在 Gensim 中可能少于 5 个

C09、C11 的循环遇到负样本等于正例时执行 `continue`，占用一次抽样机会；N02 则在 `getNegative` 中反复抽取，直到不等于正例。故 Gensim 的 `negative=k` 表示 `k` 次候选抽样，实际负更新次数可能少于 `k`；原生执行 `k` 次有效负更新。重复的其它负词仍可重复出现。

这种差异在小词表、偏斜频率下更容易显现。它是两份代码的采样协议差异，不是从论文中推导出“必须无放回抽样”。本次 B1 探针也实际利用并验证了候选被跳过的行为。

### B4. SG 饱和分数直接丢弃更新

Gensim SG + NS 在 `f_dot <= -8` 或 `f_dot >= 8` 时跳过该正／负样本。原生 logistic 将 sigmoid 截到 0／1 后仍计算 `(label-f)*lr`；Gensim CBOW + NS 也采用这种截断。特别是“正例却得到很大的负分数”时，继续求梯度与直接跳过有实质差别。

固定输入和 `S=200`、输出权重 `-0.1` 的探针：SG 的 logit 为 `-20`，输出参数增量 **0**；CBOW 的 logit 为 `-10`，输出参数增量约 **1**。证据见 C09、C11 和 `audit.json.kernel_probes`。SG + HS 同样有越界跳过分支（C10）；CBOW + HS 也会跳过（C12），因此不能把 NS 的 CBOW 截断结论推广到所有内核。

这项可能影响异常大范数、饱和样本的训练行为；本次没有统计真实项目语料中触发该条件的频率。

### B5. 平均前向的输入更新没有完整按均值链式缩放

C09–C12 将累计误差 `work` 写回每个整词／字符桶参数。以 `cbow_mean=1` 为例，前向除以特征总数，但输入更新没有再除以该总数；`cbow_mean=0` 反而在更新前缩放 `work`。这沿用 word2vec 风格的更新约定。

探针中单上下文词含整词和 1 个字符项，CBOW 输入均值为 3；实测整词输入增量约 `0.000430107`，对应未缩放梯度 `0.000430147`。若按该平均前向的严格链式导数，增量应约 `0.000215074`。

**此项与原生无监督实现共享：** 原生 `normalizeGradient` 只在 `model==sup` 时打开（N05）。所以不能把它独立列为“Gensim 相对原生的 bug”；它解释的是工程更新规则与数学平均函数的严格导数之间的区别。监督分类平均层则有独立的梯度缩放。由于不同词的特征数不同，也不能一般地用一个统一学习率换算消除。

### B6. CBOW 是扩展选项，当前项目并非一直采用论文 SG

Gensim 默认 `sg=0`。CBOW 的隐藏层把窗口内所有上下文词的整词与字符项一起相加，再按**全部输入特征项**平均，不是“先每词平均，再对词平均”；字符项较多的上下文词因此占更高权重。原生 CBOW 也是拼接全部上下文子词后平均（N04）。

子词论文的提出模型是 SG；CBOW 主要作为对比基线。库支持 CBOW 本身属于功能扩展；项目正式使用 CBOW／HS 时，实验方法已偏离该论文的 SG／NS 协议。不能把扩展功能的存在判为错误，也不能以默认值直接代表当前每条路线的设置。

## 4. 工程接口与边界行为

### B7. 普通字符哈希一致，但 EOS 和输入协议有例外

普通词的 `<word>` 边界、UTF-8 起始字节遍历、FNV-1a 的有符号字节处理、`hash % bucket` 都与原生对应。不能把中文多字节 UTF-8 当作“按字节长度截 n-gram”的错误；长度计数按 UTF-8 字符边界推进。代码见 C04–C06。

本次比较了包含中文、ASCII、重音字符、组合字符、emoji、OOV 和空字符串的 **11 个普通／非 EOS 输入**，字符桶的**有序多重列表**逐项一致；保留重复项，不只比较去重集合。以相同原生 `.bin` 导入 Gensim 后，**10 项**查询（8 个普通 IV 加 2 个 OOV）向量的最大绝对差约 `3.73e-9`，完整明细见 JSON。

特殊 token `</s>` 不符合上述结论：

- 原生词典将换行转为 EOS，并且**不为 EOS 构建字符子词**，只保留其词 ID。
- Gensim 不自动向每行补原生 EOS；如果用户词表或导入模型含 `</s>`，其普通字符桶重建逻辑仍会为它生成字符项。
- 本次导入原生模型后，EOS 向量最大绝对差约 **0.236019**；这只是当前微型模型的差值，不是一般固定误差。

见 C16、N03；本次 `hash_probes` 中原生 EOS 字符桶为空而 Gensim 不为空，`eos_import_vector_max_error` 非零。该问题直接涉及把 EOS 当作普通 token 查询、池化或导入后续训的边界。是否影响具体文章特征，要看项目是否实际输入 EOS；不能由此否定普通词导入兼容性。

分词方面，`FastText(sentences=...)` 接收外部已分词的列表；库不负责完成中文分词。原生按自己的 ASCII 空白规则读取文本。论文复现路线已经用 `TextCorpus` 对齐 ASCII 词表扫描，见 [paper_worker.py](<C:/Projects/LLM/Gensim/gensim_lab/paper_worker.py:290>)。其它路线的中文分词、句子截断则属于项目输入协议。原生额外 EOS 使 token 计数、词表及学习率进度不能简单当作完全相同。

### B8. FastText 的 loss 和 score 接口不能当作实际训练目标的完整观测

`FastText.__init__` 没有 `compute_loss` 形参；继承的 `.train(..., compute_loss=True)` 虽能接收它，但 C03 的 FastText 训练入口没有把该标志传给更新内核，C09–C12 也不累计 `running_training_loss`。

本次用 20 个训练线程训练微型语料：参数确实发生变化，但 `get_latest_training_loss()` 仍为 **0.0**。零返回值不表示训练已达到零损失。若绘制这一接口得到的“训练 loss”，会产生误导。

此外 `FastText.score is Word2Vec.score` 实测为真；它使用继承的 HS 打分路径，不能自动覆盖 FastText 的 SG 特殊分母、字符 OOV 训练语义或外部标签分类目标。项目的 [word_fasttext_score.py](<C:/Projects/LLM/Gensim/gensim_lab/word_fasttext_score.py:1>) 已把该输出标记为兼容性诊断。这里不判定项目另行计算的离线目标值无效，也不把零 loss 接口问题外推到自定义诊断。

### B9. 稠密桶、缓存和 BLAS 属于实现策略，通常不是算法不一致

当前版本为 `vectors_ngrams` 分配 `(bucket, dimension)` 稠密 `float32` 矩阵；整词参数单独存储，预先缓存每词的字符桶索引，并用 Cython／BLAS、线程私有 `work/neu1` 更新。原生将整词与字符桶放在一个输入矩阵中。物理存储拆分和缓存没有改变对应的共享参数关系。

这里不能沿用旧版 Gensim 的“只存出现过的 n-gram 桶”描述。当前 `bucket=200000`、100 维字符矩阵约占 80 MB；设置 2,000,000 则约 800 MB，尚未计算整词／输出／缓存矩阵。`wv.vectors` 是另一个导出缓存，训练参数以 `vectors_vocab` 与 `vectors_ngrams` 为准。

`train(corpus_file=...)` 使用 `fasttext_corpusfile.pyx`，但它调用的 SG／CBOW 更新函数仍来自 `fasttext_inner`（C18、C19）。所以换文件训练入口不会绕开 B1、B3、B4 的内核行为。

## 5. 当前项目实际参数：哪些是配置偏离

以下值直接读取本次审查时的 YAML；代码指纹和完整配置另存于证据 JSON。Gensim 词向量参数的直接基准是子词论文，原生监督参数的基准是分类论文，两种基准不混用。

| 参数 | 子词论文 §3.2、§4.3 | Gensim 4.4.0 默认 | `experiments.yaml` | `thucnews.yaml` | `paper_reproduction.yaml` 的 Gensim 表示 |
| --- | --- | --- | --- | --- | --- |
| 架构／输出目标 | SG + NS | CBOW + NS | SG + NS | CBOW + HS | CBOW + NS |
| 向量维度 | 300 | 100 | 100 | 100 | 分类 10；标签 50／200 |
| 字符长度 | 3–6 | 3–6 | 1–3 | 1–3 | 3–6 |
| 字符桶数 | 2,000,000 | 2,000,000 | 200,000 | 200,000 | 200,000 |
| 负样本数 | 5 | 5 | 5 | 0（HS-only） | 5 |
| 噪声指数 | 0.5 | 0.75 | 0.75（未显式设置） | 不适用 | 0.75（未显式设置） |
| 高频降采样阈值 | 0.0001 | 0.001 | 0.001 | 0.001 | 0.001 |
| 起始学习率 | 子词模型 0.05 | 0.025 | 0.025 | 0.025 | 0.025 |
| 最小词频 | 5 | 5 | 3 | 5 | 分类 1；正式标签任务覆盖为 100 |
| 训练轮次 | 主要实验 5 | 5 | 15 | 5 | 5 |
| 最大窗口 | 5，随机缩窗 | 5，`shrink_windows=True` | 5 | 5 | 5 |
| 训练线程 | 异步共享参数 | 3 | 20 | 20 | 20 |

配置定位：[experiments.yaml](<C:/Projects/LLM/Gensim/configs/experiments.yaml:12>)、[thucnews.yaml](<C:/Projects/LLM/Gensim/configs/thucnews.yaml:21>)、[paper_reproduction.yaml](<C:/Projects/LLM/Gensim/configs/paper_reproduction.yaml:3>)、[runtime.yaml](<C:/Projects/LLM/Gensim/configs/runtime.yaml:1>)。

字符长度 1–3 和较小桶数可以是中文语料上的工程选择；它们不是 Gensim 实现 bug。但应报告为本项目方法设置，而不是原论文复现参数。20 个训练线程符合本项目当前约定；异步更新不保证逐位可复现，不应恢复历史单线程要求。

监督分类的原生路线还有一项独立偏离：当前 `classification.native_loss: hs`，而项目保存的作者 `classification-results.sh` 不传 `-loss`，结合原生 `Args::parseArgs` 的 supervised 默认值对应 **softmax**。大类别标签任务用 HS 有依据，但不能把 8 个小类别分类数据集的 HS 配置也直接当成作者脚本设置。见 [当前配置](<C:/Projects/LLM/Gensim/configs/paper_reproduction.yaml:22>)、[归档脚本](<C:/Projects/LLM/Gensim/artifacts/paper_reproduction/classification-results.sh:89>)、[args.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/args.cc:109>)。

该项属于项目原生基准配置，不是 Gensim 内核缺陷。本次没有重跑分类数据集，也没有把既有消融差值当成本次新实验。

学习率也是协议差异：论文子词模型按 token 进度衰减；Gensim iterable 构造训练默认有 `total_examples`，通常按句子／样本计数调度，文件流路线则按传入的计数调度。原生按共享 token 计数更新并有 `lrUpdateRate` 粒度；Gensim 按批次或文件流进度调度且保留 `min_alpha` 下限。它们都可以称“线性衰减”，实际步长轨迹仍未对齐。见 [word2vec.py](<C:/Projects/LLM/Gensim/references/sources/gensim-4.4.0/models/word2vec.py:1199>)、[fasttext.cc](<C:/Projects/LLM/Gensim/references/sources/fastText-0.9.2/src/fasttext.cc:638>)。

## 6. 如何使用这些结论

| 目标 | 建议的工程处理 | 仅改配置能否完成 |
| --- | --- | --- |
| 与监督分类论文比较 | 将 Gensim 路线标记为“两阶段词向量基线”；原生路线对齐监督目标、词 n-gram、桶数及实验协议 | Gensim 的标签联合训练和 word bigram 不能靠当前接口补齐 |
| 与子词论文训练协议比较 | 单独设置 SG／NS、`ns_exponent=0.5`、字符 3–6、桶 2M、`sample=1e-4`，按任务核对维度、词频、轮次和步长 | 可以对齐多数参数；当前项目参数转发也需显式包含噪声指数 |
| 与原生训练内核精确对照 | 隔离 B1 分母、B3 重抽样和 B4 饱和更新进行受控修改，再评估数值及任务效果 | 不能；须修内核／编译或选用原生训练 |
| 记录训练目标变化 | 不使用 inherited 零 loss 作为有效目标曲线；用定义明确、与训练模式一致的独立观测，并注明查表／采样口径 | 换观测逻辑即可，但必须明确它测量的目标 |
| 导入原生模型 | 核验普通 IV／OOV，同时单独测试 EOS；如使用 EOS，处理其特判与后续训协议 | 特殊 token 行为需额外处理 |

本文只形成审查结果和可复核材料，未改变项目训练代码、安装环境、参数配置或既有实验结果。没有自动替换内核分母，因为这会形成不同版本的模型，需要新的训练和评估记录；本次用户请求是对比审查。

## 7. 证据与复现

核心代码：**[Gensim FastText 核心代码](<C:/Projects/LLM/Gensim/reports/Gensim_FastText核心代码.md>)**，包含编号 C01–C19、P01–P04、N01–N05 的原样连续摘录、原文件行号和源码散列。完整源码副本连同 LGPL 许可证位于 [sources 目录中的 fasttext.py](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/sources/fasttext.py>)；补齐了项目此前未存档的 `fasttext_corpusfile.pyx`、`fasttext_inner.pxd`。

数值证据：**[audit.json](<C:/Projects/LLM/Gensim/artifacts/reviews/fasttext_paper_engineering/audit.json>)**，包括本机版本、加载 `.pyd` 散列、源码一致性、固定权重内核更新、噪声分布、API 错误、字符哈希、原生导入和零 loss 探针。源码与配置指纹可用于检查后续版本是否已变化。

复现探针：**[audit_fasttext_paper_engineering.py](<C:/Projects/LLM/Gensim/tools/audit_fasttext_paper_engineering.py>)**。运行：

```powershell
& 'C:\Users\wangj\.conda\envs\Gensim\python.exe' tools\audit_fasttext_paper_engineering.py
& 'C:\Users\wangj\.conda\envs\Gensim\python.exe' tools\build_fasttext_engineering_core.py
```

所有拟合探针按 `runtime.yaml` 用 20 个 CPU 训练线程，数值库限制为 4 线程。固定权重的直接 Cython 调用是**数值参照探针**，不构成新的单线程正式实验。原生 `train_unsupervised` 的当前 Python 包装器不接受 `seed` 关键字，微型原生探针使用其固定原生默认种子 0；多线程参数更新仍可能浮动。普通向量兼容核验在同一份本次生成的 `.bin` 上进行，因此不依赖两次训练得到相同参数。

审查边界：普通哈希和导入向量为实际交叉核验；训练分母、饱和更新、梯度规则和零 loss 用编译内核数值探针验证；HS 分支、调度、项目训练结构和 YAML 偏离主要依靠逐行源码核查。没有重跑全量训练、测量这些差异造成的泛化变化，或验证其它 Gensim 版本是否已修复。报告中的“影响”是由机制推出的可能性，与已经测得的微型数值结果分开表述。

论文文件来源：

[Bag of Tricks for Efficient Text Classification.pdf](<C:/Projects/LLM/Gensim/paper/Bag of Tricks for Efficient Text Classification.pdf>)

[Enriching Word Vectors with Subword Information.pdf](<C:/Projects/LLM/Gensim/paper/Enriching Word Vectors with Subword Information.pdf>)
