# Gensim 调用审查与原生 FastText 对比核查

本报告只做核验，**未修改工程的任何既有文件**。审查使用本机 `Gensim` 环境（Gensim 4.4.0、fasttext-wheel 0.9.2、NumPy 1.26.4）实际运行探针，证据为 `artifacts/reviews/` 下的 JSON，探针源码为 `tools/audit_*.py`；这些探针与证据文件是本次新增的只读核查材料，`gensim_lab/`、`tests/`、`configs/` 均保持原样。

## 1. 结论摘要

Gensim 侧的调用**在算法语义上基本正确**：混合构造 `FastText(**params)` + `build_vocab(corpus_iterable=…)` + `train(corpus_file=…)` 与直接 `FastText(corpus_file=…)` 得到完全相同的词表与字符子词分桶；`wv.vectors` 确实等于原生 fastText 的“词向量与字符 n-gram 均值”；文档池化与原生监督模型的隐藏层池化等价；字符 n-gram 的哈希与分桶和原生**逐字节兼容**。这些是“用 Gensim 复现 fastText”能成立的前提，均已实测通过。

审查同时发现两类问题：

1. **一个静默失效的缺陷**：原生训练的进度 loss 正则永不适配，写出的 CSV 只有表头（`gensim_lab/paper_worker.py:257`）。缺陷当前被后续抽取步骤掩盖，因此不影响现有报告数值。
2. **一个尚未处理、且直接压低原生基准的协议偏离**：8 套分类数据的原生模型使用层次 softmax（`configs/paper_reproduction.yaml:22`），而作者归档脚本与论文 Table 1 用的是普通 softmax。实测该偏离让 ag_news 低约 0.53–0.68 个百分点，把与论文的残差放大约三倍。

此外，第八章报告文字中“本轮原生模型依照论文使用层次 softmax”一句对分类实验不成立，需要更正；两条路线还有若干未披露的不可比因素（L2 归一化、学习率衰减基准、参数量级、OOV 策略、分数语义）。

## 2. 已实测正确的 Gensim 调用

| 调用点 | 核验结论 | 证据 |
| --- | --- | --- |
| `FastText(**params)` + `build_vocab(corpus_iterable=TextCorpus(...))` + `train(corpus_file=..., total_words=model.corpus_total_words)`（`paper_worker.py:292-295`） | 与 `FastText(corpus_file=...)` 的词表、字符子词分桶集合、`corpus_total_words` **完全一致**（12000/12000 tokens） | `artifacts/reviews/gensim_fasttext_api_audit.json` 的 `build_vocab_equivalence` |
| 同上，不使用 `LineSentence` 的理由 | 成立。`scan_vocab` 走 `LineSentence`（`references/sources/gensim-4.4.0/models/word2vec.py:583-584`），其 `str.split()` 会切开 Unicode 空白；训练用的 Cython 文件读取器才与 fastText 的 `readWord` 一致 | 源码比对 + 第 3 节词表交叉验证 |
| `workers` 在 `corpus_file` 模式是否生效 | 生效，每个 epoch 起 `workers` 个训练线程（`word2vec.py:1362-1370`）。探针 workers=4、epochs=2 观察到 8 次线程启动 | 同上 `corpus_file_worker_threads` |
| `model.wv.vectors[index]` 作为词向量（`paper_worker.py:54`） | 等于 `(vectors_vocab + Σngram)/(1+n)`，即原生 fastText 的每词输入向量；与手动计算最大误差 1.5e-8 | 同上 `composite_word_vectors` |
| `document_vector` 的池化（`paper_worker.py:51-69`） | 在真实全量运行上等于“按 token 直接求均值的 L2 归一化结果”，最大误差 2.5e-7；“按类型频次加权”与“按 token 不加权”在代数上等价，因此池化本身与原生隐藏层一致，**唯一多出的是单位化** | `artifacts/reviews/pooling_equivalence.json` |
| 字符 n-gram 哈希兼容性 | Gensim 的 `ft_ngram_hashes` 经 `compute_ngrams_bytes`/`ft_hash_bytes`（`fasttext_inner.pyx:619-641,677`）复刻了 fastText `Dictionary::hash` 的 `int8_t` 符号扩展与 `BOW+word+EOW` 填充；29/29 个词（含中文词）分桶集合与原生 `get_subwords` 完全一致 | `artifacts/reviews/gensim_native_hash_compatibility.json` |
| 检查点续训的学习率 | `paper_worker.py:286` 的插值公式与 Gensim 自身 `_get_next_alpha` 的连续调度**逐点一致**，最大误差 3.5e-18 | `gensim_fasttext_api_audit.json` 的 `alpha_resume_schedule` |
| `model.save(path, separately=[])` + 自定义属性 `paper_epoch_records` | 属性随 pickle 保存并在 `FastText.load` 后仍存在；重载后 `wv.vectors` 与内存中**逐位相同**（误差 0.0） | 同上 `save_load` |
| `models.py` 的 `TfidfModel` 单文档下标、`corpus2csc(...).T`、`Doc2Vec.wv.key_to_index` | 单文档下标返回 `(id, weight)` 列表；`idfs` 为 dict；转置后为 文档×词；Doc2Vec 具备 `wv.key_to_index` | 同上 `models_py_calls` |
| 目标函数消融（`runner.py:139-141`） | `hs` 与 `negative` 互斥设置（hs→negative=0），避免 Gensim 同时训练两个输出头 | 源码核对 |

补充的端到端交叉验证：同一份 `ag_news` 训练文件，Gensim 路线词表 95,810，原生日志 `Number of words: 95811`，相差恰好 1，即原生额外追加的 `</s>`。两条路线的分词口径一致，词表差异只有句末标记。

## 3. 发现的问题


### 3.2 待决策：8 套分类数据的原生模型使用了层次 softmax

`configs/paper_reproduction.yaml:22` 设 `classification.native_loss: hs`，`paper_worker.py:252` 将其传入 `train_supervised`。作者归档脚本 `artifacts/paper_reproduction/classification-results.sh:89-91` 不传 `-loss`，即默认 softmax；论文正文的默认监督目标也是 softmax，层次 softmax 是在 `2.1` 节作为**大标签空间**的加速选项引入的（标签预测 312,116 类才使用）。因此该设置对 8 套分类数据（2–14 类）是协议偏离，而当前报告第 13 行的表述把它说成论文做法，缺少依据。

在被测数据上做受控对照（同一份已准备数据、`dim=10`、`epoch=5`、`lr=0.25`、`minn=maxn=0`、`bucket=10000000`、`thread=20`、种子 42/2026/7）：

| 目标函数 | wordNgrams | 本轮 Accuracy (%) | 论文 (%) | 与论文差 |
| --- | --- | --- | --- | --- |
| softmax（作者脚本） | 1 | 91.24 ± 0.12 | 91.5 | −0.26 |
| softmax（作者脚本） | 2 | 92.29 ± 0.11 | 92.5 | −0.21 |
| hs（当前配置） | 1 | 90.66 ± 0.05 | 91.5 | −0.84 |
| hs（当前配置） | 2 | 91.67 ± 0.10 | 92.5 | −0.83 |

逐种子明细见 [native_loss_ablation.csv](data/native_loss_ablation.csv) 与 `artifacts/reviews/native_loss_ablation.json`。结论：改用 softmax 后，ag_news 与论文的残差从约 0.85 降到约 0.25 个百分点，属于可接受的历史实现差异范围。

### 3.3 20 线程的抖动不小于种子效应

同一配置（softmax/ngram=1/seed=42）重复 4 次得到 91.605、91.276、91.355、91.368，极差 0.329、标准差 0.142。这与三种子标准差（0.12）同量级甚至更大，说明当前报告里“三种子样本标准差衡量训练波动”会**低估**总波动；在 ag_news 上小于约 0.3 个百分点的差异不宜解读。AGENTS.md 已批准的“多线程非逐位确定性”需要在数值解释上再明确这一量级。

### 3.4 报告文字与代码不符之处

- 第 13 行“本轮原生模型依照论文使用层次 softmax”：对分类实验不成立（见 3.2）；对 YFCC 标签实验成立。
- 第 7 行只写了“平均文章向量”，未披露 Gensim 路线**额外做了 L2 归一化**；这一步使下游 LinearSVC 变成余弦间隔分类器，是真实的几何差异（`paper_worker.py:67-69`，实测见 `pooling_equivalence.json`）。
- 未披露的其它不可比因素：Gensim 的 `alpha=0.025→1e-4` 按 epoch 衰减，原生按数据集取 0.05–0.5 并按 token 衰减；输入参数量级相差约 50 倍（原生 bigram 桶 10M×10 维 vs Gensim 桶 200k×10 维）；原生 `minn=maxn=0` 使 OOV 词被直接忽略，Gensim 侧可用字符子词生成向量；Gensim 分类分数是 SVM 间隔而非概率（报告只说明了标签任务的 log_score）。
- `paper_worker.py:99` 的 `oov_tokens/oov_rate` 统计的是“不在词表中的 token”，即使该 token 已由子词生成向量也计入，且这里不输出 `oov_vector_coverage`（主章节的 `models.py:124` 才有）。字段名容易被读成“无法表示”，建议改名或补一列。
- `run.log` 以追加方式打开、整文件重新解析（`paper_reproduction.py:85`、`paper_worker.py:256`），重试同一运行目录会写入重复进度行。

### 3.5 检查点与模型体积（已核算，非阻塞）

`model.save(path, separately=[])` 只写入 `vectors_ngrams`、`vectors_vocab`、`syn1neg` 等实际参数；派生矩阵 `wv.vectors` 被 `FastTextKeyedVectors._save_specials` 排除，并在 `load` 时由 `adjust_vectors()` 重算（实测保存文件的 `wv.__dict__` 不含 `vectors`，重载后数值逐位一致），因此每个 epoch 的 `embedding_checkpoint.gensim` 不额外多存一份副本。

按配置推算：YFCC 标签实验 200 维、`bucket=200000`、词表约 297k 时，单个 Gensim 模型约 0.64 GB，5 个轮次检查点加最终模型约 3.8 GB/运行（6 个 Gensim 运行）；原生 `bucket=10000000`、200 维的 `native.bin` 约 8.5 GB/运行。加上 `runtime_volume_estimate.json` 已记录的 200 维文档特征矩阵 73 GB、50 维 18 GB 与原始数据 32.5 GB，总量在 C 盘剩余约 723 GB 之内，可以完成；但应预留空间并保持顺序执行。

## 4. Gensim 复现与原生实现的对比现状

两条路线的差距可以分解为三块，ag_news 单标签 Accuracy：

| 环节 | 变化量 | 说明 |
| --- | --- | --- |
| 两阶段近似本身 | 约 10.5 个百分点 | softmax/单字原生 91.24 vs Gensim 两阶段 80.78 ± 0.29（三种子，见[复现报告](论文实验复现.md)） |
| 词 bigram | 约 +1.05 | 原生 wordNgrams 1→2；Gensim 路线无对应特征 |
| 目标函数 | 约 +0.58 | hs→softmax |

即：**Gensim 侧约 10 个百分点的落后主要来自方法本身**（冻结的无监督词向量 + 外部线性分类器，输入表示不随标签损失更新），而不是分词、词表或子词实现差异——后三者已被本报告第 2 节证明是对齐的。

人民日报章节是另一种情形：Gensim FastText（0.4127）远高于当时未调参的原生实现（0.0202），但补充网格把原生提到 0.4656 ± 0.0020。该章原有的原生基线存在拟合不足，工程已在[补充实验与改进报告](补充实验与改进报告.md)中如实记录，不涉及本次发现的问题。

## 5. 建议（本次均未实施）

1. **修复 3.1 的正则**，让运行目录内的 loss CSV 直接可用，并让 `tools/extract_paper_loss.py` 继续作为最终产物来源（修复后两条路径口径一致）。
2. **考虑把 `classification.native_loss` 改为 `softmax`**，与归档脚本和论文 Table 1 对齐。注意 `configs/paper_reproduction.yaml` 的哈希进入每个 spec，改动会让当前 14/90 个已完成运行全部变为“非当前”，需要重跑；但已完成的原生运行只在 ag_news 与 sogou_news 上（约 10 个，单个 1–20 秒），加上 Gensim 侧的 ag_news 3 个（各约 45 秒）与 sogou_news 1 个（约 25 分钟），总重算约半小时。若不改，应把该偏离作为协议差异写入报告，并用 3.2 的对照数值说明影响量级；无论如何，报告第 13 行“依照论文使用层次 softmax”的表述对分类实验需要更正。
3. **在报告中补齐 3.4 列出的未披露项**，特别是 L2 归一化与 20 线程抖动量级（3.3）。
4. YFCC 标签实验的 `hs` 设置符合论文，不需要调整。

以上建议都只是本次核验的结论，未对工程做任何改动。

## 6. 复现命令

```powershell
conda run -n Gensim python tools/audit_gensim_fasttext_api.py
conda run -n Gensim python tools/audit_gensim_native_hashes.py
conda run -n Gensim python tools/audit_pooling_equivalence.py
conda run -n Gensim python tools/audit_native_loss_ablation.py
```
