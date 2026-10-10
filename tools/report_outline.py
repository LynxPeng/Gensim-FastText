"""Render the six report chapters requested by the user from saved evidence."""
from __future__ import annotations

import re
from report_paper import paper_review_body, paper_reproduction_body

from report_sections import (classification_summary, data_summary, error_summary,
                             records, section, source, table, tables, word_summary)


def anchored(content, anchor):
    return content.split(f'<a id="{anchor}"></a>', 1)[1].split('<a id="', 1)[0]


def thucnews_body():
    """Include current independent-corpus evidence without changing earlier chapters."""
    text = source("THUCNews词向量实验.md")
    text = re.sub(r"(?m)^# .+\n", "", text, count=1)
    text = text.replace('<a id="thucnews-word-learning"></a>', "")
    return re.sub(r"(?m)^## ", "#### ", text).strip()


def render_report(jobs, exports, baseline):
    chunks = []

    def add(text):
        chunks.append(text.strip())

    def chapter(title, anchor):
        add(f'<a id="{anchor}"></a>\n\n## {title}')

    def subsection(number, title, anchor):
        level = "#" * (number.count(".") + 2)
        add(f'<a id="{anchor}"></a>\n\n{level} {number} {title}')

    def figure(name, caption, *, word=True):
        path = f"figures/word_evaluation/{name}.png" if word else f"figures/{name}.png"
        add(f"![{caption}]({path})\n\n{caption}")

    add("# FastText 论文与实现分析及 Gensim 文本向量化表示学习实验报告")
    chapter("摘要", "abstract")
    add("本文围绕上手 Gensim 与理解 FastText 两项目标，以 Bag of Tricks for Efficient Text Classification 为第一章的解读对象，整理论文要点、核心源码、"
        "文本向量化实验、实现差异及参数对照。已执行实验使用人民日报 1998 年一月至六月文本："
        "清洗后保留 18,108 篇，一月 3,050 篇按近重复组隔离后划分为 2,138/456/456 篇训练、"
        "验证和测试文章。类别来自版面编号 01–12，不是人工标注的主题。五种 Gensim 表示均"
        "连接统一的 LinearSVC，另以多数类和原生监督 FastText 为参照；表示与分类器仅拟合训练集，"
        "选参仅使用验证集，人民日报历史实验采用三个种子和单线程；当前默认使用 20 个训练线程。")
    add("一月 12 类主配置中，TF-IDF 测试 Macro-F1 为 0.4652，Gensim FastText 为 "
        "0.4127 ± 0.0122；WordSim-240 上 Word2Vec 与 Gensim FastText 的 Spearman rho 分别为 "
        "0.3207 ± 0.0090 和 0.4610 ± 0.0101，显示词相似度与版面分类排序不同。补充优化后的"
        "原生 FastText 达到 0.4656 ± 0.0020，但日期成对抽样的差值区间包含零，不能确认其稳定"
        "优于 TF-IDF。源码与数值核验说明 HS score 的解释限制。")
    add("第二部分使用原论文八套分类数据和完整 YFCC100M 文本数据，开展原生监督 FastText 与 "
        "Gensim 两阶段近似路线的对照。全量进度和已完成结果单独列出，未完成的配置不作为最终结论。逐运行、逐类、逐关系、逐轮和"
        "逐篇明细保存在 CSV，正文展示支撑结论的汇总与趋势。")
    add("另以清华 THUCNews 全量语料开展独立词向量实验，遍历 836,075 篇，清洗后保留原有 "
        "5,000 篇留出文章，其余训练。重新训练 TF-IDF、Word2Vec、FastText，统一做最近类别中心诊断；"
        "词向量另与此前 5 万篇结果对照，训练使用 20 个线程，"
        "实际完成数量和结果列在 4.9 节。"
        "此前原论文全量复现已经按用户要求停止。")
    add("关键词：FastText；Gensim；子词表示；文本向量化；版面分类；参数比较")

    chapter("一、FastText论文解读", "paper-reading")
    add(paper_review_body())

    chapter("二、FastText复现", "paper-reproduction")
    add(paper_reproduction_body())

    chapter("三、FastText核心代码阅读笔记", "code-reading")
    subsection("3.1", "阅读对象与调用入口", "source-entry")
    add("阅读以归档 fastText v0.9.2 的 C++ 核心为主，并与实际 Python 调用和 Gensim 对照。"
        "[Python train_supervised](../references/sources/fastText-0.9.2/python/fasttext_module/fasttext/FastText.py) "
        "将参数传入 [pybind](../references/sources/fastText-0.9.2/python/fasttext_module/fasttext/pybind/fasttext_pybind.cc)，"
        "绑定释放 GIL 后调用 C++ 训练。核心路径为词典读入与特征构造、模型创建、trainThread、"
        "supervised/skipgram/cbow、Model::update、输出损失及参数更新。上游阅读副本不等于对第三方 "
        "Windows wheel 的逐字节源码认证。")

    subsection("3.2", "字符子词与兼容哈希", "subword-code")
    add("[Dictionary::computeSubwords](../references/sources/fastText-0.9.2/src/dictionary.cc) "
        "从带词首、词尾标记的字符串提取片段。代码跳过 UTF-8 后续字节，并在增加片段长度时吸收"
        "完整字符，因此中文按字符边界计数；只包含边界标记的一元片段排除。训练词保留整词参数行，"
        "字符片段再映射到桶，体现子词词表示研究中的参数共享。")
    add("Dictionary::hash 先把每个字节转为 int8，再异或到 uint32 并乘以 16777619，"
        "按 32 位溢出。这个有符号字节规则用于兼容历史模型，不能直接用 Python hash() 或另一种"
        "无符号字节实现代替。中文、ASCII 和新词的子词/哈希探针均与独立复算一致，完整列表移入 CSV。")

    subsection("3.3", "词组合特征与隐藏表示", "feature-code")
    add("Dictionary::addWordNgrams 对相邻词的哈希递推，再映射到桶。它对应分类论文的词组合特征；"
        "关闭字符子词并不关闭词 bigram。未知词仍可能参与词组合哈希，因此单词 OOV 向量覆盖"
        "不能代替整个监督输入的覆盖。Python 参数入口在 wordNgrams≤1 且 maxn=0 时把实际 bucket "
        "设为 0，解释了 unigram 模型体积显著小于 bigram 的现象。")
    add("[Model::computeHidden](../references/sources/fastText-0.9.2/src/model.cc) 的核心代码如下，"
        "先累计输入特征所在行，再按特征数平均；输入为空时 Model::update 会提前返回。")
    add("""```cpp
hidden.zero();
for (auto it = input.cbegin(); it != input.cend(); ++it) {
  hidden.addRow(*wi_, *it);
}
hidden.mul(1.0 / input.size());
```""")
    add("Model::update 随后调用损失的 forward，获得输出参数更新及隐藏梯度，再更新输入矩阵各行；"
        "normalizeGradient_ 启用时还按输入特征数缩放梯度。这说明监督路线的表示参数会受到标签反馈。")

    subsection("3.4", "监督目标与无监督目标的分派", "objective-code")
    add("[FastText::createLoss/supervised](../references/sources/fastText-0.9.2/src/fasttext.cc) "
        "选择负采样、层次 softmax、普通 softmax 等输出目标。supervised 在没有标签或特征时返回；"
        "除 OVA 分支外，从文章标签中选目标并调用 Model::update。当前单版面文章只有一个标签，"
        "使用 softmax。无监督 skipgram/cbow 则预测词上下文，输出目标不是版面标签。")
    add("监督模式下 Dictionary::discard 返回 false，无监督的高频下采样不能直接套用到监督分类。"
        "NS/HS 改变输出事件和运算路径；Gensim 的 HS 对照明确设 hs=1、negative=0，"
        "避免同时训练两个目标。对应的分类与内在评价在第四、六部分分别报告。")

    subsection("3.5", "学习率、线程、矩阵与模型保存", "training-code")
    add("FastText::trainThread 按 `tokenCount/(epoch×ntokens)` 计算进度，学习率为 "
        "`lr×(1−progress)`；局部 token 计数超过 lrUpdateRate 才上报共享进度。训练根据模型类型"
        "调用 supervised、cbow 或 skipgram，多个线程共享输入/输出矩阵，更新顺序会变化。"
        "本节历史结果使用 thread=1；当前运行默认 thread=20，历史数据不改写为多线程结果。")
    add("[DenseMatrix](../references/sources/fastText-0.9.2/src/densematrix.cc) 按连续行存储参数。"
        "输入矩阵通常为 `(词数+桶数)×维度`，普通 softmax 输出矩阵为 `类别数×维度`。"
        "saveModel/loadModel 保留字典、参数和矩阵；只保存词向量不足以恢复分类输出头。"
        "原生小语料训练 P@1=1，保存加载预测一致，支持分类链路正确运行，但不能据此推断正式泛化质量。")

    subsection("3.6", "运行探针与工程核验", "engineering")
    add("已完成子词/哈希、OOV 聚合、最低词频、高频下采样、可重复迭代、真实任务与学习率、"
        "数组类型和保存加载探针。关闭下采样与 sample=0.001 的小语料有效训练词数分别为 "
        "6,000 和 501，说明下采样改变训练事件而不从分类文章中删除功能词。Gensim 已登录词"
        "聚合最大复算误差为 1.86e-09，OOV 聚合最大误差为 0；可生成的 OOV 向量未必具有正确语义。")
    add("数据深度核查独立重算清洗链路、词典文档频率、IDF、预测标签和各项指标。对 "
        "163,940,778 对文章的 SimHash 全对检查与生产索引得到同样的 3 对近重复；这只验证"
        "指定距离阈值，不证明所有语义重复均已识别。323 个完整预测分区重载最大分数差为 0；"
        "一月 12 类 seed 42 的七种方法独立重训后，完整验证预测 CSV 字节一致。"
        "这些结果限定于当前机器、固定依赖及单线程，不保证跨平台逐位一致。")
    add("缓存核对模型、输入、分区、预测与 SHA-256，异常缓存保留后重建；操作系统文件锁保护写入。"
        "公开预测检查包括空文本、标点、分词空白、全 OOV 和旧协议拒绝。Doc2Vec 原推断依赖文章编号，"
        "修正前 20 篇探针中有 1 篇标签改变；当前由 token 序列 SHA-256 重置推断种子，全 OOV 文章"
        "置零，并重训受影响的主矩阵、冒烟和日期隔离运行。当前适配器见"
        "[content_seed.py](../gensim_lab/content_seed.py)。")
    add("核验资源：[探针](../artifacts/evidence/probes.json)、"
        "[独立重训](../artifacts/reproducibility/checks.json)、[深度核查](../artifacts/deep_audit.json)、"
        "[预测入口检查](../artifacts/cli_prediction_checks.json)。工程测试是不同日期的保存快照："
        "2026-10-07 审查记录 43 项测试和 51 项技术验收；2026-10-08 代码基线验证 49 项测试。"
        "技术通过不代替独立基本语料要求，也不代表原论文复现完成。")

    chapter("四、基于Gensim文本向量化表示学习的各种实验", "vectorization")
    subsection("4.1", "已执行语料、预处理与评价协议", "data-quality")
    add("现有表示和分类实验均读取人民日报 1998 年 01–06 月，沿用原文分词，去除编号、词性和"
        "实体括号，执行 NFKC、控制字符及标点清理，保留标题和正文，不额外删除停用词。"
        "文章编号只用于溯源及提取版面标签。4.1–4.8 节为人民日报实验；"
        "新增 THUCNews 独立词向量实验列于 4.9 节，两套数据分别报告。")
    add(data_summary())
    add("少于 30 个有效词的文章排除；异标签精确重复全部排除，同标签精确重复保留最早文章。"
        "539 篇损失中短文 492 篇、异标签重复 45 篇、同标签重复 2 篇。不同内容的重复段落编号按"
        "物理行顺序保留，控制字符清除但不猜测补字。字符三元组 64 位 SimHash、汉明距离≤3 "
        "形成近重复组。一月用 seed 42 的 20 折 StratifiedGroupKFold 合并为 14/3/3 折；"
        "先划分 12 类，再筛选 2/4/8/12 类，分区组不交叉。")
    add("五种表示均接 `LinearSVC(class_weight='balanced', dual='auto', max_iter=20000)`，"
        "C 在验证集从 0.1/1/10 选择。主指标为各类 F1 的平均 Macro-F1，分母为零时置零，另存 "
        "Accuracy、逐类和加权 PRF、混淆计数。42/2026/7 三种子的均值 ± 样本 SD 描述训练波动，"
        "不等于置信区间；TF-IDF 与多数类为确定性单次基线，LSI 列入随机模型。")

    subsection("4.2", "表示训练与文章向量构造", "representations")
    add(table(["方法", "本项目训练或拟合", "文章向量"], [
        ["TF-IDF", "训练词典、文档频率与 log2(N/df) IDF", "稀疏词袋与 L2 归一化"],
        ["LSI", "训练 TF-IDF 矩阵的随机截断低秩投影", "至多 100 维与 L2 归一化"],
        ["Word2Vec", "主配置 Skip-gram 负采样，保持句子/段落边界", "token 频次均值，跳过 OOV，再 L2 归一化"],
        ["Doc2Vec", "DBOW，dm=0、默认 dbow_words=0", "所有分区内容种子推断 30 轮；全 OOV 置零"],
        ["Gensim FastText", "Skip-gram，整词与字符子词共享上下文目标", "组合词向量均值或训练 IDF 加权，再 L2 归一化"],
    ]))
    add("文章聚合为 `z(d)=Σ w(t,d)v(t)/Σ w(t,d)`，主配置权重为 token 频次；加权对照再乘训练 IDF，"
        "词典外可生成向量的词使用训练最大 IDF。无有效词向量时输出零向量。词向量上下文不跨文章，"
        "超长片段最多 1,000 词；SentenceCorpus 每轮可重新遍历。Doc2Vec 默认未联合训练词向量，"
        "不把其词向量作为有效词语义模型。代码入口见 [models.py](../gensim_lab/models.py)，"
        "当前 Doc2Vec 运行另使用第三部分所述内容种子适配器。")

    intrinsic = word_summary(jobs)
    subsection("4.3", "中文词相似度、类比与覆盖率", "word-similarity")
    add("人工评价不参与表示训练、词表扩充或分类选参。WordSim-240 为 240 词对；CWE 转存的 "
        "297 词对原样评价，并单列按上游说明去除 OPEC/石油后的 296 版本。CA-translated 为 "
        "953 题，CA8 形态/语义为 10,177/7,636 题。分母依据归档实际文件，不将中文基准称为原 "
        "FastText 论文的评测集。主配置两方法各三个种子的结果如下。")
    add(table(*tables(intrinsic)[0]))
    add("FastText 在两个主要词相似度基准上的相关高于 Word2Vec，但覆盖率相同，词相似度优势"
        "不能直接推出版面分类优势。Pearson/Spearman 及双侧 p 值用 Gensim evaluate_word_pairs，"
        "严格词表、大小写敏感、restrict_vocab=len(wv)，OOV 词对不参与相关。p 值未经多重比较校正，"
        "不用于选参；共同词表与 FastText 可生成 OOV 口径分别保存，不混入严格 IV 指标。")
    figure("similarity_main", "主配置 Word2Vec 与 Gensim FastText 的词相似度；点为三个种子，短线为均值，未给推断区间。")
    subsection("4.3.1", "类比条件准确率与总体结果", "analogy-results")
    add(table(*tables(section(intrinsic, "词类比的覆盖与条件准确率"))[0]))
    add("3CosAdd 使用 evaluate_word_analogies，四词均在训练词表才计覆盖，dummy4unknown=True "
        "把未覆盖题计错。FastText 形态题覆盖内准确率为 86.51%，但覆盖仅 7.11%，全部题准确率"
        "为 6.15%；高条件分数不能推广到整套基准，语义题表现也较弱。")
    figure("analogy_main", "类比方法差异；条件准确率与全部题准确率采用不同分母，不能互相替代。")
    figure("class_scale_intrinsic", "词相似度与覆盖率随训练类别范围变化；类别范围同时改变文章与词表，不是类别数量的独立因果效应。")
    add("词评价包含 99 条模型记录、594 条外部指标与 1,089 行逐关系结果；部分历史控制共享权重，"
        "不能相加为独立重复。资源来自固定 CWE/Chinese-Word-Vectors 归档，仓库分别声明 "
        "MIT/Apache-2.0，原始人工评分归属不因此改变；见[来源与校验清单](../references/word_benchmarks/manifest.json)。")

    classified = classification_summary()
    subsection("4.4", "基于文章向量的分类小实验", "main-results")
    add(table(*tables(section(classified, "方法与类别规模的比较"))[0]))
    add("表中为测试 Macro-F1。Word2Vec/Doc2Vec 在 2 类时均值较高；主配置 4、8、12 类中 "
        "TF-IDF 较高。一月 12 类 TF-IDF Accuracy=0.4803、Macro-F1=0.4652；FastText 的子词"
        "可生成覆盖达到 1，没有转化为更高分类 F1。缩减类别同时改变文章和类别分布，不能把变化"
        "全部归因于类别数。原生监督 FastText 的初始低分与后续优化分别保留，避免把初始预算下"
        "的拟合不足当成算法结论。")
    figure("main_macro_f1", "四档类别范围的测试 Macro-F1；随机模型为三种子均值与样本标准差。", word=False)

    subsection("4.5", "补充实验：冻结模型的跨月泛化", "temporal-results")
    add(table(*tables(section(classified, "冻结模型的跨月泛化"))[0]))
    add("二月至六月只评估冻结的一月模型，不重新拟合词典、IDF、表示或分类器，排除与一月任意"
        "分区同组的后续文章。原生路线为最初验证选定模型。TF-IDF 多数月份较高；月份波动同时"
        "包含事件、版面及类别分布变化，不能解释为时间流逝的单一效应。")
    figure("temporal_macro_f1", "一月冻结模型的后续月份分类结果。", word=False)

    subsection("4.6", "补充实验：扩大训练的共同测试文章对照", "scale-results")
    add("扩展训练为一月至四月 11,802 篇，五月验证 3,115 篇，六月测试 3,191 篇；后序分区排除"
        "与前序同组的文章。沿用一月表示参数，Gensim 路线只在五月重选 SVM C。下表在相同六月"
        "文章上成对比较，差值为扩展减冻结。")
    add(table(*tables(section(classified, "扩大训练的六月共同文章对照"))[0]))
    add("三条路线均改善，原生路线改善最大，扩展 TF-IDF 仍较高。训练文章、词表、事件和时间范围"
        "同时增加，规模效应不能与内容效应完全分离；相同 epoch 也不是相同训练事件预算。")

    subsection("4.7", "补充实验：日期隔离、置乱与不确定性", "robustness")
    supplemental = source("补充实验与改进报告.md")
    add("以下均为原一月及六月测试结果已经观察后的探索，不是新盲测。日期隔离使用 1–21 日训练 "
        "2,091 篇、22–26 日验证 652 篇、27–31 日测试 307 篇。主参数七方法全部完成；TF-IDF "
        "测试 F1=0.3342、Gensim FastText=0.3240 ± 0.0015。迁移改进原生参数后为 "
        "0.2666 ± 0.0050；其参数来自包含后期日期的一月随机验证集，属于敏感性分析，不是完全前瞻评估。")
    permutation = tables(section(supplemental, "标签置乱的负对照"))[0]
    add(table(permutation[0][:-1], [row[:-1] for row in permutation[1]]))
    add("仅置乱训练标签，seed=12345，文本、样本数及验证/测试真值不变。SVM 仍按同一 C 网格在"
        "验证集选择，没有锁定所有分类器参数。单次置乱用于负对照，不能当成正式置乱显著性检验"
        "或所有泄漏均不存在的证明。")
    interval_tables = tables(section(supplemental, "按日期成对抽样的不确定性"))
    combined = []
    blocks = interval_tables[2][1]
    for month, (_, rows) in zip(["一月", "六月"], interval_tables[:2], strict=True):
        for row in rows:
            if row[0] == "TF-IDF":
                continue
            matched = next(r for r in blocks if r[0] == month and r[1] == row[0])
            combined.append([month, row[0], row[3], row[4], matched[3]])
    add(table(["月份", "相对 TF-IDF 的方法", "F1点差值", "日期成对分位区间", "七日期块分位区间"], combined))
    add("两种抽样均为 2,000 次，以整日或连续 7 个已观察日期的循环块为单位，方法共享权重。"
        "模型种子未重采样，区间为固定模型下的 2.5%/97.5% 描述性分位数，未作多重比较校正。"
        "补充原生模型的差值区间包含零，不能确认稳定优势，也不能证明等效。旧原生六月结果"
        "在不同抽样定义下的方向判断不同，进一步提示日期相关性影响解释。")

    subsection("4.8", "拟合程度与错误案例", "errors")
    errors = error_summary()
    add("以下采用最初验证选定配置的 seed 42，训练指标只诊断拟合，不用于再次选参。")
    add(table(*tables(errors)[0]))
    add("TF-IDF 训练与测试差距明显；早期原生模型在训练集上也低，支持拟合不足的判断。改进配置"
        "在日期隔离训练集上约 0.997，却仅有约 0.267 的测试 F1，说明增加优化预算未消除日期泛化差距。")
    add(table(*tables(errors)[1]))
    add("这些文章的主题与其版面位置并非一一对应，语义相近文章可能跨版，同版也可能包含不同主题。"
        "样例说明标签定义，不证明误分类原因。完整逐篇预测、PRF、混淆及其他错误例留在 CSV。")

    subsection("4.9", "THUCNews 三种表示学习", "thucnews-word-learning")
    add(thucnews_body())

    subsection("4.10", "THUCNews 训练方式消融与三种子对比", "thucnews-training-ablation")
    ablation = source("THUCNews训练方式消融实验.md")
    ablation = re.sub(r"(?m)^# .+\n", "", ablation, count=1)
    ablation = ablation.replace('<a id="thucnews-training-ablation"></a>', "")
    ablation = ablation.replace('(THUCNews词向量实验.md#数据与预处理)', '(#thucnews-word-learning)')
    add(re.sub(r"(?m)^## ", "#### ", ablation).strip())

    subsection("4.11", "THUCNews 小规模训练方式与超参数搜索", "thucnews-small-search")
    small_search = source("THUCNews小规模超参数搜索.md")
    small_search = re.sub(r"(?m)^# .+\n", "", small_search, count=1)
    small_search = small_search.replace('<a id="thucnews-small-search"></a>', "")
    small_search = small_search.replace('(THUCNews训练方式消融实验.md)', '(#thucnews-training-ablation)')
    small_search = small_search.replace('(THUCNews学习率三档历史记录.md)', '(#thucnews-small-search)')
    add(re.sub(r"(?m)^## ", "#### ", small_search).strip())

    subsection("4.12", "THUCNews 向量维数搜索", "thucnews-dimensions")
    dimensions = source("THUCNews向量维数搜索.md")
    dimensions = re.sub(r"(?m)^# .+\n", "", dimensions, count=1)
    dimensions = dimensions.replace('<a id="thucnews-dimensions"></a>', "")
    dimensions = dimensions.replace('(../docs/THUCNews向量维数搜索方案.md)', '(#thucnews-dimensions)')
    add(re.sub(r"(?m)^## ", "#### ", dimensions).strip())

    chapter("五、对比Gensim的实现方法", "implementation-comparison")
    subsection("5.1", "五种 Gensim 表示的实现差异", "gensim-methods")
    add(table(["方法", "核心实现", "输入与词表", "适用及限制"], [
        ["TF-IDF", "Dictionary/TfidfModel", "文档频率过滤、稀疏词袋", "保留栏目关键词；不直接学习词语义"],
        ["LSI", "LsiModel 随机低秩投影", "训练 TF-IDF 矩阵", "压缩共现结构；投影秩受词表与样本数限制"],
        ["Word2Vec", "Word2Vec + word2vec_inner Cython", "token 频次词表、句子上下文", "上下文词表示；跳过 OOV，文章聚合丢失顺序"],
        ["Doc2Vec", "Doc2Vec + doc2vec_inner DBOW", "TaggedDocument，推断需已登录词", "文章表示；需固定内容种子与统一推断协议"],
        ["Gensim FastText", "FastText + fasttext_inner Cython", "整词表、字符子词哈希桶", "共享片段及 OOV 组合；桶体积大，可生成不等于语义正确"],
    ]))
    add("TF-IDF no_below 统计文档频次，Word2Vec/FastText min_count 统计 token 频次，两套词表"
        "不完全相同。文档频率过滤、训练上下文目标与监督目标的差异都是实现条件，不能把整个流水线"
        "的分数差当成某个单一机制的因果效应。训练集词典、IDF 和表示隔离已由独立核查确认。")

    subsection("5.2", "Gensim FastText 与原生监督 FastText", "fasttext-comparison")
    add(table(*tables(section(source("论文阅读与源码分析.md"), "两篇论文的任务不同"))[0]))
    add("Gensim 训练更新 vectors_vocab 和 vectors_ngrams，adjust_vectors 在查询前组合整词与子词。"
        "已登录词取整词行加子词行的平均，OOV 只平均片段桶，重复桶可以重复贡献。判断训练词表应"
        "查询 key_to_index；启用子词时 `word in wv` 可对 OOV 返回真。关闭子词后 OOV 查询抛 "
        "KeyError，文章聚合器跳过；未训练桶仍有初始化值，不能以非零向量声称语义已学会。")
    add("Gensim 默认负采样频次指数为 0.75，与子词词表示论文的 0.5 不同；训练更新按队列任务调度，"
        "通过 _do_train_job 调用 train_batch_any，在 nogil 区域执行。batch_words 是任务大小，"
        "不等于平均梯度 mini-batch。原生监督直接通过标签更新输入/输出矩阵，当前无额外类别重加权；"
        "Gensim 表示连接 balanced SVM。因此相同轮数不代表相同预算，二者比较的是完整路线。")
    add("Gensim 模型及词典可原生保存，完整分类另需 Bundle 中的 IDF/SVM；原生监督用 native.bin。"
        "Gensim 的子词向量 API 不提供等价监督输出头，没有用它载入监督分类 bin 进行相同预测。")

    subsection("5.3", "句子上下文评分接口及数值限制", "numeric-diagnostics")
    add("Word2Vec 的 score 只支持相应 HS 配置，逐句分数为局部上下文对数项之和。内核跳过 OOV，"
        "移除后重建窗口，使用 log-sigmoid 查表并跳过 |logit|≥6 的饱和节点。固定审计句的稳定"
        "复算差在 SG 可达 78.1978、CBOW 测试可达 6.9706，近似误差需独立解释。不能取 exp 后"
        "称为规范句子概率，也不能直接称为语言模型困惑度。")
    figure("sentence_scores", "HS 句子 score 的长度依赖、按 IV 上下文事件归一化分布与 300 句近似误差核查。")
    figure("fasttext_hs_scores", "FastText-HS 继承的 score 使用组合词向量与 HS 输出头，仍跳过 OOV；属于接口诊断，不按分数高低判断整体模型质量。")
    add("每个 HS 模型有 46,779 句，训练/验证/测试为 32,593/7,096/7,090；IV 少于两词记为无有效"
        "上下文。421,011 条评分含重复控制，不能当成相同数量独立样本。默认 Doc2Vec DBOW 未训练词向量，"
        "原生监督/TF-IDF/LSI/多数类也不作为本轮无监督词评价模型。")

    subsection("5.4", "资源成本与计算后端", "resource-comparison")
    add(table(*tables(section(classified, "十二类实验的资源代价"))[0]))
    add("本表为一月 12 类主配置，拟合秒包括表示训练和 SVM C 搜索；峰值 RSS 每 50 ms 采样，取"
        "种子最大值；模型大小为可运行 Bundle/native.bin，不重复计入相同内容的原生 Gensim 导出。"
        "TF-IDF 在本设置下兼具低成本与较好分类表现，FastText 子词桶增加存储，未带来分类优势。"
        "一次本机计时没有性能置信区间。")
    add("本机 RTX 5070 被识别，但当前固定 Gensim、原生 FastText、LinearSVC 路径使用 CPU，"
        "未执行 CUDA 训练或 GPU 基准。向量聚合等外围计算可另行改写，现有结果没有支持其加速收益。"
        "判断结合实际数组/BLAS/C++ 路径与接口检查，不能只根据缺少 device 参数推断。"
        "记录见 [CUDA 核查](../artifacts/cuda_audit.json)。")

    chapter("六、各类参数，超参数尝试比较", "parameters")
    subsection("6.1", "主配置与控制条件", "main-parameters")
    add(table(["参数组", "已执行主配置"], [
        ["训练与随机性", "split_seed=42；模型 seeds=42/2026/7；classes=2/4/8/12；历史单线程"],
        ["Dictionary", "no_below=3 文档；no_above=0.9；keep_n=50,000"],
        ["通用词表示", "vector_size=100；window=5；min_count=3 token；epochs=15；sample=0.001"],
        ["优化目标", "sg=1；negative=5；hs=0；alpha=0.025；min_alpha=0.0001；batch_words=10,000"],
        ["FastText", "字符 min_n=1/max_n=3；bucket=200,000；文章 pooling=mean"],
        ["Doc2Vec", "dm=0/dbow_words=0；infer epochs=30/alpha=0.025/min_alpha=0.0001；tokens_sha256_v1"],
        ["原生监督", "dim=100；epoch=25；lr=0.1；minCount=3；wordNgrams=2；字符关闭；softmax；bucket=200,000"],
        ["分类器", "LinearSVC C=0.1/1/10；balanced；dual=auto；max_iter=20,000"],
    ]))
    add("Gensim FastText 单因素在一月 12 类验证集、seed 42 上每次改变一项，正式模型选择只使用"
        "历史单线程候选。现行训练使用多线程，原速度对照保留。各组基线可共享缓存，不算独立重复；最优候选采用"
        "实际测得的完整配置，不拼接不同轴的单独最优值。")

    subsection("6.2", "FastText 单因素的完整比较范围", "fasttext-factors")
    parameter_text = source("参数分析.md")
    labels = {"vector_size": "维度", "window": "上下文窗口", "min_count": "最低词频", "epochs": "轮数",
              "sg": "CBOW/SG", "subwords": "字符子词范围", "bucket": "哈希桶数", "objective": "训练目标",
              "alpha": "初始学习率", "sample": "高频下采样", "pooling": "文章聚合",
              "native_word_ngrams": "原生词组合", "workers": "CPU线程（速度）", "batch_words": "任务大小"}
    factor_rows = []
    for name, label in labels.items():
        values = records(section(parameter_text, name))
        best = max(values, key=lambda row: float(row["验证Macro-F1"]))
        scores = [float(row["验证Macro-F1"]) for row in values]
        factor_rows.append([label, " / ".join(row["值"] for row in values),
                            f"{min(scores):.4f}–{max(scores):.4f}", best["值"]])
    add(table(["因素", "实际测试取值", "验证 Macro-F1 范围", "本组最高分取值"], factor_rows))
    add("negative=10 的验证 F1=0.4760，是历史单线程候选的实际最优配置。维度、轮数、架构及"
        "学习率显示局部敏感性，但不能外推为参数越大越好。窗口、最低词频、子词长度、下采样、"
        "文章聚合与任务大小的各组极差不超过 0.0091，不足以证明设置等效。最低词频从 1 增到 "
        "5 时 token OOV 从 0.0477 增至 0.1089；覆盖与分类表现需要分别解释。")

    subsection("6.3", "Word2Vec/FastText 参数与内在评价", "intrinsic-factors")
    add("Word2Vec 另完成维度、窗口、最低词频、轮数、SG/CBOW、目标、alpha、sample、"
        "batch_words 九轴单因素；适用取值沿用上表设计。已有 FastText 模型均接受相同中文词评价。"
        "每档只有一次训练，这些补充结果发生在原测试已观察之后，不重新选择主分类模型。")
    for name, caption in [
        ("word2vec_changed_similarity", "Word2Vec 参数对 WordSim-240 rho 的变化；每档 n=1，只连接数值轴实测点。"),
        ("word2vec_changed_classification", "Word2Vec 验证 F1 相对共享基线 0.4429 的百分点变化；分类轴不连线。"),
        ("fasttext_changed_similarity", "Gensim FastText 参数对 WordSim-240 rho 的变化；词相似度与分类分别评价。"),
        ("fasttext_changed_classification", "Gensim FastText 验证 F1 相对基线 0.4379 的变化；线程为速度对照，不参加选参。"),
    ]:
        figure(name, caption)
    add("项目已取消按变化幅度或极差筛选图形的限制。旧版图表与合表数量保留为历史展示记录，"
        "新图按数据类型、真实样本量和问题归属绘制；全部取值保存在 CSV，不作显著性或等效性推断。")

    subsection("6.4", "桶容量与线程的资源权衡", "resource-factors")
    bucket = records(section(parameter_text, "bucket"))
    add(table(["桶数", "碰撞率", "模型MiB", "验证F1"],
              [[r["值"], r["碰撞率"], r["模型MiB"], r["验证Macro-F1"]] for r in bucket]))
    figure("bucket_tradeoff", "桶数增大降低碰撞并增加子词数组内存，验证 F1 未单调提高；图为数组内存，表为整个模型文件大小。", word=False)
    add("碰撞率为不同子词共享桶的比例，不是查询出错概率。仅子词数组约占 `4×bucket×dim` 字节；"
        "200,000×100 为约 76.3 MiB，1,000,000×100 为约 381.5 MiB。RSS 还含其他矩阵、"
        "词典、分类器与语料，不能与数组字节或模型文件大小互换。")
    workers = records(section(parameter_text, "workers"))
    add(table(["CPU线程", "拟合秒", "相对单线程加速"],
              [[r["值"], r["训练秒"], f'{float(workers[0]["训练秒"]) / float(r["训练秒"]):.2f}×'] for r in workers]))
    add("本机 4/8 线程的单次加速约为 2.27/3.12 倍，但异步更新顺序与测得分类结果也变化，"
        "因此不参加正式选参，不作稳定性能或跨平台字节一致承诺。")

    subsection("6.5", "验证选参、原生补充优化与聚合交互", "optimization")
    add("最初验证选定配置在三个种子上的复验如下。Gensim 负采样 10 的单种子验证收益未稳定转移"
        "到测试；最初原生 unigram、25 轮路线仍拟合不足，早期结果继续保留。")
    add(table(*tables(section(classified, "最初验证选参后的三种子复验"))[0]))
    add("后续原生网格为 lr=0.1/0.5/1、epoch=25/100/200、wordNgrams=1/2，共 18 格；"
        "seed 42 验证选择 lr=0.1、epoch=200、wordNgrams=2，再三种子复验。原测试已经观察，"
        "这属于补充探索。增加轮数/学习率能改善拟合，但训练与验证差距同时提示过拟合。")
    figure("supplement_native_grid", "原生 FastText 的验证参数网格；展示学习率、轮数与词组合的交互，不按测试分数选择。", word=False)
    add(table(*tables(section(supplemental, "原生 FastText 的拟合诊断与改进"))[1]))
    add("补充原生模型一月测试 F1=0.4656 ± 0.0020，扩展六月为 0.4741 ± 0.0015，资源成本同时"
        "增加。扩展仍沿用一月选定表示参数，不在五月扩大网格；日期不确定性限制见第四部分。")
    migration = tables(section(supplemental, "原生 FastText 的拟合诊断与改进"))[2][1]
    add(table(["类别", "原生主配置测试F1", "补充固定参数测试F1"],
              [[migration[i][0].split(" / ")[0], migration[i][3], migration[i + 1][3]]
               for i in range(0, len(migration), 2)]))
    add("固定 12 类验证选定参数迁移到 2/4/8 类后也改善分类，每档仅使用对应类别训练文章；"
        "没有按测试重新选参，不能将迁移检查当成独立盲测。")
    interaction = tables(section(supplemental, "负采样与文章向量加权的交互"))[0]
    add(table(["配置", "三种子验证F1", "三种子测试F1"], [[r[0], r[2], r[3]] for r in interaction[1]]))
    add("negative=10 下 mean 与训练 IDF 加权分别复验，测试均值差很小，不支持加权必然更好。"
        "当前仅覆盖部分参数交互，验证集反复使用和三种子数量有限均限制结论。")

    subsection("6.6", "完整数据与报告复查", "data-resources")
    add(f'[CSV 总索引](data/csv_index.csv)记录全部 {len(exports["inventory"])} 份导出资源的行数、字段和校验值。'
        "当前索引覆盖 201 个分类训练/评估运行、323 个指标分区、99 条词模型记录，以及上述外部评价与评分资源；冻结评估复用模型，历史控制数量不相加为独立模型数。")
    add(table(["用途", "主要 CSV"], [
        ["分类与逐类明细", "[分类指标](data/classification_metrics.csv)、[PRF](data/per_class_metrics.csv)、[混淆计数](data/confusion_counts.csv)、[逐篇预测索引](data/prediction_file_index.csv)"],
        ["词评价与参数明细", "[外部指标](data/word_metrics.csv)、[逐关系类比](data/analogy_relations.csv)、[逐题索引](data/evaluation_file_index.csv)、[Word2Vec 因素](data/word2vec_factors.csv)、[FastText 因素](data/fasttext_factors.csv)"],
        ["训练与数值诊断", "[句子评分索引](data/sentence_score_file_index.csv)"],
        ["配置与其余完整记录", "[分类配置](data/run_configurations.csv)、[词模型配置](data/word_model_configurations.csv)、[补充运行](data/supplemental_runs.csv)、[旧报告全部表格](data/historical_report_tables.csv)"],
    ]))
    add("机器指标保留原浮点精度，historical_report_tables.csv 保留旧表展示精度；空值不是零。"
        "标签 01–12 导入表格软件时应按文本读取。file_index 的路径相对于项目根目录，既有大型"
        "逐题、逐句、逐篇 CSV 沿用原位置，不复制到正文。artifacts 与训练语料默认不纳入 Git，"
        "仅取得代码仓库时需按项目运行入口准备数据与资源。")
    add(f"代码基线为 `{baseline}`；源文档校验值见 [source_fingerprints.csv](data/source_fingerprints.csv)。"
        "项目环境为 Gensim（Python 3.12.15），固定 Gensim 4.4.0、fasttext-wheel 0.9.2、"
        "NumPy 1.26.4、SciPy 1.16.3、scikit-learn 1.7.2；精确版本见[锁定依赖](../requirements.lock.txt)。"
        "训练按 prepare、smoke、full 单种子、ablations、full 三种子、temporal/finalize、补充与词评价、"
        "evidence、深度核查及报告的顺序执行，勿并发写入同一运行索引。可用"
        "[run_all.py](../tools/run_all.py)顺序编排并复用完整缓存；这些运行是项目复现，不是第二部分的论文复现。")
    add("仅重写报告与 CSV，执行：")
    add("""```powershell
conda run -n Gensim python tools/build_consolidated_report.py
conda run -n Gensim python tools/build_consolidated_report.py --check
```""")
    add("生成器不调用训练、不修改选参或实验索引；检查六部分结构、第二部分与复现记录一致、源文档覆盖、"
        "CSV 完整性、关键数值与资源链接，结果见 [检查记录](../artifacts/consolidated_report_check.json)。"
        "所有观察限定于当前语料、版面标签、候选配置和固定版本，不声称更广泛的新闻主题分类优势。")
    return "\n\n".join(chunks) + "\n"
