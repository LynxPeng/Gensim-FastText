"""Select interpretable summaries for the paper; detailed observations live in CSV."""
from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(name):
    return (ROOT / "reports" / name).read_text(encoding="utf-8")


def tables(text):
    result, current = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            current.append([c.strip() for c in re.split(r"(?<!\\)\|", line)[1:-1]])
        elif current:
            result.append((current[0], current[2:]))
            current = []
    return result


def table(headers, rows):
    return "\n".join(["| " + " | ".join(map(str, headers)) + " |",
                      "| " + " | ".join("---" for _ in headers) + " |",
                      *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def section(text, name):
    body = text.split("## " + name + "\n", 1)[1]
    return body.split("\n## ", 1)[0]


def records(text, index=0):
    headers, rows = tables(text)[index]
    return [dict(zip(headers, row, strict=True)) for row in rows]


def data_summary():
    text = source("数据质量报告.md")
    monthly = records(text)
    return "清洗前 18,647 篇，清洗后 18,108 篇，合计排除 2.89%；压缩包与原文内容一致。\n\n" \
        + table(["月份", "原始文章", "保留文章", "保留比例"],
                [[r["月份"], r["原始文章"], r["保留文章"],
                  f'{int(r["保留文章"]) / int(r["原始文章"]):.2%}'] for r in monthly]) \
        + "\n\n一月训练/验证/测试为 2,138/456/456 篇，" \
        "各版面训练样本为 85–316 篇，存在类别不平衡；因此主指标使用 Macro-F1，" \
        "SVM 采用 balanced 类别权重。完整逐版面分布和六个月清洗损失见数据资源，" \
        "不在正文逐行展开。三处不同内容的重复段落编号均按原顺序保留；" \
        "没有根据编号缺口猜测补写内容。"


def source_index_summary():
    headers, rows = tables(source("源码位置索引.md"))[0]
    concepts = {"字符子词", "兼容 FNV 哈希", "OOV 向量", "训练 Cython 内核", "词表过滤与下采样",
                "任务生产与批次", "监督分类特征", "监督目标", "pybind 与 GIL", "连续内存"}
    return "完整 26 项源码定位与归档引用保存于 CSV。以下定位用于复查第 3.2 节的关键机制：\n\n" \
        + table(headers, [row for row in rows if row[0] in concepts])


def classification_summary():
    text = source("最终实验报告.md")
    matrix = records(section(text, "主配置完整矩阵"))
    methods = list(dict.fromkeys(row["方法"] for row in matrix))
    main = [[method, *(next(r["测试F1"] for r in matrix if r["方法"] == method and r["类别"] == str(k))
                          for k in [2, 4, 8, 12])] for method in methods]
    resources = [[r["方法"], r["训练秒"], r["峰值MiB"], r["模型MiB"]]
                 for r in matrix if r["类别"] == "12" and r["方法"] != "多数类"]
    chosen = tables(section(text, "验证集选定配置"))[0][1]
    frozen = tables(section(text, "冻结一月模型的跨时间评估"))[0][1]
    monthly = [[month, next(r[2] for r in frozen if r[0] == month),
                *(next(r[4] for r in frozen if r[0] == month and r[1] == method)
                  for method in ["TF-IDF", "Gensim FastText", "原生 FastText"])]
               for month in dict.fromkeys(r[0] for r in frozen)]
    expanded = records(section(text, "扩大训练语料"), 1)
    matched = records(section(text, "扩大训练语料"), 2)
    scaling = [[r["方法"], next(x["冻结Macro-F1"] for x in matched if x["方法"] == r["方法"]), r["测试F1"],
                next(x["差值"] for x in matched if x["方法"] == r["方法"]), r["训练秒"], r["模型MiB"]]
               for r in expanded]
    return "\n\n".join([
        "以下均为测试 Macro-F1。随机模型为 42/2026/7 三种子的均值 ± 样本标准差，"
        "确定性基线只运行一次。各方法仅拟合相应训练集。",
        "## 方法与类别规模的比较",
        table(["方法", "2 类", "4 类", "8 类", "12 类"], main),
        "2 类时 Word2Vec 和 Doc2Vec 的均值较高；4、8、12 类主配置中 TF-IDF 较高。"
        "类别规模同时改变训练数据和类别分布，不能将下降解释为类别数量的独立因果效应。"
        "一月 12 类 TF-IDF Accuracy=0.4803，其 Macro-F1=0.4652，二者分别衡量总体正确率与各类表现。",
        "![四档类别规模下的方法比较](figures/main_macro_f1.png)",
        "## 十二类实验的资源代价",
        table(["方法", "拟合秒", "峰值RSS MiB", "模型 MiB"], resources),
        "TF-IDF 在本配置下兼具分类表现与低资源代价。Gensim FastText 的子词矩阵增加内存和模型体积，"
        "OOV 可生成覆盖达到 1，并未带来更高分类 F1。训练时间包含表示拟合及 SVM 选 C，不能当作纯内核速度。",
        "## 最初验证选参后的三种子复验",
        table(["方法", "验证F1", "测试F1"], [[r[1], r[3], r[4]] for r in chosen]),
        "Gensim FastText 单种子验证最优候选的收益未稳定转移到测试；最初原生选定模型仍存在拟合不足。"
        "第 8 节另报告后续补充网格，保留这两项早期结果。",
        "## 冻结模型的跨月泛化",
        table(["月份", "文章数", "TF-IDF", "Gensim FastText", "原生 FastText"], monthly),
        "![一月冻结模型的跨月表现](figures/temporal_macro_f1.png)",
        "这里的原生路线为最初验证选定模型。二月至六月没有重新拟合词典、IDF、表示或分类器，"
        "后续文章排除与一月任意分区同组的近重复。月份间波动包含事件、版面安排和类别分布变化，"
        "不是随时间推移的单一效应。",
        "## 扩大训练的六月共同文章对照",
        "扩展训练使用一月至四月 11,802 篇，五月验证 3,115 篇，六月测试 3,191 篇；"
        "沿用一月选定的表示参数，Gensim 路线只在五月重选 SVM C。下表在相同 3,191 篇六月文章上成对比较。",
        table(["方法", "冻结F1", "扩展F1", "成对差值", "扩展拟合秒", "扩展模型MiB"], scaling),
        "三条路线均改善，原生路线改善幅度最大，但仍低于扩展 TF-IDF。增加训练文章也改变词表、内容与"
        "时间覆盖，因此不能把全部差值归因于样本数量。"
    ])


def parameter_summary():
    text = source("参数分析.md")
    changed = []
    titles = {"vector_size": "维度", "epochs": "训练轮数", "sg": "CBOW/SG",
              "objective": "训练目标", "alpha": "初始学习率"}
    for factor, label in titles.items():
        rows = tables(section(text, factor))[0][1]
        changed.append([label, "；".join(f"{r[0]}: {r[1]}" for r in rows)])
    bucket = tables(section(text, "bucket"))[0][1]
    workers = tables(section(text, "workers"))[0][1]
    return "\n\n".join([
        "单因素均在一月 12 类验证集、seed 42 下进行；每次只改变一项。以下选择能解释优化、"
        "容量和资源权衡的汇总，完整各档取值留在 CSV。单次测量不作显著性或交互因果推断。",
        "## 优化与容量对验证结果的影响",
        table(["因素", "取值及验证 Macro-F1"], changed),
        "负采样 10 的验证 F1=0.4760，是历史单线程候选中实际测得的最优配置；"
        "选参采用这个完整候选，没有拼接各因素单独最优的参数。维度、轮数、架构及学习率的结果"
        "均显示局部配置敏感性，不能推广为参数越大越好。三种子测试复验见第 5 节。",
        "## 哈希碰撞与存储代价",
        table(["桶数", "碰撞率", "模型MiB", "验证F1"], [[r[0], r[8], r[5], r[1]] for r in bucket]),
        "![桶容量、碰撞及验证表现](figures/bucket_tradeoff.png)",
        "桶数增加降低碰撞，但模型增大且验证 F1 没有单调提高。碰撞率为不同子词共享桶的比例，"
        "不是查询错误概率；该局部对照不能证明碰撞对分类的独立因果效应。图中内存为子词数组字节，"
        "表中模型大小为整个可运行 Bundle，两者计量范围不同。",
        "## 多线程速度对照",
        table(["CPU线程数", "拟合秒", "相对单线程加速"],
              [[r[0], r[3], f"{float(workers[0][3]) / float(r[3]):.2f}×"] for r in workers]),
        "线程对照显示本机加速，但更新顺序与结果可能变化，因此不参加正式模型选择。"
        "本节记录历史单线程与速度对照结果；当前默认训练 20 线程。Gensim 词向量训练仍使用 CPU。",
        "## 小幅变化与覆盖率的解释",
        "窗口、最低词频、字符子词范围、下采样、均值/IDF 聚合和 batch_words 的分类变化也完成记录，"
        "各组验证 F1 极差不超过 0.0091。它们仍可能改变成本或词表：最低词频从 1 增至 5 时，"
        "验证 token OOV 从 0.0477 增至 0.1089。这里的小幅单种子差异不足以证明这些设置等效。"
        "负采样 10 下两种聚合的三种子交互结果见第 8 节，完整设置和低分结果均保留。",
    ])


def word_summary(jobs):
    values=json.loads((ROOT/'artifacts/word_evaluation/metrics.json').read_text(encoding='utf-8'))
    ids={key for key,job in jobs.items() if job['classes']==12 and
         any(use['profile']=='main' for use in job['uses'])}
    names={'word2vec':'Word2Vec','fasttext':'Gensim FastText'}
    def group(method,benchmark):
        found=[r for r in values if r['id'] in ids and r['method']==method and r['benchmark']==benchmark]
        assert len(found)==3
        assert len({(r['covered'],r['total']) for r in found})==1
        return found
    def aggregate(rows,key,percent=False):
        numeric=[r[key]*(100 if percent else 1) for r in rows]
        return f'{statistics.mean(numeric):.4f} ± {statistics.stdev(numeric):.4f}'
    similarity=[]; analogy=[]
    for benchmark in ['wordsim240','wordsim297','wordsim296']:
        for method,name in names.items():
            rows=group(method,benchmark)
            similarity.append([benchmark,name,aggregate(rows,'spearman_rho'),aggregate(rows,'pearson_r'),
                               f"{rows[0]['covered']}/{rows[0]['total']}"])
    for benchmark in ['CA_translated','CA8_morphological','CA8_semantic']:
        for method,name in names.items():
            rows=group(method,benchmark)
            analogy.append([benchmark,name,f"{rows[0]['covered']}/{rows[0]['total']}",
                            aggregate(rows,'conditional_accuracy',True),aggregate(rows,'all_accuracy',True)])
    return "\n\n".join([
          "历史分类语料的词相似度、类比与 OOV 指标见[完整评价报告](词向量全面评价报告.md)，逐题数值保留在 CSV。",
          table(['基准','方法','Spearman ρ','Pearson r','覆盖／全部'],similarity),
          "## 词类比的覆盖与条件准确率",
          table(['基准','方法','覆盖／全部','覆盖正确率 %','全部正确率 %'],analogy),
        "覆盖题正确率与把未覆盖题按错计入的全量正确率分别解释，字符子词的可用性不等于语义正确性。",
        "## 图形选择与完整结果保留",
        "按评价目标与数据类型选择图形，不设置变化幅度门槛。历史配置仅支撑历史语料上的结论。",
        "## 关键趋势图及解释",
        "THUCNews 的 Word2Vec 与 FastText 分别记录逐轮验证指标并保存全部检查点，主指标只使用验证词对，选定模型再评价独立测试分区，详见[当前实验](THUCNews词向量实验.md)。",
        "## 句子 score 的适用边界",
        "HS score 为局部上下文对数项之和，跳过 OOV，使用查表近似；只作接口诊断，不解释为语言模型困惑度。",
    ])


def supplemental_summary():
    text = source("补充实验与改进报告.md").split("\n", 1)[1]
    # Reconstruct actual original lines to avoid altering numeric formatting.
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("| lr |"))
    end = first
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[first:end] = ["完整 18 格网格移入 CSV，保留下面的验证热力图用于比较学习率、轮数和词组合的交互。"]
    first = next(i for i, line in enumerate(lines) if line.startswith("| 原生 FastText 配置 | 种子 |"))
    end = first
    while end < len(lines) and lines[end].startswith("|"):
        end += 1
    lines[first:end] = ["各种子拟合诊断移入 CSV：原参数训练 F1 约 0.0247，改进参数约 0.997；"
                        "测试均值仍只有 0.2666，说明拟合改善未消除日期泛化差距。"]
    return "\n".join(lines)


def probe_summary():
    text = source("运行核验证据.md").split("\n", 1)[1]
    lines = text.splitlines()
    for marker, explanation in [
        ("| 词 | 1–3 子词 |", "中文、ASCII 和新词三种探针的字符子词与兼容哈希均与手工复算一致；"
         "完整子词列表与核对值移入 CSV。"),
        ("| 数组 | 形状 |", "整词与子词矩阵均为连续 float32 数组；小语料的子词桶数组占用远大于整词数组。"
         "数组形状、类型和字节数移入 CSV；正式桶容量的资源权衡见第 6.2 节。"),
        ("| 方法 | 原运行 |", "七种方法的完整验证预测 CSV 均与原运行逐字节一致。"
         "逐运行编号及配对核验结果移入数据资源。"),
    ]:
        first = next(i for i, line in enumerate(lines) if line.startswith(marker))
        end = first
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        lines[first:end] = [explanation]
    return "\n".join(lines)


def error_summary():
    text = source("错误分析.md")
    fitted = []
    for method in ["TF-IDF", "Gensim FastText", "原生 FastText"]:
        _, rows = tables(section(text, method))[0]
        fitted.append([method, rows[0][2], rows[1][2], rows[2][2]])
    examples = tables(section(text, "TF-IDF"))[3][1][:3]
    return "\n\n".join([
        "以下使用最初选定模型的 seed 42 预测解释拟合与标签性质，不用于重新选参。"
        "补充原生参数后的三种子与日期结果已在第 8 节保留。",
        "## 拟合差距与早期原生模型诊断",
        table(["方法", "训练F1（拟合样本）", "验证F1", "测试F1"], fitted),
        "TF-IDF 的训练与测试差距明显；最初原生模型在训练集上亦较低，更符合拟合不足的诊断，"
        "不能据此判定算法或绑定接口失效。增加优化预算的补充实验同时保留成本与泛化限制。",
        "## 版面标签与文章语义的差异",
        table(["文章标题", "真实版面", "预测版面"], [[r[3], r[1], r[2]] for r in examples]),
        "标题涉及人物、航空管理或外交任命，但真实目标是报纸版面位置，语义上接近的文章仍可被编辑安排"
        "在不同版面。这些案例用于说明标签含义，不构成误分类原因的因果证明。完整逐篇预测、"
        "各类 PRF、混淆计数和其他错误例均置于 CSV，正文不展开所有样本。",
    ])
