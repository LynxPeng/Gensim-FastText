from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ARTIFACTS, REPORTS, ROOT, file_hash, markdown_table, read_json, stamp, write_json
from .data import load_docs
from .reporting import NAMES, meanstd, records, select
from .supplemental import clustered_bootstrap, settings


def report():
    result = read_json(ARTIFACTS / "supplemental_results.json")
    assert "completed" in result
    rows = records()
    by_key = {r["run"]: r for r in rows}
    figures = REPORTS / "figures"
    figures.mkdir(exist_ok=True)
    cfg = settings()
    docs = {d["id"]: d for d in load_docs()}
    primary = select(rows, "main", "tfidf", 12)
    original_native = select(rows, "selected", "native_fasttext", 12)
    original_fasttext = select(rows, "selected", "fasttext", 12)
    selected_native = [by_key[key] for key in result["native_selected"]]
    expanded_native = [by_key[key] for key in result["expanded_native"]]
    june_tfidf = select(rows, "expanded", "tfidf", 12)
    june_fasttext = select(rows, "expanded", "fasttext", 12)
    june_native = select(rows, "expanded", "native_fasttext", 12)
    comparisons = {
        "january": clustered_bootstrap({"TF-IDF": [r["run"] for r in primary],
            "Gensim FastText": [r["run"] for r in original_fasttext],
            "补充原生 FastText": result["native_selected"]}, docs, cfg["bootstrap_replicates"], cfg["bootstrap_seed"]),
        "june": clustered_bootstrap({"TF-IDF": [r["run"] for r in june_tfidf],
            "Gensim FastText": [r["run"] for r in june_fasttext],
            "原选定原生 FastText": [r["run"] for r in june_native],
            "补充原生 FastText": result["expanded_native"]}, docs, cfg["bootstrap_replicates"], cfg["bootstrap_seed"]),
    }
    write_json(ARTIFACTS / "date_bootstrap.json", comparisons)
    blocks = {month: clustered_bootstrap(value["models"], docs, cfg["bootstrap_replicates"], cfg["bootstrap_seed"], block_size=7)
              for month, value in comparisons.items()}
    write_json(ARTIFACTS / "date_block_bootstrap.json", blocks)
    with (REPORTS / "补充实验数值.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["profile", "run", "method", "seed", "validation_macro_f1", "test_macro_f1", "test_accuracy", "fit_seconds", "peak_rss_mb", "model_bytes"])
        for r in rows:
            uses = sorted({u["profile"] for u in r["uses"] if u["profile"].startswith("supplement_")})
            for profile in uses:
                writer.writerow([profile, r["run"], r["method"], r["seed"], r["summary"].get("validation", {}).get("macro_f1", ""),
                                 (r["test"] or {}).get("macro_f1", ""), (r["test"] or {}).get("accuracy", ""),
                                 r["summary"]["fit_seconds"], r["summary"]["peak_rss_mb"], r["summary"]["model_bytes"]])

    # Only validation scores appear in the parameter-search heatmap.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ngrams, ax in zip(cfg["native_word_ngrams"], axes):
        grid = np.array([[next(r["validation_macro_f1"] for r in result["native_grid"]
                              if r["lr"] == lr and r["epoch"] == epoch and r["wordNgrams"] == ngrams)
                          for epoch in cfg["native_epochs"]] for lr in cfg["native_learning_rates"]])
        im = ax.imshow(grid, vmin=0, vmax=0.6, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(cfg["native_epochs"])), cfg["native_epochs"])
        ax.set_yticks(range(len(cfg["native_learning_rates"])), cfg["native_learning_rates"])
        ax.set(xlabel="Epochs", ylabel="Learning rate", title=f"wordNgrams={ngrams}; validation Macro-F1")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                ax.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", color="white" if grid[i,j] < 0.35 else "black")
    fig.colorbar(im, ax=axes, label="Validation Macro-F1")
    fig.savefig(figures / "supplement_native_grid.png", dpi=160)
    plt.close(fig)

    best = result["selected_native_config"]
    md = "# 补充实验与改进报告\n\n"
    md += f"生成于 {stamp()}。[事先固定的计划](../docs/审查与补充验证计划.md)与[配置](../configs/supplemental.yaml)规定 18 格参数网格、三种子复验、日期隔离、置乱对照和日期自助抽样。**本轮属于补充探索：原一月及六月测试结果已经被查看，不能视为新的盲测。** 选参代码仍只使用验证分数；原主实验保留。\n\n"
    md += "## 原生 FastText 的拟合诊断与改进\n\n"
    md += f"仅用一月验证集选出的补充配置是 lr={best['lr']}、epoch={best['epoch']}、wordNgrams={best['wordNgrams']}，其余参数不变，本节保存的是历史 CPU 单线程运行；现行默认训练 20 线程。[选择记录](../artifacts/supplemental_selection.json)给出分数与平分时的规则。\n\n"
    md += markdown_table(["lr", "epoch", "wordNgrams", "训练Macro-F1", "验证Macro-F1", "配置"],
        [[r["lr"], r["epoch"], r["wordNgrams"], f"{r['training_macro_f1']:.4f}", f"{r['validation_macro_f1']:.4f}",
          f"[运行](../artifacts/runs/{r['run']}/spec.json)"] for r in result["native_grid"]])
    md += "\n\n![原生 FastText 验证参数网格](figures/supplement_native_grid.png)\n\n"
    md += "训练指标是已拟合样本的检查，不能当作泛化指标。低学习率与少轮数的组合拟合不足；增加轮数或学习率可改善本数据的拟合，但高训练分数与验证差距也显示有限语料下的过拟合。网格不是普遍最优参数证明，不应单凭原固定 25 轮的结果判定 FastText 算法不适合版面分类。\n\n"
    def group_row(label, group):
        return [label, len(group), meanstd(r["summary"]["validation"]["macro_f1"] for r in group),
                meanstd(r["test"]["macro_f1"] for r in group), meanstd(r["test"]["accuracy"] for r in group),
                meanstd((r["summary"]["fit_seconds"] for r in group), 1),
                meanstd((r["summary"]["model_bytes"] / 2**20 for r in group), 1)]
    md += markdown_table(["场景/方法", "种子数", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "拟合秒", "模型MiB"], [
        group_row("一月 TF-IDF", primary), group_row("一月原选定原生 FastText", original_native),
        group_row("一月补充原生 FastText", selected_native), group_row("扩展 TF-IDF / 六月", june_tfidf),
        group_row("扩展原选定原生 FastText / 六月", june_native), group_row("扩展补充原生 FastText / 六月", expanded_native)])
    md += "\n\n新增扩展模型沿用一月验证选出的参数；五月验证没有用于扩大此网格，六月也没有用于重新选择。训练与模型体积的代价均保留，完整数据见[补充 CSV](补充实验数值.csv)。\n\n"
    scales_path = ARTIFACTS / "supplemental_scales.json"
    if scales_path.exists():
        scales = read_json(scales_path)
        md += "### 固定参数迁移到 2、4、8 类\n\n"
        md += "使用同一个 12 类验证选出的参数，沿用原划分后筛选类别；每个模型仅使用对应类别训练文章，不用测试选参。\n\n"
        md += markdown_table(["类别/方法", "种子数", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "拟合秒", "模型MiB"], [
            group_row(f"{classes} 类 / {label}", group)
            for classes in [2, 4, 8]
            for label, group in [("原主配置", select(rows, "main", "native_fasttext", classes)),
                                 ("补充固定参数", [by_key[key] for key in scales["runs"] if by_key[key]["classes"] == classes])]]) + "\n\n"

    md += "## 负采样与文章向量加权的交互\n\n"
    pooling = []
    for kind in ["mean", "tfidf"]:
        group = [by_key[r["run"]] for r in result["pooling"] if r["pooling"] == kind]
        pooling.append(group_row("negative=10 / " + kind, group))
    md += markdown_table(["配置", "种子数", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "拟合秒", "模型MiB"], pooling)
    md += "\n\n两种聚合分别复验，未按测试分数再选择。加权时 IDF 仅来自训练集，词典外 token 取训练最大 IDF，FastText 仍能生成子词向量。加权不必然优于均值。\n\n"

    md += "## 按日期隔离的七方法验证\n\n"
    md += markdown_table(["分区", "日期", "文章数"], [[part, dates, result["chronological_counts"][part]]
        for part, dates in [("train", "1–21 日"), ("validation", "22–26 日"), ("test", "27–31 日")]])
    md += "\n\n[日期划分与跨边界近重复移除](../artifacts/chronological_splits.json)。此处全部方法使用原计划主参数，各分区日期和近重复组不交叉，SVM 仅在相应日期验证集选 C。\n\n"
    chronology = []
    for method in NAMES:
        group = [by_key[key] for key in result["chronological"] if by_key[key]["method"] == method]
        chronology.append(group_row(NAMES[method], group))
    md += markdown_table(["方法", "种子数", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "拟合秒", "模型MiB"], chronology)
    md += "\n\n该切分改变训练规模、日期、版面构成和事件分布，不能将其与随机切分的差值全部解释为泄漏或时间效应；它补充检查更接近预测后续日期的场景。\n\n"

    transferred_path = ARTIFACTS / "supplemental_chronological_native.json"
    if transferred_path.exists():
        transferred = read_json(transferred_path)
        md += "### 改进参数的固定迁移检查\n\n"
        md += "发现原主配置在日期隔离语料上验证分数仍较低后，追加三种子检查；沿用已经选定的 lr=0.1、epoch=200、wordNgrams=2，不再调参。模型只拟合 1–21 日，但参数来自原一月随机划分验证，该选择已经接触部分后续日期的标签。此表是参数迁移的敏感性分析，不能当作完整隔离的前瞻评估，也不能声称改进配置从未见过目标日期信息。\n\n"
        md += markdown_table(["日期隔离模型", "种子数", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "拟合秒", "模型MiB"], [
            group_row("TF-IDF", [by_key[key] for key in result["chronological"] if by_key[key]["method"] == "tfidf"]),
            group_row("原生 FastText 原主参数", [by_key[key] for key in result["chronological"] if by_key[key]["method"] == "native_fasttext"]),
            group_row("原生 FastText 改进固定参数", [by_key[key] for key in transferred["runs"]])])
        md += "\n\n[固定参数迁移记录与全部限制](../artifacts/supplemental_chronological_native.json)。全部三个种子的模型及逐篇预测均保留。\n\n"
        md += markdown_table(["原生 FastText 配置", "种子", "训练Macro-F1", "验证Macro-F1", "测试Macro-F1"], [
            [label, by_key[key]["seed"], f"{transferred['training_macro_f1'][key]:.4f}",
             f"{by_key[key]['summary']['validation']['macro_f1']:.4f}", f"{by_key[key]['test']['macro_f1']:.4f}"]
            for label, keys in [("原主参数", [key for key in result["chronological"] if by_key[key]["method"] == "native_fasttext"]),
                                ("改进固定参数", transferred["runs"])] for key in keys])
        md += "\n\n训练分数只诊断拟合，不参与再次选择参数；随机划分上的提升不能直接推断到后续日期。\n\n"

    md += "## 标签置乱的负对照\n\n"
    original_keys = {"tfidf": primary[0]["run"], "fasttext": next(r["run"] for r in original_fasttext if r["seed"] == 42),
                     "native_fasttext": next(r["run"] for r in selected_native if r["seed"] == 42)}
    controls = []
    for key in result["label_controls"]:
        row, true = by_key[key], by_key[original_keys[by_key[key]["method"]]]
        controls.append([NAMES[row["method"]], f"{true['test']['macro_f1']:.4f}", f"{row['test']['macro_f1']:.4f}",
                         f"{true['test']['macro_f1']-row['test']['macro_f1']:.4f}", f"[对照](../artifacts/runs/{key}/spec.json)"])
    md += markdown_table(["方法 seed 42", "真实训练标签测试F1", "置乱训练标签测试F1", "差值", "证据"], controls)
    md += "\n\n置乱只作用于训练标签，文本、训练样本、标签数量和验证/测试真值保持一致，表示参数及原生 FastText 参数固定，不扩大网格。TF-IDF 和 Gensim FastText 的 SVM 仍在预先固定的 C={0.1,1,10} 中用验证集选择，与真实标签模型执行同一训练流程；这不是锁定全部分类器参数的对照。单次置乱是诊断性负对照，不能代替多次置乱的显著性检验，也不能证明所有潜在泄漏均不存在。\n\n"

    md += "## 按日期成对抽样的不确定性\n\n"
    md += "对共同测试文章，以整日为抽样单元进行 2,000 次有放回抽样；比较方法共享同一组日期权重，避免将同日文章当成完全独立样本。随机模型在每次抽样内先求各模型分数，再对三个种子求平均。种子本身未重采样；区间是当前固定模型下的描述性 2.5%/97.5% 分位数。\n\n"
    for month, value in comparisons.items():
        md += f"### {'一月' if month == 'january' else '六月'}：{value['articles']} 篇，{value['dates']} 个日期\n\n"
        md += markdown_table(["方法", "Macro-F1", "日期抽样区间", "相对 TF-IDF 差值", "成对差值区间"], [
            [name, f"{interval['macro_f1']:.4f}", " / ".join(f"{x:.4f}" for x in interval["percentile_95"]),
             f"{value['paired_differences'][name]['difference']:.4f}" if name in value["paired_differences"] else "—",
             " / ".join(f"{x:.4f}" for x in value["paired_differences"][name]["percentile_95"]) if name in value["paired_differences"] else "—"]
            for name, interval in value["intervals"].items()]) + "\n\n"
    md += "区间包含 0 时，当前重采样不能支持稳定的方向优势；不包含 0 也不是跨月份、跨报刊的效果保证。参数搜索、测试已观察和日期数量有限的限制仍然存在，未做多重比较校正。\n\n"
    md += "### 连续日期相关性的敏感性检查\n\n"
    md += "同一事件可能跨日延续，因此另以连续 7 个已观察日期为循环块进行 2,000 次抽样，拼接并截取到原日期数。该检查保留块内日期顺序；首尾循环是抽样定义，不是实际新闻时间连续的事实。区间仍只针对当前模型和月份。\n\n"
    md += markdown_table(["月份", "方法相对 TF-IDF", "点差值", "七日期块分位区间"], [
        ["一月" if month == "january" else "六月", name, f"{diff['difference']:.4f}", " / ".join(f"{x:.4f}" for x in diff["percentile_95"])]
        for month, value in blocks.items() for name, diff in value["paired_differences"].items()])
    md += "\n\n[连续日期块数值](../artifacts/date_block_bootstrap.json)。这补充日期相关性检查，不把有利的抽样定义用于重新挑选模型。\n\n"
    md += "## 工程加固与复现\n\n"
    md += "缓存现在核对模型、输入、指标、预测 ID/真值/分数及 SHA-256；损坏缓存会保留到 cache_recovery 后重建，旧选参必须匹配代码、数据与依赖。操作系统文件锁保护写入，进程退出自动释放；锁文件会保留，存在并不表示仍被占用。即使已设置 Python 哈希种子，CLI 也检查并固定 BLAS/OMP/NumExpr 线程。\n\n"
    md += "原生 Gensim 导出现在核对完整数组、词表、词典和 IDF；保存模型的完整验证及测试预测（包括冻结月份）再次加载核对。[数据与实验深度核查](数据与实验深度核查.md)独立重算原文、分组和指标。\n\n"
    md += "完整重载证据还核对分区名及样本数，不能以抽样核验冒充整个分区。层次 softmax 的 code/point 嵌套数组逐层核对数值、类型和有限性。公开 predict 入口检查七种方法及补充原生模型、分词空白变化、Doc2Vec 全未登录词、空文本与纯标点、旧 Doc2Vec 协议拒绝；见[入口核验记录](../artifacts/cli_prediction_checks.json)。\n\n"
    md += "### Doc2Vec 编号依赖修正\n\n"
    md += "原推断随机状态按文章编号重置，导致相同词序列更换编号也可能改变预测。修正前 20 篇探针中有 1 篇标签改变；当前使用词序列 SHA-256 生成推断种子，原文 ID 只用于追踪。所有四种类别规模三种子、冒烟、日期隔离 Doc2Vec 已按新协议重训；当前主矩阵中的 Doc2Vec 更新为修正结果，旧运行保留。\n\n"
    md += "[修正前探针](../artifacts/doc2vec_identity_probe_before.json)、[四种类别规模全部验证文章的编号不变性核验](../artifacts/doc2vec_identity_probe_after.json)、[内容种子实现](../gensim_lab/content_seed.py)。新协议与辅助训练入口的源码 SHA-256 随配置保存，缓存不能复用旧协议。\n\n"
    md += "```powershell\nconda run -n Gensim python -m gensim_lab.supplemental\nconda run -n Gensim python tools/deep_audit.py\nconda run -n Gensim python -m pytest --basetemp .artifact-work/pytest-audit-temp -q\nconda run -n Gensim python -m gensim_lab report\n```\n\n"
    md += "[全部补充运行](../artifacts/supplemental_results.json)、[日期抽样数值](../artifacts/date_bootstrap.json)、[工程入口](../README.md)。本轮完成已声明的核查与实验；不承诺绝对无误，也不通过不断查看测试结果追逐更高分数。\n"
    (REPORTS / "补充实验与改进报告.md").write_text(md, encoding="utf-8")
    print("Supplemental Markdown report and paired date-bootstrap results published", flush=True)
