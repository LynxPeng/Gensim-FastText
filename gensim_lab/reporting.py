from __future__ import annotations

import collections
import csv
import json
import math
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote, urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .common import ARTIFACTS, PROCESSED, REPORTS, ROOT, code_hash, config, markdown_table, read_json, stamp, write_json
from .data import load_docs
from .runner import index
from .integrity import spec_is_current

NAMES = {"majority": "多数类", "tfidf": "TF-IDF", "lsi": "LSI", "word2vec": "Word2Vec", "doc2vec": "Doc2Vec", "fasttext": "Gensim FastText", "native_fasttext": "原生 FastText"}


def records():
    result = []
    manifest = read_json(PROCESSED / "manifest.json")
    for key, rec in index()["jobs"].items():
        path = ARTIFACTS / "runs" / key
        spec = read_json(path / "spec.json")
        if not spec_is_current(spec):
            continue
        item = {**rec, "run": key, "directory": path, "spec": spec}
        for field, name in [("summary", "summary.json"), ("test", "metrics_test.json"), ("diagnostics", "embedding_diagnostics.json")]:
            item[field] = read_json(path / name) if (path / name).exists() else None
        result.append(item)
    return result


def uses(r, profile):
    return any(u["profile"] == profile for u in r["uses"])


def select(rows, profile, method=None, classes=None):
    return [r for r in rows if uses(r, profile) and (method is None or r["method"] == method) and (classes is None or r["classes"] == classes)]


def meanstd(values, digits=4):
    values = list(values)
    if not values:
        return "未完成"
    mean = np.mean(values)
    return f"{mean:.{digits}f}" + (f" ± {np.std(values, ddof=1):.{digits}f}" if len(values) > 1 else "")


def runlink(r):
    return f"[{r['run']}](../artifacts/runs/{r['run']}/spec.json)"


def summary_rows(rows, profile):
    out = []
    for classes in sorted({r["classes"] for r in select(rows, profile)}):
        for method in config()["methods"]:
            group = select(rows, profile, method, classes)
            if not group:
                continue
            out.append([classes, NAMES[method], len(group), meanstd(r["summary"]["validation"]["macro_f1"] for r in group),
                        meanstd(r["test"]["macro_f1"] for r in group), meanstd(r["test"]["accuracy"] for r in group),
                        meanstd((r["summary"]["fit_seconds"] for r in group), 1),
                        f"{max(r['summary']['peak_rss_mb'] for r in group):.1f}",
                        meanstd((r["summary"]["model_bytes"] / 2**20 for r in group), 1),
                        meanstd((r["test"].get("coverage", {}).get("oov_rate", 0) for r in group), 4),
                        meanstd((r["test"].get("coverage", {}).get("oov_vector_coverage", 0) for r in group), 4)])
    return out


def plot_main(rows, figures):
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    for method in config()["methods"]:
        groups = [select(rows, "main", method, k) for k in config()["classes"]]
        values = [np.mean([r["test"]["macro_f1"] for r in group]) for group in groups]
        error = [np.std([r["test"]["macro_f1"] for r in group], ddof=1) if len(group) > 1 else 0 for group in groups]
        ax.errorbar(config()["classes"], values, yerr=error, marker="o", label=method)
    ax.set(xlabel="Number of newspaper pages", ylabel="Test Macro-F1", xticks=config()["classes"], ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend(ncol=2, fontsize=9)
    fig.savefig(figures / "main_macro_f1.png", dpi=170)
    plt.close(fig)


def plot_confusion(r, figures):
    counts = np.array(r["test"]["confusion_matrix"])
    normalized = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    labels = [f"{i:02d}" for i in range(1, r["classes"] + 1)]
    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            if counts[i, j]:
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8, color="white" if normalized[i, j] > 0.55 else "black")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
           xlabel="Predicted page", ylabel="True page", title=f"{r['method']} K={r['classes']} seed={r['seed']}")
    fig.colorbar(im, ax=ax, label="Row-normalized fraction")
    name = f"confusion_{r['run']}.png"
    fig.savefig(figures / name, dpi=150)
    plt.close(fig)
    return name


def numeric_exports(rows):
    fields = ["run", "profile", "axis", "value", "month", "method", "classes", "seed", "chosen_svm_c", "validation_macro_f1", "test_macro_f1", "test_accuracy", "fit_seconds", "peak_rss_mb", "model_bytes", "oov_rate", "oov_vector_coverage", "collision_fraction"]
    with (REPORTS / "实验汇总.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            s, t = r["summary"], r["test"] or {}
            coverage = t.get("coverage", s.get("validation", {}).get("coverage", {}))
            seen = set()
            tuning = s.get("tuning", [])
            chosen_c = max(tuning, key=lambda v: v["validation_macro_f1"])["C"] if tuning else ""
            for use in r["uses"]:
                signature = (use["profile"], use.get("axis"), json.dumps(use.get("value")), use.get("month"))
                if signature in seen:
                    continue
                seen.add(signature)
                writer.writerow({"run": r["run"], "profile": use["profile"], "axis": use.get("axis", ""), "value": json.dumps(use.get("value", ""), ensure_ascii=False),
                    "month": use.get("month", ""), "chosen_svm_c": chosen_c,
                    "method": r["method"], "classes": r["classes"], "seed": r["seed"], "validation_macro_f1": s.get("validation", {}).get("macro_f1", ""),
                    "test_macro_f1": t.get("macro_f1", ""), "test_accuracy": t.get("accuracy", ""),
                    "fit_seconds": s["fit_seconds"], "peak_rss_mb": s["peak_rss_mb"], "model_bytes": s["model_bytes"],
                    "oov_rate": coverage.get("oov_rate", ""), "oov_vector_coverage": coverage.get("oov_vector_coverage", ""),
                    "collision_fraction": (r["diagnostics"] or {}).get("collision_fraction", "")})


def quality_supplement():
    manifest = read_json(PROCESSED / "manifest.json")
    excluded = read_json(PROCESSED / "excluded.json")
    loss = collections.Counter((e["month"], e["page_label"], e["reason"]) for e in excluded)
    data = []
    for m in manifest["months"]:
        for page in [f"{i:02d}" for i in range(1, 13)]:
            losses = [loss[(m["month"], page, reason)] for reason in ["short_article", "exact_duplicate", "conflicting_exact_duplicate"]]
            kept = m["page_counts"].get(page, 0)
            data.append([m["month"], page, kept + sum(losses), kept, *losses])
    md = "# 各版面清洗损失\n\n排除原因按优先级互斥统计：异标签精确重复优先于短文，短文优先于同标签重复。原始数=保留数+三类损失，逐篇记录见[排除清单](../data/processed/excluded.json)。\n\n"
    md += markdown_table(["月份", "版面", "原始", "保留", "不足30词", "同标签重复", "标签冲突重复"], data)
    issues = read_json(PROCESSED / "anomalies.json")
    md += "\n\n## 重复段落编号\n\n" + markdown_table(["文章", "物理行", "段落", "原因"], [[r["id"], r["line"], r["paragraph"], r["reason"]] for r in issues])
    md += "\n\n不同内容的重复段落全部保留原顺序。三月的控制字符在清洗时移除；不猜测原本可能丢失的汉字。缺失首段/段号间隙的文章保留并通过 JSONL 的 quality_flags 标记。\n"
    (REPORTS / "各版面清洗损失.md").write_text(md, encoding="utf-8")


def ablations(rows):
    baseline = next(r for r in rows if uses(r, "main") and r["method"] == "fasttext" and r["classes"] == 12 and r["seed"] == 42)
    native_base = next(r for r in rows if uses(r, "main") and r["method"] == "native_fasttext" and r["classes"] == 12 and r["seed"] == 42)
    cfg = config()
    md = "# 参数分析\n\n所有分类分数来自一月 12 类验证集，seed 42；每次只改变一个因素。测试分数不参与该表或参数选择。基线配置共享缓存，所以相同配置的资源数值也是同一运行的测量。峰值 RSS 含加载语料、拟合、验证、诊断和序列化，训练秒数包含表示及 SVM 的 C 搜索。\n\n"
    md += "![桶数、碰撞与分类效果](figures/bucket_tradeoff.png)\n\n"
    explanations = {"vector_size": "维度增加表示容量，也按比例增加桶矩阵内存。", "window": "上下文窗口影响词共现范围；句子/段落边界仍限制上下文。", "min_count": "按词出现次数过滤，低阈值增加词表并减少 OOV。", "epochs": "更多遍历同时改变计算预算与学习率下降轨迹。", "sg": "CBOW 聚合上下文预测词，Skip-gram 预测上下文；速度和下游质量需分别观察。", "subwords": "中文较短词的字符片段数量随长度设置变化，OOV 覆盖只是可生成性。", "bucket": "碰撞率按不同子词统计；桶矩阵实际字节见运行诊断。", "objective": "ns10 增加负例；hs 对照设 hs=1、negative=0。", "alpha": "学习率影响优化步幅，各实验 min_alpha 保持固定。", "sample": "下采样只改变表示训练中的有效频词，不从分类文本删除停用词。", "pooling": "TF-IDF 加权采用训练集 IDF，词典外词使用最大训练 IDF，最终 L2 归一化。", "native_word_ngrams": "原生监督路线比较 unigram/bigram；关闭字符子词不等于关闭词 n-gram 桶。", "workers": "多线程改变异步更新顺序，属于速度对照，不参加正式选择。", "batch_words": "改变任务大小及学习率更新粒度；它是调度批次，不是平均梯度 mini-batch。"}
    selected_details = {}
    for axis, values in cfg["ablations"].items():
        md += f"## {axis}\n\n{explanations[axis]}\n\n"
        table, group = [], []
        for value in values:
            found = [(r, u) for r in rows for u in r["uses"] if u.get("axis") == axis and u.get("value") == value]
            if len(found) != 1:
                raise AssertionError(f"Missing/duplicate ablation {axis}={value}: {len(found)}")
            r, _ = found[0]
            group.append(r)
            s, d = r["summary"], r["diagnostics"] or {}
            v = s["validation"]
            base = native_base if axis == "native_word_ngrams" else baseline
            delta = v["macro_f1"] - base["summary"]["validation"]["macro_f1"]
            coverage = v.get("coverage", {})
            table.append([value, f"{v['macro_f1']:.4f}", f"{delta:+.4f}", f"{s['fit_seconds']:.2f}", f"{s['peak_rss_mb']:.1f}", f"{s['model_bytes']/2**20:.1f}", f"{coverage.get('oov_rate',0):.4f}", f"{coverage.get('oov_vector_coverage',0):.4f}", f"{d['collision_fraction']:.4f}" if "collision_fraction" in d else "—", runlink(r)])
        md += markdown_table(["值", "验证Macro-F1", "较基线", "训练秒", "峰值MiB", "模型MiB", "OOV率", "OOV向量覆盖", "碰撞率", "运行"], table)
        best = max(group, key=lambda r: r["summary"]["validation"]["macro_f1"])
        worst = min(group, key=lambda r: r["summary"]["validation"]["macro_f1"])
        best_value, worst_value = values[group.index(best)], values[group.index(worst)]
        times = [r["summary"]["fit_seconds"] for r in group]
        memories = [r["summary"]["peak_rss_mb"] for r in group]
        md += f"\n\n该组最高验证 Macro-F1={best['summary']['validation']['macro_f1']:.4f}（{best_value}），最低={worst['summary']['validation']['macro_f1']:.4f}（{worst_value}）；训练时间 {min(times):.1f}–{max(times):.1f} 秒，峰值 RSS {min(memories):.1f}–{max(memories):.1f} MiB。这是单种子局部对照，不作统计显著性或因素交互的结论。\n\n"
        if axis == "bucket":
            md += "桶数增加带来的碰撞率和矩阵大小都在实测中可查；更低碰撞没有保证本次验证分类分数更高，不能仅凭桶数决定配置。\n\n"
        elif axis == "workers":
            serial = group[values.index(1)]["summary"]["fit_seconds"]
            md += "相对单线程的拟合耗时加速比：" + "，".join(f"{v} 线程 {serial/r['summary']['fit_seconds']:.2f}×" for v, r in zip(values, group)) + "。语料规模、调度及异步更新同时影响测量；速度试验只有一次，不能作为稳定性能保证。\n\n"
    chosen = read_json(ARTIFACTS / "selection.json")
    md += "## 选定配置及三种子复验\n\n选择规则为一月验证 Macro-F1 最大，排除线程/批次速度实验；同分按配置指纹排序。未将每个因素的优选值拼成未经运行的新组合。\n\n"
    for method in ["fasttext", "native_fasttext"]:
        r = next(x for x in rows if x["run"] == chosen[method])
        parameters = r["spec"]["embedding" if method == "fasttext" else "native"]
        md += f"### {NAMES[method]}\n\n{runlink(r)}；seed 42 验证 Macro-F1={chosen[method + '_validation_macro_f1']:.4f}。\n\n```json\n{json.dumps(parameters, ensure_ascii=False, indent=2)}\n```\n\n"
        if method == "native_fasttext" and parameters["wordNgrams"] == 1 and parameters["maxn"] == 0:
            md += "原生 Python 包的 _build_args 会把 unigram 且关闭字符子词的有效桶数改为 0；上述 JSON 是传入配置，实际参数见错误分析的 error_diagnosis.json。模型大小按实际文件计数。\n\n"
        group = select(rows, "selected", method, 12)
        md += markdown_table(["种子", "验证Macro-F1", "测试Macro-F1", "测试Accuracy", "运行"], [[x["seed"], f"{x['summary']['validation']['macro_f1']:.4f}", f"{x['test']['macro_f1']:.4f}", f"{x['test']['accuracy']:.4f}", runlink(x)] for x in sorted(group, key=lambda x: x["seed"])])
        md += "\n\n"
        selected_details[method] = parameters
    (REPORTS / "参数分析.md").write_text(md, encoding="utf-8")
    return selected_details


def plot_bucket_tradeoff(rows, figures):
    values = config()["ablations"]["bucket"]
    group = [next(r for r in rows if any(u.get("axis") == "bucket" and u.get("value") == value for u in r["uses"])) for value in values]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    series = [[r["diagnostics"]["collision_fraction"] for r in group],
              [r["diagnostics"]["arrays"]["wv.vectors_ngrams"]["bytes"] / 2**20 for r in group],
              [r["summary"]["validation"]["macro_f1"] for r in group]]
    for ax, numbers, title in zip(axes, series, ["Distinct subword collision fraction", "Subword array memory (MiB)", "Validation Macro-F1 (seed 42)"]):
        ax.plot(range(len(values)), numbers, marker="o", color="#245a81")
        ax.set(xticks=range(len(values)), xticklabels=[f"{v:,}" for v in values], xlabel="Hash buckets", title=title)
        ax.grid(alpha=0.2)
        ax.set_ylim(0, 1 if title != "Subword array memory (MiB)" else max(numbers) * 1.2)
        for i, v in enumerate(numbers):
            ax.annotate(f"{v:.3f}" if title != "Subword array memory (MiB)" else f"{v:.1f}", (i, v), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    fig.savefig(figures / "bucket_tradeoff.png", dpi=170)
    plt.close(fig)


def temporal(rows, figures):
    all_docs = load_docs()
    splits = read_json(ARTIFACTS / "temporal_splits.json")
    table = []
    for month in ["199802", "199803", "199804", "199805", "199806"]:
        for method in ["tfidf", "fasttext", "native_fasttext"]:
            group = [r for r in select(rows, "frozen", method) if any(u.get("month") == month for u in r["uses"])]
            assert len(group) == (1 if method == "tfidf" else 3)
            table.append([month, NAMES[method], group[0]["test"]["n"], len(group),
                          meanstd(r["test"]["macro_f1"] for r in group), meanstd(r["test"]["accuracy"] for r in group),
                          meanstd(r["test"]["coverage"]["oov_rate"] for r in group),
                          meanstd(r["test"]["coverage"]["oov_vector_coverage"] for r in group)])
    md = "## 冻结一月模型的跨时间评估\n\n冻结的是一月 12 类训练得到的模型；后续月份没有再拟合词典、IDF、表示或分类器。后续文章排除与一月任意分区同组的近重复，避免同文跨时间夸大泛化。\n\n"
    md += markdown_table(["月份", "方法", "文章", "种子数", "Macro-F1", "Accuracy", "OOV率", "OOV向量覆盖"], table)
    md += "\n\n![冻结模型跨月份](figures/temporal_macro_f1.png)\n\n"
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    months = ["199802", "199803", "199804", "199805", "199806"]
    for method in ["tfidf", "fasttext", "native_fasttext"]:
        groups = [[r for r in select(rows, "frozen", method) if any(u.get("month") == m for u in r["uses"])] for m in months]
        means = [np.mean([r["test"]["macro_f1"] for r in g]) for g in groups]
        errors = [np.std([r["test"]["macro_f1"] for r in g], ddof=1) if len(g) > 1 else 0 for g in groups]
        ax.errorbar(months, means, yerr=errors, marker="o", label=method)
    ax.set(xlabel="Evaluation month", ylabel="Frozen model Macro-F1", ylim=(0, 1))
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(figures / "temporal_macro_f1.png", dpi=170)
    plt.close(fig)
    md += "## 扩大训练语料\n\n" + markdown_table(["分区", "月份", "文章数"], [["训练", "一月至四月", len(splits["train"])], ["验证", "五月", len(splits["validation"])], ["测试", "六月", len(splits["test"])]])
    md += "\n\n后续分区排除与前序分区同组的文章。FastText 两条路线沿用一月选定的表示参数；Gensim 路线仍在五月验证集选择 SVM C。六月只做最终评估。\n\n"
    md += markdown_table(["类别", "方法", "种子数", "验证F1", "测试F1", "测试Accuracy", "训练秒", "峰值MiB", "模型MiB", "OOV率", "OOV向量覆盖"], summary_rows(rows, "expanded"))
    from .models import metrics
    aligned, comparisons = [], []
    for method in ["tfidf", "fasttext", "native_fasttext"]:
        differences, frozen_scores, expanded_scores, counts = [], [], [], []
        for expanded in select(rows, "expanded", method):
            frozen = next(r for r in select(rows, "frozen", method) if r["seed"] == expanded["seed"] and any(u.get("month") == "199806" for u in r["uses"]))
            samples = []
            for run in [frozen, expanded]:
                with (run["directory"] / "predictions_test.csv").open(encoding="utf-8-sig") as f:
                    samples.append({p["id"]: p for p in csv.DictReader(f)})
            shared = sorted(set(samples[0]) & set(samples[1]))
            labels = [f"{i:02d}" for i in range(1, 13)]
            truth = [samples[0][i]["true_label"] for i in shared]
            assert truth == [samples[1][i]["true_label"] for i in shared]
            values = [metrics(truth, [s[i]["predicted_label"] for i in shared], labels)["macro_f1"] for s in samples]
            frozen_scores.append(values[0]); expanded_scores.append(values[1]); differences.append(values[1] - values[0]); counts.append(len(shared))
            comparisons.append({"method": method, "seed": expanded["seed"], "common_articles": len(shared), "same_article_set": set(samples[0]) == set(samples[1]), "frozen_macro_f1": values[0], "expanded_macro_f1": values[1], "difference": values[1] - values[0]})
        aligned.append([NAMES[method], counts[0], meanstd(frozen_scores), meanstd(expanded_scores), meanstd(differences)])
    write_json(ARTIFACTS / "matched_june_comparison.json", comparisons)
    md += "\n\n### 六月共同文章的规模对照\n\n以下取冻结评估和扩展测试共有的文章，按相同真实标签重新汇总已有预测；不重新训练，也不重新选参。差值为扩展减冻结，各种子成对计算。\n\n"
    md += markdown_table(["方法", "共同文章", "冻结Macro-F1", "扩展Macro-F1", "差值"], aligned)
    md += "\n\n[逐种子对齐数值](../artifacts/matched_june_comparison.json)。按月变化包含版面安排、事件和类别分布变化，不能只归因于时间流逝。\n\n"
    (REPORTS / "时间扩展实验.md").write_text("# 时间扩展实验\n\n" + md, encoding="utf-8")
    return md


def per_class(rows, figures):
    chosen = [r for r in rows if r["seed"] == 42 and (uses(r, "main") or uses(r, "selected") or uses(r, "expanded"))]
    md = "# 逐类指标与混淆矩阵\n\n以下展示 seed 42 的主矩阵、选定配置和扩展实验。其他种子的完整逐类 PRF 与混淆矩阵在各运行 metrics_test.json 中；[实验汇总](实验汇总.csv)可定位运行。混淆矩阵行是真实版面、列是预测版面，图中色阶按行归一化，数字是文章数。\n\n"
    for r in chosen:
        profile = ", ".join(sorted({u["profile"] for u in r["uses"] if u["profile"] in ["main", "selected", "expanded"]}))
        md += f"## {NAMES[r['method']]} / {r['classes']} 类 / {profile}\n\n{runlink(r)}；[全部测试指标](../artifacts/runs/{r['run']}/metrics_test.json)。\n\n"
        md += markdown_table(["版面", "Precision", "Recall", "F1", "Support"], [[page, f"{v['precision']:.4f}", f"{v['recall']:.4f}", f"{v['f1-score']:.4f}", int(v["support"])] for page, v in r["test"]["per_class"].items() if re.fullmatch(r"\d{2}", page)])
        md += "\n\n"
        if r["classes"] == 12:
            name = plot_confusion(r, figures)
            md += f"![混淆矩阵 {r['method']} {profile}](figures/{name})\n\n"
    (REPORTS / "逐类指标与混淆矩阵.md").write_text(md, encoding="utf-8")


def errors(rows):
    from .models import Bundle, metrics
    docs = {d["id"]: d for d in load_docs()}
    md = "# 错误分析\n\n以下分析已选定配置的测试预测，不用于重新选参。标签是版面位置；同主题跨版、专栏、体裁和日期安排都可能造成语义上合理但版面标签不一致的预测。样例保留标题，只引用短片段供人工核对。\n\n"
    for method in ["tfidf", "fasttext", "native_fasttext"]:
        profile = "main" if method == "tfidf" else "selected"
        r = next(r for r in select(rows, profile, method, 12) if r["seed"] == 42)
        bundle = Bundle.load(r["directory"])
        train_docs = [docs[i] for i in sorted(r["spec"]["splits"]["train"])]
        train_pred, _, _ = bundle.predict(train_docs)
        train_metrics = metrics([d["page_label"] for d in train_docs], train_pred, bundle.labels)
        diagnosis = {"train": train_metrics, "validation": r["summary"]["validation"], "test": r["test"]}
        if method == "native_fasttext":
            actual = bundle.native.f.getArgs()
            diagnosis["actual_native_parameters"] = {n: getattr(actual, n) for n in ["dim", "epoch", "lr", "wordNgrams", "minn", "maxn", "bucket", "thread", "seed", "minCount"]}
        write_json(r["directory"] / "error_diagnosis.json", diagnosis)
        del bundle
        with (r["directory"] / "predictions_test.csv").open(encoding="utf-8-sig") as f:
            predictions = list(csv.DictReader(f))
        mistakes = [p for p in predictions if p["true_label"] != p["predicted_label"]]
        confusion = collections.Counter((p["true_label"], p["predicted_label"]) for p in mistakes)
        true_support = collections.Counter(p["true_label"] for p in predictions)
        predicted_support = collections.Counter(p["predicted_label"] for p in predictions)
        md += f"## {NAMES[method]}\n\n{runlink(r)}；错误 {len(mistakes)}/{len(predictions)}。\n\n"
        md += markdown_table(["分区", "文章", "Macro-F1", "Accuracy"], [[part, values["n"], f"{values['macro_f1']:.4f}", f"{values['accuracy']:.4f}"] for part, values in [("训练（已拟合样本）", train_metrics), ("验证", r["summary"]["validation"]), ("测试", r["test"])]])
        md += "\n\n训练分数仅用于检查拟合程度，不参与报告的泛化分数或参数重选；[数值核查](../artifacts/runs/" + r["run"] + "/error_diagnosis.json)。\n\n"
        md += markdown_table(["真实版面→预测版面", "数量"], [[f"{a}→{b}", n] for (a, b), n in confusion.most_common(10)])
        md += "\n\n" + markdown_table(["版面", "真实数量", "预测数量"], [[f"{i:02d}", true_support[f"{i:02d}"], predicted_support[f"{i:02d}"]] for i in range(1, 13)])
        # Deterministic samples: one example per leading confusion pair.
        examples = []
        for (a, b), _ in confusion.most_common(10):
            p = next(p for p in mistakes if p["true_label"] == a and p["predicted_label"] == b)
            doc = docs[p["id"]]
            examples.append([p["id"], a, b, doc["title"][:60] or "（缺首段）", len(doc["tokens"]), f"{float(p['score']):.4f}"])
        md += "\n\n" + markdown_table(["文章", "真实", "预测", "标题", "词数", "分数"], examples)
        md += "\n\n"
        if method == "native_fasttext":
            dominant, n = predicted_support.most_common(1)[0]
            md += f"原生监督模型预测最多的版面为 {dominant}，占 {n/len(predictions):.1%}。该分布与低 Macro-F1 一起检查类别偏置；模型在可分小语料上已通过接口核验，当前固定学习率、轮数及长文章特征平均的效果不足。不能仅凭此结果断言监督 FastText 算法普遍无效。\n\n"
    md += "## 可进一步核查的方向\n\n版面 01–04 经常都含政策、经济和社会报道，低维平均可能压掉细粒度栏目词；IDF 加权能减轻高频功能词影响。某些后部版面样本较少，逐类 Recall 和预测数量能区分分类器偏向大类与缺少特征。上述是结合任务结构的解释，未建立人工错误主题标签，不把解释当作已验证的因果结论。\n\n改进实验可另行研究原生学习率/轮数、标题权重、文章长度分层、主题标签或栏目识别；本次不使用测试集结果修改已选参数。\n"
    (REPORTS / "错误分析.md").write_text(md, encoding="utf-8")


def verify(rows):
    docs = {d["id"]: d for d in load_docs()}
    checks = {}
    def check(name, value):
        checks[name] = bool(value)
    cfg = config()
    expected_main = sum(1 if m in ["majority", "tfidf"] else 3 for m in cfg["methods"]) * len(cfg["classes"])
    main = select(rows, "main")
    check("主实验68个运行完成", len(main) == expected_main and all(r["summary"] and r["test"] for r in main))
    for classes in cfg["classes"]:
        for method in cfg["methods"]:
            group = select(rows, "main", method, classes)
            seeds = {r["seed"] for r in group}
            check(f"主矩阵{classes}类/{method}种子覆盖", seeds == ({42} if method in ["tfidf", "majority"] else {42, 2026, 7}))
    check("选定模型三种子", all({r["seed"] for r in select(rows, "selected", m, 12)} == {42, 2026, 7} for m in ["fasttext", "native_fasttext"]))
    check("时间扩展7个训练及35个月度评估", len(select(rows, "expanded")) == 7 and len(select(rows, "frozen")) == 35)
    coverage = all(any(u.get("axis") == axis and u.get("value") == value for r in rows for u in r["uses"]) for axis, values in cfg["ablations"].items() for value in values)
    check("单因素及速度实验全部取值", coverage)
    disjoint, valid_ids, no_failed = True, True, True
    for r in rows:
        spec = r["spec"]
        splits = spec["splits"]
        if "source_run" in spec:
            source = read_json(ARTIFACTS / "runs" / spec["source_run"] / "spec.json")
            splits = {"train": source["splits"]["train"] + source["splits"]["validation"] + source["splits"]["test"], "validation": [], "test": splits["test"]}
        ids = {p: set(v) for p, v in splits.items()}
        groups = {p: {docs[i]["group"] for i in v} for p, v in ids.items()}
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            disjoint &= not bool(ids[a] & ids[b] or groups[a] & groups[b])
        valid_ids &= all(i in docs for values in ids.values() for i in values)
        no_failed &= r["summary"] is not None and not (r["directory"] / "failure.json").exists()
    check("文章及近重复组分区无交叉", disjoint and valid_ids)
    check("当前指纹运行无失败", no_failed)
    trained = [r for r in rows if "source_run" not in r["spec"]]
    check("全部训练模型保存加载一致", all(r["summary"]["save_load"]["same_labels"] and r["summary"]["save_load"]["max_score_difference"] <= 1e-6 for r in trained))
    reloads = read_json(ARTIFACTS / "prediction_reload_checks.json")
    from .integrity import cache_problem, reload_check_complete
    reload_by_part = {(r["run"], r.get("part")): r for r in reloads}
    check("每个训练模型全部验证预测加载一致", all(reload_check_complete(reload_by_part.get((r["run"], "validation")), r["run"], r["spec"], "validation") for r in trained))
    check("全部正式及冻结测试预测加载一致", all(reload_check_complete(reload_by_part.get((r["run"], "test")), r["run"], r["spec"], "test") for r in rows if r["test"]))
    check("完整文件与SHA256缓存检查", all(cache_problem(r["directory"], r["spec"], "test" if "source_run" in r["spec"] else "validation") is None for r in rows))
    check("Gensim原生格式导出加载一致", all((r["directory"] / "native_export.json").exists() and read_json(r["directory"] / "native_export.json")["same_model_values_after_load"] for r in trained if r["method"] not in ["majority", "native_fasttext"]))
    check("Gensim全部数组与词典IDF原生重载一致", all(read_json(r["directory"] / "native_export.json").get("verification_version") == 2 and read_json(r["directory"] / "native_export.json").get("dictionary_and_idf_match") is True for r in trained if r["method"] not in ["majority", "native_fasttext"]))
    identity_path = ARTIFACTS / "doc2vec_identity_probe_after.json"
    identities = read_json(identity_path)["checks"] if identity_path.exists() else []
    checked_identity = {r["run"] for r in identities if r["same_vectors_labels_scores_after_id_change"]}
    check("Doc2Vec四种类别规模编号不影响文本向量及预测", all(r["run"] in checked_identity for r in main if r["method"] == "doc2vec"))
    cli_path = ARTIFACTS / "cli_prediction_checks.json"
    cli_checks = read_json(cli_path) if cli_path.exists() else {}
    check("公开预测入口模型及输入边界核验", cli_checks.get("passed") is True
          and len(cli_checks.get("model_checks", [])) in [7, 8]
          and {r["method"] for r in cli_checks.get("model_checks", [])} == set(cfg["methods"])
          and all(r.get("same_as_bundle_and_whitespace_variant") is True for r in cli_checks.get("model_checks", []))
          and len(cli_checks.get("invalid_inputs", [])) == 2
          and all(r["exit_code"] == 2 for r in cli_checks.get("invalid_inputs", []))
          and cli_checks.get("doc2vec_oov", {}).get("coverage", {}).get("zero_documents") == 1
          and cli_checks.get("legacy_doc2vec_rejected") is True)
    audit_path = ARTIFACTS / "deep_audit.json"
    audited = read_json(audit_path) if audit_path.exists() else {}
    check("当前全部运行独立重算数据分组与指标", all(audited.get("checks", {}).values()) and bool(audited.get("checks")) and {r["run"] for r in rows}.issubset({r["run"] for r in audited.get("run_checks", []) if r["passed"]}))
    check("记录有效线程数与固定哈希种子", all(r["spec"]["native"]["thread"] >= 1 and r["spec"]["embedding"]["workers"] >= 1 and r["summary"]["environment"]["hash_seed"] == "0" for r in rows if any(uses(r, p) for p in ["main", "selected", "expanded", "frozen"])))
    def finite(value):
        if isinstance(value, dict):
            return all(finite(v) for v in value.values())
        if isinstance(value, list):
            return all(finite(v) for v in value)
        return not isinstance(value, float) or math.isfinite(value)
    check("指标无NaN或Infinity", all(finite(r["summary"]) and finite(r["test"]) for r in rows))
    repeated = read_json(ARTIFACTS / "reproducibility" / "checks.json")
    check("七种方法独立重训逐篇预测一致", len(repeated) == 7 and all(r["identical"] for r in repeated))
    manifest = read_json(PROCESSED / "manifest.json")
    check("原始数量及压缩包一致", manifest["raw_total"] == 18647 and manifest["zip_matches"] is True and manifest["raw_total"] - sum(manifest["exclusion_counts"].values()) == manifest["kept_total"])
    check("高级探针与固定源码存在", (ARTIFACTS / "evidence" / "probes.json").exists() and (ROOT / "references" / "source_positions.json").exists())
    sources = read_json(ROOT / "references" / "source_manifest.json")["sources"]
    check("Gensim归档源码与4.4.0标签文本一致", all(s.get("matches_upstream_tag") is True for s in sources if s["version"] == "4.4.0"))
    check("最终测试已统一完成", all(r["test"] for r in rows if any(uses(r, p) for p in ["main", "selected", "expanded", "frozen"])))
    return checks


def links():
    files = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), *sorted(REPORTS.glob("*.md"))]
    checked, missing = 0, []
    for path in files:
        if not path.exists():
            missing.append(str(path.relative_to(ROOT)))
            continue
        content = path.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]\n]*\]\(([^)\n]+)\)", content):
            target = target.strip().strip("<>")
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target = unquote(target.split("#", 1)[0])
            checked += 1
            if not (path.parent / target).resolve().exists():
                missing.append({"file": str(path.relative_to(ROOT)), "target": target})
    return {"checked": checked, "missing": missing, "valid": not missing}


def report():
    REPORTS.mkdir(exist_ok=True)
    figures = REPORTS / "figures"
    figures.mkdir(exist_ok=True)
    rows = records()
    if not (ARTIFACTS / "finalization.json").exists() or "completed" not in read_json(ARTIFACTS / "finalization.json"):
        raise RuntimeError("Final test evaluation must complete before publishing the final report")
    for script in ["export_native.py", "check_saved_predictions.py", "check_doc2vec_identity.py", "check_cli_predictions.py"]:
        print(f"Verifying {script}", flush=True)
        output = subprocess.run([sys.executable, str(ROOT / "tools" / script)], cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        (ARTIFACTS / (script.replace(".py", "") + ".log")).write_text(output.stdout + output.stderr, encoding="utf-8")
        if output.returncode:
            raise RuntimeError(output.stderr)
    checks = verify(rows)
    if not all(checks.values()):
        write_json(ARTIFACTS / "acceptance.json", checks)
        raise AssertionError({k: v for k, v in checks.items() if not v})
    numeric_exports(rows)
    quality_supplement()
    parameters = ablations(rows)
    plot_bucket_tradeoff(rows, figures)
    plot_main(rows, figures)
    time_text = temporal(rows, figures)
    per_class(rows, figures)
    errors(rows)
    manifest = read_json(PROCESSED / "manifest.json")
    md = "# Gensim 与 FastText 最终实验报告\n\n"
    md += f"报告生成于 {stamp()}。任务为人民日报 **版面分类**，标签来自编号 `01–12`，不视为人工主题标签。原始 18,647 篇，清洗保留 {manifest['kept_total']:,} 篇；一月 3,050 篇，训练/验证/测试为 2,138/456/456。全部七种方法完成 2、4、8、12 类，随机模型三种子正式运行。\n\n"
    tf = next(r for r in select(rows, "main", "tfidf", 12))
    group = select(rows, "selected", "fasttext", 12)
    native = select(rows, "selected", "native_fasttext", 12)
    md += f"一月 12 类测试集：TF-IDF Macro-F1={tf['test']['macro_f1']:.4f}；验证集选定的 Gensim FastText 为 {meanstd(r['test']['macro_f1'] for r in group)}；选定的原生 FastText 为 {meanstd(r['test']['macro_f1'] for r in native)}。原生路线已完成分类及接口核验，其实际效果见表与错误分析，结果未作筛除。\n\n"
    improved = select(rows, "supplement_native_selected", "native_fasttext", 12)
    improved_expanded = select(rows, "supplement_expanded_native", "native_fasttext", 12)
    if improved and all(r["test"] for r in improved):
        md += f"后续审查修正了 Doc2Vec 推断依赖文章编号的问题，当前主矩阵已更新为按文本生成种子的结果，历史运行保留。补充网格改善原生 FastText 的拟合：一月 12 类测试 Macro-F1 为 {meanstd(r['test']['macro_f1'] for r in improved)}"
        if improved_expanded and all(r["test"] for r in improved_expanded):
            md += f"，扩展训练后六月为 {meanstd(r['test']['macro_f1'] for r in improved_expanded)}"
        md += "。一月与 TF-IDF 的差值区间包含零，不能确认稳定优势；固定参数的日期迁移也仍低于该划分的 TF-IDF。新增结果属于原测试已观察后的补充探索，详见[补充实验与改进报告](补充实验与改进报告.md)。\n\n"
    md += "## 实验约定\n\n词典、IDF、表示及分类器仅拟合当前类别训练集。统一的 12 类近重复组划分后筛选类别；训练输入包含标题及正文，无编号、词性、实体标签，不额外删除停用词。词向量以句子/段落构造上下文。LinearSVC 类别权重 balanced，C 在验证集从 0.1/1/10 选择。测试在选参与三种子复验后统一评估。完整配置见[实施计划](../docs/实施计划.md)与[配置文件](../configs/experiments.yaml)。\n\n"
    md += "随机模型的均值 ± 样本标准差来自 42/2026/7，确定性 TF-IDF 和多数类仅一次。本报告的历史运行使用 workers/thread=1、Python hash seed=0、BLAS/OMP=1；现行默认训练 20 线程、数值库 4 线程，历史结果未改写。训练时间包含表示及 SVM 选 C；峰值 RSS 每 50ms 采样，取各种子的最大值；模型大小为可运行 Bundle/native.bin 的文件字节，同内容的 Gensim 原生导出不重复计数。计时是本机一次测量，没有给出性能置信区间。\n\n"
    md += "OOV 是对应模型训练词表外的 token 比例。OOV 向量覆盖衡量可生成向量；TF-IDF/LSI 和 Word2Vec 为跳过，FastText 可用子词。Doc2Vec 能为文章推断向量，不单独生成 OOV 词向量。多数类无词表，表中覆盖字段记 0 并不代表其文本特征质量。\n\n"
    headers = ["类别", "方法", "种子数", "验证F1", "测试F1", "测试Accuracy", "训练秒", "峰值MiB", "模型MiB", "OOV率", "OOV向量覆盖"]
    md += "## 主配置完整矩阵\n\n" + markdown_table(headers, summary_rows(rows, "main"))
    md += "\n\n![四种类别规模](figures/main_macro_f1.png)\n\n不同类别规模改变任务难度与类别分布，不能把分数下降只解释为模型退化。低频版面的效果见[逐类指标与混淆矩阵](逐类指标与混淆矩阵.md)。\n\n"
    md += "## 验证集选定配置\n\n选定配置来自一月 12 类单因素实验候选，正式三种子复验；不是按测试分数选择。详细参数和每个因素的资源/覆盖/碰撞见[参数分析](参数分析.md)。\n\n" + markdown_table(headers, summary_rows(rows, "selected")) + "\n\n"
    md += time_text
    md += "## 证据、限制及交付\n\n所有数据损失、原始文件 SHA-256、重复段落和逐篇排除可查：[数据质量](数据质量报告.md)、[各版面清洗损失](各版面清洗损失.md)。SimHash 距离阈值是启发式；它防止检测到的近重复跨分区，不能保证识别全部语义重复。分组使用不带标签的全体文本相似度，只控制去重；不从后续文本拟合任何特征统计。\n\n"
    md += "[论文与源码分析](论文阅读与源码分析.md)解释两条路线的目标差异及周边代码；[运行核验](运行核验证据.md)包含手工向量/哈希复算、迭代/任务/学习率探针、数组布局及七种方法的独立重训。固定配置在本机重训验证预测 CSV 字节一致；跨平台依赖或浮点实现改变不保证逐位一致。\n\n"
    md += "主消融结果来自一个验证划分，补充实验检查了负采样与聚合、原生学习率与轮数和词 n-gram 的部分交互，仍未覆盖全部因素组合；三种子只衡量初始化波动。版面标签含栏目及编排信息，一月随机划分结果不能代表未来主题分类。原生 Windows 包来自第三方 fasttext-wheel 0.9.2，源码阅读固定上游标签，二进制供应链与 wheel SHA-256 见[来源清单](../references/source_manifest.json)。\n\n"
    md += "[错误分析](错误分析.md)提供主要混淆及真实文章样例；[时间实验](时间扩展实验.md)提供未来月份结果；[实验记录](实验记录.md)链接所有当前运行。原始数值见[实验汇总 CSV](实验汇总.csv)，完整模型和逐篇预测在 artifacts/runs；[运行说明与复现](../README.md)给出入口和完整执行顺序。\n"
    if (REPORTS / "补充实验与改进报告.md").exists():
        md += "\n[补充实验与改进报告](补充实验与改进报告.md)记录后续参数网格、三种子、日期隔离、标签置乱与日期自助抽样；这些新增结果属于原测试已观察后的补充探索，原主矩阵和最初选参结果保留。\n"
    (REPORTS / "最终实验报告.md").write_text(md, encoding="utf-8")
    word_evaluation = ARTIFACTS / "word_evaluation" / "figure_qa.json"
    if word_evaluation.exists() and read_json(word_evaluation)["status"] == "exported":
        from .word_figures import insert_into_main
        insert_into_main()
    ledger(rows)
    # Create the acceptance file before checking links so links to it resolve.
    checklist = "# 高级要求完成清单\n\n以下按实际运行和机器核验生成，全部要求通过才允许生成最终报告。\n\n" + markdown_table(["检查", "状态"], [[k, "通过" if v else "失败"] for k, v in checks.items()])
    checklist += "\n\n高级内容的可读解释：[论文与源码](论文阅读与源码分析.md)、[源码行号](源码位置索引.md)、[参数分析](参数分析.md)、[运行探针](运行核验证据.md)。所有文档为 Markdown；CSV/JSON、PNG、模型、日志和配置是计算或运行资源。旧 Word 文件是先前转录的历史版本，已提供 Markdown 版本，不作为最终交付格式。\n"
    word_checks = ARTIFACTS / "word_evaluation" / "verification.json"
    if word_checks.exists() and read_json(word_checks)["status"] == "complete":
        evidence = read_json(word_checks)
        checklist += "\n\n## 全面评价补充验收\n\n" + markdown_table(["检查", "状态"], [[k, "通过" if v else "失败"] for k,v in evidence["checks"].items()])
        checklist += "\n\n[词向量全面评价报告](词向量全面评价报告.md)、[原始验证记录](../artifacts/word_evaluation/verification.json)保存外部评价指标、逐句 score 与图形核查证据。\n"
    (REPORTS / "高级要求完成清单.md").write_text(checklist, encoding="utf-8")
    link_result = links()
    checks["Markdown本地链接和图片存在"] = link_result["valid"]
    write_json(ARTIFACTS / "link_check.json", link_result)
    write_json(ARTIFACTS / "acceptance.json", checks)
    with (REPORTS / "高级要求完成清单.md").open("a", encoding="utf-8") as f:
        f.write(f"\n本地链接与图片检查 {link_result['checked']} 项，结果：{'通过' if link_result['valid'] else '失败'}。[机器记录](../artifacts/link_check.json)。\n")
    if not link_result["valid"]:
        raise AssertionError(link_result)
    print(f"Published Markdown reports; {len(rows)} runs; all {len(checks)} acceptance checks passed", flush=True)


def ledger(rows):
    md = "# 实验记录\n\n每行是一个当前数据/代码指纹的实际运行，相同配置可承担多个用途。选参只读 validation；frozen 行引用 source_run 模型，训练时间为 0，未在未来月份训练。历史不同指纹缓存保留但不进入报告。\n\n"
    md += markdown_table(["用途", "方法", "类", "种子", "验证F1", "测试F1", "时间戳", "配置", "日志"], [[", ".join(dict.fromkeys(u["profile"] + (":" + u["axis"] if "axis" in u else ":" + u["month"] if "month" in u else "") for u in r["uses"])), NAMES[r["method"]], r["classes"], r["seed"], f"{r['summary']['validation']['macro_f1']:.4f}" if "validation" in r["summary"] else "—", f"{r['test']['macro_f1']:.4f}" if r["test"] else "未评估（选参/冒烟）", r["summary"]["completed"], runlink(r), f"[日志](../artifacts/runs/{r['run']}/{'evaluation.log' if 'source_run' in r['spec'] else 'console.log'})"] for r in rows])
    md += "\n\n环境：每个 summary.json 包含 Python、安装包精确版本、CPU、RAM、哈希种子。配置内保存训练/验证/测试 ID、数据与代码指纹；复用之前检查配置哈希，代码或数据变化时产生新目录。最终参数选择见[选择 JSON](../artifacts/selection.json)，测试完成列表见[最终评估](../artifacts/finalization.json)。\n"
    (REPORTS / "实验记录.md").write_text(md, encoding="utf-8")
