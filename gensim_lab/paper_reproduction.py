"""Full-scale original-data reproduction, isolated from People's Daily jobs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from .common import ROOT, environment, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .paper_data import PAPER_ARTIFACTS, PREPARED, PAPER_CONFIG, prepare_classification, prepare_tags, settings, source_ready

RUNS = PAPER_ARTIFACTS / "runs"
REPORT = ROOT / "reports/论文实验复现.md"


def configurations(task, manifest, dataset, seeds):
    configuration = settings()
    manifest_path = ROOT / manifest["directory"] / "manifest.json"
    protocol = {name: file_hash(ROOT / "gensim_lab" / name)
                for name in ["paper_data.py", "paper_worker.py", "paper_hs.py"]}
    base = {"dataset": dataset, "task": task, "manifest": manifest_path.relative_to(ROOT).as_posix(),
            "data_fingerprint": manifest["fingerprint"], "smoke": manifest["spec"]["smoke"],
            "config_sha256": file_hash(PAPER_CONFIG), "protocol_sha256": protocol,
            "configuration": configuration,
            "packages": dict(environment()["packages"], torch=version("torch"), numba=version("numba"))}
    if task == "classification":
        dimensions = [configuration["classification"]["dimension"]]
        learning_rate = configuration["classification"]["datasets"][dataset]["learning_rate"]
    else:
        dimensions = configuration["tags"]["dimensions"]
        learning_rate = configuration["tags"]["native_learning_rate"]
    for seed in seeds:
        for dimension in dimensions:
            for ngram in [1, 2]:
                yield dict(base, method="native_fasttext", seed=seed, dimension=dimension,
                           word_ngrams=ngram, learning_rate=learning_rate)
            yield dict(base, method="gensim_fasttext", seed=seed, dimension=dimension,
                       word_ngrams=1, learning_rate=0.025)


def cache_valid(directory, spec):
    summary_path = directory / "summary.json"
    if not summary_path.exists():
        return False
    try:
        summary = read_json(summary_path)
        if summary["spec"] != spec or not summary["metrics"].get("test"):
            return False
        for name, digest in summary["seal"].items():
            if not (directory / name).exists() or file_hash(directory / name) != digest:
                return False
        manifest = read_json(ROOT / spec["manifest"])
        if summary["metrics"]["test"]["n"] != manifest["partitions"]["test"]["rows"]:
            return False
        check = read_json(directory / "save_load_check.json")
        return (check["same_labels_and_scores"] is True and
                check["partitions"]["test"]["samples"] == manifest["partitions"]["test"]["rows"])
    except (ValueError, KeyError, FileNotFoundError):
        return False


def schedule(spec):
    directory = RUNS / fingerprint(spec)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "spec.json", spec)
    return directory


def execute(spec, directory):
    if cache_valid(directory, spec):
        print(f"Reuse {spec['dataset']}/{spec['method']}/{spec['dimension']}d/seed {spec['seed']}", flush=True)
        return
    print(f"Start {spec['dataset']}/{spec['method']}/{spec['dimension']}d/ngram {spec['word_ngrams']}/seed {spec['seed']}", flush=True)
    from .runtime import worker_environment
    env = worker_environment()
    env.update(PYTHONIOENCODING="utf-8", CUBLAS_WORKSPACE_CONFIG=":4096:8")
    state = {"status": "running", "run": directory.name, "spec": spec, "updated": stamp()}
    write_json(PAPER_ARTIFACTS / "progress.json", state)
    with (directory / "run.log").open("a", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-m", "gensim_lab.paper_worker", "--run", str(directory)],
                                cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    state.update(status="complete" if result.returncode == 0 else "failed", returncode=result.returncode, updated=stamp())
    write_json(PAPER_ARTIFACTS / "progress.json", state)
    if result.returncode:
        raise RuntimeError(f"Experiment failed: {directory}; inspect run.log. Full-scale result not marked complete.")
    if not cache_valid(directory, spec):
        raise RuntimeError(f"Completed experiment failed artifact verification: {directory}")


def records(*, current=True):
    completed = []
    for directory in sorted(RUNS.glob("*")) if RUNS.exists() else []:
        if (directory / "summary.json").exists():
            summary = read_json(directory / "summary.json")
            spec = summary["spec"]
            valid_protocol = all(file_hash(ROOT / "gensim_lab" / name) == digest
                                 for name, digest in spec["protocol_sha256"].items())
            if (not spec["smoke"] and (not current or
                    (spec["config_sha256"] == file_hash(PAPER_CONFIG) and valid_protocol))):
                completed.append((directory, summary))
    return completed


def report():
    configuration = settings()
    completed = records()
    data = ROOT / "reports/data"
    data.mkdir(exist_ok=True)
    headers = ["run", "dataset", "task", "method", "dimension", "word_ngrams", "seed",
               "n_test", "precision_at_1", "recall_at_1", "macro_f1", "fit_seconds", "peak_memory_mib", "model_bytes",
               "test_seconds", "total_seconds", "cpu_training_threads"]
    rows_out = []
    for directory, summary in completed:
        spec, result = summary["spec"], summary["metrics"]["test"]
        rows_out.append([directory.name, spec["dataset"], spec["task"], spec["method"], spec["dimension"],
                         spec["word_ngrams"], spec["seed"], result["n"], result["precision_at_1"],
                         result["recall_at_1"], result["macro_f1_over_training_labels"], summary["fit_seconds"],
                         summary["peak_memory_mib"], summary["model_bytes"], result["seconds"], summary["total_seconds"],
                         configuration["embedding"]["workers"] if spec["method"] == "gensim_fasttext"
                         else configuration[spec["task"]]["native_threads"]])
    with (data / "paper_reproduction_results.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows_out)
    expected = len(configuration["seeds"]) * (len(configuration["classification"]["datasets"]) * 3
                                               + len(configuration["tags"]["dimensions"]) * 3)
    def display(values):
        if not values:
            return "待运行"
        mean = statistics.mean(values) * 100
        return f"{mean:.2f} ± {statistics.stdev(values) * 100:.2f}" if len(values) == 3 else f"{mean:.2f}（n={len(values)}，未完成）"
    original_unigram = [91.5, 93.9, 98.1, 93.8, 60.4, 72.0, 55.8, 91.2]
    original_bigram = [92.5, 96.8, 98.6, 95.7, 63.9, 72.3, 60.2, 94.6]
    chunks = ["# FastText 论文实验复现", "<a id=\"reproduction-protocol\"></a>", "## 2.1 数据与实现",
        "本部分使用原论文的八套分类数据和完整 YFCC100M 文本标签数据，保留作者提供的训练、验证及测试划分。"
        "原生 FastText 使用监督训练；Gensim FastText 先学习无监督词向量，再用平均文章向量接分类器，属于原数据上的近似复现。"
        "Gensim 的字符子词与原生的词 bigram 是不同特征，结果分开解释。",
        "分类实验采用 10 维、5 轮，原生学习率沿用作者公布的选参结果；Gensim 使用 CBOW 与固定 C=1 的 LinearSVC。"
        "标签实验采用 50/200 维、5 轮，原生使用层次 softmax；Gensim 接外部 Huffman 分类头，可用时由 CUDA 训练，预测采用精确树搜索。"
        "正式训练使用种子 42、2026、7；本章经用户批准使用 20 个 CPU 训练线程，全项目已取消单线程约束，旧结果保留实际线程数。"
        "多线程更新不保证逐位确定性，三种子均值与样本标准差衡量训练波动，不构成显著性检验。"
        "预测单线程；本机、实现版本和计时范围与论文不同，耗时不直接视为论文速度的复现。",
        "来源：[原论文](../paper/Bag%20of%20Tricks%20for%20Efficient%20Text%20Classification.pdf)、"
        "[作者分类脚本](https://github.com/facebookresearch/fastText/blob/v0.9.2/classification-results.sh)、"
        "[官方 YFCC 数据入口](https://fasttext.cc/docs/en/dataset.html)、"
        "[运行配置](../configs/paper_reproduction.yaml)。",
        "<a id=\"reproduction-classification\"></a>", "## 2.2 文本分类实验"]
    dataset_rows = []
    for position, (name, values) in enumerate(configuration["classification"]["datasets"].items()):
        found = [entry for entry in rows_out if entry[1] == name]
        unigram = [entry[8] for entry in found if entry[3] == "native_fasttext" and entry[5] == 1]
        native = [entry[8] for entry in found if entry[3] == "native_fasttext" and entry[5] == 2]
        gensim = [entry[8] for entry in found if entry[3] == "gensim_fasttext"]
        dataset_rows.append([name, values["train"], values["test"], original_unigram[position], original_bigram[position],
                             display(unigram), display(native), display(gensim)])
    chunks.append(markdown_table(["数据集", "训练数", "测试数", "论文 unigram (%)", "论文 bigram (%)",
                                  "本轮 unigram (%)", "本轮 bigram (%)", "Gensim (%)"], dataset_rows))
    chunks.append("分类 Accuracy 等于单标签 P@1；完整 Macro-F1、逐类 Precision/Recall/F1 和混淆矩阵另存。"
                  "三种子近乎恒定的值保留在表中，避免用三个点放大小差异。原文数值是对照基准，未混入本轮均值。")
    chunks.extend(["<a id=\"reproduction-tags\"></a>", "## 2.3 YFCC100M 标签预测实验",
        "原训练集 91,188,648 条，原验证集和测试集各 930,497 条。按作者词表与标签最低频次 100 过滤评估，"
        "得到有效验证 542,967 条、有效测试 543,424 条，后者与论文完全一致；训练仍读取全部原始行。标签数 312,116。"
        "当前分发数据得到 297,139 个词（含句末标记），比论文的 297,141 少 2 个；此差异保留记录，未人为补词。"
        "最高分标签属于任一真实标签即记为命中，以 P@1 为主指标；R@1 和逐标签指标另存明细。"])
    tag_rows = []
    for dimension in configuration["tags"]["dimensions"]:
        for method, ngram, label in [("native_fasttext", 1, "原生 unigram"), ("native_fasttext", 2, "原生 bigram"),
                                    ("gensim_fasttext", 1, "Gensim 向量 + 外部 HS")]:
            values = [entry[8] for entry in rows_out if entry[2] == "tags" and entry[3] == method
                      and entry[4] == dimension and entry[5] == ngram]
            reference = {(50, 1): 31.2, (50, 2): 36.7, (200, 1): 41.1, (200, 2): 46.1}
            tag_rows.append([dimension, label, reference[(dimension, ngram)] if method == "native_fasttext" else "无对应模型", display(values)])
    chunks.append(markdown_table(["维度", "路线", "论文 P@1 (%)", "本轮 P@1 (%)"], tag_rows))
    chunks.append("Gensim 分类头是冻结平均向量上的外部 Huffman SGD 模型，采用批量训练；原生模型联合更新输入和标签向量。"
                  "二者训练目标和实现不同，不能把差值全部归因于 Gensim 框架。外部分类头用 CUDA，精确树搜索用 CPU，未用束搜索近似替代。")
    chunks.extend(["<a id=\"reproduction-evidence\"></a>", "## 2.4 运行证据与状态",
        f"当前完成 {len(completed)}/{expected} 个正式运行。冒烟实验不计入正式结果，缺少的配置保留为待运行，整体复现尚未完成。"
        if len(completed) != expected else f"已完成 {len(completed)}/{expected} 个正式运行。",
        "逐运行结果见[汇总 CSV](data/paper_reproduction_results.csv)，当前任务见[进度记录](../artifacts/paper_reproduction/progress.json)。"
        "模型、完整分区预测、逐类指标、训练日志和重载检查分别保存在各运行目录。"
        "预测 CSV 的 log_score 是标签对数概率，不是句子语言模型对数概率；本章的 FastText 模型不支持后者。"
        "训练时间包含词典、向量学习、文档池化及分类器拟合，测试时间包含指标与预测写盘；峰值内存为进程 RSS，CUDA 显存另存分类头记录。",
        "旧单线程运行保留在原目录，但配置与源码指纹不同，不计入当前结果。检查点在轮次边界保存，"
        "模型重载后复查全部验证和测试预测。源码、配置、数据指纹变化会生成新运行目录。",
        "复现入口：`conda run -n Gensim python -m gensim_lab.paper_reproduction --stage all --task all`。"])
    temporary = REPORT.with_suffix(".md.tmp")
    temporary.write_text("\n\n".join(chunks) + "\n", encoding="utf-8")
    temporary.replace(REPORT)
    return {"completed": len(completed), "expected": expected, "report": str(REPORT)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "train", "report", "all"], default="all")
    parser.add_argument("--task", choices=["classification", "tags", "all"], default="all")
    parser.add_argument("--datasets", default="all")
    parser.add_argument("--seeds", default="42,2026,7")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--wait-for-download", action="store_true")
    args = parser.parse_args()
    PAPER_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if args.stage == "report":
        print(json.dumps(report(), ensure_ascii=False))
        subprocess.run([sys.executable, str(ROOT / "tools/sync_paper_reports.py")], cwd=ROOT, check=True)
        return
    from .integrity import artifact_lock
    with artifact_lock():
        seeds = [int(value) for value in args.seeds.split(",")]
        jobs = []
        names = list(settings()["classification"]["datasets"]) if args.datasets == "all" else args.datasets.split(",")
        tasks = [("classification", name) for name in names] if args.task in {"classification", "all"} else []
        if args.task in {"tags", "all"}:
            tasks.append(("tags", "yfcc100m"))
        for task, name in tasks:
            while not source_ready(name):
                if not args.wait_for_download:
                    raise FileNotFoundError(f"Complete original data required: {name}")
                print(f"Waiting for original download to finish: {name}", flush=True)
                time.sleep(30)
            manifest = prepare_classification(name, smoke=args.smoke) if task == "classification" else prepare_tags(smoke=args.smoke)
            if args.stage == "prepare":
                continue
            for spec in configurations(task, manifest, name, seeds):
                directory = schedule(spec)
                jobs.append({"run": directory.name, "spec": spec})
                write_json(PAPER_ARTIFACTS / "scheduled.json", {"updated": stamp(), "jobs": jobs})
                execute(spec, directory)
                report()
                if not args.smoke:
                    subprocess.run([sys.executable, str(ROOT / "tools/sync_paper_reports.py")], cwd=ROOT, check=True)
        if not args.smoke:
            print(json.dumps(report(), ensure_ascii=False))


if __name__ == "__main__":
    main()
