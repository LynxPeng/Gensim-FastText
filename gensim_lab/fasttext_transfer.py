"""THUCNews Gensim word vectors -> native fastText, on all six PFR months."""
from __future__ import annotations

import argparse
import collections
import csv
import gc
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from .common import ROOT, PROCESSED, environment, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .runtime import settings as runtime_settings, worker_environment

BASE = ROOT / "artifacts/fasttext_transfer"
CONFIG = ROOT / "configs/fasttext_transfer.yaml"
REPORT = ROOT / "reports/FastText预训练向量分类实验.md"
VARIANTS = ["scratch", "word2vec", "fasttext"]
LABELS = [f"{i:02d}" for i in range(1, 13)]


@contextmanager
def transfer_lock():
    """This pipeline writes its own index; no main/THUCNews artifacts are mutated."""
    BASE.mkdir(parents=True, exist_ok=True)
    handle = (BASE / "orchestration.lock").open("a+b")
    locked = False
    try:
        if handle.seek(0, 2) == 0:
            handle.write(b"\0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def verify_prepared_dataset():
    """Read-only verification lets independent experiments reuse immutable prepared data."""
    manifest = read_json(PROCESSED / "manifest.json")
    seal = read_json(ROOT / "artifacts/dataset_integrity.json")
    if seal["data_fingerprint"] != manifest["fingerprint"]:
        raise ValueError("Dataset integrity manifest is stale; run the main prepare entry first")
    for name, digest in seal["files"].items():
        if file_hash(PROCESSED / name) != digest:
            raise ValueError(f"Prepared dataset checksum mismatch: {name}")
    for month in manifest["months"]:
        if file_hash(ROOT / "data/raw" / (month["month"] + ".txt")) != month["sha256"]:
            raise ValueError("Raw monthly data changed; run the main prepare entry first")
    return manifest


def save_csv(path, rows, fields=None):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_splits(docs, parameters, seed):
    """Stratify month x page, keeping existing global near-duplicate groups intact."""
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    folds = parameters["folds"]
    train_end = parameters["train_folds"]
    validation_end = train_end + parameters["validation_folds"]
    if not 0 < train_end < validation_end < folds:
        raise ValueError("Each partition must have at least one fold")
    labels = [f"{d['month']}/{d['page_label']}" for d in docs]
    # Seeded group IDs randomize ties without sklearn 1.7's shuffled count/group misalignment.
    group_ids = {d["group"]: fingerprint({"seed": seed, "group": d["group"]}) for d in docs}
    groups = [group_ids[d["group"]] for d in docs]
    if min(collections.Counter(labels).values()) < folds:
        raise ValueError("Insufficient month/page support for the requested stratification")
    result = {part: [] for part in ["train", "validation", "test"]}
    cv = StratifiedGroupKFold(n_splits=folds, shuffle=False)
    for fold, (_, held) in enumerate(cv.split(np.zeros(len(docs)), labels, groups)):
        part = "train" if fold < train_end else "validation" if fold < validation_end else "test"
        result[part].extend(docs[int(i)]["id"] for i in held)
    lookup = {d["id"]: d for d in docs}
    if len(lookup) != len(docs):
        raise ValueError("Duplicate document IDs")
    owner, assigned = {}, set()
    for part, ids in result.items():
        ids.sort()
        if assigned.intersection(ids):
            raise ValueError("Document partitions overlap")
        assigned.update(ids)
        for doc_id in ids:
            group = lookup[doc_id]["group"]
            if owner.setdefault(group, part) != part:
                raise ValueError("Near-duplicate group crosses partitions")
        if set(lookup[i]["page_label"] for i in ids) != set(LABELS):
            raise ValueError("Every partition must contain all 12 pages")
    if assigned != set(lookup):
        raise ValueError("Partitions do not exhaust the corpus")
    return result


def training_vocabulary(docs):
    counts = collections.Counter(t for d in docs for t in d["tokens"])
    if any(not t or any(c.isspace() for c in t) or t.startswith("__label__") or t == "</s>" for t in counts):
        raise ValueError("Unsafe token for fastText input/vector format")
    return counts


def export_initialization(wv, words, path, *, generate_oov):
    """Export only target TRAIN words; native fastText adds every .vec word to its dictionary."""
    import numpy as np
    rows, kinds = [], collections.Counter()
    for word in sorted(words):
        kind = "source_vocabulary" if word in wv.key_to_index else "generated_subword" if generate_oov else None
        if kind is None:
            kinds["random_initialization"] += 1
            continue
        vector = np.asarray(wv.get_vector(word), dtype=np.float32)
        if vector.shape != (wv.vector_size,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid source vector: {word}")
        if not np.any(vector):
            kinds["random_initialization"] += 1
            continue
        rows.append((word, vector))
        kinds[kind] += 1
    if not rows:
        raise ValueError("No usable pretrained training-word vectors")
    with Path(path).open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{len(rows)} {wv.vector_size}\n")
        for word, vector in rows:
            stream.write(word + " " + " ".join(format(float(x), ".9g") for x in vector) + "\n")
    probes = rows[::max(1, len(rows) // 32)][:32]
    return {"target_train_words": len(words), "exported_words": len(rows), "dimension": wv.vector_size,
            "kinds": dict(kinds), "probe_vectors": {w: v.tolist() for w, v in probes}}


def source_records(configuration):
    result = {}
    for method, relative in configuration["sources"].items():
        directory = ROOT / relative
        spec = read_json(directory / "spec.json")
        seal = read_json(directory / "seal.json")
        if spec["method"] != method or spec["smoke"] or spec["embedding"]["vector_size"] != configuration["native"]["dim"]:
            raise ValueError(f"Unsuitable pretrained model: {directory}")
        if seal["spec_fingerprint"] != fingerprint(spec):
            raise ValueError("Pretrained source spec has changed")
        hashes = {}
        for name in ["spec.json", "summary.json", *sorted(p.name for p in directory.glob("vectors.kv*"))]:
            actual = file_hash(directory / name)
            if seal["files"].get(name) != actual:
                raise ValueError(f"Pretrained source checksum mismatch: {directory / name}")
            hashes[name] = actual
        if "vectors.kv" not in hashes:
            raise ValueError("Missing pretrained KeyedVectors")
        result[method] = {"directory": relative, "files": hashes, "source_spec": spec,
                          "manifest_sha256": file_hash(ROOT / spec["manifest"])}
    if set(result) != {"word2vec", "fasttext"}:
        raise ValueError("Both Gensim word-vector sources are required")
    return result


def prepare():
    import yaml
    from gensim.models import KeyedVectors
    from .data import load_docs
    configuration = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    if configuration["native"]["minCount"] != 1 or configuration["native"]["minn"] != 0 or configuration["native"]["maxn"] != 0:
        raise ValueError("Transfer protocol requires minCount=1 and no native character subwords")
    verified = verify_prepared_dataset()
    docs = load_docs(months=configuration["months"])
    if {d["month"] for d in docs} != set(configuration["months"]):
        raise ValueError("Missing monthly data")
    if len(docs) != sum(m["kept_articles"] for m in verified["months"] if m["month"] in configuration["months"]):
        raise ValueError("Loaded document count differs from the verified manifest")
    sources = source_records(configuration)
    split = make_splits(docs, configuration["split"], configuration["split_seed"])
    dataset = {name: file_hash(PROCESSED / name) for name in ["manifest.json", *[m + ".jsonl" for m in configuration["months"]]]}
    protocol = {name: file_hash(ROOT / "gensim_lab" / name) for name in
                ["fasttext_transfer.py", "data.py", "common.py", "runtime.py", "integrity.py"]}
    plan = {"configuration": configuration, "runtime": runtime_settings(), "sources": sources,
            "dataset": dataset, "splits": split, "protocol": protocol, "packages": environment()["packages"]}
    directory = BASE / "experiments" / fingerprint(plan)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["plan_fingerprint"] != fingerprint(plan) or any(file_hash(directory / name) != digest for name, digest in manifest["files"].items()):
            raise ValueError("Prepared transfer cache changed; inspect files before rerunning")
        write_json(BASE / "current.json", {"directory": directory.relative_to(ROOT).as_posix()})
        return directory
    print(f"Prepare six months: {len(docs):,} articles; { {k: len(v) for k, v in split.items()} }", flush=True)
    write_json(directory / "plan.json", plan)
    write_json(directory / "splits.json", split)
    lookup = {d["id"]: d for d in docs}
    train = [lookup[i] for i in split["train"]]
    vocab = training_vocabulary(train)
    files = ["plan.json", "splits.json", "train.txt", "train_vocabulary.json", "split_counts.csv", "resource_context.json"]
    import psutil
    concurrent = [{"pid": p.pid, "command": p.info["cmdline"]} for p in psutil.process_iter(["cmdline"])
                  if p.pid != os.getpid() and any("gensim_lab.thucnews_validation" in a for a in p.info["cmdline"] or [])]
    write_json(directory / "resource_context.json", {"recorded": stamp(), "concurrent_thucnews_validation": concurrent,
               "timing_interpretation": "observed wall time; CPU is shared when another training job is active"})
    with (directory / "train.txt").open("w", encoding="utf-8", newline="\n") as stream:
        for d in train:
            stream.write(f"__label__p{d['page_label']} {' '.join(d['tokens'])}\n")
    write_json(directory / "train_vocabulary.json", dict(sorted(vocab.items())))
    rows = []
    for part, ids in split.items():
        counter = collections.Counter((lookup[i]["month"], lookup[i]["page_label"]) for i in ids)
        rows.extend({"partition": part, "month": month, "page_label": page, "documents": counter[month, page]}
                    for month in configuration["months"] for page in LABELS)
    save_csv(directory / "split_counts.csv", rows)
    for method, source in sources.items():
        print(f"Export {method} vectors for target training vocabulary only", flush=True)
        wv = KeyedVectors.load(str(ROOT / source["directory"] / "vectors.kv"))
        record = export_initialization(wv, vocab, directory / f"{method}.vec", generate_oov=method == "fasttext")
        record["token_coverage"] = {
            "source_vocabulary": sum(n for w, n in vocab.items() if w in wv.key_to_index) / sum(vocab.values()),
            "exported": sum(n for w, n in vocab.items() if w in wv.key_to_index or (method == "fasttext" and np_vector_nonzero(wv, w))) / sum(vocab.values()),
        }
        write_json(directory / f"{method}_initialization.json", record)
        files.extend([f"{method}.vec", f"{method}_initialization.json"])
        del wv
        gc.collect()
    write_json(manifest_path, {"created": stamp(), "plan_fingerprint": fingerprint(plan), "documents": len(docs),
                              "partition_counts": {k: len(v) for k, v in split.items()}, "near_duplicate_groups_isolated": True,
                              "files": {name: file_hash(directory / name) for name in files}})
    write_json(BASE / "current.json", {"directory": directory.relative_to(ROOT).as_posix()})
    return directory


def np_vector_nonzero(wv, word):
    import numpy as np
    return bool(np.any(wv.get_vector(word)))


def predict(model, docs):
    import numpy as np
    predictions, probabilities = [], []
    for doc in docs:
        labels, values = model.predict(" ".join(doc["tokens"]), k=1)
        label, score = labels[0].removeprefix("__label__p"), float(values[0])
        if label not in LABELS or not math.isfinite(score) or not 0 <= score <= 1.0001:
            raise ValueError("Invalid fastText prediction")
        predictions.append(label)
        probabilities.append(score)
    return predictions, np.asarray(probabilities)


def evaluate_metrics(docs, predictions):
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    truth = [d["page_label"] for d in docs]
    return {"n": len(docs), "accuracy": float(accuracy_score(truth, predictions)),
            **{f"{average}_f1": float(f1_score(truth, predictions, labels=LABELS, average=average, zero_division=0))
               for average in ["macro", "micro", "weighted"]},
            "per_class": classification_report(truth, predictions, labels=LABELS, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(truth, predictions, labels=LABELS).tolist()}


def save_evaluation(directory, part, docs, predictions, probabilities):
    result = evaluate_metrics(docs, predictions)
    write_json(directory / f"metrics_{part}.json", result)
    save_csv(directory / f"predictions_{part}.csv", ({"id": d["id"], "month": d["month"], "page_label": d["page_label"],
             "prediction": p, "probability": float(s)} for d, p, s in zip(docs, predictions, probabilities)))
    save_csv(directory / f"per_class_{part}.csv", ({"page_label": label, **result["per_class"][label]} for label in LABELS))
    save_csv(directory / f"confusion_{part}.csv", ({"true_page": label, **dict(zip(LABELS, row))}
             for label, row in zip(LABELS, result["confusion_matrix"])))
    return result


def worker(directory, phase):
    import fasttext
    import numpy as np
    from .data import load_docs
    from .common import PeakMemory
    spec = read_json(directory / "spec.json")
    experiment = ROOT / spec["experiment"]
    split = read_json(experiment / "splits.json")
    variant = spec["variant"]
    if phase == "train":
        train, validation = load_docs(split["train"]), load_docs(split["validation"])
        kwargs = dict(spec["native"], input=str(experiment / "train.txt"), seed=spec["seed"])
        initialization = None
        if variant != "scratch":
            kwargs["pretrainedVectors"] = str(experiment / f"{variant}.vec")
            record = read_json(experiment / f"{variant}_initialization.json")
            # Real installed-library probe: lr=0 preserves imported vectors exactly.
            probe = fasttext.train_supervised(**dict(kwargs, lr=0.0, epoch=1, verbose=0))
            errors = [float(np.max(np.abs(probe.get_word_vector(w) - np.asarray(v, dtype=np.float32))))
                      for w, v in record["probe_vectors"].items()]
            if any(errors):
                raise ValueError("Native fastText failed to import Gensim word vectors exactly")
            initialization = {"words_checked": len(errors), "max_absolute_difference": max(errors),
                              "installed_library_import_exact": True, "probe_lr": 0, "probe_epochs": 1}
            del probe
            gc.collect()
        print(f"Train {variant}: thread={kwargs['thread']}, epoch={kwargs['epoch']}, dim={kwargs['dim']}", flush=True)
        with PeakMemory() as memory:
            model = fasttext.train_supervised(**kwargs)
        vocabulary = set(read_json(experiment / "train_vocabulary.json")) | {"</s>"}
        if set(model.get_words()) != vocabulary:
            raise ValueError("Native classifier vocabulary differs from target train-only vocabulary")
        validation_predictions, validation_scores = predict(model, validation)
        train_predictions, train_scores = predict(model, train)
        model.save_model(str(directory / "native.bin"))
        reloaded = fasttext.load_model(str(directory / "native.bin"))
        checks = {}
        for part, docs, predictions, scores in [("train", train, train_predictions, train_scores),
                                               ("validation", validation, validation_predictions, validation_scores)]:
            pp, ss = predict(reloaded, docs)
            if predictions != pp or not np.array_equal(scores, ss):
                raise ValueError(f"Saved predictions changed: {part}")
            save_evaluation(directory, part, docs, predictions, scores)
            checks[part] = {"n": len(docs), "labels_exact": True, "probabilities_exact": True}
        write_json(directory / "save_load_check.json", checks)
        write_json(directory / "summary.json", {"variant": variant, "created": stamp(), "spec": spec,
                   "training_seconds": memory.elapsed, "peak_memory_bytes": memory.peak,
                   "model_bytes": (directory / "native.bin").stat().st_size, "vocabulary_size": len(vocabulary),
                   "thread": model.thread, "epoch": model.epoch, "dimension": model.get_dimension(),
                   "train_only_vocabulary_exact": True, "initialization_probe": initialization})
    else:
        test = load_docs(split["test"])
        model = fasttext.load_model(str(directory / "native.bin"))
        predictions, scores = predict(model, test)
        result = save_evaluation(directory, "test", test, predictions, scores)
        other = fasttext.load_model(str(directory / "native.bin"))
        pp, ss = predict(other, test)
        if pp != predictions or not np.array_equal(ss, scores):
            raise ValueError("Test predictions changed after reloading")
        checks = read_json(directory / "save_load_check.json")
        checks["test"] = {"n": len(test), "labels_exact": True, "probabilities_exact": True,
                          "comparison": "two independent native.bin loads"}
        write_json(directory / "save_load_check.json", checks)
        print(f"Test {variant}: Accuracy={result['accuracy']:.6f}, Macro-F1={result['macro_f1']:.6f}", flush=True)



def valid_run(directory, spec):
    seal_path = directory / "seal.json"
    if not seal_path.exists():
        return False
    seal = read_json(seal_path)
    return seal["spec_fingerprint"] == fingerprint(spec) and all((directory / name).exists() and file_hash(directory / name) == digest
             for name, digest in seal["files"].items())


def run_worker(directory, phase):
    with (directory / f"{phase}.log").open("w", encoding="utf-8") as log:
        result = subprocess.run([sys.executable, "-m", "gensim_lab.fasttext_transfer", "--worker", str(directory), "--phase", phase],
                                cwd=ROOT, env=worker_environment(), stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        write_json(BASE / "progress.json", {"status": "failed", "run": str(directory), "phase": phase, "returncode": result.returncode})
        raise RuntimeError(f"Transfer worker failed: {directory / (phase + '.log')}")


def train_all(experiment):
    plan = read_json(experiment / "plan.json")
    native = dict(plan["configuration"]["native"], thread=plan["runtime"]["training_threads"])
    specs = []
    for variant in VARIANTS:
        spec = {"variant": variant, "experiment": experiment.relative_to(ROOT).as_posix(),
                "plan_fingerprint": fingerprint(plan), "seed": plan["configuration"]["seed"], "native": native}
        directory = experiment / "runs" / variant
        directory.mkdir(parents=True, exist_ok=True)
        specs.append((directory, spec))
        if valid_run(directory, spec):
            print(f"Reuse {variant}", flush=True)
            continue
        write_json(directory / "spec.json", spec)
        write_json(BASE / "progress.json", {"status": "training", "variant": variant, "updated": stamp()})
        print(f"Start {variant}", flush=True)
        run_worker(directory, "train")
    best_directory, best_spec = max(specs, key=lambda item: read_json(item[0] / "metrics_validation.json")["macro_f1"])
    write_json(experiment / "selection.json", {"selected_variant": best_spec["variant"], "criterion": "validation macro_f1",
               "selected_before_test_evaluation": True, "updated": stamp(),
               "validation_scores": {spec["variant"]: read_json(directory / "metrics_validation.json")["macro_f1"] for directory, spec in specs}})
    for directory, spec in specs:
        if valid_run(directory, spec):
            continue
        write_json(BASE / "progress.json", {"status": "testing", "variant": spec["variant"], "updated": stamp()})
        run_worker(directory, "test")
        files = {p.name: file_hash(p) for p in directory.iterdir() if p.is_file() and p.name != "seal.json"}
        write_json(directory / "seal.json", {"spec_fingerprint": fingerprint(spec), "completed": stamp(), "files": files})
    write_json(BASE / "progress.json", {"status": "complete", "completed_variants": len(specs), "updated": stamp()})
    report(experiment)


def report(experiment):
    plan, manifest = read_json(experiment / "plan.json"), read_json(experiment / "manifest.json")
    selection = read_json(experiment / "selection.json")
    rows, metric_rows, coverage_rows, per_class_rows = [], [], [], []
    names = {"scratch": "随机初始化", "word2vec": "Gensim Word2Vec 初始化", "fasttext": "Gensim FastText 初始化"}
    for variant in VARIANTS:
        directory = experiment / "runs" / variant
        summary = read_json(directory / "summary.json")
        if not valid_run(directory, summary["spec"]):
            raise ValueError(f"Missing or changed run: {directory}")
        for part in ["train", "validation", "test"]:
            metric = read_json(directory / f"metrics_{part}.json")
            metric_rows.append({"variant": variant, "partition": part,
                                **{k: metric[k] for k in ["n", "accuracy", "macro_f1", "micro_f1", "weighted_f1"]}})
            per_class_rows.extend({"variant": variant, "partition": part, "page_label": label, **metric["per_class"][label]} for label in LABELS)
        validation, test = read_json(directory / "metrics_validation.json"), read_json(directory / "metrics_test.json")
        rows.append([names[variant], f"{validation['macro_f1']:.6f}", f"{test['accuracy']:.6f}", f"{test['macro_f1']:.6f}",
                     f"{test['weighted_f1']:.6f}", f"{summary['training_seconds']:.2f}"])
        if variant != "scratch":
            record = read_json(experiment / f"{variant}_initialization.json")
            coverage_rows.append([names[variant], record["target_train_words"], record["kinds"].get("source_vocabulary", 0),
                                  record["kinds"].get("generated_subword", 0), record["kinds"].get("random_initialization", 0),
                                  f"{record['token_coverage']['source_vocabulary']:.2%}"])
    save_csv(experiment / "metrics.csv", metric_rows)
    save_csv(experiment / "per_class.csv", per_class_rows)
    counts = manifest["partition_counts"]
    rel = "../" + experiment.relative_to(ROOT).as_posix()
    text = "# FastText 预训练向量分类实验\n\n"
    text += f"本次使用 9801–9806（1998 年 1–6 月）清洗后的全部 {manifest['documents']:,} 篇人民日报文章，预测 01–12 版面编号。版面 ID 是弱监督标签，不能解释为人工标注的主题。\n\n"
    text += "## 数据与协议\n\n"
    text += f"按月份 × 版面分层、按已有全语料近重复组隔离，以种子 42 做 10 折划分，其中 7 折训练、1 折验证、2 折测试。实际为训练 {counts['train']:,} 篇、验证 {counts['validation']:,} 篇、测试 {counts['test']:,} 篇。六个月均进入三种分区，这是同分布留出评价，不能据此声称未来月份泛化。\n\n"
    text += "复用 THUCNews 全量训练集的 Gensim Word2Vec/FastText（100 维、20 线程、821,609 篇训练文章），来源不包含人民日报留出集。原始词向量、分类器输入和依赖均记录 SHA-256。TF-IDF 为文档稀疏表示，无法通过 native fastText 的 pretrainedVectors 作为词向量初始化，本实验使用两个词向量模型。\n\n"
    text += "native fastText 通过 pretrainedVectors 读取 .vec，随后联合微调词向量与监督分类层。导出仅包含本次人民日报**训练集**词表：Word2Vec 未命中词随机初始化，Gensim FastText 可用已训练字符子词生成这些训练词的初值。Gensim 的原始字符桶和输出权重没有被转移；原生监督模型使用词及词 bigram，不启用新的字符子词。三组模型均核验词表恰为训练词表加 EOS。\n\n"
    text += f"固定参数：100 维、25 轮、学习率 0.1、minCount=1、wordNgrams=2、字符子词关闭、softmax、bucket=200000、{plan['runtime']['training_threads']} 个训练线程、{plan['runtime']['numerical_threads']} 个数值库线程。分类器顺序运行，各模型内部并行。没有在测试集调参；比较方案预先固定，按验证 Macro-F1 选择的是“{names[selection['selected_variant']]}”。本次只有种子 42 的一次运行，多线程训练不保证逐位确定性，不补造误差棒或显著性。\n\n"
    text += "## 留出结果\n\n" + markdown_table(["初始化", "验证 Macro-F1", "测试 Accuracy", "测试 Macro-F1", "测试 Weighted-F1", "训练秒数"], rows) + "\n\n"
    text += "三组分类器使用相同划分、词表约束与训练参数。预训练初始化是否改善本次分类，应以表中测试指标为准；不能仅凭训练集拟合判断。训练时间不含向量导出、零学习率导入探针与评估。\n\n"
    if read_json(experiment / "resource_context.json")["concurrent_thucnews_validation"]:
        text += "本次启动时另有 THUCNews 验证训练正在运行，CPU 资源共享；表中时间是本次实测墙钟时间，不能用来比较三组模型的独立计算效率。分类实验使用专用锁与独立索引，原有验证实验、源模型和公共数据只读。\n\n"
    text += "## 向量覆盖与保存加载核验\n\n" + markdown_table(["初始化", "目标训练词数", "源词表命中", "生成子词初值", "保留随机初值", "源词表 token 覆盖"], coverage_rows) + "\n\n"
    text += "对每个预训练方案，使用真实安装的 native fastText 做 lr=0 的导入探针，核验 32 个分散采样词与 Gensim 导出值 float32 完全一致。正式模型保存 native.bin，训练和验证的所有预测标签、概率与保存前逐值一致；测试结果通过两次独立载入逐值核验。各分区保存逐篇预测、逐类 PRF 和混淆矩阵。\n\n"
    text += f"## 资源与复现\n\n[完整配置](../configs/fasttext_transfer.yaml)、[方案与来源]({rel}/plan.json)、[固定划分]({rel}/splits.json)、[月份与版面分区计数]({rel}/split_counts.csv)、[全部整体指标]({rel}/metrics.csv)、[全部逐类指标]({rel}/per_class.csv)、[仅验证集选型记录]({rel}/selection.json)。\n\n"
    for variant in VARIANTS:
        run = f"{rel}/runs/{variant}"
        text += f"- {names[variant]}：[原生模型]({run}/native.bin)、[训练日志]({run}/train.log)、[测试预测]({run}/predictions_test.csv)、[测试混淆]({run}/confusion_test.csv)、[保存加载核验]({run}/save_load_check.json)"
        text += "。\n"
    text += "\n```powershell\nconda run -n Gensim python -m gensim_lab.fasttext_transfer --stage all\nconda run -n Gensim python -m gensim_lab.fasttext_transfer --stage report\n```\n\n"
    text += "配置、数据、源模型或源码变化会创建新的实验指纹目录；原实验和主项目索引保留。当前目录记录在 artifacts/fasttext_transfer/current.json，完成缓存须通过文件 SHA-256 才能复用。\n"
    REPORT.write_text(text, encoding="utf-8")
    print(f"Report: {REPORT}", flush=True)


def main():
    from .runtime import required_environment
    required = required_environment()
    if any(os.environ.get(key) != value for key, value in required.items()):
        raise SystemExit(subprocess.call([sys.executable, "-m", "gensim_lab.fasttext_transfer", *sys.argv[1:]],
                                        cwd=ROOT, env=worker_environment()))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["all", "prepare", "train", "report"], default="all")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--phase", choices=["train", "test"], default="train", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.phase)
        return
    with transfer_lock():
        BASE.mkdir(parents=True, exist_ok=True)
        if args.stage == "report":
            report(ROOT / read_json(BASE / "current.json")["directory"])
            return
        experiment = prepare()
        if args.stage in ["all", "train"]:
            train_all(experiment)


if __name__ == "__main__":
    main()
