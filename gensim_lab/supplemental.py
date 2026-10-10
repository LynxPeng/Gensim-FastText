"""Prespecified exploratory checks, separate from the original selected results."""
from __future__ import annotations

import os
import subprocess
import sys

# Set numerical process settings before importing NumPy and model code.
from .runtime import required_environment
_required = required_environment()
if __name__ == "__main__" and any(os.environ.get(k) != v for k, v in _required.items()):
    _env = os.environ.copy()
    _env.update(_required)
    raise SystemExit(subprocess.call([sys.executable, "-m", "gensim_lab.supplemental", *sys.argv[1:]], env=_env))

import argparse
import collections
import copy
import csv
import itertools

import numpy as np
import yaml

from .common import ARTIFACTS, PROCESSED, ROOT, config, file_hash, read_json, stamp, write_json
from .data import load_docs
from .integrity import artifact_lock, cache_problem, ensure_dataset, seal_run, validate_splits
from .models import Bundle, metrics
from .runner import base_spec, class_splits, execute, validated_selection, worker_environment


def settings():
    return yaml.safe_load((ROOT / "configs" / "supplemental.yaml").read_text(encoding="utf-8"))


def chronological_splits(docs, cfg):
    parts = {"train": [], "validation": [], "test": []}
    owners, excluded = {}, []
    for d in sorted(docs, key=lambda x: x["id"]):
        part = "train" if d["date"] <= cfg["chronological_train_end"] else "validation" if d["date"] <= cfg["chronological_validation_end"] else "test"
        owner = owners.setdefault(d["group"], part)
        if owner != part:
            excluded.append({"id": d["id"], "intended_split": part, "group_owner": owner})
        else:
            parts[part].append(d["id"])
    return parts, excluded


def evaluate_final(key):
    directory = ARTIFACTS / "runs" / key
    spec = read_json(directory / "spec.json")
    if cache_problem(directory, spec, "test") is None:
        return
    validate_splits(spec)
    result = subprocess.run([sys.executable, "-m", "gensim_lab.worker", str(directory), "--evaluate", "test"],
                            cwd=ROOT, env=worker_environment(), capture_output=True, text=True, encoding="utf-8")
    (directory / "evaluation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout.strip(), flush=True)
    if result.returncode:
        raise RuntimeError(result.stderr)
    problem = cache_problem(directory, spec, "test", check_hashes=False)
    if problem:
        raise RuntimeError(problem)
    seal_run(directory, spec)


def native_fit_diagnosis(key):
    directory = ARTIFACTS / "runs" / key
    spec = read_json(directory / "spec.json")
    path = directory / "supplemental_fit_diagnosis.json"
    if path.exists():
        return read_json(path)
    bundle = Bundle.load(directory)
    docs = load_docs(spec["splits"]["train"])
    predicted, _, _ = bundle.predict(docs)
    result = metrics([d["page_label"] for d in docs], predicted, bundle.labels)
    result["prediction_counts"] = dict(collections.Counter(predicted))
    write_json(path, result)
    return result


def train_all():
    ensure_dataset()
    cfg = settings()
    jan = load_docs(months=["199801"])
    original = validated_selection()
    result = {"started": stamp(), "config": cfg, "selection_basis": "January validation Macro-F1 only",
              "held_out_status": "Exploratory: original test results were already observed before this protocol.",
              "native_grid": [], "native_selected": [], "pooling": [], "chronological": [],
              "expanded_native": [], "label_controls": []}
    native_base = base_spec("native_fasttext", 12, 42, class_splits(12))
    for lr, epoch, ngrams in itertools.product(cfg["native_learning_rates"], cfg["native_epochs"], cfg["native_word_ngrams"]):
        spec = copy.deepcopy(native_base)
        spec["native"].update(lr=lr, epoch=epoch, wordNgrams=ngrams)
        key = execute(spec, {"profile": "supplement_native_grid", "lr": lr, "epoch": epoch, "wordNgrams": ngrams})
        value = read_json(ARTIFACTS / "runs" / key / "summary.json")["validation"]["macro_f1"]
        diagnosis = native_fit_diagnosis(key)
        result["native_grid"].append({"run": key, "lr": lr, "epoch": epoch, "wordNgrams": ngrams,
                                      "validation_macro_f1": value, "training_macro_f1": diagnosis["macro_f1"]})
        write_json(ARTIFACTS / "supplemental_progress.json", result)
    best = sorted(result["native_grid"], key=lambda r: (-r["validation_macro_f1"], r["epoch"], r["lr"], r["wordNgrams"]))[0]
    result["selected_native_config"] = best
    write_json(ARTIFACTS / "supplemental_selection.json", {"created": stamp(), "basis": result["selection_basis"],
                                                         "tie_break": "fewer epochs, lower lr, lower wordNgrams", **best})
    selected_spec = read_json(ARTIFACTS / "runs" / best["run"] / "spec.json")
    for seed in cfg["seeds"]:
        spec = copy.deepcopy(selected_spec)
        spec["seed"] = seed
        result["native_selected"].append(execute(spec, {"profile": "supplement_native_selected"}))
    fasttext_spec = read_json(ARTIFACTS / "runs" / original["fasttext"] / "spec.json")
    for pooling in ["mean", "tfidf"]:
        for seed in cfg["seeds"]:
            spec = copy.deepcopy(fasttext_spec)
            spec["seed"] = seed
            spec["embedding"]["pooling"] = pooling
            result["pooling"].append({"pooling": pooling, "seed": seed,
                                      "run": execute(spec, {"profile": "supplement_pooling", "pooling": pooling})})
    chrono, excluded = chronological_splits(jan, cfg)
    write_json(ARTIFACTS / "chronological_splits.json", {"splits": chrono, "excluded_groups": excluded, "config": cfg})
    result["chronological_counts"] = {part: len(ids) for part, ids in chrono.items()}
    for method in config()["methods"]:
        for seed in ([42] if method in ["majority", "tfidf"] else cfg["seeds"]):
            spec = base_spec(method, 12, seed, chrono)
            result["chronological"].append(execute(spec, {"profile": "supplement_chronological"}))
    temporal = read_json(ARTIFACTS / "temporal_splits.json")
    for seed in cfg["seeds"]:
        spec = copy.deepcopy(selected_spec)
        spec["seed"], spec["splits"] = seed, temporal
        result["expanded_native"].append(execute(spec, {"profile": "supplement_expanded_native"}))
    for method in ["tfidf", "fasttext", "native_fasttext"]:
        spec = copy.deepcopy(selected_spec if method == "native_fasttext" else fasttext_spec if method == "fasttext" else base_spec(method, 12, 42, class_splits(12)))
        spec["label_permutation_seed"] = cfg["label_permutation_seed"]
        path = "gensim_lab/supplemental_worker.py"
        spec["protocol_code"] = {path: file_hash(ROOT / path)}
        result["label_controls"].append(execute(spec, {"profile": "supplement_label_control"}))
    write_json(ARTIFACTS / "supplemental_progress.json", result)
    # Grid candidates not selected are never evaluated on the test set by this protocol.
    test_keys = result["native_selected"] + [r["run"] for r in result["pooling"]] + result["chronological"] + result["expanded_native"] + result["label_controls"]
    result["final_evaluation_runs"] = sorted(set(test_keys))
    for key in result["final_evaluation_runs"]:
        evaluate_final(key)
    result["completed"] = stamp()
    write_json(ARTIFACTS / "supplemental_results.json", result)
    print("Supplemental validation, repetitions, date splits and controls completed", flush=True)


def macro_from_confusion(confusion):
    confusion = np.asarray(confusion)
    diagonal = np.diagonal(confusion, axis1=-2, axis2=-1)
    denominator = confusion.sum(axis=-1) + confusion.sum(axis=-2)
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float), where=denominator != 0)
    return f1.mean(axis=-1)


def run_scales():
    ensure_dataset()
    previous = read_json(ARTIFACTS / "supplemental_results.json")
    assert "completed" in previous
    source = read_json(ARTIFACTS / "runs" / previous["selected_native_config"]["run"] / "spec.json")
    result = {"started": stamp(), "basis": "Fixed January K=12 validation-selected configuration; no retuning", "runs": []}
    for classes in [2, 4, 8]:
        for seed in settings()["seeds"]:
            spec = base_spec("native_fasttext", classes, seed, class_splits(classes))
            spec["native"] = copy.deepcopy(source["native"])
            key = execute(spec, {"profile": "supplement_native_scales"})
            result["runs"].append(key)
    for key in result["runs"]:
        evaluate_final(key)
    result["completed"] = stamp()
    write_json(ARTIFACTS / "supplemental_scales.json", result)
    print("Fixed-config native FastText 2/4/8-class transfer completed", flush=True)


def run_chronological_native():
    """Transfer fixed improved parameters; do not claim unseen-date selection."""
    ensure_dataset()
    previous = read_json(ARTIFACTS / "supplemental_results.json")
    assert "completed" in previous
    source = read_json(ARTIFACTS / "runs" / previous["selected_native_config"]["run"] / "spec.json")
    chrono, excluded = chronological_splits(load_docs(months=["199801"]), settings())
    validate_splits(base_spec("native_fasttext", 12, 42, chrono))
    result = {"started": stamp(), "basis": "Fixed January random-split validation-selected parameters; no retuning",
              "limitation": "Parameter selection on the original January split already used some later-date labels. This is a transfer sensitivity check, not prospective validation.",
              "splits": chrono, "excluded_groups": excluded, "runs": []}
    for seed in settings()["seeds"]:
        spec = base_spec("native_fasttext", 12, seed, chrono)
        spec["native"] = copy.deepcopy(source["native"])
        result["runs"].append(execute(spec, {"profile": "supplement_chronological_native_fixed"}))
    for key in result["runs"]:
        evaluate_final(key)
    baseline = [key for key in previous["chronological"]
                if read_json(ARTIFACTS / "runs" / key / "spec.json")["method"] == "native_fasttext"]
    result["training_macro_f1"] = {key: native_fit_diagnosis(key)["macro_f1"]
                                    for key in baseline + result["runs"]}
    result["completed"] = stamp()
    write_json(ARTIFACTS / "supplemental_chronological_native.json", result)
    print("Fixed-config native FastText chronological transfer completed", flush=True)


def clustered_bootstrap(models, docs, repeats=2000, seed=20261007, block_size=1):
    """Paired resampling of whole dates; do not resample individual articles."""
    saved = {}
    shared = None
    for name, keys in models.items():
        saved[name] = []
        for key in keys:
            path = ARTIFACTS / "runs" / key / "predictions_test.csv"
            with path.open(encoding="utf-8-sig") as f:
                prediction = {row["id"]: row for row in csv.DictReader(f)}
            shared = set(prediction) if shared is None else shared & set(prediction)
            saved[name].append(prediction)
    ids = sorted(shared)
    assert ids
    dates = sorted({docs[i]["date"] for i in ids})
    date_index = {value: i for i, value in enumerate(dates)}
    rng = np.random.default_rng(seed)
    if not 1 <= block_size <= len(dates):
        raise ValueError("Date block size must be between 1 and the number of observed dates")
    if block_size == 1:
        draws = rng.multinomial(len(dates), np.full(len(dates), 1 / len(dates)), size=repeats)
    else:
        starts = rng.integers(0, len(dates), size=(repeats, (len(dates) + block_size - 1) // block_size))
        indices = ((starts[..., None] + np.arange(block_size)) % len(dates)).reshape(repeats, -1)[:, :len(dates)]
        draws = np.zeros((repeats, len(dates)), dtype=np.int64)
        np.add.at(draws, (np.repeat(np.arange(repeats), len(dates)), indices.ravel()), 1)
    intervals, values = {}, {}
    for name, predictions in saved.items():
        counts = np.zeros((len(predictions), len(dates), 12, 12), dtype=np.int64)
        for s, prediction in enumerate(predictions):
            for i in ids:
                row = prediction[i]
                assert row["true_label"] == docs[i]["page_label"]
                counts[s, date_index[docs[i]["date"]], int(row["true_label"]) - 1, int(row["predicted_label"]) - 1] += 1
        point = float(macro_from_confusion(counts.sum(axis=1)).mean())
        sampled = np.einsum("bd,sdij->bsij", draws, counts, optimize=True)
        values[name] = macro_from_confusion(sampled).mean(axis=1)
        lo, hi = np.quantile(values[name], [0.025, 0.975])
        intervals[name] = {"macro_f1": point, "percentile_95": [float(lo), float(hi)]}
    baseline = "TF-IDF"
    differences = {}
    for name in values:
        if name == baseline:
            continue
        lo, hi = np.quantile(values[name] - values[baseline], [0.025, 0.975])
        differences[name] = {"minus": baseline, "difference": intervals[name]["macro_f1"] - intervals[baseline]["macro_f1"],
                             "percentile_95": [float(lo), float(hi)]}
    return {"articles": len(ids), "dates": len(dates), "replicates": repeats, "seed": seed,
            "sampling_unit": "date" if block_size == 1 else "circular consecutive observed-date blocks",
            "block_size": block_size, "models": models, "intervals": intervals, "paired_differences": differences}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["train", "scales", "chronological-native", "report", "all"], default="all")
    args = parser.parse_args()
    with artifact_lock():
        if args.stage in ["train", "all"]:
            train_all()
        if args.stage in ["scales", "all"]:
            run_scales()
        if args.stage in ["chronological-native", "all"]:
            run_chronological_native()
        if args.stage in ["report", "all"]:
            from .supplemental_report import report
            report()


if __name__ == "__main__":
    main()
