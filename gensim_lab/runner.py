from __future__ import annotations

import copy
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ARTIFACTS, PROCESSED, ROOT, code_hash, config, environment, file_hash, fingerprint, read_json, stamp, write_json
from .data import load_docs
from .integrity import cache_problem, ensure_dataset, seal_run, spec_is_current, validate_splits


def index():
    path = ARTIFACTS / "index.json"
    return read_json(path) if path.exists() else {"schema_version": 1, "jobs": {}}


def worker_environment():
    from .runtime import worker_environment as configured_environment
    return configured_environment()


def base_spec(method, classes, seed, splits, smoke=False):
    cfg = config()
    spec = {"schema_version": 1, "method": method, "classes": classes, "seed": seed,
            "embedding": copy.deepcopy(cfg["embedding"]), "native": copy.deepcopy(cfg["native"]),
            "dictionary": copy.deepcopy(cfg["dictionary"]), "svm_c": cfg["svm_c"],
            "splits": splits, "data_fingerprint": read_json(PROCESSED / "manifest.json")["fingerprint"],
            "runtime_code": code_hash(), "packages": environment()["packages"]}
    if method == "doc2vec":
        from .content_seed import VERSION
        spec["doc2vec_inference_seed"] = VERSION
        spec["protocol_code"] = {path: file_hash(ROOT / path) for path in
                                 ["gensim_lab/content_seed.py", "gensim_lab/supplemental_worker.py"]}
    if smoke:
        spec["embedding"].update({"vector_size": 20, "epochs": 1, "bucket": 1000, "min_count": 1})
        spec["native"].update({"dim": 20, "epoch": 1, "bucket": 1000, "minCount": 1})
        spec["dictionary"].update({"no_below": 1})
        spec["svm_c"] = [1.0]
    return spec


def class_splits(classes, smoke=False):
    split = read_json(PROCESSED / "splits.json")
    docs = {d["id"]: d for d in load_docs(months=["199801"])}
    selected = {}
    for part, ids in split.items():
        values = [x for x in ids if int(docs[x]["page_label"]) <= classes]
        if smoke:
            cap = 32 if part == "train" else 8
            values = [x for label in range(1, classes + 1) for x in [i for i in values if int(docs[i]["page_label"]) == label][:cap]]
        selected[part] = values
    return selected


def register(spec, use):
    key = fingerprint(spec)[:20]
    directory = ARTIFACTS / "runs" / key
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / "spec.json").exists():
        write_json(directory / "spec.json", spec)
    state = index()
    job = state["jobs"].setdefault(key, {"uses": [], "method": spec["method"], "classes": spec["classes"], "seed": spec["seed"]})
    if use not in job["uses"]:
        job["uses"].append(use)
    state["updated"] = stamp()
    write_json(ARTIFACTS / "index.json", state)
    return key, directory


def execute(spec, use):
    if not spec_is_current(spec):
        raise ValueError("Refusing to train/reuse a spec from stale code, data or package versions")
    validate_splits(spec)
    key, directory = register(spec, use)
    problem = cache_problem(directory, spec)
    if problem is None:
        print(f"CACHE {use} {spec['method']} K={spec['classes']} seed={spec['seed']}", flush=True)
        return key
    if any(p.name != "spec.json" for p in directory.iterdir()):
        recovery = ARTIFACTS / "cache_recovery" / f"{key}-{time.time_ns()}"
        # Both resolved targets are checked before moving a checkout-local artifact directory.
        directory.resolve().relative_to((ARTIFACTS / "runs").resolve())
        recovery.resolve().relative_to((ARTIFACTS / "cache_recovery").resolve())
        recovery.parent.mkdir(parents=True, exist_ok=True)
        directory.rename(recovery)
        directory.mkdir()
        write_json(directory / "spec.json", spec)
        write_json(recovery / "recovery_reason.json", {"time": stamp(), "reason": problem})
        print(f"RECOVER {key}: {problem}; preserved in {recovery}", flush=True)
    print(f"START {use} {spec['method']} K={spec['classes']} seed={spec['seed']} {key}", flush=True)
    module = "gensim_lab.supplemental_worker" if "label_permutation_seed" in spec or "doc2vec_inference_seed" in spec else "gensim_lab.worker"
    result = subprocess.run([sys.executable, "-m", module, str(directory)], env=worker_environment(), cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    (directory / "console.log").write_text(result.stdout + result.stderr, encoding="utf-8")
    print(result.stdout.strip(), flush=True)
    if result.returncode:
        write_json(directory / "failure.json", {"returncode": result.returncode, "stderr": result.stderr, "time": stamp()})
        raise RuntimeError(f"Job {key} failed; see {directory / 'console.log'}\n{result.stderr[-4000:]}")
    problem = cache_problem(directory, spec, check_hashes=False)
    if problem:
        raise RuntimeError(f"Worker completed but artifacts are invalid: {problem}")
    seal_run(directory, spec)
    return key


def run_main(seeds, smoke=False):
    ensure_dataset()
    cfg = config()
    for classes in ([2] if smoke else cfg["classes"]):
        splits = class_splits(classes, smoke)
        for method in cfg["methods"]:
            for seed in ([42] if smoke or method in ["tfidf", "majority"] else seeds):
                execute(base_spec(method, classes, seed, splits, smoke), {"profile": "smoke" if smoke else "main"})
    if not smoke and (ARTIFACTS / "selection.json").exists():
        selection = validated_selection()
        for method in ["fasttext", "native_fasttext"]:
            source = read_json(ARTIFACTS / "runs" / selection[method] / "spec.json")
            for seed in seeds:
                spec = copy.deepcopy(source)
                spec["seed"] = seed
                execute(spec, {"profile": "selected", "selection_basis": "January validation Macro-F1"})


def run_ablations():
    ensure_dataset()
    cfg = config()
    baseline = base_spec("fasttext", 12, 42, class_splits(12))
    for axis, values in cfg["ablations"].items():
        for value in values:
            spec = copy.deepcopy(baseline)
            if axis == "native_word_ngrams":
                spec["method"] = "native_fasttext"
                spec["native"]["wordNgrams"] = value
            elif axis == "subwords":
                spec["embedding"]["min_n"], spec["embedding"]["max_n"] = value
            elif axis == "objective":
                spec["embedding"]["hs"] = int(value == "hs")
                spec["embedding"]["negative"] = 0 if value == "hs" else 10 if value == "ns10" else 5
            else:
                spec["embedding"][axis] = value
            profile = "performance" if axis in ["workers", "batch_words"] else "ablation"
            execute(spec, {"profile": profile, "axis": axis, "value": value})
    choose()


def choose():
    state = index()
    result = {"basis": "January validation Macro-F1 only", "created": stamp()}
    for method in ["fasttext", "native_fasttext"]:
        candidates = []
        for key, rec in state["jobs"].items():
            if rec["method"] != method or rec["classes"] != 12 or rec["seed"] != 42:
                continue
            if not any(u["profile"] in ["main", "ablation"] for u in rec["uses"]):
                continue
            path = ARTIFACTS / "runs" / key / "summary.json"
            if not path.exists() or not (path.parent / "spec.json").exists():
                continue
            spec = read_json(path.parent / "spec.json")
            if spec_is_current(spec) and cache_problem(path.parent, spec) is None:
                score = read_json(path)["validation"]["macro_f1"]
                candidates.append((score, key))
        if not candidates:
            raise ValueError(f"No completed candidates for {method}")
        candidates.sort(key=lambda x: (-x[0], x[1]))
        result[method] = candidates[0][1]
        result[method + "_validation_macro_f1"] = candidates[0][0]
    write_json(ARTIFACTS / "selection.json", result)
    print("SELECTED", result, flush=True)
    return result


def validated_selection():
    path = ARTIFACTS / "selection.json"
    if path.exists():
        try:
            selected = read_json(path)
            for method in ["fasttext", "native_fasttext"]:
                directory = ARTIFACTS / "runs" / selected[method]
                spec = read_json(directory / "spec.json")
                if (spec["method"] != method or spec["classes"] != 12 or spec["seed"] != 42
                        or not spec_is_current(spec) or cache_problem(directory, spec)):
                    break
                score = read_json(directory / "summary.json")["validation"]["macro_f1"]
                if score != selected[method + "_validation_macro_f1"]:
                    break
            else:
                return selected
        except (OSError, ValueError, KeyError, TypeError):
            pass
        print("Selection is stale or incomplete; recomputing from completed current validation runs", flush=True)
    return choose()


def run_temporal(seeds):
    ensure_dataset()
    selected = validated_selection()
    all_docs = load_docs()
    january_groups = {d["group"] for d in all_docs if d["month"] == "199801"}
    train = [d for d in all_docs if d["month"] in ["199801", "199802", "199803", "199804"]]
    train_groups = {d["group"] for d in train}
    val = [d for d in all_docs if d["month"] == "199805" and d["group"] not in train_groups]
    forbidden = train_groups | {d["group"] for d in val}
    test = [d for d in all_docs if d["month"] == "199806" and d["group"] not in forbidden]
    splits = {"train": [d["id"] for d in train], "validation": [d["id"] for d in val], "test": [d["id"] for d in test]}
    write_json(ARTIFACTS / "temporal_splits.json", splits)
    for method in ["tfidf", "fasttext", "native_fasttext"]:
        for seed in ([42] if method == "tfidf" else seeds):
            if method == "tfidf":
                source = base_spec(method, 12, seed, class_splits(12))
                source_key = execute(source, {"profile": "main"})
            else:
                source = read_json(ARTIFACTS / "runs" / selected[method] / "spec.json")
                source["seed"] = seed
                source_key = execute(source, {"profile": "selected"})
            expanded = copy.deepcopy(source)
            expanded["splits"] = splits
            execute(expanded, {"profile": "expanded"})
            for month in ["199802", "199803", "199804", "199805", "199806"]:
                ids = [d["id"] for d in all_docs if d["month"] == month and d["group"] not in january_groups]
                frozen = copy.deepcopy(source)
                frozen["source_run"] = source_key
                frozen["splits"] = {"train": [], "validation": [], "test": ids}
                register(frozen, {"profile": "frozen", "month": month})
    print("Temporal training complete; frozen and final test evaluation deferred to --finalize", flush=True)


def finalize():
    ensure_dataset()
    state = index()
    tasks = []
    for key, rec in state["jobs"].items():
        if any(u["profile"] in ["main", "selected", "expanded", "frozen"] for u in rec["uses"]):
            spec = read_json(ARTIFACTS / "runs" / key / "spec.json")
            if not spec_is_current(spec):
                continue
            tasks.append(key)
    write_json(ARTIFACTS / "finalization.json", {"started": stamp(), "jobs": tasks, "selection": read_json(ARTIFACTS / "selection.json") if (ARTIFACTS / "selection.json").exists() else None})
    for key in tasks:
        directory = ARTIFACTS / "runs" / key
        spec = read_json(directory / "spec.json")
        validate_splits(spec)
        source = ARTIFACTS / "runs" / spec["source_run"] if "source_run" in spec else directory
        source_spec = read_json(source / "spec.json")
        problem = cache_problem(source, source_spec)
        if problem:
            raise RuntimeError(f"Final evaluation source is incomplete: {source.name}: {problem}")
        if cache_problem(directory, spec, "test") is None:
            continue
        result = subprocess.run([sys.executable, "-m", "gensim_lab.worker", str(directory), "--evaluate", "test"], env=worker_environment(), cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        (directory / "evaluation.log").write_text(result.stdout + result.stderr, encoding="utf-8")
        print(result.stdout.strip(), flush=True)
        if result.returncode:
            raise RuntimeError(result.stderr)
        problem = cache_problem(directory, spec, "test", check_hashes=False)
        if problem:
            raise RuntimeError(f"Invalid final evaluation: {key}: {problem}")
        seal_run(directory, spec)
    write_json(ARTIFACTS / "finalization.json", {"completed": stamp(), "jobs": tasks, "selection": read_json(ARTIFACTS / "selection.json") if (ARTIFACTS / "selection.json").exists() else None})
