"""Validate completed artifacts and guard orchestration without changing training."""
from __future__ import annotations

import csv
import math
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import psutil

from .common import ARTIFACTS, PROCESSED, ROOT, code_hash, environment, file_hash, fingerprint, read_json, stamp, write_json


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def reload_check_complete(check, key, spec, part):
    """A successful sample check cannot stand in for a complete partition."""
    if not isinstance(check, dict):
        return False
    difference = check.get("max_score_difference")
    return (check.get("run") == key and check.get("part") == part
            and check.get("method") == spec["method"]
            and check.get("samples") == len(spec["splits"][part])
            and check.get("same_labels_and_scores") is True
            and isinstance(difference, (int, float)) and math.isfinite(difference)
            and 0 <= difference <= 1e-6)


@lru_cache(maxsize=2)
def _documents(data_fingerprint):
    from .data import load_docs
    return {d["id"]: d for d in load_docs()}


def spec_is_current(spec):
    return (spec.get("runtime_code") == code_hash()
            and spec.get("data_fingerprint") == read_json(PROCESSED / "manifest.json")["fingerprint"]
            and spec.get("packages") == environment()["packages"]
            and (spec.get("method") != "doc2vec" or spec.get("doc2vec_inference_seed") == "tokens_sha256_v1")
            and all(file_hash(ROOT / path) == digest for path, digest in spec.get("protocol_code", {}).items()))


def validate_splits(spec):
    docs = _documents(spec["data_fingerprint"])
    parts = spec["splits"]
    if set(parts) != {"train", "validation", "test"}:
        raise ValueError("Expected train/validation/test splits")
    effective = parts
    if "source_run" in spec:
        source = read_json(ARTIFACTS / "runs" / spec["source_run"] / "spec.json")
        if not spec_is_current(source) or source["method"] != spec["method"] or source["seed"] != spec["seed"]:
            raise ValueError("Frozen source does not match the current experiment")
        if parts["train"] or parts["validation"]:
            raise ValueError("Frozen evaluation cannot refit training/validation data")
        effective = {"train": sum(source["splits"].values(), []), "validation": [], "test": parts["test"]}
    else:
        if not parts["train"] or not parts["validation"]:
            raise ValueError("Training and validation must both be nonempty")
        labels = {docs[i]["page_label"] for i in parts["train"] if i in docs}
        if labels != {f"{i:02d}" for i in range(1, spec["classes"] + 1)}:
            raise ValueError("Every requested class must occur in the training set")
    ids, groups = {}, {}
    for part, values in effective.items():
        ids[part] = set(values)
        if len(values) != len(ids[part]):
            raise ValueError(f"Duplicate article IDs in {part}")
        if not ids[part].issubset(docs):
            raise ValueError(f"Unknown article IDs in {part}")
        if any(not 1 <= int(docs[i]["page_label"]) <= spec["classes"] for i in values):
            raise ValueError(f"Unexpected class in {part}")
        groups[part] = {docs[i]["group"] for i in values}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if ids[a] & ids[b] or groups[a] & groups[b]:
            raise ValueError(f"Articles or near-duplicate groups overlap: {a}/{b}")


def required_files(spec, part):
    names = ["spec.json", "summary.json", f"metrics_{part}.json", f"predictions_{part}.csv"]
    if "source_run" not in spec:
        names += ["bundle.joblib", "save_load_check.json", "embedding_diagnostics.json"]
        if spec["method"] == "native_fasttext":
            names += ["native.bin", "native_train.txt"]
    return names


def cache_problem(directory, spec, part="validation", check_hashes=True):
    """Return a concrete defect; existence of summary alone is insufficient."""
    directory = Path(directory)
    try:
        for name in required_files(spec, part):
            if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
                return f"Missing/empty {name}"
        if read_json(directory / "spec.json") != spec:
            return "Saved spec differs from requested spec"
        if directory.parent.name == "runs" and directory.name != fingerprint(spec)[:20]:
            return "Run directory does not match its spec fingerprint"
        summary = read_json(directory / "summary.json")
        values = read_json(directory / f"metrics_{part}.json")
        if not finite(summary) or not finite(values):
            return "Non-finite metrics or summary"
        if any(summary.get(k) != spec[k] for k in ["method", "classes", "seed"]):
            return "Summary does not match the requested model"
        if part == "validation" and summary.get("validation") != values:
            return "Summary validation metrics do not match metrics_validation.json"
        docs = _documents(spec["data_fingerprint"])
        expected = sorted(spec["splits"][part])
        with (directory / f"predictions_{part}.csv").open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames != ["id", "true_label", "predicted_label", "score", "score_type"]:
                return "Invalid prediction CSV header"
            predictions = list(reader)
        if [r["id"] for r in predictions] != expected or values.get("n") != len(expected):
            return "Prediction IDs/counts do not match the exact split"
        labels = {f"{i:02d}" for i in range(1, spec["classes"] + 1)}
        kind = "probability" if spec["method"] == "native_fasttext" else "constant" if spec["method"] == "majority" else "decision_margin"
        for row in predictions:
            score = float(row["score"])
            if row["true_label"] != docs[row["id"]]["page_label"] or row["predicted_label"] not in labels:
                return "Prediction labels do not match the dataset or class set"
            if row["score_type"] != kind or not math.isfinite(score):
                return "Invalid prediction score/type"
            # Native fastText adds a small epsilon to probabilities at prediction time.
            if kind == "probability" and not 0 <= score <= 1.00002:
                return "Native probability outside the supported numeric range"
        counts = [[0 for _ in labels] for _ in labels]
        for row in predictions:
            counts[int(row["true_label"]) - 1][int(row["predicted_label"]) - 1] += 1
        if values.get("confusion_matrix") != counts:
            return "Confusion matrix does not match the prediction CSV"
        correct = sum(counts[i][i] for i in range(len(labels)))
        accuracy = correct / len(predictions) if predictions else 0.0
        f1 = 0.0
        for i in range(len(labels)):
            denominator = sum(counts[i]) + sum(row[i] for row in counts)
            f1 += 2 * counts[i][i] / denominator if denominator else 0.0
        f1 /= len(labels)
        if (not math.isclose(float(values["accuracy"]), accuracy, abs_tol=1e-12)
                or not math.isclose(float(values["macro_f1"]), f1, abs_tol=1e-12)):
            return "Accuracy/Macro-F1 do not match the prediction CSV"
        if "source_run" not in spec:
            saved = read_json(directory / "save_load_check.json")
            if not saved["same_labels"] or saved["max_score_difference"] > 1e-6:
                return "Save/load check failed"
        seal_path = directory / "integrity.json"
        if check_hashes and seal_path.exists():
            seal = read_json(seal_path)
            if seal["spec_fingerprint"] != fingerprint(spec):
                return "Integrity record belongs to a different spec"
            for name in required_files(spec, part):
                if name not in seal["files"]:
                    return f"Integrity record is incomplete: {name}"
                if seal["files"][name] != file_hash(directory / name):
                    return f"Checksum mismatch: {name}"
        return None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return f"Invalid cache: {error}"


def seal_run(directory, spec):
    names = set(required_files(spec, "validation" if "source_run" not in spec else "test"))
    if (Path(directory) / "metrics_test.json").exists():
        names.update(required_files(spec, "test"))
    write_json(Path(directory) / "integrity.json", {
        "created": stamp(), "spec_fingerprint": fingerprint(spec),
        "files": {name: file_hash(Path(directory) / name) for name in sorted(names)},
    })


def ensure_dataset(force=False):
    from .data import prepare
    prepare(force=force)
    if force:
        _documents.cache_clear()
    manifest = read_json(PROCESSED / "manifest.json")
    names = ["manifest.json", "splits.json", "excluded.json", "anomalies.json", "near_duplicate_pairs.json"]
    names += [m["month"] + ".jsonl" for m in manifest["months"]]
    expected_months = {m["month"] + ".jsonl" for m in manifest["months"]}
    extra_months = {p.name for p in PROCESSED.glob("1998??.jsonl")} - expected_months
    if extra_months:
        raise ValueError(f"Unexpected prepared monthly files outside the manifest: {sorted(extra_months)}")
    seal_path = ARTIFACTS / "dataset_integrity.json"
    seal = read_json(seal_path) if seal_path.exists() else None
    broken = any(not (PROCESSED / name).exists() for name in names)
    if seal and not force and seal["data_fingerprint"] == manifest["fingerprint"]:
        broken |= any(not (PROCESSED / name).exists() or file_hash(PROCESSED / name) != digest
                      for name, digest in seal["files"].items())
    if broken:
        print("Prepared artifacts are incomplete or changed; rebuilding from raw data", flush=True)
        prepare(force=True)
        manifest = read_json(PROCESSED / "manifest.json")
        _documents.cache_clear()
    if not seal or force or broken or seal["data_fingerprint"] != manifest["fingerprint"]:
        docs = _documents(manifest["fingerprint"])
        if len(docs) != manifest["kept_total"]:
            raise ValueError("Prepared article count does not match the manifest")
        write_json(seal_path, {"created": stamp(), "data_fingerprint": manifest["fingerprint"],
                               "files": {name: file_hash(PROCESSED / name) for name in names}})


@contextmanager
def artifact_lock():
    """OS byte locks are released on process death; the lock file stays in place."""
    import json
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    path = ARTIFACTS / "orchestration.lock"
    owner = {"pid": os.getpid(), "process_created": psutil.Process().create_time(), "started": stamp()}
    handle = path.open("a+b")
    locked = False
    try:
        if path.stat().st_size == 0:
            handle.write(b"\x00")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            locked = True
        except OSError as error:
            raise RuntimeError(f"Another artifact-writing process owns {path}; do not run concurrent pipelines") from error
        # Allow an already running writer from the initial PID-file implementation to finish.
        handle.seek(0)
        old = handle.read()
        if old.startswith(b"{"):
            try:
                previous = json.loads(old)
                process = psutil.Process(previous["pid"])
                live = abs(process.create_time() - previous["process_created"]) < 0.01
            except psutil.NoSuchProcess:
                live = False
            except (OSError, ValueError, KeyError, psutil.AccessDenied):
                raise RuntimeError(f"Cannot verify artifact lock owner: {path}")
            if live:
                raise RuntimeError(f"Another artifact-writing process owns {path}; do not run concurrent pipelines")
        handle.seek(0)
        handle.truncate()
        handle.write(b"\x00" + json.dumps(owner).encode())
        handle.flush()
        yield
    finally:
        if locked:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()
