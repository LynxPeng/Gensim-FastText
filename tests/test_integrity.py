import copy
import csv
import os
from pathlib import Path

import pytest

from gensim_lab import integrity
from gensim_lab.common import read_json, write_json


@pytest.mark.parametrize("change", [None, "sample_count", "part", "result", "nan", "method"])
def test_reload_evidence_requires_complete_correct_partition(change):
    spec = {"method": "tfidf", "splits": {"validation": ["a", "b"], "test": ["c"]}}
    check = {"run": "run1", "part": "validation", "method": "tfidf", "samples": 2,
             "same_labels_and_scores": True, "max_score_difference": 0.0}
    if change == "sample_count":
        check["samples"] = 1
    elif change == "part":
        check["part"] = "test"
    elif change == "result":
        check["same_labels_and_scores"] = False
    elif change == "nan":
        check["max_score_difference"] = float("nan")
    elif change == "method":
        check["method"] = "doc2vec"
    assert integrity.reload_check_complete(check, "run1", spec, "validation") is (change is None)


@pytest.fixture
def completed(tmp_path, monkeypatch):
    docs = {key: {"id": key, "page_label": label, "group": key}
            for key, label in [("a", "01"), ("b", "02"), ("c", "01"), ("d", "02"), ("e", "01"), ("f", "02")]}
    monkeypatch.setattr(integrity, "_documents", lambda value: docs)
    spec = {"method": "tfidf", "classes": 2, "seed": 42, "data_fingerprint": "data",
            "splits": {"train": ["a", "b"], "validation": ["c", "d"], "test": ["e", "f"]}}
    values = {"n": 2, "macro_f1": 1.0, "accuracy": 1.0, "confusion_matrix": [[1, 0], [0, 1]]}
    write_json(tmp_path / "spec.json", spec)
    write_json(tmp_path / "summary.json", {"method": "tfidf", "classes": 2, "seed": 42, "validation": values})
    write_json(tmp_path / "metrics_validation.json", values)
    write_json(tmp_path / "save_load_check.json", {"same_labels": True, "max_score_difference": 0.0})
    write_json(tmp_path / "embedding_diagnostics.json", {})
    (tmp_path / "bundle.joblib").write_bytes(b"model content")
    with (tmp_path / "predictions_validation.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", "predicted_label", "score", "score_type"])
        writer.writerows([["c", "01", "01", 0.5, "decision_margin"], ["d", "02", "02", 0.3, "decision_margin"]])
    return tmp_path, spec, docs


def test_summary_without_model_cannot_be_reused(completed):
    directory, spec, _ = completed
    (directory / "bundle.joblib").unlink()
    assert integrity.cache_problem(directory, spec) == "Missing/empty bundle.joblib"


def test_checksum_detects_model_corruption(completed):
    directory, spec, _ = completed
    integrity.seal_run(directory, spec)
    assert integrity.cache_problem(directory, spec) is None
    (directory / "bundle.joblib").write_bytes(b"damaged model")
    assert integrity.cache_problem(directory, spec) == "Checksum mismatch: bundle.joblib"


@pytest.mark.parametrize("change, expected", [("id", "IDs/counts"), ("truth", "labels"), ("nan", "score/type")])
def test_invalid_predictions_are_rejected(completed, change, expected):
    directory, spec, _ = completed
    path = directory / "predictions_validation.csv"
    content = path.read_text(encoding="utf-8-sig")
    content = content.replace("c,01,01", "e,01,01") if change == "id" else content.replace("c,01,01", "c,02,01") if change == "truth" else content.replace("0.5", "nan")
    path.write_text(content, encoding="utf-8-sig")
    assert expected in integrity.cache_problem(directory, spec)


def test_summary_metrics_must_match(completed):
    directory, spec, _ = completed
    write_json(directory / "metrics_validation.json", {"n": 2, "macro_f1": 0.0})
    assert "do not match" in integrity.cache_problem(directory, spec)


def test_matching_but_incorrect_summary_and_metrics_are_rejected(completed):
    directory, spec, _ = completed
    summary = read_json(directory / "summary.json")
    summary["validation"]["macro_f1"] = 0.2
    write_json(directory / "summary.json", summary)
    write_json(directory / "metrics_validation.json", summary["validation"])
    assert "Accuracy/Macro-F1" in integrity.cache_problem(directory, spec)


def test_duplicate_ids_and_near_groups_are_rejected(completed):
    _, spec, docs = completed
    integrity.validate_splits(spec)
    duplicate = copy.deepcopy(spec)
    duplicate["splits"]["validation"].append("c")
    with pytest.raises(ValueError, match="Duplicate"):
        integrity.validate_splits(duplicate)
    docs["c"]["group"] = "a"
    with pytest.raises(ValueError, match="overlap"):
        integrity.validate_splits(spec)


def test_native_cache_requires_binary_and_training_input(completed):
    directory, spec, _ = completed
    spec["method"] = "native_fasttext"
    assert integrity.cache_problem(directory, spec) == "Missing/empty native.bin"


def test_stale_spec_is_rejected_before_register(monkeypatch):
    from gensim_lab import runner
    monkeypatch.setattr(runner, "spec_is_current", lambda spec: False)
    with pytest.raises(ValueError, match="stale"):
        runner.execute({}, {"profile": "main"})


def test_incomplete_cache_is_preserved_rebuilt_and_then_reused(completed, monkeypatch, tmp_path):
    import shutil
    from types import SimpleNamespace
    from gensim_lab import runner
    from gensim_lab.common import fingerprint
    template, spec, _ = completed
    artifacts = tmp_path / "artifacts"
    monkeypatch.setattr(runner, "ARTIFACTS", artifacts)
    monkeypatch.setattr(runner, "spec_is_current", lambda value: True)
    monkeypatch.setattr(runner, "validate_splits", lambda value: None)
    key = fingerprint(spec)[:20]
    directory = artifacts / "runs" / key
    directory.mkdir(parents=True)
    files = [path for path in template.iterdir() if path.is_file()]
    for path in files:
        shutil.copy2(path, directory / path.name)
    (directory / "bundle.joblib").unlink()
    calls = []
    def worker(*args, **kwargs):
        calls.append(args)
        for path in files:
            shutil.copy2(path, directory / path.name)
        return SimpleNamespace(returncode=0, stdout="rebuilt", stderr="")
    monkeypatch.setattr(runner.subprocess, "run", worker)
    assert runner.execute(spec, {"profile": "main"}) == key
    recovered = list((artifacts / "cache_recovery").iterdir())
    assert len(recovered) == 1
    assert read_json(recovered[0] / "recovery_reason.json")["reason"] == "Missing/empty bundle.joblib"
    assert (recovered[0] / "summary.json").exists()
    assert integrity.cache_problem(directory, spec) is None
    assert runner.execute(spec, {"profile": "main"}) == key and len(calls) == 1


def test_selection_skips_missing_run_without_crashing(monkeypatch, tmp_path):
    from gensim_lab import runner
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path)
    monkeypatch.setattr(runner, "index", lambda: {"jobs": {"missing": {
        "method": "fasttext", "classes": 12, "seed": 42, "uses": [{"profile": "main"}]}}})
    with pytest.raises(ValueError, match="No completed candidates"):
        runner.choose()


def test_stale_selection_is_recomputed(monkeypatch, tmp_path):
    from gensim_lab import runner
    monkeypatch.setattr(runner, "ARTIFACTS", tmp_path)
    write_json(tmp_path / "selection.json", {"fasttext": "missing"})
    expected = {"new": "validation-only selection"}
    monkeypatch.setattr(runner, "choose", lambda: expected)
    assert runner.validated_selection() == expected


def test_live_writer_lock_and_release(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity, "ARTIFACTS", tmp_path)
    with integrity.artifact_lock():
        assert (tmp_path / "orchestration.lock").exists()
        with pytest.raises(RuntimeError, match="Another"):
            with integrity.artifact_lock():
                pass
    # Keeping the file avoids an unlink/recreate race; the OS lock is released.
    with integrity.artifact_lock():
        pass


def test_dead_writer_lock_is_reclaimed(monkeypatch, tmp_path):
    monkeypatch.setattr(integrity, "ARTIFACTS", tmp_path)
    write_json(tmp_path / "orchestration.lock", {"pid": 999999999, "process_created": 0})
    with integrity.artifact_lock():
        import json
        with (tmp_path / "orchestration.lock").open("rb") as handle:
            handle.seek(1)  # Windows correctly prohibits another handle reading the locked first byte.
            assert json.loads(handle.read())["pid"] == os.getpid()


def test_os_lock_is_released_if_owner_process_dies(monkeypatch, tmp_path):
    import subprocess
    import sys
    monkeypatch.setattr(integrity, "ARTIFACTS", tmp_path)
    code = "from pathlib import Path; import sys,time; from gensim_lab import integrity; integrity.ARTIFACTS=Path(sys.argv[1]); lock=integrity.artifact_lock(); lock.__enter__(); print('LOCKED',flush=True); time.sleep(30)"
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "LOCKED"
        with pytest.raises(RuntimeError, match="Another"):
            with integrity.artifact_lock():
                pass
        child.terminate()
        child.wait(timeout=10)
        with integrity.artifact_lock():
            pass
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=10)


def test_processed_corruption_is_rebuilt(monkeypatch, tmp_path):
    from gensim_lab import data
    processed, artifacts = tmp_path / "processed", tmp_path / "artifacts"
    processed.mkdir()
    manifest = {"fingerprint": "sample", "months": [{"month": "199801"}], "kept_total": 1}
    names = ["splits.json", "excluded.json", "anomalies.json", "near_duplicate_pairs.json"]
    original = '{"id":"a","tokens":["原始"]}\n'
    calls = []
    def prepare(force=False):
        calls.append(force)
        if force or not (processed / "manifest.json").exists():
            write_json(processed / "manifest.json", manifest)
            for name in names:
                write_json(processed / name, [])
            (processed / "199801.jsonl").write_text(original, encoding="utf-8")
    monkeypatch.setattr(data, "prepare", prepare)
    monkeypatch.setattr(data, "PROCESSED", processed)
    monkeypatch.setattr(integrity, "PROCESSED", processed)
    monkeypatch.setattr(integrity, "ARTIFACTS", artifacts)
    integrity._documents.cache_clear()
    integrity.ensure_dataset()
    (processed / "199801.jsonl").write_text('{"id":"a","tokens":["篡改"]}\n', encoding="utf-8")
    integrity.ensure_dataset()
    assert calls == [False, False, True]
    assert (processed / "199801.jsonl").read_text(encoding="utf-8") == original
    (processed / "199802.jsonl").write_text('{"id":"unexpected"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Unexpected prepared"):
        integrity.ensure_dataset()
    integrity._documents.cache_clear()


def test_cli_fixes_threads_even_if_hash_seed_already_set(monkeypatch):
    import subprocess
    import sys
    from gensim_lab import cli
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.setattr(sys, "argv", ["gensim-lab", "report"])
    captured = {}
    def launch(command, env):
        captured.update(env)
        return 0
    monkeypatch.setattr(subprocess, "call", launch)
    with pytest.raises(SystemExit):
        cli.main()
    from gensim_lab.runtime import numerical_threads
    assert captured["PYTHONHASHSEED"] == "0" and captured["OMP_NUM_THREADS"] == str(numerical_threads())
    assert captured["OPENBLAS_NUM_THREADS"] == str(numerical_threads()) and captured["NUMEXPR_NUM_THREADS"] == str(numerical_threads())
