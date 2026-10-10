from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path

import numpy as np

from .common import ARTIFACTS, PeakMemory, environment, read_json, stamp, write_json
from .data import load_docs
from .diagnostics import embedding_diagnostics
from .models import Bundle, metrics


def evaluate(bundle, docs, part, directory):
    begin = time.perf_counter()
    predicted, scores, coverage = bundle.predict(docs)
    values = metrics([d["page_label"] for d in docs], predicted, bundle.labels)
    values.update({"coverage": coverage, "evaluation_seconds": time.perf_counter() - begin})
    with (directory / f"predictions_{part}.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "true_label", "predicted_label", "score", "score_type"])
        kind = "probability" if bundle.method == "native_fasttext" else "constant" if bundle.method == "majority" else "decision_margin"
        for doc, label, score in zip(docs, predicted, scores):
            writer.writerow([doc["id"], doc["page_label"], label, score, kind])
    write_json(directory / f"metrics_{part}.json", values)
    return values


def train_job(directory):
    spec = read_json(directory / "spec.json")
    logging.basicConfig(filename=str(directory / "training.log"), level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    with PeakMemory() as monitor:
        train = load_docs(spec["splits"]["train"])
        validation = load_docs(spec["splits"]["validation"])
        # This worker never reads its test documents before finalization.
        assert set(spec["splits"]["train"]).isdisjoint(spec["splits"]["validation"])
        bundle = Bundle(spec).fit(train, validation, directory)
        if bundle.representation is not None:
            assert set(bundle.representation.fit_ids) == set(spec["splits"]["train"])
            assert set(bundle.representation.fit_ids).isdisjoint(spec["splits"]["test"])
        write_json(directory / "embedding_diagnostics.json", embedding_diagnostics(bundle))
        result = evaluate(bundle, validation, "validation", directory)
        before, scores, _ = bundle.predict(validation[:8])
        bundle.save(directory)
        loaded = Bundle.load(directory)
        after, other_scores, _ = loaded.predict(validation[:8])
        if before != after or not np.allclose(scores, other_scores, rtol=1e-6, atol=1e-6):
            raise AssertionError("Save/load prediction mismatch")
        verification = {"samples": len(before), "same_labels": before == after, "max_score_difference": float(np.max(np.abs(np.asarray(scores) - other_scores))) if scores else 0.0}
        write_json(directory / "save_load_check.json", verification)
    size = sum(p.stat().st_size for p in directory.glob("*") if p.name.endswith((".joblib", ".bin", ".npy")))
    summary = {"completed": stamp(), "method": spec["method"], "classes": spec["classes"], "seed": spec["seed"], "validation": result, "tuning": bundle.tuning, "fit_seconds": bundle.fit_seconds, "wall_seconds": monitor.elapsed, "peak_rss_mb": monitor.peak / 2**20, "model_bytes": size, "save_load": verification, "environment": environment()}
    write_json(directory / "summary.json", summary)
    print(f"DONE {directory.name} val_macro_f1={result['macro_f1']:.4f} seconds={monitor.elapsed:.1f}", flush=True)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--evaluate", choices=["test"])
    args = parser.parse_args()
    if args.evaluate:
        spec = read_json(args.directory / "spec.json")
        source = ARTIFACTS / "runs" / spec["source_run"] if "source_run" in spec else args.directory
        with PeakMemory() as monitor:
            bundle = Bundle.load(source)
            values = evaluate(bundle, load_docs(spec["splits"]["test"]), "test", args.directory)
        if "source_run" in spec:
            write_json(args.directory / "summary.json", {"completed": stamp(), "method": spec["method"], "classes": spec["classes"], "seed": spec["seed"], "source_run": spec["source_run"], "fit_seconds": 0.0, "wall_seconds": monitor.elapsed, "peak_rss_mb": monitor.peak / 2**20, "model_bytes": 0, "environment": environment()})
        print(f"TEST {args.directory.name} macro_f1={values['macro_f1']:.4f}", flush=True)
    else:
        train_job(args.directory)


if __name__ == "__main__":
    main()
