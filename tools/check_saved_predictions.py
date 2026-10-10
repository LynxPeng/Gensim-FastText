"""Check complete validation/test predictions, including frozen source models."""
import os
import subprocess
import sys
from gensim_lab.runtime import required_environment
required = required_environment()
if any(os.environ.get(k) != v for k, v in required.items()):
    env = os.environ.copy()
    env.update(required)
    raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))
import csv
import numpy as np
from gensim_lab.common import ARTIFACTS, read_json, write_json, file_hash
from gensim_lab.data import load_docs
from gensim_lab.models import Bundle
from gensim_lab.integrity import reload_check_complete
from gensim_lab.reporting import records

all_docs = {d["id"]: d for d in load_docs()}
checks = []
previous = read_json(ARTIFACTS / "prediction_reload_checks.json") if (ARTIFACTS / "prediction_reload_checks.json").exists() else []
cache = {(r["run"], r.get("part", "validation")): r for r in previous}
for r in records():
    directory = r["directory"]
    source = ARTIFACTS / "runs" / r["spec"]["source_run"] if "source_run" in r["spec"] else directory
    model_hash = file_hash(source / "bundle.joblib")
    native_hash = file_hash(source / "native.bin") if r["method"] == "native_fasttext" else None
    bundle = None
    for part in ["validation", "test"]:
        csv_path = directory / f"predictions_{part}.csv"
        if not csv_path.exists():
            continue
        csv_hash = file_hash(csv_path)
        old = cache.get((r["run"], part))
        if (reload_check_complete(old, r["run"], r["spec"], part)
                and old["model_sha256"] == model_hash and old["csv_sha256"] == csv_hash
                and old.get("native_sha256") == native_hash):
            checks.append({**old, "part": part})
            continue
        with csv_path.open(encoding="utf-8-sig") as f:
            saved = list(csv.DictReader(f))
        assert [p["id"] for p in saved] == sorted(r["spec"]["splits"][part])
        assert all(p["true_label"] == all_docs[p["id"]]["page_label"] for p in saved)
        if bundle is None:
            bundle = Bundle.load(source)
        docs = [all_docs[p["id"]] for p in saved]
        predicted, scores, _ = bundle.predict(docs)
        original = [p["predicted_label"] for p in saved]
        original_scores = np.asarray([float(p["score"]) for p in saved])
        difference = float(np.max(np.abs(np.asarray(scores) - original_scores))) if saved else 0.0
        equal = predicted == original and np.allclose(scores, original_scores, rtol=1e-6, atol=1e-6)
        assert equal, (r["run"], part)
        checks.append({"run": r["run"], "part": part, "method": r["method"], "samples": len(saved), "same_labels_and_scores": bool(equal), "max_score_difference": difference, "model_sha256": model_hash, "native_sha256": native_hash, "csv_sha256": csv_hash})
        print("RELOAD", r["run"], part, len(saved), difference, flush=True)
    del bundle
write_json(ARTIFACTS / "prediction_reload_checks.json", checks)
print(f"Verified {len(checks)} complete prediction files across {len({r['run'] for r in checks})} runs")
