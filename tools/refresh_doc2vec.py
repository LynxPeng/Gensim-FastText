"""Rebuild only the ID-sensitive Doc2Vec runs under the corrected protocol."""
import os
import subprocess
import sys

from gensim_lab.runtime import required_environment
required = required_environment()
if any(os.environ.get(k) != v for k, v in required.items()):
    env = os.environ.copy()
    env.update(required)
    raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))

import numpy as np
from gensim_lab.common import ARTIFACTS, read_json, stamp, write_json
from gensim_lab.data import load_docs
from gensim_lab.integrity import artifact_lock, ensure_dataset
from gensim_lab.models import Bundle
from gensim_lab.runner import base_spec, class_splits, execute, finalize
from gensim_lab.supplemental import evaluate_final

with artifact_lock():
    ensure_dataset()
    fixed = []
    for classes in [2, 4, 8, 12]:
        for seed in [42, 2026, 7]:
            spec = base_spec("doc2vec", classes, seed, class_splits(classes))
            fixed.append(execute(spec, {"profile": "main"}))
    execute(base_spec("doc2vec", 2, 42, class_splits(2, True), True), {"profile": "smoke"})
    finalize()
    path = ARTIFACTS / "supplemental_results.json"
    result = read_json(path)
    refreshed, historical = [], []
    for key in result["chronological"]:
        old = read_json(ARTIFACTS / "runs" / key / "spec.json")
        if old["method"] == "doc2vec":
            spec = base_spec("doc2vec", 12, old["seed"], old["splits"])
            new_key = execute(spec, {"profile": "supplement_chronological"})
            evaluate_final(new_key)
            refreshed.append(new_key)
            historical.append(key)
        else:
            refreshed.append(key)
    result["chronological"] = refreshed
    result["previous_chronological_doc2vec_runs"] = historical
    result["doc2vec_correction_time"] = stamp()
    result["final_evaluation_runs"] = sorted(set(result["final_evaluation_runs"]) - set(historical) | set(refreshed))
    write_json(path, result)
    checks = []
    for key in fixed:
        directory = ARTIFACTS / "runs" / key
        spec = read_json(directory / "spec.json")
        bundle = Bundle.load(directory)
        docs = load_docs(spec["splits"]["validation"])
        changed = [{**d, "id": "different-id-" + d["id"]} for d in docs]
        x, _ = bundle.representation.transform(docs)
        y, _ = bundle.representation.transform(changed)
        labels, scores, _ = bundle.predict(docs)
        other, other_scores, _ = bundle.predict(changed)
        equal = np.array_equal(x, y) and labels == other and np.array_equal(scores, other_scores)
        assert equal, key
        checks.append({"run": key, "samples": len(docs), "same_vectors_labels_scores_after_id_change": bool(equal)})
    write_json(ARTIFACTS / "doc2vec_identity_probe_after.json", {"completed": stamp(), "checks": checks})
    print(f"Corrected Doc2Vec ID-invariance verified across {len(checks)} main runs", flush=True)
