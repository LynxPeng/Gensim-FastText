"""Verify the public predict entry against saved models and invalid inputs."""
import os
import subprocess
import sys

from gensim_lab.runtime import required_environment
required = required_environment()
if any(os.environ.get(k) != v for k, v in required.items()):
    env = os.environ.copy()
    env.update(required)
    raise SystemExit(subprocess.call([sys.executable, *sys.argv], env=env))

import json
import math
import copy

from gensim_lab.common import ARTIFACTS, ROOT, file_hash, read_json, write_json
from gensim_lab.data import load_docs
from gensim_lab.models import Bundle
from gensim_lab.reporting import records, select


def invoke(directory, text):
    return subprocess.run([sys.executable, "-m", "gensim_lab", "predict", "--model", str(directory),
                           "--text", text, "--tokenized"], cwd=ROOT, capture_output=True,
                          text=True, encoding="utf-8")


rows = records()
tested = [r for r in select(rows, "main", classes=12) if r["seed"] == 42]
selected_path = ARTIFACTS / "supplemental_selection.json"
if selected_path.exists():
    selected = read_json(selected_path)["run"]
    tested += [r for r in rows if r["run"] == selected and r not in tested]
assert len(tested) in [7, 8]
docs = {d["id"]: d for d in load_docs()}
result = {"model_checks": [], "invalid_inputs": [], "doc2vec_oov": None, "legacy_doc2vec_rejected": False}
for row in tested:
    bundle = Bundle.load(row["directory"])
    source = docs[sorted(row["spec"]["splits"]["test"])[0]]
    tokens = source["tokens"][:80]
    expected, scores, coverage = bundle.predict([{"id": "public-predict-probe", "tokens": tokens}])
    actuals = []
    for separator in [" ", "  \t "]:
        process = invoke(row["directory"], separator.join(tokens))
        assert process.returncode == 0, process.stderr
        actual = json.loads(process.stdout)
        assert actual["page_label"] == expected[0]
        assert math.isclose(actual["score"], float(scores[0]), rel_tol=1e-6, abs_tol=1e-6)
        assert actual["coverage"] == coverage
        actuals.append(actual)
    assert actuals[0] == actuals[1]
    result["model_checks"].append({"run": row["run"], "method": row["method"],
        "model_sha256": file_hash(row["directory"] / "bundle.joblib"),
        "same_as_bundle_and_whitespace_variant": True, "output": actuals[0]})
    if row["method"] == "doc2vec":
        token = "unknown_cli_probe_a97e8bd1"
        assert token not in bundle.representation.model.wv.key_to_index
        process = invoke(row["directory"], token)
        assert process.returncode == 0, process.stderr
        actual = json.loads(process.stdout)
        assert actual["coverage"]["zero_documents"] == 1
        result["doc2vec_oov"] = actual
    del bundle

for text in ["   ", "！！！ ， 。"]:
    process = invoke(tested[0]["directory"], text)
    assert process.returncode == 2
    result["invalid_inputs"].append({"text": text, "exit_code": process.returncode})

old = [p.parent for p in sorted((ARTIFACTS / "runs").glob("*/spec.json"))
       if read_json(p).get("method") == "doc2vec" and not read_json(p).get("doc2vec_inference_seed")
       and (p.parent / "bundle.joblib").exists()]
if not old:
    # A clean reproduction has no historical runs. Make a separate malformed
    # metadata fixture to exercise the same public compatibility rejection.
    current = next(r for r in tested if r["method"] == "doc2vec")
    fixture = ROOT / ".artifact-work" / "cli-legacy-fixture"
    fixture.mkdir(parents=True, exist_ok=True)
    bundle = Bundle.load(current["directory"])
    bundle.spec = copy.deepcopy(bundle.spec)
    bundle.spec.pop("doc2vec_inference_seed")
    bundle.save(fixture)
    del bundle
    old = [fixture]
    result["legacy_fixture"] = "separate missing-protocol metadata fixture"
else:
    result["legacy_fixture"] = "historical model"
process = invoke(old[0], "经济 市场 改革 发展")
assert process.returncode == 2 and "Legacy Doc2Vec" in process.stderr
result["legacy_doc2vec_rejected"] = True
result["passed"] = True
write_json(ARTIFACTS / "cli_prediction_checks.json", result)
print(f"Public predict verified: {len(tested)} models, whitespace invariance, Doc2Vec OOV, empty/punctuation and legacy rejection")
