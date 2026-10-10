"""Check every main Doc2Vec validation vector under changed article IDs."""
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
from gensim_lab.common import ARTIFACTS, file_hash, read_json, stamp, write_json
from gensim_lab.content_seed import ContentSeedRepresentation
from gensim_lab.data import load_docs
from gensim_lab.models import Bundle
from gensim_lab.reporting import records, select

path = ARTIFACTS / "doc2vec_identity_probe_after.json"
previous = read_json(path).get("checks", []) if path.exists() else []
cached = {r["run"]: r for r in previous}
checks = []
tool_sha = file_hash(__file__)
for row in select(records(), "main", "doc2vec"):
    spec = row["spec"]
    model_sha = file_hash(row["directory"] / "bundle.joblib")
    old = cached.get(row["run"])
    if (old and old.get("model_sha256") == model_sha and old.get("tool_sha256") == tool_sha
            and old.get("data_fingerprint") == spec["data_fingerprint"]
            and old.get("samples") == len(spec["splits"]["validation"])
            and old.get("same_vectors_labels_scores_after_id_change") is True):
        checks.append(old)
        continue
    bundle = Bundle.load(row["directory"])
    assert isinstance(bundle.representation, ContentSeedRepresentation)
    docs = load_docs(spec["splits"]["validation"])
    changed = [{**d, "id": "different-id-" + d["id"]} for d in docs]
    x, before_coverage = bundle.representation.transform(docs)
    y, after_coverage = bundle.representation.transform(changed)
    equal = (np.array_equal(x, y) and before_coverage == after_coverage
             and np.array_equal(bundle.classifier.predict(x), bundle.classifier.predict(y))
             and np.array_equal(bundle.classifier.decision_function(x), bundle.classifier.decision_function(y)))
    assert equal, row["run"]
    checks.append({"run": row["run"], "samples": len(docs),
                   "same_vectors_labels_scores_after_id_change": bool(equal),
                   "model_sha256": model_sha, "tool_sha256": tool_sha,
                   "data_fingerprint": spec["data_fingerprint"]})
    print("DOC2VEC ID CHECK", row["run"], len(docs), flush=True)
    del bundle
assert len(checks) == 12
write_json(path, {"completed": stamp(), "checks": checks})
print("Doc2Vec identity invariance verified for all 12 main models")
