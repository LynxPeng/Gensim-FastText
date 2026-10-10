"""Compare the project's document_vector pooling with native fastText's supervised sentence representation.

native fastText averages the input vectors of every token (unweighted over tokens).
paper_worker.document_vector averages unique tokens weighted by frequency, then L2-normalizes.
The two means are algebraically identical; this checks it numerically on a real run and isolates
the effect of the extra L2 normalization.
Run: conda run -n Gensim python tools/audit_pooling_equivalence.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from gensim.models import FastText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.paper_data import rows
from gensim_lab.paper_worker import document_vector

OUT = ROOT / "artifacts/reviews/pooling_equivalence.json"


def main():
    candidates = []
    for directory in sorted((ROOT / "artifacts/paper_reproduction/runs").glob("*")):
        spec = directory / "spec.json"
        if (directory / "representation.gensim").exists() and spec.exists():
            record = json.loads(spec.read_text(encoding="utf-8"))
            if record.get("method") == "gensim_fasttext" and not record.get("smoke"):
                candidates.append((directory, record))
    if not candidates:
        raise SystemExit("no completed gensim representation found")
    directory, spec = candidates[0]
    dimension = spec["dimension"]
    model = FastText.load(str(directory / "representation.gensim"))
    print(f"run={directory.name} dataset={spec['dataset']} dimension={dimension} "
          f"vectors={tuple(model.wv.vectors.shape)}")

    max_pooling_error = max_norm_error = 0.0
    compared = 0
    for index, _, words in rows(_manifest_text(spec)):
        if index >= 200:
            break
        project, _, _ = document_vector(model, words, dimension)
        token_mean = np.zeros(dimension, np.float32)
        counted = 0
        for token in words:
            value_index = model.wv.key_to_index.get(token)
            if value_index is not None:
                token_mean += model.wv.vectors[value_index]
                counted += 1
            else:
                try:
                    token_mean += model.wv.get_vector(token)
                    counted += 1
                except KeyError:
                    pass
        if not counted:
            continue
        token_mean /= counted
        max_pooling_error = max(max_pooling_error, float(np.max(np.abs(project - token_mean / np.linalg.norm(token_mean)))))
        max_norm_error = max(max_norm_error, abs(float(np.linalg.norm(project)) - 1.0))
        compared += 1

    result = {
        "run": directory.name, "dataset": spec["dataset"], "dimension": dimension,
        "gensim_vocabulary_size": len(model.wv),
        "documents_compared": compared,
        "max_abs_error_project_vs_l2_normalized_token_mean": max_pooling_error,
        "max_abs_deviation_of_project_l2_norm_from_one": max_norm_error,
        "conclusion": ("document_vector equals the L2-normalized token-level mean, i.e. the same pooling "
                       "as native fastText's hidden layer plus an extra unit-norm step"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


def _manifest_text(spec):
    manifest = json.loads((ROOT / spec["manifest"]).read_text(encoding="utf-8"))
    entry = manifest["partitions"]["test"]
    return str(ROOT / entry["path"]) if "path" in entry else str(ROOT / manifest["directory"] / entry["file"])


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
