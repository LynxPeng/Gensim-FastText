"""Explicit shuffled-label control; validation/test labels are never shuffled."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np

from . import worker
from .common import read_json


def permute_training_labels(docs, seed):
    result = copy.deepcopy(docs)
    labels = np.random.default_rng(seed).permutation([d["page_label"] for d in docs])
    for doc, label in zip(result, labels):
        doc["page_label"] = str(label)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    spec = read_json(args.directory / "spec.json")
    assert "label_permutation_seed" in spec or "doc2vec_inference_seed" in spec
    if "doc2vec_inference_seed" in spec:
        from .content_seed import ContentSeedRepresentation, VERSION
        from . import models
        assert spec["method"] == "doc2vec" and spec["doc2vec_inference_seed"] == VERSION
        models.Representation = ContentSeedRepresentation
    original_loader = worker.load_docs
    def control_loader(ids=None, months=None):
        docs = original_loader(ids, months)
        return permute_training_labels(docs, spec["label_permutation_seed"]) if "label_permutation_seed" in spec and set(ids or []) == set(spec["splits"]["train"]) else docs
    worker.load_docs = control_loader
    if "label_permutation_seed" in spec:
        print(f"LABEL PERMUTATION CONTROL seed={spec['label_permutation_seed']}; held-out labels unchanged", flush=True)
    if "doc2vec_inference_seed" in spec:
        print(f"DOC2VEC INFERENCE PROTOCOL {spec['doc2vec_inference_seed']}", flush=True)
    worker.train_job(args.directory)


if __name__ == "__main__":
    main()
