"""Versioned Doc2Vec inference that depends on tokens, never article metadata."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .models import Representation

VERSION = "tokens_sha256_v1"


def inference_identity(tokens):
    canonical = json.dumps(tokens, ensure_ascii=False, separators=(",", ":"))
    return "content-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ContentSeedRepresentation(Representation):
    def transform(self, docs):
        if self.method != "doc2vec":
            raise ValueError("Content-seeded inference adapter is only for Doc2Vec")
        # Original IDs remain in fit_ids and CSVs; these copies only seed inference.
        canonical_docs = [{**d, "id": inference_identity(d["tokens"])} for d in docs]
        matrix, coverage = super().transform(canonical_docs)
        known = np.array([any(t in self.model.wv.key_to_index for t in d["tokens"]) for d in docs])
        # With no known words, infer_vector only returns its random initialization.
        # Do not expose that untrained vector as usable textual evidence.
        matrix[~known] = 0
        coverage["zero_documents"] = int(np.sum(~np.any(matrix, axis=1)))
        return matrix, coverage
