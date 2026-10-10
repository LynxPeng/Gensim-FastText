from __future__ import annotations

import numpy as np
from gensim.models.fasttext_inner import compute_ngrams_bytes, ft_hash_bytes


def embedding_diagnostics(bundle):
    """Inspect trained arrays and hash collisions without looking at held-out text."""
    if bundle.representation is None or bundle.representation.model is None:
        return {}
    model = bundle.representation.model
    if not hasattr(model, "wv"):
        return {}
    wv = model.wv
    arrays = {}
    for owner_name, owner in [("model", model), ("wv", wv)]:
        for name in ["vectors", "vectors_vocab", "vectors_ngrams", "syn1", "syn1neg", "cum_table"]:
            value = getattr(owner, name, None)
            if isinstance(value, np.ndarray):
                arrays[f"{owner_name}.{name}"] = {"shape": list(value.shape), "dtype": str(value.dtype), "bytes": value.nbytes}
    result = {"vocabulary_size": len(wv), "arrays": arrays}
    if bundle.method != "fasttext":
        return result
    p = bundle.spec["embedding"]
    grams = set()
    if p["max_n"] > 0:
        for word in wv.index_to_key:
            grams.update(compute_ngrams_bytes(word, p["min_n"], p["max_n"]))
    occupied = {ft_hash_bytes(g) % p["bucket"] for g in grams}
    result.update({"unique_subwords": len(grams), "occupied_buckets": len(occupied),
                   "collision_excess": len(grams) - len(occupied),
                   "collision_fraction": (len(grams) - len(occupied)) / len(grams) if grams else 0.0,
                   "bucket": p["bucket"], "collision_definition": "1 - occupied buckets / distinct UTF-8 subwords"})
    probe = "未登录词核验甲乙丙丁"
    while probe in wv.key_to_index:
        probe += "丁"
    try:
        vector = wv.get_vector(probe)
        ids = [ft_hash_bytes(g) % p["bucket"] for g in compute_ngrams_bytes(probe, p["min_n"], p["max_n"])] if grams else []
        manual = np.mean(wv.vectors_ngrams[ids], axis=0) if ids else np.zeros(wv.vector_size)
        result["oov_probe"] = {"word": probe, "in_vocabulary": False, "subwords": len(ids),
                               "norm": float(np.linalg.norm(vector)),
                               "manual_max_error": float(np.max(np.abs(vector - manual)))}
    except KeyError:
        result["oov_probe"] = {"word": probe, "in_vocabulary": False, "get_vector": "KeyError (subwords disabled)"}
    return result
