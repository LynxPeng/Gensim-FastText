"""Export and verify every native Gensim array, dictionary and IDF entry."""
import numpy as np
from gensim_lab.array_checks import validate_array_roundtrip
from gensim_lab.common import file_hash, write_json
from gensim_lab.models import Bundle
from gensim_lab.reporting import records

for row in records():
    if row["method"] in ["majority", "native_fasttext"] or "source_run" in row["spec"] or not row["summary"]:
        continue
    directory = row["directory"]
    manifest = directory / "native_export.json"
    bundle_sha = file_hash(directory / "bundle.joblib")
    if manifest.exists():
        from gensim_lab.common import read_json
        previous = read_json(manifest)
        if (previous.get("verification_version") == 2 and previous.get("bundle_sha256") == bundle_sha
                and all((directory / f["name"]).exists() and file_hash(directory / f["name"]) == f["sha256"] for f in previous["files"])):
            continue
    bundle = Bundle.load(directory)
    rep = bundle.representation
    model = rep.tfidf if row["method"] == "tfidf" else rep.model
    target = directory / "representation.gensim"
    model.save(str(target))
    restored = type(model).load(str(target))
    checked_arrays = {}
    if hasattr(model, "wv"):
        for owner in ["", "wv", "dv"]:
            original_owner = getattr(model, owner, None) if owner else model
            restored_owner = getattr(restored, owner, None) if owner else restored
            if original_owner is None:
                continue
            if hasattr(original_owner, "key_to_index"):
                assert original_owner.key_to_index == restored_owner.key_to_index
            for name, value in vars(original_owner).items():
                if isinstance(value, np.ndarray):
                    other = getattr(restored_owner, name)
                    checked_arrays[(owner + "." if owner else "") + name] = validate_array_roundtrip(value, other)
            for name, value in getattr(original_owner, "expandos", {}).items():
                other = restored_owner.expandos[name]
                checked_arrays[(owner + "." if owner else "") + "expandos." + name] = validate_array_roundtrip(value, other)
            if getattr(original_owner, "buckets_word", None) is not None:
                before, after = original_owner.buckets_word, restored_owner.buckets_word
                assert len(before) == len(after)
                for original, restored_bucket in zip(before, after):
                    validate_array_roundtrip(original, restored_bucket)
        equal = True
    elif row["method"] == "tfidf":
        equal = model.idfs == restored.idfs
    else:
        checked_arrays = {"projection.u": validate_array_roundtrip(model.projection.u, restored.projection.u),
                          "projection.s": validate_array_roundtrip(model.projection.s, restored.projection.s)}
        equal = True
    assert equal
    rep.dictionary.save(str(directory / "dictionary.gensim"))
    from gensim.corpora import Dictionary
    dictionary = Dictionary.load(str(directory / "dictionary.gensim"))
    assert dictionary.token2id == rep.dictionary.token2id and dictionary.dfs == rep.dictionary.dfs
    assert dictionary.num_docs == rep.dictionary.num_docs
    if row["method"] != "tfidf":
        rep.tfidf.save(str(directory / "idf.gensim"))
        from gensim.models import TfidfModel
        assert TfidfModel.load(str(directory / "idf.gensim")).idfs == rep.tfidf.idfs
    files = sorted([*directory.glob("representation.gensim*"), *directory.glob("dictionary.gensim*"), *directory.glob("idf.gensim*")])
    write_json(directory / "native_export.json", {"run": row["run"], "verification_version": 2, "bundle_sha256": bundle_sha,
               "same_model_values_after_load": bool(equal), "checked_arrays": checked_arrays,
               "dictionary_and_idf_match": True,
               "files": [{"name": p.name, "bytes": p.stat().st_size, "sha256": file_hash(p)} for p in files],
               "note": "Full arrays, keys, dictionary document frequencies and IDFs checked; duplicate exports are not counted twice in model size."})
    print("EXPORTED", row["method"], row["run"], flush=True)
    del bundle, model, restored
