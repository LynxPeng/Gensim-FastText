"""Version-bounded FastText engineering audit; writes only its own review directory.

Run with the project Gensim environment. Direct compiled-kernel probes are
deterministic numerical references, not single-thread experiment runs.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.runtime import required_environment, training_threads

os.environ.update(required_environment())
import fasttext
import gensim
import numpy as np
import yaml
from gensim.models import FastText, Word2Vec
from gensim.models.fasttext import ft_ngram_hashes, load_facebook_vectors
from gensim.models.fasttext_inner import train_batch_any
from threadpoolctl import threadpool_limits

OUT = ROOT / "artifacts/reviews/fasttext_paper_engineering"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table_sigmoid(x):
    # Same discretization and float32 arithmetic as fasttext_inner.pyx:init.
    slot = int((x + 8) * 32)
    e = np.float32(np.exp((slot / 512 * 2 - 1) * 8))
    return float(np.float32(e / np.float32(e + np.float32(1))))


def kernel_probe(sg, word="a", output_weight=0.1, input_sum=6.0):
    model = FastText(vector_size=4, sg=sg, hs=0, negative=1,
                     workers=training_threads(), min_count=1, min_n=1,
                     max_n=1, bucket=1009, sample=0, sorted_vocab=False,
                     shrink_windows=False, window=1, seed=42)
    model.build_vocab([[word, "z"]])
    source, target = [model.wv.key_to_index[w] for w in [word, "z"]]
    source_buckets = model.wv.buckets_word[source]
    target_buckets = model.wv.buckets_word[target]
    assert not set(source_buckets) & set(target_buckets)
    model.wv.vectors_vocab[:] = 0
    model.wv.vectors_ngrams[:] = 0
    model.syn1neg[:] = 0
    model.wv.vectors_vocab[source, 0] = 2
    for bucket in source_buckets:
        model.wv.vectors_ngrams[bucket, 0] = (input_sum - 2) / len(source_buckets)
    model.syn1neg[target, 0] = output_weight
    # All noise proposals equal z. For a -> z this forces the Gensim
    # continue-on-positive behavior and removes noise from that update.
    model.cum_table[:] = [0, 2**31 - 1]
    before_output = float(model.syn1neg[target, 0])
    before_input = float(model.wv.vectors_vocab[source, 0])
    before_ngrams = model.wv.vectors_ngrams[source_buckets].copy()
    alpha = 0.01
    train_batch_any(model, [[word, "z"]], alpha,
                    np.zeros(4, np.float32), np.zeros(4, np.float32))
    delta_output = float(model.syn1neg[target, 0]) - before_output
    delta_input = float(model.wv.vectors_vocab[source, 0]) - before_input
    n = len(source_buckets)
    hidden = input_sum / (n if sg else n + 1)
    dot = hidden * before_output
    skipped = bool(sg and (dot <= -8 or dot >= 8))
    f = 0.0 if dot <= -8 else 1.0 if dot >= 8 else table_sigmoid(dot)
    g = 0 if skipped else (1 - f) * alpha
    expected_output = g * hidden
    expected_input = g * before_output
    np.testing.assert_allclose(delta_output, expected_output, atol=2e-7, rtol=1e-4)
    np.testing.assert_allclose(delta_input, expected_input, atol=2e-7, rtol=1e-3)
    ngram_delta = model.wv.vectors_ngrams[source_buckets, 0] - before_ngrams[:, 0]
    quantized_expected = (before_ngrams[:, 0] + np.float32(expected_input)) - before_ngrams[:, 0]
    np.testing.assert_allclose(ngram_delta, quantized_expected, atol=3e-7, rtol=1e-3)
    return {"sg": sg, "word": word, "n_char_ngrams": n,
            "input_sum": input_sum, "export_denominator": n + 1,
            "observed_kernel_hidden_by_update": hidden,
            "score_dot": dot, "output_weight": before_output,
            "observed_delta_output": delta_output,
            "predicted_delta_output": expected_output,
            "observed_delta_word_input": delta_input,
            "predicted_unscaled_delta_word_input": expected_input,
            "observed_delta_ngram_inputs": ngram_delta.tolist(),
            "mathematical_mean_chain_rule_delta": expected_input / (n if sg else n + 1),
            "all_negative_proposals_equal_positive": True,
            "positive_saturation_skipped": skipped,
            "compiled_kernel_matches_source": True}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_dir = OUT / "sources"
    source_dir.mkdir(exist_ok=True)
    installed = Path(gensim.__file__).parent
    manifest = []
    for name in ["fasttext.py", "fasttext_inner.pyx", "fasttext_inner.pxd",
                 "fasttext_corpusfile.pyx", "word2vec.py", "keyedvectors.py"]:
        origin = installed / "models" / name
        archived = source_dir / name
        shutil.copyfile(origin, archived)
        snapshot = ROOT / "references/sources/gensim-4.4.0/models" / name
        equal = origin.read_text(encoding="utf-8") == snapshot.read_text(encoding="utf-8") if snapshot.exists() else None
        if equal is not None:
            assert equal, f"Project snapshot differs from runtime: {name}"
        manifest.append({"filename": name, "installed_path": str(origin),
                         "archive_path": str(archived.relative_to(ROOT)),
                         "sha256": digest(origin), "matches_project_snapshot_text": equal})
    shutil.copyfile(ROOT / "references/sources/gensim-4.4.0/LICENSE", source_dir / "LICENSE")
    kernel_binary = installed / "models" / Path(gensim.models.fasttext_inner.__file__).name

    probes = [kernel_probe(1), kernel_probe(1, "ab"), kernel_probe(0),
              kernel_probe(1, output_weight=-0.1, input_sum=200),
              kernel_probe(0, output_weight=-0.1, input_sum=200)]

    unsupported = None
    try:
        FastText(word_ngrams=2, workers=training_threads())
    except NotImplementedError as error:
        unsupported = str(error)
    assert unsupported

    counts_model = FastText(vector_size=4, min_count=1, bucket=17,
                           workers=training_threads(), sorted_vocab=False)
    counts_model.build_vocab([["frequent"] * 16 + ["rare"]])
    default_mass = np.diff(np.r_[0, counts_model.cum_table]).astype(float)
    default_prob = default_mass / default_mass.sum()
    counts_model.ns_exponent = 0.5
    counts_model.make_cum_table()
    aligned_mass = np.diff(np.r_[0, counts_model.cum_table]).astype(float)
    aligned_prob = aligned_mass / aligned_mass.sum()
    np.testing.assert_allclose(default_prob, [8/9, 1/9], atol=1e-8)
    np.testing.assert_allclose(aligned_prob, [4/5, 1/5], atol=1e-8)

    tokens = ["a", "ab", "where", "中国", "机器学习", "é", "e\u0301", "🙂", "</s>"]
    corpus = OUT / "probe_corpus.txt"
    # Native tiny training uses the current project 20-thread setting.
    corpus.write_text((" ".join(tokens) + "\n") * 300, encoding="utf-8")
    native = fasttext.train_unsupervised(input=str(corpus), model="skipgram",
        dim=4, epoch=1, lr=0.001, minCount=1, minn=1, maxn=3,
        bucket=1009, thread=training_threads(), verbose=0, t=1.0)
    native_path = OUT / "native_probe.bin"
    native.save_model(str(native_path))
    loaded = load_facebook_vectors(str(native_path))
    words = len(native.get_words())
    hashes = []
    for word in tokens + ["新词", "unknownzz", ""]:
        _, native_ids = native.get_subwords(word)
        ids = [int(i) - words for i in native_ids if int(i) >= words]
        gensim_ids = ft_ngram_hashes(word, 1, 3, 1009)
        hashes.append({"word": word, "native_buckets": ids, "gensim_buckets": gensim_ids,
                       "ordered_multiset_identical": ids == gensim_ids})
    assert all(row["ordered_multiset_identical"] for row in hashes if row["word"] != "</s>")
    assert not next(row for row in hashes if row["word"] == "</s>")["ordered_multiset_identical"]
    vector_errors = {word: float(np.max(np.abs(loaded.get_vector(word) - native.get_word_vector(word))))
                     for word in tokens if word != "</s>"}
    vector_errors.update({word: float(np.max(np.abs(loaded.get_vector(word) - native.get_word_vector(word))))
                          for word in ["新词", "unknownzz"]})
    assert max(vector_errors.values()) < 1e-6
    eos_error = float(np.max(np.abs(loaded.get_vector("</s>") - native.get_word_vector("</s>"))))
    assert eos_error > 1e-6

    composition = FastText(vector_size=4, min_count=1, min_n=1, max_n=1,
                           bucket=1009, workers=training_threads(), sorted_vocab=False)
    composition.build_vocab([["a", "z"]])
    composition.wv.vectors_vocab[:] = 0
    composition.wv.vectors_ngrams[:] = 0
    for word in ["a", "z"]:
        index = composition.wv.key_to_index[word]
        composition.wv.vectors_vocab[index, 0] = 2
        for bucket in composition.wv.buckets_word[index]:
            composition.wv.vectors_ngrams[bucket, 0] = 4
    composition.wv.adjust_vectors()
    iv_value = composition.wv.get_vector("a").tolist()
    # aa has two occurrences of the same char ngram, no separate word row.
    oov_value = composition.wv.get_vector("aa").tolist()
    assert iv_value[0] == 3 and oov_value[0] == 4

    toy = FastText(vector_size=4, workers=training_threads(), sg=1, negative=2,
                   hs=0, sample=0, min_count=1, bucket=101, seed=42, epochs=1)
    data = [["a", "b", "c"]] * 100
    toy.build_vocab(data)
    before = toy.wv.vectors_vocab.copy()
    toy.train(data, total_examples=len(data), epochs=1, compute_loss=True)
    changed = bool(np.any(before != toy.wv.vectors_vocab))
    loss = float(toy.get_latest_training_loss())
    assert changed and loss == 0

    configs = {str(path.relative_to(ROOT)): yaml.safe_load(path.read_text(encoding="utf-8"))
               for path in [ROOT / "configs/experiments.yaml", ROOT / "configs/thucnews.yaml",
                            ROOT / "configs/paper_reproduction.yaml", ROOT / "configs/runtime.yaml"]}
    result = {"date": "2026-10-09", "python_executable": sys.executable,
              "gensim_version": gensim.__version__,
              "fasttext_wheel_version": importlib.metadata.version("fasttext-wheel"),
              "training_threads": training_threads(), "source_manifest": manifest,
              "native_probe_seed": 0,
              "native_probe_seed_note": "train_unsupervised wrapper has no seed keyword; native default is 0",
              "compiled_kernel": {"path": str(kernel_binary), "sha256": digest(kernel_binary)},
              "kernel_probe_scope": "Direct two-token numerical references; no asynchronous model fitting",
              "kernel_probes": probes, "word_ngrams_2_error": unsupported,
              "train_supervised_method_present": hasattr(FastText, "train_supervised"),
              "predict_method_present": hasattr(FastText, "predict"),
              "negative_distribution": {"counts": [16, 1], "default_ns_exponent": 0.75,
                                         "default_probabilities": default_prob.tolist(),
                                         "ns_exponent_0_5_probabilities": aligned_prob.tolist()},
              "hash_probes": hashes, "ordinary_import_vector_errors": vector_errors,
              "eos_import_vector_max_error": eos_error,
              "composite_probe": {"iv_a": iv_value, "oov_aa": oov_value},
              "loss_probe": {"weights_changed": changed, "compute_loss_requested": True,
                             "reported_loss": loss, "workers": training_threads()},
              "score_is_inherited_word2vec_method": FastText.score is Word2Vec.score,
              "constructor_defaults": {k: p.default for k,p in inspect.signature(FastText).parameters.items()
                                       if k in ["sg", "hs", "negative", "ns_exponent", "vector_size", "min_n", "max_n",
                                                "bucket", "alpha", "min_alpha", "sample", "epochs", "workers", "min_count"]},
              "current_configs": configs,
              "source_fingerprints": {str(p.relative_to(ROOT)): digest(p) for p in
                  [ROOT / "gensim_lab/models.py", ROOT / "gensim_lab/paper_worker.py",
                   ROOT / "gensim_lab/paper_data.py", ROOT / "gensim_lab/word_fasttext_score.py",
                   ROOT / "configs/experiments.yaml", ROOT / "configs/thucnews.yaml",
                   ROOT / "configs/paper_reproduction.yaml", ROOT / "configs/runtime.yaml",
                   ROOT / "paper/Bag of Tricks for Efficient Text Classification.pdf",
                   ROOT / "paper/Enriching Word Vectors with Subword Information.pdf"]}}
    output = OUT / "audit.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"evidence": str(output), "kernel_probes": probes,
                      "eos_import_vector_max_error": eos_error,
                      "ordinary_vector_max_error": max(vector_errors.values()),
                      "loss_probe": result["loss_probe"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    with threadpool_limits(limits=4):
        main()
