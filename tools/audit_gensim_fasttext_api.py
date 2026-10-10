"""Probe the exact Gensim 4.4.0 FastText semantics this project relies on.

Read-only with respect to experiment data: writes only its JSON evidence file.
Run: conda run -n Gensim python tools/audit_gensim_fasttext_api.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
from gensim.matutils import corpus2csc
from gensim.models import Doc2Vec, FastText, TfidfModel, Word2Vec
from gensim.corpora import Dictionary
from gensim.models.doc2vec import TaggedDocument

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gensim_lab.runtime import training_threads
OUT = ROOT / "artifacts/reviews/gensim_fasttext_api_audit.json"

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
         "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi"]
PARAMS = dict(vector_size=20, window=5, min_count=1, sg=0, negative=5, hs=0,
              sample=0.0, alpha=0.025, min_alpha=0.0001, workers=training_threads(), epochs=5,
              min_n=3, max_n=6, bucket=1000, batch_words=10000)


def corpus_lines(seed=42, documents=300, length=40):
    rng = np.random.default_rng(seed)
    lines = []
    for _ in range(documents):
        lines.append(" ".join(rng.choice(WORDS, size=length)))
    return lines


def write_corpus(path: Path, lines) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sum(len(line.split()) for line in lines)


class ListCorpus:
    def __init__(self, lines):
        self.lines = lines

    def __iter__(self):
        for line in self.lines:
            yield line.split()


def probe_build_vocab_equivalence(path, lines, tokens):
    """FastText(**params)+build_vocab(iterable)+train(corpus_file) vs FastText(corpus_file=...)."""
    hybrid = FastText(**PARAMS)
    hybrid.build_vocab(corpus_iterable=ListCorpus(lines))
    hybrid_words = list(hybrid.wv.index_to_key)
    hybrid_total = int(hybrid.corpus_total_words)
    hybrid_count = int(hybrid.corpus_count)
    hybrid_buckets = {w: [int(x) for x in hybrid.wv.buckets_word[i]] for i, w in enumerate(hybrid_words)}
    hybrid.train(corpus_file=str(path), total_words=hybrid.corpus_total_words,
                 epochs=PARAMS["epochs"], callbacks=[])

    direct = FastText(corpus_file=str(path), **PARAMS)

    same_words = hybrid_words == list(direct.wv.index_to_key)
    same_buckets = hybrid_buckets == {w: [int(x) for x in direct.wv.buckets_word[i]]
                                      for i, w in enumerate(direct.wv.index_to_key)}
    return {
        "hybrid_corpus_total_words": hybrid_total,
        "direct_corpus_total_words": int(direct.corpus_total_words),
        "actual_file_tokens": tokens,
        "corpus_total_words_matches_file": hybrid_total == tokens,
        "same_vocabulary": same_words,
        "same_ngram_buckets": same_buckets,
        "vector_size": int(hybrid.wv.vector_size),
        "bucket": int(hybrid.wv.bucket),
        "vocabulary_size": len(hybrid_words),
    }, hybrid


def probe_composite_vectors(model):
    """wv.vectors[i] must be the native-equivalent mean of word vector and its n-gram vectors."""
    word = model.wv.index_to_key[0]
    index = model.wv.key_to_index[word]
    buckets = list(model.wv.buckets_word[index])
    manual = (model.wv.vectors_vocab[index] + model.wv.vectors_ngrams[buckets].sum(axis=0)) / (len(buckets) + 1)
    return {
        "word": word,
        "ngram_count": len(buckets),
        "max_abs_error_vectors_vs_composite": float(np.max(np.abs(model.wv.vectors[index] - manual))),
        "max_abs_error_get_vector_vs_composite": float(np.max(np.abs(model.wv.get_vector(word) - manual))),
    }


def probe_short_word_no_ngrams(model):
    """Character n-grams are computed on '<word>', so even a 2-character word with min_n=3 has n-grams.

    gensim's ft_ngram_hashes uses compute_ngrams_bytes, the byte-level port of fastText's
    Dictionary::computeSubwords, which is also fed BOW+word+EOW. So a non-zero vector is expected here.
    """
    probe = "ab"
    try:
        vector = model.wv.get_vector(probe)
        status = "returned"
    except KeyError:
        vector, status = np.zeros(model.wv.vector_size), "KeyError"
    return {"word": probe, "min_n": int(model.wv.min_n), "status": status,
            "is_zero_vector": bool(not np.any(vector)),
            "note": "non-zero is correct: n-grams are taken over '<ab>'"} 


def probe_save_load(directory, model):
    directory.mkdir(parents=True, exist_ok=True)
    saved = directory / "probe.gensim"
    model.paper_epoch_records = [{"epoch": 1, "probe": True}]
    model.save(str(saved), separately=[])
    restored = FastText.load(str(saved))
    diff = float(np.max(np.abs(restored.wv.vectors - model.wv.vectors)))
    return {
        "bytes": saved.stat().st_size,
        "max_abs_error_wv_vectors_after_load": diff,
        "custom_attribute_survived": getattr(restored, "paper_epoch_records", None) == [{"epoch": 1, "probe": True}],
        "corpus_total_words_survived": int(getattr(restored, "corpus_total_words", -1)),
    }


def probe_alpha_resume_schedule():
    """Compare the project's checkpoint-resume alpha against gensim's own schedule."""
    model = FastText(**PARAMS)
    model.alpha, model.min_alpha, model.epochs = PARAMS["alpha"], PARAMS["min_alpha"], PARAMS["epochs"]
    continuous = [model._get_next_alpha(0.0, epoch) for epoch in range(PARAMS["epochs"])]
    resumed = {}
    for done in range(1, PARAMS["epochs"]):
        start = PARAMS["alpha"] + (PARAMS["min_alpha"] - PARAMS["alpha"]) * done / PARAMS["epochs"]
        model.alpha, model.min_alpha, model.epochs = start, PARAMS["min_alpha"], PARAMS["epochs"] - done
        resumed[done] = [model._get_next_alpha(0.0, epoch) for epoch in range(PARAMS["epochs"] - done)]
    max_error = max(abs(resumed[done][k] - continuous[done + k])
                    for done in resumed for k in range(len(resumed[done])))
    return {"continuous_alpha_at_epoch_start": continuous,
            "max_abs_error_resume_vs_continuous": max_error,
            "resume_schedule_identical": max_error < 1e-15}


def probe_corpusfile_worker_threads(path, workers, epochs=2):
    """Confirm corpus_file training really spawns `workers` training threads per epoch in 4.4.0."""
    seen, lock = set(), threading.Lock()
    original = Word2Vec._worker_loop_corpusfile

    def patched(self, *args, **kwargs):
        with lock:
            seen.add(threading.current_thread().name)
        return original(self, *args, **kwargs)

    Word2Vec._worker_loop_corpusfile = patched
    try:
        FastText(corpus_file=str(path), **dict(PARAMS, workers=workers, epochs=epochs))
    finally:
        Word2Vec._worker_loop_corpusfile = original
    starts = len(seen)
    return {"requested_workers": workers, "epochs": epochs,
            "observed_worker_thread_starts": starts,
            "expected_thread_starts": workers * epochs,
            "workers_per_epoch_confirmed": starts == workers * epochs,
            "workers_honoured": starts >= workers * epochs}


def probe_models_py_calls(tmp: Path):
    """Check the gensim calls used by gensim_lab/models.py."""
    documents = [line.split() for line in corpus_lines(documents=40, length=30)]
    dictionary = Dictionary(documents)
    dictionary.filter_extremes(no_below=3, no_above=0.9, keep_n=50000)
    bows = [dictionary.doc2bow(words) for words in documents]
    tfidf = TfidfModel(bows, dictionary=dictionary, normalize=True)
    single = tfidf[bows[0]]
    single_type = type(single).__name__
    single_is_pairs = bool(single) and isinstance(single[0], tuple)
    corpus_matrix = corpus2csc([tfidf[bow] for bow in bows], num_terms=len(dictionary),
                               num_docs=len(documents), dtype=np.float32)
    tagged = [TaggedDocument(words, [index]) for index, words in enumerate(documents)]
    doc2vec = Doc2Vec(documents=tagged, dm=0, vector_size=20, window=5, min_count=1, epochs=2,
                      negative=5, hs=0, sample=0.0, alpha=0.025, min_alpha=0.0001, workers=training_threads(),
                      batch_words=10000, seed=42)
    return {
        "tfidf_single_document_indexing_returns": single_type,
        "tfidf_single_document_is_(id,weight)_pairs": single_is_pairs,
        "tfidf_idfs_is_dict": isinstance(tfidf.idfs, dict),
        "corpus2csc_shape_terms_by_docs": list(corpus_matrix.shape),
        "corpus2csc_transposed_shape_docs_by_terms": list(corpus_matrix.T.shape),
        "doc2vec_has_wv": hasattr(doc2vec, "wv"),
        "doc2vec_wv_has_key_to_index": hasattr(getattr(doc2vec, "wv", None), "key_to_index"),
        "dictionary_len_after_filtering": len(dictionary),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=ROOT / ".artifact-work/gensim_api_audit")
    args = parser.parse_args()
    directory = args.directory
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "corpus.txt"
    lines = corpus_lines()
    tokens = write_corpus(path, lines)

    result = {"gensim_version": __import__("gensim").__version__,
              "numpy_version": np.__version__, "corpus_tokens": tokens, "params": PARAMS}
    started = time.perf_counter()
    result["build_vocab_equivalence"], model = probe_build_vocab_equivalence(path, lines, tokens)
    result["composite_word_vectors"] = probe_composite_vectors(model)
    result["short_word_without_ngrams"] = probe_short_word_no_ngrams(model)
    result["save_load"] = probe_save_load(directory, model)
    result["alpha_resume_schedule"] = probe_alpha_resume_schedule()
    result["corpus_file_worker_threads"] = probe_corpusfile_worker_threads(path, 4, epochs=2)
    result["models_py_calls"] = probe_models_py_calls(directory)
    result["seconds"] = time.perf_counter() - started
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    main()
