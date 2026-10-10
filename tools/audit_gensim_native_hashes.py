"""Check whether Gensim's FastText character n-grams hash into the same buckets as native fastText.

This decides whether "Gensim 复现 FastText" shares the native subword feature extractor.
Run: conda run -n Gensim python tools/audit_gensim_native_hashes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import fasttext
import numpy as np
from gensim.models import FastText

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gensim_lab.runtime import training_threads
OUT = ROOT / "artifacts/reviews/gensim_native_hash_compatibility.json"
MIN_N, MAX_N, BUCKET = 3, 6, 200000

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "internationalization counterrevolutionary antidisestablishmentarianism",
    "word embeddings subword information bag of tricks",
    "机器学习 自然语言处理 词向量 子词",
    "a bb ccc dddd eeeee ffffff ggggggg",
]


def main():
    directory = ROOT / ".artifact-work/gensim_hash_audit"
    directory.mkdir(parents=True, exist_ok=True)
    corpus = directory / "corpus.txt"
    corpus.write_text("\n".join(TEXTS) + "\n", encoding="utf-8")

    native = fasttext.train_supervised(input=str(corpus), dim=10, epoch=1, minCount=1, minn=MIN_N,
                                      maxn=MAX_N, bucket=BUCKET, wordNgrams=1, loss="softmax", thread=training_threads(),
                                      seed=42, verbose=0)
    gensim_model = FastText(corpus_file=str(corpus), vector_size=10, min_count=1, min_n=MIN_N, max_n=MAX_N,
                            bucket=BUCKET, epochs=1, workers=training_threads(), sg=0, negative=5, hs=0, seed=42, sample=0.0)

    words = sorted({word for text in TEXTS for word in text.split()})
    offset = len(native.get_words())  # native shares one id space: ngram ids are offset by nwords
    rows, matched, mismatched = [], 0, []
    for word in words:
        subwords, native_hashes = native.get_subwords(word)
        # Dictionary::initNgrams stores the word's own id first, then the char-n-gram buckets.
        word_id, native_ngrams = int(native_hashes[0]), [int(value) for value in native_hashes[1:]]
        assert word_id == native.get_word_id(word), "first get_subwords entry must be the word id"
        native_buckets = sorted(value - offset for value in native_ngrams)
        if word in gensim_model.wv.key_to_index:
            index = gensim_model.wv.key_to_index[word]
            gensim_buckets = sorted(int(value) for value in gensim_model.wv.buckets_word[index])
            identical = gensim_buckets == native_buckets
        else:
            gensim_buckets, identical = None, None
        matched += bool(identical)
        if identical is False:
            mismatched.append(word)
        rows.append({"word": word, "word_id": word_id, "native_ngram_count": len(native_buckets),
                     "gensim_ngram_count": None if gensim_buckets is None else len(gensim_buckets),
                     "bucket_sets_identical": identical,
                     "native_subword_strings": subwords[:8],
                     "native_first_buckets": native_buckets[:6],
                     "gensim_first_buckets": None if gensim_buckets is None else gensim_buckets[:6]})

    comparable = [row for row in rows if row["bucket_sets_identical"] is not None]
    result = {
        "min_n": MIN_N, "max_n": MAX_N, "bucket": BUCKET,
        "native_vocabulary_size": len(native.get_words()),
        "gensim_vocabulary_size": len(gensim_model.wv),
        "vocabulary_identical": native.get_words() == list(gensim_model.wv.index_to_key)
                                or sorted(native.get_words()) == sorted(gensim_model.wv.index_to_key),
        "words_compared": len(comparable),
        "words_with_identical_bucket_sets": matched,
        "mismatched_words": mismatched,
        "all_comparable_words_match": matched == len(comparable) and bool(comparable),
        "detail": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "detail"}, indent=2, ensure_ascii=False))
    print("\nper-word:")
    for row in rows:
        print(f"  {row['word']:<28} native={row['native_ngram_count']:<3} "
              f"gensim={row['gensim_ngram_count']} identical={row['bucket_sets_identical']}")
    print(f"\nWrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
