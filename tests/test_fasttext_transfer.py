import collections

import fasttext
import numpy as np
from gensim.models import FastText, KeyedVectors

from gensim_lab.fasttext_transfer import export_initialization, make_splits, training_vocabulary


def test_six_month_split_exhaustive_stratified_and_group_isolated():
    docs = [{"id": f"{month}-{page:02d}-{i:03d}", "month": month, "page_label": f"{page:02d}",
             "group": f"{month}-{page:02d}-{i // 2:03d}"}
            for month in [f"1998{i:02d}" for i in range(1, 7)] for page in range(1, 13) for i in range(40)]
    # A real near-duplicate component can cross both month and page strata.
    docs[0]["group"] = docs[-1]["group"]
    parameters = {"folds": 10, "train_folds": 7, "validation_folds": 1}
    split = make_splits(docs, parameters, 42)
    assert split == make_splits(docs, parameters, 42)
    assert sum(map(len, split.values())) == len(docs)
    lookup = {d["id"]: d for d in docs}
    owners = {}
    for part, ids in split.items():
        counts = collections.Counter((lookup[i]["month"], lookup[i]["page_label"]) for i in ids)
        assert len(counts) == 72
        for i in ids:
            assert owners.setdefault(lookup[i]["group"], part) == part


def test_train_only_export_and_installed_native_vector_import(tmp_path):
    source = KeyedVectors(vector_size=4)
    vectors = np.asarray([[0.5, 0.25, -0.75, 1], [-1, 2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
    source.add_vectors(["训练", "新闻", "仅在测试"], vectors)
    words = training_vocabulary([{"tokens": ["训练", "新闻", "训练集未登录"]}])
    exported = tmp_path / "pretrained.vec"
    record = export_initialization(source, words, exported, generate_oov=False)
    assert record["exported_words"] == 2
    assert record["kinds"] == {"source_vocabulary": 2, "random_initialization": 1}
    loaded = KeyedVectors.load_word2vec_format(str(exported))
    assert set(loaded.key_to_index) == {"训练", "新闻"}
    assert np.array_equal(loaded["训练"], source["训练"])
    train = tmp_path / "train.txt"
    train.write_text("__label__p01 训练 新闻 训练集未登录\n__label__p02 新闻 训练\n" * 200, encoding="utf-8")
    model = fasttext.train_supervised(input=str(train), pretrainedVectors=str(exported), dim=4,
                                      epoch=1, lr=0, minn=0, maxn=0, wordNgrams=1,
                                      minCount=1, thread=2, verbose=0)
    assert np.array_equal(model.get_word_vector("训练"), source["训练"])
    assert set(model.get_words()) == set(words) | {"</s>"}
    assert "仅在测试" not in model.get_words()


def test_gensim_subwords_export_only_requested_training_oov(tmp_path):
    model = FastText(vector_size=6, min_count=1, min_n=1, max_n=3, bucket=100,
                     workers=2, seed=42)
    model.build_vocab(corpus_iterable=[["经济", "新闻", "经济发展"]] * 100)
    model.train(corpus_iterable=[["经济", "新闻", "经济发展"]] * 100, total_examples=100, epochs=1)
    path = tmp_path / "fasttext.vec"
    record = export_initialization(model.wv, ["经济", "经济新闻发展"], path, generate_oov=True)
    assert record["kinds"] == {"source_vocabulary": 1, "generated_subword": 1}
    exported = KeyedVectors.load_word2vec_format(str(path))
    assert set(exported.key_to_index) == {"经济", "经济新闻发展"}
    assert np.array_equal(exported["经济新闻发展"], model.wv["经济新闻发展"])
