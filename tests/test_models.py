import copy
import numpy as np
import pytest
from gensim_lab.common import config
from gensim_lab.models import Bundle, Representation


def documents():
    return [{"id": f"train-{i}", "page_label": "01" if i < 6 else "02",
             "tokens": ["共同", "新闻", "经济" if i < 6 else "体育"] * 8,
             "sentences": [["共同", "新闻", "经济" if i < 6 else "体育"]] * 8} for i in range(12)]


def spec(method):
    cfg = config()
    p = copy.deepcopy(cfg["embedding"])
    p.update(vector_size=10, min_count=1, epochs=2, bucket=100)
    return {"method": method, "classes": 2, "seed": 42, "embedding": p,
            "dictionary": {"no_below": 1, "no_above": 1.0, "keep_n": 100}, "svm_c": [1]}


@pytest.mark.parametrize("method", ["tfidf", "fasttext", "doc2vec"])
def test_save_load_and_training_dictionary(tmp_path, method):
    s = spec(method)
    validation = [{"id": "held-out", "page_label": "01", "tokens": ["验证专属词", "经济"]}]
    bundle = Bundle(s).fit(documents(), validation, tmp_path)
    assert "验证专属词" not in bundle.representation.dictionary.token2id
    before, score, _ = bundle.predict(validation)
    bundle.save(tmp_path)
    after, other, _ = Bundle.load(tmp_path).predict(validation)
    assert before == after
    np.testing.assert_allclose(score, other, rtol=1e-6, atol=1e-6)


def test_oov_fasttext_vs_word2vec():
    doc = [{"id": "unseen", "tokens": ["共同新词"]}]
    outputs = {}
    for method in ["word2vec", "fasttext"]:
        s = spec(method)
        model = Representation(method, s["embedding"], s["dictionary"], 42).fit(documents())
        x, coverage = model.transform(doc)
        outputs[method] = coverage
        assert np.all(np.isfinite(x))
    assert outputs["word2vec"]["zero_documents"] == 1
    assert outputs["word2vec"]["oov_vector_coverage"] == 0
    assert outputs["fasttext"]["zero_documents"] == 0
    assert outputs["fasttext"]["oov_vector_coverage"] == 1


def test_doc2vec_inference_independent_of_order():
    s = spec("doc2vec")
    model = Representation("doc2vec", s["embedding"], s["dictionary"], 42).fit(documents())
    docs = [{"id": "a", "tokens": ["经济", "新闻"]}, {"id": "b", "tokens": ["体育", "共同"]}]
    a, _ = model.transform(docs)
    b, _ = model.transform(list(reversed(docs)))
    np.testing.assert_array_equal(a, b[::-1])
