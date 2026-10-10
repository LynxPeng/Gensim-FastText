import math
import numpy as np
import pytest

from gensim_lab.paper_data import normalize_classification, parse_labeled, TextCorpus
from gensim_lab.paper_hs import huffman, exact_top1


def test_label_tokens_are_excluded_from_gensim_corpus(tmp_path):
    path = tmp_path / "labels.txt"
    path.write_text("__label__cat cat runs\n__label__dog __label__pet dog runs\n", encoding="utf-8")
    assert list(TextCorpus(path)) == [["cat", "runs"], ["dog", "runs"]]
    assert parse_labeled("__label__dog __label__pet dog runs")[0] == ["dog", "pet"]


def test_author_normalization_keeps_local_word_order_and_separates_punctuation():
    labels, words = normalize_classification('1,"A Dog!","Dog <br /> runs."\n')
    assert labels == ["1"]
    assert words == [",", "a", "dog", "!", ",", "dog", "runs", "."]


def test_native_ascii_token_boundaries_preserve_unicode_spaces():
    labels, words = parse_labeled("__label__a\u00a0b foo\u00a0bar\tend\x00last")
    assert labels == ["a\u00a0b"]
    assert words == ["foo\u00a0bar", "end", "last"]


@pytest.mark.parametrize("counts", [[1, 1], [100, 1, 1, 1, 1], list(range(1, 18))])
def test_exact_tree_search_matches_exhaustive_label_probabilities(counts):
    tree = huffman(counts)
    rng = np.random.default_rng(42)
    features = rng.normal(size=(15, 7)).astype(np.float32)
    weights = rng.normal(size=(len(counts) - 1, 7)).astype(np.float32)
    predictions, log_probability, visited = exact_top1(features, weights, tree["left"], tree["right"], tree["root"])
    exhaustive = []
    for feature in features:
        probabilities = []
        for label in range(len(counts)):
            valid = tree["mask"][label].astype(bool)
            scores = weights[tree["paths"][label][valid]].astype(np.float64) @ feature.astype(np.float64)
            bits = tree["targets"][label][valid]
            probabilities.append(float((-np.logaddexp(0, (1 - 2 * bits) * scores)).sum()))
        exhaustive.append(probabilities)
    exhaustive = np.array(exhaustive)
    assert np.array_equal(predictions, exhaustive.argmax(axis=1))
    assert np.allclose(log_probability, exhaustive.max(axis=1), atol=1e-7)
    assert np.allclose(np.exp(exhaustive).sum(axis=1), 1.0)
    assert np.all(visited <= 2 * len(counts) - 1)


def test_invalid_training_label_frequencies_fail():
    with pytest.raises(ValueError):
        huffman([1])
    with pytest.raises(ValueError):
        huffman([1, 0])


def test_document_pooling_matches_repeated_token_mean_and_handles_oov():
    from gensim.models import FastText
    from gensim_lab.paper_worker import document_vector
    model = FastText(vector_size=7, min_count=1, bucket=1000, workers=1, seed=42)
    model.build_vocab([['cat', 'dog', 'runs']])
    words = ['cat', 'cat', 'dog', 'unknown']
    expected = np.mean([model.wv.get_vector(word) for word in words], axis=0)
    expected /= np.linalg.norm(expected)
    actual, missing, empty = document_vector(model, words, 7, {})
    assert np.allclose(actual, expected, atol=1e-6)
    assert missing == 1 and not empty
    assert not document_vector(model, [], 7)[0].any()




def test_external_hs_learns_and_preserves_empty_tail_on_reload(tmp_path):
    import torch
    from gensim_lab.paper_hs import HuffmanClassifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features = np.array([[-1., 0.], [1., 0.], [0., 0.], [0., 0.]], dtype=np.float32)
    offsets = np.array([0, 1, 2, 2, 2], dtype=np.int64)
    targets = np.array([0, 1], dtype=np.int32)
    classifier = HuffmanClassifier(["negative", "positive"], [1, 1], 2, 42)
    classifier.fit(features, offsets, targets, tmp_path, epochs=8, learning_rate=1.0,
                   batch_size=2, device=device)
    assert len(classifier.history) == 8
    assert all(set(row)=={'epoch','documents'} for row in classifier.history)
    assert classifier.history[-1]["documents"] == 2
    predictions, scores, _ = classifier.predict(features[:2])
    assert predictions.tolist() == [0, 1]
    classifier.save(tmp_path)
    reloaded = HuffmanClassifier.load(tmp_path)
    result = reloaded.predict(features[:2])
    assert np.array_equal(result[0], predictions)
    assert np.array_equal(result[1], scores)
