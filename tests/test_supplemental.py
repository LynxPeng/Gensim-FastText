import collections
import csv

import numpy as np
from sklearn.metrics import f1_score

from gensim_lab import supplemental
from gensim_lab.supplemental_worker import permute_training_labels


def test_permutation_preserves_inputs_counts_and_is_deterministic():
    docs = [{"id": str(i), "page_label": "01" if i < 6 else "02", "tokens": [str(i)]} for i in range(12)]
    shuffled = permute_training_labels(docs, 12345)
    assert shuffled == permute_training_labels(docs, 12345)
    assert collections.Counter(d["page_label"] for d in shuffled) == collections.Counter(d["page_label"] for d in docs)
    assert [d["tokens"] for d in shuffled] == [d["tokens"] for d in docs]
    assert any(a["page_label"] != b["page_label"] for a, b in zip(docs, shuffled))
    assert docs[0]["page_label"] == "01"


def test_chronological_groups_never_cross_date_boundaries():
    docs = [{"id": date + "-" + group, "date": date, "group": group}
            for date, group in [("19980121", "shared"), ("19980122", "shared"),
                                ("19980122", "val"), ("19980127", "val"), ("19980127", "test")]]
    parts, excluded = supplemental.chronological_splits(docs, {
        "chronological_train_end": "19980121", "chronological_validation_end": "19980126"})
    assert list(map(len, parts.values())) == [1, 1, 1]
    assert len(excluded) == 2
    assert excluded[0]["group_owner"] == "train" and excluded[1]["group_owner"] == "validation"


def test_confusion_macro_matches_sklearn_with_absent_classes():
    matrix = np.array([[3, 1, 0], [2, 5, 0], [0, 0, 0]])
    truth, predicted = [], []
    for i in range(3):
        for j in range(3):
            truth.extend([i] * int(matrix[i, j]))
            predicted.extend([j] * int(matrix[i, j]))
    expected = f1_score(truth, predicted, labels=[0, 1, 2], average="macro", zero_division=0)
    np.testing.assert_allclose(supplemental.macro_from_confusion(matrix), expected, atol=1e-15)
    np.testing.assert_allclose(supplemental.macro_from_confusion(np.stack([matrix, matrix])), [expected, expected])


def test_date_bootstrap_is_paired_and_reproducible(tmp_path, monkeypatch):
    monkeypatch.setattr(supplemental, "ARTIFACTS", tmp_path)
    docs = {str(i): {"date": "19980101" if i < 3 else "19980102", "page_label": "01" if i % 2 == 0 else "02"} for i in range(4)}
    for key in ["a", "b"]:
        directory = tmp_path / "runs" / key
        directory.mkdir(parents=True)
        with (directory / "predictions_test.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "true_label", "predicted_label"])
            writer.writerows([[key, doc["page_label"], doc["page_label"]] for key, doc in docs.items()])
    models = {"TF-IDF": ["a"], "same": ["b"]}
    result = supplemental.clustered_bootstrap(models, docs, repeats=100, seed=7)
    assert result == supplemental.clustered_bootstrap(models, docs, repeats=100, seed=7)
    assert result["dates"] == 2 and result["articles"] == 4
    assert result["paired_differences"]["same"]["difference"] == 0
    assert result["paired_differences"]["same"]["percentile_95"] == [0.0, 0.0]
    block = supplemental.clustered_bootstrap(models, docs, repeats=100, seed=7, block_size=2)
    assert block["block_size"] == 2
    assert block["paired_differences"]["same"]["percentile_95"] == [0.0, 0.0]
    assert block["intervals"]["TF-IDF"]["percentile_95"] == [block["intervals"]["TF-IDF"]["macro_f1"]] * 2
