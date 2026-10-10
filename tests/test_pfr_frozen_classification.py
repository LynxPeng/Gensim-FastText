"""Leakage, pooling and paired uncertainty guards for the new transfer task."""
import numpy as np
import pytest
from scipy import sparse

from gensim_lab import pfr_frozen_classification as task
from gensim_lab.common import write_json


def test_partition_rejects_near_duplicate_and_content_leakage():
    lookup = {'a': {'id': 'a', 'content_hash': 'A', 'group': 'g'},
              'b': {'id': 'b', 'content_hash': 'B', 'group': 'g'}}
    with pytest.raises(ValueError, match='group'):
        task.validate_partitions({'train': ['a'], 'test': ['b']}, lookup)
    lookup['b'].update(content_hash='A', group='other')
    with pytest.raises(ValueError, match='content_hash'):
        task.validate_partitions({'train': ['a'], 'test': ['b']}, lookup)


def test_temporal_excludes_groups_seen_in_earlier_partitions(tmp_path, monkeypatch):
    monkeypatch.setattr(task, 'PROCESSED', tmp_path)
    write_json(tmp_path / 'splits.json', {'train': ['jan'], 'validation': [], 'test': []})
    descriptions = [('jan', '199801', 'a'), ('april', '199804', 'b'),
                    ('may_duplicate', '199805', 'b'), ('may', '199805', 'c'),
                    ('june_duplicate', '199806', 'c'), ('june', '199806', 'd')]
    docs = [{'id': i, 'month': month, 'group': group, 'content_hash': i, 'page_label': '01'}
            for i, month, group in descriptions]
    profiles, future = task.make_profiles(docs, {'classes': [2]})
    assert profiles['temporal_12']['splits'] == {'train': ['jan', 'april'], 'validation': ['may'], 'test': ['june']}
    assert set(future['199806']) == {'june_duplicate', 'june'}


def test_word_occurrences_and_oov_follow_frozen_pooling_contract():
    class Vectors:
        key_to_index = {'known_a': 0, 'known_b': 1}

        def get_vector(self, word):
            return {'known_a': np.array([1., 0.]), 'known_b': np.array([0., 1.]),
                    'unknown': np.array([0., 1.])}[word]

    representation = task.FrozenRepresentation.__new__(task.FrozenRepresentation)
    representation.width, representation.wv = 2, Vectors()
    doc = {'tokens': ['known_a', 'known_a', 'known_b', 'unknown'], 'id': 'article', 'month': '199801'}
    representation.method = 'word2vec'
    vector, stats = representation.one(doc)
    assert np.allclose(vector, np.array([2, 1]) / np.sqrt(5))
    assert stats['iv_tokens'] == stats['usable_tokens'] == 3
    representation.method = 'fasttext'
    vector, stats = representation.one(doc)
    assert np.allclose(vector, np.array([1, 1]) / np.sqrt(2))
    assert stats['iv_tokens'] == 3 and stats['usable_tokens'] == 4


def test_bootstrap_is_paired_when_model_predictions_are_identical():
    truth = ['01', '01', '02', '02', '02']
    same = ['01', '02', '02', '02', '01']
    predictions = {m: same for m in task.METHODS}
    result = task.paired_bootstrap(truth, predictions, ['01', '02'], 20, 42)
    assert all(row['difference'] == row['ci_low'] == row['ci_high'] == 0 for row in result['differences'])


def test_corrupted_classifier_cannot_be_reused(tmp_path):
    spec = {'only_train': True}
    for name in ['spec.json', 'summary.json', 'metrics_validation.json']:
        write_json(tmp_path / name, {})
    (tmp_path / 'predictions_validation.csv').write_text('id,label\na,01\n')
    task.seal_run(tmp_path, spec)
    assert task.cached_run(tmp_path, spec)
    (tmp_path / 'predictions_validation.csv').write_text('id,label\na,02\n')
    with pytest.raises(ValueError, match='cache changed'):
        task.cached_run(tmp_path, spec)


def test_softmax_training_handles_windows_integer_targets(tmp_path):
    # Exercise the actual sparse and dense worker, rather than only a CUDA toy.
    labels = task.labels_for(12)
    docs = [{'id': f'{part}_{label}', 'page_label': label}
            for part in ['train', 'validation'] for label in labels]
    splits = {part: [d['id'] for d in docs if d['id'].startswith(part)] for part in ['train', 'validation']}
    options = {'learning_rates': [0.01], 'batch_size': 8, 'weight_decay': 0.0001,
               'min_epochs': 2, 'max_epochs': 2, 'patience': 2, 'min_delta': 0.001}
    plan = {'configuration': {'seeds': [42], 'softmax': options}, 'runtime': {'numerical_threads': 4},
            'document_order': [d['id'] for d in docs],
            'profiles': {p: {'splits': splits} for p in ['january_12', 'temporal_12']}}
    write_json(tmp_path / 'plan.json', plan)
    write_json(tmp_path / 'index.json', {'runs': {}, 'selection': {}})
    folder = tmp_path / 'features'
    folder.mkdir()
    x = np.tile(np.eye(12, dtype=np.float32), (2, 1))
    sparse.save_npz(folder / 'tfidf.npz', sparse.csr_matrix(x))
    for method in ['word2vec', 'fasttext']:
        np.save(folder / f'{method}.npy', x)
    task.train_softmax(tmp_path, docs)
    assert len(task.read_json(tmp_path / 'index.json')['selection']) == 6
