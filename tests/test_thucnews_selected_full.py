"""Guard publication: a failed second model or audit must retain the baseline."""
import copy

import pytest
import yaml

from gensim_lab.common import read_json, write_json
from gensim_lab import thucnews_selected_full as selected


def test_changed_candidate_cannot_silently_retrain(tmp_path, monkeypatch):
    cfg = yaml.safe_load(selected.CONFIG.read_text(encoding='utf-8'))
    cfg['model_parameters']['word2vec']['alpha'] = 0.05
    path = tmp_path / 'changed_candidate.yaml'
    path.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    monkeypatch.setattr(selected, 'CONFIG', path)
    with pytest.raises(ValueError, match='Frozen candidate differs'):
        selected.load_configuration()


def fixture_plan(tmp_path, monkeypatch):
    base = tmp_path / 'artifacts/selected'
    official = tmp_path / 'artifacts/validation'
    experiment = base / 'experiments/test'
    experiment.mkdir(parents=True)
    old = {'models': {'word2vec': 'old_word2vec', 'fasttext': 'old_fasttext'}}
    write_json(official / 'latest.json', old)
    plan = {'configuration': {'models': ['word2vec', 'fasttext']},
            'specs': {'word2vec': {}, 'fasttext': {}},
            'runs': {'word2vec': 'runs/word2vec', 'fasttext': 'runs/fasttext'},
            'completed_models': [], 'experiment': 'artifacts/selected/experiments/test'}
    for run in plan['runs'].values():
        (tmp_path / run).mkdir(parents=True)
    monkeypatch.setattr(selected, 'ROOT', tmp_path)
    monkeypatch.setattr(selected, 'BASE', base)
    monkeypatch.setattr(selected, 'VALIDATION_BASE', official)
    monkeypatch.setattr(selected, 'prepare', lambda: (experiment, copy.deepcopy(plan)))
    return experiment, official, old


def test_second_training_failure_does_not_publish_mixed_models(tmp_path, monkeypatch):
    experiment, official, old = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(selected, 'cache_valid', lambda directory, spec: directory.name == 'word2vec')

    class FailedWorker:
        pid = 999999

        def __init__(self, *args, **kwargs):
            pass

        def wait(self):
            return 1

    monkeypatch.setattr(selected.subprocess, 'Popen', FailedWorker)
    with pytest.raises(RuntimeError, match='Training failed'):
        selected.run()
    assert read_json(official / 'latest.json') == old
    result = read_json(experiment / 'index.json')
    assert result['status'] == 'failed'
    assert result['completed_models'] == ['word2vec']


def test_audit_failure_does_not_publish_unverified_models(tmp_path, monkeypatch):
    experiment, official, old = fixture_plan(tmp_path, monkeypatch)
    monkeypatch.setattr(selected, 'cache_valid', lambda directory, spec: True)

    def failed_audit(experiment):
        raise ValueError('checkpoint mismatch')

    monkeypatch.setattr(selected, 'audit', failed_audit)
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        selected.run()
    assert read_json(official / 'latest.json') == old
    assert read_json(experiment / 'index.json')['status'] == 'failed'
