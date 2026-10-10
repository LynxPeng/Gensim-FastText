import math
from copy import deepcopy

import pytest

from gensim_lab.common import ROOT, write_json
from gensim_lab.thucnews_ablation import (fixed_split, frozen_entry, load_configuration,
    make_spec, pick_candidate, rank_replicates, replication_tasks, screen_tasks)


def test_training_seed_never_repartitions_benchmarks():
    cfg = load_configuration()
    split = fixed_split(cfg)
    assert split['benchmarks']['wordsim240']['validation']['n'] == 95
    corpus = ROOT / 'configs/thucnews_ablation.yaml'
    a = make_spec(cfg, 'word2vec', 'cbow_hs', 42, corpus, 'fixture')
    b = make_spec(cfg, 'word2vec', 'cbow_hs', 7, corpus, 'fixture')
    assert a['model_seed'] != b['model_seed']
    assert a['split_seed'] == b['split_seed'] == 42
    assert a['benchmark_manifest'] == b['benchmark_manifest']
    assert a['benchmark_manifest_sha256'] == b['benchmark_manifest_sha256']


def test_search_excludes_baseline_for_new_candidate_and_ignores_test_scores():
    rows = [{'method': 'word2vec', 'combination': c, 'validation_score': v, 'test_score': t}
            for c, v, t in [('cbow_hs', .9, 0), ('sg_ns5', .4, .1), ('sg_ns10', .3, .99)]]
    assert pick_candidate(rows, 'word2vec', 'cbow_hs') == 'sg_ns5'
    rows[1]['validation_score'] = math.nan
    with pytest.raises(ValueError):
        pick_candidate(rows, 'word2vec', 'cbow_hs')


def test_replicated_validation_mean_decides_winner_and_exact_twenty_runs():
    cfg = load_configuration()
    candidates = {'word2vec': 'sg_ns5', 'fasttext': 'cbow_ns5'}
    assert len(set(screen_tasks(cfg) + replication_tasks(cfg, candidates))) == 20
    rows = []
    for method in cfg['models']:
        for combination, scores in [(cfg['baseline'], [.5, .5, .5]), (candidates[method], [.8, .2, .2])]:
            for seed, score in zip([42, 2026, 7], scores, strict=True):
                rows.append({'method': method, 'combination': combination, 'model_seed': seed,
                             'validation_score': score, 'test_score': .99 if combination != cfg['baseline'] else 0})
    ranked = rank_replicates(rows, cfg, candidates)
    assert all(r['winner'] == 'cbow_hs' for r in ranked.values())
    assert all(r['comparisons'][0]['sd'] == 0 for r in ranked.values())
    with pytest.raises(ValueError):
        rank_replicates(rows[:-1], cfg, candidates)


def test_final_test_requires_complete_frozen_authorization(tmp_path):
    path = tmp_path / 'frozen.json'
    write_json(path, {'status': 'searching', 'search_completed': 19, 'test_runs': []})
    with pytest.raises(ValueError, match='forbidden'):
        frozen_entry(path, ROOT / 'artifacts/thucnews_ablation/runs/unselected')
    write_json(path, {'status': 'frozen', 'search_completed': 20, 'test_runs': []})
    with pytest.raises(ValueError, match='not authorized'):
        frozen_entry(path, ROOT / 'artifacts/thucnews_ablation/runs/unselected')


def test_submission_refresh_preserves_the_users_later_chapters(tmp_path, monkeypatch):
    import gensim_lab.thucnews_ablation_report as reporting
    folder = tmp_path / 'reports'
    folder.mkdir()
    path = folder / 'Gensim&FastText.md'
    original = '# 项目\n\n### 1.2 文本分类实验\n\n用户表格与公式 $a+b$\n'
    path.write_text(original, encoding='utf-8')
    monkeypatch.setattr(reporting, 'ROOT', tmp_path)
    reporting.update_submission('第一次')
    reporting.update_submission('第二次')
    actual = path.read_text(encoding='utf-8')
    assert actual.count(reporting.START) == 1
    assert '第一次' not in actual and '第二次' in actual
    assert actual.split('### 1.2', 1)[1] == original.split('### 1.2', 1)[1]
