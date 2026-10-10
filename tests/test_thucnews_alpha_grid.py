import copy

import pytest

from gensim_lab.thucnews_alpha_grid import rank_grid


@pytest.fixture
def grid():
    return {'training_methods': {'word2vec': 'sg_ns5', 'fasttext': 'sg_ns10'},
            'learning_rates': [.005, .01, .025, .05, .1], 'seeds': [42, 2026, 7], 'reference_alpha': .025}


@pytest.fixture
def rows(grid):
    return [{'method': m, 'combination': c, 'alpha': a, 'model_seed': s,
             'validation_score': .5 if a == .025 else .4}
            for m, c in grid['training_methods'].items() for a in grid['learning_rates'] for s in grid['seeds']]


def test_incomplete_or_duplicate_seed_blocks_selection(grid, rows):
    with pytest.raises(ValueError, match='three distinct seeds'):
        rank_grid(rows[:-1], grid)
    bad = copy.deepcopy(rows)
    bad[-1] = copy.deepcopy(bad[-2])
    with pytest.raises(ValueError, match='duplicates'):
        rank_grid(bad, grid)


def test_selects_three_seed_mean_instead_of_single_seed_peak(grid, rows):
    for r in rows:
        if r['method'] == 'word2vec' and r['alpha'] == .1:
            r['validation_score'] = .99 if r['model_seed'] == 42 else .1
        if r['method'] == 'fasttext' and r['alpha'] == .05:
            r['validation_score'] = .6
    ranks = rank_grid(rows, grid)
    assert ranks['word2vec']['winner'] == .025
    assert ranks['fasttext']['winner'] == .05
    assert all(c['n'] == 3 for rank in ranks.values() for c in rank['comparisons'])
    assert len(ranks['word2vec']['comparisons']) == 5


def test_exact_tie_prefers_reference_and_reports_sample_sd(grid, rows):
    for r in rows:
        if r['alpha'] == .01:
            r['validation_score'] = .5
        if r['alpha'] == .1:
            r['validation_score'] = {42: .1, 2026: .2, 7: .3}[r['model_seed']]
    ranks = rank_grid(rows, grid)
    assert ranks['word2vec']['winner'] == .025
    result = next(c for c in ranks['word2vec']['comparisons'] if c['choice'] == .1)
    assert result['mean'] == pytest.approx(.2)
    assert result['sd'] == pytest.approx(.1)


def test_nonfinite_or_changed_training_method_is_rejected(grid, rows):
    bad = copy.deepcopy(rows)
    bad[0]['validation_score'] = float('nan')
    with pytest.raises(ValueError, match='Non-finite'):
        rank_grid(bad, grid)
    bad = copy.deepcopy(rows)
    bad[0]['combination'] = 'cbow_hs'
    with pytest.raises(ValueError, match='remain fixed'):
        rank_grid(bad, grid)
