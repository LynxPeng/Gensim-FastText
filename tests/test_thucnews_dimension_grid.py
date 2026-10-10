import copy

import pytest

from gensim_lab.thucnews_dimension_grid import rank_grid
from gensim_lab.thucnews_dimension_grid_report import START, END, replace_block


@pytest.fixture
def grid():
    return {'training_methods': {'word2vec': 'sg_ns5', 'fasttext': 'sg_ns10'},
            'learning_rates': {'word2vec': .025, 'fasttext': .05},
            'vector_sizes': [50, 100, 150, 200, 300], 'seeds': [42, 2026, 7]}


@pytest.fixture
def rows(grid):
    return [{'method': m, 'combination': c, 'alpha': grid['learning_rates'][m],
             'vector_size': d, 'model_seed': s, 'validation_score': .5 if d == 100 else .4}
            for m, c in grid['training_methods'].items() for d in grid['vector_sizes'] for s in grid['seeds']]


def test_incomplete_duplicate_or_extra_grid_blocks_selection(grid, rows):
    for candidate in [rows[:-1], [*rows, rows[0]], [{**rows[0], 'vector_size': 999}, *rows[1:]]]:
        with pytest.raises(ValueError, match='three distinct seeds'):
            rank_grid(candidate, grid)


def test_mean_wins_instead_of_single_seed_peak_and_exact_tie_prefers_lower_dimension(grid, rows):
    for row in rows:
        if row['method'] == 'word2vec' and row['vector_size'] == 300:
            row['validation_score'] = .99 if row['model_seed'] == 42 else .1
        if row['method'] == 'fasttext' and row['vector_size'] in [50, 100]:
            row['validation_score'] = .5
    ranking = rank_grid(rows, grid)
    assert ranking['word2vec']['winner'] == 100
    assert ranking['fasttext']['winner'] == 50


def test_changed_training_method_learning_rate_and_nonfinite_score_rejected(grid, rows):
    for key, value in [('combination', 'cbow_hs'), ('alpha', .1), ('validation_score', float('nan'))]:
        changed = copy.deepcopy(rows)
        changed[0][key] = value
        with pytest.raises(ValueError):
            rank_grid(changed, grid)


def test_sample_sd_and_each_model_selected_independently(grid, rows):
    for row in rows:
        if row['method'] == 'fasttext' and row['vector_size'] == 200:
            row['validation_score'] = {42: .6, 2026: .7, 7: .8}[row['model_seed']]
    ranking = rank_grid(rows, grid)
    assert ranking['word2vec']['winner'] == 100
    assert ranking['fasttext']['winner'] == 200
    result = ranking['fasttext']['comparisons'][0]
    assert result['mean'] == pytest.approx(.7)
    assert result['sd'] == pytest.approx(.1)
    assert result['n'] == 3


def test_report_insertion_preserves_editorial_prose_and_updates_without_duplicate_sections():
    original = ('prefix\n<!-- thucnews-ablation:start -->\n'
                '#### 1.1.3 超参数搜索\napproved methods and learning-rate prose\n'
                '##### 1.1.3.3 候选参数与结论范围\napproved scope\n'
                '<!-- thucnews-ablation:end -->\nsuffix')
    first = replace_block(original, 'dimension results')
    assert 'approved methods and learning-rate prose' in first and 'approved scope' in first
    assert '##### 1.1.3.4 候选参数与结论范围' in first
    assert first.startswith('prefix') and first.endswith('suffix')
    second = replace_block(first, 'updated dimension results')
    assert second.count(START) == second.count(END) == 1
    assert 'updated dimension results' in second
    assert second == replace_block(second, 'updated dimension results')
