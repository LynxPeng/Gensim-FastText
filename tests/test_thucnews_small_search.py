import math
from collections import Counter
from copy import deepcopy

import pytest

from gensim_lab.thucnews_small_data import proportional_quotas, retain, sampling_rank
from gensim_lab.thucnews_small_search import choose_new, load_configuration, rank_three, update_selection


def trajectory(scores):
    rules = load_configuration()['validation']
    state = {}
    for epoch, score in enumerate(scores, 1):
        assert not state.get('stop')
        state = update_selection(state, score, epoch, rules)
    return state


def test_initial_plateau_stops_at_three_but_rising_curve_continues():
    assert trajectory([.3, .3, .3])['stop']
    assert trajectory([.3, .4, .5])['decision'] == 'continue_upward_trend'
    assert not trajectory([.3, .4, .5])['stop']
    assert trajectory([.3, .4, .5, .5, .5, .5])['stop']


def test_recovering_curve_is_not_pruned_for_lagging_past_peak():
    state = trajectory([.8, .2, .3, .4, .5])
    assert state['best_epoch'] == 1 and not state['stop']
    assert state['stale_epochs'] == 4 and state['rising']
    with pytest.raises(ValueError):
        trajectory([math.nan])


def test_small_improvement_still_updates_selected_checkpoint():
    state = trajectory([.3, .4, .5, .5005, .5006, .5007])
    assert state['stop'] and state['best_epoch'] == 6


def test_proportional_sampling_and_hash_selection_are_order_independent():
    quotas = proportional_quotas({'a': 80, 'b': 15, 'c': 5}, 21)
    assert sum(quotas.values()) == 21 and quotas == {'a': 17, 'b': 3, 'c': 1}
    identifiers = [str(n) for n in range(100)]
    def choose(order):
        heap = []
        for i in order:
            retain(heap, sampling_rank(i, 42), int(i), i, 10)
        return {item[2] for item in heap}
    assert choose(identifiers) == choose(identifiers[::-1])
    with pytest.raises(ValueError):
        proportional_quotas({'a': 2}, 3)


def test_selection_uses_validation_and_requires_three_fixed_seeds():
    cfg = load_configuration()
    rows = [{'method': 'word2vec', 'alpha': alpha, 'model_seed': seed, 'validation_score': score,
             'test_score': .99 if alpha == .01 else 0}
            for alpha, scores in [(.025, [.5, .5, .5]), (.01, [.8, .2, .2])]
            for seed, score in zip([42, 2026, 7], scores, strict=True)]
    assert rank_three(rows, 'word2vec', 'alpha', [.025, .01], cfg)['winner'] == .025
    assert choose_new(rows, 'word2vec', 'alpha', .025, lambda r: r['model_seed'] == 42) == .01
    with pytest.raises(ValueError):
        rank_three(rows[:-1], 'word2vec', 'alpha', [.025, .01], cfg)


def test_small_search_submission_preserves_user_chapters(tmp_path, monkeypatch):
    import gensim_lab.thucnews_small_search_report as reporting
    (tmp_path / 'reports').mkdir()
    path = tmp_path / 'reports/Gensim&FastText.md'
    text = '# 项目\n\n### 1.2 文本分类实验\n\n用户正文 $a+b$\n'
    path.write_text(text, encoding='utf-8')
    monkeypatch.setattr(reporting, 'ROOT', tmp_path)
    reporting.update_submission('小规模实验')
    assert path.read_text(encoding='utf-8').split('### 1.2', 1)[1] == text.split('### 1.2', 1)[1]
