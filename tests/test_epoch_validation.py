import csv
import math
from pathlib import Path

import pytest
import yaml

from gensim_lab.common import ROOT, read_json
from gensim_lab.thucnews_validation import question_key, partition, update_selection


def test_equivalent_questions_share_partition():
    pair = question_key('similarity', ['Ａbc', '北京'])
    assert pair == question_key('similarity', ['北京', 'ABC'])
    original = question_key('analogy', ['中国', '北京', '日本', '东京'])
    for words in [['日本','东京','中国','北京'],['北京','中国','东京','日本'],['东京','日本','北京','中国']]:
        assert original == question_key('analogy', words)
        assert partition(original,42,.4) == partition(question_key('analogy',words),42,.4)


def test_primary_improvements_patience_and_actual_best_checkpoint():
    rules = {'min_epochs':5,'patience':3,'min_delta':.001}
    state = {}
    for epoch, value in enumerate([.4,.45,.45,.449,.4505],1):
        state = update_selection(state,value,epoch,rules)
        if epoch < 5:
            assert not state['stop']
    assert state['stop'] and state['stale_epochs'] == 3
    # A tiny gain can be the actual best model while not resetting patience.
    assert state['best_epoch'] == 5 and state['best_score'] == .4505
    with pytest.raises(ValueError):
        update_selection(state,math.nan,6,rules)


def test_completed_smoke_runs_have_each_checkpoint_and_disjoint_partitions():
    directories = []
    for path in (ROOT/'artifacts/thucnews_validation/runs').glob('*/summary.json'):
        summary = read_json(path)
        if summary['spec']['smoke'] and summary['spec']['embedding']['vector_size'] == 16:
            directories.append((path.parent,summary))
    if {s['spec']['method'] for _,s in directories} != {'word2vec','fasttext'}:
        pytest.skip('运行两个原生模型冒烟后可检查缓存检查点；选择规则由独立单元测试覆盖')
    for directory,summary in directories:
        rows = read_json(directory/'epochs.json')
        assert len(rows) == summary['checkpoints']
        assert (directory/rows[summary['selected_epoch']-1]['checkpoint']).exists()
        assert all((directory/r['checkpoint']).exists() for r in rows)
        assert all('training_loss' not in r for r in rows)
        assert read_json(directory/'save_load_check.json')['all_validation_metrics_equal']
        split = read_json(ROOT/summary['spec']['benchmark_manifest'])
        with (ROOT/split['directory']/'assignments.csv').open(encoding='utf-8-sig') as stream:
            assignments=list(csv.DictReader(stream))
        validation={r['question_key'] for r in assignments if r['part']=='validation'}
        test={r['question_key'] for r in assignments if r['part']=='test'}
        assert not validation & test
