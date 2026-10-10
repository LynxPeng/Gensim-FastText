"""Resume the final matrix replication and finish without learning-rate search.

This controller preserves the sealed worker protocol and existing run IDs.
It deliberately cannot start any further parameter-search configuration.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import ROOT, file_hash, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .thucnews_ablation import cache_valid, result_row
from .thucnews_small_search import BASE, launch_worker, rank_three, update_selection


def latest():
    return ROOT / read_json(BASE / 'latest.json')['experiment']


def finalize(experiment):
    plan = read_json(experiment / 'index.json')
    cfg, rows = plan['configuration'], plan['runs']
    assert len(rows) == 20 and all(r['search_phase'].startswith('matrix_') for r in rows)
    assert all(r['alpha'] == cfg['embedding']['alpha'] for r in rows)
    rankings = {m: rank_three(rows, m, 'combination', [cfg['baseline'], plan['matrix_candidates'][m]], cfg)
                for m in cfg['models']}
    selected = {m: {'combination': rank['winner'], 'alpha': cfg['embedding']['alpha'],
                   'validation_mean': rank['comparisons'][0]['mean'],
                   'validation_sd': rank['comparisons'][0]['sd'], 'n': 3}
                for m, rank in rankings.items()}
    if plan.get('error'):
        plan.setdefault('recovered_errors', []).append(plan.pop('error'))
    plan.update(matrix_ranking=rankings, selected=selected, status='complete', phase='complete',
                scope='training_methods_only', expected_search_runs=20,
                learning_rate_search_deferred=True, completed=stamp())
    write_json(experiment / 'index.json', plan)
    write_json(experiment / 'matrix_selection.json', {'stage': 'validation_only', 'ranking': rankings, 'created': stamp()})
    write_json(experiment / 'selection.json', {'status': 'frozen_training_method_comparison', 'selected': selected,
               'sample_documents': 50000, 'search_runs': 20, 'learning_rate_search_deferred': True,
               'test_evaluated': False, 'full_scale_training_started': False,
               'interpretation': 'Exploratory validation search; alpha fixed, not optimized; rankings may change with corpus scale',
               'created': stamp()})
    save_csv(experiment / 'search_results.csv', rows)
    write_json(BASE / 'progress.json', {'status': 'complete', 'phase': 'complete', 'experiment': plan['experiment'],
               'completed_search_runs': 20, 'expected_search_runs': 20, 'scope': plan['scope'],
               'learning_rate_search_deferred': True, 'smoke': False, 'updated': stamp()})
    write_json(experiment / 'controller.json', {'entry': 'gensim_lab.thucnews_small_matrix',
               'controller_sha256': file_hash(Path(__file__)), 'worker_protocol_preserved': True, 'completed': stamp()})
    print(f'Complete: 20 training-method runs; learning-rate search deferred. {selected}', flush=True)


def audit(experiment):
    plan = read_json(experiment / 'index.json')
    cfg, rows = plan['configuration'], plan['runs']
    from .thucnews_ablation import fixed_split
    fixed_split(cfg)
    assert plan['status'] == 'complete' and plan['scope'] == 'training_methods_only' and len(rows) == 20
    assert len({(r['method'], r['combination'], r['alpha'], r['model_seed']) for r in rows}) == 20
    assert len({r['vocabulary_sha256'] for r in rows}) == len({r['frequency_sha256'] for r in rows}) == 1
    coverage, epoch_count = set(), 0
    for row in rows:
        directory = ROOT / row['run']
        spec = read_json(directory / 'spec.json')
        assert spec['stage'] == 'validation_only' and row['alpha'] == .025
        assert row['search_phase'].startswith('matrix_') and cache_valid(directory, spec)
        assert all(file_hash(ROOT / p) == sha for p, sha in spec['protocol_sha256'].items())
        summary, state = read_json(directory / 'summary.json'), read_json(directory / 'state.json')
        assert summary['test_evaluated'] is False and not (directory / 'test').exists()
        selection = {}
        for epoch in state['epochs']:
            assert not selection.get('stop')
            selection = update_selection(selection, epoch['primary_score'], epoch['epoch'], spec['validation'])
            assert file_hash(directory / epoch['checkpoint']) == epoch['checkpoint_sha256']
            assert epoch['raw_words'] == spec['expected_corpus']['tokens']
            assert all(m['part'] == 'validation' for m in epoch['metrics'])
            coverage.add(next(m for m in epoch['metrics'] if m['benchmark'] == 'wordsim240')['covered'])
            epoch_count += 1
        assert selection == state['selection'] and row['validation_score'] == selection['best_score']
        assert read_json(directory / 'save_load_check.json')['all_validation_metrics_equal']
    assert len(coverage) == 1 and coverage == {92}
    result = {'status': 'passed', 'search_runs': 20, 'saved_epochs': epoch_count, 'test_runs': 0,
              'learning_rate_search_runs': 0, 'full_scale_training_started': False,
              'fixed_split_seed': cfg['split_seed'], 'identical_vocabularies_and_primary_coverage': True,
              'selection_and_checkpoints_recomputed': True, 'worker_protocol_unchanged': True, 'checked': stamp()}
    write_json(experiment / 'verification.json', result)
    print(result, flush=True)


def resume_last(experiment):
    plan = read_json(experiment / 'index.json')
    amendment = read_json(experiment / 'scope_amendment.json')
    assert not plan['smoke'] and amendment['scope'] == 'training_methods_only'
    assert len(plan['runs']) in (19, 20)
    target = ROOT / amendment['target_run']
    spec = read_json(target / 'spec.json')
    assert (spec['method'], spec['combination'], spec['model_seed'], spec['embedding']['alpha']) == ('fasttext', 'sg_ns10', 7, .025)
    assert all(file_hash(ROOT / p) == sha for p, sha in spec['protocol_sha256'].items())
    if len(plan['runs']) == 19:
        stop = BASE / 'stop.request.json'
        if stop.exists() and read_json(stop)['experiment'] == plan['experiment']:
            stop.unlink()
        state = read_json(target / 'state.json')
        assert file_hash(target / state['checkpoint']) == state['checkpoint_sha256']
        write_json(experiment / 'last_replication_resume.json', {'run': amendment['target_run'],
                   'resume_after_epoch': len(state['epochs']), 'checkpoint_sha256': state['checkpoint_sha256'],
                   'reason': 'User requested continuing yesterday final replication', 'requested': stamp()})
        write_json(BASE / 'progress.json', {'status': 'starting', 'phase': 'matrix_replication',
                   'method': spec['method'], 'combination': spec['combination'], 'model_seed': spec['model_seed'],
                   'alpha': .025, 'completed_search_runs': 19, 'expected_search_runs': 20,
                   'experiment': plan['experiment'], 'run': amendment['target_run'], 'smoke': False, 'updated': stamp()})
        if not cache_valid(target, spec):
            launch_worker(target)
        assert cache_valid(target, spec)
        plan['runs'].append({**result_row(target), 'alpha': .025, 'search_phase': 'matrix_replication',
                             'stop_reason': read_json(target / 'summary.json')['stop_reason']})
        write_json(experiment / 'index.json', plan)
    finalize(experiment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['run', 'audit', 'report'], default='run')
    args = parser.parse_args()
    experiment = latest()
    if args.stage == 'run':
        with artifact_lock():
            resume_last(experiment)
    elif args.stage == 'audit':
        audit(experiment)
    else:
        from .thucnews_small_search_report import report
        print(report(experiment))


if __name__ == '__main__':
    main()
