"""Complete five-rate, three-seed validation grid with sealed-run reuse."""
from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

import yaml

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .thucnews_ablation import cache_valid, result_row
from .thucnews_small_search import BASE, StopRequested, launch_worker, make_spec, protocol, update_selection

CONFIG = ROOT / 'configs/thucnews_alpha_grid.yaml'


def rank_grid(rows, grid):
    """Require the complete balanced grid before choosing any learning rate."""
    expected = {(m, a, s) for m in grid['training_methods'] for a in grid['learning_rates'] for s in grid['seeds']}
    observed = [(r['method'], r['alpha'], r['model_seed']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Every model/rate requires all three distinct seeds; duplicates are forbidden')
    ranking = {}
    for method in grid['training_methods']:
        comparisons = []
        for alpha in grid['learning_rates']:
            group = [r for r in rows if r['method'] == method and r['alpha'] == alpha]
            if any(r['combination'] != grid['training_methods'][method] for r in group):
                raise ValueError('Training methods must remain fixed during learning-rate comparison')
            scores = [r['validation_score'] for r in group]
            if not all(math.isfinite(v) for v in scores):
                raise ValueError('Non-finite validation score')
            comparisons.append({'choice': alpha, 'mean': statistics.mean(scores),
                                'sd': statistics.stdev(scores), 'n': len(scores)})
        comparisons.sort(key=lambda r: (-r['mean'], r['choice'] != grid['reference_alpha'], r['choice']))
        ranking[method] = {'winner': comparisons[0]['choice'], 'comparisons': comparisons}
    return ranking


def prepare():
    grid = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert len(set(grid['learning_rates'])) >= 5 and grid['seeds'] == [42, 2026, 7]
    assert all(math.isfinite(a) and a > 0 for a in grid['learning_rates'])
    assert grid['expected_runs'] == len(grid['training_methods']) * len(grid['learning_rates']) * len(grid['seeds'])
    source = ROOT / grid['source_experiment']
    previous = read_json(source / 'index.json')
    assert previous['status'] == 'complete'
    cfg = read_json(source / 'spec.json')['configuration']
    assert cfg == previous['configuration']
    assert protocol() == read_json(source / 'spec.json')['protocol_sha256']
    assert {m: r['winner'] for m, r in previous['matrix_ranking'].items()} == grid['training_methods']
    manifest_path = ROOT / previous['manifest']
    manifest = read_json(manifest_path)
    corpus = ROOT / manifest['directory'] / 'train.txt'
    corpus_sha = file_hash(corpus)
    specs = []
    for method, combination in grid['training_methods'].items():
        for alpha in grid['learning_rates']:
            for seed in grid['seeds']:
                spec = make_spec(cfg, manifest_path, method, combination, seed, alpha,
                                 corpus, corpus_sha, manifest['partitions']['train'])
                specs.append({'method': method, 'alpha': alpha, 'model_seed': seed,
                              'run': (BASE / 'runs' / fingerprint(spec)[:24]).relative_to(ROOT).as_posix(), 'spec': spec})
    old_rows = {r['run']: r for r in previous['runs']}
    for item in specs:
        if item['run'] in old_rows:
            assert read_json(ROOT / item['run'] / 'spec.json') == item['spec']
    spec = {'kind': 'full_alpha_grid', 'grid': grid, 'source_index_sha256': file_hash(source / 'index.json'),
            'configuration': cfg, 'manifest': previous['manifest'], 'manifest_sha256': file_hash(manifest_path),
            'corpus_sha256': corpus_sha, 'protocol_sha256': protocol(),
            'controller_sha256': file_hash(Path(__file__)), 'run_specs': specs}
    experiment = BASE / 'experiments' / fingerprint(spec)[:24]
    experiment.mkdir(parents=True, exist_ok=True)
    if (experiment / 'spec.json').exists():
        assert read_json(experiment / 'spec.json') == spec
    else:
        write_json(experiment / 'spec.json', spec)
        write_json(experiment / 'index.json', {'kind': 'full_alpha_grid', 'configuration': cfg, 'grid': grid,
                   'manifest': previous['manifest'], 'smoke': False, 'source_experiment': grid['source_experiment'],
                   'experiment': experiment.relative_to(ROOT).as_posix(), 'runs': [], 'status': 'prepared',
                   'phase': 'alpha_grid', 'expected_search_runs': grid['expected_runs'], 'started': stamp()})
    write_json(BASE / 'latest.json', {'experiment': experiment.relative_to(ROOT).as_posix(), 'smoke': False})
    return experiment


def run():
    experiment = prepare()
    plan = read_json(experiment / 'index.json')
    spec = read_json(experiment / 'spec.json')
    old_rows = {r['run']: r for r in read_json(ROOT / plan['source_experiment'] / 'index.json')['runs']}
    stop = BASE / 'stop.request.json'
    if stop.exists() and read_json(stop)['experiment'] == plan['experiment']:
        stop.unlink()  # An explicit run resumes this grid.
    plan.update(status='running', phase='alpha_grid')
    write_json(experiment / 'index.json', plan)
    try:
        for item in spec['run_specs']:
            directory = ROOT / item['run']
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'spec.json').exists():
                assert read_json(directory / 'spec.json') == item['spec']
            else:
                write_json(directory / 'spec.json', item['spec'])
            progress = {'experiment': plan['experiment'], 'kind': 'full_alpha_grid', 'phase': 'alpha_grid',
                        'completed': len(plan['runs']), 'expected': plan['expected_search_runs'],
                        'run': item['run'], 'method': item['method'], 'alpha': item['alpha'],
                        'model_seed': item['model_seed'], 'updated': stamp()}
            write_json(BASE / 'progress.json', progress)
            if stop.exists() and read_json(stop)['experiment'] == plan['experiment']:
                raise StopRequested('Paused before next configuration')
            if not cache_valid(directory, item['spec']):
                print(f'TRAIN {item["method"]} alpha={item["alpha"]} seed={item["model_seed"]}', flush=True)
                launch_worker(directory)
                assert cache_valid(directory, item['spec'])
            row = result_row(directory)
            row.update(alpha=item['alpha'], search_phase='alpha_grid', reused_from_previous=item['run'] in old_rows,
                       stop_reason=read_json(directory / 'summary.json')['stop_reason'])
            matches = [r for r in plan['runs'] if r['run'] == row['run']]
            if matches:
                assert matches == [row]
            else:
                plan['runs'].append(row)
            write_json(experiment / 'index.json', plan)
            print(f'{len(plan["runs"])}/30 {row["method"]} alpha={row["alpha"]} seed={row["model_seed"]} '
                  f'rho={row["validation_score"]:.6f} epoch={row["selected_epoch"]}/{row["trained_epochs"]} '
                  f'reused={row["reused_from_previous"]}', flush=True)
        plan['alpha_ranking'] = rank_grid(plan['runs'], plan['grid'])
        plan['selected'] = {m: {'combination': plan['grid']['training_methods'][m], 'alpha': rank['winner'],
                            'validation_mean': rank['comparisons'][0]['mean'],
                            'validation_sd': rank['comparisons'][0]['sd'], 'n': 3}
                            for m, rank in plan['alpha_ranking'].items()}
        plan.update(status='complete', phase='complete', completed=stamp())
    except StopRequested as exc:
        plan.update(status='paused', pause_reason=str(exc), updated=stamp())
        print(str(exc), flush=True)
    except BaseException as exc:
        plan.update(status='failed', failure=str(exc), updated=stamp())
        raise
    finally:
        write_json(experiment / 'index.json', plan)
        write_json(BASE / 'progress.json', {'experiment': plan['experiment'], 'kind': 'full_alpha_grid',
                   'phase': plan['phase'], 'status': plan['status'], 'completed': len(plan['runs']),
                   'expected': plan['expected_search_runs'], 'updated': stamp()})
    if plan['status'] == 'complete':
        audit(experiment)


def audit(experiment):
    experiment = Path(experiment)
    plan, spec = read_json(experiment / 'index.json'), read_json(experiment / 'spec.json')
    assert plan['status'] == 'complete' and rank_grid(plan['runs'], plan['grid']) == plan['alpha_ranking']
    assert protocol() == spec['protocol_sha256']
    assert file_hash(ROOT / plan['source_experiment'] / 'index.json') == spec['source_index_sha256']
    assert file_hash(ROOT / plan['manifest']) == spec['manifest_sha256']
    assert file_hash(ROOT / spec['run_specs'][0]['spec']['training_corpus']) == spec['corpus_sha256']
    vocabulary, frequencies, coverage, epochs = set(), set(), set(), 0
    expected_specs = {i['run']: i['spec'] for i in spec['run_specs']}
    for row in plan['runs']:
        directory = ROOT / row['run']
        run_spec = read_json(directory / 'spec.json')
        assert run_spec == expected_specs[row['run']] and cache_valid(directory, run_spec)
        summary, state = read_json(directory / 'summary.json'), read_json(directory / 'state.json')
        assert summary['test_evaluated'] is False and not (directory / 'test').exists()
        assert read_json(directory / 'save_load_check.json')['all_validation_metrics_equal']
        assert row['seal_sha256'] == file_hash(directory / 'seal.json')
        selection = {}
        for epoch in state['epochs']:
            assert not selection.get('stop')
            selection = update_selection(selection, epoch['primary_score'], epoch['epoch'], run_spec['validation'])
            assert epoch['raw_words'] == run_spec['expected_corpus']['tokens']
            assert all(m['part'] == 'validation' for m in epoch['metrics'])
            assert file_hash(directory / epoch['checkpoint']) == epoch['checkpoint_sha256']
            coverage.add(next(m['covered'] for m in epoch['metrics'] if m['benchmark'] == 'wordsim240'))
            epochs += 1
        assert selection == state['selection']
        assert row['validation_score'] == selection['best_score'] and row['selected_epoch'] == selection['best_epoch']
        vocabulary.add(row['vocabulary_sha256'])
        frequencies.add(row['frequency_sha256'])
    assert len(vocabulary) == len(frequencies) == len(coverage) == 1
    save_csv(experiment / 'results.csv', plan['runs'])
    write_json(experiment / 'verification.json', {'status': 'passed', 'search_runs': len(plan['runs']),
               'rates_per_model': 5, 'seeds_per_rate': 3, 'test_runs': 0, 'complete_epochs': epochs,
               'reused_runs': sum(r['reused_from_previous'] for r in plan['runs']),
               'source_experiment_preserved': True, 'protocol_unchanged': True,
               'identical_vocabularies_frequencies_and_coverage': True,
               'selection_checkpoints_and_reload_verified': True, 'checked': stamp()})
    print('AUDIT PASSED: complete balanced 2 × 5 × 3 grid', flush=True)


def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'audit', 'report'], default='run')
    args = parser.parse_args()
    if args.stage in ['prepare', 'run']:
        with artifact_lock():
            print(prepare()) if args.stage == 'prepare' else run()
    else:
        experiment = ROOT / read_json(BASE / 'latest.json')['experiment']
        if args.stage == 'audit':
            audit(experiment)
        else:
            from .thucnews_alpha_grid_report import report
            report(experiment)


if __name__ == '__main__':
    main()
