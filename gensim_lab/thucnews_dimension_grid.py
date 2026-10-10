"""Three-seed embedding dimension search following frozen learning-rate selection."""
from __future__ import annotations

import argparse
import copy
import math
import statistics
from pathlib import Path

import yaml

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .thucnews_ablation import cache_valid, fixed_split, result_row
from .thucnews_alpha_grid import rank_grid as rank_alpha_grid
from .thucnews_small_search import BASE, StopRequested, launch_worker, make_spec, protocol, update_selection

CONFIG = ROOT / 'configs/thucnews_dimension_grid.yaml'


def rank_grid(rows, grid):
    expected = {(m, d, s) for m in grid['training_methods'] for d in grid['vector_sizes'] for s in grid['seeds']}
    observed = [(r['method'], r['vector_size'], r['model_seed']) for r in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError('Every dimension requires all three distinct seeds; duplicates are forbidden')
    ranking = {}
    for method, combination in grid['training_methods'].items():
        comparisons = []
        for dimension in grid['vector_sizes']:
            group = [r for r in rows if r['method'] == method and r['vector_size'] == dimension]
            if any(r['combination'] != combination or r['alpha'] != grid['learning_rates'][method] for r in group):
                raise ValueError('Training method and learning rate must remain fixed during dimension search')
            scores = [r['validation_score'] for r in group]
            if not all(math.isfinite(s) for s in scores):
                raise ValueError('Non-finite validation score')
            comparisons.append({'choice': dimension, 'mean': statistics.mean(scores),
                                'sd': statistics.stdev(scores), 'n': len(scores)})
        comparisons.sort(key=lambda r: (-r['mean'], r['choice']))
        ranking[method] = {'winner': comparisons[0]['choice'], 'comparisons': comparisons}
    return ranking


def prepare():
    grid = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert len(set(grid['vector_sizes'])) == len(grid['vector_sizes']) >= 5
    assert all(type(d) is int and d > 0 for d in grid['vector_sizes'])
    assert grid['seeds'] == [42, 2026, 7]
    assert grid['reference_vector_size'] in grid['vector_sizes']
    assert grid['exact_tie_break'] == 'smaller_vector_size'
    assert grid['expected_runs'] == len(grid['training_methods']) * len(grid['vector_sizes']) * len(grid['seeds'])
    source = ROOT / grid['source_experiment']
    previous, source_spec = read_json(source / 'index.json'), read_json(source / 'spec.json')
    assert previous['status'] == 'complete' and previous['kind'] == 'full_alpha_grid'
    assert rank_alpha_grid(previous['runs'], previous['grid']) == previous['alpha_ranking']
    assert {m: r['combination'] for m, r in previous['selected'].items()} == grid['training_methods']
    assert {m: r['alpha'] for m, r in previous['selected'].items()} == grid['learning_rates']
    cfg = previous['configuration']
    assert cfg == source_spec['configuration'] and cfg['embedding']['vector_size'] == grid['reference_vector_size']
    assert protocol() == source_spec['protocol_sha256']
    fixed_split(cfg)
    manifest_path = ROOT / previous['manifest']
    manifest = read_json(manifest_path)
    corpus = ROOT / manifest['directory'] / 'train.txt'
    corpus_sha = file_hash(corpus)
    old_rows = {r['run']: r for r in previous['runs']}
    specs = []
    for method, combination in grid['training_methods'].items():
        for dimension in grid['vector_sizes']:
            run_cfg = copy.deepcopy(cfg)
            run_cfg['embedding']['vector_size'] = dimension
            for seed in grid['seeds']:
                spec = make_spec(run_cfg, manifest_path, method, combination, seed, grid['learning_rates'][method],
                                 corpus, corpus_sha, manifest['partitions']['train'])
                run = (BASE / 'runs' / fingerprint(spec)[:24]).relative_to(ROOT).as_posix()
                if dimension == grid['reference_vector_size']:
                    assert run in old_rows and read_json(ROOT / run / 'spec.json') == spec
                specs.append({'method': method, 'vector_size': dimension, 'model_seed': seed, 'run': run, 'spec': spec})
    spec = {'kind': 'dimension_grid', 'grid': grid, 'configuration': cfg,
            'source_index_sha256': file_hash(source / 'index.json'), 'manifest': previous['manifest'],
            'manifest_sha256': file_hash(manifest_path), 'corpus_sha256': corpus_sha,
            'protocol_sha256': protocol(), 'controller_sha256': file_hash(Path(__file__)), 'run_specs': specs}
    experiment = BASE / 'experiments' / fingerprint(spec)[:24]
    experiment.mkdir(parents=True, exist_ok=True)
    if (experiment / 'spec.json').exists():
        assert read_json(experiment / 'spec.json') == spec
    else:
        write_json(experiment / 'spec.json', spec)
        write_json(experiment / 'index.json', {'kind': 'dimension_grid', 'configuration': cfg, 'grid': grid,
                   'manifest': previous['manifest'], 'smoke': False, 'source_experiment': grid['source_experiment'],
                   'experiment': experiment.relative_to(ROOT).as_posix(), 'runs': [], 'status': 'prepared',
                   'phase': 'dimension_grid', 'expected_search_runs': grid['expected_runs'], 'started': stamp()})
    write_json(BASE / 'latest.json', {'experiment': experiment.relative_to(ROOT).as_posix(), 'smoke': False})
    return experiment


def run():
    experiment = prepare()
    plan, spec = read_json(experiment / 'index.json'), read_json(experiment / 'spec.json')
    old_rows = {r['run']: r for r in read_json(ROOT / plan['source_experiment'] / 'index.json')['runs']}
    stop = BASE / 'stop.request.json'
    if stop.exists() and read_json(stop)['experiment'] == plan['experiment']:
        stop.unlink()
    plan.update(status='running', phase='dimension_grid')
    write_json(experiment / 'index.json', plan)
    try:
        # Check and register the six references before training new dimensions.
        tasks = sorted(spec['run_specs'], key=lambda r: r['vector_size'] != plan['grid']['reference_vector_size'])
        for item in tasks:
            directory = ROOT / item['run']
            directory.mkdir(parents=True, exist_ok=True)
            if (directory / 'spec.json').exists():
                assert read_json(directory / 'spec.json') == item['spec']
            else:
                write_json(directory / 'spec.json', item['spec'])
            progress = {'experiment': plan['experiment'], 'kind': 'dimension_grid', 'phase': 'dimension_grid',
                        'completed': len(plan['runs']), 'expected': plan['expected_search_runs'],
                        'run': item['run'], 'method': item['method'], 'vector_size': item['vector_size'],
                        'alpha': item['spec']['embedding']['alpha'], 'model_seed': item['model_seed'], 'updated': stamp()}
            write_json(BASE / 'progress.json', progress)
            if stop.exists() and read_json(stop)['experiment'] == plan['experiment']:
                raise StopRequested('Paused before next dimension configuration')
            if not cache_valid(directory, item['spec']):
                print(f'TRAIN {item["method"]} dimension={item["vector_size"]} seed={item["model_seed"]}', flush=True)
                launch_worker(directory)
                assert cache_valid(directory, item['spec'])
            row = result_row(directory)
            row.update(vector_size=item['vector_size'], alpha=item['spec']['embedding']['alpha'],
                       search_phase='dimension_grid', reused_from_previous=item['run'] in old_rows,
                       stop_reason=read_json(directory / 'summary.json')['stop_reason'])
            matches = [r for r in plan['runs'] if r['run'] == row['run']]
            if matches:
                assert matches == [row]
            else:
                plan['runs'].append(row)
            write_json(experiment / 'index.json', plan)
            print(f'{len(plan["runs"])}/{plan["expected_search_runs"]} {row["method"]} dimension={row["vector_size"]} '
                  f'seed={row["model_seed"]} rho={row["validation_score"]:.6f} '
                  f'epoch={row["selected_epoch"]}/{row["trained_epochs"]} reused={row["reused_from_previous"]}', flush=True)
        plan['dimension_ranking'] = rank_grid(plan['runs'], plan['grid'])
        plan['selected'] = {m: {'combination': plan['grid']['training_methods'][m],
                            'alpha': plan['grid']['learning_rates'][m], 'vector_size': rank['winner'],
                            'validation_mean': rank['comparisons'][0]['mean'],
                            'validation_sd': rank['comparisons'][0]['sd'], 'n': 3}
                            for m, rank in plan['dimension_ranking'].items()}
        plan.update(status='complete', phase='complete', completed=stamp())
    except StopRequested as exc:
        plan.update(status='paused', pause_reason=str(exc), updated=stamp())
        print(str(exc), flush=True)
    except BaseException as exc:
        plan.update(status='failed', failure=str(exc), updated=stamp())
        raise
    finally:
        write_json(experiment / 'index.json', plan)
        write_json(BASE / 'progress.json', {'experiment': plan['experiment'], 'kind': 'dimension_grid',
                   'phase': plan['phase'], 'status': plan['status'], 'completed': len(plan['runs']),
                   'expected': plan['expected_search_runs'], 'updated': stamp()})
    if plan['status'] == 'complete':
        audit(experiment)


def audit(experiment):
    experiment = Path(experiment)
    plan, spec = read_json(experiment / 'index.json'), read_json(experiment / 'spec.json')
    assert plan['status'] == 'complete' and rank_grid(plan['runs'], plan['grid']) == plan['dimension_ranking']
    assert protocol() == spec['protocol_sha256']
    assert file_hash(ROOT / plan['source_experiment'] / 'index.json') == spec['source_index_sha256']
    assert file_hash(ROOT / plan['manifest']) == spec['manifest_sha256']
    assert file_hash(ROOT / spec['run_specs'][0]['spec']['training_corpus']) == spec['corpus_sha256']
    vocabularies, frequencies, coverage, epochs = set(), set(), set(), 0
    expected_specs = {i['run']: i['spec'] for i in spec['run_specs']}
    for row in plan['runs']:
        directory = ROOT / row['run']
        run_spec = read_json(directory / 'spec.json')
        assert run_spec == expected_specs[row['run']] and cache_valid(directory, run_spec)
        assert run_spec['embedding']['vector_size'] == row['vector_size']
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
            primary = next(m for m in epoch['metrics'] if m['benchmark'] == 'wordsim240')
            coverage.add((primary['covered'], primary['total']))
            epochs += 1
        assert selection == state['selection']
        assert row['validation_score'] == selection['best_score'] and row['selected_epoch'] == selection['best_epoch']
        vocabularies.add(row['vocabulary_sha256'])
        frequencies.add(row['frequency_sha256'])
    assert len(vocabularies) == len(frequencies) == len(coverage) == 1
    save_csv(experiment / 'results.csv', plan['runs'])
    write_json(experiment / 'verification.json', {'status': 'passed', 'search_runs': len(plan['runs']),
               'dimensions_per_model': len(plan['grid']['vector_sizes']), 'seeds_per_dimension': 3,
               'test_runs': 0, 'complete_epochs': epochs, 'reused_runs': sum(r['reused_from_previous'] for r in plan['runs']),
               'source_experiment_preserved': True, 'protocol_unchanged': True,
               'identical_vocabularies_frequencies_and_coverage': True,
               'selection_checkpoints_and_reload_verified': True, 'checked': stamp()})
    print('AUDIT PASSED: complete balanced 2 × 5 × 3 dimension grid', flush=True)


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
            from .thucnews_dimension_grid_report import report
            report(experiment)


if __name__ == '__main__':
    main()
