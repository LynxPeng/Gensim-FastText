"""Validation-only 50k-article training-method and learning-rate search.

The old full-corpus experiment is preserved. No test evaluation or full-corpus
training is performed by this entry point. Every configuration starts fresh.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from gensim.models.word2vec import LineSentence
from threadpoolctl import threadpool_limits

from .common import ROOT, PeakMemory, environment, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .runtime import numerical_threads, worker_environment
from .thucnews_ablation import cache_valid, fixed_split, model_class, result_row, seal, validate_model
from .thucnews_small_data import prepare
from .thucnews_validation import evaluate_partition, save_model

BASE = ROOT / 'artifacts/thucnews_small_search'
CONFIG = ROOT / 'configs/thucnews_small_search.yaml'
PHASES = {'matrix_screen': '训练方式初筛', 'matrix_replication': '训练方式三种子复验',
          'alpha_screen': '学习率初筛', 'alpha_replication': '学习率三种子复验', 'complete': '搜索完成'}


class StopRequested(Exception):
    pass


def load_configuration(smoke=False):
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    assert cfg['models'] == ['word2vec', 'fasttext']
    assert len({c['name'] for c in cfg['combinations']}) == 6
    assert {(c['sg'], c['hs'], c['negative']) for c in cfg['combinations']} == {
        (sg, hs, n) for sg in [0, 1] for hs, n in [(1, 0), (0, 5), (0, 10)]}
    assert len(set([cfg['screen_seed'], *cfg['replicate_seeds']])) == 3
    assert cfg['embedding']['alpha'] in cfg['learning_rates'] and len(set(cfg['learning_rates'])) == 3
    if smoke:
        cfg['embedding'].update(vector_size=16, min_count=2, workers=4)
        cfg['fasttext']['bucket'] = 5000
        cfg['validation'].update(min_epochs=2, max_epochs=2, patience=2, minimum_primary_coverage=0)
    return cfg


def protocol():
    names = ['thucnews_small_search.py', 'thucnews_small_data.py', 'thucnews_ablation.py',
             'thucnews_validation.py', 'benchmark_metrics.py', 'evaluation_support.py',
             'runtime.py', 'common.py', 'integrity.py']
    return {f'gensim_lab/{n}': file_hash(ROOT / 'gensim_lab' / n) for n in names}


def update_selection(state, score, epoch, rules):
    """Keep rising curves; stop flat initial curves or exhausted patience.

    'Rising' means a >min_delta gain over either the previous round or two
    rounds earlier. This is an operational rule, not a significance test.
    """
    if not math.isfinite(score):
        raise ValueError('Primary validation score must be finite')
    state = dict(state)
    scores = [*state.get('scores', []), float(score)]
    assert len(scores) == epoch
    if state.get('best_score') is None or score > state['best_score']:
        state.update(best_score=float(score), best_epoch=epoch)
    if state.get('reference_score') is None or score > state['reference_score'] + rules['min_delta']:
        state.update(reference_score=float(score), stale_epochs=0)
    else:
        state['stale_epochs'] = state.get('stale_epochs', 0) + 1
    gain1 = score - scores[-2] if len(scores) >= 2 else None
    gain2 = score - scores[-3] if len(scores) >= 3 else None
    rising = any(g is not None and g > rules['min_delta'] for g in [gain1, gain2])
    initial_flat = epoch == rules['min_epochs'] and not rising
    patience_exhausted = epoch >= rules['min_epochs'] and state['stale_epochs'] >= rules['patience'] and not rising
    stop = initial_flat or patience_exhausted
    decision = ('minimum_rounds' if epoch < rules['min_epochs'] else
                'initial_no_upward_trend' if initial_flat else
                'patience_without_upward_trend' if patience_exhausted else
                'continue_upward_trend' if rising else 'continue_within_patience')
    state.update(scores=scores, gain_one_epoch=gain1, gain_two_epochs=gain2,
                 rising=rising, stop=stop, decision=decision)
    return state


def make_spec(cfg, manifest_path, method, combination, seed, alpha, corpus, corpus_sha, expected, smoke=False):
    choice = next(c for c in cfg['combinations'] if c['name'] == combination)
    return {'method': method, 'combination': combination, 'model_seed': seed, 'split_seed': cfg['split_seed'],
            'smoke': smoke, 'stage': 'validation_only',
            'embedding': {**cfg['embedding'], **{k: choice[k] for k in ['sg', 'hs', 'negative']}, 'alpha': alpha},
            'subwords': cfg['fasttext'] if method == 'fasttext' else {}, 'validation': cfg['validation'],
            'manifest': manifest_path.relative_to(ROOT).as_posix(), 'manifest_sha256': file_hash(manifest_path),
            'training_corpus': corpus.relative_to(ROOT).as_posix(), 'training_corpus_sha256': corpus_sha,
            'expected_corpus': expected, 'benchmark_manifest': cfg['benchmark_manifest'],
            'benchmark_manifest_sha256': cfg['benchmark_manifest_sha256'],
            'protocol_sha256': protocol(), 'packages': environment()['packages']}


def stop_requested():
    path = BASE / 'stop.request.json'
    if not path.exists():
        return False
    progress = read_json(BASE / 'progress.json')
    return read_json(path)['experiment'] == progress['experiment']


def worker(directory):
    directory = Path(directory).resolve()
    assert directory.is_relative_to((BASE / 'runs').resolve())
    spec = read_json(directory / 'spec.json')
    assert spec['stage'] == 'validation_only'
    assert all(file_hash(ROOT / p) == sha for p, sha in spec['protocol_sha256'].items())
    for key in ['manifest', 'benchmark_manifest', 'training_corpus']:
        assert file_hash(ROOT / spec[key]) == spec[key + '_sha256']
    split = read_json(ROOT / spec['benchmark_manifest'])
    corpus, cls, rules = ROOT / spec['training_corpus'], model_class(spec['method']), spec['validation']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=numerical_threads()), PeakMemory() as memory:
        if (directory / 'state.json').exists():
            state = read_json(directory / 'state.json')
            assert state['spec_fingerprint'] == fingerprint(spec)
            assert file_hash(directory / state['checkpoint']) == state['checkpoint_sha256']
            model = cls.load(str(directory / state['checkpoint']))
            epochs, selection = state['epochs'], state['selection']
        else:
            model = cls(**spec['embedding'], **spec['subwords'], seed=spec['model_seed'], epochs=rules['max_epochs'])
            begin = time.perf_counter()
            model.build_vocab(corpus_file=str(corpus))
            state = {'spec_fingerprint': fingerprint(spec), 'vocabulary_seconds': time.perf_counter() - begin}
            epochs, selection = [], {}
        validate_model(model, spec)
        assert model.corpus_total_words == spec['expected_corpus']['tokens']
        assert model.corpus_count == spec['expected_corpus']['sentences']
        for epoch in range(len(epochs) + 1, rules['max_epochs'] + 1):
            if stop_requested():
                raise StopRequested('Stopped before next epoch')
            if selection.get('stop'):
                break
            alpha = spec['embedding']['alpha'] + (spec['embedding']['min_alpha'] - spec['embedding']['alpha']) * (epoch - 1) / rules['max_epochs']
            end_alpha = spec['embedding']['alpha'] + (spec['embedding']['min_alpha'] - spec['embedding']['alpha']) * epoch / rules['max_epochs']
            begin = time.perf_counter()
            effective, raw = model.train(corpus_iterable=LineSentence(str(corpus)), total_examples=model.corpus_count,
                                        total_words=model.corpus_total_words, epochs=1, start_alpha=alpha, end_alpha=end_alpha)
            train_seconds = time.perf_counter() - begin
            assert raw == model.corpus_total_words, 'Incomplete epoch'
            begin = time.perf_counter()
            metrics = evaluate_partition(model, split, 'validation', directory / 'validation' / f'epoch_{epoch:02d}')
            validation_seconds = time.perf_counter() - begin
            primary_metric = next(m for m in metrics if m['benchmark'] == rules['primary_benchmark'])
            assert primary_metric['coverage'] >= rules['minimum_primary_coverage']
            primary = primary_metric[rules['primary_metric']]
            selection = update_selection(selection, primary, epoch, rules)
            checkpoint = directory / 'checkpoints' / f'epoch_{epoch:02d}.gensim'
            begin = time.perf_counter()
            save_model(model, checkpoint)
            row = {'epoch': epoch, 'training_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'checkpoint_seconds': time.perf_counter() - begin, 'raw_words': raw, 'effective_words': effective,
                   'start_alpha': alpha, 'end_alpha': end_alpha, 'primary_score': primary,
                   'best_epoch_so_far': selection['best_epoch'], 'stale_epochs': selection['stale_epochs'],
                   'gain_one_epoch': selection['gain_one_epoch'], 'gain_two_epochs': selection['gain_two_epochs'],
                   'rising': selection['rising'], 'decision': selection['decision'],
                   'checkpoint': checkpoint.relative_to(directory).as_posix(), 'checkpoint_sha256': file_hash(checkpoint),
                   'metrics': metrics}
            epochs.append(row)
            state.update(epochs=epochs, selection=selection, checkpoint=row['checkpoint'],
                         checkpoint_sha256=row['checkpoint_sha256'], updated=stamp())
            write_json(directory / 'state.json', state)
            write_json(directory / 'epochs.json', epochs)
            save_csv(directory / 'epochs.csv', [{k: v for k, v in e.items() if k != 'metrics'} for e in epochs])
            save_csv(directory / 'validation_metrics.csv', [{'epoch': e['epoch'], **{k: v for k, v in m.items() if k != 'sections'}}
                     for e in epochs for m in e['metrics']])
            progress = read_json(BASE / 'progress.json')
            progress.update(status='running', epoch=epoch, best_epoch=selection['best_epoch'],
                            primary_score=primary, decision=selection['decision'], updated=stamp())
            write_json(BASE / 'progress.json', progress)
            print(f'{spec["method"]}/{spec["combination"]}/alpha{spec["embedding"]["alpha"]}/seed{spec["model_seed"]} '
                  f'epoch{epoch}: rho={primary:.6f}; {selection["decision"]}; train={train_seconds:.1f}s', flush=True)
            if stop_requested():
                raise StopRequested('Stopped after saving complete epoch')
        best = epochs[selection['best_epoch'] - 1]
        del model
        model = cls.load(str(directory / best['checkpoint']))
        validate_model(model, spec)
        rechecked = evaluate_partition(model, split, 'validation', directory / 'reload_validation')
        for a, b in zip(best['metrics'], rechecked, strict=True):
            for key in ['pearson_r', 'spearman_rho', 'conditional_accuracy', 'all_accuracy']:
                if a.get(key) is not None:
                    assert np.isclose(a[key], b[key], rtol=0, atol=1e-12)
        write_json(directory / 'save_load_check.json', {'all_validation_metrics_equal': True,
                   'selected_epoch': selection['best_epoch'], 'checkpoint_sha256': best['checkpoint_sha256']})
        summary = {'spec': spec, 'selection': selection, 'trained_epochs': len(epochs),
                   'selected_epoch': selection['best_epoch'], 'validation_score': selection['best_score'],
                   'validation_metrics': rechecked, 'selected_checkpoint': best['checkpoint'],
                   'selected_checkpoint_sha256': best['checkpoint_sha256'], 'test_evaluated': False,
                   'fit_seconds': sum(e['training_seconds'] for e in epochs),
                   'validation_seconds': sum(e['validation_seconds'] for e in epochs),
                   'checkpoint_seconds': sum(e['checkpoint_seconds'] for e in epochs),
                   'vocabulary_seconds': state['vocabulary_seconds'], 'vocabulary_size': len(model.wv),
                   'vocabulary_sha256': fingerprint(model.wv.index_to_key),
                   'frequency_sha256': fingerprint([(w, int(model.wv.get_vecattr(w, 'count'))) for w in model.wv.index_to_key]),
                   'model_bytes': (directory / best['checkpoint']).stat().st_size,
                   'stop_reason': selection['decision'] if selection['stop'] else 'max_epochs', 'completed': stamp()}
    summary['peak_rss_mib'] = memory.peak / 2**20
    write_json(directory / 'summary.json', summary)
    seal(directory, spec)


def choose_new(rows, method, key, reference, predicate=lambda r: True):
    choices = [r for r in rows if r['method'] == method and r[key] != reference and predicate(r)]
    if not choices or not all(math.isfinite(r['validation_score']) for r in choices):
        raise ValueError('Finite validation candidates required')
    return sorted(choices, key=lambda r: (-r['validation_score'], str(r[key])))[0][key]


def rank_three(rows, method, key, choices, cfg, predicate=lambda r: True):
    seeds = {cfg['screen_seed'], *cfg['replicate_seeds']}
    comparisons = []
    for choice in choices:
        group = [r for r in rows if r['method'] == method and r[key] == choice and predicate(r)]
        if len(group) != 3 or {r['model_seed'] for r in group} != seeds:
            raise ValueError('Three training seeds required for each candidate')
        scores = [r['validation_score'] for r in group]
        if not all(math.isfinite(s) for s in scores):
            raise ValueError('Non-finite candidate score')
        comparisons.append({'choice': choice, 'mean': statistics.mean(scores), 'sd': statistics.stdev(scores), 'n': 3})
    comparisons.sort(key=lambda r: (-r['mean'], r['choice'] != choices[0], str(r['choice'])))
    return {'winner': comparisons[0]['choice'], 'comparisons': comparisons}


def launch_worker(directory):
    with (directory / 'run.log').open('a', encoding='utf-8') as stream:
        process = subprocess.Popen([sys.executable, '-u', '-m', 'gensim_lab.thucnews_small_search', '--worker', str(directory)],
                                   cwd=ROOT, env=worker_environment(), stdout=stream, stderr=subprocess.STDOUT)
        write_json(BASE / 'processes.json', {'parent_pid': os.getpid(), 'worker_pid': process.pid,
                   'run': directory.relative_to(ROOT).as_posix(), 'started': stamp()})
        try:
            code = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if code == 75:
        raise StopRequested('Worker saved checkpoint and paused')
    if code:
        raise RuntimeError(f'Worker exited {code}: {directory / "run.log"}')


def run(smoke=False):
    cfg = load_configuration(smoke)
    fixed_split(cfg)
    manifest_path = prepare(load_configuration())
    manifest = read_json(manifest_path)
    corpus = ROOT / manifest['directory'] / 'train.txt'
    expected = manifest['partitions']['train']
    BASE.mkdir(parents=True, exist_ok=True)
    if smoke:
        corpus_smoke = BASE / 'smoke_train.txt'
        expected = {'sentences': 0, 'tokens': 0}
        with corpus.open(encoding='utf-8') as source, corpus_smoke.open('w', encoding='utf-8', newline='\n') as out:
            for i, line in enumerate(source):
                if i == 5000:
                    break
                out.write(line)
                expected['sentences'] += 1
                expected['tokens'] += len(line.split())
        corpus = corpus_smoke
    if shutil.disk_usage(ROOT).free < 40 * 2**30:
        raise RuntimeError('Reserve 40 GiB for all native checkpoints')
    corpus_sha = file_hash(corpus)
    experiment_spec = {'configuration': cfg, 'manifest_sha256': file_hash(manifest_path), 'smoke': smoke,
                       'protocol_sha256': protocol(), 'corpus_sha256': corpus_sha, 'packages': environment()['packages']}
    experiment = BASE / 'experiments' / fingerprint(experiment_spec)[:24]
    experiment.mkdir(parents=True, exist_ok=True)
    write_json(experiment / 'spec.json', experiment_spec)
    plan = (read_json(experiment / 'index.json') if (experiment / 'index.json').exists() else
            {'configuration': cfg, 'manifest': manifest_path.relative_to(ROOT).as_posix(), 'smoke': smoke,
             'experiment': experiment.relative_to(ROOT).as_posix(), 'runs': [], 'phase': 'matrix_screen', 'started': stamp()})
    plan['status'] = 'running'
    write_json(experiment / 'index.json', plan)
    write_json(BASE / ('smoke_latest.json' if smoke else 'latest.json'), {'experiment': plan['experiment'], 'smoke': smoke})
    stop_path = BASE / 'stop.request.json'
    if stop_path.exists() and read_json(stop_path)['experiment'] == plan['experiment']:
        stop_path.unlink()
    packages = environment()
    write_json(experiment / 'environment.json', packages)
    from .thucnews_small_search_report import report

    def train(method, combination, seed, alpha, phase):
        if stop_requested():
            raise StopRequested('Stopped before next configuration')
        spec = make_spec(cfg, manifest_path, method, combination, seed, alpha, corpus, corpus_sha, expected, smoke)
        directory = BASE / 'runs' / fingerprint(spec)[:24]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'spec.json', spec)
        plan.update(phase=phase, status='running', updated=stamp())
        write_json(experiment / 'index.json', plan)
        write_json(BASE / 'progress.json', {'status': 'starting', 'phase': phase, 'method': method,
                   'combination': combination, 'alpha': alpha, 'model_seed': seed, 'completed_search_runs': len(plan['runs']),
                   'expected_search_runs': 28, 'experiment': plan['experiment'], 'run': directory.relative_to(ROOT).as_posix(),
                   'smoke': smoke, 'updated': stamp()})
        if not cache_valid(directory, spec):
            print(f'Start {phase}: {method}/{combination}/alpha{alpha}/seed{seed}', flush=True)
            launch_worker(directory)
        assert cache_valid(directory, spec)
        row = {**result_row(directory), 'alpha': alpha, 'search_phase': phase,
               'stop_reason': read_json(directory / 'summary.json')['stop_reason']}
        if not any(r['run'] == row['run'] for r in plan['runs']):
            plan['runs'].append(row)
        write_json(experiment / 'index.json', plan)
        save_csv(experiment / 'search_results.csv', plan['runs'])
        if not smoke:
            report(experiment)
        print(f'Complete ({len(plan["runs"])}/28): {method}/{combination}/alpha{alpha}/seed{seed}; '
              f'rho={row["validation_score"]:.6f}; epochs={row["selected_epoch"]}/{row["trained_epochs"]}', flush=True)

    try:
        base_alpha, seed = cfg['embedding']['alpha'], cfg['screen_seed']
        for method in cfg['models']:
            for combination in cfg['combinations']:
                train(method, combination['name'], seed, base_alpha, 'matrix_screen')
        screen = [r for r in plan['runs'] if r['search_phase'] == 'matrix_screen']
        assert len(screen) == 12
        candidates = {m: choose_new(screen, m, 'combination', cfg['baseline']) for m in cfg['models']}
        plan['matrix_candidates'] = candidates
        write_json(experiment / 'matrix_screen_selection.json', {'stage': 'validation_only', 'candidates': candidates, 'created': stamp()})
        for method in cfg['models']:
            for combination in [cfg['baseline'], candidates[method]]:
                for model_seed in cfg['replicate_seeds']:
                    train(method, combination, model_seed, base_alpha, 'matrix_replication')
        matrix_rows = [r for r in plan['runs'] if r['search_phase'].startswith('matrix_')]
        matrix_rank = {m: rank_three(matrix_rows, m, 'combination', [cfg['baseline'], candidates[m]], cfg) for m in cfg['models']}
        plan['matrix_ranking'] = matrix_rank
        write_json(experiment / 'matrix_selection.json', {'stage': 'validation_only', 'ranking': matrix_rank, 'created': stamp()})
        for method in cfg['models']:
            for alpha in cfg['learning_rates']:
                if alpha != base_alpha:
                    train(method, matrix_rank[method]['winner'], seed, alpha, 'alpha_screen')
        alpha_candidates = {m: choose_new(plan['runs'], m, 'alpha', base_alpha,
                             lambda r, m=m: r['combination'] == matrix_rank[m]['winner'] and r['model_seed'] == seed)
                            for m in cfg['models']}
        plan['alpha_candidates'] = alpha_candidates
        write_json(experiment / 'alpha_screen_selection.json', {'stage': 'validation_only', 'candidates': alpha_candidates, 'created': stamp()})
        for method in cfg['models']:
            for model_seed in cfg['replicate_seeds']:
                train(method, matrix_rank[method]['winner'], model_seed, alpha_candidates[method], 'alpha_replication')
        ranking = {m: rank_three(plan['runs'], m, 'alpha', [base_alpha, alpha_candidates[m]], cfg,
                                lambda r, m=m: r['combination'] == matrix_rank[m]['winner']) for m in cfg['models']}
        selected = {m: {'combination': matrix_rank[m]['winner'], 'alpha': ranking[m]['winner'],
                       'validation_mean': ranking[m]['comparisons'][0]['mean'], 'validation_sd': ranking[m]['comparisons'][0]['sd'],
                       'n': 3} for m in cfg['models']}
        plan.update(alpha_ranking=ranking, selected=selected, phase='complete', status='complete', completed=stamp())
        write_json(experiment / 'index.json', plan)
        write_json(experiment / 'selection.json', {'status': 'frozen_small_scale_search', 'selected': selected,
                   'sample_documents': manifest['partitions']['train']['documents'], 'search_runs': len(plan['runs']),
                   'test_evaluated': False, 'full_scale_training_started': False,
                   'next_step': 'Validate shortlisted configurations at larger scale; reselect epoch count at full scale',
                   'interpretation': 'Exploratory validation search; rankings may change with corpus scale', 'created': stamp()})
        audit(experiment)
        write_json(BASE / 'progress.json', {'status': 'complete', 'phase': 'complete', 'completed_search_runs': len(plan['runs']),
                   'expected_search_runs': 28, 'experiment': plan['experiment'], 'smoke': smoke, 'updated': stamp()})
        if not smoke:
            report(experiment)
            from .thucnews_small_search_figures import build
            build(experiment)
        print(f'All small-scale search runs complete: {experiment}', flush=True)
    except BaseException as exc:
        paused = isinstance(exc, (StopRequested, KeyboardInterrupt))
        plan.update(status='paused' if paused else 'failed', error=str(exc), updated=stamp())
        write_json(experiment / 'index.json', plan)
        progress = read_json(BASE / 'progress.json') if (BASE / 'progress.json').exists() else {}
        progress.update(status=plan['status'], error=str(exc), updated=stamp())
        write_json(BASE / 'progress.json', progress)
        if not smoke:
            report(experiment)
        if not paused:
            raise


def audit(experiment):
    plan = read_json(Path(experiment) / 'index.json')
    cfg, rows = plan['configuration'], plan['runs']
    fixed_split(cfg)
    assert plan['status'] == 'complete' and len(rows) == 28
    assert len({r['vocabulary_sha256'] for r in rows}) == 1
    assert len({r['frequency_sha256'] for r in rows}) == 1
    assert len({(r['method'], r['combination'], r['alpha'], r['model_seed']) for r in rows}) == 28
    primary_coverage = set()
    for row in rows:
        directory = ROOT / row['run']
        spec = read_json(directory / 'spec.json')
        assert cache_valid(directory, spec) and spec['stage'] == 'validation_only'
        summary, state = read_json(directory / 'summary.json'), read_json(directory / 'state.json')
        assert summary['test_evaluated'] is False and not (directory / 'test').exists()
        selection = {}
        for epoch in state['epochs']:
            assert not selection.get('stop')
            selection = update_selection(selection, epoch['primary_score'], epoch['epoch'], spec['validation'])
            assert file_hash(directory / epoch['checkpoint']) == epoch['checkpoint_sha256']
            assert all(m['part'] == 'validation' for m in epoch['metrics'])
            assert epoch['raw_words'] == spec['expected_corpus']['tokens']
            metric = next(m for m in epoch['metrics'] if m['benchmark'] == cfg['validation']['primary_benchmark'])
            primary_coverage.add(metric['covered'])
        assert selection == state['selection'] and row['validation_score'] == selection['best_score']
        assert read_json(directory / 'save_load_check.json')['all_validation_metrics_equal']
    assert len(primary_coverage) == 1
    write_json(Path(experiment) / 'verification.json', {'status': 'passed', 'search_runs': 28,
               'test_runs': 0, 'fixed_split_seed': cfg['split_seed'], 'identical_vocabularies_and_primary_coverage': True,
               'selection_and_checkpoints_recomputed': True, 'checked': stamp()})


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'report', 'audit', 'stop'], default='run')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        try:
            worker(args.worker)
        except StopRequested as exc:
            print(str(exc), flush=True)
            raise SystemExit(75)
    elif args.stage == 'prepare':
        print(prepare(load_configuration()))
    elif args.stage == 'run':
        with artifact_lock():
            run(args.smoke)
    elif args.stage == 'stop':
        progress = read_json(BASE / 'progress.json')
        write_json(BASE / 'stop.request.json', {'experiment': progress['experiment'], 'reason': 'user_requested_stop', 'requested': stamp()})
        print('Stop requested; worker saves current complete epoch, then exits.')
    else:
        experiment = ROOT / read_json(BASE / ('smoke_latest.json' if args.smoke else 'latest.json'))['experiment']
        if args.stage == 'audit':
            audit(experiment)
        else:
            from .thucnews_small_search_report import report
            report(experiment)


if __name__ == '__main__':
    main()
