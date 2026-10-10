"""Validation-only factorial search, fixed split, replications, then frozen testing.

Search artifacts are sealed independently of test artifacts. Changing a model
seed never changes the benchmark partition. No loss instrumentation is used.
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence
from threadpoolctl import threadpool_limits

from .common import ROOT, PeakMemory, environment, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .runtime import numerical_threads, worker_environment
from .thucnews_validation import coverage, evaluate_partition, save_model, update_selection

BASE = ROOT / 'artifacts/thucnews_ablation'
CONFIG = ROOT / 'configs/thucnews_ablation.yaml'


def load_configuration(smoke=False):
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if smoke:
        cfg['embedding'].update(vector_size=16, min_count=2, workers=4)
        cfg['fasttext']['bucket'] = 5000
        cfg['validation'].update(min_epochs=2, max_epochs=3, patience=2)
    assert cfg['screen_seed'] not in cfg['replicate_seeds']
    assert len(set(cfg['replicate_seeds'])) == len(cfg['replicate_seeds']) == 2
    assert cfg['models'] == ['word2vec', 'fasttext']
    assert len({c['name'] for c in cfg['combinations']}) == 6
    assert {(c['sg'], c['hs'], c['negative']) for c in cfg['combinations']} == {
        (sg, hs, negative) for sg in [0, 1] for hs, negative in [(1, 0), (0, 5), (0, 10)]}
    assert cfg['baseline'] in {c['name'] for c in cfg['combinations']}
    return cfg


def fixed_split(cfg):
    path = ROOT / cfg['benchmark_manifest']
    assert file_hash(path) == cfg['benchmark_manifest_sha256'], 'Benchmark manifest changed'
    split = read_json(path)
    assert split['spec']['seed'] == cfg['split_seed']
    assert split['spec']['validation']['primary_benchmark'] == cfg['validation']['primary_benchmark']
    assert split['overlap_groups'] == 0
    # Validate both partition files without computing any test metric.
    for entry in split['benchmarks'].values():
        for key in ['validation', 'validation_probe', 'test']:
            if key in entry:
                assert file_hash(ROOT / entry[key]['path']) == entry[key]['sha256']
    with (ROOT / split['directory'] / 'assignments.csv').open(encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    assert not ({r['question_key'] for r in rows if r['part'] == 'validation'} &
                {r['question_key'] for r in rows if r['part'] == 'test'})
    return split


def screen_tasks(cfg):
    return [(method, c['name'], cfg['screen_seed'])
            for method in cfg['models'] for c in cfg['combinations']]


def pick_candidate(rows, method, baseline):
    choices = [r for r in rows if r['method'] == method and r['combination'] != baseline]
    if not choices or any(not math.isfinite(r['validation_score']) for r in choices):
        raise ValueError('Candidate selection requires finite validation scores')
    # Stable lexical tie break, independent of execution order and test outcomes.
    return sorted(choices, key=lambda r: (-r['validation_score'], r['combination']))[0]['combination']


def replication_tasks(cfg, candidates):
    return [(method, combination, seed) for method in cfg['models']
            for combination in [cfg['baseline'], candidates[method]] for seed in cfg['replicate_seeds']]


def rank_replicates(rows, cfg, candidates):
    seeds = {cfg['screen_seed'], *cfg['replicate_seeds']}
    ranked = {}
    for method in cfg['models']:
        comparisons = []
        for combination in [cfg['baseline'], candidates[method]]:
            group = [r for r in rows if r['method'] == method and r['combination'] == combination]
            if len(group) != 3 or {r['model_seed'] for r in group} != seeds:
                raise ValueError('Freeze requires all three independent training seeds')
            scores = [r['validation_score'] for r in group]
            if not all(math.isfinite(s) for s in scores):
                raise ValueError('Non-finite validation score')
            comparisons.append({'combination': combination, 'n': 3, 'mean': statistics.mean(scores),
                                'sd': statistics.stdev(scores), 'seeds': sorted(seeds)})
        comparisons.sort(key=lambda r: (-r['mean'], r['combination'] != cfg['baseline'], r['combination']))
        ranked[method] = {'winner': comparisons[0]['combination'], 'comparisons': comparisons}
    return ranked


def protocol():
    return {f'gensim_lab/{name}': file_hash(ROOT / 'gensim_lab' / name) for name in
            ['thucnews_ablation.py', 'thucnews_validation.py', 'benchmark_metrics.py',
             'evaluation_support.py', 'runtime.py', 'common.py', 'integrity.py']}


def make_spec(cfg, method, combination, seed, corpus, corpus_sha, smoke=False):
    choice = next(c for c in cfg['combinations'] if c['name'] == combination)
    return {'method': method, 'combination': combination, 'model_seed': seed, 'split_seed': cfg['split_seed'],
            'smoke': smoke, 'stage': 'validation_only', 'embedding': {**cfg['embedding'],
                **{k: choice[k] for k in ['sg', 'hs', 'negative']}},
            'subwords': cfg['fasttext'] if method == 'fasttext' else {},
            'validation': cfg['validation'], 'manifest': cfg['manifest'],
            'manifest_sha256': file_hash(ROOT / cfg['manifest']),
            'training_corpus': corpus.relative_to(ROOT).as_posix(), 'training_corpus_sha256': corpus_sha,
            'benchmark_manifest': cfg['benchmark_manifest'],
            'benchmark_manifest_sha256': cfg['benchmark_manifest_sha256'],
            'protocol_sha256': protocol(), 'packages': environment()['packages']}


def seal(directory, spec):
    files = {p.relative_to(directory).as_posix(): file_hash(p) for p in directory.rglob('*')
             if p.is_file() and p.name not in ['run.log', 'seal.json']}
    write_json(directory / 'seal.json', {'spec_fingerprint': fingerprint(spec), 'files': files, 'completed': stamp()})


def cache_valid(directory, spec):
    path = directory / 'seal.json'
    if not path.exists():
        return False
    item = read_json(path)
    return item['spec_fingerprint'] == fingerprint(spec) and all(
        (directory / name).is_file() and file_hash(directory / name) == sha for name, sha in item['files'].items())


def model_class(method):
    return Word2Vec if method == 'word2vec' else FastText


def validate_model(model, spec):
    for name, value in spec['embedding'].items():
        if name in ['alpha', 'min_alpha']:
            continue  # train() stores the most recent start/end rates here.
        assert getattr(model, name) == value, f'Model parameter mismatch: {name}'
    assert spec['embedding']['min_alpha'] - 1e-12 <= model.min_alpha <= model.alpha <= spec['embedding']['alpha'] + 1e-12
    for name, value in spec['subwords'].items():
        assert getattr(model.wv, name) == value, f'Subword parameter mismatch: {name}'
    assert model.seed == spec['model_seed']
    assert model.compute_loss is False


def worker(directory):
    directory = Path(directory).resolve()
    assert directory.is_relative_to((BASE / 'runs').resolve())
    spec = read_json(directory / 'spec.json')
    assert spec['stage'] == 'validation_only'
    assert all(file_hash(ROOT / p) == sha for p, sha in spec['protocol_sha256'].items())
    assert file_hash(ROOT / spec['manifest']) == spec['manifest_sha256']
    assert file_hash(ROOT / spec['benchmark_manifest']) == spec['benchmark_manifest_sha256']
    split = read_json(ROOT / spec['benchmark_manifest'])
    assert split['spec']['seed'] == spec['split_seed']
    corpus = ROOT / spec['training_corpus']
    assert file_hash(corpus) == spec['training_corpus_sha256']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cls, rules = model_class(spec['method']), spec['validation']
    started = time.perf_counter()
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
        if not spec['smoke']:
            expected = read_json(ROOT / spec['manifest'])['partitions']['train']
            assert model.corpus_total_words == expected['tokens']
            assert model.corpus_count == expected['sentences']
        for epoch in range(len(epochs) + 1, rules['max_epochs'] + 1):
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
            primary = next(m for m in metrics if m['benchmark'] == rules['primary_benchmark'])[rules['primary_metric']]
            selection = update_selection(selection, primary, epoch, rules)
            checkpoint = directory / 'checkpoints' / f'epoch_{epoch:02d}.gensim'
            begin = time.perf_counter()
            save_model(model, checkpoint)
            row = {'epoch': epoch, 'training_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'checkpoint_seconds': time.perf_counter() - begin, 'raw_words': raw, 'effective_words': effective,
                   'start_alpha': alpha, 'end_alpha': end_alpha, 'primary_score': primary,
                   'best_epoch_so_far': selection['best_epoch'], 'stale_epochs': selection['stale_epochs'],
                   'checkpoint': checkpoint.relative_to(directory).as_posix(), 'checkpoint_sha256': file_hash(checkpoint),
                   'metrics': metrics}
            epochs.append(row)
            state.update(epochs=epochs, selection=selection, checkpoint=row['checkpoint'],
                         checkpoint_sha256=row['checkpoint_sha256'], updated=stamp())
            write_json(directory / 'state.json', state)
            write_json(directory / 'epochs.json', epochs)
            save_csv(directory / 'epochs.csv', [{k: v for k, v in r.items() if k != 'metrics'} for r in epochs])
            save_csv(directory / 'validation_metrics.csv', [{'epoch': r['epoch'], **{k: v for k, v in m.items() if k != 'sections'}}
                     for r in epochs for m in r['metrics']])
            progress = read_json(BASE / 'progress.json') if (BASE / 'progress.json').exists() else {}
            progress.update(status='running', method=spec['method'], combination=spec['combination'],
                            model_seed=spec['model_seed'], epoch=epoch, best_epoch=selection['best_epoch'],
                            primary_score=primary, run=directory.relative_to(ROOT).as_posix(), updated=stamp())
            write_json(BASE / 'progress.json', progress)
            print(f'{spec["method"]}/{spec["combination"]}/seed{spec["model_seed"]} epoch {epoch}: validation={primary:.6f}, best={selection["best_epoch"]}; train={train_seconds:.1f}s', flush=True)
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
                   'selected_checkpoint_sha256': best['checkpoint_sha256'],
                   'fit_seconds': sum(r['training_seconds'] for r in epochs),
                   'validation_seconds': sum(r['validation_seconds'] for r in epochs),
                   'checkpoint_seconds': sum(r['checkpoint_seconds'] for r in epochs),
                   'vocabulary_seconds': state['vocabulary_seconds'], 'vocabulary_size': len(model.wv),
                   'vocabulary_sha256': fingerprint(model.wv.index_to_key),
                   'frequency_sha256': fingerprint([(w, int(model.wv.get_vecattr(w, 'count'))) for w in model.wv.index_to_key]),
                   'model_bytes': (directory / best['checkpoint']).stat().st_size,
                   'stop_reason': 'validation_patience' if selection['stop'] else 'max_epochs',
                   'test_evaluated': False, 'completed': stamp(), 'process_elapsed_seconds': time.perf_counter() - started}
    summary['peak_rss_mib'] = memory.peak / 2**20
    assert not (directory / 'test').exists() and not (directory / 'metrics.json').exists()
    write_json(directory / 'summary.json', summary)
    seal(directory, spec)


def result_row(directory):
    summary = read_json(directory / 'summary.json')
    spec = summary['spec']
    return {**{k: spec[k] for k in ['method', 'combination', 'model_seed', 'split_seed']},
            **{k: summary[k] for k in ['validation_score', 'selected_epoch', 'trained_epochs', 'fit_seconds',
                 'validation_seconds', 'checkpoint_seconds', 'vocabulary_seconds', 'peak_rss_mib',
                 'model_bytes', 'vocabulary_size', 'vocabulary_sha256', 'frequency_sha256']},
            'run': directory.relative_to(ROOT).as_posix(), 'seal_sha256': file_hash(directory / 'seal.json'),
            'checkpoint': summary['selected_checkpoint'], 'checkpoint_sha256': summary['selected_checkpoint_sha256']}


def frozen_entry(frozen_path, directory):
    frozen = read_json(frozen_path)
    if frozen.get('status') != 'frozen' or frozen.get('search_completed') != 20:
        raise ValueError('Testing is forbidden before all 20 validation-only runs are frozen')
    path = directory.resolve().relative_to(ROOT.resolve()).as_posix()
    matches = [r for r in frozen['test_runs'] if r['run'] == path]
    if len(matches) != 1:
        raise ValueError('Run not authorized by the frozen validation selection')
    item = matches[0]
    assert file_hash(directory / 'seal.json') == item['seal_sha256']
    assert file_hash(directory / item['checkpoint']) == item['checkpoint_sha256']
    return frozen, item


def test_worker(directory, frozen_path):
    directory, frozen_path = Path(directory).resolve(), Path(frozen_path).resolve()
    assert directory.is_relative_to((BASE / 'runs').resolve())
    assert frozen_path.is_relative_to((BASE / 'experiments').resolve())
    frozen, item = frozen_entry(frozen_path, directory)
    spec = read_json(directory / 'spec.json')
    test_spec = {'search_run': item, 'frozen_manifest_sha256': file_hash(frozen_path),
                 'protocol_sha256': protocol(), 'stage': 'frozen_test'}
    target = frozen_path.parent / 'tests' / directory.name
    if cache_valid(target, test_spec):
        return
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / 'spec.json', test_spec)
    assert cache_valid(directory, spec)
    split = read_json(ROOT / spec['benchmark_manifest'])
    assert file_hash(ROOT / spec['benchmark_manifest']) == spec['benchmark_manifest_sha256']
    with threadpool_limits(limits=numerical_threads()):
        model = model_class(spec['method']).load(str(directory / item['checkpoint']))
        validate_model(model, spec)
        begin = time.perf_counter()
        metrics = evaluate_partition(model, split, 'test', target / 'test')
        manifest = read_json(ROOT / spec['manifest'])
        heldout = coverage(model, ROOT / manifest['directory'] / 'holdout.txt')
    write_json(target / 'metrics.json', metrics)
    write_json(target / 'summary.json', {'spec': test_spec, 'metrics': metrics, 'heldout': heldout,
        'method': spec['method'], 'combination': spec['combination'], 'model_seed': spec['model_seed'],
        'is_validation_winner': frozen['ranking'][spec['method']]['winner'] == spec['combination'],
        'selected_epoch': item['selected_epoch'], 'test_seconds': time.perf_counter() - begin,
        'score_api_status': 'HS context score supported; not a selection criterion' if spec['embedding']['hs'] else
                            'not applicable: Gensim score requires hierarchical softmax', 'completed': stamp()})
    seal(target, test_spec)


def launch_worker(module_args, log):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a', encoding='utf-8') as stream:
        result = subprocess.run([sys.executable, '-m', 'gensim_lab.thucnews_ablation', *module_args],
                                cwd=ROOT, env=worker_environment(), stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Worker exit {result.returncode}: {log}')


def run(smoke=False):
    cfg = load_configuration(smoke)
    fixed_split(cfg)
    manifest = read_json(ROOT / cfg['manifest'])
    corpus = ROOT / manifest['directory'] / 'train.txt'
    BASE.mkdir(parents=True, exist_ok=True)
    if smoke:
        target = BASE / 'smoke_train.txt'
        with corpus.open(encoding='utf-8') as source, target.open('w', encoding='utf-8', newline='\n') as output:
            for i, line in enumerate(source):
                if i == 20000:
                    break
                output.write(line)
        corpus = target
    elif shutil.disk_usage(ROOT).free < 120 * 2**30:
        raise RuntimeError('Reserve at least 120 GiB for all native epoch checkpoints')
    corpus_sha = file_hash(corpus)
    experiment_spec = {'configuration': cfg, 'smoke': smoke, 'protocol_sha256': protocol(),
                       'training_corpus_sha256': corpus_sha, 'packages': environment()['packages']}
    experiment = BASE / 'experiments' / fingerprint(experiment_spec)[:24]
    experiment.mkdir(parents=True, exist_ok=True)
    write_json(experiment / 'spec.json', experiment_spec)
    plan = {'phase': 'screen', 'configuration': cfg, 'smoke': smoke, 'experiment': experiment.relative_to(ROOT).as_posix(),
            'runs': [], 'started': stamp(), 'expected_search_runs': 20}
    if (experiment / 'index.json').exists():
        plan = read_json(experiment / 'index.json')
    else:
        write_json(experiment / 'index.json', plan)
        save_csv(experiment / 'search_results.csv', [], ['method', 'combination', 'model_seed', 'validation_score'])
    index_name = 'smoke_latest.json' if smoke else 'latest.json'
    write_json(BASE / index_name, {'experiment': experiment.relative_to(ROOT).as_posix(), 'smoke': smoke})
    if not smoke:
        from .thucnews_ablation_report import report
        report(experiment)

    def train(task, phase):
        method, combination, seed = task
        spec = make_spec(cfg, method, combination, seed, corpus, corpus_sha, smoke)
        directory = BASE / 'runs' / fingerprint(spec)[:24]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(directory / 'spec.json', spec)
        write_json(BASE / 'progress.json', {'status': 'starting', 'phase': phase, 'method': method,
                   'combination': combination, 'model_seed': seed, 'completed_search_runs': len(plan['runs']),
                   'expected_search_runs': 20, 'run': directory.relative_to(ROOT).as_posix(),
                   'experiment': plan['experiment'], 'smoke': smoke, 'updated': stamp()})
        if not cache_valid(directory, spec):
            print(f'Start {phase}: {method}/{combination}/seed{seed}', flush=True)
            launch_worker(['--worker', str(directory)], directory / 'run.log')
            assert cache_valid(directory, spec)
        row = result_row(directory)
        plan['runs'] = [r for r in plan['runs'] if r['run'] != row['run']] + [row]
        plan.update(phase=phase, updated=stamp())
        write_json(experiment / 'index.json', plan)
        save_csv(experiment / 'search_results.csv', plan['runs'])
        print(f'Complete ({len(plan["runs"])}/20): {method}/{combination}/seed{seed}; validation={row["validation_score"]:.6f}', flush=True)
        if not smoke:
            from .thucnews_ablation_report import report
            report(experiment)

    try:
        for task in screen_tasks(cfg):
            train(task, 'screen')
        screen = [r for r in plan['runs'] if r['model_seed'] == cfg['screen_seed']]
        assert len(screen) == 12
        candidates = {m: pick_candidate(screen, m, cfg['baseline']) for m in cfg['models']}
        decision = {'stage': 'validation_only', 'candidates': candidates, 'screen': screen,
                    'selection_metric': cfg['validation']['primary_metric'], 'created': stamp()}
        if (experiment / 'screen_selection.json').exists():
            old = read_json(experiment / 'screen_selection.json')
            assert old['candidates'] == candidates and old['screen'] == screen
        else:
            write_json(experiment / 'screen_selection.json', decision)
        plan['candidates'] = candidates
        write_json(experiment / 'index.json', plan)
        if not smoke:
            from .thucnews_ablation_figures import build
            build(experiment)
        for task in replication_tasks(cfg, candidates):
            train(task, 'replication')
        assert len(plan['runs']) == 20
        assert len({r['vocabulary_sha256'] for r in plan['runs']}) == 1
        assert len({r['frequency_sha256'] for r in plan['runs']}) == 1
        ranking = rank_replicates(plan['runs'], cfg, candidates)
        selected = [r for r in plan['runs'] if r['combination'] in [cfg['baseline'], candidates[r['method']]]]
        frozen_path = experiment / 'frozen.json'
        search_snapshot = {'configuration': cfg, 'runs': plan['runs'], 'candidates': candidates}
        if (experiment / 'search_snapshot.json').exists():
            assert read_json(experiment / 'search_snapshot.json') == search_snapshot
        else:
            write_json(experiment / 'search_snapshot.json', search_snapshot)
        frozen = {'status': 'frozen', 'search_completed': 20, 'ranking': ranking, 'test_runs': selected,
                  'search_snapshot_sha256': file_hash(experiment / 'search_snapshot.json'), 'benchmark_manifest': cfg['benchmark_manifest'],
                  'benchmark_manifest_sha256': cfg['benchmark_manifest_sha256'], 'created': stamp(),
                  'interpretation': 'Exploratory: original benchmark results have previously been observed'}
        if frozen_path.exists():
            old = read_json(frozen_path)
            assert old['ranking'] == ranking and old['test_runs'] == selected
        else:
            write_json(frozen_path, frozen)
        plan.update(phase='frozen_test', ranking=ranking, frozen_sha256=file_hash(frozen_path))
        write_json(experiment / 'index.json', plan)
        for number, item in enumerate(selected, 1):
            directory = ROOT / item['run']
            write_json(BASE / 'progress.json', {'status': 'testing', 'phase': 'frozen_test',
                       'method': item['method'], 'combination': item['combination'], 'model_seed': item['model_seed'],
                       'completed_search_runs': 20, 'test_number': number, 'expected_test_runs': len(selected),
                       'experiment': plan['experiment'], 'smoke': smoke, 'updated': stamp()})
            launch_worker(['--test-worker', str(directory), '--frozen', str(frozen_path)],
                          experiment / 'tests' / directory.name / 'run.log')
            if not smoke:
                from .thucnews_ablation_report import report
                report(experiment)
        plan.update(phase='complete', completed=stamp())
        write_json(experiment / 'index.json', plan)
        audit(experiment)
        write_json(BASE / 'progress.json', {'status': 'complete', 'completed_search_runs': 20,
                   'completed_test_runs': len(selected), 'experiment': plan['experiment'], 'smoke': smoke, 'updated': stamp()})
        if not smoke:
            from .thucnews_ablation_figures import build
            build(experiment)
            from .thucnews_ablation_report import report
            report(experiment)
        print(f'All experiments complete: {experiment}', flush=True)
    except Exception as exc:
        write_json(BASE / 'progress.json', {'status': 'failed', 'phase': plan['phase'], 'error': str(exc),
                   'completed_search_runs': len(plan['runs']), 'experiment': plan['experiment'],
                   'smoke': smoke, 'updated': stamp()})
        raise


def audit(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    cfg = plan['configuration']
    fixed_split(cfg)
    assert len(plan['runs']) == 20 and plan['phase'] == 'complete'
    candidates = read_json(experiment / 'screen_selection.json')['candidates']
    assert rank_replicates(plan['runs'], cfg, candidates) == read_json(experiment / 'frozen.json')['ranking']
    expected = set(screen_tasks(cfg) + replication_tasks(cfg, candidates))
    assert {(r['method'], r['combination'], r['model_seed']) for r in plan['runs']} == expected
    for row in plan['runs']:
        directory = ROOT / row['run']
        spec = read_json(directory / 'spec.json')
        assert cache_valid(directory, spec)
        assert spec['stage'] == 'validation_only' and spec['split_seed'] == cfg['split_seed']
        assert not (directory / 'test').exists()
        assert read_json(directory / 'summary.json')['test_evaluated'] is False
        state, selection = read_json(directory / 'state.json'), {}
        for epoch in state['epochs']:
            assert not selection.get('stop'), 'Training continued past patience stop'
            selection = update_selection(selection, epoch['primary_score'], epoch['epoch'], spec['validation'])
            path = directory / epoch['checkpoint']
            assert file_hash(path) == epoch['checkpoint_sha256']
            assert all(m['part'] == 'validation' for m in epoch['metrics'])
            assert epoch['raw_words'] == state['epochs'][0]['raw_words']
            assert 'training_loss' not in epoch
        assert selection == state['selection']
        assert row['validation_score'] == selection['best_score']
        assert read_json(directory / 'save_load_check.json')['all_validation_metrics_equal']
    frozen_path = experiment / 'frozen.json'
    assert file_hash(experiment / 'search_snapshot.json') == read_json(frozen_path)['search_snapshot_sha256']
    for item in read_json(frozen_path)['test_runs']:
        frozen_entry(frozen_path, ROOT / item['run'])
        directory = experiment / 'tests' / Path(item['run']).name
        test_spec = read_json(directory / 'spec.json')
        assert test_spec['frozen_manifest_sha256'] == file_hash(frozen_path)
        assert cache_valid(directory, test_spec)
        assert all(m['part'] == 'test' for m in read_json(directory / 'metrics.json'))
    write_json(experiment / 'verification.json', {'status': 'passed', 'search_runs': 20, 'test_runs': 12,
               'fixed_split_seed': cfg['split_seed'], 'no_test_metrics_in_search': True,
               'checkpoint_hashes_and_selection_recomputed': True, 'checked': stamp()})


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stage', choices=['run', 'report', 'audit'], default='run')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--test-worker', type=Path)
    parser.add_argument('--frozen', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.test_worker:
        if not args.frozen:
            parser.error('--test-worker requires --frozen')
        test_worker(args.test_worker, args.frozen)
    elif args.stage == 'run':
        with artifact_lock():
            run(args.smoke)
    else:
        experiment = ROOT / read_json(BASE / ('smoke_latest.json' if args.smoke else 'latest.json'))['experiment']
        if args.stage == 'audit':
            audit(experiment)
        else:
            from .thucnews_ablation_report import report
            report(experiment)


if __name__ == '__main__':
    main()
