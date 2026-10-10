"""Retrain frozen small-search candidates on the existing full training split.

Reuse the established epoch worker without modifying any sealed search protocol.
Publish the two new models together, after independent checkpoint verification.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml
from gensim.models import FastText, KeyedVectors, Word2Vec
from gensim.models.word2vec import LineSentence
from threadpoolctl import threadpool_limits

from .common import ROOT, environment, file_hash, fingerprint, read_json, stamp, write_json
from .integrity import artifact_lock
from .runtime import numerical_threads, worker_environment
from .thucnews_ablation import fixed_split, validate_model
from .thucnews_audit import metric_check
from .thucnews_validation import BASE as VALIDATION_BASE, cache_valid, update_selection

BASE = ROOT / 'artifacts/thucnews_selected_full'
CONFIG = ROOT / 'configs/thucnews_selected_full.yaml'


def load_configuration():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    source = read_json(ROOT / cfg['source_selection'])
    if source['status'] != 'complete':
        raise ValueError('Candidate search must be complete before full training')
    if cfg['models'] != ['word2vec', 'fasttext']:
        raise ValueError('This retraining run must contain both embedding models')
    for method in cfg['models']:
        chosen = source['selected'][method]
        params = cfg['model_parameters'][method]
        actual = (f"sg_ns{params['negative']}", params['alpha'], params['vector_size'])
        expected = (chosen['combination'], chosen['alpha'], chosen['vector_size'])
        if actual != expected or params['sg'] != 1 or params['hs'] != 0 or chosen['n'] != 3:
            raise ValueError(f'Frozen candidate differs from search selection: {method}')
    fixed_split(cfg)
    return cfg


def protocol():
    names = ['thucnews_selected_full.py', 'thucnews_validation.py', 'thucnews_ablation.py',
             'thucnews_audit.py', 'benchmark_metrics.py', 'evaluation_support.py',
             'runtime.py', 'common.py', 'integrity.py']
    return {f'gensim_lab/{name}': file_hash(ROOT / 'gensim_lab' / name) for name in names}


def make_spec(cfg, method, manifest, corpus_sha, packages):
    return {
        'method': method, 'seed': cfg['seed'], 'model_seed': cfg['seed'],
        'split_seed': cfg['split_seed'], 'smoke': False,
        'stage': 'frozen_candidate_full_retraining',
        'manifest': cfg['manifest'], 'manifest_sha256': file_hash(ROOT / cfg['manifest']),
        'embedding': {**cfg['embedding'], **cfg['model_parameters'][method]},
        'subwords': copy.deepcopy(cfg['fasttext']) if method == 'fasttext' else {},
        'validation': copy.deepcopy(cfg['validation']),
        'training_corpus': (Path(manifest['directory']) / 'train.txt').as_posix(),
        'training_corpus_sha256': corpus_sha,
        'expected_corpus': manifest['partitions']['train'],
        'benchmark_manifest': cfg['benchmark_manifest'],
        'benchmark_manifest_sha256': cfg['benchmark_manifest_sha256'],
        'source_selection': cfg['source_selection'],
        'source_selection_sha256': file_hash(ROOT / cfg['source_selection']),
        'configuration_sha256': file_hash(CONFIG),
        'protocol_sha256': protocol(), 'packages': packages,
    }


def prepare():
    cfg = load_configuration()
    manifest = read_json(ROOT / cfg['manifest'])
    if manifest['spec']['sample']['mode'] != 'full':
        raise ValueError('Expected the full prepared training corpus')
    corpus = ROOT / manifest['directory'] / 'train.txt'
    corpus_sha = file_hash(corpus)
    if corpus_sha != manifest['files']['train.txt']:
        raise ValueError('Training corpus checksum differs from the prepared manifest')
    packages = environment()['packages']
    specs = {method: make_spec(cfg, method, manifest, corpus_sha, packages) for method in cfg['models']}
    experiment = BASE / 'experiments' / fingerprint(specs)[:24]
    experiment.mkdir(parents=True, exist_ok=True)
    path = experiment / 'index.json'
    if path.exists():
        plan = read_json(path)
        if plan['specs'] != specs:
            raise ValueError('Existing experiment specification changed')
    else:
        baseline = read_json(VALIDATION_BASE / 'latest.json')
        runs = {method: (VALIDATION_BASE / 'runs' / fingerprint(spec)[:24]).relative_to(ROOT).as_posix()
                for method, spec in specs.items()}
        plan = {'status': 'prepared', 'created': stamp(), 'configuration': cfg, 'specs': specs,
                'runs': runs, 'completed_models': [], 'historical_baseline': baseline,
                'historical_baseline_sha256': file_hash(VALIDATION_BASE / 'latest.json'),
                'environment': environment(), 'experiment': experiment.relative_to(ROOT).as_posix()}
        for method, spec in specs.items():
            directory = ROOT / runs[method]
            directory.mkdir(parents=True, exist_ok=True)
            spec_path = directory / 'spec.json'
            if spec_path.exists() and read_json(spec_path) != spec:
                raise ValueError(f'Existing run configuration differs: {directory}')
            write_json(spec_path, spec)
        write_json(experiment / 'historical_baseline.json', baseline)
        write_json(path, plan)
    write_json(BASE / 'latest.json', {'experiment': plan['experiment'], 'updated': stamp()})
    return experiment, plan


def audit(experiment):
    plan = read_json(experiment / 'index.json')
    cfg = plan['configuration']
    fixed_split(cfg)
    split = read_json(ROOT / cfg['benchmark_manifest'])
    manifest = read_json(ROOT / cfg['manifest'])
    corpus = ROOT / manifest['directory'] / 'train.txt'
    frequencies = collections.Counter()
    sentences = 0
    for sentence in LineSentence(str(corpus)):
        frequencies.update(sentence)
        sentences += 1
        if sentences % 4000000 == 0:
            print(f'Audit: independently counted {sentences:,} training sentences', flush=True)
    assert sentences == manifest['partitions']['train']['sentences']
    assert sum(frequencies.values()) == manifest['partitions']['train']['tokens']
    checks = []
    for method in cfg['models']:
        spec = plan['specs'][method]
        directory = ROOT / plan['runs'][method]
        assert cache_valid(directory, spec), 'Complete sealed training artifacts required'
        assert all(file_hash(ROOT / p) == h for p, h in spec['protocol_sha256'].items())
        assert file_hash(corpus) == spec['training_corpus_sha256']
        summary = read_json(directory / 'summary.json')
        epochs = read_json(directory / 'epochs.json')
        rules = spec['validation']
        assert rules['min_epochs'] <= len(epochs) <= rules['max_epochs']
        state, checkpoint_checks = {}, []
        cls = Word2Vec if method == 'word2vec' else FastText
        for i, row in enumerate(epochs, 1):
            assert row['epoch'] == i and row['raw_words'] == spec['expected_corpus']['tokens']
            checkpoint = directory / row['checkpoint']
            assert file_hash(checkpoint) == row['checkpoint_sha256']
            model = cls.load(str(checkpoint))
            validate_model(model, spec)
            source = ROOT / split['benchmarks'][rules['primary_benchmark']]['validation']['path']
            _, rho, _ = model.wv.evaluate_word_pairs(str(source), delimiter='\t',
                                                    restrict_vocab=len(model.wv), case_insensitive=False)
            assert np.isclose(rho[0], row['primary_score'], rtol=0, atol=1e-12)
            for metric in row['metrics']:
                metric_check(directory / 'validation' / f'epoch_{i:02d}', metric)
            state = update_selection(state, row['primary_score'], i, rules)
            assert row['best_epoch_so_far'] == state['best_epoch']
            assert row['stale_epochs'] == state['stale_epochs']
            assert not state['stop'] or i == len(epochs)
            checkpoint_checks.append({'epoch': i, 'raw_words_exact': True,
                                      'parameters_and_primary_rechecked': True})
            del model
        assert state == summary['selection']
        assert summary['selected_epoch'] == state['best_epoch']
        chosen = epochs[state['best_epoch'] - 1]
        assert file_hash(directory / 'model.gensim') == chosen['checkpoint_sha256']
        model = cls.load(str(directory / 'model.gensim'))
        saved = KeyedVectors.load(str(directory / 'vectors.kv'))
        expected = {word for word, count in frequencies.items() if count >= spec['embedding']['min_count']}
        assert set(model.wv.key_to_index) == expected
        assert all(int(model.wv.get_vecattr(w, 'count')) == frequencies[w] for w in expected)
        assert model.corpus_count == sentences and model.corpus_total_words == sum(frequencies.values())
        assert model.wv.index_to_key == saved.index_to_key
        assert np.array_equal(model.wv.vectors, saved.vectors)
        assert np.isfinite(saved.vectors).all()
        assert (np.linalg.norm(saved.vectors, axis=1) > 0).all()
        if method == 'fasttext':
            for name in ['vectors_vocab', 'vectors_ngrams']:
                assert np.array_equal(getattr(model.wv, name), getattr(saved, name))
            for word in ['新能源汽车产业测试未登录', '量子计算芯片测试未登录']:
                assert word not in expected and np.array_equal(model.wv[word], saved[word])
        for metric in summary['metrics']:
            metric_check(directory / 'test', metric)
            assert metric['total'] == split['benchmarks'][metric['benchmark']]['test']['n']
        assert read_json(directory / 'save_load_check.json')['all_validation_metrics_equal']
        checks.append({'method': method, 'run': plan['runs'][method], 'selected_epoch': state['best_epoch'],
                       'checkpoints': checkpoint_checks, 'full_training_vocab_and_counts_exact': True,
                       'saved_vectors_equal': True, 'test_item_metrics_rechecked': True,
                       'context_log_score': 'not_applicable_negative_sampling'})
        del model, saved
        print(f'Audit passed: {method}', flush=True)
    result = {'status': 'passed', 'completed': stamp(), 'benchmark_group_overlap': 0,
              'training_sentences': sentences, 'training_tokens': sum(frequencies.values()), 'checks': checks}
    write_json(experiment / 'verification.json', result)
    return result


def run():
    experiment, plan = prepare()
    if shutil.disk_usage(ROOT).free < 30 * 2**30:
        raise RuntimeError('Reserve at least 30 GiB for native full-corpus checkpoints')
    try:
        plan.update(status='running', updated=stamp())
        write_json(experiment / 'index.json', plan)
        for method in plan['configuration']['models']:
            directory, spec = ROOT / plan['runs'][method], plan['specs'][method]
            if not cache_valid(directory, spec):
                write_json(BASE / 'progress.json', {'status': 'training', 'method': method,
                           'experiment': plan['experiment'], 'run': plan['runs'][method],
                           'completed_models': plan['completed_models'], 'updated': stamp()})
                print(f'Start {method}: {directory}', flush=True)
                with (directory / 'run.log').open('a', encoding='utf-8') as log:
                    process = subprocess.Popen([sys.executable, '-u', '-m', 'gensim_lab.thucnews_validation',
                                                '--worker', str(directory)], cwd=ROOT, env=worker_environment(),
                                               stdout=log, stderr=subprocess.STDOUT,
                                               creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                    write_json(BASE / 'processes.json', {'parent_pid': os.getpid(), 'worker_pid': process.pid,
                               'method': method, 'run': plan['runs'][method], 'started': stamp()})
                    try:
                        code = process.wait()
                    except BaseException:
                        process.terminate()
                        process.wait(timeout=15)
                        raise
                if code or not cache_valid(directory, spec):
                    raise RuntimeError(f'Training failed ({code}); see {directory / "run.log"}')
            if method not in plan['completed_models']:
                plan['completed_models'].append(method)
            plan['updated'] = stamp()
            write_json(experiment / 'index.json', plan)
            print(f'Completed {method}', flush=True)
        write_json(BASE / 'progress.json', {'status': 'auditing', 'experiment': plan['experiment'], 'updated': stamp()})
        with threadpool_limits(limits=numerical_threads()):
            audit(experiment)
        # Publish both together. Until this point the official index still refers to the old baseline.
        write_json(VALIDATION_BASE / 'latest.json', {'models': plan['runs'], 'updated': stamp(),
                   'benchmark_manifest': plan['configuration']['benchmark_manifest'],
                   'full_retraining_experiment': plan['experiment']})
        plan.update(status='complete', completed=stamp())
        write_json(experiment / 'index.json', plan)
        write_json(BASE / 'progress.json', {'status': 'complete', 'experiment': plan['experiment'],
                   'completed_models': plan['completed_models'], 'updated': stamp()})
        write_json(VALIDATION_BASE / 'progress.json', {'status': 'complete', 'models': list(plan['runs']),
                   'full_retraining_experiment': plan['experiment'], 'smoke': False, 'updated': stamp()})
        print(f'Full candidate retraining and verification complete: {experiment}', flush=True)
    except BaseException as exc:
        plan.update(status='failed', error=str(exc), updated=stamp())
        write_json(experiment / 'index.json', plan)
        write_json(BASE / 'progress.json', {'status': 'failed', 'error': str(exc),
                   'experiment': plan['experiment'], 'updated': stamp()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['prepare', 'run', 'audit'], default='run')
    args = parser.parse_args()
    with artifact_lock():
        if args.stage == 'prepare':
            experiment, plan = prepare()
            print(plan['experiment'])
        elif args.stage == 'audit':
            experiment = ROOT / read_json(BASE / 'latest.json')['experiment']
            with threadpool_limits(limits=numerical_threads()):
                audit(experiment)
        else:
            run()


if __name__ == '__main__':
    main()
