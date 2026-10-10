"""Separate, validation-selected embedding runs with native epoch checkpoints."""
from __future__ import annotations

import argparse
import collections
import csv
import json
import logging
import math
import shutil
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import yaml
from gensim.models import FastText, Word2Vec
from gensim.models.word2vec import LineSentence
from threadpoolctl import threadpool_limits

from .benchmark_metrics import evaluate_pairs, evaluate_analogies
from .common import ROOT, PeakMemory, environment, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .runtime import numerical_threads, worker_environment

BASE = ROOT / 'artifacts/thucnews_validation'
CONFIG = ROOT / 'configs/thucnews_validation.yaml'


def question_key(kind, words):
    words = tuple(unicodedata.normalize('NFKC', w).upper() for w in words)
    if kind == 'similarity':
        return ('similarity', *sorted(words))
    a, b, c, d = words
    # Same pair-of-relations, including exchange and simultaneous reversal.
    return ('analogy', *min(words, (c, d, a, b), (b, a, d, c), (d, c, b, a)))


def partition(key, seed, fraction):
    value = int(fingerprint({'seed': seed, 'question': key})[:16], 16) / 2**64
    return 'validation' if value < fraction else 'test'


def write_questions(path, kind, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == 'similarity':
        content = ''.join('\t'.join([*r['words'], r['gold']]) + '\n' for r in rows)
    else:
        sections = collections.defaultdict(list)
        for r in rows:
            sections[r['section']].append(' '.join(r['words']))
        content = ''.join(': '+name+'\n'+'\n'.join(items)+'\n' for name, items in sections.items())
    path.write_text(content, encoding='utf-8')


def prepare_benchmark_split(configuration):
    original = read_json(ROOT / 'artifacts/word_evaluation/benchmarks/manifest.json')
    split_spec = {'source': original, 'validation': configuration['validation'], 'seed': configuration['seed'],
                  'normalization': 'NFKC uppercase; unordered pairs; equivalent analogy relations; first duplicate retained',
                  'protocol_sha256': file_hash(Path(__file__))}
    target = BASE / 'benchmarks' / fingerprint(split_spec)[:20]
    target.mkdir(parents=True, exist_ok=True)
    result, assignments, conflicts = {}, [], []
    val_keys, test_keys = set(), set()
    primary = original[configuration['validation']['primary_benchmark']]
    primary_keys = {question_key('similarity', line.split()[:2])
                    for line in (ROOT/primary['path']).read_text(encoding='utf-8').splitlines() if line.strip()}
    ordered = sorted(primary_keys, key=lambda key: fingerprint({'seed':configuration['seed'],'question':key}))
    quota = round(len(ordered)*configuration['validation']['fraction'])
    primary_assignments = {key:'validation' if i < quota else 'test' for i,key in enumerate(ordered)}
    for name, spec in original.items():
        source = ROOT / spec['path']
        assert file_hash(source) == spec['sha256']
        rows, seen, section, duplicate = [], {}, None, 0
        for line_number, line in enumerate(source.read_text(encoding='utf-8').splitlines(), 1):
            if not line.strip():
                continue
            if line.startswith(':'):
                section = line[1:].strip()
                continue
            tokens = line.split()
            words = tokens[:2] if spec['kind'] == 'similarity' else tokens
            key = question_key(spec['kind'], words)
            if key in seen:
                duplicate += 1
                if spec['kind'] == 'similarity' and seen[key]['gold'] != tokens[2]:
                    conflicts.append({'benchmark': name, 'line': line_number, 'key': key,
                                      'kept_gold': seen[key]['gold'], 'excluded_gold': tokens[2]})
                continue
            part = primary_assignments.get(key, partition(key, configuration['seed'], configuration['validation']['fraction']))
            row = {'key': key, 'words': words, 'gold': tokens[2] if spec['kind'] == 'similarity' else None,
                   'section': section, 'part': part, 'line': line_number}
            rows.append(row)
            seen[key] = row
            (val_keys if part == 'validation' else test_keys).add(key)
        split = {'kind': spec['kind'], 'original_n': spec['n'], 'duplicates_removed': duplicate}
        for part in ['validation', 'test']:
            chosen = [r for r in rows if r['part'] == part]
            assert len(chosen) >= 3
            path = target / f'{name}_{part}.txt'
            write_questions(path, spec['kind'], chosen)
            split[part] = {'n': len(chosen), 'path': path.relative_to(ROOT).as_posix(), 'sha256': file_hash(path)}
            if part == 'validation' and spec['kind'] == 'analogy':
                # A fixed, relation-stratified diagnostic subset keeps epoch evaluation affordable.
                groups = collections.defaultdict(list)
                for row in chosen:
                    groups[row['section']].append(row)
                queues = [sorted(group, key=lambda r: fingerprint(r['key'])) for group in groups.values()]
                probe = []
                while queues and len(probe) < configuration['validation']['analogy_probe_per_benchmark']:
                    queues = [q for q in queues if q]
                    for queue in queues:
                        if len(probe) < configuration['validation']['analogy_probe_per_benchmark']:
                            probe.append(queue.pop(0))
                path = target / f'{name}_validation_probe.txt'
                write_questions(path, 'analogy', probe)
                split['validation_probe'] = {'n': len(probe), 'path': path.relative_to(ROOT).as_posix(),
                                              'sha256': file_hash(path)}
        result[name] = split
        assignments.extend({'benchmark': name, 'line': r['line'], 'part': r['part'],
                            'question_key': json.dumps(r['key'], ensure_ascii=False),
                            'words': ' '.join(r['words'])} for r in rows)
    assert not val_keys & test_keys
    save_csv(target / 'assignments.csv', assignments)
    split = {'spec': split_spec, 'benchmarks': result, 'validation_unique_groups': len(val_keys),
             'test_unique_groups': len(test_keys), 'overlap_groups': 0, 'conflicting_duplicates': conflicts,
             'directory': target.relative_to(ROOT).as_posix()}
    write_json(target / 'manifest.json', split)
    return split


def update_selection(state, score, epoch, rules):
    if not math.isfinite(score):
        raise ValueError('The primary validation score must be finite')
    state = dict(state)
    if state.get('best_score') is None or score > state['best_score']:
        state.update(best_score=score, best_epoch=epoch)
    if state.get('reference_score') is None or score > state['reference_score'] + rules['min_delta']:
        state.update(reference_score=score, stale_epochs=0)
    else:
        state['stale_epochs'] = state.get('stale_epochs', 0) + 1
    state['stop'] = epoch >= rules['min_epochs'] and state['stale_epochs'] >= rules['patience']
    return state


def save_model(model, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    model.save(str(temp), separately=[], sep_limit=sys.maxsize)
    assert not list(path.parent.glob(temp.name+'.*'))
    temp.replace(path)


def evaluate_partition(model, split, part, output):
    result = []
    model.wv.fill_norms(force=True)
    for name, entry in split['benchmarks'].items():
        source = entry.get('validation_probe', entry[part]) if part == 'validation' else entry[part]
        path = ROOT / source['path']
        assert file_hash(path) == source['sha256']
        fn = evaluate_pairs if entry['kind'] == 'similarity' else evaluate_analogies
        metric = fn(model.wv, path, output, name)
        metric['part'] = part
        metric['source_sha256'] = source['sha256']
        metric['validation_pool_n'] = entry['validation']['n'] if part == 'validation' else None
        result.append(metric)
    return result


def coverage(model, heldout):
    frequencies = collections.Counter(t for s in LineSentence(str(heldout)) for t in s)
    total = sum(frequencies.values())
    oov = {t: n for t, n in frequencies.items() if t not in model.wv.key_to_index}
    generated = 0
    if isinstance(model, FastText):
        for word, count in oov.items():
            vector = model.wv.get_vector(word)
            if np.isfinite(vector).all() and np.linalg.norm(vector) > 0:
                generated += count
    return {'tokens': total, 'oov_tokens': sum(oov.values()), 'oov_types': len(oov),
            'vocabulary_token_coverage': 1-sum(oov.values())/total,
            'generated_oov_tokens': generated, 'usable_vector_token_coverage': 1-(sum(oov.values())-generated)/total}


def cache_valid(directory, spec):
    if not (directory / 'seal.json').exists():
        return False
    seal = read_json(directory / 'seal.json')
    return seal['spec_fingerprint'] == fingerprint(spec) and all(
        (directory / p).is_file() and file_hash(directory / p) == sha for p, sha in seal['files'].items())


def worker(directory):
    directory = Path(directory).resolve()
    if not directory.is_relative_to((BASE/'runs').resolve()):
        raise ValueError('Run must stay in the validation experiment directory')
    spec = read_json(directory / 'spec.json')
    assert all(file_hash(ROOT/p) == h for p,h in spec['protocol_sha256'].items())
    split = read_json(ROOT / spec['benchmark_manifest'])
    assert file_hash(ROOT / spec['benchmark_manifest']) == spec['benchmark_manifest_sha256']
    manifest = read_json(ROOT / spec['manifest'])
    data = ROOT / manifest['directory']
    corpus = ROOT / spec['training_corpus']
    assert file_hash(corpus) == spec['training_corpus_sha256']
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    cls = Word2Vec if spec['method'] == 'word2vec' else FastText
    rules = spec['validation']
    state_path = directory / 'state.json'
    with threadpool_limits(limits=numerical_threads()), PeakMemory() as memory:
        if state_path.exists():
            state = read_json(state_path)
            assert state['spec_fingerprint'] == fingerprint(spec)
            path = directory / state['checkpoint']
            assert file_hash(path) == state['checkpoint_sha256']
            model = cls.load(str(path))
            epochs, selection = state['epochs'], state['selection']
        else:
            model = cls(**spec['embedding'], **spec['subwords'], seed=spec['seed'], epochs=rules['max_epochs'])
            before = time.perf_counter()
            model.build_vocab(corpus_file=str(corpus))
            state = {'spec_fingerprint': fingerprint(spec), 'vocabulary_seconds': time.perf_counter()-before}
            epochs, selection = [], {}
        for epoch in range(len(epochs)+1, rules['max_epochs']+1):
            if selection.get('stop'):
                break
            alpha = spec['embedding']['alpha'] + (spec['embedding']['min_alpha']-spec['embedding']['alpha'])*(epoch-1)/rules['max_epochs']
            end_alpha = spec['embedding']['alpha'] + (spec['embedding']['min_alpha']-spec['embedding']['alpha'])*epoch/rules['max_epochs']
            begin = time.perf_counter()
            effective, raw = model.train(corpus_iterable=LineSentence(str(corpus)),
                                         total_examples=model.corpus_count, total_words=model.corpus_total_words,
                                         epochs=1, start_alpha=alpha, end_alpha=end_alpha)
            train_seconds = time.perf_counter()-begin
            assert raw == model.corpus_total_words
            assert model.compute_loss is False
            begin = time.perf_counter()
            metrics = evaluate_partition(model, split, 'validation', directory / 'validation' / f'epoch_{epoch:02d}')
            validation_seconds = time.perf_counter()-begin
            primary = next(m for m in metrics if m['benchmark'] == rules['primary_benchmark'])[rules['primary_metric']]
            selection = update_selection(selection, primary, epoch, rules)
            checkpoint = directory / 'checkpoints' / f'epoch_{epoch:02d}.gensim'
            begin = time.perf_counter()
            save_model(model, checkpoint)
            row = {'epoch': epoch, 'training_seconds': train_seconds, 'validation_seconds': validation_seconds,
                   'checkpoint_seconds': time.perf_counter()-begin, 'effective_words': effective, 'raw_words': raw,
                   'start_alpha': alpha, 'end_alpha': end_alpha, 'primary_score': primary,
                   'best_epoch_so_far': selection['best_epoch'], 'stale_epochs': selection['stale_epochs'],
                   'checkpoint': checkpoint.relative_to(directory).as_posix(), 'checkpoint_sha256': file_hash(checkpoint),
                   'metrics': metrics}
            epochs.append(row)
            state.update(epochs=epochs, selection=selection, checkpoint=row['checkpoint'],
                         checkpoint_sha256=row['checkpoint_sha256'], updated=stamp())
            write_json(state_path, state)
            write_json(directory/'epochs.json', epochs)
            save_csv(directory/'epochs.csv', [{k:v for k,v in r.items() if k != 'metrics'} for r in epochs])
            save_csv(directory/'validation_metrics.csv', [{'epoch':r['epoch'], **{k:v for k,v in m.items() if k != 'sections'}}
                                                         for r in epochs for m in r['metrics']])
            write_json(BASE/'progress.json', {'status':'running','method':spec['method'], 'run':directory.name,
                       'epoch':epoch, 'max_epochs':rules['max_epochs'], 'best_epoch':selection['best_epoch'],
                       'primary_score':primary,'updated':stamp(),'smoke':spec['smoke']})
            print(f'{spec["method"]} epoch {epoch}: validation rho={primary:.6f}, best={selection["best_epoch"]}, stale={selection["stale_epochs"]}; train={train_seconds:.1f}s',flush=True)
        assert epochs
        best = epochs[selection['best_epoch']-1]
        checkpoint = directory / best['checkpoint']
        assert file_hash(checkpoint) == best['checkpoint_sha256']
        del model
        model = cls.load(str(checkpoint))
        # Recompute all chosen-epoch validation metrics from the saved model.
        rechecked = evaluate_partition(model, split, 'validation', directory/'reload_validation')
        for original, loaded in zip(best['metrics'], rechecked, strict=True):
            for metric in ['pearson_r','spearman_rho','conditional_accuracy','all_accuracy']:
                if original.get(metric) is not None:
                    assert np.isclose(original[metric], loaded[metric], rtol=0, atol=1e-12)
        write_json(directory/'save_load_check.json', {'selected_epoch':selection['best_epoch'],
                   'all_validation_metrics_equal':True,'checkpoint_sha256':best['checkpoint_sha256']})
        # Final test is never entered before selection has finished.
        test = evaluate_partition(model, split, 'test', directory/'test')
        write_json(directory/'metrics.json',test)
        heldout = coverage(model, data/'holdout.txt')
        write_json(directory/'heldout_summary.json',heldout)
        shutil.copy2(checkpoint,directory/'model.gensim')
        model.wv.save(str(directory/'vectors.kv'), separately=[], sep_limit=sys.maxsize)
        model.wv.save_word2vec_format(str(directory/'vectors.vec'),binary=False)
        queries = {}
        for word in ['经济','股票','新能源汽车产业']:
            try:
                queries[word] = {'in_vocabulary':word in model.wv.key_to_index,
                                 'nearest_neighbors':[(w,float(s)) for w,s in model.wv.most_similar(word,topn=10)]}
            except KeyError:
                queries[word] = {'status':'no_vector'}
        write_json(directory/'queries.json',{'queries':queries})
        summary = {'spec':spec,'completed':stamp(),'selection':selection,'trained_epochs':len(epochs),
                   'selected_epoch':selection['best_epoch'],'stop_reason':'validation_patience' if selection['stop'] else 'max_epochs',
                   'fit_seconds':sum(r['training_seconds'] for r in epochs),
                   'validation_seconds':sum(r['validation_seconds'] for r in epochs),
                   'checkpoint_seconds':sum(r['checkpoint_seconds'] for r in epochs),
                   'vocabulary_seconds':state['vocabulary_seconds'],'vocabulary_size':len(model.wv),
                   'vocabulary_sha256':fingerprint(model.wv.index_to_key),
                   'heldout':heldout,'metrics':test,'model_bytes':(directory/'model.gensim').stat().st_size,
                   'benchmark_split':split['benchmarks'],'checkpoints':len(epochs)}
    summary['peak_rss_mib'] = memory.peak/2**20
    write_json(directory/'summary.json',summary)
    files={p.relative_to(directory).as_posix():file_hash(p) for p in directory.rglob('*')
           if p.is_file() and p.name not in ['run.log','seal.json']}
    write_json(directory/'seal.json', {'spec_fingerprint':fingerprint(spec),'files':files,'completed':stamp()})
    print(f'Completed {spec["method"]}: selected epoch {selection["best_epoch"]}/{len(epochs)}',flush=True)


def run(models=None, smoke=False):
    configuration = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    if smoke:
        configuration['validation'].update(min_epochs=2,max_epochs=3,patience=2,analogy_probe_per_benchmark=8)
        configuration['embedding'].update(vector_size=16,min_count=2,workers=4)
        configuration['fasttext']['bucket'] = 5000
    split = prepare_benchmark_split(configuration)
    manifest = read_json(ROOT / configuration['manifest'])
    corpus = ROOT / manifest['directory'] / 'train.txt'
    if smoke:
        target = BASE/'smoke_train.txt'
        target.parent.mkdir(parents=True,exist_ok=True)
        with corpus.open(encoding='utf-8') as source, target.open('w',encoding='utf-8',newline='\n') as output:
            for i,line in enumerate(source):
                if i == 20000:
                    break
                output.write(line)
        corpus = target
    protocol = {f'gensim_lab/{name}':file_hash(ROOT/'gensim_lab'/name) for name in
                ['thucnews_validation.py','benchmark_metrics.py','evaluation_support.py','runtime.py']}
    current = read_json(BASE/'latest.json') if (BASE/'latest.json').exists() and not smoke else {'models':{}}
    for method in models or configuration['models']:
        spec={'method':method,'seed':configuration['seed'],'smoke':smoke,'manifest':configuration['manifest'],
              'manifest_sha256':file_hash(ROOT/configuration['manifest']), 'embedding':configuration['embedding'],
              'subwords':configuration['fasttext'] if method=='fasttext' else {},'validation':configuration['validation'],
              'training_corpus':corpus.relative_to(ROOT).as_posix(),'training_corpus_sha256':file_hash(corpus),
              'benchmark_manifest':(ROOT/split['directory']/'manifest.json').relative_to(ROOT).as_posix(),
              'benchmark_manifest_sha256':file_hash(ROOT/split['directory']/'manifest.json'),
              'protocol_sha256':protocol,'packages':environment()['packages']}
        directory=BASE/'runs'/fingerprint(spec)[:24]
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/'spec.json',spec)
        if not cache_valid(directory,spec):
            write_json(BASE/'progress.json',{'status':'starting','method':method,'run':directory.name,'updated':stamp(),'smoke':smoke})
            print(f'Start {method} {directory.name}',flush=True)
            with (directory/'run.log').open('a',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-m','gensim_lab.thucnews_validation','--worker',str(directory)],
                                      cwd=ROOT,env=worker_environment(),stdout=log,stderr=subprocess.STDOUT)
            if result.returncode or not cache_valid(directory,spec):
                write_json(BASE/'progress.json',{'status':'failed','method':method,'run':directory.name,'returncode':result.returncode,'updated':stamp()})
                raise RuntimeError(f'Validation-selected training failed: {directory}/run.log')
        print(f'Complete {method} {directory.name}',flush=True)
        if not smoke:
            current['models'][method] = directory.relative_to(ROOT).as_posix()
            current.update(updated=stamp(),benchmark_manifest=spec['benchmark_manifest'])
            write_json(BASE/'latest.json',current)
    write_json(BASE/'progress.json',{'status':'complete','models':models or configuration['models'],'smoke':smoke,'updated':stamp()})


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['all','word2vec','fasttext'],default='all')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker',type=Path)
    args=parser.parse_args()
    if args.worker:
        worker(args.worker)
    else:
        with artifact_lock():
            run(None if args.model=='all' else [args.model],args.smoke)


if __name__=='__main__':
    main()
