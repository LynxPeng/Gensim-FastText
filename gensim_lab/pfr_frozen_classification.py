"""Frozen THUCNews representations, classifiers fitted ONLY on PFR 9801--9806."""
from __future__ import annotations

import argparse
import collections
from concurrent.futures import ThreadPoolExecutor
import copy
import gc
import json
import os
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
import yaml

from .common import ROOT, PROCESSED, PeakMemory, environment, file_hash, fingerprint, read_json, stamp, write_json
from .data import load_docs
from .evaluation_support import save_csv
from .fasttext_transfer import verify_prepared_dataset, export_initialization, training_vocabulary
from .integrity import artifact_lock
from .runtime import settings as runtime_settings

BASE = ROOT / 'artifacts/pfr_frozen_classification'
CONFIG = ROOT / 'configs/pfr_frozen_classification.yaml'
METHODS = ['tfidf', 'word2vec', 'fasttext']


def labels_for(n):
    return [f'{i:02d}' for i in range(1, n + 1)]


def validate_partitions(splits, lookup):
    """Reject leakage of IDs, content or global near-duplicate groups."""
    for attribute in ['id', 'content_hash', 'group']:
        seen = set()
        for part, ids in splits.items():
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate ID in {part}')
            values = {lookup[i][attribute] for i in ids}
            if values & seen:
                raise ValueError(f'{attribute} crosses partitions')
            seen.update(values)


def make_profiles(docs, cfg):
    lookup = {d['id']: d for d in docs}
    january = read_json(PROCESSED / 'splits.json')
    profiles = {}
    for n in cfg['classes']:
        allowed = set(labels_for(n))
        splits = {part: [i for i in ids if lookup[i]['page_label'] in allowed]
                  for part, ids in january.items()}
        validate_partitions(splits, lookup)
        profiles[f'january_{n}'] = {'classes': n, 'splits': splits, 'kind': 'january'}
    train = [d['id'] for d in docs if d['month'] <= '199804']
    groups = {lookup[i]['group'] for i in train}
    val = [d['id'] for d in docs if d['month'] == '199805' and d['group'] not in groups]
    forbidden = groups | {lookup[i]['group'] for i in val}
    test = [d['id'] for d in docs if d['month'] == '199806' and d['group'] not in forbidden]
    splits = {'train': train, 'validation': val, 'test': test}
    validate_partitions(splits, lookup)
    profiles['temporal_12'] = {'classes': 12, 'splits': splits, 'kind': 'temporal'}
    january_groups = {d['group'] for d in docs if d['month'] == '199801'}
    future = {month: [d['id'] for d in docs if d['month'] == month and d['group'] not in january_groups]
              for month in ['199802', '199803', '199804', '199805', '199806']}
    return profiles, future


def prepare():
    cfg = yaml.safe_load(CONFIG.read_text(encoding='utf-8'))
    manifest = verify_prepared_dataset()
    docs = load_docs()
    assert len(docs) == manifest['kept_total'] == 18108
    assert {d['month'] for d in docs} == {f'1998{i:02d}' for i in range(1, 7)}
    sources = {}
    for method, relative in cfg['sources'].items():
        directory = ROOT / relative
        seal, summary = read_json(directory / 'seal.json'), read_json(directory / 'summary.json')
        names = ['summary.json', 'model.gensim']
        names += ['dictionary.gensim'] if method == 'tfidf' else ['vectors.kv']
        hashes = {name: file_hash(directory / name) for name in names}
        if any(seal['files'].get(name) != digest for name, digest in hashes.items()):
            raise ValueError(f'Frozen source changed: {method}')
        if summary['spec']['method'] != method:
            raise ValueError('Wrong representation source')
        sources[method] = {'directory': relative, 'files': hashes, 'spec': summary['spec']}
    profiles, future = make_profiles(docs, cfg)
    plan = {'configuration': cfg, 'configuration_sha256': file_hash(CONFIG),
            'runtime': runtime_settings(), 'sources': sources, 'profiles': profiles, 'future': future,
            'dataset': read_json(ROOT / 'artifacts/dataset_integrity.json'),
            'code_sha256': file_hash(Path(__file__)), 'environment': environment(),
            'document_order': [d['id'] for d in docs]}
    # Exclude transient platform details from the cache identity.
    identity = {k: v for k, v in plan.items() if k != 'environment'}
    directory = BASE / 'experiments' / fingerprint(identity)[:24]
    directory.mkdir(parents=True, exist_ok=True)
    if not (directory / 'plan.json').exists():
        write_json(directory / 'plan.json', plan)
    else:
        assert {k: v for k, v in read_json(directory / 'plan.json').items() if k != 'environment'} == identity
    write_json(BASE / 'latest.json', {'experiment': directory.relative_to(ROOT).as_posix()})
    if not (directory / 'index.json').exists():
        write_json(directory / 'index.json', {'started': stamp(), 'status': 'prepared', 'runs': {}, 'selection': {}})
    rows = []
    lookup = {d['id']: d for d in docs}
    for name, p in profiles.items():
        for part, ids in p['splits'].items():
            count = collections.Counter((lookup[i]['month'], lookup[i]['page_label']) for i in ids)
            rows.extend({'profile': name, 'partition': part, 'month': month, 'page_label': page,
                         'documents': n} for (month, page), n in sorted(count.items()))
    save_csv(directory / 'split_counts.csv', rows)
    print('PFR partitions:', {p: {s: len(v) for s, v in rec['splits'].items()} for p, rec in profiles.items()}, flush=True)
    return directory, docs


class FrozenRepresentation:
    """No fit method; TF-IDF statistics and word/subword parameters are read-only."""
    def __init__(self, method, source):
        from gensim.corpora import Dictionary
        from gensim.models import KeyedVectors, TfidfModel
        self.method = method
        directory = ROOT / source['directory']
        if method == 'tfidf':
            self.dictionary = Dictionary.load(str(directory / 'dictionary.gensim'))
            self.model = TfidfModel.load(str(directory / 'model.gensim'))
            self.width = len(self.dictionary)
        else:
            self.wv = KeyedVectors.load(str(directory / 'vectors.kv'))
            self.width = self.wv.vector_size

    def one(self, doc):
        counts = collections.Counter(doc['tokens'])
        total = len(doc['tokens'])
        if self.method == 'tfidf':
            iv = sum(n for w, n in counts.items() if w in self.dictionary.token2id)
            row = self.model[self.dictionary.doc2bow(doc['tokens'])]
            columns, values = [i for i, _ in row], [v for _, v in row]
            usable, zero = iv, not bool(row)
            result = (columns, values)
        else:
            iv, usable = 0, 0
            result = np.zeros(self.width, dtype=np.float64)
            for word, count in counts.items():
                present = word in self.wv.key_to_index
                iv += count if present else 0
                if not present and self.method == 'word2vec':
                    continue
                try:
                    vector = self.wv.get_vector(word)
                except KeyError:
                    continue
                if np.isfinite(vector).all() and np.any(vector):
                    result += count * vector.astype(np.float64)
                    usable += count
            if usable:
                result /= usable
            norm = np.linalg.norm(result)
            if norm:
                result /= norm
            zero = norm == 0
        stats = {'id': doc['id'], 'month': doc['month'], 'tokens': total, 'iv_tokens': iv,
                 'usable_tokens': usable, 'zero_vector': int(zero)}
        return result, stats

    def transform(self, docs, threads):
        with ThreadPoolExecutor(max_workers=threads) as executor:
            records = list(executor.map(self.one, docs))
        if self.method == 'tfidf':
            values, indices, indptr = [], [], [0]
            for (columns, weights), _ in records:
                indices.extend(columns); values.extend(weights); indptr.append(len(indices))
            x = sparse.csr_matrix((np.array(values, dtype=np.float32), np.array(indices, dtype=np.int32),
                                   np.array(indptr, dtype=np.int32)), shape=(len(docs), self.width))
        else:
            x = np.array([r[0] for r in records], dtype=np.float32)
        assert np.isfinite(x.data if sparse.issparse(x) else x).all()
        return x, [r[1] for r in records]


def features(directory, docs):
    plan = read_json(directory / 'plan.json')
    out = directory / 'features'
    out.mkdir(exist_ok=True)
    if (out / 'seal.json').exists():
        for name, digest in read_json(out / 'seal.json').items():
            assert file_hash(out / name) == digest
        return
    names, summaries = [], {}
    for method in METHODS:
        start = time.perf_counter()
        print('Frozen PFR features:', method, flush=True)
        with PeakMemory() as memory:
            representation = FrozenRepresentation(method, plan['sources'][method])
            x, stats = representation.transform(docs, plan['runtime']['preprocessing_workers'])
            suffix = '.npz' if sparse.issparse(x) else '.npy'
            path = out / (method + suffix)
            if sparse.issparse(x):
                sparse.save_npz(path, x, compressed=False)
            else:
                np.save(path, x)
            save_csv(out / f'{method}_coverage.csv', stats)
            norm = np.sqrt(np.asarray(x.multiply(x).sum(axis=1)).ravel()) if sparse.issparse(x) else np.linalg.norm(x, axis=1)
            assert np.allclose(norm[norm > 0], 1, atol=1e-5)
            summaries[method] = {'dimension': x.shape[1], 'documents': x.shape[0],
                                 'seconds': time.perf_counter() - start, 'peak_rss_mib': memory.peak / 2**20,
                                 'source_frozen': True}
            names.extend([path.name, f'{method}_coverage.csv'])
            del representation, x, stats
            gc.collect()
    write_json(out / 'summary.json', summaries)
    names.append('summary.json')
    write_json(out / 'seal.json', {name: file_hash(out / name) for name in names})


def load_features(directory, method):
    folder = directory / 'features'
    return sparse.load_npz(folder / 'tfidf.npz') if method == 'tfidf' else np.load(folder / f'{method}.npy', mmap_mode='r')


def metrics(truth, prediction, labels):
    return {'n': len(truth), 'accuracy': float(accuracy_score(truth, prediction)),
            **{f'{average}_f1': float(f1_score(truth, prediction, labels=labels, average=average, zero_division=0))
               for average in ['macro', 'micro', 'weighted']},
            'per_class': classification_report(truth, prediction, labels=labels, output_dict=True, zero_division=0),
            'confusion_matrix': confusion_matrix(truth, prediction, labels=labels).tolist()}


def evaluate(out, part, ids, truth, prediction, scores, labels, score_kind):
    result = metrics(truth, prediction, labels)
    assert len(ids) == len(prediction) == len(scores)
    assert np.isfinite(scores).all()
    write_json(out / f'metrics_{part}.json', result)
    save_csv(out / f'predictions_{part}.csv', [{'id': i, 'true_label': y, 'predicted_label': p,
              'score': float(s), 'score_kind': score_kind} for i, y, p, s in zip(ids, truth, prediction, scores)])
    save_csv(out / f'per_class_{part}.csv', [{'page_label': label, **result['per_class'][label]} for label in labels])
    save_csv(out / f'confusion_{part}.csv', [{'true_label': label, **dict(zip(labels, row))}
              for label, row in zip(labels, result['confusion_matrix'])])
    return result


def cached_run(out, spec):
    if not (out / 'seal.json').exists():
        return False
    seal = read_json(out / 'seal.json')
    if seal['spec'] != fingerprint(spec):
        raise ValueError('Run cache identity changed')
    for name, digest in seal['files'].items():
        if file_hash(out / name) != digest:
            raise ValueError(f'Run cache changed: {out / name}')
    return True


def seal_run(out, spec):
    names = ['spec.json', 'summary.json', 'metrics_validation.json', 'predictions_validation.csv']
    names += [n for n in ['classifier.joblib', 'softmax.npz', 'native.bin', 'epochs.csv'] if (out / n).exists()]
    write_json(out / 'seal.json', {'spec': fingerprint(spec), 'files': {n: file_hash(out / n) for n in names}})


def freeze_selection(directory, profile, method, head, candidates, seed):
    # No test information enters this function. Ties prefer the first/smaller candidate.
    best = max(candidates, key=lambda r: r['macro_f1'])
    index = read_json(directory / 'index.json')
    key = f'{profile}/{method}/{head}/{seed}'
    index['selection'][key] = {'run': best['run'], 'validation_macro_f1': best['macro_f1'],
                               'parameter': best['parameter'], 'seed': seed}
    for candidate in candidates:
        index['runs'][candidate['run']] = {'profile': profile, 'method': method, 'head': head, 'seed': seed}
    write_json(directory / 'index.json', index)
    return best


def train_linear(directory, docs):
    plan = read_json(directory / 'plan.json')
    cfg, positions = plan['configuration'], {i: n for n, i in enumerate(plan['document_order'])}
    lookup = {d['id']: d for d in docs}
    for method in METHODS:
        all_x = load_features(directory, method)
        for profile, rec in plan['profiles'].items():
            splits, labels = rec['splits'], labels_for(rec['classes'])
            train, val = splits['train'], splits['validation']
            x = all_x[[positions[i] for i in train]]
            xv = all_x[[positions[i] for i in val]]
            y, yv = [lookup[i]['page_label'] for i in train], [lookup[i]['page_label'] for i in val]
            for head in ['svm', 'logistic']:
                candidates = []
                for c in cfg['linear']['C']:
                    spec = {'profile': profile, 'method': method, 'head': head, 'C': c, 'seed': cfg['seed'],
                            'experiment': directory.name, 'class_weight': 'balanced'}
                    key = fingerprint(spec)[:24]
                    out = directory / 'runs' / key
                    out.mkdir(parents=True, exist_ok=True)
                    if not cached_run(out, spec):
                        print(f'Fit {profile} {method} {head} C={c}', flush=True)
                        write_json(out / 'spec.json', spec)
                        clf = (LinearSVC(C=c, class_weight='balanced', dual='auto', random_state=cfg['seed'], max_iter=cfg['linear']['max_iter'])
                               if head == 'svm' else LogisticRegression(C=c, class_weight='balanced', solver='lbfgs',
                                       random_state=cfg['seed'], max_iter=cfg['linear']['max_iter']))
                        with PeakMemory() as memory, warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always', ConvergenceWarning)
                            clf.fit(x, y)
                            if any(issubclass(w.category, ConvergenceWarning) for w in caught):
                                raise RuntimeError(f'Classifier did not converge: {spec}')
                        scores = clf.decision_function(xv)
                        margins = abs(scores) if scores.ndim == 1 else scores.max(axis=1)
                        prediction = clf.predict(xv)
                        result = evaluate(out, 'validation', val, yv, prediction, margins, labels, 'decision_margin')
                        joblib.dump(clf, out / 'classifier.joblib')
                        loaded = joblib.load(out / 'classifier.joblib')
                        assert np.array_equal(loaded.predict(xv), prediction)
                        assert np.array_equal(loaded.decision_function(xv), scores)
                        write_json(out / 'summary.json', {'spec': spec, 'fit_seconds': memory.elapsed,
                                   'peak_rss_mib': memory.peak / 2**20, 'reload_validation_exact': True,
                                   'model_bytes': (out / 'classifier.joblib').stat().st_size,
                                   'n_iter': np.asarray(clf.n_iter_).tolist(), 'validation': result})
                        seal_run(out, spec)
                    result = read_json(out / 'metrics_validation.json')
                    candidates.append({'run': key, 'parameter': c, 'macro_f1': result['macro_f1']})
                best = freeze_selection(directory, profile, method, head, candidates, cfg['seed'])
                print(f'Selected {profile} {method} {head}: C={best["parameter"]}, val={best["macro_f1"]:.4f}', flush=True)
        del all_x


def tensor_batch(x, device):
    import torch
    if sparse.issparse(x):
        coo = x.tocoo()
        indices = torch.from_numpy(np.vstack([coo.row, coo.col]).astype(np.int64)).to(device)
        values = torch.from_numpy(coo.data.astype(np.float32, copy=False)).to(device)
        return torch.sparse_coo_tensor(indices, values, coo.shape, device=device).coalesce()
    return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)


def softmax_scores(weights, bias, x):
    return np.asarray(x @ weights.T + bias)


def train_softmax(directory, docs):
    import torch
    plan = read_json(directory / 'plan.json')
    cfg, options = plan['configuration'], plan['configuration']['softmax']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(plan['runtime']['numerical_threads'])
    positions, lookup = {i: n for n, i in enumerate(plan['document_order'])}, {d['id']: d for d in docs}
    for profile in ['january_12', 'temporal_12']:
        splits = plan['profiles'][profile]['splits']
        labels = labels_for(12)
        y = np.array([labels.index(lookup[i]['page_label']) for i in splits['train']], dtype=np.int64)
        yv = [lookup[i]['page_label'] for i in splits['validation']]
        for method in METHODS:
            all_x = load_features(directory, method)
            x = all_x[[positions[i] for i in splits['train']]]
            xv = all_x[[positions[i] for i in splits['validation']]]
            selected_lr = None
            for seed in cfg['seeds']:
                candidates = []
                grid = options['learning_rates'] if selected_lr is None else [selected_lr]
                for lr in grid:
                    spec = {'profile': profile, 'method': method, 'head': 'softmax', 'learning_rate': lr,
                            'seed': seed, 'experiment': directory.name, 'options': options, 'device': device}
                    key = fingerprint(spec)[:24]
                    out = directory / 'runs' / key
                    out.mkdir(parents=True, exist_ok=True)
                    if not cached_run(out, spec):
                        print(f'CUDA Softmax {profile} {method} lr={lr} seed={seed}', flush=True)
                        torch.manual_seed(seed)
                        if device == 'cuda':
                            torch.cuda.manual_seed_all(seed)
                            torch.cuda.reset_peak_memory_stats()
                        rng = np.random.default_rng(seed)
                        layer = torch.nn.Linear(x.shape[1], len(labels), device=device)
                        optimizer = torch.optim.AdamW(layer.parameters(), lr=lr, weight_decay=options['weight_decay'])
                        weights = torch.tensor(len(y) / (len(labels) * np.bincount(y, minlength=len(labels))), dtype=torch.float32, device=device)
                        epochs, best, patience = [], None, 0
                        with PeakMemory() as memory:
                            for epoch in range(1, options['max_epochs'] + 1):
                                order = rng.permutation(len(y))
                                for start in range(0, len(y), options['batch_size']):
                                    ids = order[start:start + options['batch_size']]
                                    xb = tensor_batch(x[ids], device)
                                    target = torch.as_tensor(y[ids], dtype=torch.long, device=device)
                                    optimizer.zero_grad(set_to_none=True)
                                    logits = (torch.sparse.mm(xb, layer.weight.T) + layer.bias if xb.is_sparse else layer(xb))
                                    objective = torch.nn.functional.cross_entropy(logits, target, weight=weights)
                                    objective.backward(); optimizer.step()
                                w, b = layer.weight.detach().cpu().numpy().copy(), layer.bias.detach().cpu().numpy().copy()
                                scores = softmax_scores(w, b, xv)
                                prediction = [labels[i] for i in scores.argmax(axis=1)]
                                result = metrics(yv, prediction, labels)
                                improved = best is None or result['macro_f1'] > best['score'] + options['min_delta']
                                if improved:
                                    best = {'epoch': epoch, 'score': result['macro_f1'], 'w': w, 'b': b}
                                    patience = 0
                                else:
                                    patience += 1
                                epochs.append({'epoch': epoch, 'validation_macro_f1': result['macro_f1'],
                                               'validation_accuracy': result['accuracy'], 'new_best': improved})
                                if epoch >= options['min_epochs'] and patience >= options['patience']:
                                    break
                        write_json(out / 'spec.json', spec)
                        np.savez(out / 'softmax.npz', weights=best['w'], bias=best['b'])
                        loaded = np.load(out / 'softmax.npz')
                        scores = softmax_scores(loaded['weights'], loaded['bias'], xv)
                        assert np.array_equal(scores, softmax_scores(best['w'], best['b'], xv))
                        # Stable softmax, no probabilistic interpretation for the SVM margins.
                        exponent = np.exp(scores - scores.max(axis=1, keepdims=True))
                        probabilities = exponent / exponent.sum(axis=1, keepdims=True)
                        prediction = [labels[i] for i in scores.argmax(axis=1)]
                        result = evaluate(out, 'validation', splits['validation'], yv, prediction, probabilities.max(axis=1), labels, 'probability')
                        save_csv(out / 'epochs.csv', epochs)
                        write_json(out / 'summary.json', {'spec': spec, 'fit_seconds': memory.elapsed,
                                   'peak_rss_mib': memory.peak / 2**20, 'trained_epochs': len(epochs),
                                   'selected_epoch': best['epoch'], 'reload_validation_exact': True,
                                   'cuda_peak_mib': torch.cuda.max_memory_allocated() / 2**20 if device == 'cuda' else 0,
                                   'model_bytes': (out / 'softmax.npz').stat().st_size, 'validation': result})
                        seal_run(out, spec)
                    candidates.append({'run': key, 'parameter': lr, 'macro_f1': read_json(out / 'metrics_validation.json')['macro_f1']})
                best = freeze_selection(directory, profile, method, 'softmax', candidates, seed)
                if selected_lr is None:
                    selected_lr = best['parameter']
            del all_x, x, xv
            gc.collect()


def native_prediction(model, docs):
    prediction, scores = [], []
    for doc in docs:
        label, probability = model.predict(' '.join(doc['tokens']), k=1)
        prediction.append(label[0].removeprefix('__label__p'))
        scores.append(float(probability[0]))
    return prediction, np.asarray(scores)


def train_native(directory, docs):
    import fasttext
    from gensim.models import KeyedVectors
    plan = read_json(directory / 'plan.json')
    cfg, lookup = plan['configuration'], {d['id']: d for d in docs}
    for profile in ['january_12', 'temporal_12']:
        splits = plan['profiles'][profile]['splits']
        train, val = [lookup[i] for i in splits['train']], [lookup[i] for i in splits['validation']]
        folder = directory / 'native_input' / profile
        folder.mkdir(parents=True, exist_ok=True)
        text = ''.join(f"__label__p{d['page_label']} {' '.join(d['tokens'])}\n" for d in train)
        (folder / 'train.txt').write_text(text, encoding='utf-8', newline='\n')
        if not (folder / 'initialization.json').exists():
            wv = KeyedVectors.load(str(ROOT / plan['sources']['fasttext']['directory'] / 'vectors.kv'))
            record = export_initialization(wv, training_vocabulary(train), folder / 'fasttext.vec', generate_oov=True)
            write_json(folder / 'initialization.json', record)
            del wv
            gc.collect()
        native = {k: v for k, v in cfg['native'].items() if k != 'epochs'}
        native['thread'] = plan['runtime']['training_threads']
        for variant in ['native_scratch', 'native_pretrained']:
            selected_epoch = None
            for seed in cfg['seeds']:
                grid = cfg['native']['epochs'] if selected_epoch is None else [selected_epoch]
                candidates = []
                for epoch in grid:
                    spec = {'profile': profile, 'method': variant, 'head': 'native', 'seed': seed,
                            'epoch': epoch, 'parameters': native, 'experiment': directory.name}
                    key = fingerprint(spec)[:24]
                    out = directory / 'runs' / key
                    out.mkdir(parents=True, exist_ok=True)
                    if not cached_run(out, spec):
                        kwargs = dict(native, input=str(folder / 'train.txt'), epoch=epoch, seed=seed, verbose=0)
                        if variant == 'native_pretrained':
                            kwargs['pretrainedVectors'] = str(folder / 'fasttext.vec')
                            probe = fasttext.train_supervised(**dict(kwargs, epoch=1, lr=0.0))
                            vectors = read_json(folder / 'initialization.json')['probe_vectors']
                            assert all(np.array_equal(probe.get_word_vector(w), np.array(v, dtype=np.float32)) for w, v in vectors.items())
                            del probe
                        print(f'Native {profile} {variant} epoch={epoch} seed={seed}', flush=True)
                        with PeakMemory() as memory:
                            model = fasttext.train_supervised(**kwargs)
                        allowed = set(training_vocabulary(train)) | {'</s>'}
                        assert set(model.get_words()) == allowed
                        prediction, scores = native_prediction(model, val)
                        result = evaluate(out, 'validation', splits['validation'], [d['page_label'] for d in val], prediction, scores, labels_for(12), 'probability')
                        model.save_model(str(out / 'native.bin'))
                        loaded = fasttext.load_model(str(out / 'native.bin'))
                        pp, ss = native_prediction(loaded, val)
                        assert pp == prediction and np.array_equal(ss, scores)
                        change = None
                        if variant == 'native_pretrained':
                            probes = read_json(folder / 'initialization.json')['probe_vectors']
                            changes = [float(np.linalg.norm(model.get_word_vector(w) - np.array(v))) for w, v in probes.items()]
                            change = {'words': len(changes), 'mean_l2_change': float(np.mean(changes)),
                                      'maximum_l2_change': max(changes), 'changed_words': sum(v > 1e-6 for v in changes)}
                        write_json(out / 'spec.json', spec)
                        write_json(out / 'summary.json', {'spec': spec, 'fit_seconds': memory.elapsed,
                                   'peak_rss_mib': memory.peak / 2**20, 'reload_validation_exact': True,
                                   'source_vector_change': change, 'training_vocabulary_only': True,
                                   'model_bytes': (out / 'native.bin').stat().st_size, 'validation': result})
                        seal_run(out, spec)
                        del model, loaded
                        gc.collect()
                    candidates.append({'run': key, 'parameter': epoch, 'macro_f1': read_json(out / 'metrics_validation.json')['macro_f1']})
                best = freeze_selection(directory, profile, variant, 'native', candidates, seed)
                if selected_epoch is None:
                    selected_epoch = best['parameter']


def predict_saved(out, method, head, x, docs, labels):
    if head == 'native':
        import fasttext
        return native_prediction(fasttext.load_model(str(out / 'native.bin')), docs)
    if head == 'softmax':
        loaded = np.load(out / 'softmax.npz')
        scores = softmax_scores(loaded['weights'], loaded['bias'], x)
        ex = np.exp(scores - scores.max(axis=1, keepdims=True))
        return [labels[i] for i in scores.argmax(axis=1)], (ex / ex.sum(axis=1, keepdims=True)).max(axis=1)
    model = joblib.load(out / 'classifier.joblib')
    scores = model.decision_function(x)
    return model.predict(x).tolist(), abs(scores) if scores.ndim == 1 else scores.max(axis=1)


def paired_bootstrap(truth, predictions, labels, repetitions, seed):
    """Paired label-stratified bootstrap; same resampled articles for all models."""
    truth = np.asarray(truth)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(truth == label) for label in labels]
    if any(len(s) == 0 for s in strata):
        raise ValueError('Bootstrap requires every evaluated page')
    values = {method: np.empty(repetitions) for method in predictions}
    for r in range(repetitions):
        selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        for method, predicted in predictions.items():
            values[method][r] = f1_score(truth[selected], np.asarray(predicted)[selected], labels=labels,
                                       average='macro', zero_division=0)
    intervals = {method: np.quantile(scores, [.025, .975]).tolist() for method, scores in values.items()}
    pairs = []
    for a, b in [('word2vec', 'tfidf'), ('fasttext', 'tfidf'), ('fasttext', 'word2vec')]:
        delta = values[a] - values[b]
        pairs.append({'a': a, 'b': b, 'difference': float(f1_score(truth, predictions[a], labels=labels, average='macro', zero_division=0)
                      - f1_score(truth, predictions[b], labels=labels, average='macro', zero_division=0)),
                      'ci_low': float(np.quantile(delta, .025)), 'ci_high': float(np.quantile(delta, .975))})
    return {'repetitions': repetitions, 'seed': seed, 'kind': 'paired label-stratified percentile bootstrap',
            'intervals': intervals, 'differences': pairs}


def finalize(directory, docs):
    plan, index = read_json(directory / 'plan.json'), read_json(directory / 'index.json')
    cfg = plan['configuration']
    expected = len(plan['profiles']) * len(METHODS) * 2 + 2 * (len(METHODS) + 2) * len(cfg['seeds'])
    if len(index['selection']) != expected:
        raise ValueError(f'Incomplete selection: {len(index["selection"])}/{expected}')
    # Persist a frozen selection record before reading any target test outcomes.
    selected = copy.deepcopy(index['selection'])
    path = directory / 'selection_frozen.json'
    if path.exists():
        assert read_json(path)['selection'] == selected
    else:
        write_json(path, {'frozen': stamp(), 'selection': selected, 'selection_metric': 'validation_macro_f1'})
    index['status'] = 'selection_frozen'
    write_json(directory / 'index.json', index)
    positions, lookup = {i: n for n, i in enumerate(plan['document_order'])}, {d['id']: d for d in docs}
    arrays = {method: load_features(directory, method) for method in METHODS}
    rows, evidence = [], {}
    for key, rec in selected.items():
        profile, method, head, seed = key.split('/')
        splits = plan['profiles'][profile]['splits']
        labels = labels_for(plan['profiles'][profile]['classes'])
        out = directory / 'runs' / rec['run']
        cached_run(out, read_json(out / 'spec.json'))
        parts = {'test': splits['test']}
        if profile == 'january_12':
            parts.update({f'future_{month}': ids for month, ids in plan['future'].items()})
        for part, ids in parts.items():
            target = [lookup[i] for i in ids]
            x = None if head == 'native' else arrays[method][[positions[i] for i in ids]]
            start = time.perf_counter()
            predicted, scores = predict_saved(out, method, head, x, target, labels)
            predict_seconds = time.perf_counter() - start
            again, other_scores = predict_saved(out, method, head, x, target, labels)
            assert predicted == again and np.array_equal(scores, other_scores)
            truth = [d['page_label'] for d in target]
            result = evaluate(out, part, ids, truth, predicted, scores, labels,
                              'probability' if head in ['native', 'softmax'] else 'decision_margin')
            summary = read_json(out / 'summary.json')
            rows.append({'profile': profile, 'method': method, 'head': head, 'seed': int(seed), 'partition': part,
                         'selected_parameter': rec['parameter'], 'validation_macro_f1': rec['validation_macro_f1'],
                         **{k: result[k] for k in ['n', 'accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']},
                         'fit_seconds': summary['fit_seconds'], 'predict_and_load_seconds': predict_seconds,
                         'peak_rss_mib': summary['peak_rss_mib'], 'model_bytes': summary['model_bytes'], 'run': rec['run']})
            if part == 'test' and head == 'svm' and method in METHODS:
                evidence.setdefault(profile, {'truth': truth, 'predictions': {}})['predictions'][method] = predicted
            write_json(out / f'reload_{part}.json', {'n': len(ids), 'labels_exact': True, 'scores_exact': True})
    for profile, rec in plan['profiles'].items():
        labels, ids = labels_for(rec['classes']), rec['splits']['test']
        groups = [lookup[i]['group'] for i in ids]
        assert len(groups) == len(set(groups)), 'Use a grouped bootstrap if future test sets have repeated groups'
        truth = [lookup[i]['page_label'] for i in ids]
        counts = collections.Counter(lookup[i]['page_label'] for i in rec['splits']['train'])
        majority = max(sorted(counts), key=counts.get)
        result = metrics(truth, [majority] * len(truth), labels)
        rows.append({'profile': profile, 'method': 'majority', 'head': 'constant', 'seed': cfg['seed'], 'partition': 'test',
                     **{k: result[k] for k in ['n', 'accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']}})
        ev = evidence[profile]
        write_json(directory / f'bootstrap_{profile}.json', paired_bootstrap(ev['truth'], ev['predictions'], labels,
                   cfg['bootstrap']['repetitions'], cfg['bootstrap']['seed']))
    save_csv(directory / 'results.csv', rows)
    # Seal all saved prediction resources after finalization, separately from the train-only seal.
    test_files = {str(p.relative_to(directory)): file_hash(p) for p in directory.glob('runs/*/*')
                  if p.is_file() and any(tag in p.name for tag in ['test', 'future_'])}
    write_json(directory / 'test_seal.json', test_files)
    source_checks = {}
    for method, source in plan['sources'].items():
        source_checks[method] = all(file_hash(ROOT / source['directory'] / name) == digest for name, digest in source['files'].items())
        assert source_checks[method], 'Frozen representation was modified'
    for profile in plan['profiles'].values():
        validate_partitions(profile['splits'], lookup)
    write_json(directory / 'verification.json', {'verified': stamp(), 'status': 'passed',
               'classification_corpus': 'People Daily 9801--9806 ONLY', 'frozen_sources_unchanged': source_checks,
               'partitions_disjoint_by_id_content_group': True, 'selection_before_test': True,
               'selected_runs': len(selected), 'evaluation_rows': len(rows), 'all_saved_predictions_reload_exact': True})
    index['status'], index['completed'] = 'complete', stamp()
    write_json(directory / 'index.json', index)
    print('Classification complete; sources frozen, reloads verified.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['all', 'prepare', 'linear', 'softmax', 'native', 'finalize'], default='all')
    args = parser.parse_args()
    with artifact_lock(), threadpool_limits(limits=runtime_settings()['numerical_threads']):
        directory, docs = prepare()
        features(directory, docs)
        if args.stage in ['all', 'linear']:
            train_linear(directory, docs)
        if args.stage in ['all', 'softmax']:
            train_softmax(directory, docs)
        if args.stage in ['all', 'native']:
            train_native(directory, docs)
        if args.stage in ['all', 'finalize']:
            finalize(directory, docs)


if __name__ == '__main__':
    main()
