"""Train-only category-centroid diagnostic shared by the three representations."""
from __future__ import annotations
import collections
from concurrent.futures import ProcessPoolExecutor
import itertools
import json
import time
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import KeyedVectors, TfidfModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from .common import ROOT, read_json, write_json
from .evaluation_support import save_csv

STATE = {}


def ordered_results(executor, function, items, capacity):
    iterator = iter(items)
    pending = collections.deque(executor.submit(function, item) for item in itertools.islice(iterator, capacity))
    while pending:
        yield pending.popleft().result()
        item = next(iterator, None)
        if item is not None:
            pending.append(executor.submit(function, item))


def chunks(path, part, size=512):
    batch = []
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            # Serialized dictionaries write part before tokens/sentences.
            if f'"part":"{part}"' not in line and f'"part": "{part}"' not in line:
                continue
            batch.append(line)
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def initialize(method, directory, labels, mode):
    global STATE
    directory = ROOT / directory
    STATE = {'method': method, 'labels': labels, 'mode': mode}
    if method == 'tfidf':
        STATE['dictionary'] = Dictionary.load(str(directory/'dictionary.gensim'))
        STATE['model'] = TfidfModel.load(str(directory/'model.gensim'))
        STATE['width'] = len(STATE['dictionary'])
    else:
        STATE['wv'] = KeyedVectors.load(str(directory/'vectors.kv'))
        STATE['width'] = STATE['wv'].vector_size
    if mode == 'holdout':
        STATE['centroids'] = np.load(directory/'category_centroids.npy')
        STATE['majority'] = read_json(directory/'centroid_training.json')['majority']


def features(document):
    tokens = [word for sentence in document['sentences'] for word in sentence]
    if STATE['method'] == 'tfidf':
        dictionary = STATE['dictionary']
        iv = sum(word in dictionary.token2id for word in tokens)
        weights = STATE['model'][dictionary.doc2bow(tokens)]
        columns = np.asarray([i for i, _ in weights], dtype=np.int32)
        values = np.asarray([v for _, v in weights], dtype=np.float64)
        return (columns, values), len(tokens), iv, iv, len(values) == 0
    wv = STATE['wv']
    indices = [wv.key_to_index[word] for word in tokens if word in wv.key_to_index]
    iv, usable = len(indices), len(indices)
    vector = wv.vectors[indices].sum(axis=0, dtype=np.float64) if indices else np.zeros(STATE['width'])
    if STATE['method'] == 'fasttext':
        for word, count in collections.Counter(word for word in tokens if word not in wv.key_to_index).items():
            try:
                generated = wv.get_vector(word)
                if np.isfinite(generated).all() and np.linalg.norm(generated) > 0:
                    vector += count * generated
                    usable += count
            except KeyError:
                pass
    if usable:
        vector /= usable
    norm = np.linalg.norm(vector)
    if norm:
        vector /= norm
    assert np.isfinite(vector).all()
    return vector, len(tokens), iv, usable, norm == 0


def process(lines):
    labels = STATE['labels']
    sums = np.zeros((len(labels), STATE['width']), dtype=np.float64)
    counts, statistics, predictions = collections.Counter(), collections.Counter(), []
    for line in lines:
        document = json.loads(line)
        vector, tokens, iv, usable, zero = features(document)
        label = document['category']
        counts[label] += 1
        statistics.update(documents=1, tokens=tokens, iv_tokens=iv, usable_tokens=usable, zero_vectors=int(zero))
        if STATE['mode'] == 'train':
            row = labels.index(label)
            if STATE['method'] == 'tfidf':
                columns, values = vector
                sums[row, columns] += values
            else:
                sums[row] += vector
        else:
            if STATE['method'] == 'tfidf':
                columns, values = vector
                scores = STATE['centroids'][:, columns] @ values
            else:
                scores = STATE['centroids'] @ vector
            assert np.isfinite(scores).all()
            predicted = STATE['majority'] if zero else labels[int(np.argmax(scores))]
            predictions.append({'document_id': document['id'], 'true_category': label,
                                'predicted_category': predicted, 'cosine_score': float(scores[labels.index(predicted)]),
                                'tokens': tokens, 'iv_tokens': iv, 'usable_tokens': usable, 'zero_vector': bool(zero)})
    packed = []
    if STATE['mode'] == 'train':
        for row in sums:
            columns = np.flatnonzero(row)
            packed.append((columns, row[columns]))
    return dict(counts), dict(statistics), packed, predictions


def diagnostics(directory, spec, manifest):
    started = time.perf_counter()
    labels = sorted(manifest['source']['categories'])
    width = len(Dictionary.load(str(directory/'dictionary.gensim'))) if spec['method'] == 'tfidf' else spec['embedding']['vector_size']
    sums = np.zeros((len(labels), width), dtype=np.float64)
    counts, totals = collections.Counter(), collections.Counter()
    workers = 2 if spec['smoke'] else spec['runtime']['preprocessing_workers']
    path = ROOT / manifest['directory'] / 'documents.jsonl'
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(spec['method'], directory.relative_to(ROOT).as_posix(), labels, 'train')) as executor:
        last = time.monotonic()
        for group_counts, stats, packed, _ in ordered_results(executor, process, chunks(path, 'train'), workers*2):
            counts.update(group_counts)
            totals.update(stats)
            for index, (columns, values) in enumerate(packed):
                sums[index, columns] += values
            if time.monotonic()-last > 15:
                print(f"Document centroids {totals['documents']}/{manifest['partitions']['train']['documents']}", flush=True)
                last = time.monotonic()
    assert totals['documents'] == manifest['partitions']['train']['documents']
    assert totals['tokens'] == manifest['partitions']['train']['tokens']
    assert counts == {label: manifest['categories'][label]['train'] for label in labels}
    for index, label in enumerate(labels):
        sums[index] /= counts[label]
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    assert np.isfinite(sums).all() and (norms > 0).all()
    centroids = sums / norms
    np.save(directory/'category_centroids.npy', centroids)
    assert np.array_equal(centroids, np.load(directory/'category_centroids.npy'))
    majority = max(labels, key=lambda label: counts[label])
    write_json(directory/'centroid_training.json', {'labels': labels, 'counts': dict(counts),
               'majority': majority, 'representation_training_labels_used': False,
               'diagnostic_head_training_labels_used': True, 'statistics': dict(totals)})
    rows, heldout = [], collections.Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize,
                             initargs=(spec['method'], directory.relative_to(ROOT).as_posix(), labels, 'holdout')) as executor:
        for _, stats, _, predictions in ordered_results(executor, process, chunks(path, 'holdout', 256), workers*2):
            rows.extend(predictions)
            heldout.update(stats)
    assert heldout['documents'] == manifest['partitions']['holdout']['documents']
    assert heldout['tokens'] == manifest['partitions']['holdout']['tokens']
    assert len({r['document_id'] for r in rows}) == len(rows)
    actual, predicted = [r['true_category'] for r in rows], [r['predicted_category'] for r in rows]
    detail = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    matrix = confusion_matrix(actual, predicted, labels=labels)
    assert int(matrix.sum()) == len(rows)
    save_csv(directory/'document_predictions.csv', rows)
    save_csv(directory/'category_metrics.csv', [{'category': label, **detail[label]} for label in labels])
    save_csv(directory/'category_confusion.csv', [{'actual': label, **{other: int(matrix[i,j]) for j,other in enumerate(labels)}} for i,label in enumerate(labels)])
    result = {'evaluation': 'Training-label nearest-centroid diagnostic; no tuning or embedding feedback',
              'labels': labels, 'dimensions': width, 'accuracy': float(accuracy_score(actual,predicted)),
              'macro_f1': float(f1_score(actual,predicted,labels=labels,average='macro',zero_division=0)),
              'micro_f1': float(f1_score(actual,predicted,labels=labels,average='micro',zero_division=0)),
              'weighted_f1': float(f1_score(actual,predicted,labels=labels,average='weighted',zero_division=0)),
              'classification_report': detail, 'train': dict(totals), 'holdout': dict(heldout),
              'elapsed_seconds': time.perf_counter()-started, 'workers': workers,
              'centroid_save_load_equal': True, 'source': 'saved representation model files'}
    write_json(directory/'document_diagnostics.json', result)
    return result
