"""Deterministic, category-proportional article subsets of the prepared train set."""
from __future__ import annotations

import collections
import hashlib
import heapq
import json
from pathlib import Path

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv

BASE = ROOT / 'data/thucnews/search_subsets'


def proportional_quotas(counts, size):
    total = sum(counts.values())
    if not 0 < size <= total:
        raise ValueError('Sample size must fit the training population')
    quotas = {category: size * count // total for category, count in counts.items()}
    order = sorted(counts, key=lambda c: (-(size * counts[c] % total), c))
    for category in order[:size - sum(quotas.values())]:
        quotas[category] += 1
    assert sum(quotas.values()) == size
    return quotas


def sampling_rank(identifier, seed):
    return int(fingerprint({'article': identifier, 'sample_seed': seed}), 16)


def retain(heap, rank, offset, identifier, capacity):
    """Keep the smallest deterministic ranks, without storing article contents."""
    entry = (-rank, offset, identifier)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif capacity and rank < -heap[0][0]:
        heapq.heapreplace(heap, entry)


def prepare(cfg, smoke=False):
    source_path = ROOT / cfg['source_manifest']
    source = read_json(source_path)
    spec = {'source_manifest': cfg['source_manifest'], 'source_manifest_sha256': file_hash(source_path),
            'sample': cfg['sample'], 'code_sha256': file_hash(Path(__file__)), 'smoke': smoke,
            'protocol': 'proportional largest-remainder quotas; smallest seeded SHA256 ranks per category; train only'}
    target = BASE / fingerprint(spec)[:24]
    manifest_path = target / 'manifest.json'
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest['spec'] == spec
        assert all(file_hash(target / p) == h for p, h in manifest['files'].items())
        return manifest_path
    counts = {c: item['train'] for c, item in source['categories'].items()}
    quotas = proportional_quotas(counts, cfg['sample']['documents'])
    heaps = {c: [] for c in counts}
    observed = collections.Counter()
    heldout_ids = set()
    document_file = ROOT / source['directory'] / 'documents.jsonl'
    digest = hashlib.sha256()
    print(f'Selecting {sum(quotas.values()):,} complete articles from {sum(counts.values()):,} training articles', flush=True)
    with document_file.open('rb') as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            digest.update(line)
            document = json.loads(line)
            if document['part'] == 'holdout':
                heldout_ids.add(document['id'])
                continue
            assert document['part'] == 'train'
            category = document['category']
            observed[category] += 1
            retain(heaps[category], sampling_rank(document['id'], cfg['sample']['seed']),
                   offset, document['id'], quotas[category])
    assert digest.hexdigest() == source['files']['documents.jsonl']
    assert dict(observed) == counts
    chosen = sorted((offset, identifier) for heap in heaps.values() for _, offset, identifier in heap)
    assert len(chosen) == cfg['sample']['documents']
    assert not heldout_ids.intersection(identifier for _, identifier in chosen)
    target.mkdir(parents=True, exist_ok=True)
    selected, vocabulary = [], collections.Counter()
    statistics = {'documents': 0, 'sentences': 0, 'tokens': 0}
    seen_hashes = set()
    with document_file.open('rb') as source_stream, (target / 'train.txt').open('w', encoding='utf-8', newline='\n') as corpus:
        for offset, identifier in chosen:
            source_stream.seek(offset)
            document = json.loads(source_stream.readline())
            assert document['id'] == identifier and document['part'] == 'train'
            assert document['text_sha256'] not in seen_hashes
            seen_hashes.add(document['text_sha256'])
            sentences = document['sentences']
            words = sum(map(len, sentences))
            assert words == document['tokens']
            for sentence in sentences:
                corpus.write(' '.join(sentence) + '\n')
                vocabulary.update(sentence)
            statistics['documents'] += 1
            statistics['sentences'] += len(sentences)
            statistics['tokens'] += words
            selected.append({'id': identifier, 'category': document['category'], 'tokens': words,
                             'sentences': len(sentences), 'text_sha256': document['text_sha256'],
                             'source_offset': offset, 'sampling_rank': str(sampling_rank(identifier, cfg['sample']['seed']))})
    save_csv(target / 'selected_articles.csv', selected)
    kept = {w: n for w, n in vocabulary.items() if n >= cfg['embedding']['min_count']}
    write_json(target / 'vocabulary_counts.json', kept)
    split = read_json(ROOT / cfg['benchmark_manifest'])
    pairs = (ROOT / split['benchmarks']['wordsim240']['validation']['path']).read_text(encoding='utf-8').splitlines()
    covered = [i for i, line in enumerate(pairs, 1) if all(w in kept for w in line.split('\t')[:2])]
    assert len(covered) / len(pairs) >= cfg['validation']['minimum_primary_coverage'], 'Insufficient primary validation coverage'
    manifest = {'prepared': stamp(), 'spec': spec, 'directory': target.relative_to(ROOT).as_posix(),
                'partitions': {'train': statistics}, 'categories': quotas,
                'training_population_documents': sum(counts.values()), 'holdout_overlap': 0,
                'vocabulary_size': len(kept), 'vocabulary_min_count': cfg['embedding']['min_count'],
                'primary_validation': {'total': len(pairs), 'covered': len(covered), 'covered_items': covered,
                                       'coverage': len(covered) / len(pairs),
                                       'selection': 'same strict-IV pairs across all configurations; no benchmark-driven sampling'},
                'files': {p.name: file_hash(p) for p in target.iterdir() if p.is_file()},
                'context': source['context'], 'order': 'original prepared article order; complete contexts retained'}
    write_json(manifest_path, manifest)
    print(f'Prepared subset: {statistics}; primary validation {len(covered)}/{len(pairs)}; {manifest_path}', flush=True)
    return manifest_path
