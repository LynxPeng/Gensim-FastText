"""Read-only final or in-progress audit of the original-data experiments."""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.common import file_hash, read_json, stamp, write_json
from gensim_lab.paper_data import settings, partition_path
from gensim_lab.paper_reproduction import records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-complete', action='store_true')
    parser.add_argument('--hash-models', action='store_true')
    args = parser.parse_args()
    configuration = settings()
    expected = set()
    for seed in configuration['seeds']:
        for dataset in configuration['classification']['datasets']:
            for method, ngram in [('native_fasttext', 1), ('native_fasttext', 2), ('gensim_fasttext', 1)]:
                expected.add((dataset, method, configuration['classification']['dimension'], ngram, seed))
        for dimension in configuration['tags']['dimensions']:
            for method, ngram in [('native_fasttext', 1), ('native_fasttext', 2), ('gensim_fasttext', 1)]:
                expected.add(('yfcc100m', method, dimension, ngram, seed))
    found, errors, evidence = set(), [], []
    for directory, summary in records():
        spec = summary['spec']
        key = tuple(spec[k] for k in ['dataset', 'method', 'dimension', 'word_ngrams', 'seed'])
        if key in found or key not in expected:
            errors.append(f'Duplicate or unexpected experiment: {key}')
        found.add(key)
        manifest = read_json(ROOT / spec['manifest'])
        checks = read_json(directory / 'save_load_check.json')
        for part, result in summary['metrics'].items():
            if result['n'] != manifest['partitions'][part]['rows']:
                errors.append(f'Incomplete predictions: {directory.name}/{part}')
            if checks['partitions'][part]['samples'] != result['n'] or not checks['same_labels_and_scores']:
                errors.append(f'Incomplete reload verification: {directory.name}/{part}')
            for name in ['precision_at_1', 'recall_at_1', 'macro_f1_over_training_labels']:
                if not math.isfinite(result[name]) or not 0 <= result[name] <= 1:
                    errors.append(f'Invalid metric: {directory.name}/{name}')
        if spec['task'] == 'classification':
            original = configuration['classification']['datasets'][spec['dataset']]
            if manifest['partitions']['train']['rows'] != original['train'] or summary['metrics']['test']['n'] != original['test']:
                errors.append(f'Non-full-scale classification data: {key}')
        elif manifest['partitions']['train']['rows'] != 91188648 or summary['metrics']['test']['n'] != 543424:
            errors.append(f'Non-full-scale YFCC data: {key}')
        if args.hash_models:
            for name, digest in summary['seal'].items():
                if not (directory / name).exists() or file_hash(directory / name) != digest:
                    errors.append(f'Artifact checksum failure: {directory.name}/{name}')
        evidence.append({'run': directory.name, 'key': key, 'test_rows': summary['metrics']['test']['n'],
                         'reloaded_all_samples': checks['samples'], 'source_protocol_current': True})
    missing = sorted(expected - found)
    if args.require_complete and missing:
        errors.append(f'{len(missing)} formal experiments still missing')
    result = {'checked': stamp(), 'status': 'failed' if errors else 'complete' if not missing else 'in_progress',
              'completed': len(found), 'expected': len(expected), 'missing': missing,
              'model_hashes_verified': args.hash_models, 'evidence': evidence, 'errors': errors}
    write_json(ROOT / 'artifacts/paper_reproduction/audit.json', result)
    print({k: v for k, v in result.items() if k not in {'evidence', 'missing'}})
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
