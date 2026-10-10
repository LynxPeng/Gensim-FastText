"""Exploratory validation-only extension prompted by native underfitting diagnostics."""
from __future__ import annotations
import copy
import gc
import json
from pathlib import Path
import shutil
import time

from threadpoolctl import threadpool_limits

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .data import load_docs
from .evaluation_support import save_csv
from .integrity import artifact_lock
from .pfr_frozen_classification import BASE, train_native, native_prediction, evaluate, labels_for, metrics
from .runtime import settings


def main():
    import fasttext
    import numpy as np
    with artifact_lock(), threadpool_limits(limits=settings()['numerical_threads']):
        parent = ROOT / read_json(BASE / 'latest.json')['experiment']
        assert read_json(parent / 'verification.json')['status'] == 'passed'
        original = read_json(parent / 'plan.json')
        plan = copy.deepcopy(original)
        plan['configuration']['native']['epochs'] = [100, 200]
        plan['supplement'] = {'parent': parent.relative_to(ROOT).as_posix(), 'parent_plan_sha256': file_hash(parent / 'plan.json'),
                             'code_sha256': file_hash(Path(__file__)), 'reason': 'low native validation F1 and existing 200-epoch fitting evidence',
                             'inference': 'exploratory; parent test results already observed; no change to frozen representation comparison'}
        directory = parent / 'native_budget' / fingerprint(plan)[:24]
        directory.mkdir(parents=True, exist_ok=True)
        write_json(parent / 'native_budget_latest.json', {'directory': directory.relative_to(ROOT).as_posix()})
        if not (directory / 'plan.json').exists():
            write_json(directory / 'plan.json', plan)
            write_json(directory / 'index.json', {'status': 'prepared', 'runs': {}, 'selection': {}})
            for profile in ['january_12', 'temporal_12']:
                source = parent / 'native_input' / profile
                dest = directory / 'native_input' / profile
                dest.mkdir(parents=True, exist_ok=True)
                for name in ['train.txt', 'fasttext.vec', 'initialization.json']:
                    shutil.copy2(source / name, dest / name)
                    assert file_hash(dest / name) == file_hash(source / name)
        else:
            assert read_json(directory / 'plan.json') == plan
        docs = load_docs()
        lookup = {d['id']: d for d in docs}
        train_native(directory, docs)
        index = read_json(directory / 'index.json')
        assert len(index['selection']) == 12
        frozen = {'frozen': stamp(), 'selection': index['selection'], 'exploratory': True}
        write_json(directory / 'selection_frozen.json', frozen)
        rows = []
        for key, rec in index['selection'].items():
            profile, method, head, seed = key.split('/')
            out = directory / 'runs' / rec['run']
            model = fasttext.load_model(str(out / 'native.bin'))
            split = plan['profiles'][profile]['splits']
            train = [lookup[i] for i in split['train']]
            pp, ss = native_prediction(model, train)
            training = evaluate(out, 'train', split['train'], [d['page_label'] for d in train], pp, ss, labels_for(12), 'probability')
            target = [lookup[i] for i in split['test']]
            pp, ss = native_prediction(model, target)
            testing = evaluate(out, 'test', split['test'], [d['page_label'] for d in target], pp, ss, labels_for(12), 'probability')
            loaded = fasttext.load_model(str(out / 'native.bin'))
            qq, tt = native_prediction(loaded, target)
            assert pp == qq and np.array_equal(ss, tt)
            summary = read_json(out / 'summary.json')
            rows.append({'profile': profile, 'method': method, 'seed': int(seed), 'epochs': rec['parameter'],
                         'train_macro_f1': training['macro_f1'], 'validation_macro_f1': rec['validation_macro_f1'],
                         'test_macro_f1': testing['macro_f1'], 'test_accuracy': testing['accuracy'],
                         'fit_seconds': summary['fit_seconds'], 'model_bytes': summary['model_bytes'], 'run': rec['run']})
            write_json(out / 'reload_test.json', {'labels_exact': True, 'scores_exact': True, 'n': len(target)})
            del model, loaded
            gc.collect()
        save_csv(directory / 'results.csv', rows)
        source_checks = {method: all(file_hash(ROOT / source['directory'] / name) == digest for name, digest in source['files'].items())
                         for method, source in plan['sources'].items()}
        assert all(source_checks.values())
        write_json(directory / 'verification.json', {'status': 'passed', 'verified': stamp(), 'exploratory': True,
                   'selection_used_validation_only': True, 'parent_test_already_observed': True,
                   'pretrained_sources_unchanged': source_checks, 'reload_all_test_predictions_exact': True})
        index['status'] = 'complete'
        write_json(directory / 'index.json', index)
        print('Exploratory native budget extension complete:', directory, flush=True)


if __name__ == '__main__':
    main()
