"""One-shot stop at the final FastText replication's verified completion boundary."""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.common import fingerprint, read_json, stamp, write_json


def main():
    base = ROOT / 'artifacts/thucnews_small_search'
    experiment = 'artifacts/thucnews_small_search/experiments/e5cc57786fa3cd4195698419'
    scope = read_json(ROOT / experiment / 'scope_amendment.json')
    if not scope.get('learning_rate_search_deferred', False):
        print('The matrix-only stop was superseded by user-authorized learning-rate search.', flush=True)
        return
    spec = read_json(base / 'runs/36dc0d3874b33d53966c9cef/spec.json')
    spec['model_seed'] = 7
    target = base / 'runs' / fingerprint(spec)[:24]
    request = {'experiment': experiment, 'scope': 'training_methods_only',
               'expected_search_runs': 20, 'learning_rate_search_deferred': True,
               'reason': 'User requested stopping after FastText replication, before learning-rate search',
               'target_run': target.relative_to(ROOT).as_posix(), 'requested': stamp()}
    write_json(ROOT / experiment / 'scope_amendment.json', request)
    print(f'Waiting for final replication reload check: {target}', flush=True)
    deadline = time.monotonic() + 7200
    check = target / 'save_load_check.json'
    while not check.exists():
        if time.monotonic() > deadline:
            raise TimeoutError('Final replication did not reach its reload check within two hours')
        time.sleep(0.02)
    assert read_json(check)['all_validation_metrics_equal'] is True
    # The worker has passed its final cooperative stop check. It can finish
    # writing and sealing this run; the parent stops before its next task.
    write_json(base / 'stop.request.json', {**request, 'triggered': stamp()})
    write_json(ROOT / experiment / 'boundary_stop.json', {
        **request, 'trigger': check.relative_to(ROOT).as_posix(), 'triggered': stamp(),
        'last_worker_allowed_to_finish_sealing': True})
    print('Final FastText reload verified; stop flag set before learning-rate search.', flush=True)


if __name__ == '__main__':
    main()
