"""Record user-authorized continuation from the frozen matrix into alpha search."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.common import file_hash, read_json, stamp, write_json
from gensim_lab.thucnews_small_search import BASE, load_configuration, protocol


def main():
    experiment = ROOT / read_json(BASE / 'latest.json')['experiment']
    plan, spec = read_json(experiment / 'index.json'), read_json(experiment / 'spec.json')
    assert plan['status'] == 'complete' and plan['scope'] == 'training_methods_only'
    assert len(plan['runs']) == 20
    assert spec['configuration'] == load_configuration() and spec['protocol_sha256'] == protocol()
    assert read_json(experiment / 'verification.json')['status'] == 'passed'
    assert read_json(experiment / 'verification.json')['search_runs'] == 20
    assert all(r['search_phase'].startswith('matrix_') for r in plan['runs'])
    snapshot = experiment / 'matrix_stage_snapshot'
    snapshot.mkdir(exist_ok=True)
    for name in ['index.json', 'selection.json', 'verification.json', 'scope_amendment.json', 'controller.json']:
        source = experiment / name
        target = snapshot / name
        assert not target.exists(), 'Preserve the original matrix-stage snapshot'
        target.write_bytes(source.read_bytes())
    plan['matrix_selected'] = plan.pop('selected')
    plan['matrix_completed'] = plan.pop('completed')
    plan.update(scope='training_methods_and_learning_rates', expected_search_runs=28,
                learning_rate_search_deferred=False, status='running', phase='alpha_screen', updated=stamp())
    write_json(experiment / 'index.json', plan)
    write_json(experiment / 'scope_amendment.json', {
        'experiment': plan['experiment'], 'scope': plan['scope'], 'expected_search_runs': 28,
        'learning_rate_search_deferred': False, 'new_training_runs': 8,
        'reason': 'User requested learning-rate search after the completed training-method comparison',
        'learning_rates': spec['configuration']['learning_rates'], 'matrix_winners': plan['matrix_selected'],
        'snapshot_index_sha256': file_hash(snapshot / 'index.json'), 'requested': stamp()})
    print('Enabled alpha search: 20 matrix runs retained; 8 new runs planned.', flush=True)


if __name__ == '__main__':
    main()
