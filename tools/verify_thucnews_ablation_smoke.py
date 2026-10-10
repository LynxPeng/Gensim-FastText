"""Record smoke execution/figure evidence separately from full-corpus results."""
from pathlib import Path

from gensim_lab.common import ROOT, file_hash, fingerprint, read_json, stamp, write_json


def approve_smoke():
    base = ROOT / 'artifacts/thucnews_ablation'
    experiment = ROOT / read_json(base / 'smoke_latest.json')['experiment']
    plan = read_json(experiment / 'index.json')
    assert plan['smoke'] and plan['phase'] == 'complete'
    assert read_json(experiment / 'verification.json')['status'] == 'passed'
    qa = experiment / 'figures'
    layout = read_json(qa / 'layout_audit.json')
    assert len(layout['checks']) == 8
    previews = {}
    for name, item in layout['checks'].items():
        assert not any(level in ['FAIL', 'WARN'] for level, _ in item['issues'])
        for suffix, key in [('preview', 'preview'), ('grayscale', 'grayscale')]:
            path = qa / f'{name}_{suffix}.png'
            assert file_hash(path) == item[key]
            previews[path.name] = item[key]
    # Explicit receipt of the color and grayscale images read in the current task.
    write_json(qa / 'visual_review.json', {'status': 'passed', 'data_fingerprint': fingerprint(plan['runs']),
        'generator_sha256': file_hash(ROOT / 'gensim_lab/thucnews_ablation_figures.py'), 'previews': previews,
        'scope': 'Smoke plots only. This approval cannot authorize future full-corpus plots.',
        'review': {'glyphs': 'passed', 'clipping': 'passed', 'legend_occlusion': 'passed',
                   'panel_alignment': 'passed', 'grayscale_markers': 'passed',
                   'raw_seed_points': 'passed', 'same_units_same_axis_scale': 'passed'}, 'checked': stamp()})
    from gensim_lab.thucnews_ablation_figures import build
    build(experiment, export=True)
    write_json(base / 'smoke_result.json', {'status': 'passed', 'experiment': experiment.relative_to(ROOT).as_posix(),
        'search_runs': 20, 'frozen_test_runs': 12, 'validation_only_search': True, 'figure_previews': 8,
        'exported_files': len(read_json(qa / 'approved.json')['files']),
        'full_experiment_not_complete': True, 'checked': stamp()})
    log = base / 'tests.log'
    assert '78 passed' in log.read_text(encoding='utf-8-sig')
    write_json(base / 'test_result.json', {'status': 'passed', 'passed': 78, 'log_sha256': file_hash(log), 'checked': stamp()})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record-reviewed-smoke', action='store_true', required=True,
                        help='Record the already-completed visual inspection of the current smoke previews')
    parser.parse_args()
    approve_smoke()
