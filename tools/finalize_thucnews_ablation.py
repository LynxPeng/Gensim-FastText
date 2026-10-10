"""One-time dependent report refresh after the already-running experiment.

Does not launch training, schedule future tasks, approve plots, or modify indices.
"""
import argparse
import subprocess
import sys
import time

from gensim_lab.common import ROOT, fingerprint, read_json, stamp, write_json
from gensim_lab.thucnews_ablation_report import report


def run(wait=False):
    base = ROOT / 'artifacts/thucnews_ablation'
    experiment = ROOT / read_json(base / 'latest.json')['experiment']
    state = {'experiment': experiment.relative_to(ROOT).as_posix(), 'started': stamp()}
    deadline = time.monotonic() + 24 * 3600
    while True:
        if read_json(base / 'latest.json')['experiment'] != state['experiment']:
            state.update(status='superseded', completed=stamp())
            write_json(base / 'finalization.json', state)
            return
        progress = read_json(base / 'progress.json')
        if progress.get('experiment') == state['experiment'] and progress.get('status') == 'failed':
            report(experiment)
            state.update(status='training_failed', error=progress['error'], completed=stamp())
            write_json(base / 'finalization.json', state)
            raise RuntimeError(progress['error'])
        plan = read_json(experiment / 'index.json')
        status_path, layout_path = experiment / 'report_status.json', experiment / 'figures/layout_audit.json'
        ready = (progress.get('status') == 'complete' and not progress.get('smoke')
                 and plan['phase'] == 'complete' and status_path.exists() and layout_path.exists())
        if ready:
            ready = (read_json(status_path)['generated'] >= progress['updated']
                     and read_json(layout_path)['data_fingerprint'] == fingerprint(plan['runs']))
        if ready:
            break
        if not wait:
            raise RuntimeError('Full experiment is still running; use --wait for a one-time dependent refresh')
        if time.monotonic() > deadline:
            raise TimeoutError('One-time report refresher exceeded 24 hours; training is untouched')
        state.update(status='waiting_for_training', checked=stamp())
        write_json(base / 'finalization.json', state)
        time.sleep(30)
    report(experiment)
    result = subprocess.run([sys.executable, str(ROOT / 'tools/build_consolidated_report.py')], cwd=ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    (base / 'consolidated_build.log').write_text(result.stdout, encoding='utf-8')
    state.update(status='complete' if result.returncode == 0 else 'report_check_failed', completed=stamp(),
                 figures='Full-data color/grayscale previews require visual review before publication')
    write_json(base / 'finalization.json', state)
    if result.returncode:
        raise RuntimeError('Report check failed: artifacts/thucnews_ablation/consolidated_build.log')
    print('Refreshed full experiment and consolidated reports. Numerical results are complete; figure review is separate.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait', action='store_true')
    run(parser.parse_args().wait)
