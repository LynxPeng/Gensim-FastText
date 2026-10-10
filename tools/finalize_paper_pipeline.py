"""Finish this single full run after its terminal log marker, without training."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gensim_lab.common import stamp, write_json
from gensim_lab.paper_data import PAPER_ARTIFACTS
from gensim_lab.integrity import artifact_lock


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=Path, required=True)
    args = parser.parse_args()
    path = args.log.resolve()
    if not path.is_relative_to(PAPER_ARTIFACTS.resolve()):
        raise ValueError('Pipeline log must be within this project\'s paper artifacts')
    state_path = PAPER_ARTIFACTS / 'finalization.json'
    started = stamp()
    terminal, failed = None, False
    write_json(state_path, {'status': 'waiting_for_training', 'started': started, 'log': str(path)})
    while terminal is None and not failed:
        if path.exists():
            with path.open('rb') as stream:
                stream.seek(max(0, path.stat().st_size - 32768))
                tail = stream.read().decode('utf-8', errors='replace')
            for line in reversed(tail.splitlines()):
                try:
                    candidate = json.loads(line)
                except ValueError:
                    continue
                if isinstance(candidate, dict) and candidate.get('expected') == 90 and 'completed' in candidate:
                    terminal = candidate
                    break
            failed = 'Traceback (most recent call last):' in tail
        if terminal is None and not failed:
            time.sleep(10)
    # The training process has printed its last report or a terminal traceback.
    # Acquire the same lock before touching its final artifacts.
    for attempt in range(60):
        try:
            with artifact_lock():
                complete = bool(terminal and terminal['completed'] == terminal['expected'] and not failed)
                audit = [sys.executable, 'tools/audit_paper_reproduction.py', '--hash-models']
                if complete:
                    audit.append('--require-complete')
                checked = subprocess.run(audit, cwd=ROOT)
                if checked.returncode:
                    complete = False
                subprocess.run([sys.executable, '-m', 'gensim_lab.paper_reproduction', '--stage', 'report'], cwd=ROOT, check=True)
                previews = []
                if complete:
                    for task in ['classification', 'tags']:
                        subprocess.run([sys.executable, '-m', 'gensim_lab.paper_figures', '--stage', 'prepare', '--task', task], cwd=ROOT, check=True)
                        previews.append(f'artifacts/paper_reproduction/figures/{task}_preview.json')
                result = {'status': 'training_verified_pending_full_figure_review' if complete else 'training_incomplete',
                          'started': started, 'finished': stamp(), 'terminal': terminal,
                          'training_log_failure': failed, 'audit_returncode': checked.returncode,
                          'figure_previews': previews,
                          'note': 'Final full-data figure previews require actual visual review before publication.'}
                write_json(state_path, result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
                return
        except RuntimeError as error:
            if 'Another artifact-writing process' not in str(error):
                raise
            time.sleep(1)
    raise RuntimeError('Finalizer could not acquire the artifact lock; training outputs preserved')


if __name__ == '__main__':
    main()
