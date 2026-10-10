"""Read-only live process/resource audit; no training or experiment-index writes."""
import argparse
import re
from pathlib import Path
from urllib.parse import unquote

import psutil
from PIL import Image, ImageDraw

from gensim_lab.common import ROOT, file_hash, read_json, stamp, write_json


def process_receipt():
    matches = []
    for process in psutil.process_iter(['pid', 'cmdline', 'create_time']):
        try:
            cmd = process.info['cmdline'] or []
            if 'gensim_lab.thucnews_ablation' in cmd and '--worker' not in cmd and '--test-worker' not in cmd and '--smoke' not in cmd:
                matches.append({'pid': process.pid, 'create_time': process.info['create_time'], 'cmdline': cmd,
                                'children': [p.pid for p in process.children(recursive=True)]})
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    assert len(matches) <= 1, 'Multiple full-training orchestrators'
    result = {'checked': stamp(), 'visible_processes': matches,
              'visibility_note': 'Process inventory is sandbox-scoped; an empty list does not prove training stopped.',
              'free_disk_gib': psutil.disk_usage(str(ROOT)).free / 2**30}
    write_json(ROOT / 'artifacts/thucnews_ablation/process.json', result)
    return result


def links():
    errors, n = [], 0
    for name in ['reports/THUCNews训练方式消融实验.md', 'reports/Gensim&FastText.md', 'docs/THUCNews训练方式消融方案.md']:
        source = ROOT / name
        content = source.read_text(encoding='utf-8')
        assert not re.search(r'(?m)^\|.*\b(?:nan|inf)\b', content, re.I)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', content):
            path = unquote(target.strip('<>').split('#', 1)[0])
            if not path or re.match(r'[a-zA-Z]+://', path):
                continue
            if not (source.parent / path).exists():
                errors.append(f'{name}: {target}')
            n += 1
    assert not errors, errors
    result = {'status': 'passed', 'checked': stamp(), 'local_links': n}
    write_json(ROOT / 'artifacts/thucnews_ablation/document_check.json', result)
    return result


def contact(smoke=False):
    base = ROOT / 'artifacts/thucnews_ablation'
    experiment = ROOT / read_json(base / ('smoke_latest.json' if smoke else 'latest.json'))['experiment']
    qa = experiment / 'figures'
    for suffix in ['preview', 'grayscale']:
        paths = sorted(p for p in qa.glob(f'*_{suffix}.png') if not p.name.startswith('contact_'))
        if not paths:
            continue
        contact = Image.new('RGB', (1400, 440 * ((len(paths) + 1) // 2)), 'white')
        draw = ImageDraw.Draw(contact)
        for index, path in enumerate(paths):
            x, y = (index % 2) * 700, (index // 2) * 440
            draw.text((x + 8, y + 8), path.name, fill='black')
            with Image.open(path) as image:
                image.thumbnail((680, 410))
                contact.paste(image.convert('RGB'), (x + (700 - image.width) // 2, y + 26))
        contact.save(qa / f'contact_{suffix}.png')
    return qa


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--contact', action='store_true')
    args = parser.parse_args()
    if args.contact:
        print(contact(args.smoke))
    else:
        print(process_receipt())
        print(links())
