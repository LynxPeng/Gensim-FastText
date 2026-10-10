"""Check subset traceability, Markdown resources, and preview contact sheets."""
import argparse
import collections
import csv
import re
from urllib.parse import unquote

from PIL import Image, ImageDraw

from gensim_lab.common import ROOT, file_hash, read_json, stamp, write_json


def check():
    base = ROOT / 'artifacts/thucnews_small_search'
    experiment = ROOT / read_json(base / 'latest.json')['experiment']
    plan = read_json(experiment / 'index.json')
    manifest = read_json(ROOT / plan['manifest'])
    directory = ROOT / manifest['directory']
    with (directory / 'selected_articles.csv').open(encoding='utf-8-sig') as stream:
        articles = list(csv.DictReader(stream))
    counts = collections.Counter(r['category'] for r in articles)
    assert dict(counts) == manifest['categories']
    assert len(articles) == len({r['id'] for r in articles}) == len({r['text_sha256'] for r in articles}) == 50000
    assert sum(int(r['tokens']) for r in articles) == manifest['partitions']['train']['tokens']
    assert sum(int(r['sentences']) for r in articles) == manifest['partitions']['train']['sentences']
    assert manifest['holdout_overlap'] == 0
    assert all(file_hash(directory / name) == sha for name, sha in manifest['files'].items())
    paths = ['README.md', 'docs/THUCNews小规模搜索方案.md', 'reports/THUCNews小规模超参数搜索.md', 'reports/Gensim&FastText.md']
    errors, n = [], 0
    for relative in paths:
        path = ROOT / relative
        text = path.read_text(encoding='utf-8')
        assert not re.search(r'(?m)^\|.*\b(?:nan|inf)\b', text, re.I)
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', text):
            local = unquote(target.strip('<>').split('#', 1)[0])
            if not local or re.match(r'[a-zA-Z]+://', local):
                continue
            n += 1
            if not (path.parent / local).exists():
                errors.append(f'{relative}: {target}')
    assert not errors, errors
    result = {'status': 'passed', 'checked': stamp(), 'sample_documents': len(articles),
              'sample_file_hashes_verified': True, 'local_links': n, 'experiment': plan['experiment']}
    write_json(base / 'document_check.json', result)
    print(result)


def contact():
    base = ROOT / 'artifacts/thucnews_small_search'
    experiment = ROOT / read_json(base / 'latest.json')['experiment']
    qa = experiment / 'figures'
    for suffix in ['preview', 'grayscale']:
        paths = sorted(p for p in qa.glob(f'*_{suffix}.png') if not p.name.startswith('contact_'))
        if not paths:
            continue
        sheet = Image.new('RGB', (1800, 550 * ((len(paths)+1)//2)), 'white')
        draw = ImageDraw.Draw(sheet)
        for i, path in enumerate(paths):
            x, y = (i % 2) * 900, (i // 2) * 550
            draw.text((x+8, y+8), path.name, fill='black')
            with Image.open(path) as original:
                original.thumbnail((880, 510))
                sheet.paste(original.convert('RGB'), (x+(900-original.width)//2, y+30))
        sheet.save(qa / f'contact_{suffix}.png')
    print(qa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--contact', action='store_true')
    args = parser.parse_args()
    contact() if args.contact else check()
