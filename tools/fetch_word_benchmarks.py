"""Archive author-published Chinese evaluation data; never execute remote code."""
import json
import urllib.request
from pathlib import Path
from gensim_lab.common import ROOT, file_hash, stamp, write_json

DEST = ROOT / 'references' / 'word_benchmarks'
PINNED_COMMITS = {'Leonard-Xu/CWE': 'bc55e88a4be9480d585f009e36ce169ec041ae15',
                  'Embedding/Chinese-Word-Vectors': '3096a475f79f218dde7b413dcc6b507d302b7f46'}

def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Gensim-coursework-evaluation'})
    return urllib.request.urlopen(req, timeout=60).read()

def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / 'manifest.json').exists():
        previous = json.loads((DEST / 'manifest.json').read_text(encoding='utf-8'))
        for item in previous['files']:
            if file_hash(ROOT / item['path']) != item['sha256']:
                raise ValueError('Existing archive checksum mismatch; do not overwrite pinned data')
        print('Pinned benchmark archive already verified; no network request')
        return
    manifest = {'created': stamp(), 'repositories': [], 'files': []}
    for repo in ['Leonard-Xu/CWE', 'Embedding/Chinese-Word-Vectors']:
        info = json.loads(get(f'https://api.github.com/repos/{repo}'))
        sha = PINNED_COMMITS[repo]
        tree = json.loads(get(f'https://api.github.com/repos/{repo}/git/trees/{sha}?recursive=1'))['tree']
        paths = [x['path'] for x in tree if x['type'] == 'blob' and
                 (x['path'].startswith('data/') if repo.endswith('CWE') else x['path'].startswith('testsets/'))]
        print(repo, sha, paths, flush=True)
        # All evaluation text files are archived; no vectors or executable source.
        selected = [p for p in paths if p.endswith(('.txt', '.md'))]
        selected += [x['path'] for x in tree if x['path'].lower() in ['readme.md', 'license', 'license.txt']]
        manifest['repositories'].append({'repository': repo, 'commit': sha, 'license_api': info.get('license')})
        for p in selected:
            url = f'https://raw.githubusercontent.com/{repo}/{sha}/{p}'
            data = get(url)
            if len(data) > 2_000_000:
                raise ValueError(f'Unexpectedly large benchmark: {p}')
            out = DEST / repo.split('/')[-1] / p
            out.parent.mkdir(parents=True, exist_ok=True)
            # Upstream readmes stored as Markdown; plain numerical datasets remain text resources.
            out.write_bytes(data)
            manifest['files'].append({'path': str(out.relative_to(ROOT)).replace('\\', '/'),
                                      'url': url, 'sha256': file_hash(out), 'bytes': len(data)})
    write_json(DEST / 'manifest.json', manifest)

if __name__ == '__main__':
    main()
