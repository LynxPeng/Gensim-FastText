"""Contiguous sentence-boundary corpus blocks for input-boundary diagnostics."""
from pathlib import Path
from .common import ROOT, file_hash, fingerprint, read_json, write_json


def blocks(manifest, requested):
    source = ROOT / manifest['directory'] / 'train.txt'
    spec = {'train_sha256': manifest['files']['train.txt'], 'blocks': requested, 'code_sha256': file_hash(Path(__file__))}
    directory = ROOT / 'data/thucnews/corpus_blocks' / fingerprint(spec)[:20]
    marker = directory / 'manifest.json'
    if marker.exists():
        result = read_json(marker)
        assert result['spec'] == spec
        assert all(file_hash(ROOT / r['path']) == r['sha256'] for r in result['blocks'])
        return result['blocks']
    directory.mkdir(parents=True, exist_ok=True)
    target = max(1, (source.stat().st_size + requested - 1) // requested)
    rows, output, size, words, sentences = [], None, 0, 0, 0
    def close():
        if output is not None:
            output.close()
            rows.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': size, 'words': words,
                         'sentences': sentences, 'sha256': file_hash(path)})
    with source.open('rb') as stream:
        for line in stream:
            if output is None or size >= target and len(rows) < requested - 1:
                close()
                path = directory / f'block_{len(rows):03d}.txt'
                output = path.open('wb')
                size, words, sentences = 0, 0, 0
            assert line.endswith(b'\n') and line.strip()
            output.write(line)
            size += len(line)
            words += line.count(b' ') + 1
            sentences += 1
    close()
    assert sum(r['words'] for r in rows) == manifest['partitions']['train']['tokens']
    assert sum(r['sentences'] for r in rows) == manifest['partitions']['train']['sentences']
    assert sum(r['bytes'] for r in rows) == source.stat().st_size
    write_json(marker, {'spec': spec, 'blocks': rows})
    return rows
