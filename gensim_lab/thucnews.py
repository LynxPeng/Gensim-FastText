"""THUCNews preparation, native representations and validation-selected training."""
from __future__ import annotations
import argparse
import json
import sys
import subprocess
from pathlib import Path
from gensim.models import KeyedVectors
from .common import ROOT, file_hash, fingerprint, read_json, write_json
from .thucnews_data import PREPARED, fetch, prepare, settings

BASE = ROOT / "artifacts/thucnews"
RUNS = BASE / "runs"
TFIDF_REFERENCE = RUNS / "928ed691978c06e8cc9d3d1e"
TFIDF_INDEX = BASE / 'tfidf_latest.json'


def records():
    result = []
    reference = ROOT/read_json(TFIDF_INDEX)['directory'] if TFIDF_INDEX.exists() else TFIDF_REFERENCE
    if (reference / 'seal.json').exists():
        summary = read_json(reference / 'summary.json')
        expected = settings()['tfidf']
        if all(summary['spec']['dictionary'][k] == expected[k] for k in ['no_below','no_above','keep_n']):
            seal = read_json(reference / 'seal.json')
            assert file_hash(reference / 'summary.json') == seal['files']['summary.json']
            result.append((reference, summary))
    index = ROOT / 'artifacts/thucnews_validation/latest.json'
    if index.exists():
        for method, relative in read_json(index)['models'].items():
            directory = ROOT / relative
            summary = read_json(directory / 'summary.json')
            seal = read_json(directory / 'seal.json')
            assert not summary['spec']['smoke'] and summary['spec']['method'] == method
            assert file_hash(directory / 'summary.json') == seal['files']['summary.json']
            result.append((directory, summary))
    return sorted(result, key=lambda item: item[1]['spec']['method'])


def save_training_model(model, path):
    """Write native Gensim arrays into a single portable checkpoint."""
    model.save(str(path), separately=[], sep_limit=sys.maxsize)


def ensure_tfidf(manifest):
    from .runtime import worker_environment
    matching=[(d,s) for d,s in records() if s['spec']['method']=='tfidf'
              and read_json(ROOT/s['spec']['manifest'])['fingerprint']==manifest['fingerprint']]
    if matching:
        print(f'Reuse unchanged TF-IDF {matching[0][0].name}',flush=True)
        return matching[0][0]
    spec={'method':'tfidf','seed':42,'smoke':False,'dictionary':settings()['tfidf'],
          'manifest':(ROOT/manifest['directory']/'manifest.json').relative_to(ROOT).as_posix(),
          'manifest_sha256':file_hash(ROOT/manifest['directory']/'manifest.json'),
          'data_fingerprint':manifest['fingerprint'],
          'protocol_sha256':{name:file_hash(ROOT/'gensim_lab'/name) for name in ['thucnews_tfidf.py','thucnews.py']}}
    directory=RUNS/fingerprint(spec)[:24]; directory.mkdir(parents=True,exist_ok=True)
    write_json(directory/'spec.json',spec)
    from .thucnews_validation import cache_valid
    if not cache_valid(directory,spec):
        with (directory/'run.log').open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-m','gensim_lab.thucnews','--worker',str(directory)],
                           cwd=ROOT,env=worker_environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    assert cache_valid(directory,spec)
    write_json(TFIDF_INDEX,{'directory':directory.relative_to(ROOT).as_posix()})
    return directory


def train(manifest=None, *, smoke=False):
    from .thucnews_validation import run
    if not smoke:
        if manifest is None:
            import yaml
            from .thucnews_validation import CONFIG
            manifest=read_json(ROOT/yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['manifest'])
        ensure_tfidf(manifest)
        import yaml
        from .thucnews_validation import CONFIG
        # The validation input is explicitly pinned; never silently train another corpus.
        fixed=read_json(ROOT/yaml.safe_load(CONFIG.read_text(encoding='utf-8'))['manifest'])
        if fixed['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Prepared data differs from configs/thucnews_validation.yaml; update its manifest first')
    run(smoke=smoke)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','fetch','prepare','train','report','query'],default='all')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--model',choices=['tfidf','word2vec','fasttext'],default='fasttext')
    parser.add_argument('--word',default='经济')
    args = parser.parse_args()
    if args.worker:
        spec = read_json(args.worker/'spec.json')
        if spec['method'] != 'tfidf':
            raise ValueError('Embedding workers now use gensim_lab.thucnews_validation')
        from .thucnews_tfidf import worker
        worker(args.worker,spec,read_json(ROOT/spec['manifest']))
        return
    if args.stage == 'query':
        directory, summary = next(x for x in records() if x[1]['spec']['method'] == args.model)
        if args.model == 'tfidf':
            from gensim.corpora import Dictionary
            from gensim.models import TfidfModel
            import jieba
            dictionary = Dictionary.load(str(directory/'dictionary.gensim'))
            model = TfidfModel.load(str(directory/'model.gensim'))
            weights = model[dictionary.doc2bow(list(jieba.cut(args.word)))]
            print(json.dumps({'model':args.model,'terms':[(dictionary[i],v) for i,v in weights]},ensure_ascii=False))
        else:
            vectors = KeyedVectors.load(str(directory/'vectors.kv'))
            print(json.dumps({'model':args.model,'word':args.word,'selected_epoch':summary['selected_epoch'],
                              'neighbors':vectors.most_similar(args.word,topn=10)},ensure_ascii=False))
        return
    from .integrity import artifact_lock
    with artifact_lock():
        manifest=None
        if args.stage in ['all','fetch','prepare']:
            source = fetch()
            if args.stage == 'fetch':
                return
            manifest = prepare(source,documents=300 if args.smoke else None)
            if not args.smoke:
                write_json(PREPARED/'latest.json',{'manifest':(ROOT/manifest['directory']/'manifest.json').relative_to(ROOT).as_posix()})
        if args.stage in ['all','train']:
            train(manifest,smoke=args.smoke)
        if args.stage in ['all','report'] and not args.smoke:
            from .thucnews_report import report
            report()


if __name__ == '__main__':
    main()
