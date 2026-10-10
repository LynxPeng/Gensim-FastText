"""Predict a segmented article with a saved PFR classification run."""
import argparse
import json
from pathlib import Path

from .common import ROOT, file_hash, read_json
from .pfr_frozen_classification import FrozenRepresentation, cached_run, labels_for, predict_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--tokens', required=True, help='Clean, whitespace-separated article tokens, without IDs or POS suffixes')
    args = parser.parse_args()
    out = args.run if args.run.is_absolute() else ROOT / args.run
    spec, plan = read_json(out / 'spec.json'), read_json(out.parent.parent / 'plan.json')
    assert cached_run(out, spec)
    tokens = args.tokens.split()
    if not tokens or any(w.startswith('__label__') for w in tokens):
        raise ValueError('Provide nonempty segmented article text without labels')
    doc = {'id': 'inference', 'month': 'inference', 'tokens': tokens}
    method, head = spec['method'], spec['head']
    x, coverage = None, None
    if head != 'native':
        source = plan['sources'][method]
        assert all(file_hash(ROOT / source['directory'] / name) == digest for name, digest in source['files'].items())
        x, stats = FrozenRepresentation(method, source).transform([doc], 1)
        coverage = stats[0]
    labels = labels_for(plan['profiles'][spec['profile']]['classes'])
    prediction, scores = predict_saved(out, method, head, x, [doc], labels)
    print(json.dumps({'page_label': prediction[0], 'score': float(scores[0]),
                     'score_kind': 'probability' if head in ['softmax', 'native'] else 'decision_margin',
                     'coverage': coverage}, ensure_ascii=False))


if __name__ == '__main__':
    main()
