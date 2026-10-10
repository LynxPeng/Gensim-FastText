"""Label-free diagnostics for the existing full-data TF-IDF representation."""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.models.word2vec import LineSentence
from PIL import Image
from threadpoolctl import threadpool_limits

from gensim_lab.common import ROOT, file_hash, read_json, stamp, write_json
from gensim_lab.figure_support.profile_data import profile_data, render_report
from gensim_lab.runtime import numerical_threads
from gensim_lab.submission_figures import safe

SOURCE = ROOT / 'artifacts/thucnews/runs/928ed691978c06e8cc9d3d1e'
QA = ROOT / 'artifacts/tfidf_submission_diagnostics' / SOURCE.name
OUT = ROOT / 'reports/figures/section11/tfidf_document_diagnostics'


def summarize(values):
    return {name: float(value) for name, value in zip(
        ['min', 'q05', 'q25', 'median', 'q75', 'q95', 'max'],
        np.quantile(values, [0, .05, .25, .5, .75, .95, 1]))} | {'mean': float(np.mean(values))}


def analyze():
    QA.mkdir(parents=True, exist_ok=True)
    seal = read_json(SOURCE / 'seal.json')
    summary = read_json(SOURCE / 'summary.json')
    sources = {}
    for name in ['dictionary.gensim', 'model.gensim', 'summary.json', 'heldout_summary.json']:
        assert file_hash(SOURCE / name) == seal['files'][name], name
        sources[(SOURCE / name).relative_to(ROOT).as_posix()] = file_hash(SOURCE / name)
    manifest_path = ROOT / summary['spec']['manifest']
    manifest = read_json(manifest_path)
    data = ROOT / manifest['directory']
    for name in ['holdout.txt', 'holdout_sentences.csv']:
        assert file_hash(data / name) == manifest['files'][name], name
        sources[(data / name).relative_to(ROOT).as_posix()] = file_hash(data / name)
    sources[manifest_path.relative_to(ROOT).as_posix()] = file_hash(manifest_path)
    dictionary = Dictionary.load(str(SOURCE / 'dictionary.gensim'))
    model = TfidfModel.load(str(SOURCE / 'model.gensim'))
    reloaded = TfidfModel.load(str(SOURCE / 'model.gensim'))
    assert dictionary.num_docs == manifest['partitions']['train']['documents']
    assert len(dictionary) == summary['vocabulary_size'] == 50000
    documents = collections.OrderedDict()
    seen_sentences = 0
    with (data / 'holdout_sentences.csv').open(encoding='utf-8-sig', newline='') as stream:
        metadata = csv.DictReader(stream)
        for tokens, row in itertools.zip_longest(LineSentence(str(data / 'holdout.txt')), metadata):
            assert tokens is not None and row is not None
            assert len(tokens) == int(row['tokens'])
            identifier = row['document_id']
            item = documents.setdefault(identifier, {'counts': collections.Counter(), 'sentences': 0})
            assert int(row['sentence_index']) == item['sentences']
            item['counts'].update(tokens)
            item['sentences'] += 1
            seen_sentences += 1
    assert seen_sentences == manifest['partitions']['holdout']['sentences']
    assert len(documents) == manifest['partitions']['holdout']['documents'] == 5000
    rows = []
    for identifier, item in documents.items():
        counts = item['counts']
        bow = sorted((dictionary.token2id[word], count) for word, count in counts.items()
                     if word in dictionary.token2id)
        vector = model[bow]
        assert vector == reloaded[bow]
        assert all(math.isfinite(weight) and weight > 0 for _, weight in vector)
        norm = math.sqrt(math.fsum(weight * weight for _, weight in vector))
        assert not vector or abs(norm - 1) < 1e-12
        tokens = sum(counts.values())
        iv = sum(count for _, count in bow)
        rows.append({'document_id': identifier, 'tokens': tokens, 'iv_tokens': iv,
                     'vocabulary_percent': 100 * iv / tokens, 'nonzero_features': len(vector),
                     'density_percent': 100 * len(vector) / len(dictionary), 'l2_norm': norm})
    frame = pd.DataFrame(rows)
    assert int(frame.tokens.sum()) == summary['heldout']['tokens']
    assert int(frame.iv_tokens.sum()) == summary['heldout']['iv_tokens']
    assert frame.nonzero_features.min() >= 0 and frame.nonzero_features.max() <= len(dictionary)
    frame.to_csv(QA / 'holdout_documents.csv', index=False, encoding='utf-8-sig')
    result = {'generated': stamp(), 'model_retrained': False, 'labels_used': False,
              'source_run': SOURCE.relative_to(ROOT).as_posix(), 'sources': sources,
              'generator_sha256': file_hash(Path(__file__)), 'documents': len(frame),
              'sentences': seen_sentences, 'dimensions': len(dictionary),
              'zero_vectors': int((frame.nonzero_features == 0).sum()),
              'word_occurrences': int(frame.tokens.sum()), 'retained_word_occurrences': int(frame.iv_tokens.sum()),
              'weighted_vocabulary_percent': 100 * frame.iv_tokens.sum() / frame.tokens.sum(),
              'nonzero_features': summarize(frame.nonzero_features),
              'density_percent': summarize(frame.density_percent),
              'document_vocabulary_percent': summarize(frame.vocabulary_percent),
              'l2_norm': summarize(frame.l2_norm),
              'max_normalization_error': float(np.abs(frame.loc[frame.nonzero_features > 0, 'l2_norm'] - 1).max()),
              'reload_vectors_exact': True}
    write_json(QA / 'summary.json', result)
    profile = profile_data(frame.drop(columns='document_id'))
    write_json(QA / 'data_profile.json', safe(profile))
    (QA / '数据剖析与图型选择.md').write_text(
        render_report(profile) + '\n\n## 展示目标与图型\n\n'
        '展示 5,000 篇留出文章的稀疏度和词表覆盖差异。使用两面板直方图，分别显示非零特征数'
        '和逐篇词次覆盖率；完整保留观测范围，纵轴为文章数。备选为 ECDF 或箱线图。'
        '这些文章属于一次模型运行，不作为独立训练重复，不绘制种子误差或显著性。'
        '归一化与重载一致性作为数值核验，不绘制近乎常量曲线。\n', encoding='utf-8')
    print({key: result[key] for key in ['documents', 'zero_vectors', 'weighted_vocabulary_percent',
                                      'nonzero_features', 'density_percent', 'max_normalization_error']})


def figure(export=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gensim_lab.figure_support.setup_style import setup_style
    from gensim_lab.figure_support.visual_qa import audit_layout, render_preview
    from gensim_lab.figure_support.export_figure import export_figure
    from gensim_lab.figure_support.check_figure import check_figure

    result = read_json(QA / 'summary.json')
    assert result['generator_sha256'] == file_hash(Path(__file__))
    for path, digest in result['sources'].items():
        assert file_hash(ROOT / path) == digest
    frame = pd.read_csv(QA / 'holdout_documents.csv', dtype={'document_id': str})
    setup_style(journal='general', lang='zh', use_sciplots=False)
    plt.rcParams.update({'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 10,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout='constrained')
    settings = [('nonzero_features', 'a 每篇文章的非零特征数', '非零特征数',
                 np.linspace(0, frame.nonzero_features.max() + 1, 41)),
                ('vocabulary_percent', 'b 每篇文章的词表覆盖率', '占该文章词次的比例（%）',
                 np.linspace(0, 100, 41))]
    bins_record = []
    for ax, (column, title, label, bins) in zip(axes, settings):
        counts, edges, _ = ax.hist(frame[column], bins=bins, color='#666666',
                                  edgecolor='white', linewidth=.35)
        assert counts.sum() == len(frame)
        ax.set(title=title, xlabel=label, ylabel='文章数', xlim=(bins[0], bins[-1]), ylim=(0, counts.max() * 1.12))
        ax.grid(axis='y', alpha=.15)
        bins_record.extend({'measure': column, 'left': left, 'right': right, 'documents': int(count)}
                           for count, left, right in zip(counts, edges[:-1], edges[1:]))
    preview = QA / 'preview.png'
    render_preview(fig, str(preview), dpi=150)
    issues = audit_layout(fig)
    assert not any(level in ['WARN', 'FAIL'] for level, _ in issues), issues
    with Image.open(preview) as im:
        im.convert('L').save(QA / 'preview_grayscale.png', dpi=(150, 150))
    pd.DataFrame(bins_record).to_csv(QA / 'histogram_bins.csv', index=False, encoding='utf-8-sig')
    audit = {'generated': stamp(), 'issues': issues, 'preview_sha256': file_hash(preview)}
    if export:
        review = read_json(QA / 'visual_review.json')
        assert review['status'] == 'approved' and review['preview_sha256'] == file_hash(preview)
        assert review['grayscale_sha256'] == file_hash(QA / 'preview_grayscale.png')
        paths = export_figure(fig, str(OUT), formats=['png', 'svg'], dpi=300,
                              size_inches=tuple(fig.get_size_inches()), tight=False)
        audit['exports'] = []
        for path in paths:
            checks, metadata = check_figure(path, min_dpi=300,
                target_inches=tuple(fig.get_size_inches()) if path.endswith('.png') else None)
            assert not any(level in ['WARN', 'FAIL'] for level, _ in checks), checks
            audit['exports'].append({'path': str(Path(path).relative_to(ROOT)), 'sha256': file_hash(path),
                                     'checks': checks, 'metadata': metadata})
    write_json(QA / 'figure_audit.json', audit)
    plt.close(fig)
    print('Exported' if export else 'Previewed', preview)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['analyze', 'preview', 'export'], default='analyze')
    args = parser.parse_args()
    with threadpool_limits(limits=numerical_threads()):
        if args.stage == 'analyze':
            analyze()
        else:
            figure(args.stage == 'export')


if __name__ == '__main__':
    main()
