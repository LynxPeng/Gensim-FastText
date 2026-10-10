"""Evidence-bound paper figures: profile, preview, review, then export."""
from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path

from .common import ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .paper_reproduction import records, REPORT
from .paper_data import PAPER_ARTIFACTS, settings

WORK = PAPER_ARTIFACTS / 'figures'
OUTPUT = ROOT / 'reports/figures/paper_reproduction'
SKILL = Path(r'C:/Users/wangj/.codex/skills/scipilot-figure-skill/scripts')
METHODS = [('native_fasttext', 1, '原生 unigram', '#0072B2', 'o'),
           ('native_fasttext', 2, '原生 bigram', '#D55E00', 's'),
           ('gensim_fasttext', 1, 'Gensim 两阶段', '#009E73', '^')]


def groups(task, datasets=None):
    result = {}
    for directory, summary in records():
        spec = summary['spec']
        if spec['task'] != task or (datasets and spec['dataset'] not in datasets):
            continue
        key = (spec['dataset'], spec['dimension'], spec['method'], spec['word_ngrams'])
        result.setdefault(key, []).append({'run': directory.name, 'seed': spec['seed'],
                                          'value': summary['metrics']['test']['precision_at_1']})
    return {key: sorted(values, key=lambda value: value['seed']) for key, values in result.items()
            if {value['seed'] for value in values} == set(settings()['seeds']) and len(values) == 3}


def figure_data(task, datasets=None):
    grouped = groups(task, datasets)
    candidates = [(name, dimension) for name in settings()['classification']['datasets']
                  for dimension in [settings()['classification']['dimension']]] if task == 'classification' else [
                      ('yfcc100m', dimension) for dimension in settings()['tags']['dimensions']]
    candidates = [(name, dim) for name, dim in candidates if all(
        (name, dim, method, ngram) in grouped for method, ngram, *_ in METHODS)]
    data = [{'dataset': name, 'dimension': dim, 'method': method, 'word_ngrams': ngram,
             'values': grouped[name, dim, method, ngram]}
            for name, dim in candidates for method, ngram, *_ in METHODS]
    return data, candidates


def draw(task, data, candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(SKILL))
    from setup_style import setup_style
    setup_style(journal='nature', lang='zh', use_sciplots=False)
    plt.rcParams['figure.constrained_layout.use'] = False
    plt.rcParams['figure.autolayout'] = False
    figure, axis = plt.subplots(figsize=(7.2, max(2.8, len(candidates) * .68 + 1.35)))
    for position, (name, dimension) in enumerate(candidates):
        for method_index, (method, ngram, label, color, marker) in enumerate(METHODS):
            row = next(row for row in data if (row['dataset'], row['dimension'], row['method'], row['word_ngrams'])
                       == (name, dimension, method, ngram))
            values = [entry['value'] * 100 for entry in row['values']]
            center = position + (method_index - 1) * .19
            axis.scatter(values, [center + shift for shift in [-.04, 0, .04]],
                         marker=marker, s=23, color=color, edgecolors='white', linewidths=.3,
                         label=label if position == 0 else None, zorder=3)
    labels = [name.replace('_', ' ') if task == 'classification' else f'YFCC100M · {dimension} 维'
              for name, dimension in candidates]
    axis.set_yticks(range(len(candidates)), labels)
    axis.set_ylim(len(candidates) - .5, -.5)
    axis.set_xlim(0, 100)
    axis.set_xlabel('测试 Accuracy (%)' if task == 'classification' else '测试 P@1 (%)')
    axis.grid(axis='x', color='#dddddd', linewidth=.5)
    axis.legend(loc='lower center', bbox_to_anchor=(.5, 1.04), ncol=3, frameon=False)
    figure.subplots_adjust(left=.27, right=.97, bottom=.24 if len(candidates) == 1 else .12,
                           top=.76 if len(candidates) == 1 else .86)
    return figure


def prepare(task):
    sys.path.insert(0, str(SKILL))
    from visual_qa import render_preview, audit_layout
    data, candidates = figure_data(task)
    WORK.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in data:
        values = [entry['value'] for entry in row['values']]
        rows.append([row['dataset'], row['dimension'], row['method'], row['word_ngrams'], 3,
                     f'{statistics.mean(values)*100:.3f}', f'{statistics.stdev(values)*100:.3f}',
                     f'{(max(values)-min(values))*100:.3f}'])
    variation = max((max(statistics.mean([v['value'] for v in row['values']]) for row in data
                         if (row['dataset'], row['dimension']) == candidate)
                     - min(statistics.mean([v['value'] for v in row['values']]) for row in data
                           if (row['dataset'], row['dimension']) == candidate)
                     for candidate in candidates), default=0)
    metadata = {'task': task, 'prepared': stamp(), 'data': data, 'data_fingerprint': fingerprint(data),
                'datasets': [name for name, _ in candidates], 'dimensions': [dim for _, dim in candidates],
                'between_method_max_range_pp': variation * 100,
                'decision': 'figure' if variation >= .005 else 'table', 'issues': []}
    profile = [f'# {task} 复现结果绘图剖析',
               '每组为三个训练种子，统计单位是训练运行，不是测试文章。直接列出各点，不画小样本均值柱，'
               '不连接不同数据集。横轴从 0 到 100，颜色与点形双重区分路线。'
               '路线均值最大差小于 0.5 个百分点时仅保留表格；阈值仅用于展示，不是显著性标准。',
               markdown_table(['数据集','维度','路线','词 ngram','n','均值 (%)','样本 SD (百分点)','种子极差 (百分点)'], rows),
               f'当前有 {len(candidates)} 个数据集/维度组合完成全部路线和三个种子；仅展示这些组合。']
    (WORK / f'{task}_profile.md').write_text('\n\n'.join(profile)+'\n', encoding='utf-8')
    if metadata['decision'] == 'figure':
        import matplotlib.pyplot as plt
        figure = draw(task, data, candidates)
        preview = WORK / f'{task}_preview.png'
        render_preview(figure, str(preview), dpi=150)
        metadata['issues'] = audit_layout(figure)
        metadata['preview_sha256'] = file_hash(preview)
        plt.close(figure)
    write_json(WORK / f'{task}_preview.json', metadata)
    return metadata


def publish(task):
    sys.path.insert(0, str(SKILL))
    from export_figure import export_figure
    from check_figure import check_figure
    metadata = read_json(WORK / f'{task}_preview.json')
    review = read_json(WORK / f'{task}_visual_review.json')
    data, candidates = figure_data(task, metadata['datasets'])
    if (review.get('passed') is not True or review['preview_sha256'] != metadata['preview_sha256']
            or fingerprint(data) != metadata['data_fingerprint'] or any(level == 'FAIL' for level, _ in metadata['issues'])):
        raise ValueError('Preview review or underlying data no longer matches')
    import matplotlib.pyplot as plt
    figure = draw(task, data, candidates)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = export_figure(figure, str(OUTPUT / task), formats=['png', 'svg'], dpi=300,
                          size_inches=tuple(figure.get_size_inches()), grayscale_preview=True, tight=False)
    # Pillow's grayscale conversion in the skill exporter drops DPI metadata.
    from PIL import Image
    grayscale = OUTPUT / f'{task}_grayscale.png'
    with Image.open(grayscale) as original:
        pixels = original.copy()
    pixels.save(grayscale, dpi=(300, 300))
    issues = []
    for path in paths:
        checked, _ = check_figure(path)
        issues.extend(checked)
    plt.close(figure)
    if any(level == 'FAIL' for level, _ in issues):
        raise ValueError(f'Figure export failed audit: {issues}')
    metadata.update(published=stamp(), exported_files={Path(path).name: file_hash(path) for path in paths},
                    export_issues=issues, visual_review=review)
    write_json(WORK / f'{task}_published.json', metadata)


def attach_published():
    text = REPORT.read_text(encoding='utf-8')
    for task, anchor in [('classification', 'reproduction-tags'), ('tags', 'reproduction-evidence')]:
        receipt = WORK / f'{task}_published.json'
        if not receipt.exists():
            continue
        metadata = read_json(receipt)
        data, _ = figure_data(task, metadata['datasets'])
        if fingerprint(data) != metadata['data_fingerprint']:
            continue
        if any(not (OUTPUT / name).exists() or file_hash(OUTPUT / name) != digest
               for name, digest in metadata['exported_files'].items()):
            continue
        caption = ('分类 Accuracy' if task == 'classification' else '标签 P@1') + '：' + ', '.join(metadata['datasets']) + '。每组显示三个种子的实际结果，未进行显著性检验。'
        block = f'![{caption}](figures/paper_reproduction/{task}.png)\n\n{caption}\n\n'
        if f'figures/paper_reproduction/{task}.png' not in text:
            text = text.replace(f'<a id="{anchor}"></a>', block + f'<a id="{anchor}"></a>')
    REPORT.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'publish', 'attach'], default='prepare')
    parser.add_argument('--task', choices=['classification', 'tags'], default='classification')
    args = parser.parse_args()
    if args.stage == 'prepare':
        print({k:v for k,v in prepare(args.task).items() if k != 'data'})
    elif args.stage == 'publish':
        publish(args.task)
    else:
        attach_published()


if __name__ == '__main__':
    main()
