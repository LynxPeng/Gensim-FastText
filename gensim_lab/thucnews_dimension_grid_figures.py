"""Separate model figures comparing five embedding dimensions and their costs."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .figure_support.profile_data import profile_data, render_report
from .figure_support.setup_style import setup_style
from .figure_support.visual_qa import audit_layout, render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure
from .thucnews_dimension_grid import rank_grid
from .thucnews_small_search import BASE

COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
STYLES = ['-', '--', '-.', ':', (0, (5, 1, 1, 1, 1, 1))]
MARKERS = ['o', 's', '^', 'D', 'v']
SEED_COLORS = ['#0072B2', '#D55E00', '#009E73']
NAMES = {'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText'}


def build(experiment, export=False):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    assert plan['status'] == 'complete' and plan['kind'] == 'dimension_grid'
    assert rank_grid(plan['runs'], plan['grid']) == plan['dimension_ranking']
    rows, grid = plan['runs'], plan['grid']
    dimensions, seeds = sorted(grid['vector_sizes']), grid['seeds']
    qa = experiment / 'figures'
    qa.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'reports/figures/thucnews_dimension_grid' / experiment.name
    data_fingerprint = fingerprint(rows)
    columns = ['method', 'vector_size', 'model_seed', 'validation_score', 'selected_epoch', 'trained_epochs', 'fit_seconds', 'peak_rss_mib', 'model_bytes']
    save_csv(qa / 'plot_data.csv', [{k: r[k] for k in columns} for r in rows])
    profile = profile_data(str(qa / 'plot_data.csv'), group_cols=['method', 'vector_size'])
    (qa / '数据剖析与图型选择.md').write_text('# 五档向量维数图型选择\n\n' + render_report(profile) +
        '\n\n每模型五个有序数值向量维数，每档真实 n=3；横轴采用实际向量维数的线性数值坐标，显示全部五档刻度。'
        '维数为有序整数，连线仅为已测点的读图引导。主图展示三种子原始点及连线、均值±样本 SD（ddof=1）；连线只帮助比较已测点，不声称中间值。'
        '备选为三种子明细表，不用均值柱或密度图。逐轮图每个种子一个面板，同一面板叠加全部五档向量维数，'
        '两种模型分别出图；星号标记各自最佳检查点，不平滑、不填充早停后数据，也不对不同样本数的轮次求均值。'
        '辅助指标分别展示词相似度与类比探针；同样保留三个原始点及 SD。主指标轴从 0 到 1，'
        '遇到负值扩展到 −1；辅助面板按各指标实测值及SD范围设置纵轴，标清刻度，不比较不同指标的线条高度。'
        '资源指标另用四面板展示累计/每轮时间、峰值 RSS 与原生模型大小，不用双 Y 轴，无显著性标记。样本 SD 反映训练波动，不是泛化误差保证。'
        '颜色、线型和标记冗余编码；机器与彩色/灰度视觉审查通过后导出 PNG 300 DPI、SVG。\n', encoding='utf-8')
    setup_style(journal='general', lang='zh')
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none'})
    review = read_json(qa / 'visual_review.json') if export else None
    if export:
        assert review['status'] == 'passed' and review['data_fingerprint'] == data_fingerprint
        assert review['generator_sha256'] == file_hash(Path(__file__))
    checks, figures, captions, hashes = {}, {}, {}, {}

    def finish(fig, name, caption):
        preview, gray = qa / f'{name}_preview.png', qa / f'{name}_grayscale.png'
        if export:
            assert all(file_hash(p) == review['previews'][p.name] for p in [preview, gray])
        else:
            render_preview(fig, str(preview), dpi=150)
            with Image.open(preview) as original:
                original.convert('L').save(gray)
        issues = audit_layout(fig)
        assert not any(level in ['FAIL', 'WARN'] for level, _ in issues), (name, issues)
        checks[name] = {'issues': issues, 'preview': file_hash(preview), 'grayscale': file_hash(gray)}
        if export:
            outputs = export_figure(fig, str(out / name), formats=['png', 'svg'], dpi=300,
                                    size_inches=tuple(fig.get_size_inches()), tight=False)
            for path in outputs:
                problems, _ = check_figure(path, min_dpi=300, target_inches=tuple(fig.get_size_inches()))
                assert not any(level in ['FAIL', 'WARN'] for level, _ in problems), (path, problems)
                hashes[Path(path).relative_to(ROOT).as_posix()] = file_hash(path)
            figures[name] = (out / f'{name}.png').relative_to(ROOT).as_posix()
            captions[name] = caption
        plt.close(fig)

    summaries = {r['run']: read_json(ROOT / r['run'] / 'summary.json') for r in rows}
    plot_values, curve_values = [], []

    def comparison(ax, method, value, ylabel, metric_name=None, auxiliary=False):
        values = np.array([[value(next(r for r in rows if r['method'] == method and r['vector_size'] == a
                                      and r['model_seed'] == s)) for a in dimensions] for s in seeds])
        assert np.isfinite(values).all()
        for i, seed in enumerate(seeds):
            ax.plot(dimensions, values[i], color=SEED_COLORS[i], marker=MARKERS[i], linestyle=STYLES[i],
                    markersize=5, markerfacecolor='none', markeredgewidth=1,
                    linewidth=.8, alpha=.85, label=f'种子 {seed}', zorder=2)
        mean, sd = values.mean(axis=0), values.std(axis=0, ddof=1)
        ax.errorbar(dimensions, mean, yerr=sd, color='#222222', marker='D', markersize=5,
                    linewidth=1.5, capsize=4, label='均值 ± 样本 SD（n=3）', zorder=4)
        ax.set_xscale('linear')
        ax.set_xticks(dimensions, [f'{a:g}' for a in dimensions])
        ax.minorticks_off()
        ax.set(xlim=(dimensions[0]/1.18, dimensions[-1]*1.18), ylim=(-1 if min(0, (mean-sd).min(), values.min()) < 0 else 0, 1),
               xlabel='向量维数 d', ylabel=ylabel)
        if auxiliary:
            low, high = min(values.min(), (mean-sd).min()), max(values.max(), (mean+sd).max())
            margin = max(.005, float(high-low)*.12)
            ax.set_ylim(min(0, float(low)-margin) if low < 0 else 0, max(.025, float(high)+margin))
        ax.grid(axis='y', alpha=.25)
        for a, avg, dev in zip(dimensions, mean, sd, strict=True):
            plot_values.append({'method': method, 'metric': metric_name or ylabel, 'vector_size': a,
                                'mean': float(avg), 'sample_sd': float(dev), 'n': 3})

    diagnostic = [('wordsim240', 'pearson_r', 'WordSim240 · Pearson r'),
                  ('wordsim297', 'spearman_rho', 'WordSim297 · Spearman ρ'),
                  ('wordsim296', 'spearman_rho', 'WordSim296 · Spearman ρ'),
                  ('CA_translated', 'all_accuracy', '中文类比 · 全题正确率'),
                  ('CA8_morphological', 'all_accuracy', 'CA8 形态 · 全题正确率'),
                  ('CA8_semantic', 'all_accuracy', 'CA8 语义 · 全题正确率')]
    for method in grid['training_methods']:
        title = NAMES[method]
        fig, ax = plt.subplots(figsize=(7.2, 4.5), layout='constrained')
        comparison(ax, method, lambda r: r['validation_score'], 'WordSim240 验证 Spearman ρ')
        ax.set_title(f'{title} · 五档向量维数 · 各三种子', loc='left')
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
        finish(fig, f'dimension_effect_{method}', f'{title} 的五档向量维数比较；各档全部三种子实测点，'
               '黑线为均值±样本 SD（n=3），每次运行采用独立选择的最佳轮次。横轴为实际维数，连线为已测点的读图引导。')
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.8), layout='constrained')
        for ax, seed in zip(axes, seeds, strict=True):
            vals = []
            for i, dimension in enumerate(dimensions):
                row = next(r for r in rows if r['method'] == method and r['vector_size'] == dimension and r['model_seed'] == seed)
                epochs = read_json(ROOT / row['run'] / 'epochs.json')
                x, y = [e['epoch'] for e in epochs], [e['primary_score'] for e in epochs]
                vals.extend(y)
                ax.plot(x, y, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i],
                        markersize=3, linewidth=1.2, label=f'd={dimension:g}')
                ax.scatter(row['selected_epoch'], row['validation_score'], marker='*', s=80,
                           facecolor=COLORS[i], edgecolor='#333333', linewidth=.4, zorder=5)
                curve_values.extend({'method': method, 'vector_size': dimension, 'seed': seed, 'epoch': e['epoch'],
                                     'validation_spearman': e['primary_score'], 'selected': e['epoch'] == row['selected_epoch']}
                                    for e in epochs)
            ax.set(title=f'{title} · 种子 {seed}', xlabel='训练轮次', ylabel='WordSim240 验证 Spearman ρ',
                   xlim=(.5, 15.5), ylim=(-1 if min(vals) < 0 else 0, 1))
            ax.set_xticks([1, 3, 5, 10, 15])
            ax.grid(axis='y', alpha=.25)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=5, frameon=False)
        finish(fig, f'dimension_curves_{method}', f'{title} 五档向量维数的逐轮验证曲线；每个面板为同一训练种子，'
               '五条曲线共用坐标；星号为各自最佳检查点。保留原始波动和早停，不平滑、不补造未训练轮次。')
        fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.3), layout='constrained')
        for ax, (benchmark, key, label) in zip(axes.flat, diagnostic, strict=True):
            def metric(row):
                return next(m[key] for m in summaries[row['run']]['validation_metrics'] if m['benchmark'] == benchmark)
            comparison(ax, method, metric, '验证指标', f'{benchmark}/{key}', auxiliary=True)
            ax.set_title(label, loc='left')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle(f'{title} · 向量维数辅助诊断', fontsize=11)
        finish(fig, f'dimension_diagnostics_{method}', f'{title} 的五档向量维数辅助验证指标；每档三个实测点和均值±样本 SD（n=3），'
               '采用各自最佳主指标轮次，各指标纵轴范围分别标注。类比为每套固定 128 题验证探针，'
               '词表未覆盖题计错。连线为已测维数间的读图引导。')
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.8), layout='constrained')
        cost_metrics = [
            ('fit_seconds', lambda r: r['fit_seconds'], '累计训练时间（秒）'),
            ('per_epoch_seconds', lambda r: r['fit_seconds']/r['trained_epochs'], '平均每轮训练时间（秒）'),
            ('peak_rss_mib', lambda r: r['peak_rss_mib'], '进程峰值内存（MiB）'),
            ('model_mib', lambda r: r['model_bytes']/2**20, '原生模型大小（MiB）'),
        ]
        for ax, (key, value, label), letter in zip(axes.flat, cost_metrics, 'abcd', strict=True):
            comparison(ax, method, value, label, key, auxiliary=True)
            ax.set_title(f'{letter}  {label}', loc='left')
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
        fig.suptitle(f'{title} · 向量维数与计算成本', fontsize=11)
        finish(fig, f'dimension_cost_{method}', f'{title} 的维数与计算成本比较；各档全部三种子实测点及均值±样本 SD（n=3）。'
               '累计训练时间受实际轮数影响，训练时间不含建词表、评价和保存；每轮平均时间为实际训练耗时除以实际轮数。'
               '峰值为进程 RSS，模型大小为最佳检查点的原生模型文件大小。连线仅为已测维数间的读图引导。')
    save_csv(qa / 'aggregated_plot_data.csv', plot_values)
    save_csv(qa / 'epoch_plot_data.csv', curve_values)
    write_json(qa / 'layout_checks.json', {'status': 'passed', 'checks': checks, 'data_fingerprint': data_fingerprint,
               'generator_sha256': file_hash(Path(__file__)), 'checked': stamp()})
    if export:
        write_json(qa / 'approved.json', {'data_fingerprint': data_fingerprint, 'files': hashes, 'figures': figures,
                   'captions': captions, 'review_sha256': file_hash(qa / 'visual_review.json'),
                   'generator_sha256': file_hash(Path(__file__)), 'generated': stamp()})
    print(qa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    build(ROOT / read_json(BASE / 'latest.json')['experiment'], args.export)
