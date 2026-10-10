"""Factorial matrices, native epoch curves, and individual three-seed results."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from PIL import Image

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .evaluation_support import save_csv
from .figure_support.profile_data import profile_data, render_report
from .figure_support.setup_style import setup_style
from .figure_support.visual_qa import audit_layout, render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure
from .thucnews_ablation_report import LABELS

COLORS = {'word2vec': '#0072B2', 'fasttext': '#D55E00'}
NAMES = {'word2vec': 'Word2Vec', 'fasttext': 'FastText'}
ORDER = ['cbow_hs', 'cbow_ns5', 'cbow_ns10', 'sg_hs', 'sg_ns5', 'sg_ns10']
SHORT = ['CBOW+HS', 'CBOW+NS5', 'CBOW+NS10', 'SG+HS', 'SG+NS5', 'SG+NS10']
SEEDS = [42, 2026, 7]
SEED_MARKERS = ['o', 's', '^']
SEED_COLORS = ['#0072B2', '#D55E00', '#009E73']


def limits(ax, values, percent=False):
    lower = min(-1, min(values) * 1.05) if any(v < 0 for v in values) else 0
    ax.set_ylim(lower, 100 if percent else 1)
    ax.grid(axis='y', alpha=.18)


def seed_points(ax, groups, field, percent=False):
    all_values = []
    for x, group in enumerate(groups):
        values = []
        for offset, seed, marker, color in zip([-.12, 0, .12], SEEDS, SEED_MARKERS, SEED_COLORS, strict=True):
            row = next(r for r in group if r['model_seed'] == seed)
            value = row[field] * (100 if percent else 1)
            values.append(value)
            ax.scatter(x + offset, value, marker=marker, color=color, s=27, zorder=4, clip_on=False)
        mean, sd = np.mean(values), np.std(values, ddof=1)
        ax.errorbar(x, mean, yerr=sd, fmt='D', color='#222222', markersize=4, capsize=4,
                    markerfacecolor='white', zorder=3, clip_on=False)
        all_values.extend(values)
        all_values.extend([mean - sd, mean + sd])
    limits(ax, all_values, percent)
    ax.set_xlim(-.5, 1.5)


def seed_legend():
    return [Line2D([], [], marker=m, linestyle='none', color=c, label=f'种子 {s}', markersize=5)
            for s, m, c in zip(SEEDS, SEED_MARKERS, SEED_COLORS, strict=True)] + [
            Line2D([], [], marker='D', linestyle='none', markerfacecolor='white', color='#222222',
                       label='均值±样本 SD（n=3）', markersize=5)]


def build(experiment, export=False):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    rows, cfg = plan['runs'], plan['configuration']
    screen = [r for r in rows if r['model_seed'] == cfg['screen_seed']]
    if len(screen) != 12:
        raise ValueError('All twelve screening runs must finish before matrix plotting')
    qa = experiment / 'figures'
    qa.mkdir(parents=True, exist_ok=True)
    out = qa / 'exported' if plan['smoke'] else ROOT / 'reports/figures/thucnews_ablation' / experiment.name
    data_fingerprint = fingerprint(rows)
    frame = pd.DataFrame(rows)
    columns = ['method', 'combination', 'model_seed', 'validation_score', 'selected_epoch', 'trained_epochs',
               'fit_seconds', 'peak_rss_mib', 'model_bytes']
    save_csv(qa / 'plot_data.csv', frame[columns].to_dict('records'))
    profile = profile_data(str(qa / 'plot_data.csv'), group_cols=['method', 'combination'])
    (qa / '数据剖析与图型选择.md').write_text(
        '# 消融实验图型选择\n\n' + render_report(profile) +
        '\n\n## 论证目标与图型\n\n初筛每个组合 n=1，用 2×3 数值热力图展示算法和训练目标的组合效果；'
        '备选为完整数值表，避免用折线连接无序配置。逐轮 Spearman/Pearson 使用原始轮次折线，'
        '不插值或平滑。复验 n=3，展示所有种子的点及均值±样本 SD（ddof=1）；'
        '备选为种子明细表，不使用箱线、密度图或只显示均值的柱状图。'
        '成本是每次运行的直接测量，用横条明确单位，累计时间和平均每轮时间分开。'
        '所有相关系数坐标保持完整 0–1（出现负数扩展至 -1），正确率 0–100%。'
        '无显著性标记、无双 Y 轴；PNG 300 DPI，SVG 矢量；彩色和灰度分别核验。\n', encoding='utf-8')
    setup_style(journal='general', lang='zh')
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'xtick.labelsize': 8,
                         'ytick.labelsize': 8, 'legend.fontsize': 8, 'svg.fonttype': 'none'})
    reviews = read_json(qa / 'visual_review.json') if export else None
    if export:
        assert reviews['status'] == 'passed' and reviews['data_fingerprint'] == data_fingerprint
        assert reviews['generator_sha256'] == file_hash(Path(__file__))
    checks, figures, captions, hashes = {}, {}, {}, {}

    def finish(fig, name, caption):
        preview = qa / f'{name}_preview.png'
        gray = qa / f'{name}_grayscale.png'
        if export:
            for path in [preview, gray]:
                assert file_hash(path) == reviews['previews'][path.name]
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
                problems, info = check_figure(path, min_dpi=300, target_inches=tuple(fig.get_size_inches()))
                assert not any(level in ['FAIL', 'WARN'] for level, _ in problems), (path, problems)
                hashes[Path(path).relative_to(ROOT).as_posix()] = file_hash(path)
            figures[name] = (out / f'{name}.png').relative_to(ROOT).as_posix()
            captions[name] = caption
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), layout='constrained')
    values = [r['validation_score'] for r in screen]
    low = -1 if min(values) < 0 else 0
    for ax, method, letter in zip(axes, cfg['models'], 'ab', strict=True):
        matrix = np.array([next(r['validation_score'] for r in screen if r['method'] == method and r['combination'] == c)
                           for c in ORDER]).reshape(2, 3)
        image = ax.pcolormesh(np.arange(4)-.5, np.arange(3)-.5, matrix,
                              cmap='viridis', vmin=low, vmax=1, shading='flat', rasterized=False)
        ax.set_xlim(-.5, 2.5)
        ax.set_ylim(1.5, -.5)
        for i in range(2):
            for j in range(3):
                row = next(r for r in screen if r['method'] == method and r['combination'] == ORDER[i*3+j])
                ax.text(j, i, f'{row["validation_score"]:.4f}\n第 {row["selected_epoch"]} 轮',
                        ha='center', va='center', color='white' if matrix[i, j] < .55 else '#222222', fontsize=10)
        ax.set_xticks([0, 1, 2], ['HS', 'NS5', 'NS10'])
        ax.set_yticks([0, 1], ['CBOW', 'Skip-gram'])
        ax.set_title(f'{letter}  {NAMES[method]} · 初筛 n=1', loc='left')
        colorbar = fig.colorbar(image, ax=ax, label='验证 Spearman ρ', shrink=.8)
        colorbar.solids.set_rasterized(False)
    finish(fig, 'screen_matrix', '种子 42 的训练方式矩阵；每组合 n=1，数值为 WordSim240 验证 Spearman，附所选轮次。')

    for method in cfg['models']:
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.8), layout='constrained')
        model_values = [m[key] for r in screen if r['method'] == method
                        for epoch in read_json(ROOT / r['run'] / 'epochs.json')
                        for m in epoch['metrics'] if m['benchmark'] == 'wordsim240'
                        for key in ['spearman_rho', 'pearson_r']]
        for ax, combination, letter in zip(axes.flat, ORDER, 'abcdef', strict=True):
            row = next(r for r in screen if r['method'] == method and r['combination'] == combination)
            epochs = read_json(ROOT / row['run'] / 'epochs.json')
            numbers = [r['epoch'] for r in epochs]
            rho, pearson = [], []
            for epoch in epochs:
                metric = next(m for m in epoch['metrics'] if m['benchmark'] == 'wordsim240')
                rho.append(metric['spearman_rho'])
                pearson.append(metric['pearson_r'])
            ax.plot(numbers, rho, color=COLORS[method], marker='o', markersize=3, label='Spearman ρ')
            ax.plot(numbers, pearson, color='#666666', marker='s', markersize=3, linestyle='--', label='Pearson r')
            ax.axvline(row['selected_epoch'], color='#888888', linestyle=':', linewidth=1)
            limits(ax, model_values)
            ax.set(xlim=(.5, cfg['validation']['max_epochs'] + .5), xlabel='训练轮次', ylabel='验证相关系数')
            ax.set_xticks([1, 5, 10, 15] if not plan['smoke'] else [1, 2, 3])
            ax.set_title(f'{letter}  {SHORT[ORDER.index(combination)]} · 选第 {row["selected_epoch"]} 轮', loc='left')
        fig.legend(handles=axes[0, 0].get_legend_handles_labels()[0], labels=['Spearman ρ', 'Pearson r'],
                   loc='outside lower center', ncol=2, frameon=False)
        finish(fig, f'curves_{method}', f'{NAMES[method]} 六组初筛逐轮验证；虚线标记各配置最佳轮次；原始轮次，无平滑，无重复实验误差带。')

    for name, measurements in [('screen_cost', [('fit_seconds', '累计训练时间（分钟）', 60),
                                               ('fit_seconds', '平均每轮训练时间（秒）', None)]),
                               ('screen_storage', [('peak_rss_mib', '主进程峰值 RSS（MiB）', 1),
                                                   ('model_bytes', '所选原生模型（MiB）', 2**20)])]:
        fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.5), layout='constrained')
        maxima = [max(r[field] / (divisor if divisor else r['trained_epochs']) for r in screen)
                  for field, _, divisor in measurements]
        for col, method in enumerate(cfg['models']):
            group = [next(r for r in screen if r['method'] == method and r['combination'] == c) for c in ORDER]
            for row_index, (field, label, divisor) in enumerate(measurements):
                ax = axes[row_index, col]
                values = [r[field] / (divisor if divisor else r['trained_epochs']) for r in group]
                ax.barh(range(6), values, color=COLORS[method], edgecolor='#333333', hatch='' if method == 'word2vec' else '//', height=.6)
                ax.set_yticks(range(6), SHORT)
                ax.invert_yaxis()
                ax.set(xlabel=label, xlim=(0, maxima[row_index] * 1.3))
                for i, value in enumerate(values):
                    ax.text(value + maxima[row_index]*.02, i, f'{value:.1f}' if value >= .05 else f'{value:.4f}', va='center', fontsize=8)
                ax.set_title(f'{"abcd"[row_index*2+col]}  {NAMES[method]}', loc='left')
                ax.grid(axis='x', alpha=.18)
        finish(fig, name, '六组初筛的直接实测成本；每组合一次运行。训练耗时不含建词表、验证和检查点保存，累计耗时受停止轮次影响。'
               if name == 'screen_cost' else '六组初筛的主进程峰值内存与所选原生模型大小；模型大小不含所有轮次检查点总占用。')

    frozen_path = experiment / 'frozen.json'
    if frozen_path.exists():
        frozen = read_json(frozen_path)
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout='constrained')
        for ax, method, letter in zip(axes, cfg['models'], 'ab', strict=True):
            choices = [cfg['baseline'], plan['candidates'][method]]
            groups = [[r for r in rows if r['method'] == method and r['combination'] == c] for c in choices]
            seed_points(ax, groups, 'validation_score')
            ax.set_xticks([0, 1], [LABELS[c].replace('＋', '\n＋') for c in choices])
            ax.set_ylabel('WordSim240 验证 Spearman ρ')
            ax.set_title(f'{letter}  {NAMES[method]}', loc='left')
        fig.legend(handles=seed_legend(), loc='outside lower center', ncol=4, frameon=False)
        finish(fig, 'replication', '基线与最佳新配置的三种子验证结果；每个点为独立训练，空心菱形与误差棒为均值±样本 SD，n=3；不作显著性判断。')

        for method in cfg['models']:
            tests = [read_json(p) for p in (experiment / 'tests').glob('*/summary.json')]
            tests = [r for r in tests if r['method'] == method]
            if len(tests) != 6:
                continue
            choices = [cfg['baseline'], plan['candidates'][method]]
            fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.5), layout='constrained')
            benchmarks = ['wordsim240', 'wordsim297', 'CA_translated', 'CA8_morphological', 'CA8_semantic']
            for ax, benchmark, letter in zip(axes.flat, benchmarks, 'abcde'):
                groups = []
                for combination in choices:
                    group = []
                    for test in tests:
                        if test['combination'] == combination:
                            metric = next(m for m in test['metrics'] if m['benchmark'] == benchmark)
                            if benchmark == 'wordsim297':
                                other = next(m for m in test['metrics'] if m['benchmark'] == 'wordsim296')
                                assert metric['spearman_rho'] == other['spearman_rho']
                            field = 'spearman_rho' if metric['kind'] == 'similarity' else 'all_accuracy'
                            group.append({'model_seed': test['model_seed'], 'value': metric[field]})
                    groups.append(group)
                is_analogy = benchmark.startswith('CA')
                seed_points(ax, groups, 'value', percent=is_analogy)
                ax.set_xticks([0, 1], [SHORT[ORDER.index(c)].replace('+', '\n+') for c in choices])
                ax.set_ylabel('全部测试题正确率（%）' if is_analogy else '测试 Spearman ρ')
                label = {'wordsim240': 'WordSim240', 'wordsim297': 'WordSim297/296', 'CA_translated': 'CA translated',
                         'CA8_morphological': 'CA8 形态', 'CA8_semantic': 'CA8 语义'}[benchmark]
                ax.set_title(f'{letter}  {label}', loc='left')
            axes[1, 2].axis('off')
            axes[1, 2].legend(handles=seed_legend(), loc='center', frameon=False)
            fig.suptitle(f'{NAMES[method]} · 冻结后的统一测试', fontsize=11)
            finish(fig, f'test_{method}', f'{NAMES[method]} 冻结后测试；每方案三个最佳检查点，全部实测点与均值±样本 SD。WordSim297/296 本次测试题及分数一致，合并展示；类比 OOV 按错计入。')

    write_json(qa / 'layout_audit.json', {'data_fingerprint': data_fingerprint, 'checks': checks,
               'generator_sha256': file_hash(Path(__file__)), 'generated': stamp()})
    if export:
        write_json(qa / 'approved.json', {'data_fingerprint': data_fingerprint, 'files': hashes,
                   'figures': figures, 'captions': captions, 'review_sha256': file_hash(qa / 'visual_review.json'),
                   'generator_sha256': file_hash(Path(__file__)), 'generated': stamp()})
        if not plan['smoke']:
            from .thucnews_ablation_report import report
            report(experiment)
    print(f'{"Exported" if export else "Previewed"} {len(checks)} figures: {qa}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preview', 'export'], default='preview')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    index = ROOT / 'artifacts/thucnews_ablation' / ('smoke_latest.json' if args.smoke else 'latest.json')
    build(ROOT / read_json(index)['experiment'], export=args.stage == 'export')


if __name__ == '__main__':
    main()
