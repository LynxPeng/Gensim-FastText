"""Native validation curves and observed configuration/seed comparisons."""
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
from .thucnews_ablation_figures import ORDER, SHORT, NAMES, COLORS, SEEDS, SEED_COLORS, limits, seed_points, seed_legend
from .thucnews_ablation_report import LABELS


def build(experiment, export=False):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    deferred = plan.get('learning_rate_search_deferred', False)
    assert plan['status'] == 'complete' and len(plan['runs']) == (20 if deferred else 28)
    rows, cfg = plan['runs'], plan['configuration']
    matrix = [r for r in rows if r['search_phase'] == 'matrix_screen']
    qa = experiment / 'figures'
    qa.mkdir(parents=True, exist_ok=True)
    out = ROOT / 'reports/figures/thucnews_small_search' / experiment.name
    data_fingerprint = fingerprint(rows)
    columns = ['method', 'combination', 'alpha', 'model_seed', 'validation_score', 'selected_epoch',
               'trained_epochs', 'fit_seconds', 'peak_rss_mib', 'model_bytes']
    save_csv(qa / 'plot_data.csv', [{k: row[k] for k in columns} for row in rows])
    profile = profile_data(str(qa / 'plot_data.csv'), group_cols=['method', 'combination', 'alpha'])
    (qa / '数据剖析与图型选择.md').write_text('# 小规模搜索图型选择\n\n' + render_report(profile) +
        '\n\n## 图型与证据范围\n\n训练方式是无序类别，用 2×3 热力图；备选为数值表，不连接类别。'
        '轮次有序，用完整原始验证折线并标记检查点，不平滑；备选为逐轮数值表。'
        '三种子展示全部实测点及均值±样本 SD（ddof=1），学习率复验用细线连接相同种子的两次运行，'
        '表示固定种子对应关系，不表示学习率之间的插值。不做均值柱、显著性标记或密度估计。'
        '成本用直接测量的横条，区分总训练和每轮训练时间。相关系数理论范围为 −1 至 1，'
        '热力图对所有模型使用共同色阶；折线坐标按观测范围设置并明确刻度。'
        '学习率暂缓时不生成其比较图。只描述验证数据，未计算测试指标。'
        '彩色/灰度审查后导出 PNG 300 DPI 和 SVG。\n', encoding='utf-8')
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

    def curve(ax, row, title):
        epochs = read_json(ROOT / row['run'] / 'epochs.json')
        x, rho, pearson = [], [], []
        for epoch in epochs:
            metric = next(m for m in epoch['metrics'] if m['benchmark'] == 'wordsim240')
            x.append(epoch['epoch'])
            rho.append(metric['spearman_rho'])
            pearson.append(metric['pearson_r'])
        ax.plot(x, rho, color=COLORS[row['method']], marker='o', markersize=3, label='Spearman ρ')
        ax.plot(x, pearson, color='#666666', linestyle='--', marker='s', markersize=3, label='Pearson r')
        ax.axvline(row['selected_epoch'], color='#999999', linestyle=':', linewidth=1)
        limits(ax, rho + pearson)
        ax.set(xlim=(.5, cfg['validation']['max_epochs'] + .5), xlabel='训练轮次', ylabel='验证相关系数')
        ax.set_xticks([1, 3, 5, 10, 15])
        ax.set_title(title + f' · 选第 {row["selected_epoch"]} 轮', loc='left')

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), layout='constrained')
    low = -1 if any(r['validation_score'] < 0 for r in matrix) else 0
    for ax, method, letter in zip(axes, cfg['models'], 'ab', strict=True):
        values = np.array([next(r['validation_score'] for r in matrix if r['method'] == method and r['combination'] == c)
                           for c in ORDER]).reshape(2, 3)
        im = ax.pcolormesh(np.arange(4)-.5, np.arange(3)-.5, values, cmap='viridis', vmin=low, vmax=1)
        ax.set(xlim=(-.5, 2.5), ylim=(1.5, -.5))
        ax.set_xticks([0, 1, 2], ['HS', 'NS5', 'NS10'])
        ax.set_yticks([0, 1], ['CBOW', 'Skip-gram'])
        for i in range(2):
            for j in range(3):
                row = next(r for r in matrix if r['method'] == method and r['combination'] == ORDER[i*3+j])
                ax.text(j, i, f'{values[i,j]:.4f}\n第 {row["selected_epoch"]} 轮', ha='center', va='center',
                        color='white' if im.norm(values[i,j]) < .55 else '#222222', fontsize=10)
        ax.set_title(f'{letter}  {NAMES[method]} · 5 万篇 · n=1', loc='left')
        bar = fig.colorbar(im, ax=ax, label='验证 Spearman ρ', shrink=.8)
        bar.solids.set_rasterized(False)
    finish(fig, 'training_matrix', '固定 5 万篇、种子 42 的训练方式初筛；每配置 n=1，数值为验证 Spearman，附最佳轮次。')

    for method in cfg['models']:
        fig, axes = plt.subplots(2, 3, figsize=(8.5, 4.8), layout='constrained')
        for ax, combination, letter in zip(axes.flat, ORDER, 'abcdef', strict=True):
            row = next(r for r in matrix if r['method'] == method and r['combination'] == combination)
            curve(ax, row, f'{letter}  {SHORT[ORDER.index(combination)]}')
        fig.legend(*axes[0,0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
        finish(fig, f'training_curves_{method}', f'{NAMES[method]} 六种训练方式逐轮验证；虚线为各自最佳检查点，无平滑；未画出的后续轮次未训练。')

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 5.3), layout='constrained')
    measurements = [('fit_seconds', '累计训练时间（分钟）', 60), ('fit_seconds', '平均每轮训练时间（秒）', None)]
    for i, (field, label, divisor) in enumerate(measurements):
        maximum = max(r[field] / (divisor or r['trained_epochs']) for r in matrix)
        for j, method in enumerate(cfg['models']):
            ax = axes[i,j]
            group = [next(r for r in matrix if r['method'] == method and r['combination'] == c) for c in ORDER]
            values = [r[field] / (divisor or r['trained_epochs']) for r in group]
            ax.barh(range(6), values, color=COLORS[method], edgecolor='#333333', hatch='' if j == 0 else '//', height=.6)
            ax.set_yticks(range(6), SHORT)
            ax.invert_yaxis()
            ax.set(xlim=(0, maximum*1.3), xlabel=label)
            ax.set_title(f'{"abcd"[i*2+j]}  {NAMES[method]}', loc='left')
            for k, value in enumerate(values):
                ax.text(value + maximum*.02, k, f'{value:.1f}', va='center', fontsize=8)
            ax.grid(axis='x', alpha=.18)
    finish(fig, 'training_cost', '训练方式初筛的直接实测训练成本；每配置一次运行，累计时间受选轮规则影响，训练时间不含建词表、评价和保存。')

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout='constrained')
    for ax, method, letter in zip(axes, cfg['models'], 'ab', strict=True):
        choices = [cfg['baseline'], plan['matrix_candidates'][method]]
        groups = [[r for r in rows if r['method'] == method and r['combination'] == c
                   and r['alpha'] == cfg['embedding']['alpha']] for c in choices]
        seed_points(ax, groups, 'validation_score')
        ax.set_xticks([0, 1], [LABELS[c].replace('＋', '\n＋') for c in choices])
        ax.set_ylabel('WordSim240 验证 Spearman ρ')
        ax.set_title(f'{letter}  {NAMES[method]}', loc='left')
    fig.legend(handles=seed_legend(), loc='outside lower center', ncol=4, frameon=False)
    finish(fig, 'training_replication', '基线与最佳新训练方式的三种子结果；全部点和均值±样本 SD（n=3），不作显著性判断。')

    for method in ([] if deferred else cfg['models']):
        fig, axes = plt.subplots(1, 3, figsize=(8.5, 3.5), layout='constrained')
        combination = plan['matrix_ranking'][method]['winner']
        for ax, alpha, letter in zip(axes, cfg['learning_rates'], 'abc', strict=True):
            row = next(r for r in rows if r['method'] == method and r['combination'] == combination
                       and r['alpha'] == alpha and r['model_seed'] == cfg['screen_seed'])
            curve(ax, row, f'{letter}  alpha={alpha}')
        fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=2, frameon=False)
        finish(fig, f'alpha_curves_{method}', f'{NAMES[method]} 在 {LABELS[combination]} 下三种学习率的逐轮验证；每配置种子 42、n=1；各自选最佳轮次。')

    if not deferred:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.7), layout='constrained')
        for ax, method, letter in zip(axes, cfg['models'], 'ab', strict=True):
            choices = [cfg['embedding']['alpha'], plan['alpha_candidates'][method]]
            groups = [[r for r in rows if r['method'] == method and r['alpha'] == alpha
                       and r['combination'] == plan['matrix_ranking'][method]['winner']] for alpha in choices]
            for offset, seed, color, style in zip([-.12, 0, .12], SEEDS, SEED_COLORS,
                                                   ['-', '--', ':'], strict=True):
                values = [next(r['validation_score'] for r in group if r['model_seed'] == seed) for group in groups]
                ax.plot([offset, 1 + offset], values, color=color, linestyle=style,
                        linewidth=.8, alpha=.65, zorder=1)
            seed_points(ax, groups, 'validation_score')
            ax.set_xticks([0, 1], [f'alpha={a}' for a in choices])
            ax.set_ylabel('WordSim240 验证 Spearman ρ')
            ax.set_title(f'{letter}  {NAMES[method]}', loc='left')
        fig.legend(handles=seed_legend(), loc='outside lower center', ncol=4, frameon=False)
        finish(fig, 'alpha_replication', '参考学习率与最佳新学习率的三种子比较；细线连接相同种子的两次运行，全部点和均值±样本 SD（n=3）；只比较已复验方案，不作显著性判断。')

    write_json(qa / 'layout_audit.json', {'data_fingerprint': data_fingerprint, 'checks': checks,
               'generator_sha256': file_hash(Path(__file__)), 'generated': stamp()})
    if export:
        write_json(qa / 'approved.json', {'data_fingerprint': data_fingerprint, 'files': hashes,
                   'figures': figures, 'captions': captions, 'review_sha256': file_hash(qa / 'visual_review.json'),
                   'generator_sha256': file_hash(Path(__file__)), 'generated': stamp()})
        from .thucnews_small_search_report import report
        report(experiment)
    print(f'{"Exported" if export else "Previewed"} {len(checks)} figures: {qa}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['preview', 'export'], default='preview')
    args = parser.parse_args()
    experiment = ROOT / read_json(ROOT / 'artifacts/thucnews_small_search/latest.json')['experiment']
    build(experiment, export=args.stage == 'export')
