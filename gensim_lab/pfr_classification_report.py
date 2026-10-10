"""Scientific figures and scoped section 1.2 publication from sealed PFR results."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .common import ROOT, file_hash, markdown_table, read_json, stamp, write_json
from .data import load_docs
from .evaluation_support import save_csv
from .figure_support.setup_style import setup_style
from .figure_support.visual_qa import audit_layout, render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure
from .figure_support.profile_data import profile_data, render_report
from .pfr_frozen_classification import BASE, METHODS, labels_for

NAMES = {'tfidf': 'TF-IDF', 'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText',
         'native_scratch': '原生 FastText：随机初始化', 'native_pretrained': '原生 FastText：预训练初始化'}
COLORS = ['#0072B2', '#E69F00', '#009E73']
FIGURES = ['performance', 'page_f1', 'confusion', 'future', 'softmax_validation']
REPORTS = [ROOT / 'reports/Gensim&FastText-修改版.md', ROOT / 'reports/Gensim&FastText.md']


def load():
    directory = ROOT / read_json(BASE / 'latest.json')['experiment']
    assert read_json(directory / 'index.json')['status'] == 'complete'
    assert read_json(directory / 'verification.json')['status'] == 'passed'
    for name, digest in read_json(directory / 'test_seal.json').items():
        assert file_hash(directory / name) == digest
    plan = read_json(directory / 'plan.json')
    df = pd.read_csv(directory / 'results.csv')
    assert np.isfinite(df[['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']].to_numpy()).all()
    return directory, plan, df


def selected_row(df, profile, method, head='svm', seed=42, part='test'):
    rows = df[(df.profile == profile) & (df.method == method) & (df['head'] == head)
              & (df.seed == seed) & (df.partition == part)]
    assert len(rows) == 1
    return rows.iloc[0]


def figures(directory, plan, df):
    setup_style(journal='general', lang='zh', use_sciplots=False)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5})
    result = {}
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), constrained_layout=True)
    for row, profile in enumerate(['january_12', 'temporal_12']):
        boot = read_json(directory / f'bootstrap_{profile}.json')
        for col, metric in enumerate(['macro_f1', 'accuracy']):
            ax = axes[row, col]
            for i, method in enumerate(METHODS):
                rec = selected_row(df, profile, method)
                value = 100 * rec[metric]
                error = None
                label_x = value + 3
                if metric == 'macro_f1':
                    low, high = np.asarray(boot['intervals'][method]) * 100
                    error = np.array([[max(0., value - low)], [max(0., high - value)]])
                    label_x = max(label_x, high + 2)
                ax.barh(i, value, color=COLORS[i], height=.56, xerr=error,
                        error_kw={'capsize': 3, 'elinewidth': 1, 'ecolor': '#222222'})
                ax.text(min(label_x, 96), i, f'{value:.2f}', va='center', fontsize=8)
            ax.set_yticks(range(3), [NAMES[m].replace('Gensim ', '') for m in METHODS])
            ax.invert_yaxis(); ax.set_xlim(0, 105)
            ax.set_xlabel(('Macro-F1' if col == 0 else 'Accuracy') + ' (%)')
            ax.set_title(('一月测试（n=456）' if row == 0 else '六月测试（n=3191）'))
            ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    result['performance'] = fig

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), constrained_layout=True)
    for ax, profile in zip(axes, ['january_12', 'temporal_12']):
        details = []
        for method in METHODS:
            row = selected_row(df, profile, method)
            details.append(read_json(directory / 'runs' / row.run / 'metrics_test.json')['per_class'])
        image = np.array([[details[j][page]['f1-score'] for j in range(3)] for page in labels_for(12)])
        im = ax.pcolormesh(np.arange(4) - .5, np.arange(13) - .5, image,
                           cmap='cividis', vmin=0, vmax=1, shading='flat')
        ax.set_xlim(-.5, 2.5); ax.set_ylim(11.5, -.5)
        for i in range(12):
            for j in range(3):
                ax.text(j, i, f'{image[i,j]:.2f}', ha='center', va='center',
                        color='white' if image[i,j] < .5 else 'black', fontsize=7)
        ax.set_xticks(range(3), ['TF-IDF', 'Word2Vec', 'FastText'])
        ax.set_yticks(range(12), [f'{p} (n={int(details[0][p]["support"])})' for p in labels_for(12)])
        ax.set_xlabel('固定表示 + SVM'); ax.set_ylabel('真实版面（测试文章数）')
        ax.set_title('一月测试' if profile == 'january_12' else '六月测试')
    cb = fig.colorbar(im, ax=axes, label='各版面 F1', shrink=.8)
    cb.solids.set_rasterized(False)
    result['page_f1'] = fig

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 6.3), constrained_layout=True)
    entries = [(m, 'svm') for m in METHODS] + [('native_pretrained', 'native')]
    for ax, (method, head) in zip(axes.ravel(), entries):
        row = selected_row(df, 'temporal_12', method, head)
        matrix = np.asarray(read_json(directory / 'runs' / row.run / 'metrics_test.json')['confusion_matrix'])
        rate = matrix / matrix.sum(axis=1, keepdims=True)
        im = ax.pcolormesh(np.arange(13) - .5, np.arange(13) - .5, rate,
                           cmap='cividis', vmin=0, vmax=1, shading='flat')
        ax.set_xlim(-.5, 11.5); ax.set_ylim(11.5, -.5); ax.set_aspect('equal')
        ax.set_xticks(range(12), labels_for(12)); ax.set_yticks(range(12), labels_for(12))
        ax.tick_params(labelsize=6.5)
        ax.set_xlabel('预测版面'); ax.set_ylabel('真实版面')
        ax.set_title(NAMES[method].replace('Gensim ', ''))
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label='按真实版面归一化比例', shrink=.8)
    cb.solids.set_rasterized(False)
    result['confusion'] = fig

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), constrained_layout=True)
    months = [f'1998{i:02d}' for i in range(2, 7)]
    for ax, metric in zip(axes, ['macro_f1', 'accuracy']):
        for method, color, marker, style in zip(METHODS, COLORS, ['o', 's', '^'], ['-', '--', '-.']):
            values = [selected_row(df, 'january_12', method, part='future_' + month)[metric] * 100 for month in months]
            ax.plot(range(2, 7), values, marker=marker, linestyle=style, color=color, label=NAMES[method].replace('Gensim ', ''))
        ax.set_xticks(range(2, 7), [f'{i}月' for i in range(2, 7)])
        ax.set_xlabel('目标月份（一月模型保持冻结）'); ax.set_ylabel(metric.replace('macro_f1', 'Macro-F1').replace('accuracy', 'Accuracy') + ' (%)')
        ax.set_ylim(0, 100); ax.grid(alpha=.2)
        ax.legend(loc='lower left', frameon=False)
    result['future'] = fig

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), constrained_layout=True)
    index = read_json(directory / 'index.json')
    for ax, profile in zip(axes, ['january_12', 'temporal_12']):
        for method, color, marker, style in zip(METHODS, COLORS, ['o', 's', '^'], ['-', '--', '-.']):
            rec = index['selection'][f'{profile}/{method}/softmax/42']
            folder = directory / 'runs' / rec['run']
            epochs = pd.read_csv(folder / 'epochs.csv')
            summary = read_json(folder / 'summary.json')
            ax.plot(epochs.epoch, epochs.validation_macro_f1 * 100, color=color, linestyle=style,
                    label=f'{NAMES[method].replace("Gensim ", "")} (lr={rec["parameter"]})')
            chosen = summary['selected_epoch']
            value = epochs.loc[epochs.epoch == chosen, 'validation_macro_f1'].iloc[0] * 100
            ax.scatter([chosen], [value], color=color, marker=marker, s=28, zorder=5)
        ax.set_xlabel('分类器训练轮次'); ax.set_ylabel('验证 Macro-F1 (%)'); ax.set_ylim(0, 100)
        ax.set_title('一月分类器' if profile == 'january_12' else '一月至四月分类器')
        ax.grid(alpha=.2); ax.legend(loc='lower right', frameon=False)
    result['softmax_validation'] = fig
    return result


def preview(directory, plan, df):
    out = directory / 'figure_preview'
    out.mkdir(exist_ok=True)
    profile = profile_data(df, group_cols=['profile', 'method', 'head', 'partition'])
    write_json(out / 'data_profile.json', json.loads(json.dumps(profile, default=str)))
    (out / '数据剖析与图型选择.md').write_text(render_report(profile) + '\n\n'
        '本轮比较固定分类系统的文章级聚合指标，不把单次模型训练当作大量独立重复。'
        '主要性能用水平条形图加文章分层配对 bootstrap 区间；各版面 F1 和混淆矩阵用统一色阶热力图；'
        '月份与真实训练轮次有顺序，使用折线。Softmax、原生模型的三种子波动另入表，不混入文章 bootstrap 区间。'
        '所有准确率／F1 轴从 0 起，不显示显著性星号。\n', encoding='utf-8')
    audits = {}
    for name, fig in figures(directory, plan, df).items():
        path = out / f'{name}.png'
        render_preview(fig, str(path), dpi=150)
        issues = audit_layout(fig)
        ImageOps.grayscale(Image.open(path)).convert('RGB').save(out / f'{name}_grayscale.png')
        audits[name] = {'issues': issues, 'preview_sha256': file_hash(path),
                        'grayscale_sha256': file_hash(out / f'{name}_grayscale.png')}
        plt.close(fig)
    write_json(out / 'layout_audit.json', audits)
    print(json.dumps(audits, ensure_ascii=False, indent=2))


def export(directory, plan, df):
    review = read_json(directory / 'figure_preview/visual_review.json')
    assert review['status'] == 'approved'
    for name in FIGURES:
        assert review['previews'][name]['sha256'] == file_hash(directory / f'figure_preview/{name}.png')
        assert review['previews'][name]['grayscale_sha256'] == file_hash(directory / f'figure_preview/{name}_grayscale.png')
    out = ROOT / 'reports/figures/pfr_frozen_classification' / directory.name
    out.mkdir(parents=True, exist_ok=True)
    audits = {}
    for name, fig in figures(directory, plan, df).items():
        files = export_figure(fig, str(out / name), formats=['png', 'svg'], dpi=300, grayscale_preview=True)
        for path in files:
            if path.endswith('_grayscale.png'):
                with Image.open(path) as image:
                    image.save(path, dpi=(300, 300))
        audits[name] = {Path(path).name: {'issues': check_figure(path)[0], 'sha256': file_hash(path)} for path in files}
        assert not any(issue[0] in ['WARN', 'FAIL'] for rec in audits[name].values() for issue in rec['issues'])
        plt.close(fig)
    write_json(directory / 'figure_export_audit.json', audits)


def percentage(value):
    return f'{100 * value:.2f}'


def figure(name, caption, directory):
    path = f'figures/pfr_frozen_classification/{directory.name}/{name}.png'
    assert (ROOT / 'reports' / path).exists()
    return f'![{caption}]({path})\n\n{caption}\n'


def manuscript(directory, plan, df):
    manifest = read_json(ROOT / 'data/processed/manifest.json')
    p = ['### 1.2 文本分类实验', '', '#### 1.2.1 数据集与评价划分（9801–9806）', '',
         '分类实验使用人民日报 1998 年一月至六月语料（9801–9806），共 18,647 篇原始文章，清洗后保留 18,108 篇。'
         '标签为 01–12 的版面编号，任务称为版面分类。THUCNews 仅用于此前的表示学习，本节没有使用其新闻类别训练分类器。', '',
         '文章按日期、版面和文章序号聚合，标题与正文共同进入特征。沿用原有分词，去除编号、词性后缀、实体标注括号和标点，'
         '进行 NFKC 规范化并清除控制字符；不同内容的重复段落编号按原顺序保留。少于 30 个有效词的文章排除；同文本同标签只保留最早一篇，'
         '标签冲突的相同文本全部排除。字符三元组 SimHash 汉明距离不超过 3 的文章归入同组。日期、编号和版面 ID 不进入文本表示。', '',
         markdown_table(['月份', '原始文章', '保留文章'], [[m['month'], m['raw_articles'], m['kept_articles']] for m in manifest['months']]), '',
         '一月实验复用既有种子 42 的分层、近重复组划分，先划分 12 类，再筛选较小的版面范围。'
         '时间扩展用一月至四月训练、五月验证、六月测试，并排除跨分区近重复组。两种设置的分类器均只读取相应训练标签，验证集选参，测试集用于最后评价。', '',
         markdown_table(['设置', '训练', '验证', '测试'], [[name.replace('january_', '一月 ').replace('temporal_12', '一至四月→六月'),
                       *[len(rec['splits'][s]) for s in ['train', 'validation', 'test']]] for name, rec in plan['profiles'].items()]), '',
         '#### 1.2.2 表示方式、分类器与参数选择', '',
         '主要实验将现有三套表示保持冻结，分别训练相同的 LinearSVC。TF-IDF 使用 THUCNews 的 50,000 词词典和原 IDF；'
         'Word2Vec 使用 Skip-gram＋NS5、300 维的最佳检查点；Gensim FastText 使用 Skip-gram＋NS10、100 维的最佳检查点。'
         '词向量按词次取均值后做 L2 归一化，Word2Vec 跳过未登录词，FastText 纳入能够生成的子词向量；TF-IDF 原始权重也做 L2 归一化。'
         '本节比较的是三套现成表示，维数、容量和预训练目标并未统一，因此不将差异归因于架构本身。', '',
         'SVM 和多项逻辑回归采用类别平衡权重，分别在 C=0.1、1、10 中选择验证 Macro-F1 最高的配置；并列时保留较小 C。'
         '自行实现的线性 Softmax 使用同一固定输入表示，只训练输出权重和偏置，采用 CUDA、AdamW、批次 256、权重衰减 0.0001。'
         '种子 42 比较学习率 0.001、0.01、0.05，固定最佳学习率后补充 2026、7；每次独立按逐轮验证结果选检查点。'
         '至少训练 10 轮，最多 60 轮，连续 8 轮未提高超过 0.001 时停止。训练过程不记录 loss，完整验证曲线另存 CSV。', '',
         '原生监督 FastText 作为辅助对照，比较随机初始化与现有 Gensim FastText 向量初始化。只导出对应人民日报训练词表中的向量，'
         '包括由子词生成的训练词向量；没有导入 Gensim 的字符哈希桶。两者均为 100 维、词 bigram、200,000 桶、平坦 softmax、'
         '学习率 0.1、最低词频 1，关闭原生字符子词，使用 20 个 CPU 线程。种子 42 在 5、15、25 轮中按验证 Macro-F1 选择，'
         '冻结轮数后补充另外两个种子。标签训练同时更新输入词向量与分类输出，因而与固定表示接 SVM 的比较口径不同。', '',
         '#### 1.2.3 三个表示模型的分类效果', '']
    winners = {}
    rows = []
    for profile in ['january_12', 'temporal_12']:
        candidates = [selected_row(df, profile, m) for m in METHODS]
        best = max(candidates, key=lambda r: r.macro_f1)
        winners[profile] = best.method
        boot = read_json(directory / f'bootstrap_{profile}.json')
        for r in candidates:
            low, high = boot['intervals'][r.method]
            rows.append(['一月' if profile == 'january_12' else '六月', NAMES[r.method], r.selected_parameter,
                         percentage(r.accuracy), percentage(r.macro_f1), f'{percentage(low)}–{percentage(high)}'])
    jan, june = [selected_row(df, name, winners[name]) for name in ['january_12', 'temporal_12']]
    p += [f'在固定表示接相同 SVM 的主要比较中，一月 12 类测试的最高 Macro-F1 来自 {NAMES[jan.method]}（{percentage(jan.macro_f1)}%）；'
          f'一月至四月训练后的六月测试以 {NAMES[june.method]} 最高（{percentage(june.macro_f1)}%）。'
          '下表同时给出准确率与 Macro-F1，后者等权计算各版面效果，避免高频版面主导结论。', '',
          markdown_table(['测试', '固定表示＋SVM', '验证选定 C', 'Accuracy (%)', 'Macro-F1 (%)', 'Macro-F1 95% 区间 (%)'], rows), '',
          figure('performance', '图1.2-1 三种固定表示的主要分类结果。Macro-F1 误差线为同一测试文章按版面分层、配对 bootstrap 2,000 次的百分位 95% 区间；一月 n=456，六月 n=3,191。Accuracy 未附误差线。每个表示只有一套预训练模型，该区间不代表训练种子的波动。', directory), '']
    comparisons = []
    for profile in ['january_12', 'temporal_12']:
        boot = read_json(directory / f'bootstrap_{profile}.json')
        for pair in boot['differences']:
            comparisons.append(['一月' if profile == 'january_12' else '六月', f'{NAMES[pair["a"]]} − {NAMES[pair["b"]]}',
                                percentage(pair['difference']), f'{percentage(pair["ci_low"])}–{percentage(pair["ci_high"])}'])
    p += [markdown_table(['测试', '配对差值', 'Macro-F1 差值（百分点）', '95% 差值区间（百分点）'], comparisons), '',
          '差值区间使用同一批重采样文章计算，可直接观察两种预测的差异；不通过两个模型区间是否重叠来判断。'
          '这些区间没有做多重比较校正，仅用于探索性差异描述。本项目的一月测试集已有历史观察，当前结果也不能视为新的独立确认性证据。', '',
          '一月的三个两两差值区间均跨零，不能据点估计确认稳定优劣。六月 TF-IDF 相对 Word2Vec、FastText 的优势分别为 '
          '4.52 和 5.70 个百分点，对应差值区间为 2.93–6.07 和 4.09–7.24 个百分点；Word2Vec 与 FastText 的差值区间仍跨零。'
          '六月测试因此为 TF-IDF 的优势提供了更明确的支持，但尚不足以区分两套词向量表示的稳定排序。', '',
          '##### 1.2.3.1 类别规模与分类器敏感性', '']
    rows = []
    for n in [2, 4, 8, 12]:
        values = [selected_row(df, f'january_{n}', m) for m in METHODS]
        base = df[(df.profile == f'january_{n}') & (df.method == 'majority')].iloc[0]
        rows.append([n, int(values[0].n), percentage(base.macro_f1), *[percentage(v.macro_f1) for v in values]])
    p += [markdown_table(['版面数', '测试篇数', '多数类 Macro-F1 (%)', 'TF-IDF', 'Word2Vec', 'FastText'], rows), '']
    rows, sensitivity = [], {}
    for profile in ['january_12', 'temporal_12']:
        winners_head = {}
        for head in ['svm', 'logistic', 'softmax']:
            means = {}
            cells = []
            for method in METHODS:
                values = df[(df.profile == profile) & (df.method == method) & (df['head'] == head) & (df.partition == 'test')].macro_f1.to_numpy()
                means[method] = values.mean()
                cells.append(percentage(values[0]) if len(values) == 1 else f'{percentage(values.mean())}±{percentage(values.std(ddof=1))}')
            winners_head[head] = max(means, key=means.get)
            rows.append(['一月' if profile == 'january_12' else '六月', {'svm': 'SVM', 'logistic': '逻辑回归', 'softmax': '线性 Softmax'}[head], *cells])
        sensitivity[profile] = winners_head
    p += [markdown_table(['测试', '相同分类器', 'TF-IDF Macro-F1 (%)', 'Word2Vec', 'FastText'], rows), '',
          '线性 Softmax 为三种子均值±样本标准差（n=3），其余为固定种子 42 的一次分类器拟合；三种子复验没有重新训练输入表示。'
          'SVM 与逻辑回归分别选择 C，Softmax 则独立选择学习率和轮数，因而同一表示在各列具有各自的验证选参结果。', '',
          f'一月的 SVM、逻辑回归、Softmax 优胜表示分别为 {"、".join(NAMES[v] for v in sensitivity["january_12"].values())}；'
          f'六月对应为 {"、".join(NAMES[v] for v in sensitivity["temporal_12"].values())}。'
          '这些结果用来判断主要排名对分类器的依赖，不能把不同分类器的最高测试分重新当作选参依据。', '',
          figure('softmax_validation', '图1.2-2 自行实现的冻结表示＋线性 Softmax 验证曲线。显示种子 42、验证选定学习率的实际轮次，点标记保存的最佳检查点；不进行插值或平滑，不画单次运行误差带。其余两个种子的完整曲线保存在运行目录。', directory), '',
          '##### 1.2.3.2 各版面表现与错误分布', '',
          figure('page_f1', '图1.2-3 各版面测试 F1。三种固定表示均接 SVM，两个面板使用相同的 0–1 色阶；括号标出各版面的测试篇数。逐类 Precision、Recall、F1 和原始混淆计数另存 CSV。', directory), '']
    confusions = []
    for method in METHODS:
        row = selected_row(df, 'temporal_12', method)
        matrix = np.array(read_json(directory / 'runs' / row.run / 'metrics_test.json')['confusion_matrix'])
        candidates = [(int(matrix[i,j]), i, j) for i in range(12) for j in range(12) if i != j and matrix[i,j] > 0]
        confusions.extend([NAMES[method], labels_for(12)[i] + '→' + labels_for(12)[j], count]
                          for count, i, j in sorted(candidates, reverse=True)[:3])
    p += [markdown_table(['表示＋SVM', '六月最常见误判方向', '篇数'], confusions), '',
          figure('confusion', '图1.2-4 六月测试的按行归一化混淆矩阵。前三幅为固定表示＋SVM；第四幅为原生 FastText 预训练初始化、种子 42 的辅助对照。颜色表示每个真实版面的预测比例，所有面板共用色阶。', directory), '',
          '##### 1.2.3.3 覆盖率与计算成本', '']
    ids = set(plan['profiles']['temporal_12']['splits']['test'])
    costs = []
    for method in METHODS:
        coverage = pd.read_csv(directory / f'features/{method}_coverage.csv')
        cov = coverage[coverage.id.isin(ids)]
        row = selected_row(df, 'temporal_12', method)
        source = ROOT / plan['sources'][method]['directory']
        source_size = sum((source / name).stat().st_size for name in (['dictionary.gensim', 'model.gensim'] if method == 'tfidf' else ['vectors.kv']))
        costs.append([NAMES[method], percentage(cov.iv_tokens.sum()/cov.tokens.sum()),
                      percentage(cov.usable_tokens.sum()/cov.tokens.sum()), int(cov.zero_vector.sum()),
                      f'{row.fit_seconds:.3f}', f'{source_size/2**20:.2f}', f'{row.model_bytes/2**10:.2f}'])
    p += [markdown_table(['表示', '六月词表覆盖 (%)', '可用表示覆盖 (%)', '零向量文章', '选定 SVM 拟合 (s)', '推理表示文件 (MiB)', '分类头 (KiB)'], costs), '',
          '词表覆盖按词次计算，可用表示覆盖还计入 FastText 可生成的未登录词。较高覆盖率并不保证正确的类别预测。'
          '拟合时间只包含选定 SVM，不包含预训练、特征提取和其余 C 候选；原始数值另保存完整运行耗时与进程 RSS 峰值。'
          '词向量推理只需 vectors.kv，故本表的表示文件大小与 1.1 中完整训练模型的大小不同。', '',
          '##### 1.2.3.4 时间迁移与原生 FastText 对照', '',
          figure('future', '图1.2-5 冻结一月 12 类 SVM，在二月至六月逐月评价。所有目标月份均不用于改参数；与任意一月文章同近重复组的文章排除。曲线连接真实时间点，训练语料和表示保持固定。', directory), '']
    rows = []
    for profile in ['january_12', 'temporal_12']:
        for method in ['native_scratch', 'native_pretrained']:
            subset = df[(df.profile == profile) & (df.method == method) & (df.partition == 'test')]
            assert len(subset) == 3
            rows.append(['一月' if profile == 'january_12' else '六月', NAMES[method],
                         int(subset.selected_parameter.iloc[0]),
                         f'{percentage(subset.accuracy.mean())}±{percentage(subset.accuracy.std(ddof=1))}',
                         f'{percentage(subset.macro_f1.mean())}±{percentage(subset.macro_f1.std(ddof=1))}'])
    p += ['以下为原先 5、15、25 轮预算中的验证选参结果。随机初始化在一月的验证 Macro-F1 仅 0.0495，'
          '说明这组预算下模型仍不能充分利用训练标签；该现象与项目此前的原生 FastText 拟合诊断一致。', '',
          markdown_table(['测试', '标签联合训练路线', '验证选定轮数', 'Accuracy (%)', 'Macro-F1 (%)'], rows), '',
          '上表为种子 42、2026、7 的均值±样本标准差（n=3），原生 FastText 的多线程更新不保证逐位确定性。'
          '预训练初始化会改变导入的词向量，向量差值探针和导入前零学习率核验均已保存；原始 Gensim 模型的 SHA-256 保持不变。'
          '预训练与随机初始化是相同原生结构下的对照，其结果与固定表示＋SVM 的差异同时包含输入微调、词 bigram 和优化目标的改变。', '',
          ]
    budget_pointer = directory / 'native_budget_latest.json'
    if budget_pointer.exists():
        supplement = ROOT / read_json(budget_pointer)['directory']
        assert read_json(supplement / 'verification.json')['status'] == 'passed'
        extra = pd.read_csv(supplement / 'results.csv')
        rows = []
        for profile in ['january_12', 'temporal_12']:
            for method in ['native_scratch', 'native_pretrained']:
                s = extra[(extra.profile == profile) & (extra.method == method)]
                assert len(s) == 3
                cells = [f'{percentage(s[c].mean())}±{percentage(s[c].std(ddof=1))}'
                         for c in ['train_macro_f1', 'validation_macro_f1', 'test_macro_f1', 'test_accuracy']]
                rows.append(['一月' if profile == 'january_12' else '六月', NAMES[method], int(s.epochs.iloc[0]), *cells])
        p += ['##### 1.2.3.5 扩大原生 FastText 训练预算的探索性检查', '',
              '为检查低验证分数是否与训练预算有关，额外比较 100、200 轮，保持原生模型其余参数不变。'
              '种子 42 仅使用原验证集选轮，固定轮数后补充 2026、7；三个固定表示接 SVM 的模型与主要结果保持封存。'
              '补充启动时初始测试结果已经观察，因此本节属于探索性检查，不能宣称新的独立确认性证据。', '',
              markdown_table(['测试', '原生路线', '补充验证选定轮数', '训练 Macro-F1 (%)', '验证 Macro-F1 (%)', '测试 Macro-F1 (%)', 'Accuracy (%)'], rows), '',
              '以上为三种子均值±样本标准差（n=3）。训练分数只检查拟合程度；训练与验证差距提示过拟合风险。'
              '增加训练预算的结果用于解释初始配置的限制，不能用最初少轮数的低分认定原生 FastText 不适合本任务。', '',
              '一月扩大预算后，随机初始化和预训练初始化的测试 Macro-F1 分别达到 45.97% 和 48.59%；'
              '其训练 Macro-F1 已接近 100%，验证与测试仍约为 50%，说明增加轮数同时带来了较大的泛化差距。'
              '六月的结果没有全面提高：预训练初始化从初始 25 轮的 48.66% 降至补充 100 轮的 47.26%。'
              '该路线种子 42 的验证 Macro-F1 也由 0.4815 降为 0.4730，综合两组已测预算时仍应保留原来的 25 轮配置。', '',
              f'[补充原始结果](../{supplement.relative_to(ROOT).as_posix()}/results.csv) · '
              f'[补充核验](../{supplement.relative_to(ROOT).as_posix()}/verification.json)', '']
    p += ['#### 1.2.4 小结', '',
          f'在现有三套固定表示接相同 SVM 的比较中，{NAMES[winners["january_12"]]} 的一月点估计最高，但差值区间均跨零；'
          f'六月时间扩展测试的 {NAMES[winners["temporal_12"]]} 优势得到配对区间支持。'
          'TF-IDF 在逻辑回归及三种子 Softmax 下也取得了较高的 Macro-F1，是当前配置较合适的版面分类基线。'
          '若选择不同分类器，应结合前述分类器对照，而不单凭词相似度或未登录词覆盖决定表示。'
          '各版面 F1 和时间迁移结果进一步限定了整体分数的适用范围：版面内容随日期变化，版面 ID 与稳定的人工主题标签不能等同。', '',
          'TF-IDF 的优势可能与保留区分版面的稀疏词项有关，均值词向量则将这些信号压缩进较小的固定空间。'
          '本轮没有分离词项权重、维数和池化因素，因此这一解释仍需消融验证。原生 FastText 的预训练初始化可以通过标签训练'
          '改善一月的分类效果；六月初始配置虽取得了略高的点估计，但不能把这项不同训练路线的结果等同于固定 FastText 表示优于 TF-IDF。', '',
          '现有 Word2Vec 和 FastText 的维数不同，表示预训练均只有种子 42，且选参范围有限；Softmax 和原生模型的三种子波动只反映分类训练阶段。'
          '因此，当前结论适用于这些固定模型、人民日报版面标签和既定划分。三套表示均未通过分类损失继续拟合人民日报文本，'
          '只有原生监督 FastText 的辅助路线进行了标签联合训练。主实验选参在打开其测试结果前完成，扩大原生预算的检查按前述探索性边界解释；'
          '保存加载后的逐篇标签及分数一致。', '',
          f'[完整实验结果与运行记录](人民日报固定表示分类实验.md) · [固定方案](../docs/人民日报固定表示分类实验方案.md) · '
          f'[结果 CSV](../{directory.relative_to(ROOT).as_posix()}/results.csv) · '
          f'[运行核验](../{directory.relative_to(ROOT).as_posix()}/verification.json)', '']
    return '\n'.join(p)


def publish(directory, plan, df):
    assert (directory / 'figure_export_audit.json').exists()
    body = manuscript(directory, plan, df)
    full = ROOT / 'reports/人民日报固定表示分类实验.md'
    full.write_text('# 人民日报固定表示与 FastText 分类实验\n\n' + f'更新：{stamp()}。\n\n' + body, encoding='utf-8')
    backup = directory / 'report_before'
    backup.mkdir(exist_ok=True)
    edits = []
    for path in REPORTS:
        raw = path.read_bytes()
        text = raw.decode('utf-8')
        start = text.index('### 1.2 文本分类实验')
        end = start + re.search(r'(?m)^## part2 ', text[start:]).start()
        newline = '\r\n' if '\r\n' in text else '\n'
        replacement = body.replace('\n', newline) + newline
        updated = (text[:start] + replacement + text[end:]).encode('utf-8')
        if not (backup / path.name).exists():
            (backup / path.name).write_bytes(raw)
        assert updated.startswith(text[:start].encode('utf-8'))
        assert updated.endswith(text[end:].encode('utf-8'))
        path.write_bytes(updated)
        edits.append({'path': path.relative_to(ROOT).as_posix(), 'before_sha256': hashlib.sha256(raw).hexdigest(),
                      'after_sha256': file_hash(path), 'outside_1_2_preserved': True,
                      'section_line': text[:start].count('\n') + 1})
    write_json(directory / 'report_edit.json', {'updated': stamp(), 'edits': edits})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['preview', 'export', 'publish'], default='preview')
    args = parser.parse_args()
    directory, plan, df = load()
    {'preview': preview, 'export': export, 'publish': publish}[args.stage](directory, plan, df)


if __name__ == '__main__':
    main()
