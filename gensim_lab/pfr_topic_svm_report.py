"""Scientific figures and a scoped report for the frozen topic SVM comparison."""
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
from .evaluation_support import save_csv
from .figure_support.setup_style import setup_style
from .figure_support.profile_data import profile_data, render_report
from .figure_support.visual_qa import audit_layout, render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure
from .pfr_topic_svm import BASE, METHODS
from .pfr_topics import load_jsonl

NAMES = {'tfidf': 'TF-IDF', 'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText'}
COLORS = ['#0072B2', '#D55E00', '#009E73']
MARKERS = ['o', 's', '^']
FIGURES = ['performance', 'validation', 'topic_f1', 'confusion', 'solver_accuracy']
REPORTS = [ROOT/'reports/Gensim&FastText.md', ROOT/'reports/Gensim&FastText-修改版.md']
HEADING = '#### 1.2.6 三种冻结表示与同一 LinearSVC 的比较'


def load():
    directory = ROOT/read_json(BASE/'latest.json')['experiment']
    assert read_json(directory/'index.json')['status'] == 'complete'
    assert read_json(directory/'verification.json')['status'] == 'passed'
    for name, digest in read_json(directory/'test_seal.json').items():
        assert file_hash(directory/name) == digest
    df = pd.read_csv(directory/'results.csv').set_index('method').loc[METHODS].reset_index()
    assert len(df) == 3 and np.isfinite(df[['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']]).all().all()
    return directory, read_json(directory/'plan.json'), df


def iterations(directory):
    folder = directory/read_json(directory/'iteration_diagnostic/latest.json')['directory']
    for name, digest in read_json(folder/'seal.json').items(): assert file_hash(folder/name) == digest
    assert read_json(folder/'verification.json')['status'] == 'passed'
    return folder, pd.read_csv(folder/'accuracy_curve.csv')


def resources(directory, plan):
    index = read_json(directory/'index.json')
    grid = []
    for run, record in index['runs'].items():
        out = directory/'runs'/run
        result = read_json(out/'metrics_validation.json')
        summary = read_json(out/'summary.json')
        grid.append({**record, 'run': run, **{k: result[k] for k in ['accuracy', 'macro_f1', 'micro_f1', 'weighted_f1']},
                     'fit_seconds': summary['fit_seconds'], 'n_iter': summary['n_iter']})
    save_csv(directory/'validation_grid.csv', grid)
    rows = []
    for method in METHODS:
        frame = pd.read_csv(directory/f'features/{method}_coverage.csv', dtype={'id': str}).set_index('id')
        for part, ids in plan['splits'].items():
            selected = frame.loc[ids]
            n = int(selected.tokens.sum())
            iv, usable = int(selected.iv_tokens.sum()), int(selected.usable_tokens.sum())
            rows.append({'method': method, 'partition': part, 'articles': len(ids), 'tokens': n,
                         'iv_tokens': iv, 'usable_tokens': usable, 'iv_coverage': iv/n, 'usable_coverage': usable/n,
                         'zero_vectors': int(selected.zero_vector.sum())})
    save_csv(directory/'coverage_summary.csv', rows)
    classes = []
    predictions = {}
    for method in METHODS:
        run = index['selection'][method]['run']
        result = read_json(directory/'runs'/run/'metrics_test.json')
        classes.extend({'method': method, 'topic': label, **result['per_class'][label]} for label in plan['labels'])
        predictions[method] = pd.read_csv(directory/'runs'/run/'predictions_test.csv', dtype={'id': str})
    save_csv(directory/'per_class_test.csv', classes)
    docs = {d['id']: d for d in load_jsonl(ROOT/plan['article_file'])}
    examples = []
    for j, article in enumerate(plan['splits']['test']):
        answers = {m: predictions[m].predicted_topic.iloc[j] for m in METHODS}
        if len(set(answers.values())) > 1:
            d = docs[article]
            examples.append({'id': article, 'title': d.get('title', '') or ''.join(d['tokens'][:20]),
                             'automatic_topic': d['topic'], **answers,
                             'unanimous_correct': sum(a == d['topic'] for a in answers.values())})
    save_csv(directory/'disagreement_examples.csv', examples)


def figures(directory, plan, df):
    setup_style(journal='general', lang='zh', use_sciplots=False)
    plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8,
                         'xtick.labelsize': 7, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5})
    index = read_json(directory/'index.json')
    bootstrap = read_json(directory/'bootstrap.json')
    result = {}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ['macro_f1', 'accuracy'], ['a  测试 Macro-F1', 'b  测试 Accuracy']):
        for j, (method, color, marker) in enumerate(zip(METHODS, COLORS, MARKERS)):
            value = float(df.loc[df.method == method, metric].iloc[0])*100
            if metric == 'macro_f1':
                low, high = np.array(bootstrap['intervals'][method])*100
                ax.errorbar(value, j, xerr=[[value-low], [high-value]], fmt=marker, color=color, capsize=3, markersize=5)
            else:
                ax.scatter(value, j, marker=marker, color=color, s=30)
            ax.text(103, j, f'{value:.2f}', va='center', fontsize=7.5)
        ax.set_yticks(range(3), [NAMES[m] for m in METHODS])
        ax.set_ylim(2.6, -.6); ax.set_xlim(0, 119); ax.set_xticks([0, 20, 40, 60, 80, 100])
        ax.set_xlabel('分数 (%)'); ax.set_title(title); ax.grid(axis='x', alpha=.18); ax.set_axisbelow(True)
    result['performance'] = fig
    grid = pd.read_csv(directory/'validation_grid.csv')
    fig, ax = plt.subplots(figsize=(7.2, 3.0), constrained_layout=True)
    for method, color, marker in zip(METHODS, COLORS, MARKERS):
        sub = grid[grid.method == method].sort_values('C')
        ax.scatter(sub.C, sub.macro_f1*100, color=color, marker=marker, s=40, label=NAMES[method])
        selected = index['selection'][method]
        ax.scatter([selected['C']], [selected['validation_macro_f1']*100], facecolors='none', edgecolors=color, s=110, linewidths=1.3)
    ax.set_xscale('log'); ax.set_xticks([.1, 1, 10], ['0.1', '1', '10']); ax.set_xlim(.06, 17)
    ax.set_ylim(0, 100); ax.set_xlabel('正则化参数 C（对数刻度）'); ax.set_ylabel('验证 Macro-F1 (%)')
    ax.legend(loc='lower right', frameon=False); ax.grid(axis='y', alpha=.18)
    result['validation'] = fig
    classes = pd.read_csv(directory/'per_class_test.csv')
    array = classes.pivot(index='topic', columns='method', values='f1-score').loc[plan['labels'], METHODS].to_numpy()
    counts = classes[classes.method == 'tfidf'].set_index('topic').loc[plan['labels'], 'support'].astype(int)
    fig, ax = plt.subplots(figsize=(7.2, 3.8), constrained_layout=True)
    im = ax.pcolormesh(np.arange(4)-.5, np.arange(9)-.5, array, vmin=0, vmax=1, cmap='cividis', rasterized=False)
    ax.set_xticks(range(3), [NAMES[m] for m in METHODS]); ax.set_yticks(range(8), [f'{t} (n={n})' for t,n in zip(plan['labels'], counts)])
    ax.set_ylim(7.5, -.5); ax.set_ylabel('自动主题标签')
    for r in range(8):
        for c in range(3):
            ax.text(c, r, f'{array[r,c]:.2f}', ha='center', va='center', color='white' if array[r,c]<.45 else '#222222')
    cb = fig.colorbar(im, ax=ax, shrink=.85, pad=.02); cb.set_label('逐主题 F1'); cb.solids.set_rasterized(False)
    result['topic_f1'] = fig
    fig = plt.figure(figsize=(7.2, 3.4), layout='none')
    axes = fig.subplots(1, 3)
    fig.subplots_adjust(left=.065, right=.98, bottom=.32, top=.91, wspace=.45)
    for ax, method, letter in zip(axes, METHODS, 'abc'):
        run = index['selection'][method]['run']
        matrix = np.asarray(read_json(directory/'runs'/run/'metrics_test.json')['confusion_matrix'])
        matrix = matrix/matrix.sum(axis=1, keepdims=True)
        im = ax.pcolormesh(np.arange(9)-.5, np.arange(9)-.5, matrix, vmin=0, vmax=1, cmap='cividis', rasterized=False)
        ax.set_xticks(range(8), plan['labels'], rotation=45, ha='right', fontsize=6.5)
        ax.set_yticks(range(8), plan['labels'], fontsize=6.5)
        ax.set_ylim(7.5, -.5); ax.set_aspect('equal'); ax.set_title(f'{letter}  {NAMES[method]}')
        ax.set_xlabel('预测主题'); ax.set_ylabel('自动主题标签')
        for r in range(8):
            for c in range(8):
                if matrix[r,c] >= .1:
                    ax.text(c,r,f'{matrix[r,c]:.2f}',ha='center',va='center',fontsize=6,color='white' if matrix[r,c]<.45 else '#222222')
    cb = fig.colorbar(im, cax=fig.add_axes([.25,.13,.50,.035]), orientation='horizontal')
    cb.set_label('按行归一化比例', labelpad=3); cb.solids.set_rasterized(False)
    result['confusion'] = fig
    _, curve = iterations(directory)
    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=True)
    for method, color, marker, style in zip(METHODS, COLORS, MARKERS, ['-', '--', ':']):
        sub = curve[curve.method == method].sort_values('budget')
        ax.plot(sub.budget, sub.validation_accuracy*100, marker=marker, markersize=4,
                linestyle=style, color=color, linewidth=1.2, label=NAMES[method])
    ax.set_xlim(.5,20.5); ax.set_xticks([1,2,3,5,10,15,20]); ax.set_ylim(0,100)
    ax.set_xlabel('LinearSVC 求解迭代预算（max_iter，非 epoch）')
    ax.set_ylabel('验证 Accuracy (%)'); ax.grid(alpha=.18)
    ax.legend(loc='lower right', frameon=False)
    result['solver_accuracy'] = fig
    return result


def preview(directory, plan, df):
    resources(directory, plan)
    folder = directory/'figure_preview'; folder.mkdir(exist_ok=True)
    profile = profile_data(df, group_cols=['method'])
    # A one-fit group has no estimable SD; keep this absence as null, not NaN.
    def finite_json(value):
        if isinstance(value, dict): return {k: finite_json(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)): return [finite_json(v) for v in value]
        if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
        return value
    write_json(folder/'data_profile.json', finite_json(json.loads(json.dumps(profile, default=str))))
    (folder/'数据剖析与图型选择.md').write_text(render_report(profile)+'\n\n'
        '每种表示对应一个已选定的凸优化分类器，不把三个方法视为三个训练重复。性能图使用点估计，'
        'Macro-F1横线为1910篇测试文章的按主题分层配对bootstrap百分位95%区间（2000次）；'
        'Accuracy只有单次点估计。C只有三个候选，用对数横轴散点而不插值，空圈标记验证选择。'
        '逐类F1和混淆矩阵使用统一0—1的cividis色阶、共享色条与数字标注。比例轴从零开始，'
        '类别使用颜色和点型双重编码，不补造训练误差或显著性符号。求解迭代曲线有20个实际预算，'
        '每预算从头拟合，用连线引导阅读，不解释为逐epoch或同一次训练轨迹。\n', encoding='utf-8')
    audits = {}
    for name, fig in figures(directory, plan, df).items():
        path = folder/f'{name}.png'; render_preview(fig, str(path), dpi=150)
        ImageOps.grayscale(Image.open(path)).convert('RGB').save(folder/f'{name}_grayscale.png')
        audits[name] = {'issues': audit_layout(fig), 'sha256': file_hash(path),
                        'grayscale_sha256': file_hash(folder/f'{name}_grayscale.png')}
        plt.close(fig)
    write_json(folder/'layout_audit.json', audits)
    print(json.dumps(audits, ensure_ascii=False))


def export(directory, plan, df):
    review = read_json(directory/'figure_preview/visual_review.json'); assert review['status'] == 'approved'
    for name in FIGURES:
        assert file_hash(directory/f'figure_preview/{name}.png') == review['previews'][name]['sha256']
        assert file_hash(directory/f'figure_preview/{name}_grayscale.png') == review['previews'][name]['grayscale_sha256']
    folder = ROOT/'reports/figures/pfr_topic_svm'/directory.name; folder.mkdir(parents=True, exist_ok=True)
    audits = {}
    for name, fig in figures(directory, plan, df).items():
        paths = export_figure(fig, str(folder/name), formats=['png', 'svg'], dpi=300, grayscale_preview=True)
        for path in paths:
            if path.endswith('_grayscale.png'):
                with Image.open(path) as im: im.save(path, dpi=(300,300))
        audits[name] = {Path(p).name: {'issues': check_figure(p)[0], 'sha256': file_hash(p)} for p in paths}
        assert not any(i[0] in ['WARN', 'FAIL'] for rec in audits[name].values() for i in rec['issues'])
        plt.close(fig)
    write_json(directory/'figure_export_audit.json', audits)


def pct(value): return f'{value*100:.2f}'


def figure(directory, name, caption):
    path = f'figures/pfr_topic_svm/{directory.name}/{name}.png'
    assert (ROOT/'reports'/path).exists()
    return f'![{caption}]({path})\n\n{caption}\n'


def manuscript(directory, plan, df):
    index = read_json(directory/'index.json'); stats = read_json(directory/'features/summary.json')
    boot = read_json(directory/'bootstrap.json')
    grid = pd.read_csv(directory/'validation_grid.csv')
    coverage = pd.read_csv(directory/'coverage_summary.csv')
    classes = pd.read_csv(directory/'per_class_test.csv')
    ranked = df.sort_values('macro_f1', ascending=False)
    leader = ranked.iloc[0]
    p = [HEADING, '', '##### 1.2.6.1 比较条件与验证选参', '',
        '复用此前THUCNews学到的TF-IDF、Word2Vec和Gensim FastText，冻结词典、IDF、词向量及字符子词参数；'
        '本轮只训练LinearSVC分类器，未新增主题标注或重新训练表示模型。使用与原生FastText部分完全相同的八类主题、'
        '5,333篇一至二月训练文章、820篇三月验证文章和1,910篇三月测试文章。文本、重复组和划分均不变，版号不进入特征。', '',
        'TF-IDF沿用50,000词词典及训练IDF，输出稀疏文章向量；Word2Vec和FastText按文章中词的出现次数加权求均值。'
        'Word2Vec跳过未登录词，FastText允许由字符子词生成词向量。三种文章向量均做L2归一化。'
        '现有Word2Vec为300维、Skip-gram＋NS5、alpha=0.025；FastText为100维、Skip-gram＋NS10、alpha=0.05，'
        '子词1—3、200,000桶。这里比较现有表示的实际使用效果，维数和预训练参数未匹配，不能单独归因于算法结构。', '',
        '分类器统一采用LinearSVC，L2惩罚、squared-hinge损失、one-vs-rest多类策略和balanced类别权重。'
        '使用原始问题求解（dual=False），容差为10⁻⁴，最大20,000次迭代；九组拟合均收敛。'
        '各方法只在训练文章上拟合，在验证集从C={0.1,1,10}选择Macro-F1最高者，平局选较小C。'
        '参数全部冻结后评估测试集，不用验证或测试文章更新表示，不合并验证集重新拟合分类器。', '',
        'LinearSVC使用CPU的liblinear求解器，数值库限制4线程，特征转换使用8个线程。'
        '每组求解一个凸优化分类器，不增加无意义的分类器种子重复；本轮也未重复预训练表示，因此没有词向量训练种子的误差估计。', '',
        markdown_table(['表示', 'C=0.1验证Macro-F1 (%)', 'C=1验证Macro-F1 (%)', 'C=10验证Macro-F1 (%)', '选定C'],
           [[NAMES[m], *[pct(grid[(grid.method==m)&(grid.C==c)].macro_f1.iloc[0]) for c in [.1,1,10]], index['selection'][m]['C']] for m in METHODS]), '',
        figure(directory,'validation','图1.2-6　同一LinearSVC的验证选参。n=820；每个点对应一次独立拟合，空圈为选中C。只显示三个实际候选，不插值。'), '',
        'Word2Vec选中C=10，位于本轮候选上界；该结果只说明它在三个已测C中验证Macro-F1最高，'
        '不声称C=10是全局最佳参数。后续迭代诊断固定此处选定的C，不重新按曲线峰值选参。', '',
        figure(directory,'solver_accuracy','图1.2-7　三种表示接LinearSVC的验证准确率与求解迭代预算。n=820；每种表示实际拟合max_iter=1—20共20个预算，使用同一训练集和各自已选C。三条线置于同图；连接线仅引导阅读独立预算结果，不是按epoch采集的训练曲线。'), '',
        'LinearSVC使用liblinear求解，没有逐epoch回调，也不将一次求解迭代等同于遍历一次语料。'
        '因此该图横轴为求解预算；每个预算均从头拟合，保存模型、完整验证决策分数、实际n_iter和收敛状态。'
        '预算不足的ConvergenceWarning作为预期诊断记录，不当作正式已收敛结果。'
        'TF-IDF和FastText在预算12时首次不再报告未收敛，实际各迭代11次；Word2Vec在预算18时首次收敛，实际迭代17次。'
        '预算20的三个模型均与正式选定分类器的验证预测和决策分数完全一致。'
        '较小预算的验证准确率可能高于收敛值；优化目标收敛并不保证验证准确率单调提高。'
        '本次诊断没有重新读取测试结果或据此改变正式模型。', '',
        '##### 1.2.6.2 整体效果与配对差值', '',
        markdown_table(['表示', 'C', 'Accuracy (%)', 'Macro-F1 (%)', 'Macro-F1 95%区间 (%)', 'Micro-F1 (%)', 'Weighted-F1 (%)'],
           [[NAMES[r.method],r.C,pct(r.accuracy),pct(r.macro_f1),f'[{pct(boot["intervals"][r.method][0])}, {pct(boot["intervals"][r.method][1])}]',pct(r.micro_f1),pct(r.weighted_f1)] for r in df.itertuples()]), '',
        f'{NAMES[leader.method]}的测试Macro-F1点估计最高，为{pct(leader.macro_f1)}%；'
        '排名依次为TF-IDF、Word2Vec、Gensim FastText。Word2Vec在验证集最高，但测试排序改变，'
        '说明单个验证留出集上的小幅差异不能保证在另一批文章中保持。单标签任务中Micro-F1等于Accuracy，二者均保存用于核验。', '',
        figure(directory,'performance','图1.2-8　三种冻结表示接同一LinearSVC的三月留出测试。n=1,910；Macro-F1误差线为按自动主题分层、文章配对重采样2,000次的百分位95%区间；Accuracy仅显示点估计。每种表示使用一个分类器，误差线不是训练种子标准差。'), '',
        markdown_table(['配对差值（前者－后者）', 'Macro-F1差值（百分点）', '95%区间（百分点）'],
           [[f'{NAMES[r["a"]]}－{NAMES[r["b"]]}',f'{100*r["difference"]:.2f}',f'[{100*r["ci_low"]:.2f}, {100*r["ci_high"]:.2f}]'] for r in boot['differences']]), '',
        '三组配对差值区间都包含零。当前数据支持TF-IDF点估计较高，但不足以确认它稳定优于另外两种表示；'
        '不能把包含零解释为三种方法等效。重采样对三种方法使用同一批文章，保持各主题测试数量，'
        '本轮所有近重复组均只有一篇，文章可作为重采样单位。区间以既有弱标签和已训练模型为条件，'
        '不涵盖自动标注误差或词向量预训练波动，也未做多重比较校正，作为探索性比较报告。', '',
        '##### 1.2.6.3 逐类表现与覆盖率', '',
        figure(directory,'topic_f1','图1.2-9　三种表示的逐主题F1。n为该主题测试文章数；统一0—1色阶，格内数字为单个已选模型的F1。'), '',
        markdown_table(['主题', '表示', '测试篇数', 'Precision (%)', 'Recall (%)', 'F1 (%)'],
           [[t,NAMES[m],int(r.support),pct(r.precision),pct(r.recall),pct(r['f1-score'])] for t in plan['labels'] for m in METHODS
            for _,r in classes[(classes.topic==t)&(classes.method==m)].iterrows()]), '',
        figure(directory,'confusion','图1.2-10　三种表示的测试混淆矩阵。行是真实保存的自动主题标签，列是预测主题，按行归一化，三图共享0—1色阶；标出占比至少10%的格，原始计数完整保存。'), '',
        markdown_table(['表示', '测试词token数', '词典内覆盖率 (%)', '可用向量覆盖率 (%)', '零向量文章'],
           [[NAMES[r.method],int(r.tokens),pct(r.iv_coverage),pct(r.usable_coverage),int(r.zero_vectors)] for r in coverage[coverage.partition=='test'].itertuples()]), '',
        '覆盖率按词token计数。TF-IDF词典内覆盖指保留词典中的token；Word2Vec可用向量覆盖与词典内覆盖一致；'
        'FastText可用覆盖还包含子词生成的未登录词向量。高覆盖只说明模型能够构造特征，不保证特征对主题识别有用，'
        '因此不能用FastText的覆盖率直接推断分类优势。词均值压缩也会丢失词袋中的部分区分信息，'
        '这是TF-IDF表现较好的一个可能原因，本轮没有进行因果验证。', '',
        '##### 1.2.6.4 计算成本、差异案例与结论', '',
        markdown_table(['表示', '特征维数', '8063篇特征转换 (s)', '选定分类器拟合 (s)', '加载及1910篇预测 (s)', '拟合进程峰值RSS (MiB)', '分类器文件 (MiB)'],
           [[NAMES[r.method],stats[r.method]['width'],f'{stats[r.method]["seconds"]:.2f}',f'{r.fit_seconds:.3f}',f'{r.predict_and_load_seconds:.4f}',f'{r.peak_rss_mib:.2f}',f'{r.model_bytes/2**20:.4f}'] for r in df.itertuples()]), '',
        '特征转换包含表示文件加载和8063篇文章转换；分类器拟合计fit调用，预测列仅计分类器加载和已缓存测试特征上的推断。'
        'RSS为共享编排进程采样峰值，受已加载库、表示和缓存影响；分类器文件大小不包含外部词典或词向量，不能当作完整部署体积。'
        '这些秒级单次测量用于描述本机成本，不用于推断稳定的速度倍数。FastText分类器较小，'
        '但子词向量生成使本轮特征转换耗时最高；TF-IDF虽然有50,000维，其稀疏计算仍使分类拟合保持在秒以内。', '']
    examples = pd.read_csv(directory/'disagreement_examples.csv').sort_values(['unanimous_correct','id'],ascending=[False,True]).head(4)
    p.extend([markdown_table(['文章标题', '自动标签', 'TF-IDF', 'Word2Vec', 'Gensim FastText'],
         [[r.title,r.automatic_topic,r.tfidf,r.word2vec,r.fasttext] for r in examples.itertuples()]), '',
        '上表从预测不一致文章中，先按与自动标签一致的方法数量降序、再按匿名ID升序取四篇，'
        '只用于展示具体分歧，不代表错误分布。自动标签也可能有误，未经人工核验不能将某个模型的不同判断直接定为语义错误。', '',
        '本轮结论是：在同一LinearSVC、固定语料划分与既有预训练表示下，TF-IDF取得最高整体点估计，'
        'Word2Vec次之，Gensim FastText略低，配对不确定性尚不能确认稳定排序。三套表示的排名与1.2.3的原生FastText初始化实验属于不同问题：'
        '这里表示完全冻结，原生监督训练则会更新输入词向量，并学习词bigram和分类输出权重。'
        '前一部分的初始化收益不能直接转化为本部分冻结表示的优势。', '',
        '三月验证文章参与选参，且此测试集已有原生实验历史观察，结果按探索性的三月留出测试解释。'
        '主题来自单次大模型标注，因此分数衡量与这套弱标签的一致性，不等同于人工真值上的主题识别准确率。', '',
        f'完整资源见[验证网格](../{directory.relative_to(ROOT).as_posix()}/validation_grid.csv)、'
        f'[测试指标](../{directory.relative_to(ROOT).as_posix()}/results.csv)、'
        f'[逐类指标](../{directory.relative_to(ROOT).as_posix()}/per_class_test.csv)、'
        f'[分区覆盖率](../{directory.relative_to(ROOT).as_posix()}/coverage_summary.csv)、'
        f'[配对区间](../{directory.relative_to(ROOT).as_posix()}/bootstrap.json)、'
        f'[求解迭代曲线数值](../{iterations(directory)[0].relative_to(ROOT).as_posix()}/accuracy_curve.csv)、'
        f'[全部分歧案例](../{directory.relative_to(ROOT).as_posix()}/disagreement_examples.csv)和'
        f'[核验](../{directory.relative_to(ROOT).as_posix()}/verification.json)。复现方法见[主题分类指南](../docs/人民日报主题分类复现指南.md)。', ''])
    return '\n'.join(p)


def publish(directory, plan, df):
    body = manuscript(directory, plan, df)
    backup = directory/'report_before'; backup.mkdir(exist_ok=True)
    edit = {}
    for path in REPORTS:
        original = path.read_bytes(); text = original.decode('utf-8')
        if not (backup/path.name).exists(): (backup/path.name).write_bytes(original)
        start = text.index('### 1.2 '); end = text.index('## part2 ', start)
        chapter = text[start:end]
        if HEADING in chapter: chapter = chapter[:chapter.index(HEADING)]
        chapter = chapter.replace('### 1.2 原生 FastText 八类主题分类实验', '### 1.2 八类主题分类：原生 FastText 与三种表示比较')
        chapter = chapter.replace('本轮仅比较原生FastText', '原生监督部分比较原生FastText')
        chapter = chapter.replace('本轮只比较两种初始化方式，没有运行TF-IDF、Word2Vec接分类器或其他分类头，因而不能给三种表示学习方法排分类名次。',
                                  '原生监督部分比较两种初始化方式；三种冻结表示接同一LinearSVC的结果另见1.2.6。')
        changed = text[:start]+chapter.rstrip()+'\n\n'+body+'\n'+text[end:]
        assert changed[:start] == text[:start] and changed.split('## part2 ',1)[1] == text[end:].split('## part2 ',1)[1]
        path.write_bytes(changed.encode('utf-8'))
        edit[path.name] = {'before_sha256': hashlib.sha256(original).hexdigest(), 'after_sha256': file_hash(path),
                           'only_section_1_2_changed': True}
    (ROOT/'reports/人民日报三种表示LinearSVC对比.md').write_text('# 人民日报三种表示与LinearSVC对比\n\n'+body, encoding='utf-8')
    write_json(directory/'report_edit.json', edit)
    print('Published section 1.2.6; sections 1.1 and part2 preserved.')


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['preview','export','publish'])
    args = parser.parse_args(); directory, plan, df = load()
    if args.stage == 'preview': preview(directory, plan, df)
    elif args.stage == 'export': export(directory, plan, df)
    else: publish(directory, plan, df)


if __name__ == '__main__': main()
