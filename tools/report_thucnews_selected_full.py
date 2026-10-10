"""Publish a separate full-candidate report without regenerating user-edited prose."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gensim_lab.common import ROOT, file_hash, markdown_table, read_json, stamp, write_json
from gensim_lab.evaluation_support import save_csv
from gensim_lab.runtime import numerical_threads
from gensim_lab.thucnews_selected_full import BASE

REPORT = ROOT / 'reports/THUCNews候选参数全量重训练.md'
LABELS = {'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText'}


def load():
    experiment = ROOT / read_json(BASE / 'latest.json')['experiment']
    plan = read_json(experiment / 'index.json')
    for spec in plan['specs'].values():
        for name in ['manifest', 'benchmark_manifest', 'source_selection']:
            assert file_hash(ROOT / spec[name]) == spec[name + '_sha256']
        assert file_hash(ROOT / 'configs/thucnews_selected_full.yaml') == spec['configuration_sha256']
    records = {}
    for method, path in plan['runs'].items():
        directory = ROOT / path
        if (directory / 'seal.json').exists():
            seal = read_json(directory / 'seal.json')
            for name in ['summary.json', 'epochs.json', 'validation_metrics.csv']:
                assert file_hash(directory / name) == seal['files'][name]
            summary = read_json(directory / 'summary.json')
            assert summary['spec'] == plan['specs'][method]
            records[method] = (directory, summary)
    return experiment, plan, records


def link(path, label):
    import posixpath
    target = posixpath.relpath(Path(path).relative_to(ROOT).as_posix(), 'reports')
    return f'[{label}]({target})'


def metric(summary, benchmark='wordsim240'):
    return next(m for m in summary['metrics'] if m['benchmark'] == benchmark)


def number(value, digits=4):
    return '不适用' if value is None else f'{value:.{digits}f}'


def report():
    experiment, plan, records = load()
    cfg = plan['configuration']
    manifest = read_json(ROOT / cfg['manifest'])
    out = ROOT / 'reports/figures/thucnews_selected_full' / experiment.name
    paragraphs = ['# THUCNews 候选参数全量重训练', '',
                  f'更新：{stamp()}。实验状态：`{plan["status"]}`，完成模型 {len(records)}/2。', '',
                  '## 数据与固定参数', '',
                  f'复用既有全量训练分区：{manifest["partitions"]["train"]["documents"]:,} 篇文章、'
                  f'{manifest["partitions"]["train"]["sentences"]:,} 个句段、'
                  f'{manifest["partitions"]["train"]["tokens"]:,} 个词。5,000 篇留出文章不进入训练。'
                  '清洗、分词、输入顺序和评价分区均沿用原实验；不重新抽样，不使用类别标签。', '',
                  markdown_table(['参数', 'Word2Vec', 'Gensim FastText'], [
                      ['训练方式', 'Skip-gram＋NS5', 'Skip-gram＋NS10'],
                      ['HS / negative', '0 / 5', '0 / 10'],
                      ['初始 / 最低学习率', '0.025 / 0.0001', '0.05 / 0.0001'],
                      ['向量维数', '300', '100'],
                      ['窗口 / 最低词频 / 下采样', '5 / 5 / 0.001', '5 / 5 / 0.001'],
                      ['训练线程 / 种子', '20 / 42', '20 / 42'],
                      ['字符子词 / 哈希桶', '不适用', '1–3 / 200,000'],
                  ]), '',
                  '候选参数来自固定 5 万篇语料的三阶段搜索，冻结后在全量语料上从头训练。'
                  '本轮每个模型训练一个种子，不能将搜索阶段的三种子标准差作为全量结果的误差。', '',
                  link(ROOT / cfg['source_selection'], '候选参数来源') + ' · ' +
                  link(ROOT / 'configs/thucnews_selected_full.yaml', '完整配置') + ' · ' +
                  link(ROOT / cfg['manifest'], '数据清单'), '',
                  '## 逐轮验证与独立选轮', '',
                  'Word2Vec 和 FastText 分别使用 WordSim240 验证 Spearman 选取最佳检查点。'
                  '至少训练 5 轮，最多 15 轮，连续 3 轮无超过 0.001 的有效提升时停止。'
                  '学习率按 15 轮线性日程衰减；训练的最后一轮不会自动替代最佳轮。'
                  '每轮完整遍历训练集，保存原生检查点及验证明细，完成选轮后才评价测试分区。'
                  '本轮不记录训练 loss；负采样模型不调用 HS-only 的上下文对数评分。', '']
    rows, comparisons, tests, costs = [], [], [], []
    for method in cfg['models']:
        directory = ROOT / plan['runs'][method]
        paragraphs += [f'### {LABELS[method]}', '']
        if method not in records:
            paragraphs += ['尚未完成；最终得分、选中轮次和测试结论暂不填写。', '']
        else:
            _, summary = records[method]
            epochs = read_json(directory / 'epochs.json')
            chosen = epochs[summary['selected_epoch'] - 1]
            score = chosen['primary_score']
            reason = '验证耐心规则' if summary['stop_reason'] == 'validation_patience' else '15 轮上限'
            paragraphs += [f'实际训练 {summary["trained_epochs"]} 轮，按{reason}结束。'
                           f'独立选中第 {summary["selected_epoch"]} 轮，WordSim240 验证 Spearman 为 {score:.4f}。'
                           '其余验证题组用于诊断，不参与候选参数重选。', '']
            rows.append([LABELS[method], summary['trained_epochs'], summary['selected_epoch'], number(score),
                         reason, summary['vocabulary_size']])
            image = out / f'{method}_validation.png'
            if image.exists():
                paragraphs += [f'![{LABELS[method]} 全量逐轮验证曲线。虚线标记独立选中的轮次。'
                               '类比使用每题组至多128道固定验证题，全部题正确率将OOV按错误计。'
                               '各点为实际完成轮次，不插值、不平滑；每个模型只有一次全量运行，无误差带。]'
                               f'({os.path.relpath(image, REPORT.parent).replace(chr(92), "/")})', '',
                               link(image.with_suffix('.svg'), 'SVG 矢量图'), '']
            old_directory = ROOT / plan['historical_baseline']['models'][method]
            old = read_json(old_directory / 'summary.json')
            assert old['spec']['training_corpus_sha256'] == summary['spec']['training_corpus_sha256']
            assert old['spec']['benchmark_manifest_sha256'] == summary['spec']['benchmark_manifest_sha256']
            old_epochs = read_json(old_directory / 'epochs.json')
            comparisons.append([LABELS[method], number(old_epochs[old['selected_epoch'] - 1]['primary_score']),
                                number(score), number(metric(old)['spearman_rho']),
                                number(metric(summary)['spearman_rho']),
                                number(metric(summary)['spearman_rho'] - metric(old)['spearman_rho'])])
            for item in summary['metrics']:
                tests.append({'method': method, **{k: v for k, v in item.items() if k != 'sections'}})
            costs.append({'method': method, 'trained_epochs': summary['trained_epochs'],
                          'fit_seconds': summary['fit_seconds'],
                          'seconds_per_epoch': summary['fit_seconds'] / summary['trained_epochs'],
                          'peak_rss_mib': summary['peak_rss_mib'], 'model_mib': summary['model_bytes'] / 2**20,
                          **summary['heldout']})
        names = [('spec.json', '配置'), ('run.log', '日志'), ('epochs.csv', '逐轮记录'),
                 ('validation_metrics.csv', '逐轮指标'), ('metrics.json', '最终测试'),
                 ('model.gensim', '选定模型'), ('vectors.kv', 'KV'),
                 ('save_load_check.json', '重载核查')]
        paragraphs += [' · '.join(link(directory / name, label) for name, label in names
                                  if (directory / name).exists()), '']
        probes = ROOT / 'artifacts/thucnews_validation/posthoc' / directory.name / 'qualitative_probes.json'
        if probes.exists():
            paragraphs += [link(probes, '近邻、词组相似度、类比和离群词接口诊断'), '']
    if rows:
        paragraphs += [markdown_table(['模型', '实际轮数', '选中轮', '验证 Spearman', '结束原因', '词表大小'], rows), '',
                       '## 与原全量基线比较', '',
                       '旧基线为 CBOW＋HS、100 维、alpha=0.025，种子、全量训练语料及评价分区相同。'
                       '新方案同时改变训练方式、目标和部分模型的维数或学习率，差异反映整套候选配置，'
                       '不能归因于某一个参数。旧基线与本轮各为一次全量运行，不作显著性结论。', '',
                       markdown_table(['模型', '旧验证 ρ', '新验证 ρ', '旧测试 ρ', '新测试 ρ', '测试差值'], comparisons), '']
    if tests:
        save_csv(experiment / 'test_metrics.csv', tests)
        save_csv(experiment / 'resource_and_coverage.csv', costs)
        paragraphs += ['## 完整测试指标', '',
                       '词相似度相关系数仅在词表内词对上计算；FastText 子词生成 OOV 的相关系数另存原始指标，'
                       '不混入主指标。类比分别给出覆盖题正确率和将 OOV 按错误计算的全部题正确率。'
                       'WordSim297 与 WordSim296 存在题目重叠，其结果不能视为两份独立验证。', '',
                       markdown_table(['模型', '题组', '覆盖 / 总题数', 'Pearson r', 'Spearman ρ',
                                       '覆盖题正确率', '全部题正确率'],
                                      [[LABELS[r['method']], r['benchmark'], f"{r['covered']}/{r['total']}",
                                        number(r.get('pearson_r')), number(r.get('spearman_rho')),
                                        number(r.get('conditional_accuracy')), number(r.get('all_accuracy'))]
                                       for r in tests]), '',
                       '## 计算成本与留出语料覆盖', '',
                       markdown_table(['模型', '累计训练秒', '每轮训练秒', '峰值 RSS / MiB', '模型 / MiB',
                                       '词表覆盖', '可用向量覆盖'],
                                      [[LABELS[r['method']], number(r['fit_seconds'], 1),
                                        number(r['seconds_per_epoch'], 1), number(r['peak_rss_mib'], 1),
                                        number(r['model_mib'], 1), number(r['vocabulary_token_coverage']),
                                        number(r['usable_vector_token_coverage'])] for r in costs]), '',
                       '训练耗时仅计参数更新；建词表、验证、检查点保存和测试另有计时。'
                       '累计训练时间受提前停止轮次影响，不能直接解释为单位轮次速度。'
                       'RSS 为模型主进程峰值；模型大小为最佳轮的完整原生文件。', '']
        figures = [('resources', '计算成本与词表大小；TF-IDF 沿用既有统计模型，本轮未重训'),
                   ('similarity_metrics', '测试词相似度及模型间差值'),
                   ('analogy_accuracy', '测试类比正确率'),
                   ('coverage', '留出语料的词表、可用向量与未登录词覆盖'),
                   ('wordsim_pairs', 'WordSim240 逐词对人工得分与模型余弦相似度')]
        for name, caption in figures:
            image = out / f'{name}.png'
            if image.exists():
                paragraphs += [f'![{caption}]({os.path.relpath(image, REPORT.parent).replace(chr(92), "/")})', '',
                               link(image.with_suffix('.svg'), 'SVG 矢量图'), '']
    paragraphs += ['## 核验与复现', '',
                   '参数已冻结，本轮不依据测试结果回调参数。基准此前已有观察，结论仍属探索性；'
                   '单种子全量结果不能排除多线程训练波动。本轮只重训两个固定候选，未在全量语料上比较'
                   '其余训练方式、学习率或维数，因而不能确认小规模搜索的参数排序在全量语料上仍然成立。', '',
                   link(experiment / 'index.json', '实验索引') + ' · ' +
                   link(experiment / 'historical_baseline.json', '旧基线索引快照') + ' · ' +
                   link(BASE / 'progress.json', '实时进度'), '']
    verification = experiment / 'verification.json'
    if verification.exists():
        assert read_json(verification)['status'] == 'passed'
        paragraphs += ['独立核验已通过：完整训练语料计数及词频、每轮模型参数和主指标重算、'
                       '最佳轮次、逐题测试指标、原生模型与保存词向量一致。两个模型核验通过后统一更新正式索引。', '',
                       link(verification, '独立核验') + ' · ' + link(experiment / 'test_metrics.csv', '测试原始数值') +
                       ' · ' + link(experiment / 'resource_and_coverage.csv', '资源与覆盖原始数值'), '']
    review = experiment / 'figure_review/approved.json'
    if review.exists():
        paragraphs += [link(review, '图表来源与导出核验') + ' · ' +
                       link(review.parent / 'visual_review.json', '彩色与灰度视觉核验'), '']
    paragraphs += ['```powershell', 'conda run -n Gensim python -m gensim_lab.thucnews_selected_full',
                   'conda run -n Gensim python -m gensim_lab.thucnews_selected_full --stage audit',
                   'conda run -n Gensim python tools/report_thucnews_selected_full.py --stage preview',
                   '# 核验彩色与灰度预览并记录 visual_review.json 后导出',
                   'conda run -n Gensim python tools/report_thucnews_selected_full.py --stage export',
                   '```', '']
    REPORT.write_text('\n'.join(paragraphs), encoding='utf-8')
    print(REPORT)


def adjust_validation_annotations(figure, frame):
    best = frame.attrs['selected_epoch']
    for annotation in figure.axes[0].texts:
        if annotation.get_text().startswith('第 '):
            annotation.set_position((10 if best == min(frame.epoch) else 0, 16))
    return figure


def adjust_difference_annotations(ax):
    for annotation in ax.texts:
        if annotation.get_text().startswith('-'):
            annotation.xy = (0, annotation.xy[1])
            annotation.set_position((6, 0))
            annotation.set_horizontalalignment('left')


def figures(export):
    from gensim_lab import submission_figures as fig
    experiment, plan, records = load()
    assert plan['status'] == 'complete' and len(records) == 2
    assert read_json(experiment / 'verification.json')['status'] == 'passed'
    assert read_json(ROOT / 'artifacts/thucnews_validation/latest.json')['models'] == plan['runs']
    fig.QA = experiment / 'figure_review'
    fig.OUT = ROOT / 'reports/figures/thucnews_selected_full' / experiment.name
    original_curves = fig.validation_curves
    fig.validation_curves = lambda frame, method: adjust_validation_annotations(original_curves(frame, method), frame)
    original_differences = fig.difference_bars
    def difference_bars(ax, *args, **kwargs):
        original_differences(ax, *args, **kwargs)
        adjust_difference_annotations(ax)
    fig.difference_bars = difference_bars
    fig.build(export=export)
    if export:
        approval = read_json(fig.QA / 'approved.json')
        approval['full_report_generator_sha256'] = file_hash(Path(__file__))
        write_json(fig.QA / 'approved.json', approval)
    report()


def main():
    from threadpoolctl import threadpool_limits
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['report', 'preview', 'export'], default='report')
    args = parser.parse_args()
    with threadpool_limits(limits=numerical_threads()):
        if args.stage == 'report':
            report()
        else:
            figures(args.stage == 'export')


if __name__ == '__main__':
    main()
