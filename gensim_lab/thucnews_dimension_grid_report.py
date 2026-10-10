"""Publish dimension-search results without regenerating the approved earlier prose."""
from __future__ import annotations

import re
import statistics
from pathlib import Path

from .common import ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .evaluation_support import save_csv
from .thucnews_dimension_grid import rank_grid

REPORT = ROOT / 'reports/THUCNews向量维数搜索.md'
NAMES = {'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText'}
START, END = '<!-- thucnews-dimension:start -->', '<!-- thucnews-dimension:end -->'


def aggregate_metrics(experiment, rows):
    summaries = {r['run']: read_json(ROOT / r['run'] / 'summary.json') for r in rows}
    output = []
    for method in NAMES:
        for dimension in sorted({r['vector_size'] for r in rows}):
            group = [r for r in rows if r['method'] == method and r['vector_size'] == dimension]
            for benchmark in [m['benchmark'] for m in summaries[group[0]['run']]['validation_metrics']]:
                metrics = [next(m for m in summaries[r['run']]['validation_metrics'] if m['benchmark'] == benchmark)
                           for r in group]
                assert len({(m['covered'], m['total']) for m in metrics}) == 1
                record = {'method': method, 'vector_size': dimension, 'benchmark': benchmark,
                          'n': len(group), 'covered': metrics[0]['covered'], 'total': metrics[0]['total']}
                for key in ['pearson_r', 'spearman_rho', 'all_accuracy', 'conditional_accuracy',
                            'generatable_spearman_rho', 'oov_ratio_percent']:
                    values = [m[key] for m in metrics if m.get(key) is not None]
                    if values:
                        assert len(values) == len(group)
                        record.update({key + '_mean': statistics.mean(values), key + '_sd': statistics.stdev(values)})
                output.append(record)
    save_csv(experiment / 'dimension_validation_metrics.csv', output)
    costs = []
    for method in NAMES:
        for dimension in sorted({r['vector_size'] for r in rows}):
            group = [r for r in rows if r['method'] == method and r['vector_size'] == dimension]
            record = {'method': method, 'vector_size': dimension, 'n': len(group)}
            for key, getter in [('fit_seconds', lambda r: r['fit_seconds']),
                                ('per_epoch_seconds', lambda r: r['fit_seconds']/r['trained_epochs']),
                                ('peak_rss_mib', lambda r: r['peak_rss_mib']),
                                ('model_mib', lambda r: r['model_bytes']/2**20)]:
                values = [getter(r) for r in group]
                record[key + '_mean'], record[key + '_sd'] = statistics.mean(values), statistics.stdev(values)
            costs.append(record)
    save_csv(experiment / 'dimension_cost_metrics.csv', costs)
    return output, costs


def replace_block(text, body):
    block = START + '\n\n' + body.strip() + '\n\n' + END
    if START in text:
        assert text.count(START) == text.count(END) == 1
        return text.split(START, 1)[0] + block + text.split(END, 1)[1]
    marker = '##### 1.1.3.3 候选参数与结论范围'
    if marker in text:
        text = text.replace(marker, '##### 1.1.3.4 候选参数与结论范围', 1)
        return text.replace('##### 1.1.3.4 候选参数与结论范围', block + '\n\n##### 1.1.3.4 候选参数与结论范围', 1)
    return text.replace('<!-- thucnews-ablation:end -->', block + '\n\n<!-- thucnews-ablation:end -->', 1)


def update_submission(body, selected):
    for name in ['Gensim&FastText.md', 'Gensim&FastText-去AI味版.md']:
        path = ROOT / 'reports' / name
        if not path.exists():
            continue
        original_bytes = path.read_bytes()
        text = path.read_text(encoding='utf-8')
        assert '<!-- thucnews-ablation:end -->' in text
        text = replace_block(text, body)
        if name.endswith('-去AI味版.md'):
            if '训练方式与学习率阶段的两类模型均设为 100 维' not in text:
                text = text.replace('两类模型均设为 100 维，窗口大小为 5，',
                                    '训练方式与学习率阶段的两类模型均设为 100 维，窗口大小为 5，', 1)
            old = re.search(r'(?s)(##### 1\.1\.3\.4 候选参数与结论范围\n\n).*?(?=\n\n这一选择同时受)', text)
            assert old
            configurations = '；'.join(f'{NAMES[m]} 为 {r["vector_size"]} 维、Skip-gram＋NS{5 if m == "word2vec" else 10}、初始学习率 {r["alpha"]:g}'
                                      for m, r in selected.items())
            paragraph = ('三个阶段按训练方式、学习率、向量维数依次搜索，最终候选为：' + configurations + '。'
                         '维数比较继承各模型在 100 维时选定的学习率，因而候选代表分阶段搜索结果。'
                         'WordSim297 与 WordSim296 存在题目重叠，其相近趋势属于相关证据；'
                         '辅助类比结果与计算成本共同限定了这些候选配置的适用目标。')
            text = text[:old.start()] + old.group(1) + paragraph + text[old.end():]
        # Preserve every byte outside the edited hyperparameter-search section.
        opening, closing = b'<!-- thucnews-ablation:start -->', b'<!-- thucnews-ablation:end -->'
        prefix, rest = original_bytes.split(opening, 1)
        _, suffix = rest.split(closing, 1)
        new_body = text.split(opening.decode(), 1)[1].split(closing.decode(), 1)[0]
        newline = b'\r\n' if original_bytes.count(b'\r\n') > original_bytes.count(b'\n')/2 else b'\n'
        path.write_bytes(prefix + opening + new_body.encode('utf-8').replace(b'\n', newline) + closing + suffix)


def report(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    assert plan['kind'] == 'dimension_grid'
    prefix = '../' + experiment.relative_to(ROOT).as_posix()
    if plan['status'] != 'complete':
        REPORT.write_text('# THUCNews 向量维数搜索\n\n<a id="thucnews-dimensions"></a>\n\n'
                          f'更新时间：{stamp()}。\n\n'
                          f'当前状态：{plan["status"]}，完整配置已登记 {len(plan["runs"])}/{plan["expected_search_runs"]}。'
                          '搜索尚未结束，各模型的候选维数在完整三种子网格核验后确定。\n\n'
                          f'[实验索引]({prefix}/index.json) · [搜索方案](../docs/THUCNews向量维数搜索方案.md)\n',
                          encoding='utf-8', newline='\n')
        return REPORT
    assert rank_grid(plan['runs'], plan['grid']) == plan['dimension_ranking']
    assert read_json(experiment / 'verification.json')['status'] == 'passed'
    receipt = read_json(experiment / 'figures/approved.json')
    assert receipt['data_fingerprint'] == fingerprint(plan['runs'])
    assert all(file_hash(ROOT / p) == h for p, h in receipt['files'].items())
    rows, grid = plan['runs'], plan['grid']
    metrics, costs = aggregate_metrics(experiment, rows)
    intro = ('在固定 50,000 篇、23,080,089 词训练子集上，比较 50、100、150、200、300 维。'
             'Word2Vec 固定 Skip-gram＋NS5、初始学习率 0.025；Gensim FastText 固定 Skip-gram＋NS10、初始学习率 0.05。'
             '每档均完成种子 42、2026、7，共 30 组结果；100 维六组经封存核验后复用，其余 24 组从头训练。'
             '窗口、最低词频、下采样、子词和线程设置沿用前述搜索，词表及主验证词对保持一致。')
    protocol = ('各次运行通过逐轮 WordSim240 验证 Spearman 独立选择最佳检查点，至少 3 轮、最多 15 轮，'
                '停止规则及线性学习率日程沿用学习率阶段。第 3 轮若当前得分相对前一轮与前两轮均未提高超过 0.001，即停止；'
                '此后若连续 3 轮未形成有效提升，且最近一至两轮也无超过 0.001 的增幅，则停止。'
                '任何严格更高的得分均更新最佳检查点，0.001 仅作为停止规则的增幅阈值；'
                '学习率按 15 轮上限线性降至 0.0001，提前停止不重新缩放日程。'
                '主指标使用相同的 92 个词表内词对，验证集共 95 题。'
                '维数选择依据三次运行的验证均值，精确同分时优先较低维数。下述误差均为样本标准差（n=3），'
                '反映固定题目下的训练波动，未进行显著性检验。')
    standalone = ['# THUCNews 向量维数搜索', '<a id="thucnews-dimensions"></a>',
                  '## 实验设计', intro, protocol]
    body = ['##### 1.1.3.3 向量维数搜索', intro, protocol]
    for i, method in enumerate(NAMES, 1):
        name, ranking = NAMES[method], plan['dimension_ranking'][method]
        winner = ranking['comparisons'][0]
        reference = next(c for c in ranking['comparisons'] if c['choice'] == grid['reference_vector_size'])
        gap = winner['mean'] - reference['mean']
        paragraph = (f'{name} 的最高验证均值出现在 {winner["choice"]} 维，为 '
                     f'{winner["mean"]:.4f}±{winner["sd"]:.4f}。')
        if winner['choice'] == grid['reference_vector_size']:
            second = ranking['comparisons'][1]
            paragraph += (f'第二高均值为 {second["choice"]} 维的 {second["mean"]:.4f}，'
                          f'低于 100 维 {winner["mean"]-second["mean"]:.4f}。'
                          '当前网格中增减维数均未获得更高的验证均值，因此保留 100 维作为候选。')
        else:
            second = ranking['comparisons'][1]
            paragraph += (f'100 维参考结果为 {reference["mean"]:.4f}±{reference["sd"]:.4f}，两者均值差为 {gap:.4f}。'
                          f'第二高均值为 {second["choice"]} 维的 {second["mean"]:.4f}，'
                          f'与最高均值相差 {winner["mean"]-second["mean"]:.4f}；各档的全部种子结果列于下表。')
        if winner['choice'] == max(grid['vector_sizes']):
            paragraph += f'最高均值位于本次网格的上界 {winner["choice"]} 维，更大维数下的趋势尚未确定。'
        elif winner['choice'] == min(grid['vector_sizes']):
            paragraph += f'最高均值位于本次网格的下界 {winner["choice"]} 维，更小维数尚未比较。'
        records = []
        for dimension in sorted(grid['vector_sizes']):
            group = [next(r for r in rows if r['method'] == method and r['vector_size'] == dimension
                          and r['model_seed'] == s) for s in grid['seeds']]
            summary = next(c for c in ranking['comparisons'] if c['choice'] == dimension)
            records.append([dimension, *[f'{r["validation_score"]:.4f}' for r in group],
                            f'{summary["mean"]:.4f}±{summary["sd"]:.4f}', '选中' if dimension == winner['choice'] else ''])
        table = markdown_table(['向量维数', '种子42', '种子2026', '种子7', '验证均值±样本SD', '选择结果'], records)
        standalone.extend([f'## {name}', paragraph, table])
        body.extend([f'###### 1.1.3.3.{i} {name}', paragraph, table])
        selected_runs = [next(r for r in rows if r['method'] == method and r['vector_size'] == winner['choice']
                             and r['model_seed'] == s) for s in grid['seeds']]
        trajectory = ('候选维数下，三个种子的最佳轮次依次为 ' + '、'.join(str(r['selected_epoch']) for r in selected_runs) +
                      '，实际训练轮数为 ' + '、'.join(str(r['trained_epochs']) for r in selected_runs) +
                      '。逐轮曲线保留各维数的实测轨迹，以区分表示维数变化与选定轮次差异。')
        initial_stops = [r for r in rows if r['method'] == method and r['stop_reason'] == 'initial_no_upward_trend']
        if initial_stops:
            groups = '、'.join(f'{r["vector_size"]} 维／种子 {r["model_seed"]}' for r in initial_stops)
            trajectory += f'{groups} 在第 3 轮按初始趋势规则停止，其低分只能说明当前学习率与训练预算下的结果，不能据此判定该维数的充分训练性能。'
        at_limit = [r for r in rows if r['method'] == method
                    and r['trained_epochs'] == plan['configuration']['validation']['max_epochs']]
        if at_limit:
            limited = '、'.join(f'{r["vector_size"]} 维／种子 {r["model_seed"]}' for r in at_limit)
            trajectory += f'本模型全部 15 次运行中，{limited} 达到 15 轮训练上限，这些配置的评分仍受共同预算约束。'
        preferences = {}
        for benchmark, key, label in [('wordsim297', 'spearman_rho_mean', 'WordSim297'),
                                      ('wordsim296', 'spearman_rho_mean', 'WordSim296'),
                                      ('CA_translated', 'all_accuracy_mean', 'CA_translated 类比'),
                                      ('CA8_morphological', 'all_accuracy_mean', 'CA8 形态类比'),
                                      ('CA8_semantic', 'all_accuracy_mean', 'CA8 语义类比')]:
            candidates = [r for r in metrics if r['method'] == method and r['benchmark'] == benchmark]
            best = sorted(candidates, key=lambda r: (-r[key], r['vector_size']))[0]
            preferences[benchmark] = {'dimension': best['vector_size'], 'mean': best[key], 'label': label}
        sim297, sim296 = preferences['wordsim297'], preferences['wordsim296']
        if sim297['dimension'] == sim296['dimension']:
            diagnosis = (f'WordSim297 与 WordSim296 均在 {sim297["dimension"]} 维达到最高验证 Spearman 均值，'
                         f'分别为 {sim297["mean"]:.4f} 和 {sim296["mean"]:.4f}。')
        else:
            diagnosis = (f'WordSim297 的最高验证 Spearman 均值为 {sim297["mean"]:.4f}（{sim297["dimension"]} 维），'
                         f'WordSim296 为 {sim296["mean"]:.4f}（{sim296["dimension"]} 维）。')
        diagnosis += '类比探针的最高全题正确率均值分别为：' + '、'.join(
            f'{preferences[b]["label"]} {preferences[b]["mean"]:.4f}（{preferences[b]["dimension"]} 维）'
            for b in ['CA_translated', 'CA8_morphological', 'CA8_semantic']) + '。'
        diagnosis += '这些辅助指标在主指标选定的检查点上评价；不同任务的维数偏好共同限定了候选配置的适用目标，选参仍使用预设主指标。'
        selected_cost = next(r for r in costs if r['method'] == method and r['vector_size'] == winner['choice'])
        comparison_dimension = (max(grid['vector_sizes']) if winner['choice'] == grid['reference_vector_size']
                                else grid['reference_vector_size'])
        baseline_cost = next(r for r in costs if r['method'] == method and r['vector_size'] == comparison_dimension)
        cost = (f'候选维数的平均每轮训练耗时为 {selected_cost["per_epoch_seconds_mean"]:.1f} 秒，'
                f'平均进程峰值内存为 {selected_cost["peak_rss_mib_mean"]:.1f} MiB，'
                f'平均模型文件大小为 {selected_cost["model_mib_mean"]:.1f} MiB。'
                f'{comparison_dimension} 维配置的对应值为 {baseline_cost["per_epoch_seconds_mean"]:.1f} 秒、'
                f'{baseline_cost["peak_rss_mib_mean"]:.1f} MiB、{baseline_cost["model_mib_mean"]:.1f} MiB。'
                '累计耗时还受实际训练轮数影响，资源图同时报告累计时间与平均每轮时间；参数更新时间不含建词表、评价及保存。')
        for target in [standalone, body]:
            for kind, text in [('effect', None), ('curves', trajectory), ('diagnostics', diagnosis), ('cost', cost)]:
                if text:
                    target.append(text)
                key = f'dimension_{kind}_{method}'
                relative = Path(receipt['figures'][key]).relative_to('reports').as_posix()
                target.append(f'![{receipt["captions"][key]}]({relative})')
    scope = ('维数搜索在各模型已经选定的训练方式与学习率下进行，所得候选属于分阶段条件搜索，'
             '尚未系统比较维数与学习率的交互。相同语料和词表使各档的覆盖率一致，维数差异主要通过验证效果及计算成本评价。'
             '三种子和较小验证集提供探索性选参依据；较低学习率或较高维数在共同轮数预算之外的表现仍需另行确认。'
             '100 维的资源统计继承前一阶段封存运行，耗时比较属于同机实测记录，仍受运行时段与系统负载影响。'
             '本阶段没有使用测试分区或启动全量重训。')
    resources = (f'[完整逐次结果]({prefix}/results.csv) · [辅助指标]({prefix}/dimension_validation_metrics.csv) · '
                 f'[资源指标]({prefix}/dimension_cost_metrics.csv) · [运行核验]({prefix}/verification.json)')
    standalone.extend(['## 选择范围与复现', scope, resources,
                       '[配置](../configs/thucnews_dimension_grid.yaml) · [方案](../docs/THUCNews向量维数搜索方案.md)',
                       '```powershell\nconda run -n Gensim python -m gensim_lab.thucnews_dimension_grid\n'
                       'conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage audit\n'
                       'conda run -n Gensim python -m gensim_lab.thucnews_dimension_grid --stage report\n```'])
    body.extend([scope, resources])
    REPORT.write_text('\n\n'.join(standalone) + '\n', encoding='utf-8', newline='\n')
    update_submission('\n\n'.join(body), plan['selected'])
    write_json(experiment / 'report_status.json', {'status': 'complete', 'search_runs': len(rows),
               'figures_approved': True, 'report_sha256': file_hash(REPORT), 'generated': stamp()})
    return REPORT
