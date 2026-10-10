"""Balanced learning-rate evidence and separate model sections in Markdown."""
from pathlib import Path
import statistics

from .common import ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .evaluation_support import save_csv
from .thucnews_ablation_report import LABELS
from .thucnews_small_search_report import REPORT, update_submission

HISTORY = ROOT / 'reports/THUCNews学习率三档历史记录.md'
NAMES = {'word2vec': 'Word2Vec', 'fasttext': 'Gensim FastText'}


def report(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    assert plan['kind'] == 'full_alpha_grid'
    source = ROOT / plan['source_experiment']
    previous = read_json(source / 'index.json')
    grid, rows = plan['grid'], plan['runs']
    rates, seeds = sorted(grid['learning_rates']), grid['seeds']
    manifest = read_json(ROOT / plan['manifest'])
    stats, coverage = manifest['partitions']['train'], manifest['primary_validation']
    complete = plan['status'] == 'complete'
    status = '完成' if complete else {'running': '进行中', 'paused': '已暂停', 'prepared': '已准备', 'failed': '失败'}[plan['status']]
    prefix = '../' + experiment.relative_to(ROOT).as_posix()
    matrix_receipt = read_json(source / 'figures/approved.json')
    assert all(file_hash(ROOT / p) == sha for p, sha in matrix_receipt['files'].items() if '/training_' in p)
    approved = experiment / 'figures/approved.json'
    receipt = read_json(approved) if approved.exists() else None
    if receipt:
        assert complete and receipt['data_fingerprint'] == fingerprint(rows)
        assert all(file_hash(ROOT / p) == sha for p, sha in receipt['files'].items())
    history = HISTORY.read_text(encoding='utf-8')
    matrix_text = history.split('## 抽样、清洗与评价覆盖', 1)[1].split('## 学习率搜索状态', 1)[0]
    matrix_text = '\n\n'.join(p for p in matrix_text.split('\n\n') if not p.startswith('第二阶段：'))
    description = ('固定 5 万篇及验证划分，在 Word2Vec 的 Skip-gram＋NS5、FastText 的 Skip-gram＋NS10 下，'
                   '完整比较初始学习率 **0.005、0.01、0.025、0.05、0.1**；每档均运行 **42、2026、7** 三种子。'
                   '学习率阶段共 **30 组**，复用原有 14 组、补充 16 组；加上此前训练方式阶段，'
                   '共 44 次不同配置训练（两阶段共享 6 组）。每个模型分别按五档三种子验证均值选参，'
                   '不再先以单种子筛掉学习率。')
    protocol = ('固定 100 维、窗口 5、最低词频 5、下采样 0.001、20 个训练线程、4 个数值库线程；'
                'FastText 子词 1–3、200,000 桶。min_alpha=0.0001，按最多 15 轮线性下降，早停不重排日程。'
                '至少 3 轮；最近一轮或两轮验证 Spearman 增幅 >0.001 时继续，结合连续 3 轮无有效提升判断停止，'
                '最多 15 轮。每次运行独立选最佳轮数，保存逐轮指标、检查点和重载验证；不记录 loss。')
    scope = ('主选参指标为 WordSim240 验证集严格词表内 Spearman，覆盖 92/95 题；'
             '所有配置的词表、词频和覆盖题一致。三种子固定为 42、2026、7，SD 是样本标准差（ddof=1），'
             '反映训练波动，不能保证消除误差或替代独立测试。评价划分种子固定为 42，多线程不保证逐位确定性。'
             '本轮为小规模探索性选参，未计算测试指标，未启动全量训练或新闻类别预测。'
             'WordSim297/296 验证题不同，分别报告；类比只使用每套固定 128 题验证探针。'
             '学习率仅在各自选定训练方式下比较，尚未穷举训练方式交互。')
    lines = ['# THUCNews 小规模训练方式与超参数搜索', '<a id="thucnews-small-search"></a>',
             '## 搜索范围与当前状态', f'更新时间：{stamp()}。**{status}；学习率网格已登记 {len(rows)}/30 组**。',
             description, protocol, scope,
             '此前三档阶段比较0.01/0.025/0.05，共28次不同配置训练；0.01仅做种子42，'
             '0.025与0.05各有三种子。本轮补齐五档全部三种子，三档阶段模型和逐轮记录保留。',
             '[配置](../configs/thucnews_alpha_grid.yaml) · [三档历史记录](THUCNews学习率三档历史记录.md) · '
             '[历史全量记录](THUCNews训练方式消融实验.md)。', '## 抽样、清洗与评价覆盖' + matrix_text]
    for name, caption in matrix_receipt['captions'].items():
        if name.startswith('training_'):
            relative = Path(matrix_receipt['figures'][name]).relative_to('reports').as_posix()
            lines.append(f'![{caption}]({relative})')
    lines.extend(['## 五档学习率搜索', description, protocol])
    body = (f'按类别原比例、种子 42 固定抽取 **{stats["documents"]:,} 篇**完整训练文章，'
            f'共 {stats["tokens"]:,} 词；沿用既有清洗、jieba 分词和文章上下文，留出文章交集为 0。'
            f'所有配置主验证词对覆盖 **{coverage["covered"]}/{coverage["total"]}**。\n\n'
            '##### 1.1.3.1 训练方式搜索\n\n'
            '两个模型各比较 CBOW／Skip-gram × HS／NS5／NS10，固定初始 alpha=0.025。'
            '初筛种子 42；每模型基线和最佳新方案补 2026、7，训练方式阶段共 20 组。'
            '按三种子验证均值选中：' + '；'.join(
                f'{NAMES[m]}：**{LABELS[r["winner"]]}**（{r["comparisons"][0]["mean"]:.4f}±'
                f'{r["comparisons"][0]["sd"]:.4f}，样本 SD，n=3）' for m, r in previous['matrix_ranking'].items()) + '。')
    for name, caption in matrix_receipt['captions'].items():
        if name.startswith('training_'):
            relative = Path(matrix_receipt['figures'][name]).relative_to('reports').as_posix()
            body += f'\n\n![{caption}]({relative})'
    body += '\n\n##### 1.1.3.2 学习率搜索\n\n' + description + '\n\n' + protocol + '\n\n' + scope
    body += f'\n\n**{status}；已登记 {len(rows)}/30 组。**'
    auxiliary = []
    for i, method in enumerate(grid['training_methods'], 1):
        title = f'{NAMES[method]}：{LABELS[grid["training_methods"][method]]}'
        sections = [f'### {title}']
        comparisons = []
        for alpha in rates:
            group = sorted([r for r in rows if r['method'] == method and r['alpha'] == alpha],
                           key=lambda r: seeds.index(r['model_seed']))
            scores = {r['model_seed']: r['validation_score'] for r in group}
            full = len(group) == 3
            comparisons.append([alpha, *[f'{scores[s]:.4f}' if s in scores else '待完成' for s in seeds],
                                f'{statistics.mean(scores.values()):.4f}±{statistics.stdev(scores.values()):.4f}' if full else '待齐三种子',
                                len(group), '是' if complete and plan['selected'][method]['alpha'] == alpha else ''])
            if full:
                summaries = [read_json(ROOT / r['run'] / 'summary.json')['validation_metrics'] for r in group]
                for metric in summaries[0]:
                    key = metric['benchmark']
                    metrics = [next(m for m in ms if m['benchmark'] == key) for ms in summaries]
                    entry = {'method': method, 'alpha': alpha, 'benchmark': key, 'kind': metric['kind'], 'n': 3,
                             'covered': metric['covered'], 'total': metric['total']}
                    for field in ['pearson_r', 'spearman_rho', 'all_accuracy', 'conditional_accuracy',
                                  'generatable_spearman_rho', 'oov_ratio_percent']:
                        values = [m.get(field) for m in metrics]
                        entry[field + '_mean'] = statistics.mean(values) if all(v is not None for v in values) else None
                        entry[field + '_sd'] = statistics.stdev(values) if all(v is not None for v in values) else None
                    auxiliary.append(entry)
        sections.append(markdown_table(['初始 alpha', '种子42', '种子2026', '种子7', '验证均值±样本SD', 'n', '选中'], comparisons))
        if complete:
            choice = plan['selected'][method]
            rank = plan['alpha_ranking'][method]['comparisons']
            second = rank[1]
            sections.append(f'五档三种子均值选中 **alpha={choice["alpha"]}**：'
                            f'{choice["validation_mean"]:.4f}±{choice["validation_sd"]:.4f}，n=3；'
                            f'次高均值 alpha={second["choice"]} 为 {second["mean"]:.4f}，'
                            f'均值差 {choice["validation_mean"]-second["mean"]:.4f}。此差异为描述性比较，未作显著性检验。')
            if choice['validation_mean'] - second['mean'] < max(choice['validation_sd'], second['sd']):
                sections.append('前两档的均值差小于至少一档的种子标准差，当前排名用于候选筛选，不能据此声称显著优越。')
            if choice['alpha'] != grid['reference_alpha']:
                gains = []
                for seed in seeds:
                    selected_row = next(r for r in rows if r['method'] == method and r['alpha'] == choice['alpha'] and r['model_seed'] == seed)
                    reference_row = next(r for r in rows if r['method'] == method and r['alpha'] == grid['reference_alpha'] and r['model_seed'] == seed)
                    gains.append(selected_row['validation_score']-reference_row['validation_score'])
                sections.append('相对参考 alpha=0.025，同种子验证分差（42、2026、7）为 ' +
                                '、'.join(f'{v:+.4f}' for v in gains) + '；只描述这三次实测，不作为独立测试结论。')
            if choice['alpha'] in [rates[0], rates[-1]]:
                sections.append('最佳值位于当前搜索边界，只能称为五档中的最佳候选，不能据此判定全局最优。')
            lowest = [r for r in rows if r['method'] == method and r['alpha'] == rates[0]]
            capped = sum(r['trained_epochs'] == plan['configuration']['validation']['max_epochs'] for r in lowest)
            sections.append(f'最低学习率 alpha={rates[0]} 有 {capped}/3 次达到15轮预算上限。'
                            '这里比较的是固定日程及选轮预算下的表现，不能将低分解释为增加训练轮数后仍一定更差；'
                            '三种子也只反映训练波动，无法排除验证题抽样误差。')
            peaks = []
            for benchmark in ['wordsim297', 'wordsim296']:
                candidates = [r for r in auxiliary if r['method'] == method and r['benchmark'] == benchmark]
                best = max(candidates, key=lambda r: r['spearman_rho_mean'])
                peaks.append(f'{benchmark} 在 alpha={best["alpha"]} 的验证 Spearman 均值最高（{best["spearman_rho_mean"]:.4f}）')
            sections.append('辅助指标与主指标的趋势并不完全一致：' + '；'.join(peaks) +
                            '。类比探针趋势另见图。最终候选对应预设 WordSim240 目标，不能泛称对所有语义任务最优。')
        else:
            sections.append('等待全部五档、每档三种子完成后统一选参；缺失种子不补值、不宣布最终学习率。')
        if receipt:
            for name in [f'alpha_effect_{method}', f'alpha_curves_{method}', f'alpha_diagnostics_{method}']:
                relative = Path(receipt['figures'][name]).relative_to('reports').as_posix()
                sections.append(f'![{receipt["captions"][name]}]({relative})')
        text = '\n\n'.join(sections)
        lines.append(text)
        body += '\n\n' + text.replace('### ' + title, f'###### 1.1.3.2.{i} ' + title, 1)
    save_csv(experiment / 'rate_validation_metrics.csv', auxiliary)
    lines.extend(['## 辅助指标与运行证据',
                  f'[全部学习率诊断 CSV]({prefix}/rate_validation_metrics.csv)保留 Pearson/Spearman、'
                  '类比全部题/覆盖题正确率、OOV 和可生成词对诊断的三种子均值、样本 SD；原始 p 值保留逐轮，'
                  '不作跨种子平均。辅助指标不替换主选参指标。',
                  '时间不含建词表、验证和保存；RSS 为各运行进程观测峰值。复用的 FastText/0.025/种子7'
                  '曾从完整第2轮检查点恢复，RSS 不覆盖此前中断进程，详情见三档历史记录。'])
    for method in grid['training_methods']:
        lines.append('### ' + NAMES[method] + ' 运行明细')
        lines.append(markdown_table(['alpha', '种子', '最佳/训练轮数', '训练秒', '峰值RSS MiB', '最佳模型 MiB', '复用'],
                      [[r['alpha'], r['model_seed'], f'{r["selected_epoch"]}/{r["trained_epochs"]}',
                        f'{r["fit_seconds"]:.1f}', f'{r["peak_rss_mib"]:.1f}', f'{r["model_bytes"]/2**20:.1f}',
                        '是' if r['reused_from_previous'] else '否'] for r in rows if r['method'] == method]))
    for r in rows:
        p = '../' + r['run']
        lines.append(f'{r["method"]}/alpha={r["alpha"]}/seed={r["model_seed"]}：'
                     f'[日志]({p}/run.log) · [逐轮指标]({p}/validation_metrics.csv) · '
                     f'[轮次与学习率]({p}/epochs.csv) · [最佳模型]({p}/{r["checkpoint"]}) · '
                     f'[保存加载核验]({p}/save_load_check.json)。')
    lines.append(f'[完整运行索引]({prefix}/index.json) · [进度](../artifacts/thucnews_small_search/progress.json) · '
                 f'[原阶段索引](../{plan["source_experiment"]}/index.json)。原训练协议与旧索引保持不变。')
    if (experiment / 'results.csv').exists():
        lines.append(f'[全部30组原始结果 CSV]({prefix}/results.csv)保留完整精度及配置、种子、模型与检查点定位。')
    if (experiment / 'verification.json').exists():
        lines.append(f'[核验通过]({prefix}/verification.json)：30 组完整网格、每档三种子；封存文件、'
                     '逐轮选轮、检查点哈希、保存加载一致；全部使用验证分区，词表、词频及覆盖一致。')
    lines.append('## 复现入口\n\n```powershell\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage stop\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid --stage audit\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_alpha_grid --stage report\n```')
    lines.append(scope)
    REPORT.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    body += ('\n\n1.1.4 保留此前全量 CBOW＋HS 基线结果。本节候选尚未做全量确认，旧全量消融保持暂停。'
             '[完整搜索报告](THUCNews小规模超参数搜索.md) · [三档历史记录](THUCNews学习率三档历史记录.md)。')
    update_submission(body)
    write_json(experiment / 'report_status.json', {'status': plan['status'], 'search_runs': len(rows),
               'test_metric_rows': 0, 'figures_approved': receipt is not None, 'report_sha256': file_hash(REPORT),
               'history_sha256': file_hash(HISTORY), 'generated': stamp()})
    return REPORT
