"""Markdown evidence for the validation-only small-scale search."""
from pathlib import Path
import statistics

from .common import ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .thucnews_ablation_report import START, END, LABELS, number
from .thucnews_small_search import PHASES
from .submission_layout import place_hyperparameter_section

REPORT = ROOT / 'reports/THUCNews小规模超参数搜索.md'
STOP_LABELS = {'initial_no_upward_trend': '第3轮未持续上升',
               'patience_without_upward_trend': '无有效提升且不再上升', 'max_epochs': '达到15轮上限'}


def update_submission(body):
    path = ROOT / 'reports/Gensim&FastText.md'
    text = path.read_text(encoding='utf-8')
    marker = '### 1.2 文本分类实验'
    assert text.count(marker) == 1
    tail = text.split(marker, 1)[1]
    path.write_text(place_hyperparameter_section(text, body), encoding='utf-8')
    assert path.read_text(encoding='utf-8').split(marker, 1)[1] == tail


def report(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    if plan.get('kind') == 'full_alpha_grid':
        from .thucnews_alpha_grid_report import report as grid_report
        return grid_report(experiment)
    assert not plan['smoke'], 'Smoke must not overwrite formal reports'
    cfg, rows = plan['configuration'], plan['runs']
    amendment_path = experiment / 'scope_amendment.json'
    amendment = read_json(amendment_path) if amendment_path.exists() else {}
    deferred = plan.get('learning_rate_search_deferred', amendment.get('learning_rate_search_deferred', False))
    expected = 20 if deferred else 28
    manifest = read_json(ROOT / plan['manifest'])
    stats, coverage = manifest['partitions']['train'], manifest['primary_validation']
    prefix = '../' + experiment.relative_to(ROOT).as_posix()
    status = {'running': '进行中', 'complete': '完成', 'paused': '已暂停', 'failed': '失败'}[plan['status']]
    current = f'{status}，{PHASES[plan["phase"]]}；已完成 {len(rows)}/{expected} 次训练'
    lines = ['# THUCNews 小规模训练方式与超参数搜索', '<a id="thucnews-small-search"></a>',
             '## 搜索范围与当前状态', f'更新时间：{stamp()}。**{current}**。',
             ('本轮在固定 5 万篇上比较 Word2Vec、FastText 的六种训练方式，完成基线与最佳新方案的三种子复验。'
              '按用户要求，FastText 复验结束后停止，学习率固定为 0.025，未开展学习率搜索。' if deferred else
              '本轮在固定 5 万篇上重新搜索 Word2Vec、FastText 的训练方式与学习率。') +
             '所有模型从头训练，不继承旧全量模型；当前只做表示学习与验证选参，没有新闻类别预测。'
             '此前全量搜索保持暂停，4 组完整结果和第 5 组的 8 轮检查点保留在'
             '[历史记录](THUCNews训练方式消融实验.md)。',
             '## 抽样、清洗与评价覆盖',
             f'从清洗去重后的 821,609 篇训练文章中，按 14 个新闻类别的原比例抽取 {stats["documents"]:,} 篇；'
             f'包含 {stats["sentences"]:,} 个句子/分块、{stats["tokens"]:,} 词。'
             '各类别配额按最大余数法确定；类别内按文章 ID 与种子 42 的 SHA-256 排名选择。'
             '只抽已有训练文章，和 5,000 篇留出文章的 ID 交集为 0。保留完整文章的句子及原分块顺序，'
             '沿用已有 NFKC、HTML/URL/控制字符清洗和 jieba 分词，不重复清洗、不删除停用词，不跨文章连接上下文。',
             f'在 min_count=5 下有 {manifest["vocabulary_size"]:,} 个词。WordSim240 验证覆盖 '
             f'**{coverage["covered"]}/{coverage["total"]}** 对，覆盖率 {coverage["coverage"]:.2%}；'
             '同一词表和同一组严格词表内词对用于所有搜索配置。FastText 的 OOV 子词生成分数另列为诊断，'
             '不会替换 Word2Vec 无法覆盖的题来参与选参。抽样不依据评价词、人工分数或测试题。'
             '覆盖题与全量不同，当前小规模分数不直接和旧全量分数相减。',
             f'[抽样清单](../{manifest["directory"]}/selected_articles.csv) · '
             f'[子集清单](../{plan["manifest"]}) · [配置](../configs/thucnews_small_search.yaml)。',
             markdown_table(['类别', '全量训练篇数', '本轮抽样篇数'],
                 [[c, read_json(ROOT / cfg['source_manifest'])['categories'][c]['train'], n]
                  for c, n in manifest['categories'].items()]),
             '## 逐轮验证、上升判断与检查点',
             '先训练 3 轮。“仍明显上升”具体定义为：当前主指标相比上一轮或两轮前增幅 **>0.001**。'
             '第 3 轮若两种增幅均未达阈值，停止该配置；仍上升则继续。随后当最佳有效提升参照连续 3 轮'
             '未提高超过 0.001，且最近一轮和两轮均无上升时停止，最多 15 轮。'
             '0.001 是计算预算规则，不代表统计显著性；仍有上升趋势的恢复曲线不会仅因落后于历史峰值被停止。'
             '所有轮次中实际最高验证分都能成为最佳检查点，即使增幅小于阈值。',
             '学习率按最多 15 轮从指定 alpha 线性降至 0.0001；早停不重排已用学习率。'
             'Word2Vec、FastText 分别判断上升并分别选轮数，不强制使用相同 epoch。'
             '每轮保存原生检查点、逐题评价、Pearson、Spearman、类比正确率、覆盖率、耗时和上升/停止理由。'
             '不记录训练 loss；最终选中模型重载后重新验证，核对预测指标一致。',
             '## 训练方式搜索与三种子复验',
             '第一阶段：固定 100 维、window=5、min_count=5、sample=0.001、alpha=0.025，'
             '两个模型各运行 CBOW／Skip-gram × HS／NS5／NS10 六组，共 12 次。'
             '每个模型的 CBOW＋HS 基线和最佳新方案再补 2026、7 两种子，共 20 次，按三种子验证均值选训练方式。',
             ('本轮实际范围为 **20 次独立训练**。原计划的 alpha=0.01、0.025、0.05 搜索暂缓，'
              '没有运行新的学习率配置；0.025 是固定值，不能称为最优学习率。' if deferred else
             '第二阶段：在每个模型选中的训练方式下比较 alpha=0.01、0.025、0.05。'
             '0.025 的三种子结果复用本轮已完成模型；其余两个学习率先各做种子 42，共新增 4 次。'
             '每个模型选最佳新学习率，再补 2026、7 共新增 4 次，与 0.025 的三种子均值比较。'
             '总计 **28 次独立训练**，最终学习率完全同分时保留 0.025。仅两个学习率有三种子，'
             '未复验学习率的单次分数不能解释为稳定排名。'),
             'FastText 字符子词 1–3、桶 200,000 固定；本轮不混入子词、桶数、窗口和维度搜索。'
             '20 个训练线程、4 个数值库线程；固定数据、评价种子 42 和模型种子 42/2026/7，'
             '多线程不保证逐位确定性。TF-IDF 保留统计对照。',
             '## 训练方式初筛']
    matrix = [r for r in rows if r['search_phase'] == 'matrix_screen']
    headers = ['模型', '训练方式', '验证 Spearman', '最佳/实际轮数', '停止原因', '训练秒']
    lines.append(markdown_table(headers, [[r['method'], LABELS[r['combination']], number(r['validation_score']),
                 f'{r["selected_epoch"]}/{r["trained_epochs"]}', STOP_LABELS.get(r['stop_reason'], r['stop_reason']),
                 f'{r["fit_seconds"]:.1f}'] for r in matrix]))
    lines.append('### 训练速度、内存与模型大小')
    lines.append(markdown_table(['模型', '训练方式', '每轮秒', '峰值 RSS MiB', '最佳原生模型 MiB'],
        [[r['method'], LABELS[r['combination']], f'{r["fit_seconds"]/r["trained_epochs"]:.1f}',
          f'{r["peak_rss_mib"]:.1f}', f'{r["model_bytes"]/2**20:.1f}'] for r in matrix]))
    lines.append('训练时间不含建词表、评价、检查点保存和封存；峰值 RSS 统计训练子进程，'
                 '模型大小仅统计最佳原生模型，不包含所有轮次检查点总占用。速度为本机本次运行实测，'
                 '不是独立硬件性能基准。')
    if 'matrix_ranking' in plan:
        lines.extend(['## 训练方式三种子比较', markdown_table(['模型', '训练方式', '验证均值', '样本 SD', 'n', '选中'],
            [[m, LABELS[c['choice']], number(c['mean']), number(c['sd']), c['n'], '是' if c['choice'] == rank['winner'] else '否']
             for m, rank in plan['matrix_ranking'].items() for c in rank['comparisons']])])
    lines.append('## 学习率搜索状态')
    if deferred:
        lines.append('按用户要求暂缓。全部 20 次训练初始 alpha=0.025，min_alpha=0.0001；未比较其他学习率。')
    elif 'matrix_ranking' in plan:
        alpha_rows = [r for r in rows if r['model_seed'] == cfg['screen_seed']
                      and r['combination'] == plan['matrix_ranking'][r['method']]['winner']]
        lines.append('下表为种子 42、各自最佳轮次的初筛分数，每配置 n=1；三种子均值另列。')
        lines.append(markdown_table(['模型', '训练方式', 'alpha', '验证 Spearman', '最佳/实际轮数', '训练秒'],
            [[r['method'], LABELS[r['combination']], r['alpha'], number(r['validation_score']),
              f'{r["selected_epoch"]}/{r["trained_epochs"]}', f'{r["fit_seconds"]:.1f}'] for r in alpha_rows]))
    else:
        lines.append('训练方式复验完成后开始学习率搜索。')
    if 'alpha_ranking' in plan:
        lines.append(markdown_table(['模型', 'alpha', '验证均值', '样本 SD', 'n', '选中'],
            [[m, c['choice'], number(c['mean']), number(c['sd']), c['n'], '是' if c['choice'] == rank['winner'] else '否']
             for m, rank in plan['alpha_ranking'].items() for c in rank['comparisons']]))
    lines.extend(['## 当前选择与结论边界'])
    if 'selected' in plan:
        for method, selected in plan['selected'].items():
            lines.append(f'**{method}：{LABELS[selected["combination"]]}，alpha={selected["alpha"]}**；'
                         f'验证均值 {selected["validation_mean"]:.4f}，样本 SD={selected["validation_sd"]:.4f}，n=3。')
        lines.append('### 选中配置的其他验证指标')
        auxiliary = []
        for method, selected in plan['selected'].items():
            group = [r for r in rows if r['method'] == method and r['combination'] == selected['combination']
                     and r['alpha'] == selected['alpha']]
            assert len(group) == 3
            summaries = [read_json(ROOT / r['run'] / 'summary.json') for r in group]
            for benchmark in read_json(ROOT / cfg['benchmark_manifest'])['benchmarks']:
                metrics = [next(m for m in summary['validation_metrics'] if m['benchmark'] == benchmark) for summary in summaries]
                entry = {'method': method, 'benchmark': benchmark, 'kind': metrics[0]['kind'], 'n': 3,
                         'covered': metrics[0]['covered'], 'total': metrics[0]['total']}
                for field in ['pearson_r', 'spearman_rho', 'all_accuracy', 'conditional_accuracy',
                              'generatable_spearman_rho', 'oov_ratio_percent']:
                    values = [m.get(field) for m in metrics]
                    entry[field + '_mean'] = statistics.mean(values) if all(v is not None for v in values) else None
                    entry[field + '_sd'] = statistics.stdev(values) if all(v is not None for v in values) else None
                auxiliary.append(entry)
        from .evaluation_support import save_csv
        save_csv(experiment / 'selected_validation_metrics.csv', auxiliary)
        lines.append(markdown_table(['模型', '词对基准', 'Pearson均值±SD', 'Spearman均值±SD', '覆盖/题数'],
            [[r['method'], r['benchmark'], number(r['pearson_r_mean']) + '±' + number(r['pearson_r_sd']),
              number(r['spearman_rho_mean']) + '±' + number(r['spearman_rho_sd']), f'{r["covered"]}/{r["total"]}']
             for r in auxiliary if r['kind'] == 'similarity']))
        lines.append(markdown_table(['模型', '类比验证探针', '全部题正确率均值±SD', '覆盖题正确率均值±SD', '覆盖/题数'],
            [[r['method'], r['benchmark'], number(r['all_accuracy_mean']) + '±' + number(r['all_accuracy_sd']),
              number(r['conditional_accuracy_mean']) + '±' + number(r['conditional_accuracy_sd']), f'{r["covered"]}/{r["total"]}']
             for r in auxiliary if r['kind'] == 'analogy']))
        lines.append(f'以上为各自所选轮数下三种子验证均值±样本 SD，n=3；类比每套固定 128 题探针，'
                     '并非全量类比测试。WordSim297 与 WordSim296 的验证题不同，分别报告。'
                     f'[完整诊断 CSV]({prefix}/selected_validation_metrics.csv)保留 OOV、子词可生成词对等指标；'
                     '逐题和每次训练的原始 p 值保留于逐轮记录，不把多个种子的 p 值作简单平均。')
    else:
        lines.append('搜索尚未结束，未宣布最终超参数。')
    lines.append('本轮 **不计算测试指标，也不自动启动全量训练**。固定验证基准曾被历史实验观察，'
                 '本轮为探索性选参；三种子只描述训练波动。小规模胜出配置未必在全量胜出，'
                 '下一步应在更大语料复核少量候选，最终全量从头训练并重新选择轮数。'
                 '只对基线与初筛最佳新训练方式做三种子复验，未复验的方案不构成稳定排名。'
                 + ('学习率保持固定，尚不能判断训练方式与学习率的交互。' if deferred else
                    '学习率仅在各自选中的训练方式下搜索，未穷举训练方式与学习率的全部交互。'))
    lines.append('## 验证曲线与对比图')
    approved = experiment / 'figures/approved.json'
    receipt = read_json(approved) if approved.exists() else None
    matrix_rows = [r for r in rows if r['search_phase'].startswith('matrix_')]
    matrix_receipt = (receipt and receipt['data_fingerprint'] == fingerprint(matrix_rows)
                      and all(name.startswith('training_') for name in receipt['captions']))
    if receipt and (receipt['data_fingerprint'] == fingerprint(rows) or matrix_receipt):
        assert all(file_hash(ROOT / p) == sha for p, sha in receipt['files'].items())
        for name, caption in receipt['captions'].items():
            relative = Path(receipt['figures'][name]).relative_to('reports').as_posix()
            lines.append(f'![{caption}]({relative})')
    else:
        receipt = None
        lines.append('完整结果出图后执行数据剖析、机器布局检查及彩色/灰度视觉复核，再嵌入本报告。')
    lines.extend(['## 运行证据与复现', f'[搜索 CSV]({prefix}/search_results.csv) · [运行索引]({prefix}/index.json) · '
                  '[进度](../artifacts/thucnews_small_search/progress.json)。'])
    recovery_path = ROOT / 'artifacts/thucnews_small_search/recovery.json'
    if recovery_path.exists():
        lines.append('FastText/CBOW＋NS5 的第 1 轮完成后，Windows 替换进度文件发生一次 WinError 5，'
                     '该轮模型与状态已完整保存。核对检查点 SHA-256 后按相同数据、参数和训练代码从第 2 轮恢复，'
                     '未把文件错误作为模型失败或重新选参。实时监控改用允许删除共享的读取方式。'
                     '该配置的 RSS 为恢复进程观测峰值；此前进程峰值未独立保存，不能声称已合并两个进程的峰值。'
                     '[恢复记录](../artifacts/thucnews_small_search/recovery.json)。')
    if (experiment / 'last_replication_resume.json').exists():
        lines.append('矩阵阶段最后一组 FastText／Skip-gram＋NS10／种子 7 曾中断于第 3 轮中途。'
                     '核对已保存第 2 轮检查点后，丢弃未保存的第 3 轮中间状态，从第 3 轮完整重跑；'
                     '数据、参数、选轮规则与训练代码未改变。峰值 RSS 为恢复进程观测值，'
                     '不能声称覆盖此前中断进程的峰值。'
                     f'[续训记录]({prefix}/last_replication_resume.json)。')
    for row in rows:
        p = '../' + row['run']
        lines.append(f'{row["method"]}/{row["combination"]}/alpha{row["alpha"]}/seed{row["model_seed"]}：'
                     f'[日志]({p}/run.log) · [逐轮指标]({p}/validation_metrics.csv) · '
                     f'[上升判定]({p}/epochs.csv) · [最佳模型]({p}/{row["checkpoint"]}) · '
                     f'[保存加载核验]({p}/save_load_check.json)。')
    if (experiment / 'verification.json').exists():
        verified = read_json(experiment / 'verification.json')['search_runs']
        lines.append(f'[核验通过]({prefix}/verification.json)：已核验 {verified} 次训练，无测试评价，词表与评价覆盖一致，'
                     '选轮规则复算、所有检查点哈希及保存加载验证通过。')
    if (experiment / 'alpha_stage_verification.json').exists():
        lines.append(f'[学习率阶段核验]({prefix}/alpha_stage_verification.json)：新增 4 次初筛与 4 次复验，'
                     '原矩阵阶段 20 次结果逐项保留，训练协议未改变，三种子学习率选择复算一致。'
                     f'[矩阵阶段快照]({prefix}/matrix_stage_snapshot/index.json)保存继续搜索前的原索引。')
    if deferred:
        lines.append('当前使用限定范围入口，仅续训最后一组或复用已完成结果，不进入学习率阶段：\n\n'
                     '```powershell\nconda run -n Gensim python -m gensim_lab.thucnews_small_matrix\n'
                     'conda run -n Gensim python -m gensim_lab.thucnews_small_matrix --stage audit\n'
                     'conda run -n Gensim python -m gensim_lab.thucnews_small_matrix --stage report\n```')
    else:
        lines.append('```powershell\nconda run -n Gensim python -m gensim_lab.thucnews_small_search --stage prepare\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search --smoke\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage stop\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage audit\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_small_search --stage report\n```')
    REPORT.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    body = (f'按类别原比例、种子 42 从完整训练集固定抽取 **{stats["documents"]:,} 篇**，'
            f'共 {stats["tokens"]:,} 词。保留已有清洗、分词及完整文章上下文；留出文章交集为 0。'
            f'主验证词对覆盖 **{coverage["covered"]}/{coverage["total"]}**，所有配置使用同一覆盖口径。\n\n'
            '##### 1.1.3.1 训练方式搜索\n\n'
            'Word2Vec、FastText 分别比较六种训练方式；先跑 3 轮，最近一轮或两轮 Spearman 增幅 >0.001 '
            '则继续，结合连续 3 轮无有效提升判断停止，最多 15 轮，每轮保留验证指标与检查点。'
            + ('按三种子验证均值选训练方式，总计 20 次训练。FastText 复验结束后停止；'
             '学习率固定为 0.025，暂不搜索学习率。\n\n' if deferred else
             '按三种子验证均值选训练方式，再搜索 alpha=0.01/0.025/0.05 并复验，总计 28 次训练。\n\n') +
            f'**{current}**。')
    if 'matrix_ranking' in plan:
        body += '\n\n训练方式三种子复验（初始 alpha=0.025）：' + '；'.join(
            f'{m}：**{LABELS[rank["winner"]]}**（验证均值 {rank["comparisons"][0]["mean"]:.4f}'
            f'±{rank["comparisons"][0]["sd"]:.4f}，样本 SD，n=3）'
            for m, rank in plan['matrix_ranking'].items()) + '。'
    if receipt:
        for name, caption in receipt['captions'].items():
            if not name.startswith('training_'):
                continue
            relative = Path(receipt['figures'][name]).relative_to('reports').as_posix()
            body += f'\n\n![{caption}]({relative})'
    if not deferred:
        body += ('\n\n##### 1.1.3.2 学习率搜索\n\n'
                 '固定上节选中的训练方式，比较初始 alpha=0.01、0.025、0.05；'
                 'min_alpha=0.0001，按最多 15 轮线性下降，早停不重排日程。'
                 '0.025 复用已有三种子结果，其他两个学习率先做种子 42 初筛，'
                 '每个模型最佳新学习率补种子 2026、7，再按三种子验证均值与 0.025 比较。'
                 '新增 8 次训练；未复验的学习率仅有单次结果，不能解释为稳定排名。')
        alpha_screen = [r for r in rows if r['model_seed'] == cfg['screen_seed']
                        and 'matrix_ranking' in plan
                        and r['combination'] == plan['matrix_ranking'][r['method']]['winner']]
        if alpha_screen:
            body += '\n\n初筛对比（种子 42，每配置 n=1，各自最佳轮次）：\n\n' + markdown_table(['模型', 'alpha', '验证 Spearman', '最佳/实际轮数'],
                [[r['method'], r['alpha'], number(r['validation_score']),
                  f'{r["selected_epoch"]}/{r["trained_epochs"]}'] for r in alpha_screen])
        if 'alpha_ranking' in plan:
            body += '\n\n' + markdown_table(['模型', 'alpha', '验证均值', '样本 SD', 'n', '选中'],
                [[m, c['choice'], number(c['mean']), number(c['sd']), c['n'],
                  '是' if c['choice'] == rank['winner'] else '否']
                 for m, rank in plan['alpha_ranking'].items() for c in rank['comparisons']])
        if 'selected' in plan:
            body += '\n\n三种子比较选中：' + '；'.join(
                f'{m}：**{LABELS[s["combination"]]}、alpha={s["alpha"]}**'
                f'（验证均值 {s["validation_mean"]:.4f}±{s["validation_sd"]:.4f}，样本 SD，n=3）'
                for m, s in plan['selected'].items()) + '。'
        else:
            body += '\n\n学习率搜索进行中，尚未冻结最终学习率。'
        if receipt:
            for name, caption in receipt['captions'].items():
                if not name.startswith('alpha_'):
                    continue
                relative = Path(receipt['figures'][name]).relative_to('reports').as_posix()
                body += f'\n\n![{caption}]({relative})'
    body += ('\n\n当前是小规模探索性选参，未计算本轮测试指标，未开展全量确认。'
             '1.1.4 保留此前全量 CBOW＋HS 基线的训练结果；本节候选的全量训练尚未开展。'
             '旧全量消融保持暂停；没有新增类别预测。'
             '[完整搜索报告](THUCNews小规模超参数搜索.md) · [历史全量记录](THUCNews训练方式消融实验.md)。')
    update_submission(body)
    write_json(experiment / 'report_status.json', {'generated': stamp(), 'status': plan['status'],
               'search_runs': len(rows), 'test_metric_rows': 0, 'figures_approved': receipt is not None,
               'report_sha256': file_hash(REPORT)})
    return REPORT
