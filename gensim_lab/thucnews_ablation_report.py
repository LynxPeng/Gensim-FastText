"""Markdown reports from the isolated THUCNews factorial experiment index."""
from pathlib import Path

from .common import ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from .evaluation_support import save_csv
from .submission_layout import place_hyperparameter_section

REPORT = ROOT / 'reports/THUCNews训练方式消融实验.md'
START = '<!-- thucnews-ablation:start -->'
END = '<!-- thucnews-ablation:end -->'
LABELS = {'cbow_hs': 'CBOW＋HS', 'cbow_ns5': 'CBOW＋NS5', 'cbow_ns10': 'CBOW＋NS10',
          'sg_hs': 'Skip-gram＋HS', 'sg_ns5': 'Skip-gram＋NS5', 'sg_ns10': 'Skip-gram＋NS10'}


def number(value):
    return '不适用' if value is None else f'{value:.4f}'


def update_submission(body):
    path = ROOT / 'reports/Gensim&FastText.md'
    text = path.read_text(encoding='utf-8')
    marker = '### 1.2 文本分类实验'
    assert text.count(marker) == 1
    tail = text.split(marker, 1)[1]
    path.write_text(place_hyperparameter_section(text, body), encoding='utf-8')
    assert path.read_text(encoding='utf-8').split(marker, 1)[1] == tail, 'Do not change user chapter 1.2 or later'


def figure_receipt(experiment, rows):
    path = experiment / 'figures/approved.json'
    if not path.exists():
        return None
    receipt = read_json(path)
    if receipt['data_fingerprint'] != fingerprint(rows):
        return None
    if not all((ROOT / p).exists() and file_hash(ROOT / p) == h for p, h in receipt['files'].items()):
        return None
    return receipt


def report(experiment):
    experiment = Path(experiment)
    plan = read_json(experiment / 'index.json')
    assert not plan['smoke'], 'Smoke artifacts must not replace the full experiment report'
    cfg, rows = plan['configuration'], plan['runs']
    prefix = '../' + experiment.relative_to(ROOT).as_posix()
    phase = {'screen': '种子 42 初筛', 'replication': '三种子复验', 'frozen_test': '冻结后的统一测试',
             'complete': '完成'}[plan['phase']]
    lines = ['# THUCNews 训练方式消融与对比实验', '<a id="thucnews-training-ablation"></a>',
             '## 研究问题与运行状态',
             '比较 Word2Vec 与 FastText 的上下文预测方式（CBOW／Skip-gram）及训练目标（HS／NS5／NS10）。'
             '两因素构成 2×3 小矩阵，每个模型 6 组；NS5→NS10 比较负例数的收益与成本。'
             '本轮只做词向量表示学习和内在评价，没有训练新闻类别预测器。',
             f'更新时间：{stamp()}。当前阶段：**{phase}**；已完成 **{len(rows)}/20** 次正式训练。'
             '本页只纳入已封存且重载验证通过的结果，未完成项不填充猜测数值。',
             '## 固定条件与评价隔离',
             '沿用完整 THUCNews 清洗与去重后的 821,609 篇训练文章、378,641,761 词。'
             '5,000 篇文章只用于向量覆盖率诊断。清洗、jieba 分词、分句和文章顺序保持原协议：'
             '[数据处理说明](THUCNews词向量实验.md#数据与预处理)。新闻类别标签不参与训练或选参。',
             '100 维、window=5、min_count=5、sample=0.001、20 个训练线程、4 个数值库线程，'
             'batch_words=9,000。学习率 0.025→0.0001 按最多 15 轮线性调度；至少训练 5 轮，'
             '验证主指标连续 3 轮未超过有效提升参照 0.001 时停止。更高的小幅分数仍更新最佳检查点。'
             'FastText 固定 min_n=1、max_n=3、bucket=200,000；不记录训练 loss。',
             '额外显式固定 ns_exponent=0.75、CBOW 上下文均值、动态窗口（shrink_windows=True）和频次排序词表。'
             '所有方案从头训练，不沿用已训练基线权重。每轮核对完整原始词数，并保存原生检查点与逐题验证结果。'
             '多线程更新不保证逐位确定性，种子重复也包括线程调度差异。',
             f'**评价划分种子固定为 {cfg["split_seed"]}，训练种子独立变化为 42、2026、7。**'
             '直接锁定已有去重评价分区的路径和 SHA-256；训练种子不会重新分配题目。'
             'WordSim240 使用相同 95 个验证词对，主指标为严格词表内 Spearman ρ。'
             '三个词对基准逐轮计算 Pearson／Spearman／覆盖率；类比使用每套固定 128 题验证探针作辅助诊断。'
             '正式测试使用全部测试题；辅助指标不参与配置或检查点选择。',
             f'[固定分区](../{cfg["benchmark_manifest"]}) · [配置](../configs/thucnews_ablation.yaml) · '
             '[代码](../gensim_lab/thucnews_ablation.py)。',
             '## 搜索、复验与冻结规则',
             '1. 种子 42 运行两个模型各 6 组，共 12 次，搜索阶段只计算验证指标。\n'
             '2. 每个模型在五个新方案中选验证分最高者；与 CBOW＋HS 基线各补 2026、7 两种子，共增加 8 次训练。\n'
             '3. 对基线和新方案分别计算三个种子的验证均值与样本标准差（ddof=1），按均值选最终训练方式。'
             '完全同分时优先基线，初筛新方案同分按配置名称稳定排序。\n'
             '4. 封存全部选择及检查点哈希后，统一评价两模型的基线和候选各三个种子，共 12 个最佳检查点。'
             '比较两方案测试分数用于描述效果，不反过来改选配置。',
             '每次训练独立按验证选择轮数，因此比较的是“训练方式＋相同选轮规则”的效果，'
             '各方案实际轮数可能不同；训练时间同时报告每轮与累计值，不能把累计时间差全部归因于算法。'
             'TF-IDF 保留已有统计对照；其无词向量，未用 WordSim 主指标搜索过滤参数。',
             '## 种子 42 初筛结果']
    screen = [r for r in rows if r['model_seed'] == cfg['screen_seed']]
    lines.append(markdown_table(['模型', '训练方式', '验证 Spearman', '最佳/实际轮数', '训练秒', '每轮秒', '峰值 RSS MiB', '原生模型 MiB'], [
        [r['method'], LABELS[r['combination']], number(r['validation_score']), f'{r["selected_epoch"]}/{r["trained_epochs"]}',
         f'{r["fit_seconds"]:.1f}', f'{r["fit_seconds"]/r["trained_epochs"]:.1f}', f'{r["peak_rss_mib"]:.1f}',
         f'{r["model_bytes"]/2**20:.1f}'] for r in screen]))
    if len(screen) == 12:
        lines.append('### 算法、训练目标与负例数的对比')
        for method in cfg['models']:
            group = {r['combination']: r for r in screen if r['method'] == method}
            best = max(group.values(), key=lambda r: r['validation_score'])
            differences = [group[f'sg_{objective}']['validation_score'] - group[f'cbow_{objective}']['validation_score']
                           for objective in ['hs', 'ns5', 'ns10']]
            lines.append(f'{method} 的种子 42 初筛最高分为 {LABELS[best["combination"]]}，'
                         f'ρ={best["validation_score"]:.4f}，相对 CBOW＋HS 为 '
                         f'{best["validation_score"]-group["cbow_hs"]["validation_score"]:+.4f}。'
                         '在 HS、NS5、NS10 下，Skip-gram 减 CBOW 的验证差值分别为 '
                         + '、'.join(f'{v:+.4f}' for v in differences) + '。'
                         f'负例从 5 增至 10 时，CBOW 的差值为 '
                         f'{group["cbow_ns10"]["validation_score"]-group["cbow_ns5"]["validation_score"]:+.4f}，'
                         f'Skip-gram 为 {group["sg_ns10"]["validation_score"]-group["sg_ns5"]["validation_score"]:+.4f}。'
                         '这些是单次、独立选轮后的描述差异，不能据此判断统计显著性或固定轮次下的纯算法效应。')
    if (experiment / 'screen_selection.json').exists():
        candidates = read_json(experiment / 'screen_selection.json')['candidates']
        lines.append('进入复验的新方案：' + '；'.join(f'{m}：**{LABELS[c]}**' for m, c in candidates.items()) + '。')
    else:
        lines.append('12 组初筛未全部完成，尚未选择新方案。')
    lines.append('## 三种子复验')
    reps = [r for r in rows if r['model_seed'] != cfg['screen_seed']]
    if reps:
        lines.append(markdown_table(['模型', '训练方式', '训练种子', '验证 Spearman', '最佳/实际轮数', '训练秒'], [
            [r['method'], LABELS[r['combination']], r['model_seed'], number(r['validation_score']),
             f'{r["selected_epoch"]}/{r["trained_epochs"]}', f'{r["fit_seconds"]:.1f}'] for r in reps]))
    else:
        lines.append('尚未进入复验；初筛单次结果不报告标准差或显著性。')
    frozen = read_json(experiment / 'frozen.json') if (experiment / 'frozen.json').exists() else None
    if frozen:
        lines.append(markdown_table(['模型', '训练方式', '验证均值', '样本 SD', 'n', '选中'], [
            [method, LABELS[r['combination']], number(r['mean']), number(r['sd']), r['n'],
             '是' if r['combination'] == rank['winner'] else '否']
            for method, rank in frozen['ranking'].items() for r in rank['comparisons']]))
        lines.append(f'[冻结决定]({prefix}/frozen.json)。只对复验的两个方案做均值选择，'
                     '不能据此断言未复验的四个训练方式更差。n=3 反映训练随机波动，不用于声称总体显著性。'
                     '种子 42 已参与候选筛选，也计入三种子均值；2026、7 是新增训练重复，仍使用同一验证题，'
                     '三种子均值不是独立评价数据上的确认性证据。')
    else:
        lines.append('尚未冻结配置，**本轮未计算测试指标**。')
    test_rows = []
    lines.append('## 冻结后的统一测试')
    for path in sorted((experiment / 'tests').glob('*/summary.json')):
        item = read_json(path)
        for metric in item['metrics']:
            test_rows.append({'method': item['method'], 'combination': item['combination'], 'model_seed': item['model_seed'],
                              'is_validation_winner': item['is_validation_winner'], **{k: v for k, v in metric.items() if k != 'sections'}})
    if test_rows:
        aggregate = []
        import statistics
        for method in cfg['models']:
            combinations = [cfg['baseline'], plan['candidates'][method]]
            for combination in combinations:
                for benchmark in ['wordsim240', 'wordsim297', 'wordsim296', 'CA_translated', 'CA8_morphological', 'CA8_semantic']:
                    group = [r for r in test_rows if r['method'] == method and r['combination'] == combination and r['benchmark'] == benchmark]
                    if len(group) != 3:
                        continue
                    field = 'spearman_rho' if group[0]['kind'] == 'similarity' else 'all_accuracy'
                    values = [r[field] for r in group]
                    if all(v is not None for v in values):
                        aggregate.append({'method': method, 'combination': combination, 'benchmark': benchmark,
                                          'metric': field, 'n': 3, 'mean': statistics.mean(values), 'sd': statistics.stdev(values)})
        lines.append(markdown_table(['模型', '训练方式', '测试基准', '指标', '均值', '样本 SD', 'n'], [
            [r['method'], LABELS[r['combination']], r['benchmark'], r['metric'], number(r['mean']), number(r['sd']), r['n']]
            for r in aggregate]))
        if aggregate:
            save_csv(experiment / 'test_aggregate.csv', aggregate)
        lines.append(markdown_table(['模型', '训练方式', '种子', '测试基准', '覆盖/题数', 'Spearman/全部题正确率'], [
            [r['method'], LABELS[r['combination']], r['model_seed'], r['benchmark'], f'{r["covered"]}/{r["total"]}',
             number(r.get('spearman_rho') if r['kind'] == 'similarity' else r.get('all_accuracy'))] for r in test_rows]))
        save_csv(experiment / 'test_results.csv', test_rows)
    else:
        lines.append('等待 20 次训练和配置冻结，不显示历史测试分数作为本轮结果。')
    lines.extend(['## 结果图与解释范围',
        '初筛为每模型 2×3 矩阵，适合带数值标注的热力图；逐轮指标适合折线。'
        '三种子比较展示每次实测值和均值±样本 SD；不连接无序训练方式、不做曲线平滑，不用均值柱遮住 n=3。'])
    receipt = figure_receipt(experiment, rows)
    if receipt:
        for name, caption in receipt['captions'].items():
            path = receipt['figures'][name]
            relative = Path(path).relative_to('reports').as_posix()
            lines.append(f'![{caption}]({relative})')
    else:
        lines.append('完整实验图在数据剖析、机器布局检查及彩色／灰度视觉复核后发布；当前数据可从原始 CSV 追溯。')
    lines.extend(['**结论边界：探索性选参。** 这些基准的完整结果曾被历史实验观察，本轮固定分区和测试隔离'
                  '避免当前搜索再用测试分数选参，但不能恢复未观察过的盲测条件。验证基准较小且尝试多个配置，'
                  '最高分存在选择偏差；三种子复验不能消除基准偏差。新的确认性结论需要独立评价数据。',
                  '学习率敏感性列为本矩阵后的独立后续阶段，暂不混入本次最多 20 次训练的结果。'
                  '子词范围、桶数、窗口与维度也未在本矩阵中改变。',
                  '## 日志、检查点与复现',
                  f'[搜索汇总 CSV]({prefix}/search_results.csv) · [编排索引]({prefix}/index.json) · '
                  '[进度](../artifacts/thucnews_ablation/progress.json)。'])
    for row in rows:
        run_prefix = '../' + row['run']
        lines.append(f'{row["method"]}/{row["combination"]}/seed{row["model_seed"]}：'
                     f'[日志]({run_prefix}/run.log) · [逐轮指标]({run_prefix}/validation_metrics.csv) · '
                     f'[逐轮记录]({run_prefix}/epochs.csv) · [所选原生模型]({run_prefix}/{row["checkpoint"]}) · '
                     f'[保存加载核验]({run_prefix}/save_load_check.json)。')
    if (experiment / 'verification.json').exists():
        lines.append(f'[实验核验]({prefix}/verification.json)：固定分区无重叠、20 次搜索无测试指标、'
                     '词表及频次一致、逐轮选轮规则复算、全部检查点哈希和冻结后的测试资源核对通过。')
    lines.append('```powershell\nconda run -n Gensim python -m gensim_lab.thucnews_ablation --smoke\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_ablation\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage audit\n'
                 'conda run -n Gensim python -m gensim_lab.thucnews_ablation --stage report\n```')
    lines.append('同一指纹再次运行时复用封存结果；中断训练从最后完整原生检查点恢复，'
                 '数据／参数／训练代码改变则创建新运行，不混用旧模型。原论文全量复现保持停止。'
                 '[Gensim Word2Vec 参数](https://radimrehurek.com/gensim/models/word2vec.html) · '
                 '[Gensim FastText 参数](https://radimrehurek.com/gensim/models/fasttext.html)。'
                 '网页显示的文档版本与本机 4.4.0 可能不同，本轮参数以保存的版本和原生模型属性核验为准。')
    REPORT.write_text('\n\n'.join(lines) + '\n', encoding='utf-8')
    submission = (f'在相同语料、100 维、窗口 5、最低词频 5、学习率日程及逐轮选轮规则下，'
                  f'比较 CBOW／Skip-gram × HS／NS5／NS10。固定评价划分种子 42，训练种子独立变化；'
                  f'先做 12 组初筛，再对每个模型的基线与最佳新配置做三种子复验，总计 20 次训练。'
                  f'搜索只用验证分，全部冻结后统一测试。\n\n当前阶段：**{phase}**；'
                  f'已完成 **{len(rows)}/20** 次训练。')
    if frozen:
        submission += '\n\n按三种子验证均值选择：' + '；'.join(
            f'{m} 使用 {LABELS[rank["winner"]]}（ρ={rank["comparisons"][0]["mean"]:.4f}，'
            f'SD={rank["comparisons"][0]["sd"]:.4f}，n=3）' for m, rank in frozen['ranking'].items()) + '。'
    if receipt:
        for name, caption in receipt['captions'].items():
            relative = Path(receipt['figures'][name]).relative_to('reports').as_posix()
            submission += f'\n\n![{caption}]({relative})'
    submission += ('\n\n本轮为探索性选参：评价基准曾被历史实验观察，三种子标准差仅描述训练波动。'
                   'TF-IDF 保留统计对照；本节没有训练类别预测器。'
                   '[完整消融记录](THUCNews训练方式消融实验.md)保存配置、逐轮验证、检查点、耗时和测试隔离证据。')
    update_submission(submission)
    write_json(experiment / 'report_status.json', {'generated': stamp(), 'report_sha256': file_hash(REPORT),
               'search_runs': len(rows), 'test_metric_rows': len(test_rows), 'figures_approved': receipt is not None})
    return REPORT
