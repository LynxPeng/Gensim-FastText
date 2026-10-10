"""Current representation report: independently selected native embeddings."""
from .common import ROOT, file_hash, markdown_table, read_json, stamp, write_json

REPORT = ROOT/'reports/THUCNews词向量实验.md'


def fmt(value):
    return '不适用' if value is None else f'{value:.4f}'


def report():
    from .thucnews import records
    from .thucnews_validation import BASE, CONFIG
    entries = records()
    neural = [(d,s) for d,s in entries if s['spec']['method'] != 'tfidf']
    manifest = read_json(ROOT/entries[0][1]['spec']['manifest'])
    lines = ['# THUCNews 三种表示学习实验', '<a id="thucnews-word-learning"></a>',
        '## 数据与预处理',
        f"原始 {manifest['source']['article_files']:,} 篇文章，清洗、短文过滤与完全去重后保留 {manifest['eligible_articles']:,} 篇。"
        '标题和正文合并；统一全半角，处理 HTML/实体、网址、控制字符与空白；jieba 关闭 HMM 分词，不删除停用词。'
        '少于 30 词的文章排除，每句最多 256 词。固定文章顺序，独立保留 5,000 篇文章做覆盖率诊断。'
        '新闻类别不用于本轮词向量训练和轮数选择。本轮没有训练类别预测器。',
        markdown_table(['分区','文章','句子/分块','词数'], [[name,p['documents'],p['sentences'],p['tokens']]
            for name,p in manifest['partitions'].items()]),
        markdown_table(['排除原因','文章'],list(manifest['excluded'].items())),
        f"[数据清单](../{manifest['directory']}/manifest.json)。语料已排除完全重复，未按近重复新闻分组。",
        '## 模型与线程',
        'TF-IDF 沿用同一训练集拟合的统计模型：文档频次至少 3、最高占比 0.9、至多 50,000 词，log2(N/DF)，L2 归一化。'
        '其结果与训练设置未改变，可直接复用。Word2Vec 与 FastText 分别从头训练原生 Gensim 模型：'
        'CBOW、100 维、窗口 5、最低词频 5、HS=1/negative=0、下采样 0.001。FastText 使用字符子词 1–3、200,000 桶。'
        '两个模型各使用 20 个 CPU 训练线程，数值库 4 线程；固定种子 42、输入顺序和 Python hash seed=0。'
        '多线程更新不保证逐位确定性；每个模型仅一次正式训练（n=1），不报告重复实验误差或显著性。',
        '使用 LineSentence 和 Gensim 有界任务队列；每个任务最多 9,000 原始词，保留句子边界。'
        '学习率 0.025→0.0001 按最多 15 轮线性调度；提前停止时尚未走完该调度。'
        '因此，本次与历史固定 5 轮训练的差异还包含学习率日程，不能只归因于增加轮数。',
        f'[配置](../{CONFIG.relative_to(ROOT).as_posix()}) · [训练入口](../gensim_lab/thucnews_validation.py)。',
        '## 独立验证与测试',
        '评价题目先去重，再固定划分验证与测试。相同无序词对及关系交换/同时反向的等价类比题跨基准共享分区，避免同题泄漏。'
        '同一基准的重复题保留第一题，冲突评分单独记录；不同版本的人工评分不合并。'
        'WordSim240 237 个唯一词对固定分为 95 验证、142 测试；其他基准按同一题组规则划分。'
        '逐轮记录三个词对集全部验证词对的 Pearson、Spearman、覆盖率与 p 值，'
        '以及每套类比验证池固定抽取最多 128 题的正确率。测试阶段使用全部测试题。',
        'Word2Vec、FastText 分别按 WordSim240 验证集严格词表内 Spearman 选择最佳检查点。'
        '至少 5 轮、最多 15 轮；连续 3 轮未比有效提升参照增加超过 0.001 时停止。'
        '任何更高分都更新最佳检查点，但不超过 0.001 的小幅提升不重置耐心计数。'
        'Pearson 与类比只作辅助诊断，不参与选择。两个模型独立完成选择并重载核对后，才计算各自最终测试结果。'
        '验证曲线只能支持当前语料、配置和评价任务的轮数选择，不能证明全局最优或所有下游任务均已收敛。']
    if neural:
        split = read_json(ROOT/neural[0][1]['spec']['benchmark_manifest'])
        lines.append(markdown_table(['基准','原题数','等价重复删除','验证池','逐轮评价','测试'],[
            [name,b['original_n'],b['duplicates_removed'],b['validation']['n'],
             b.get('validation_probe',b['validation'])['n'],b['test']['n']]
            for name,b in split['benchmarks'].items()]))
        lines.append(f"[分区清单](../{split['directory']}/manifest.json) · [题组分配](../{split['directory']}/assignments.csv)。"
        '分区题组交集为 0。WordSim297/296 的本次测试题完全相同，图中合并标注；两份指标分别保存。'
        '该协议不同于历史完整基准直接测试，历史分数不能直接作为本次提升幅度。原完整基准已被历史实验观察过，'
        '本次保留集不能声称是未观察过的新基准盲测；当前选轮程序不读取测试分数。')
    lines.append('## 逐轮验证与模型检查点')
    for directory,s in neural:
        method=s['spec']['method']; selected=s['selected_epoch']; epochs=read_json(directory/'epochs.json')
        reason='验证主指标连续三轮无有效提升' if s['stop_reason']=='validation_patience' else '达到最多 15 轮'
        lines.extend([f'### {method}',
            f"实际训练 {s['trained_epochs']} 轮，因{reason}结束；选择第 {selected} 轮，"
            f"验证 Spearman={s['selection']['best_score']:.4f}。每轮均保存完整原生检查点、配置和逐题评价，"
            '最终 model.gensim 为所选检查点，训练最后一轮与部署模型可能不同。',
            f'![{method} 逐轮验证](figures/section11/{method}_validation.png)',
            markdown_table(['轮次','验证主指标 ρ','最佳轮次','耐心计数','训练秒','评价秒','保存秒'],[
                [r['epoch'],fmt(r['primary_score']),r['best_epoch_so_far'],r['stale_epochs'],
                 f"{r['training_seconds']:.1f}",f"{r['validation_seconds']:.1f}",f"{r['checkpoint_seconds']:.1f}"] for r in epochs])])
    lines.extend(['## 最终测试与覆盖率',
        '词相似度仅对严格词表内词对计算正式相关系数。FastText 子词生成的 OOV 结果另存探索指标。'
        '类比同时报告覆盖题正确率与 OOV 按错误计入的全部测试题正确率；TF-IDF 不适用这些词向量评价。'])
    metrics=[]
    for _,s in neural:
        for m in s['metrics']:
            metrics.append({'model':s['spec']['method'],**{k:v for k,v in m.items() if k!='sections'}})
    lines.append(markdown_table(['方法','词对集','覆盖/测试','Spearman ρ','Pearson r','含生成 OOV 的 ρ（探索）'],[
        [m['model'],m['benchmark'],f"{m['covered']}/{m['total']}",fmt(m['spearman_rho']),fmt(m['pearson_r']),
         fmt(m['generatable_spearman_rho']) if m['model']=='fasttext' else '—'] for m in metrics if m['kind']=='similarity']))
    lines.append(markdown_table(['方法','类比集','覆盖/测试','正确题','覆盖题正确率','全部测试正确率'],[
        [m['model'],m['benchmark'],f"{m['covered']}/{m['total']}",m['correct'],fmt(m['conditional_accuracy']),fmt(m['all_accuracy'])]
        for m in metrics if m['kind']=='analogy']))
    lines.append(markdown_table(['方法','词表/特征数','训练/拟合秒','主进程峰值 RSS MiB','原生模型 MiB','留出可用词覆盖率'],[
        [s['spec']['method'],s['vocabulary_size'],f"{s['fit_seconds']:.1f}",f"{s['peak_rss_mib']:.1f}",
         f"{s['model_bytes']/2**20:.1f}",fmt(s['heldout']['usable_vector_token_coverage'])] for _,s in entries]))
    lines.append('神经模型训练时间仅累计所有实际训练轮的训练调用，不包含词表构建、验证评价、检查点保存或最终测试；'
        '上述开销分别保存。TF-IDF 拟合包含 DF/CF 统计和 IDF 计算，计时范围不同。'
        '原生模型体积不含所有检查点总占用；TF-IDF 包含词典。峰值内存为模型主进程 RSS。')
    lines.append('## 句子上下文评分与核验')
    for directory,s in neural:
        post=BASE/'posthoc'/directory.name/'summary.json'
        if post.exists():
            item=read_json(post)
            if item['model_sha256']==file_hash(directory/'model.gensim'):
                lines.append(f"{s['spec']['method']} 的最佳模型保存 {item['sentences']:,} 条留出句子上下文评分，"
                    f"[逐句 CSV](../{post.parent.relative_to(ROOT).as_posix()}/heldout_sentence_scores.csv)。")
                probes=post.parent/'qualitative_probes.json'
                if probes.exists():
                    lines.append(f"[定性 API 探针](../{probes.relative_to(ROOT).as_posix()})保存近邻、词／词组相似度、乘法类比与异类词结果；不作为训练停止指标。")
    lines.append('HS 上下文 score 是固定窗口中词预测事件的分数，忽略 OOV；FastText 继承 Word2Vec 的评分路径，'
        '这里仅作兼容性诊断。它不是整句生成概率，不计算困惑度，也不用于早停。训练过程只使用外部验证指标。')
    check=BASE/'verification.json'
    if check.exists():
        result=read_json(check)
        if result['status']=='passed' and {r['seal_sha256'] for r in result['checks']}=={file_hash(d/'seal.json') for d,_ in neural}:
            lines.append('独立核验通过：题组无交叉、每轮主指标复算、原生检查点重载、最佳轮次与耐心规则、训练词数、'
                '训练词表频次、逐题测试分母和保存向量均一致。'
                '[核验记录](../artifacts/thucnews_validation/verification.json)。')
    lines.append('## 资源与复现')
    test=BASE/'test_result.json'
    if test.exists() and read_json(test)['status']=='passed':
        lines.append(f"项目测试 {read_json(test)['passed']} 项通过。[测试日志](../artifacts/thucnews_validation/tests.log)。")
    for d,s in neural:
        prefix='../'+d.relative_to(ROOT).as_posix()
        resources=[('配置','spec.json'),('日志','run.log'),('逐轮记录','epochs.csv'),('逐轮指标','validation_metrics.csv'),
                   ('最佳模型','model.gensim'),('KV','vectors.kv'),('测试指标','metrics.json'),('重载核对','save_load_check.json')]
        lines.append(s['spec']['method']+'：'+' · '.join(f'[{title}]({prefix}/{name})' for title,name in resources))
    lines.append('```powershell\nconda run -n Gensim python -m gensim_lab.thucnews_validation --smoke\n'
        'conda run -n Gensim python -m gensim_lab.thucnews_validation\n'
        'conda run -n Gensim python -m gensim_lab.thucnews_audit\n'
        'conda run -n Gensim python -m gensim_lab.thucnews --stage report\n```')
    lines.append('中断后按配置/数据/代码指纹从最后完整检查点恢复；已封存结果复用缓存。历史固定 5 轮和分类实验资源保留，'
        '当前报告采用上述新分区。原论文全量复现仍保持停止。'
        '[THUNLP](https://github.com/thunlp/THUCTC) · [CWE 词对](https://github.com/Leonard-Xu/CWE) · '
        '[Chinese Word Vectors 类比](https://github.com/Embedding/Chinese-Word-Vectors)。')
    REPORT.write_text('\n\n'.join(lines)+'\n',encoding='utf-8')
    from .evaluation_support import save_csv
    save_csv(BASE/'metrics.csv',metrics)
    write_json(BASE/'report_status.json',{'generated':stamp(),'models':len(entries),'report':REPORT.relative_to(ROOT).as_posix()})
    print(f'Report saved: {REPORT}',flush=True)
    return REPORT
