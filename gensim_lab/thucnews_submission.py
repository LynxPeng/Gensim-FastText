"""Update only the user's THUCNews section, preserving the rest of the submission."""
from .common import ROOT, file_hash, read_json, write_json, stamp, markdown_table
from .thucnews import records
from .thucnews_validation import BASE
from .submission_layout import place_hyperparameter_section

TARGET=ROOT/'reports/Gensim&FastText.md'


def figure(name, caption):
    return f'![{caption}](figures/section11/{name}.png)\n\n[SVG 矢量图](figures/section11/{name}.svg)'


def update():
    entries={s['spec']['method']:(d,s) for d,s in records()}
    assert set(entries)=={'tfidf','word2vec','fasttext'}
    approved=read_json(ROOT/'artifacts/submission_section11_figures/approved.json')
    assert approved['status']=='approved' if 'status' in approved else True
    neural={m:entries[m][1] for m in ['word2vec','fasttext']}
    split=read_json(ROOT/neural['word2vec']['spec']['benchmark_manifest'])
    for d,s in entries.values():
        source=(d/'summary.json').relative_to(ROOT).as_posix()
        assert approved['inputs'][source]==file_hash(d/'summary.json')
    original=TARGET.read_text(encoding='utf-8-sig')
    first='#### 1.1.2 实验模型、参数与对比设置'
    boundary='### 1.2 文本分类实验'
    prefix,remaining=original.split(first,1); _,tail=remaining.split(boundary,1)
    old_tail=boundary+tail
    marker='##### 1.1.1.5 内在评价数据的验证／测试划分'
    prefix=prefix.split(marker,1)[0].rstrip()+'\n\n'+marker+'\n\n'
    primary=split['benchmarks']['wordsim240']
    prefix+=f"词相似度与类比题用于评价表示质量，不是新闻文章分区。先对题目去重，再按固定种子 42 划分验证／测试：相同无序词对、关系交换或同时反向的等价类比题在不同基准中共享同一分区，题组交集为 0。单个基准的重复题保留首题，评分冲突另记。WordSim240 的 {primary['original_n']} 个原始词对去重为 {primary['validation']['n']+primary['test']['n']} 对，分为 {primary['validation']['n']} 验证、{primary['test']['n']} 测试；其他基准按同题组规则分配，目标验证比例 40%。\n\n"
    prefix+='每轮评价三个词对集的全部验证题；每套类比从验证池按关系固定抽取最多 128 题，用作辅助曲线。选轮完成后才评价全部测试题。WordSim297 与 WordSim296 大量重叠，不能当作独立重复。历史实验已观察过原完整基准，本次测试子集不能声称是从未被观察过的新盲测；当前训练程序不读取测试分数选轮。'
    prefix+=f"[基准分区及去重清单](../{split['directory']}/manifest.json)、[题组分配](../{split['directory']}/assignments.csv)。\n\n"
    settings=first+'\n\n'
    settings+='**TF-IDF** 使用文章词频乘以 $\\log_2(N/df_t)$，再做 L2 归一化，得到稀疏文章表示。相同训练集的统计模型不受轮数选择影响，本轮直接复用。\n\n'
    settings+='**Word2Vec** 使用 CBOW，以附近词预测中心词，学习每个整词的 100 维向量，词表外词没有直接表示。**FastText** 保持相同目标，在整词表示中加入字符子词参数共享，词表外词可组合子词得到向量。这里使用 Gensim 无监督词向量模型；字符 n-gram 与监督分类的词 n-gram 是两类特征。\n\n'
    settings+=markdown_table(['设置','TF-IDF','Word2Vec','FastText'],[
        ['表示维度','最多 50,000 特征','100','100'],
        ['词表过滤','DF≥3、DF/N≤0.9','训练词频≥5','训练词频≥5'],
        ['训练算法','文章 DF／IDF 统计','CBOW','CBOW'],
        ['上下文窗口','不适用','5','5'],
        ['训练轮数','一次统计遍历',f"实际 {neural['word2vec']['trained_epochs']}／选择 {neural['word2vec']['selected_epoch']}",f"实际 {neural['fasttext']['trained_epochs']}／选择 {neural['fasttext']['selected_epoch']}"],
        ['训练目标','不适用','HS=1、negative=0','HS=1、negative=0'],
        ['学习率','不适用','0.025→0.0001，按最多 15 轮调度','同 Word2Vec'],
        ['高频下采样','不适用','0.001','0.001'],
        ['字符子词','不适用','无','长度 1–3、200,000 桶'],
        ['并行','8 统计进程','20 训练线程','20 训练线程']])+'\n\n'
    settings+='两个词向量模型分别从头训练，至少 5 轮、最多 15 轮，以 **WordSim240 验证 Spearman** 独立选择检查点。连续 3 轮未比有效提升参照增加超过 0.001 时停止；任何更高分都会更新最佳模型，不足 0.001 的改善不重置耐心计数。Pearson、其他词对集和类比只作辅助诊断。每轮保存完整原生检查点，结束后重载得分最高的模型，再评价测试集。学习率按 15 轮上限预先线性调度，提前停止时尚未走完；与历史固定 5 轮日程不同，差异不能只归因于轮数。\n\n'
    settings+='使用 Python 3.12、Gensim 4.4.0、NumPy 1.26.4，种子 42，输入顺序和 Python hash seed=0 固定，数值库 4 线程。原生 LineSentence 和有界任务队列每批最多 9,000 原始词，各轮核对完整词数。训练使用 CPU；多线程不保证独立重训逐位一致，每个模型本轮只有一次正式运行，不作显著性判断。当前未做 THUCNews 的维度、窗口或子词范围消融，所选轮次仅针对当前配置和验证任务。'
    settings+='[词向量配置](../configs/thucnews_validation.yaml)、[TF-IDF 配置](../configs/thucnews.yaml)。\n\n'
    results='#### 1.1.3 训练结果与评估\n\n'
    for i,method in enumerate(['word2vec','fasttext'],1):
        s=neural[method]; directory=entries[method][0]; name='Word2Vec' if method=='word2vec' else 'FastText'
        primary_metric=next(m for m in read_json(directory/'epochs.json')[s['selected_epoch']-1]['metrics'] if m['benchmark']=='wordsim240')
        reason='连续 3 轮没有有效提升而提前停止' if s['stop_reason']=='validation_patience' else '达到 15 轮上限而停止'
        results+=f"##### 1.1.3.{i} {name} 逐轮验证与模型选择\n\n"
        results+=f"{name} 实际训练 **{s['trained_epochs']} 轮**，{reason}。验证主指标最高的是 **第 {s['selected_epoch']} 轮，Spearman ρ={s['selection']['best_score']:.4f}**，该轮检查点作为最终模型。训练最后一轮没有自动替代最佳轮。\n\n"
        caption=f"图 1.{i} {name} 逐轮验证曲线。a：Spearman；b：Pearson；c：覆盖类比题正确率；d：全部探针正确率（OOV 按错）。虚线标记独立选定的第 {s['selected_epoch']} 轮。主指标使用 WordSim240 验证集 {primary_metric['covered']}/{primary_metric['total']} 个词表内词对；类比每套固定最多 128 验证题。轮次来自同一次训练，没有重复实验误差带，不做平滑或插值。"
        results+=figure(method+'_validation',caption)+'\n\n'
        rel='../'+directory.relative_to(ROOT).as_posix()
        results+=f"[逐轮完整指标]({rel}/validation_metrics.csv)、[轮次／时间／检查点索引]({rel}/epochs.csv)、[保存加载复核]({rel}/save_load_check.json)。\n\n"
    results+='两个模型使用同一评价分区，但分别计算指标、分别决定停止和最佳轮数。验证曲线用于选择当前配置的训练轮数；单次曲线与有限词对不能证明所有语义任务已经收敛，也不能据此作模型差异显著性判断。\n\n'
    results+='##### 1.1.3.3 训练耗时、峰值内存与模型大小\n\n'
    results+=figure('resources','图 1.3 三种表示的特征数、训练／拟合时间、主进程峰值 RSS 和完整原生模型大小。柱长为单次实测值；每面板使用独立单位。')+'\n\n'
    results+=f"Word2Vec 与 FastText 均保留 {neural['word2vec']['vocabulary_size']:,} 词，TF-IDF 为 50,000 文章特征。词向量时间累计所有实际训练轮的训练调用：Word2Vec {neural['word2vec']['fit_seconds']:.1f} 秒、FastText {neural['fasttext']['fit_seconds']:.1f} 秒；不包含词表构建、验证、检查点保存、最终测试。TF-IDF 包含 DF／CF 统计和 IDF 计算，计时范围不同，不能直接当作同算法加速比；两个词向量模型实际轮数不同，累计时间也不代表单位轮次速度比较。TF-IDF 的资源记录沿用历史相同统计模型，峰值包含当时的文档诊断。峰值 RSS 为模型主进程，未累计子进程；TF-IDF 体积含词典，词向量为所选完整原生模型，不含全部检查点总占用。\n\n"
    results+='##### 1.1.3.4 词相似度：Pearson 与 Spearman\n\n'
    results+='使用模型余弦相似度与人工评分比较；Pearson 衡量线性相关，Spearman 衡量排序相关。它们不是类别预测准确率。下面仅评价所选模型的独立测试词表内词对，两个模型的词表和覆盖题相同。\n\n'
    counts='、'.join(f"{m['benchmark']} {m['covered']}/{m['total']}" for m in neural['word2vec']['metrics'] if m['kind']=='similarity')
    results+=figure('similarity_metrics','图 1.4 测试词对 Pearson／Spearman 及 FastText 减 Word2Vec 的差值。上排绝对相关系数，下排描述差值；覆盖数为 '+counts+'。每模型 n=1，不判断差异显著性。')+'\n\n'
    results+=figure('wordsim_pairs','图 1.5 WordSim240 独立测试词对人工评分与模型余弦相似度。每点为一个覆盖词对，两个面板使用相同题目；展示逐题差异，不把词对当作模型独立重复。')+'\n\n'
    w=next(m for m in neural['word2vec']['metrics'] if m['benchmark']=='wordsim240'); f=next(m for m in neural['fasttext']['metrics'] if m['benchmark']=='wordsim240')
    results+=f"WordSim240 测试 Spearman 分别为 Word2Vec {w['spearman_rho']:.4f}、FastText {f['spearman_rho']:.4f}，差值 {f['spearman_rho']-w['spearman_rho']:+.4f}。各基准差值按本次实测解释，不能推广为稳定优势。WordSim296 去掉 OPEC–石油词对，且与 WordSim297 共享大多数题目。本次两个版本的测试题完全相同，图中合并显示，原始指标分别保存。FastText 可生成 OOV 的探索分数另存完整记录，不混入正式词表内相关。\n\n"
    results+='##### 1.1.3.5 词类比：覆盖题与全部题正确率\n\n'
    counts='、'.join(f"{m['benchmark']} {m['covered']}/{m['total']}" for m in neural['word2vec']['metrics'] if m['kind']=='analogy')
    results+=figure('analogy_accuracy','图 1.6 所选模型的全部测试类比题；覆盖题正确率与全题正确率分面，单位为百分比，OOV 在全题口径按错。两模型覆盖数相同：'+counts+'。')+'\n\n'
    results+='类比通过“B−A+C”寻找最相近词，正式评价只搜索训练词表，字符子词能生成 OOV 向量不等于能完成所有 OOV 类比。形态与语义关系分别报告；与逐轮小探针的分母不同，不能用图中测试分数回选轮数。TF-IDF 是文档表示，词相似度和类比不适用。\n\n'
    results+='##### 1.1.3.6 词表与可用表示覆盖率\n\n'
    results+=figure('coverage','图 1.7 同一 5,000 篇留出文章、2,337,756 词次的词表／可用表示覆盖率与 OOV 次数。比例和计数分面；浅灰斜线为未覆盖部分，生成向量不代表语义正确。')+'\n\n'
    h=neural['fasttext']['heldout']
    results+=f"FastText 为 {h['generated_oov_tokens']:,}/{h['oov_tokens']:,} 次 OOV 出现生成可用向量；Word2Vec 跳过这些词。TF-IDF 覆盖受 50,000 特征限制，不能把覆盖率当作语义质量排序。\n\n"
    results+='##### 1.1.3.7 近邻词与句子上下文评分\n\n'
    for method in ['word2vec','fasttext']:
        directory,s=entries[method]; queries=read_json(directory/'queries.json')['queries']
        economy='、'.join(w for w,_ in queries['经济'].get('nearest_neighbors',[])[:3])
        oov='、'.join(w for w,_ in queries['新能源汽车产业'].get('nearest_neighbors',[])[:3]) or '无向量'
        results+=f"{method} 查询“经济”的前三近邻为“{economy}”；查询“新能源汽车产业”得到“{oov}”。近邻仅作定性检查，不能代替基准指标。\n\n"
        post=BASE/'posthoc'/directory.name/'summary.json'; assert post.exists()
        check=read_json(post); assert check['model_sha256']==file_hash(directory/'model.gensim')
        results+=f"[所选模型的 {check['sentences']:,} 条留出句子评分](../{post.parent.relative_to(ROOT).as_posix()}/heldout_sentence_scores.csv)。\n\n"
        probes=post.parent/'qualitative_probes.json'; assert probes.exists()
        results+=f"[定性 API 探针](../{probes.relative_to(ROOT).as_posix()})另存词／词组相似度、乘法类比和异类词结果，不作为模型排名或选轮指标。\n\n"
    results+='HS 上下文 score 累计窗口内词预测事件，忽略 OOV；FastText 继承 Word2Vec HS 评分路径，仅作兼容性诊断。它不是整句生成对数概率，不计算语言模型困惑度，也不参与本轮早停。TF-IDF 不提供这种评分。\n\n'
    results+='##### 1.1.3.8 核验与结果范围\n\n'
    results+='本节提供三种表示的资源和覆盖率比较，以及分别按验证指标选择的 Word2Vec／FastText 测试结果。没有训练类别预测器，也没有据此确定哪种表示用于文本分类最好。选轮、每轮检查点重载、训练词表频次、原始词数、逐题测试指标及保存向量通过[独立核查](../artifacts/thucnews_validation/verification.json)。'
    results+='[完整实验记录](THUCNews词向量实验.md)保存 p 值、原始指标、日志、时间和协议边界；[绘图记录](../artifacts/submission_section11_figures/数据剖析与图型选择.md)保存完整精度数据、来源、图型选择及视觉核查。\n\n'
    tests=read_json(BASE/'test_result.json')
    results+=f"项目测试 {tests['passed']} 项通过，见[测试日志](../artifacts/thucnews_validation/tests.log)。\n\n"
    ablation=''
    if '<!-- thucnews-ablation:start -->' in original:
        ablation='<!-- thucnews-ablation:start -->'+original.split('<!-- thucnews-ablation:start -->',1)[1].split('<!-- thucnews-ablation:end -->',1)[0]+'<!-- thucnews-ablation:end -->\n\n'
    content=prefix+settings+results+old_tail
    if ablation:
        body=ablation.split('<!-- thucnews-ablation:start -->',1)[1].split('<!-- thucnews-ablation:end -->',1)[0]
        body=body.strip().split('\n',1)[1].lstrip()
        content=place_hyperparameter_section(content,body)
    assert content.split(boundary,1)[1]==tail
    assert 'loss' not in content.split('### 1.1 Gensim',1)[1].split(boundary,1)[0].lower()
    assert not any(line.startswith('|') for line in results.splitlines())
    TARGET.write_text(content,encoding='utf-8')
    write_json(BASE/'submission_update.json',{'updated':stamp(),'tail_preserved':True,'tail_sha256':__import__('hashlib').sha256(old_tail.encode('utf-8')).hexdigest(),
        'selected_epochs':{m:s['selected_epoch'] for m,s in neural.items()},'report_sha256':file_hash(TARGET)})
    print(f'Updated submission section 1.1: {TARGET}',flush=True)


if __name__=='__main__':
    update()
