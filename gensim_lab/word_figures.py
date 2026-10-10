"""Report archived intrinsic metrics; epoch selection is handled by THUCNews validation."""
import re
import pandas as pd
from .common import ROOT, REPORTS, markdown_table, read_json
from .word_evaluation import BASE
from .figure_support.profile_data import profile_data, render_report


def fmt(value):
    return '不适用' if value is None or pd.isna(value) else f'{float(value):.4f}'


def report(export=False):
    data = read_json(BASE/'metrics.json')
    models = read_json(BASE/'evaluated_models.json')
    frame = pd.DataFrame(data)
    info = profile_data(frame,group_cols=['method','benchmark','origin'])
    (REPORTS/'词向量评价数据剖析.md').write_text('# 词向量评价数据剖析\n\n'+render_report(info)+
        '\n\n模型与种子是分组标识；原始分类语料的历史评价与 THUCNews 新逐轮验证分开解释。\n',encoding='utf-8')
    md = '# 词向量全面评价与句子上下文评分\n\n'
    md += '本文保留历史分类语料的外部词表示评价。THUCNews 已改用独立验证与测试分区，分别选择 Word2Vec 和 FastText 轮数，见[当前词向量实验](THUCNews词向量实验.md)。\n\n'
    md += '## 口径与适用范围\n\n词相似度使用 Pearson、Spearman 和双侧 p 值；类比同时给出覆盖题与全量正确率。OOV 生成结果独立标注。句子上下文评分只作兼容性诊断，不作为语言模型困惑度。\n\n'
    md += '## 全部外部指标\n\n'
    similarity = [r for r in data if r['kind']=='similarity']
    analogy = [r for r in data if r['kind']=='analogy']
    md += markdown_table(['模型','方法/类/种子','评测','覆盖/全部','Pearson r','Spearman ρ','OOV%'],[
        [r['id'],f"{r['method']}/{r['classes']}/{r['seed']}",r['benchmark'],f"{r['covered']}/{r['total']}",
         fmt(r['pearson_r']),fmt(r['spearman_rho']),fmt(r['oov_ratio_percent'])] for r in similarity])+'\n\n'
    md += markdown_table(['模型','评测','覆盖/全部','正确题','覆盖题正确率','全量正确率'],[
        [r['id'],r['benchmark'],f"{r['covered']}/{r['total']}",r['correct'],fmt(r['conditional_accuracy']),fmt(r['all_accuracy'])]
        for r in analogy])+'\n\n'
    md += '## 历史评价图\n\n以下为已有实测资源，历史参数对照不计作当前 THUCNews 的逐轮验证。\n\n'
    approved=BASE/'figures_approved.json'
    if not approved.exists():
        approved=BASE/'approved_figures.json'
    if not approved.exists():
        approved=BASE/'visual_review.json'
    if approved.exists():
        receipt=read_json(approved)
        for item in receipt.get('figures',[]):
            if isinstance(item,dict):
                name=item.get('name','')
                path=item.get('relative_report_path',f'figures/word_evaluation/{name}.png')
                caption=item.get('caption',name)
            else:
                name=item;path=f'figures/word_evaluation/{name}.png';caption=name
            if 'loss' not in name.lower() and (REPORTS/path).exists():
                md+=f'![{caption}]({path})\n\n'
    md += '## 保存与复现\n\n全部 p 值及逐题数值见[指标 CSV](../artifacts/word_evaluation/metrics.csv)。历史数值与模型保留，当前训练入口为 `python -m gensim_lab.thucnews_validation`。\n'
    (REPORTS/'词向量全面评价报告.md').write_text(md,encoding='utf-8')
    insert_into_main()
    print(f'Archived intrinsic report: {len(models)} models')


def insert_into_main():
    target=REPORTS/'最终实验报告.md'
    if not target.exists():
        return
    begin='<!-- word-evaluation:start -->';end='<!-- word-evaluation:end -->'
    content=target.read_text(encoding='utf-8')
    block=begin+'\n\n## 词向量内置评价与句子上下文评分\n\n词相似度、类比、OOV 与上下文评分的原始历史数值见[完整评价报告](词向量全面评价报告.md)。当前 THUCNews 分别采用 Word2Vec、FastText 的逐轮验证与检查点选择，见[当前实验](THUCNews词向量实验.md)。\n\n'+end+'\n'
    content=re.sub(re.escape(begin)+r'.*?'+re.escape(end)+'\n?','',content,flags=re.S).rstrip()+'\n\n'+block
    target.write_text(content,encoding='utf-8')
