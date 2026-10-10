"""Evidence-based native fastText topic report, figures, and scoped 1.2 replacement."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image,ImageOps

from .common import ROOT,file_hash,markdown_table,read_json,stamp,write_json
from .evaluation_support import save_csv
from .figure_support.setup_style import setup_style
from .figure_support.profile_data import profile_data,render_report
from .figure_support.visual_qa import audit_layout,render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure
from .pfr_topic_native import VARIANTS
from .pfr_topics import BASE,load_jsonl

NAMES={'native_scratch':'随机初始化','native_pretrained':'Gensim FastText 初始化'}
COLORS=['#0072B2','#D55E00']
FIGURES=['distribution','validation','performance','topic_f1','confusion']
REPORTS=[ROOT/'reports/Gensim&FastText-去AI味版.md',ROOT/'reports/Gensim&FastText.md']


def load():
    directory=ROOT/read_json(BASE/'latest_experiment.json')['experiment']
    assert read_json(directory/'index.json')['status']=='complete'
    assert read_json(directory/'verification.json')['status']=='passed'
    for n,h in read_json(directory/'test_seal.json').items():
        assert file_hash(directory/n)==h
    plan=read_json(directory/'plan.json')
    df=pd.read_csv(directory/'results.csv')
    assert len(df)==6 and np.isfinite(df[['accuracy','macro_f1','micro_f1','weighted_f1']]).all().all()
    return directory,plan,df


def resources(directory,plan):
    docs=load_jsonl(directory/'labelled_articles.jsonl');lookup={d['id']:d for d in docs}
    vocab=collections.Counter(w for i in plan['splits']['train'] for w in lookup[i]['tokens'])
    rows=[]
    for part,ids in plan['splits'].items():
        counts=collections.Counter(w for i in ids for w in lookup[i]['tokens'])
        total=sum(counts.values());iv=sum(n for w,n in counts.items() if w in vocab)
        rows.append({'partition':part,'articles':len(ids),'tokens':total,'types':len(counts),
                     'iv_tokens':iv,'oov_tokens':total-iv,'token_coverage':iv/total,
                     'all_word_oov_articles':sum(not any(w in vocab for w in lookup[i]['tokens']) for i in ids)})
    save_csv(directory/'word_coverage.csv',rows)
    save_csv(directory/'validation_grid.csv',[{'method':rec['method'],'seed':rec['seed'],'epoch':rec['epoch'],
              **{k:read_json(directory/'runs'/run/'metrics_validation.json')[k] for k in ['accuracy','macro_f1','micro_f1','weighted_f1']},
              'fit_seconds':read_json(directory/'runs'/run/'summary.json')['fit_seconds'],'run':run}
              for run,rec in read_json(directory/'index.json')['runs'].items()])


def figures(directory,plan,df):
    setup_style(journal='general',lang='zh',use_sciplots=False)
    plt.rcParams.update({'font.size':8,'axes.titlesize':9,'axes.labelsize':8,
                        'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5})
    labels=plan['labels'];index=read_json(directory/'index.json');result={}
    counts=pd.read_csv(directory/'split_counts.csv')
    fig,axes=plt.subplots(1,3,figsize=(7.2,3.2),constrained_layout=True)
    for ax,part,title in zip(axes,['train','validation','test'],['训练：一至二月','验证：三月30%','测试：三月70%']):
        subset=counts[counts.partition==part].set_index('topic').loc[labels]
        ax.barh(np.arange(8),subset.articles,color='#0072B2',height=.6)
        ax.set_yticks(np.arange(8),labels);ax.invert_yaxis();ax.set_xlim(0,subset.articles.max()*1.3)
        for j,n in enumerate(subset.articles):
            ax.text(n+subset.articles.max()*.03,j,str(n),va='center',fontsize=7)
        ax.set_xlabel('文章数');ax.set_title(f'{title} (n={subset.articles.sum()})')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    result['distribution']=fig
    grid=pd.read_csv(directory/'validation_grid.csv')
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.2),constrained_layout=True)
    for ax,metric,caption in zip(axes,['macro_f1','accuracy'],['验证 Macro-F1 (%)','验证 Accuracy (%)']):
        for method,color,marker,style in zip(VARIANTS,COLORS,['o','s'],['-','--']):
            subset=grid[(grid.method==method)&(grid.seed==42)].sort_values('epoch')
            ax.plot(subset.epoch,subset[metric]*100,color=color,marker=marker,linestyle=style,label=NAMES[method])
            selected=index['selection'][f'{method}/42']['parameter']
            value=subset.loc[subset.epoch==selected,metric].iloc[0]*100
            ax.scatter([selected],[value],s=90,facecolors='none',edgecolors=color,linewidths=1.5,zorder=5)
        ax.set_xlabel('独立训练预算（轮）');ax.set_ylabel(caption);ax.set_ylim(0,100)
        ax.set_xticks([5,25,50,100,200]);ax.grid(alpha=.2)
        ax.legend(loc='lower right',frameon=False)
    result['validation']=fig
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0),constrained_layout=True)
    for ax,metric,caption in zip(axes,['macro_f1','accuracy'],['测试 Macro-F1 (%)','测试 Accuracy (%)']):
        for j,(method,color) in enumerate(zip(VARIANTS,COLORS)):
            sub=df[df.method==method].sort_values('seed')
            for offset,(_,row),marker in zip([-.13,0,.13],sub.iterrows(),['o','s','^']):
                ax.scatter(row[metric]*100,j+offset,color=color,marker=marker,s=35,zorder=4)
            mean,sd=sub[metric].mean()*100,sub[metric].std(ddof=1)*100
            ax.errorbar(mean,j,xerr=sd,color='#222222',marker='|',markersize=12,capsize=3,zorder=3)
            ax.text(102,j,f'{mean:.2f}±{sd:.2f}',va='center',fontsize=7.5)
        ax.set_yticks([0,1],[NAMES[m] for m in VARIANTS]);ax.invert_yaxis();ax.set_xlim(0,125)
        ax.set_xticks([0,20,40,60,80,100]);ax.set_xlabel(caption);ax.grid(axis='x',alpha=.2)
        ax.set_title('原始三种子点＋均值±样本SD')
    result['performance']=fig
    classes=[]
    for m in VARIANTS:
        for seed in plan['configuration']['seeds']:
            run=index['selection'][f'{m}/{seed}']['run']
            classes.append(read_json(directory/'runs'/run/'metrics_test.json')['per_class'])
    array=np.array([[c[label]['f1-score'] for c in classes] for label in labels])
    fig,ax=plt.subplots(figsize=(7.2,3.8),constrained_layout=True)
    im=ax.pcolormesh(np.arange(7)-.5,np.arange(9)-.5,array,vmin=0,vmax=1,cmap='cividis',rasterized=False)
    test_counts=counts[counts.partition=='test'].set_index('topic').loc[labels].articles
    ax.set_yticks(range(8),[f'{t} (n={n})' for t,n in zip(labels,test_counts)])
    ax.set_xticks(range(6),[f'{"随机" if m=="native_scratch" else "预训练"}\n种子{s}' for m in VARIANTS for s in plan['configuration']['seeds']],rotation=0,fontsize=7)
    ax.set_ylim(7.5,-.5);ax.set_ylabel('自动主题标签');ax.set_aspect('auto')
    for r in range(8):
        for c in range(6):
            ax.text(c,r,f'{array[r,c]:.2f}',ha='center',va='center',fontsize=7,color='white' if array[r,c]<.45 else '#222222')
    cb=fig.colorbar(im,ax=ax,shrink=.85,pad=.02);cb.set_label('逐主题 F1');cb.solids.set_rasterized(False)
    result['topic_f1']=fig
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5),constrained_layout=True)
    for ax,method in zip(axes,VARIANTS):
        run=index['selection'][f'{method}/42']['run']
        raw=np.asarray(read_json(directory/'runs'/run/'metrics_test.json')['confusion_matrix'])
        matrix=raw/raw.sum(axis=1,keepdims=True)
        im=ax.pcolormesh(np.arange(9)-.5,np.arange(9)-.5,matrix,vmin=0,vmax=1,cmap='cividis',rasterized=False)
        ax.set_xticks(range(8),labels,rotation=45,ha='right');ax.set_yticks(range(8),labels)
        ax.set_ylim(7.5,-.5);ax.set_aspect('equal');ax.set_title(NAMES[method]+'（种子42）')
        ax.set_xlabel('预测主题');ax.set_ylabel('自动主题标签')
        for r in range(8):
            for c in range(8):
                if matrix[r,c]>=.1:
                    ax.text(c,r,f'{matrix[r,c]:.2f}',ha='center',va='center',fontsize=7,color='white' if matrix[r,c]<.45 else '#222222')
    cb=fig.colorbar(im,ax=list(axes),shrink=.8,pad=.03);cb.set_label('按自动主题归一化比例');cb.solids.set_rasterized(False)
    result['confusion']=fig
    return result


def preview(directory,plan,df):
    resources(directory,plan)
    folder=directory/'figure_preview';folder.mkdir(exist_ok=True)
    profile=profile_data(df,group_cols=['method'])
    write_json(folder/'data_profile.json',json.loads(json.dumps(profile,default=str)))
    (folder/'数据剖析与图型选择.md').write_text(render_report(profile)+'\n\n'
        '两条原生FastText路线各三种子，直接显示种子点与均值±样本SD，不用均值柱图隐藏小样本。'
        '轮数是有序预算，连接六个独立训练终点，不能解释为单次训练逐轮轨迹。主题数量使用横向条形图，'
        '逐主题F1和混淆矩阵使用统一0—1色阶。准确率和F1轴从零开始。文章bootstrap与种子SD分开解释。\n',encoding='utf-8')
    audits={}
    for name,fig in figures(directory,plan,df).items():
        path=folder/f'{name}.png';render_preview(fig,str(path),dpi=150)
        ImageOps.grayscale(Image.open(path)).convert('RGB').save(folder/f'{name}_grayscale.png')
        audits[name]={'issues':audit_layout(fig),'preview_sha256':file_hash(path),
                      'grayscale_sha256':file_hash(folder/f'{name}_grayscale.png')}
        plt.close(fig)
    write_json(folder/'layout_audit.json',audits)
    print(json.dumps(audits,ensure_ascii=False))


def export(directory,plan,df):
    review=read_json(directory/'figure_preview/visual_review.json');assert review['status']=='approved'
    for name in FIGURES:
        assert file_hash(directory/f'figure_preview/{name}.png')==review['previews'][name]['sha256']
        assert file_hash(directory/f'figure_preview/{name}_grayscale.png')==review['previews'][name]['grayscale_sha256']
    folder=ROOT/'reports/figures/pfr_topics'/directory.name;folder.mkdir(parents=True,exist_ok=True)
    audits={}
    for name,fig in figures(directory,plan,df).items():
        files=export_figure(fig,str(folder/name),formats=['png','svg'],dpi=300,grayscale_preview=True)
        for path in files:
            if path.endswith('_grayscale.png'):
                with Image.open(path) as im:
                    im.save(path,dpi=(300,300))
        audits[name]={Path(p).name:{'issues':check_figure(p)[0],'sha256':file_hash(p)} for p in files}
        assert not any(issue[0] in ['WARN','FAIL'] for rec in audits[name].values() for issue in rec['issues'])
        plt.close(fig)
    write_json(directory/'figure_export_audit.json',audits)


def pct(value):
    return f'{100*value:.2f}'


def mean_sd(values,scale=1):
    arr=np.array(values,dtype=float)*scale
    return f'{arr.mean():.2f} ± {arr.std(ddof=1):.2f}'


def figure(directory,name,caption):
    path=f'figures/pfr_topics/{directory.name}/{name}.png'
    assert (ROOT/'reports'/path).exists()
    return f'![{caption}]({path})\n\n{caption}\n'


def manuscript(directory,plan,df):
    corpus=ROOT/plan['dataset'];manifest=read_json(corpus/'manifest.json')
    annotations=load_jsonl(corpus/'annotations.jsonl');requests=load_jsonl(corpus/'requests.jsonl')
    labels=plan['labels'];index=read_json(directory/'index.json');docs=load_jsonl(directory/'labelled_articles.jsonl')
    lookup={d['id']:d for d in docs};init=read_json(directory/'native_input/initialization.json')
    counts=pd.read_csv(directory/'split_counts.csv')
    statuses=collections.Counter(r['status'] for r in annotations)
    confidence=collections.Counter(r['confidence'] for r in annotations)
    total_input=sum(r.get('usage',{}).get('prompt_tokens',0) for r in requests)
    total_output=sum(r.get('usage',{}).get('completion_tokens',0) for r in requests)
    amount=sum(r['cost_cny'] for r in requests)
    missing_quotes=sum(r['status']=='out_of_scope' and not r.get('evidence') for r in annotations)
    repaired_quotes=sum(bool(r.get('evidence_original')) for r in annotations)
    unknown_usage=sum(not r.get('usage') for r in requests)
    parts={p:len(v) for p,v in plan['splits'].items()}
    chosen={m:index['selection'][f'{m}/42']['parameter'] for m in VARIANTS}
    upper=[NAMES[m] for m in VARIANTS if chosen[m]==max(plan['configuration']['native']['epochs'])]
    budget_limit=('、'.join(upper)+'选中200轮，即本轮搜索上界；这只能说明它在已测预算中取得最高验证Macro-F1，'
                  '尚不能证明继续训练无收益、模型已经收敛或该轮数为全局最优。' if upper else
                  '两条路线的选定轮数均位于已测预算范围内，结论仅适用于本轮固定参数和六档预算。')
    change=read_json(directory/'runs'/index['selection']['native_pretrained/42']['run']/'summary.json')['source_vector_change']
    definitions=['国内政治、党政活动、公共政策与政府治理','金融、贸易、企业经营、产业及农业经济',
                 '科学研究、技术研发及技术应用','比赛、运动员、训练与群众体育','影视、音乐、演出及文艺活动',
                 '社会生活、教育、医疗服务、法治、民生与环境问题','国防、军队、军事装备、演习及军事行动',
                 '外国政局、外交关系及国际组织事务']
    p=['### 1.2 原生 FastText 八类主题分类实验','','#### 1.2.1 数据处理、主题标注与评价划分','',
       f'本节使用人民日报1998年一月至三月语料（9801—9803），任务是根据文章内容确定主主题。'
       f'原始{manifest["raw_articles"]:,}篇，文本清洗后{manifest["kept_articles"]:,}篇。'
       '八类为政治、财经、科技、体育、娱乐、社会、军事、国际；版面编号不作为分类标签。'
       'THUCNews只提供此前学到的FastText词向量，不提供本轮分类训练文章或主题标签。','',
       '原文已经分词并标注词性。按文章编号聚合段落并保留文件顺序，去除编号、词性后缀和实体括号，'
       '使用NFKC统一全半角、清除控制字符。分类输入保留原分词和功能词，标题与正文一并使用；'
       '带标点和段落的版本另供主题标注。少于30个有效词的文章排除，完全重复文本保留最早一篇，'
       '字符三元组SimHash汉明距离不超过3的文章归为近重复组。重复文章处理和划分均不依赖版号。','',
       f'本轮实际排除{manifest["exclusions"].get("short_article",0)}篇短文章，合格文章中完全重复文本'
       f'{manifest["exclusions"].get("exact_duplicate",0)}篇、近重复文章对{manifest["near_duplicate_pairs"]}对；'
       '清洗损失及段落编号异常均单独保存，编号重复且内容不同的段落仍按文件顺序保留。','',
       markdown_table(['月份','原始文章','清洗保留'],[[m,manifest['raw_month_counts'][m],manifest['kept_month_counts'][m]] for m in plan['configuration']['months']]),'',
       markdown_table(['主题','判定范围'],zip(labels,definitions)),'',
       '主题由DeepSeek `deepseek-flash`单次自动标注。请求关闭思考模式，以匿名编号提交完整文章，'
       '不提供日期、版号等源文件元数据或分类器预测；文章正文中的日期仍保留。'
       '按中心事件判定主题：国外财经、体育和科技报道仍归相应主题，不因外国地名而归国际；'
       '政治人物参加某领域活动也不自动归政治。每篇保存分类依据和高／中／低自评等级，进入分类实验的文章还必须有连续原文证据。'
       '这些等级没有经过正确率校准。接口采用20请求并发、每批至多4篇，仅对格式、遗漏或网络失败重试，'
       '成功文章不再做第二次语义标注。','',
       f'全部{len(annotations):,}篇均获得有效结构化标注，其中正常标注{statuses["ok"]:,}篇、'
       f'带歧义标记{statuses["ambiguous"]:,}篇、无法归入单一新闻主题{statuses["out_of_scope"]:,}篇。'
       '后者包括纯文学创作、天气预报和无关主题汇编等，保留排除原因，不强行归入八类。'
       f'自评高／中／低等级分别为{confidence["high"]:,}／{confidence["medium"]:,}／{confidence["low"]:,}篇。'
       f'另有{missing_quotes}篇无法归类文章未返回证据原句，已显式记录并退出分类；'
       f'{repaired_quotes}篇引用通过截取已有响应中真实连续的原文片段或缩短过长引用完成格式修复，'
       '保留原始引用与修复标记，未改变主题或新增接口调用。'
       f'共{len(requests):,}次请求，接口记录输入{total_input:,}、输出{total_output:,} tokens，'
       f'累计计入费用{amount:.4f}元（包括100篇试标及失败重试），低于35元预算。'
       f'费用依据响应usage与调用时段核算，缺少usage的请求共{unknown_usage}次；若发生未知用量的网络失败，则按预留上界单列。'
       '价格查阅于2026-10-10，见[官方计价](https://api-docs.deepseek.com/zh-cn/quick_start/pricing/)。','',
       f'一月和二月全部可归类文章用于训练（n={parts["train"]:,}）；三月按主题分层、按重复组以种子42划为'
       f'验证集（n={parts["validation"]:,}）和测试集（n={parts["test"]:,}），目标比例30%／70%。'
       '验证集选择轮数，配置冻结后才计算测试结果。三月参与选参，因此本节称为三月留出测试，'
       '不将整个三月视为未用于模型选择的数据。','',
       '具体划分先按种子42生成组顺序，再采用不内部打乱的分组分层十折分配，取三折验证、七折测试。'
       '实际各主题验证篇数与30%目标相差不超过2篇；文章ID、相同文本和近重复组均未跨分区。','',
       figure(directory,'distribution','图1.2-1 八类主题在训练、验证和测试集中的文章数。数值为自动标注后实际进入实验的文章数量，三个面板各自从零起，横轴范围不同。'),
       '#### 1.2.2 原生监督模型与预训练向量初始化','',
       '原生监督部分比较原生FastText的随机初始化与Gensim FastText向量初始化两条路线。'
       '原生模型将分词后的词和词bigram映射到输入向量，汇总成文章表示，再通过线性输出层和softmax预测主题。'
       '监督训练同时更新输入向量和输出分类权重。相比仅训练外部分类器，这里考察的是词向量在主题标签监督下继续调整后的效果。','',
       '预测保存原生接口分数及去除数值偏移、归一化后的八类softmax概率。该数值处理保持类别排序，'
       '并不构成概率校准，不能把高分直接解释为相同水平的正确率。','',
       '两条路线统一采用100维、学习率0.1、minCount=1、词bigram、关闭字符子词、softmax、200000哈希桶及20个CPU训练线程。'
       '预训练路线使用此前在THUCNews上训练的Gensim FastText模型：Skip-gram、负采样10、100维、字符子词1—3，验证选中第11轮。'
       '只导出本轮训练集中出现的词；源词表外词允许用Gensim子词生成初值。原生模型导入的是完整词向量，'
       '没有导入Gensim的字符子词桶；词bigram参数仍由监督训练学习。','',
       f'训练词表有{init["target_train_words"]:,}个词，导出{init["exported_words"]:,}个初值，'
       f'其中源词表向量{init["kinds"].get("source_vocabulary",0):,}个、子词生成向量'
       f'{init["kinds"].get("generated_subword",0):,}个。以学习率为零的导入探针核验，'
       f'{len(init["probe_vectors"])}个检查词的初值逐位一致；监督训练后的向量变化另外保存。'
       '更新的是原生模型自己的向量副本，封存的Gensim源模型保持不变。'
       f'选定的种子42模型中，{change["changed_words"]}/{change["words"]}个检查词向量发生变化，'
       f'平均L2变化为{change["mean_l2_change"]:.4f}，最大为{change["maximum_l2_change"]:.4f}。'
       '这是监督训练实际调整输入向量的运行证据，不是仅对初值另接一个冻结分类器。','',
       '#### 1.2.3 验证轮数选择与三种子测试结果','',
       '种子42分别比较5、15、25、50、100、200轮，以验证Macro-F1最高值选轮，若相同则优先较少轮数。'
       '选定预算后用2026、7复验，各自保持同样轮数。轮数对照来自六次独立训练的终点，'
       '并非同一模型连续训练中的逐轮曲线；各预算模型和验证预测均保存为检查点。','',
       budget_limit,'',
       figure(directory,'validation','图1.2-2 两条原生FastText路线的训练预算对照。种子42，每个点对应一次独立训练终点；空心圈标出验证Macro-F1选中的轮数。连线只展示有序预算变化，不代表逐轮连续监测。')]
    grid=pd.read_csv(directory/'validation_grid.csv')
    p.extend([markdown_table(['路线','5轮验证F1','15轮','25轮','50轮','100轮','200轮','选中轮数'],
       [[NAMES[m],*[pct(grid[(grid.method==m)&(grid.seed==42)&(grid.epoch==e)].macro_f1.iloc[0]) for e in plan['configuration']['native']['epochs']],
         index['selection'][f'{m}/42']['parameter']] for m in VARIANTS]),'',
       f'预训练初始化在25轮的验证Macro-F1为{pct(grid[(grid.method=="native_pretrained")&(grid.seed==42)&(grid.epoch==25)].macro_f1.iloc[0])}%，'
       f'已经高于随机初始化200轮的{pct(grid[(grid.method=="native_scratch")&(grid.seed==42)&(grid.epoch==200)].macro_f1.iloc[0])}%。'
       '在这组固定监督参数下，预训练向量改善了较低训练预算时的验证表现；'
       '这里仍包含此前THUCNews预训练提供的额外信息，不能解释为相同数据成本下的效率优势。','',
       '下表以三种子均值±样本标准差报告最终测试表现。Macro-F1等权计入八类，Micro-F1在单标签任务中与Accuracy相同；'
       'Weighted-F1按测试类频加权。种子波动只包含监督训练阶段，源Gensim词向量仍只有此前选定的一套。','',
       markdown_table(['路线','轮数','验证Macro-F1 (%)','测试Macro-F1 (%)','Accuracy / Micro-F1 (%)','Weighted-F1 (%)'],
       [[NAMES[m],index['selection'][f'{m}/42']['parameter'],mean_sd(df[df.method==m].validation_macro_f1,100),
         mean_sd(df[df.method==m].macro_f1,100),mean_sd(df[df.method==m].accuracy,100),
         mean_sd(df[df.method==m].weighted_f1,100)] for m in VARIANTS]),'',
       figure(directory,'performance','图1.2-3 三种子测试表现。圆形、方形、三角形依次表示种子7、42、2026；黑色标记为均值±样本SD，每条路线n=3。此处误差线描述训练波动，不是文章抽样置信区间。')])
    boot=read_json(directory/'bootstrap.json');delta=boot['difference']
    p.extend([f'在种子42的同一测试文章上进行按主题分层的配对bootstrap（2,000次），'
       f'预训练初始化相对随机初始化的Macro-F1差值为{100*delta["estimate"]:.2f}个百分点，'
       f'95%百分位区间为[{100*delta["ci_low"]:.2f}, {100*delta["ci_high"]:.2f}]个百分点。'
       +('该区间跨零，当前测试抽样证据不能区分两条路线。' if delta['ci_low']<=0<=delta['ci_high'] else
         '该区间未跨零，支持这两套已选配置在当前自动标签测试集上的差异。')+
       '这个区间与三种子SD回答不同的不确定性问题。两条路线各自选择预算，差值不是同轮数下初始化因素的单独因果效应。','',
       '#### 1.2.4 逐主题表现、词表覆盖与不一致案例','',
       figure(directory,'topic_f1','图1.2-4 两条路线、三个种子的逐主题F1。所有单元格共用0—1色阶，括号标明测试文章数。各格为对应种子的实际结果，不是平均后生成的重复点。'),
       figure(directory,'confusion','图1.2-5 种子42测试混淆矩阵，按自动主题标签归一化。两幅图共用0—1色阶；数值标出比例至少0.10的单元格。纵轴标签来自大模型，不能视为已核验的人工真值。')])
    class_rows=[];class_means={m:{} for m in VARIANTS}
    for label in labels:
        for m in VARIANTS:
            records=[read_json(directory/'runs'/index['selection'][f'{m}/{s}']['run']/'metrics_test.json')['per_class'][label]
                     for s in plan['configuration']['seeds']]
            class_rows.append([label,NAMES[m],int(records[0]['support']),
                *[mean_sd([r[k] for r in records],100) for k in ['precision','recall','f1-score']]])
            class_means[m][label]=float(np.mean([r['f1-score'] for r in records]))
    p.extend([markdown_table(['主题','路线','测试篇数','Precision (%)','Recall (%)','F1 (%)'],class_rows),''])
    differences={label:class_means['native_pretrained'][label]-class_means['native_scratch'][label] for label in labels}
    most=max(differences,key=differences.get)
    weakest=min(class_means['native_pretrained'],key=class_means['native_pretrained'].get)
    p.extend([f'初始化收益并非均匀分布于八类。平均F1增加最多的是{most}，提高{100*differences[most]:.2f}个百分点；'
       f'预训练路线表现最低的是{weakest}，平均F1为{pct(class_means["native_pretrained"][weakest])}%，'
       f'该类测试样本只有{int(counts[(counts.partition=="test")&(counts.topic==weakest)].articles.iloc[0])}篇。'
       '逐类数值既受模型识别能力影响，也受类别样本数量和自动标签边界影响，不能将总体均值的提升理解为每一类均有改善。',''])
    coverage=pd.read_csv(directory/'word_coverage.csv')
    p.extend(['两条原生模型使用相同的监督训练词表，minCount=1。下表统计词token在该词表内的覆盖率；'
       '原生字符子词已关闭，预训练路线在训练时生成初值，并不意味着推断时可为任意新词生成字符子词向量。','',
       markdown_table(['分区','词token数','词表外token数','词token覆盖率 (%)','全部词均在词表外的文章'],
          [[r.partition,int(r.tokens),int(r.oov_tokens),pct(r.token_coverage),int(r.all_word_oov_articles)] for _,r in coverage.iterrows()]),''])
    run=index['selection']['native_pretrained/42']['run']
    pred=pd.read_csv(directory/'runs'/run/'predictions_test.csv')
    wrong=pred[pred.true_topic!=pred.predicted_topic].sort_values('score',ascending=False).head(4)
    p.extend(['以下为预训练路线种子42与自动标签不一致的高分案例。它们可用于检查类别边界或模型失误，'
              '但单次自动标签不足以判定哪一方正确。','',markdown_table(['文章标题','自动标签','预测主题','预测分数','标注原文证据'],
              [[lookup[r.id]['title'][:45],r.true_topic,r.predicted_topic,f'{r.score:.3f}',
                next(a['evidence'] for a in annotations if a['id']==r.id)] for _,r in wrong.iterrows()]),'',
              '#### 1.2.5 资源成本与实验小结','',
              markdown_table(['路线','训练耗时 (s)','加载及测试预测耗时 (s)','进程峰值RSS (MiB)','模型大小 (MiB)'],
              [[NAMES[m],mean_sd(df[df.method==m].fit_seconds),mean_sd(df[df.method==m].predict_and_load_seconds),
                mean_sd(df[df.method==m].peak_rss_mib),mean_sd(df[df.method==m].model_bytes,1/2**20)] for m in VARIANTS]),'',
              '训练耗时计原生train_supervised调用，包含读取训练输入及预训练向量；不含此前THUCNews预训练、主题标注、向量导出和导入探针。'
              '峰值RSS为进程采样峰值，不能当作模型本体大小。所有候选及三种子模型均保留，保存加载后的验证和测试标签及分数一致。',''])
    means=df.groupby('method').macro_f1.mean();winner=means.idxmax()
    p.extend([f'在本轮三月留出测试中，{NAMES[winner]}取得较高的平均Macro-F1（{pct(means[winner])}%）。'
              '该结论限定于本轮原生FastText配置、验证选轮和自动主题标签；是否存在稳定优势同时结合三种子波动与配对区间判断。'
              '原生监督部分只比较两种初始化方式；冻结表示接分类器的表现应由独立对照实验判断。','',
              '主题标签均来自大模型一次判断，原文证据校验只能检查引用是否真实，不能证明主题一定正确。'
              '标签误差、类别边界和无法归类文章的排除会影响评价范围，测试分数衡量模型与这套弱标签体系的一致性。'
              '后续若需要确认真实主题识别准确率，应另行建立独立人工标注评价集。','',
              f'复现入口见[主题分类指南](../docs/人民日报主题分类复现指南.md)，'
              f'原始结果见[三种子指标](../{directory.relative_to(ROOT).as_posix()}/results.csv)、'
              f'[轮数对照](../{directory.relative_to(ROOT).as_posix()}/validation_grid.csv)和'
              f'[核验记录](../{directory.relative_to(ROOT).as_posix()}/verification.json)。',''])
    return '\n'.join(p)


def publish(directory,plan,df):
    body=manuscript(directory,plan,df)
    backup=directory/'report_before';backup.mkdir(exist_ok=True)
    audit={}
    for path in REPORTS:
        original=path.read_bytes();text=original.decode('utf-8')
        start=text.index('### 1.2 ')
        end=start+re.search(r'(?m)^## part2 ',text[start:]).start()
        if not (backup/path.name).exists():
            (backup/path.name).write_bytes(original)
        # Keep the separately generated frozen-representation comparison intact.
        heading='#### 1.2.6 三种冻结表示与同一 LinearSVC 的比较'
        extra=text[text.index(heading,start):end] if heading in text[start:end] else ''
        combined=body
        if extra:
            combined=combined.replace('### 1.2 原生 FastText 八类主题分类实验','### 1.2 八类主题分类：原生 FastText 与三种表示比较')
            combined+='\n'+extra.rstrip()+'\n'
        changed=text[:start]+combined+'\n'+text[end:]
        path.write_bytes(changed.encode('utf-8'))
        assert changed[:start]==text[:start] and changed[changed.index('## part2 '):]==text[end:]
        audit[path.name]={'before_sha256':__import__('hashlib').sha256(original).hexdigest(),
                         'after_sha256':file_hash(path),'only_section_1_2_changed':True}
    archive=ROOT/'reports/archive';archive.mkdir(exist_ok=True)
    historical=ROOT/'reports/人民日报固定表示分类实验.md'
    if historical.exists():
        archived=archive/'人民日报版面分类_历史_20261010.md'
        if not archived.exists():
            old=historical.read_text(encoding='utf-8').replace('](figures/','](../figures/').replace('](../docs/','](../../docs/').replace('](../artifacts/','](../../artifacts/').replace('](../configs/','](../../configs/').replace('](人民日报固定表示分类实验.md)','](人民日报版面分类_历史_20261010.md)')
            archived.write_text('# 历史版面代理分类记录\n\n该实验预测版号，已退出当前提交报告，不能用于判断主题分类效果。\n\n'+old,encoding='utf-8')
        historical.write_text('# 人民日报版面分类历史记录\n\n'
             '该实验预测版号，已退出当前主题分类报告。原文、图表和数据入口保存在'
             '[历史归档](archive/人民日报版面分类_历史_20261010.md)。\n\n'
             '当前结果见[原生FastText主题分类](人民日报原生FastText主题分类.md)。\n',encoding='utf-8')
    (ROOT/'reports/人民日报原生FastText主题分类.md').write_text('# 人民日报原生FastText主题分类\n\n'+body,encoding='utf-8')
    write_json(directory/'report_edit.json',audit)
    write_json(directory/'result_allocation.json',{'core_discovery':'Native random vs pretrained topic classification',
       'necessary_support':['validation budget','three seeds','paired interval'],
       'qualification':['single-pass weak labels','March participates in selection','different selected budgets'],
       'heterogeneity':'per-theme F1 and disagreement examples','provenance_detail':'CSV/JSON and reproduction guide'})
    print('Published native topic section 1.2; surrounding user text preserved.',flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['preview','export','publish'])
    args=parser.parse_args();directory,plan,df=load()
    if args.stage=='preview':preview(directory,plan,df)
    elif args.stage=='export':export(directory,plan,df)
    else:publish(directory,plan,df)


if __name__=='__main__':main()
