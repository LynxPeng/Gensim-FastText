"""Source-linked figures for section 1.1.3; no retraining or amplitude filter."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
import numpy as np
import pandas as pd
from PIL import Image

from .common import ROOT, file_hash, read_json, write_json, stamp
from .figure_support.profile_data import profile_data, render_report
from .figure_support.setup_style import setup_style
from .figure_support.visual_qa import audit_layout, render_preview
from .figure_support.export_figure import export_figure
from .figure_support.check_figure import check_figure

QA = ROOT / "artifacts/submission_section11_figures"
OUT = ROOT / "reports/figures/section11"
LABELS = {"tfidf": "TF-IDF", "word2vec": "Word2Vec", "fasttext": "FastText"}
COLORS = {"tfidf": "#666666", "word2vec": "#0072B2", "fasttext": "#D55E00"}
MARKERS = {"tfidf": "D", "word2vec": "o", "fasttext": "s"}
HATCHES = {"tfidf": "xx", "word2vec": "", "fasttext": "//"}
METHODS = ["tfidf", "word2vec", "fasttext"]
BENCHMARKS = ["wordsim240", "wordsim297", "wordsim296"]
ANALOGIES = ["CA_translated", "CA8_morphological", "CA8_semantic"]
NAMES = ["word2vec_validation", "fasttext_validation", "resources", "similarity_metrics", "analogy_accuracy", "coverage", "wordsim_pairs"]


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.generic):
        return safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def frames():
    from .thucnews import records
    entries = {summary['spec']['method']:(directory,summary) for directory,summary in records()}
    if set(entries) != set(METHODS):
        raise ValueError('Both independently selected embedding runs must finish before publishing comparisons')
    inputs, resources, coverage, similarity, analogy, validation, pairs = {}, [], [], [], [], {}, []
    for method in METHODS:
        directory, summary = entries[method]
        seal = read_json(directory/'seal.json')
        names = ['summary.json'] if method=='tfidf' else ['summary.json','epochs.json','validation_metrics.csv','test/wordsim240_pairs.csv']
        for name in names:
            path=directory/name
            assert file_hash(path)==seal['files'][name]
            inputs[path.relative_to(ROOT).as_posix()]=file_hash(path)
        resources.append({'model':method,'features':summary['vocabulary_size'],'fit_seconds':summary['fit_seconds'],
                          'peak_rss_mib':summary['peak_rss_mib'],'model_mib':summary['model_bytes']/2**20})
        h=summary['heldout']
        coverage.append({'model':method,'vocabulary_percent':100*h['vocabulary_token_coverage'],
                         'usable_percent':100*h['usable_vector_token_coverage'],'oov_tokens':h['oov_tokens'],
                         'generated_oov_tokens':h['generated_oov_tokens']})
        for m in summary['metrics']:
            row={'model':method,'benchmark':m['benchmark'],'covered':m['covered'],'total':m['total']}
            if m['kind']=='similarity':
                similarity.append({**row,'pearson_r':m['pearson_r'],'spearman_rho':m['spearman_rho']})
            else:
                analogy.append({**row,'conditional_percent':100*m['conditional_accuracy'],'all_percent':100*m['all_accuracy']})
        if method!='tfidf':
            f=pd.read_csv(directory/'validation_metrics.csv')
            f['model']=method
            f.attrs['selected_epoch']=summary['selected_epoch']
            f.attrs['trained_epochs']=summary['trained_epochs']
            validation[method+'_validation']=f
            f=pd.read_csv(directory/'test/wordsim240_pairs.csv')
            f=f[f.status=='ok'].copy()
            f['model']=method
            pairs.append(f)
    data={'resources':pd.DataFrame(resources),'coverage':pd.DataFrame(coverage),
          'similarity':pd.DataFrame(similarity),'analogy':pd.DataFrame(analogy),
          **validation,'wordsim_pairs':pd.concat(pairs,ignore_index=True)}
    return data,inputs


def prepare():
    QA.mkdir(parents=True, exist_ok=True)
    data, inputs = frames()
    text = ("# 1.1.3 图表数据剖析与选择\n\n"
            "图表回答训练过程、模型资源成本及词表示质量的差异；不按变化幅度筛选。"
            "每个模型只有一次正式运行、seed 42；逐轮记录来自有序训练过程，基准题目是评价单元，均不构成模型独立重复。"
            "不添加误差棒或显著性。采用中文通用科研图规范，宽 7.2 英寸，PNG 300 DPI 与可编辑 SVG。\n\n")
    choices = {
        "word2vec_validation": "Word2Vec 四面板真实轮次曲线，分别表示验证 Spearman、Pearson、类比覆盖题和全量正确率；标明独立选中的轮次。WordSim297/296 的验证词对分别为 109/108，保留两条曲线，不能与相同的测试题集合混淆。备选为逐指标单图。",
        "fasttext_validation": "FastText 独立四面板验证曲线，使用同一分区，单独选轮数；不沿用 Word2Vec 的最优轮数。备选为逐指标单图。",
        "resources": "四面板水平条形图，柱长直接比较特征数、耗时、主进程 RSS 和模型大小，各自标明单位；备选为资源比值图。柱长为单次实测值，不是重复试验均值。",
        "similarity": "上排为 Pearson 与 Spearman 并列条形图，下排为 FastText 减 Word2Vec 的实际差值；兼顾绝对水平与小差异。新分区的 WordSim297/296 测试题完全相同，合并标为 WordSim297/296，避免重复展示同一组结果。备选为数值热图。差值轴保留零点，容纳正负差值，不作显著性判断。",
        "analogy": "覆盖题与全量正确率并列条形图，柱长和标注直接比较两个方法，百分比轴从零开始，覆盖分母写入图注；备选为带注释的分面热图。",
        "coverage": "覆盖和未覆盖、可用和不可用分别用百分比堆叠条形图表示；OOV 词次及可生成 OOV 向量的词次另用条形图，比例与计数不混用坐标。备选为两个比例图加计数热图。",
        "wordsim_pairs": "独立测试词对逐题散点，比较人工评分与余弦相似度；每点是一个评价词对，不是独立模型复验。备选为误差分布图。",
    }
    for name, frame in data.items():
        path = QA / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        info = profile_data(frame, group_cols=["model"])
        write_json(QA / f"{name}_profile.json", safe(info))
        text += f"## {name}\n\n{choices[name]}\n\n" + render_report(info) + "\n\n"
    delta = data["similarity"].pivot(index="benchmark", columns="model", values=["pearson_r", "spearman_rho"])
    differences = pd.DataFrame({column + "_delta": delta[column]["fasttext"] - delta[column]["word2vec"]
                                for column in ["pearson_r", "spearman_rho"]}).loc[BENCHMARKS].reset_index()
    differences.to_csv(QA / "similarity_differences.csv", index=False, encoding="utf-8-sig")
    text += ("## 剖析解释\n\nmodel、benchmark 是分类标签；epoch 是有序轮次，不是独立重复。验证和最终测试题目去重分组后划分，不将两个 WordSim 版本当独立重复。"
             "整数特征数和词次数量按数值轴绘制，不能因唯一值较少将它们当分类变量。"
             "相关系数和准确率的题目覆盖子集固定；词覆盖与语义准确性分别解释。"
             "条形图直接展示一个训练模型的实测成本或其基准评价统计量，不表示多次重训的均值或分布；因此不补造误差棒。"
             "相关系数差值按同一基准的 FastText 减 Word2Vec 计算，不把不同 WordSim 版本当作独立重复。"
             "所有图使用原始全精度数值，注释只做显示舍入，坐标从零起，不做平滑或插值。\n")
    (QA / "数据剖析与图型选择.md").write_text(text, encoding="utf-8")
    write_json(QA / "sources.json", {"inputs": inputs, "selected_epochs": {m:data[m+"_validation"].attrs["selected_epoch"] for m in ["word2vec","fasttext"]},
                                    "selection": "all requested metrics; no amplitude threshold",
                                    "models_n": 1, "seed": 42})
    return data, inputs


def base_axes(rows=1, columns=2, height=3.2):
    fig, axes = plt.subplots(rows, columns, figsize=(7.2, height), squeeze=False, layout="constrained")
    fig.set_constrained_layout_pads(w_pad=.12, h_pad=.12, wspace=.12, hspace=.12)
    return fig, axes


def title(ax, label, text):
    ax.set_title(f"{label}  {text}", loc="left", fontweight="normal")


def method_bars(ax, frame, column, maximum, xlabel, formatter):
    ordered = frame.set_index("model").loc[METHODS]
    for y, method in enumerate(METHODS):
        value = float(ordered.loc[method, column])
        ax.barh(y, value, height=.48, color=COLORS[method], hatch=HATCHES[method],
                edgecolor="#333333", linewidth=.4, zorder=3)
        ax.annotate(formatter(value), (value, y), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8)
    ax.set(yticks=range(3), yticklabels=[LABELS[m] for m in METHODS],
           ylim=(2.5, -.65), xlim=(0, maximum), xlabel=xlabel)
    ax.grid(axis="x", alpha=.15)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))


def paired_bars(ax, frame, column, benchmarks, labels, maximum, xlabel, digits):
    for i, method in enumerate(["word2vec", "fasttext"]):
        sub = frame[frame.model == method].set_index("benchmark").loc[benchmarks]
        y = np.arange(len(benchmarks)) + [-.18, .18][i]
        values = sub[column].to_numpy(float)
        ax.barh(y, values, height=.28, color=COLORS[method], hatch=HATCHES[method],
                edgecolor="#333333", linewidth=.4, zorder=3)
        for x, yi in zip(values, y):
            ax.annotate(f"{x:.{digits}f}", (x, yi), xytext=(7, 0), textcoords="offset points",
                        va="center", fontsize=8)
    ax.set(yticks=range(len(benchmarks)), yticklabels=labels, ylim=(len(benchmarks)-.5, -.55),
           xlim=(0, maximum), xlabel=xlabel)
    ax.grid(axis="x", alpha=.15)


def legend(fig):
    handles = [Patch(facecolor=COLORS[m], hatch=HATCHES[m], edgecolor="#333333", linewidth=.4,
                     label=LABELS[m]) for m in ["word2vec", "fasttext"]]
    fig.legend(handles=handles, loc="outside upper center", ncol=2, frameon=False)


def difference_bars(ax, frame, column, benchmarks, labels, maximum):
    values = frame.pivot(index="benchmark", columns="model", values=column).loc[benchmarks]
    differences = values.fasttext.to_numpy(float) - values.word2vec.to_numpy(float)
    ax.barh(np.arange(len(labels)), differences, height=.48, color=COLORS["fasttext"],
            hatch=HATCHES["fasttext"], edgecolor="#333333", linewidth=.4, zorder=3)
    for y, value in enumerate(differences):
        ax.annotate(f"{value:+.4f}", (value, y), xytext=(6 if value>=0 else -6, 0), textcoords="offset points",
                    ha="left" if value>=0 else "right", va="center", fontsize=8)
    ax.set(yticks=range(len(labels)), yticklabels=labels, ylim=(len(labels)-.5, -.55),
           xlim=(min(0,float(differences.min())*1.5),max(.001,float(differences.max())*1.5)), xlabel="相关系数差（FT − W2V）")
    ax.grid(axis="x", alpha=.15)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))


def coverage_bars(ax, frame, column):
    ordered = frame.set_index("model").loc[METHODS]
    for y, method in enumerate(METHODS):
        value = float(ordered.loc[method, column])
        ax.barh(y, value, height=.48, color=COLORS[method], hatch=HATCHES[method],
                edgecolor="#333333", linewidth=.4, zorder=3)
        ax.barh(y, 100-value, left=value, height=.48, color="#EEEEEE", hatch="\\\\",
                edgecolor="#999999", linewidth=.4, zorder=3)
        ax.text(value / 2, y, f"{value:.2f}%", ha="center", va="center", fontsize=8, color="white",
                bbox={"facecolor": COLORS[method], "edgecolor": "none", "pad": 2})
    ax.set(yticks=range(3), yticklabels=[LABELS[m] for m in METHODS],
           ylim=(2.5, -.65), xlim=(0, 100), xlabel="占留出词次的比例（%）")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.grid(axis="x", alpha=.15)


def validation_curves(frame, method):
    palette=['#0072B2','#009E73','#D55E00']
    shapes=['o','^','s']
    styles=['-','--',':']
    fig, axes=base_axes(2,2,5.9)
    best=frame.attrs['selected_epoch']
    for ax,column,benchmarks,labels,heading,letter in zip(axes.flat,
            ['spearman_rho','pearson_r','conditional_accuracy','all_accuracy'],
            [BENCHMARKS,BENCHMARKS,ANALOGIES,ANALOGIES],
            [['WordSim240（主指标）','WordSim297','WordSim296'],['WordSim240','WordSim297','WordSim296'],
             ['CA translated','CA8 形态','CA8 语义'],['CA translated','CA8 形态','CA8 语义']],
            ['验证 Spearman ρ','验证 Pearson r','验证类比：覆盖题正确率','验证类比：全量正确率'],'abcd'):
        if column in ['pearson_r','spearman_rho']:
            left=frame[frame.benchmark=='wordsim297'].sort_values('epoch')[column].to_numpy(float)
            right=frame[frame.benchmark=='wordsim296'].sort_values('epoch')[column].to_numpy(float)
            if np.array_equal(left,right):
                benchmarks=benchmarks[:2]; labels=[labels[0],'WordSim297/296（相同 IV 词对）']
        for i,(benchmark,label) in enumerate(zip(benchmarks,labels)):
            rows=frame[frame.benchmark==benchmark].sort_values('epoch')
            y=rows[column].to_numpy(float)
            if 'accuracy' in column:
                y=y*100
            ax.plot(rows.epoch,y,color=palette[i],marker=shapes[i],linestyle=styles[i],ms=4,lw=1.1,label=label)
        ax.axvline(best,color='#777777',ls='--',lw=.8)
        x=sorted(frame.epoch.unique())
        ax.set(xlabel='训练轮次（epoch）',xticks=x,xlim=(.6,max(x)+.4),
               ylabel='正确率（%）' if 'accuracy' in column else '相关系数')
        ax.set_ylim((0,100) if 'accuracy' in column else (-1 if frame[column].min()<0 else 0,1))
        title(ax,letter,heading)
        ax.grid(axis='y',alpha=.15)
        ax.legend(loc='upper right',fontsize=7,frameon=False)
        if column=='spearman_rho':
            primary=float(frame[(frame.benchmark=='wordsim240') & (frame.epoch==best)][column].iloc[0])
            ax.scatter([best],[primary],s=50,facecolors='none',edgecolors='#333333',zorder=5)
            first=best==min(frame.epoch)
            ax.annotate(f'第 {best} 轮：ρ={primary:.4f}',(best,primary),xytext=(10 if first else 0,-20),
                        textcoords='offset points',ha='left' if first else 'center',fontsize=8)
    fig.supxlabel(f'{LABELS[method]} · 虚线为选定的第 {best} 轮；每模型一次正式运行',fontsize=9)
    return fig


def build(export=False):
    data, inputs = prepare()
    OUT.mkdir(parents=True, exist_ok=True)
    setup_style(journal="general", lang="zh", use_sciplots=False)
    plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "axes.titlesize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 9, "legend.fontsize": 9,
                         "font.weight": "normal", "svg.fonttype": "none"})
    receipt = read_json(QA / "visual_review.json") if export and (QA / "visual_review.json").exists() else {}
    checks, exports = [], []

    def finish(fig, name):
        fig.canvas.draw()
        for ax in fig.axes:
            bounds = [*sorted(ax.get_xlim()), *sorted(ax.get_ylim())]
            points = [np.column_stack([line.get_xdata(), line.get_ydata()]) for line in ax.lines]
            points += [np.asarray(c.get_offsets()) for c in ax.collections]
            for bar in ax.patches:
                points.append(np.array([[bar.get_x(), bar.get_y()],
                                        [bar.get_x()+bar.get_width(), bar.get_y()+bar.get_height()]]))
            for p in points:
                assert np.isfinite(p).all()
                assert ((p[:, 0] >= bounds[0]) & (p[:, 0] <= bounds[1]) &
                        (p[:, 1] >= bounds[2]) & (p[:, 1] <= bounds[3])).all(), name
        preview = QA / f"{name}_preview.png"
        render_preview(fig, str(preview), dpi=150)
        issues = audit_layout(fig)
        assert not any(level in ["WARN", "FAIL"] for level, _ in issues), (name, issues)
        with Image.open(preview) as im:
            im.convert("L").save(QA / f"{name}_preview_grayscale.png", dpi=(150, 150))
        digest = file_hash(preview)
        checks.append({"name": name, "preview_sha256": digest, "issues": issues,
                       "axes_ranges": [{"x": list(a.get_xlim()), "y": list(a.get_ylim())} for a in fig.axes]})
        if export:
            assert receipt.get("status") == "approved" and receipt["previews"][name] == digest, name
            paths = export_figure(fig, str(OUT / name), formats=["png", "svg"], dpi=300,
                                  size_inches=tuple(fig.get_size_inches()), tight=False)
            checks[-1]["exports"] = []
            for path in paths:
                result, metadata = check_figure(path, min_dpi=300,
                                               target_inches=tuple(fig.get_size_inches()) if path.endswith(".png") else None)
                assert not any(level in ["WARN", "FAIL"] for level, _ in result), (path, result)
                checks[-1]["exports"].append({"path": str(Path(path).relative_to(ROOT)),
                                              "sha256": file_hash(path), "issues": result, "metadata": metadata})
            exports.append(name)
        plt.close(fig)

    for method in ['word2vec','fasttext']:
        finish(validation_curves(data[method+'_validation'],method),method+'_validation')

    fig, axes = base_axes(2, 2, 5.3)
    for ax, (column, heading, maximum, xlabel, formatter), letter in zip(axes.flat, [
        ("features", "词表 / 特征数", 320000, "词项数", lambda x: f"{x:,.0f}"),
        ("fit_seconds", "拟合耗时", 1400, "时间（秒）", lambda x: f"{x:,.1f}"),
        ("peak_rss_mib", "主进程峰值 RSS", 1650, "内存（MiB）", lambda x: f"{x:,.1f}"),
        ("model_mib", "完整原生模型大小", 400, "模型大小（MiB）", lambda x: f"{x:.1f}"),
    ], "abcd"):
        method_bars(ax, data["resources"], column, max(1, float(data["resources"][column].max())*1.35), xlabel, formatter)
        title(ax, letter, heading)
    finish(fig, "resources")

    fig, axes = base_axes(2, 2, 5.7)
    chart_benchmarks=BENCHMARKS
    labels = ["WordSim240", "WordSim297", "WordSim296"]
    left=data['similarity'][data['similarity'].benchmark=='wordsim297'].set_index('model')[['pearson_r','spearman_rho']]
    right=data['similarity'][data['similarity'].benchmark=='wordsim296'].set_index('model')[['pearson_r','spearman_rho']]
    if left.equals(right):
        chart_benchmarks=BENCHMARKS[:2]; labels=['WordSim240','WordSim297/296']
    for ax, column, heading, letter in zip(axes[0], ["pearson_r", "spearman_rho"], ["Pearson r", "Spearman ρ"], "ab"):
        paired_bars(ax, data["similarity"], column, chart_benchmarks, labels, 1, "相关系数", 4)
        title(ax, letter, heading)
    for ax, column, heading, letter in zip(axes[1], ["pearson_r", "spearman_rho"], ["Pearson 差值", "Spearman 差值"], "cd"):
        difference_bars(ax, data["similarity"], column, chart_benchmarks, labels, .05)
        title(ax, letter, heading)
    legend(fig)
    finish(fig, "similarity_metrics")

    fig, axes = base_axes(height=3.8)
    labels = ["CA translated", "CA8 形态", "CA8 语义"]
    for ax, column, heading, letter in zip(axes.flat, ["conditional_percent", "all_percent"],
                                         ["覆盖题正确率", "全量正确率"], "ab"):
        paired_bars(ax, data["analogy"], column, ANALOGIES, labels, 100, "正确率（%）", 2)
        title(ax, letter, heading)
    legend(fig)
    finish(fig, "analogy_accuracy")

    fig, axes = base_axes(2, 2, 5.3)
    for ax, column, heading, letter in zip(axes[0], ["vocabulary_percent", "usable_percent"],
                                         ["留出词表覆盖率", "可用表示覆盖率"], "ab"):
        coverage_bars(ax, data["coverage"], column)
        title(ax, letter, heading)
    for ax, (column, heading, maximum, xlabel, formatter), letter in zip(axes[1], [
        ("oov_tokens", "词表外词次", 230000, "OOV 词次", lambda x: f"{x:,.0f}"),
        ("generated_oov_tokens", "可生成 OOV 向量的词次", 5600, "OOV 词次", lambda x: f"{x:,.0f}"),
    ], "cd"):
        method_bars(ax, data["coverage"], column, maximum, xlabel, formatter)
        title(ax, letter, heading)
    fig.legend(handles=[Patch(facecolor="#EEEEEE", hatch="\\\\", edgecolor="#999999",
                              label="浅灰斜线：未覆盖 / 不可用")], loc="outside upper center", frameon=False)
    finish(fig, "coverage")

    fig,axes=base_axes(height=3.6)
    for ax,method,letter in zip(axes.flat,['word2vec','fasttext'],'ab'):
        frame=data['wordsim_pairs'][data['wordsim_pairs'].model==method]
        ax.scatter(frame.human_similarity,frame.cosine_similarity,color=COLORS[method],marker=MARKERS[method],s=10,alpha=.6)
        ax.set(xlabel='人工相似度评分',ylabel='词向量余弦相似度',ylim=(-1,1))
        ax.grid(alpha=.15)
        title(ax,letter,f'{LABELS[method]} · 测试词对 n={len(frame)}')
    finish(fig,'wordsim_pairs')
    write_json(QA / "layout_audit.json", safe(checks))
    if export:
        write_json(QA / "approved.json", {"generated": stamp(), "inputs": inputs, "figures": exports,
                                           "generator_sha256": file_hash(Path(__file__)),
                                           "review_sha256": file_hash(QA / "visual_review.json")})
    print(json.dumps({"status": "exported" if export else "previewed", "figures": len(checks)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["prepare", "preview", "export"], default="preview")
    stage = parser.parse_args().stage
    if stage == "prepare":
        prepare()
        print("EDA and source CSVs prepared")
    else:
        build(export=stage == "export")
