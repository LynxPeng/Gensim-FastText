"""Refresh the combined report while preserving the editorial chapters."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from report_paper import paper_reproduction_body
from report_outline import thucnews_body

def main():
    source = ROOT / 'reports/论文实验复现.md'
    text = source.read_text(encoding='utf-8')
    note = ('协议差异：本轮原生模型依照论文使用层次 softmax；归档 v0.9.2 分类脚本采用默认 softmax、'
            '4 线程及随机打乱行序。本轮固定原始行序、使用 20 线程，学习率沿用该脚本公布的值。'
            '这些差异与版本差异均可能影响结果；全量数据和任务指标对齐，不等于历史数值逐项一致。')
    if note not in text:
        text = text.replace('<a id="reproduction-classification"></a>', note + '\n\n<a id="reproduction-classification"></a>')
        source.write_text(text, encoding='utf-8')
    api_note = ('能力与计时边界：Gensim 4.4.0 的 FastText 实现对应子词词向量论文，'
                '没有 `train_supervised`，且不支持 `word_ngrams != 1`。本机已核查编译内核，'
                '实际训练使用 `corpus_file` 和 20 个线程。当前两条路线的耗时比较反映不同训练方案的成本，'
                '不能用于认定哪个框架更快；同算法速度基准应对照原生 `train_unsupervised`。'
                '依据：[Gensim 4.4.0 源码](https://github.com/piskvorky/gensim/blob/4.4.0/gensim/models/fasttext.py)、'
                '[原生训练 API](https://fasttext.cc/docs/en/python-module.html)。')
    if api_note not in text:
        text = text.replace('<a id="reproduction-classification"></a>', api_note + '\n\n<a id="reproduction-classification"></a>')
        source.write_text(text, encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'gensim_lab.paper_figures', '--stage', 'attach'], cwd=ROOT, check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/build_consolidated_report.py")], cwd=ROOT, check=True)
    edition = ROOT / "reports/综合研究报告-修改版.md"
    text = edition.read_text(encoding="utf-8")
    text = text.replace('正式随机模型采用三个种子、单线程训练。',
                        '人民日报历史随机模型采用三个种子、单线程训练；当前默认训练 20 线程，历史记录保留原始口径。')
    text = text.replace('正式单线程','历史单线程')
    text = text.replace('正式选参只使用单线程候选，多线程另作速度对照。',
                        '本节历史选参使用单线程候选，多线程另作速度对照；现行正式训练默认多线程。')
    start, end = '<a id="paper-reproduction"></a>', '<a id="code-reading"></a>'
    prefix, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    text = prefix + start + "\n\n## 二、FastText复现\n\n" + paper_reproduction_body() + "\n\n" + end + tail
    thuc_anchor = '<a id="thucnews-word-learning"></a>'
    next_anchor = '<a id="implementation-comparison"></a>'
    if (ROOT / 'reports/THUCNews词向量实验.md').exists():
        if thuc_anchor in text:
            before, rest = text.split(thuc_anchor, 1)
            _, after = rest.split(next_anchor, 1)
        else:
            before, after = text.split(next_anchor, 1)
        text = (before + thuc_anchor + '\n\n### 4.9 THUCNews 三种表示学习\n\n'
                + thucnews_body() + '\n\n' + next_anchor + after)
        text = text.replace('独立基本语料实验和原论文复现仍未完成。',
                            '独立 THUCNews 词向量实验已完成，原论文全量复现按用户要求停止。')
        text = text.replace('作业要求的独立基本语料尚未接入，THUCNews 等外部语料也尚未用于本项目实验。',
                            '4.1–4.8 节报告人民日报实验，新增 THUCNews 独立词向量实验另列于 4.9 节。')
    text = text.replace("检查包括章节结构、第二部分留空、源文档覆盖", "检查包括章节结构、第二部分与复现记录一致、源文档覆盖")
    text = text.replace("原论文的实验协议与数值结果尚未严格复现，独立基本语料也尚未接入。",
                        "第二部分已接入原论文完整数据，原生监督模型与 Gensim 两阶段近似路线分开报告，当前进度以运行记录为准。")
    edition.write_text(text, encoding="utf-8", newline="\n")
    subprocess.run([sys.executable, str(ROOT / "tools/build_consolidated_report.py"), "--check", "--report", str(edition)], cwd=ROOT, check=True)

if __name__ == "__main__":
    main()
