"""Export exact, line-addressed implementation excerpts for the engineering review."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/Gensim_FastText核心代码.md"
REVIEW = ROOT / "artifacts/reviews/fasttext_paper_engineering"
G = ROOT / "references/sources/gensim-4.4.0/models"
N = ROOT / "references/sources/fastText-0.9.2/src"


def link(path, line=None):
    target = path.as_posix() + (f":{line}" if line else "")
    return f"[{path.name}](<{target}>)"


# Inclusive, one-based source ranges. All snippets are verbatim contiguous text.
SNIPPETS = [
    ("C01", "构造函数和默认值", G / "fasttext.py", 274, 279, "python"),
    ("C02", "词 n-gram 限制、参数表与 Word2Vec 继承", G / "fasttext.py", 418, 442, "python"),
    ("C03", "训练入口：文件流与 iterable 分派", G / "fasttext.py", 499, 546, "python"),
    ("C04", "UTF-8 字符 n-gram 提取", G / "fasttext_inner.pyx", 677, 720, "cython"),
    ("C05", "FNV-1a 字节哈希", G / "fasttext_inner.pyx", 619, 641, "cython"),
    ("C06", "字符 n-gram 分桶入口", G / "fasttext.py", 1326, 1349, "python"),
    ("C07", "训练批次的词表过滤与子词缓存绑定", G / "fasttext_inner.pyx", 470, 510, "cython"),
    ("C08", "训练窗口和 SG／CBOW、HS／NS 分派", G / "fasttext_inner.pyx", 513, 569, "cython"),
    ("C09", "Skip-gram + Negative Sampling 完整核心函数", G / "fasttext_inner.pyx", 75, 145, "cython"),
    ("C10", "Skip-gram + Hierarchical Softmax 完整核心函数", G / "fasttext_inner.pyx", 148, 221, "cython"),
    ("C11", "CBOW + Negative Sampling 完整核心函数", G / "fasttext_inner.pyx", 224, 306, "cython"),
    ("C12", "CBOW + Hierarchical Softmax 完整核心函数", G / "fasttext_inner.pyx", 309, 373, "cython"),
    ("C13", "负采样累计分布", G / "word2vec.py", 824, 851, "python"),
    ("C14", "IV／OOV 向量查询", G / "fasttext.py", 1085, 1130, "python"),
    ("C15", "稠密参数分配", G / "fasttext.py", 1149, 1161, "python"),
    ("C16", "导出词向量合成与字符桶缓存", G / "fasttext.py", 1191, 1227, "python"),
    ("C17", "句向量接口", G / "fasttext.py", 1132, 1147, "python"),
    ("C18", "corpus_file 复用相同训练内核", REVIEW / "sources/fasttext_corpusfile.pyx", 18, 28, "cython"),
    ("C19", "corpus_file 的 SG 更新与学习率调度", REVIEW / "sources/fasttext_corpusfile.pyx", 158, 187, "cython"),
    ("P01", "项目表示模型训练", ROOT / "gensim_lab/models.py", 50, 65, "python"),
    ("P02", "项目词向量池化与 L2 单位化", ROOT / "gensim_lab/models.py", 95, 125, "python"),
    ("P03", "项目外部 SVM 分类器", ROOT / "gensim_lab/models.py", 155, 173, "python"),
    ("P04", "论文复现路线的文档池化", ROOT / "gensim_lab/paper_worker.py", 48, 73, "python"),
    ("N01", "原生模型平均隐藏层与梯度更新", N / "model.cc", 42, 91, "cpp"),
    ("N02", "原生负采样分布与重抽样", N / "loss.cc", 145, 190, "cpp"),
    ("N03", "原生输入词 ID 与 EOS 特判", N / "dictionary.cc", 196, 205, "cpp"),
    ("N04", "原生 SG／CBOW 的输入特征构建", N / "fasttext.cc", 387, 421, "cpp"),
    ("N05", "原生梯度归一化开关", N / "fasttext.cc", 762, 766, "cpp"),
]


def main():
    audit = json.loads((REVIEW / "audit.json").read_text(encoding="utf-8"))
    assert audit["gensim_version"] == "4.4.0"
    parts = ["# Gensim 4.4.0 FastText 核心代码\n",
             "日期：2026-10-09。以下代码是当前环境对应源文件的**原样连续摘录**，没有用教学伪代码替换实现。"
             "这里只输出核心调用链；完整源文件、许可证和 SHA-256 见末尾。代码块之间省略的外围逻辑不属于可独立运行模块。\n",
             f"对比结论：{link(ROOT / 'reports/Gensim_FastText与论文工程差异.md')}。"
             f"本次数值核查：{link(REVIEW / 'audit.json')}。\n",
             "## 参数与训练调用链\n",
             "`FastText → Word2Vec.build_vocab/train → _do_train_job / _do_train_epoch → "
             "fasttext_inner / fasttext_corpusfile → SG/CBOW × NS/HS → adjust_vectors`。\n",
             "训练修改 `vectors_vocab`（词输入参数）、`vectors_ngrams`（字符桶输入参数）和 "
             "`syn1neg`／`syn1`（词输出参数）；`wv.vectors` 是查询使用的合成矩阵。"
             "输出层的行对应词或词树节点。外部标签分类器属于项目代码。\n",
             "## 代码索引\n", "| 编号 | 内容 | 源文件与行号 |\n| --- | --- | --- |\n"]
    for code, title, path, start, end, lang in SNIPPETS:
        parts[-1] += f"| {code} | {title} | {link(path, start)}，{start}–{end} |\n"
    for code, title, path, start, end, lang in SNIPPETS:
        text = path.read_text(encoding="utf-8")
        selected = "\n".join(text.splitlines()[start-1:end])
        parts.extend([f"\n## {code} {title}\n\n",
                      f"来源：{link(path, start)}，原文件第 {start}–{end} 行。\n\n",
                      f"```{lang}\n{selected}\n```\n"])
    parts.append("\n## 完整源码、授权与版本边界\n\n")
    parts.append("Gensim 源码遵循 LGPL-2.1；原生 fastText 源码遵循 MIT。摘录保留原始注释；"
                 "再分发源码应同时保留原许可证。本项目只审查 4.4.0／0.9.2，不把结论外推到其它版本。\n\n")
    parts.append(f"Gensim 许可证：{link(REVIEW / 'sources/LICENSE')}；"
                 f"fastText 许可证：{link(ROOT / 'references/sources/fastText-0.9.2/LICENSE')}。\n\n")
    parts.append("| 当前环境源码副本 | SHA-256 | 与项目已有副本比较 |\n| --- | --- | --- |\n")
    for row in audit["source_manifest"]:
        state = "文本一致（换行统一后）" if row["matches_project_snapshot_text"] else "本次新增存档"
        parts[-1] += f"| {link(ROOT / row['archive_path'])} | `{row['sha256']}` | {state} |\n"
    parts.append("\n上述散列针对随 wheel 安装的源码。已单独记录实际加载 `.pyd` 的散列，并用数值探针"
                 "验证本报告涉及的编译内核行为；没有声称只凭 `.pyx` 的散列能证明二进制构建完全相同。\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"Exported {len(SNIPPETS)} exact excerpts: {OUT}")
    verify()


def verify():
    core = OUT.read_text(encoding="utf-8")
    blocks = re.findall(r"```[^\n]*\n(.*?)\n```", core, re.S)
    assert len(blocks) == len(SNIPPETS)
    for block, (code, title, path, start, end, lang) in zip(blocks, SNIPPETS):
        lines = path.read_text(encoding="utf-8").splitlines()
        assert block == "\n".join(lines[start-1:end]), code
        if code in ["C09", "C10", "C11", "C12"]:
            next_function = next(i for i in range(start, len(lines)) if lines[i].startswith("cdef "))
            assert block.rstrip() == "\n".join(lines[start-1:next_function]).rstrip(), code
    review = ROOT / "reports/Gensim_FastText与论文工程差异.md"
    errors, tables = [], 0
    for path in [OUT, review]:
        text = path.read_text(encoding="utf-8")
        fence, width = False, None
        for number, row in enumerate(text.splitlines(), 1):
            if row.startswith("```"):
                fence = not fence
                width = None
                continue
            if fence:
                continue
            if row.startswith("|"):
                count = len(re.findall(r"(?<!\\)\|", row)) - 1
                if width is None:
                    width = count
                    tables += 1
                elif count != width:
                    errors.append([path.name, number, "table column mismatch"])
            else:
                width = None
        assert not fence, path
        assert "\ufffd" not in text, path
        for target in re.findall(r"\]\(<([^>]+)>\)", text):
            file = Path(re.sub(r":\d+$", "", target))
            if not file.exists():
                errors.append([path.name, target, "missing local link"])
    audit = json.loads((REVIEW / "audit.json").read_text(encoding="utf-8"))
    for name, expected in audit["source_fingerprints"].items():
        import hashlib
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    assert not errors, errors
    result = {"verbatim_code_excerpts": len(blocks), "tables": tables,
              "full_kernel_functions_verified": 4, "missing_links": 0,
              "source_fingerprints_unchanged": True, "errors": errors}
    (REVIEW / "document_check.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
