"""Build the user's six-part Markdown report and complete CSV detail resources.

This reporting operation does not train models or modify experiment indices.
Run in the project's Gensim conda environment.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote

from report_data import DATA, export_details, read_csv
from report_outline import anchored, render_report
from report_paper import PRIMARY_PAPER, paper_review_body, paper_reproduction_body
from report_sections import tables

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/综合研究报告.md"
BASELINE = "f16410ec42742d1c64036e93ccb7792f3b7ed07e"
CHAPTER_TITLES = ["摘要", "一、FastText论文解读", "二、FastText复现", "三、FastText核心代码阅读笔记",
                  "四、基于Gensim文本向量化表示学习的各种实验", "五、对比Gensim的实现方法",
                  "六、各类参数，超参数尝试比较"]
SOURCE_ANCHORS = {
    "最终实验报告.md": "main-results", "时间扩展实验.md": "temporal-results",
    "数据质量报告.md": "data-quality", "各版面清洗损失.md": "data-quality",
    "论文阅读与源码分析.md": "paper-reading", "源码位置索引.md": "code-reading",
    "参数分析.md": "fasttext-factors", "词向量全面评价报告.md": "word-similarity",
    "补充实验与改进报告.md": "optimization", "错误分析.md": "errors",
    "运行核验证据.md": "engineering", "数据与实验深度核查.md": "engineering",
    "CUDA可用性评估.md": "resource-comparison", "逐类指标与混淆矩阵.md": "data-resources",
    "实验记录.md": "data-resources", "词向量评价数据剖析.md": "intrinsic-factors",
    "高级要求完成清单.md": "engineering", "作业原文转录.md": "paper-reading",
    "作业需求说明.md": "data-quality", "实施计划.md": "main-parameters",
    "审查与补充验证计划.md": "robustness", "全面评价与绘图计划.md": "intrinsic-factors",
    "README.md": "data-resources",
    "FastText论文解读与综述.md": "paper-reading",
    "论文实验复现.md": "paper-reproduction",
    "性能瓶颈与耗时估算.md": "paper-reproduction",
    "Gensim调用与FastText对比审查.md": "engineering",
    "THUCNews词向量实验.md": "thucnews-word-learning",
    "THUCNews5万篇词向量实验.md": "thucnews-word-learning",
    "THUCNews单线程全量阶段记录.md": "thucnews-word-learning",
    "THUCNews词向量实验方案.md": "thucnews-word-learning",
    "多线程迁移与输入路径核验.md": "thucnews-word-learning",
    "Gensim&FastText.md": "thucnews-word-learning",
    "THUCNews训练方式消融实验.md": "thucnews-training-ablation",
    "THUCNews训练方式消融方案.md": "thucnews-training-ablation",
    "THUCNews小规模超参数搜索.md": "thucnews-small-search",
    "THUCNews小规模搜索方案.md": "thucnews-small-search",
    "THUCNews学习率三档历史记录.md": "thucnews-small-search",
    "THUCNews向量维数搜索.md": "thucnews-dimensions",
    "THUCNews向量维数搜索方案.md": "thucnews-dimensions",
    "Gensim_FastText核心代码.md": "code-reading",
    "Gensim_FastText重点代码筛选.md": "code-reading",
    "Gensim_FastText与论文工程差异.md": "implementation-comparison",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def current_jobs():
    from gensim_lab.integrity import spec_is_current
    indexed = read_json(ROOT / "artifacts/index.json")["jobs"]
    return {run_id: job for run_id, job in indexed.items()
            if spec_is_current(read_json(ROOT / "artifacts/runs" / run_id / "spec.json"))}


def build():
    jobs = current_jobs()
    models = read_json(ROOT / "artifacts/word_evaluation/evaluated_models.json")
    metrics = read_csv(ROOT / "artifacts/word_evaluation/metrics.csv")
    assert (len(jobs), len(models), len(metrics)) == (201, 99, 594), "Review conclusions when evidence changes"
    sources = [ROOT / ("README.md" if name == "README.md" else
                       ("reports/" if (ROOT / "reports" / name).exists() else "docs/") + name)
               for name in SOURCE_ANCHORS]
    exports = export_details(jobs, models, sources)
    assert exports["classification_metric_partitions"] == 323
    output = render_report(jobs, exports, BASELINE)
    output = "\n".join(line.rstrip() for line in output.splitlines()) + "\n"
    OUT.write_text(output, encoding="utf-8", newline="\n")
    coverage = {p.relative_to(ROOT).as_posix(): {
        "anchor": SOURCE_ANCHORS[p.name], "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        "body": ("user-maintained submission outline; not numerical experiment evidence" if p.name == 'Gensim&FastText.md' else
                 "completed work summarized in six chapters; complete original tables retained in CSV"),
    } for p in sources}
    evidence = {"built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "baseline": BASELINE, "sources": coverage,
                "primary_paper": {"path": PRIMARY_PAPER.relative_to(ROOT).as_posix(),
                                  "sha256": hashlib.sha256(PRIMARY_PAPER.read_bytes()).hexdigest(),
                                  "version": "arXiv:1607.01759v3, 2016-08-09"},
                "current_runs": len(jobs),
                "word_models": len(models), "external_rows": len(metrics),
                "classification_metric_partitions": exports["classification_metric_partitions"],
                "analogy_relation_rows": exports["analogy_relation_rows"],
                "sentence_scores": exports["sentence_scores"], "csv_exports": exports["inventory"]}
    (ROOT / "artifacts/consolidated_report_coverage.json").write_text(
        json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    return validate()


def validate(report_path=OUT):
    report_path = report_path.resolve()
    content = report_path.read_text(encoding="utf-8")
    code, width, errors, table_count = False, None, [], 0
    chapter_titles = re.findall(r"(?m)^## (.+)$", content)
    if chapter_titles != CHAPTER_TITLES:
        errors.append(f"unexpected report chapters: {chapter_titles}")
    first = content.split('<a id="paper-reading"></a>', 1)[1].split(
        '<a id="paper-reproduction"></a>', 1)[0].strip()
    expected_first = "## 一、FastText论文解读\n\n" + paper_review_body()
    if first != expected_first:
        errors.append("chapter one differs from the approved Markdown literature review")
    if first.count("$$") % 2:
        errors.append("unclosed display math in the paper review")
    reproduction = content.split('<a id="paper-reproduction"></a>', 1)[1].split(
        '<a id="code-reading"></a>', 1)[0].strip()
    if reproduction != "## 二、FastText复现\n\n" + paper_reproduction_body():
        errors.append("chapter two differs from the recorded original-data reproduction")
    headings, image_count, local_links = [], 0, 0
    for number, line in enumerate(content.splitlines(), 1):
        if line.startswith("```"):
            code = not code
            width = None
            continue
        if code:
            continue
        if line.startswith("# "):
            headings.append(number)
        if line.startswith("|"):
            columns = len(re.findall(r"(?<!\\)\|", line)) - 1
            if width is None:
                table_count += 1
                width = columns
            elif columns != width:
                errors.append(f"table width at line {number}: {width} != {columns}")
        else:
            width = None
    if code:
        errors.append("unclosed code fence")
    if len(headings) != 1:
        errors.append(f"expected one document title: {headings}")
    anchors = re.findall(r'<a id="([^"]+)"></a>', content)
    if len(anchors) != len(set(anchors)):
        errors.append("duplicate explicit anchors")
    # Links within code fences are literal data, not active document references.
    prose = re.sub(r"```[^\n]*\n.*?\n```", "", content, flags=re.S)
    for match in re.finditer(r"(!?\[[^\]\n]*\])\(([^)\n]+)\)", prose):
        target = unquote(match.group(2))
        if target.startswith("#"):
            if target[1:] not in anchors:
                errors.append(f"unknown anchor: {target}")
            continue
        if re.match(r"(?:https?://|mailto:)", target):
            continue
        resource = (report_path.parent / target.split("#", 1)[0]).resolve()
        local_links += 1
        if match.group(1).startswith("!"):
            image_count += 1
        if resource.suffix.lower() == ".md":
            errors.append(f"external prose document: {target}")
        if not resource.exists():
            errors.append(f"missing local resource: {target}")
    evidence = read_json(ROOT / "artifacts/consolidated_report_coverage.json")
    paper = evidence.get("primary_paper", {})
    if (paper.get("path") != PRIMARY_PAPER.relative_to(ROOT).as_posix()
            or paper.get("sha256") != hashlib.sha256(PRIMARY_PAPER.read_bytes()).hexdigest()):
        errors.append("primary paper changed or is missing from report provenance")
    # Editorial versions derive from covered reports and add no experiment evidence.
    derived_reports = {OUT.name, f"{OUT.stem}-修改版.md", report_path.name,
                       "Gensim&FastText-修改版.md"}
    expected = {str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "reports").glob("*.md")
                if p.name not in derived_reports | {"Untitled.md"}}
    for path in sorted(expected - evidence["sources"].keys()):
        errors.append(f"uncovered report: {path}")
    for path, item in evidence["sources"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"source changed since report build: {path}")
    jobs = current_jobs()
    numeric_checks = verify_selected_values(content, errors)
    exported_runs = {row["run"] for row in read_csv(DATA / "run_configurations.csv")}
    if exported_runs != set(jobs):
        errors.append("CSV does not match the complete current run set")
    for item in evidence["csv_exports"]:
        path = DATA / item["file"]
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            errors.append(f"missing or changed CSV: {item['file']}")
        elif len(read_csv(path)) != item["rows"]:
            errors.append(f"CSV row count differs: {item['file']}")
    if read_csv(DATA / "csv_index.csv") != [
        {key: str(value) for key, value in item.items()} for item in evidence["csv_exports"]
    ]:
        errors.append("CSV inventory differs from exported resource checksums")
    for name in ["evaluation_file_index.csv", "sentence_score_file_index.csv", "prediction_file_index.csv"]:
        for row in read_csv(DATA / name):
            if not (ROOT / row["path_from_project_root"]).is_file():
                errors.append(f"missing indexed observation file: {row['path_from_project_root']}")
    blocks = re.findall(r"(?m)(?:^\|[^\n]*\n)+", content)
    max_rows = max((len(block.strip().splitlines()) - 2 for block in blocks), default=0)
    if max_rows > 24:
        errors.append(f"detail table leaked into report: {max_rows} rows")
    if re.search(r"(?m)^\| [0-9a-f]{20} \|", content):
        errors.append("raw per-run/model ID rows leaked into report")
    if "```json" in content:
        errors.append("raw JSON dump leaked into report")
    result = {"report": str(report_path), "bytes": report_path.stat().st_size,
              "lines": len(content.splitlines()), "tables": table_count, "images": image_count,
              "local_resources": local_links, "internal_anchors": len(anchors),
              "covered_sources": len(evidence["sources"]), "current_runs": len(jobs),
              "csv_exports": len(evidence["csv_exports"]), "maximum_table_rows": max_rows,
              "numeric_summary_checks": numeric_checks,
              "chapters": chapter_titles, "paper_reproduction_empty": not bool(reproduction),
              "external_markdown_links": 0 if not any("external prose document" in e for e in errors) else "failed",
              "errors": errors}
    check_file = ("consolidated_report_check.json" if report_path == OUT
                  else "consolidated_report_editorial_check.json")
    (ROOT / "artifacts" / check_file).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return result


def verify_selected_values(content, errors):
    """Check published summaries against saved machine values, not table positions."""
    metrics = read_csv(DATA / "classification_metrics.csv")
    names = {"多数类": "majority", "TF-IDF": "tfidf", "LSI": "lsi", "Word2Vec": "word2vec",
             "Doc2Vec": "doc2vec", "Gensim FastText": "fasttext", "原生 FastText": "native_fasttext"}
    main_rows = tables(anchored(content, "main-results"))[0][1]
    resource_rows = tables(anchored(content, "resource-comparison"))[0][1]
    scaling_rows = tables(anchored(content, "scale-results"))[0][1]
    checks = 0

    def group(method, classes, profile):
        found = [r for r in metrics if r["method"] == names[method] and r["classes"] == str(classes)
                 and r["part"] == "test" and any(use["profile"] == profile for use in json.loads(r["uses"]))]
        assert len(found) == (1 if method in {"多数类", "TF-IDF"} else 3), (method, classes, profile)
        return found

    def check(display, values, label, precision=4, aggregate="mean"):
        nonlocal checks
        observed = [float(v.strip()) for v in display.split("±")]
        expected = [max(values) if aggregate == "max" else statistics.mean(values)]
        if len(observed) == 2:
            expected.append(statistics.stdev(values))
        tolerance = 0.51 * 10 ** (-precision)
        if any(abs(actual - wanted) > tolerance for actual, wanted in zip(observed, expected, strict=True)):
            errors.append(f"reported summary differs from machine values: {label}: {display}")
        checks += len(observed)

    for method, *scores in main_rows:
        for classes, score in zip([2, 4, 8, 12], scores, strict=True):
            check(score, [float(r["macro_f1"]) for r in group(method, classes, "main")], f"main {method}/{classes}")
    for method, fit, peak, size in resource_rows:
        found = group(method, 12, "main")
        check(fit, [float(r["fit_seconds"]) for r in found], f"fit {method}", 1)
        check(peak, [float(r["peak_rss_mb"]) for r in found], f"RSS {method}", 1, "max")
        check(size, [float(r["model_bytes"]) / 2**20 for r in found], f"model size {method}", 1)
    for method, _, score, _, fit, size in scaling_rows:
        found = group(method, 12, "expanded")
        check(score, [float(r["macro_f1"]) for r in found], f"expanded {method}")
        check(fit, [float(r["fit_seconds"]) for r in found], f"expanded fit {method}", 1)
        check(size, [float(r["model_bytes"]) / 2**20 for r in found], f"expanded size {method}", 1)
    word_rows = read_csv(DATA / "word_metrics.csv")
    configs = read_csv(DATA / "run_configurations.csv")
    main_ids = {r["run"] for r in configs if r["classes"] == "12"
                and any(u["profile"] == "main" for u in json.loads(r["uses"]))}
    word = anchored(content, "word-similarity")
    for benchmark, method, rho, _, _ in tables(word)[0][1]:
        found = [r for r in word_rows if r["id"] in main_ids and r["method"] == names[method]
                 and r["benchmark"] == benchmark]
        assert len(found) == 3
        check(rho, [float(r["spearman_rho"]) for r in found], f"Spearman {method}/{benchmark}")
    return checks


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate the saved report without rebuilding it")
    parser.add_argument("--report", type=Path, help="Saved report to validate; requires --check")
    args = parser.parse_args()
    if args.report is not None and not args.check:
        parser.error("--report requires --check; report generation always preserves editorial versions")
    validate(args.report or OUT) if args.check else build()
