"""Export report details as lossless CSV resources, without fitting any model."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "reports/data"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def export_details(jobs, models, sources):
    DATA.mkdir(exist_ok=True)
    inventory = []

    def write(name, rows, description, columns=None):
        rows = list(rows)
        fields = columns or list(dict.fromkeys(key for row in rows for key in row))
        path = DATA / name
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        # Check an actual read-back, including Unicode and scalar precision.
        reread = read_csv(path)
        expected = [{key: "" if row.get(key) is None else str(row.get(key, "")) for key in fields} for row in rows]
        assert reread == expected, name
        inventory.append({"file": name, "rows": len(rows), "columns": len(fields),
                          "description": description,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})

    def flat_params(spec):
        values = {}
        for family in ["dictionary", "embedding", "native"]:
            values.update({f"{family}.{key}": value for key, value in spec.get(family, {}).items()})
        for key in ["source_run", "month", "doc2vec_inference_seed", "label_permutation_seed", "runtime_code", "data_fingerprint", "counter", "loss_counter"]:
            if key in spec:
                values[key] = spec[key]
        return values

    run_rows, configuration_rows, class_rows, matrix_rows, prediction_files = [], [], [], [], []
    partition_count = 0
    for run, job in jobs.items():
        directory = ROOT / "artifacts/runs" / run
        spec, summary = load(directory / "spec.json"), load(directory / "summary.json")
        common = {"run": run, "method": job["method"], "classes": job["classes"], "seed": job["seed"]}
        uses = json.dumps(job["uses"], ensure_ascii=False, separators=(",", ":"))
        configuration_rows.append({**common, "uses": uses, "spec_path": (directory / "spec.json").relative_to(ROOT).as_posix(),
                                   **{f"n_{part}": len(ids) for part, ids in spec["splits"].items()},
                                   **flat_params(spec)})
        best = max(summary.get("tuning", []), key=lambda point: point["validation_macro_f1"], default={})
        resources = {key: summary.get(key) for key in ["completed", "fit_seconds", "wall_seconds", "peak_rss_mb", "model_bytes"]}
        for path in sorted(directory.glob("metrics_*.json")):
            metric = load(path)
            part = path.stem.removeprefix("metrics_")
            partition_count += 1
            prediction = directory / f"predictions_{part}.csv"
            assert prediction.is_file(), prediction
            prediction_files.append({**common, "part": part, "rows": metric["n"],
                                     "path_from_project_root": prediction.relative_to(ROOT).as_posix()})
            run_rows.append({**common, "part": part, "uses": uses, "n": metric["n"],
                             "accuracy": metric["accuracy"], "macro_f1": metric["macro_f1"],
                             "selected_c": best.get("C"), **resources})
            for label, values in metric["per_class"].items():
                if isinstance(values, dict):
                    class_rows.append({**common, "part": part, "label_or_average": label, **values})
            labels = [f"{i:02d}" for i in range(1, job["classes"] + 1)]
            matrix = metric["confusion_matrix"]
            assert sum(map(sum, matrix)) == metric["n"]
            for i, truth in enumerate(labels):
                for j, prediction in enumerate(labels):
                    matrix_rows.append({**common, "part": part, "true_label": truth,
                                        "predicted_label": prediction, "count": matrix[i][j]})
    write("classification_metrics.csv", run_rows, "全部当前运行各分区的分类与资源指标；保留浮点精度")
    write("run_configurations.csv", configuration_rows, "201 个当前运行的完整参数、用途、规模与协议")
    write("per_class_metrics.csv", class_rows, "各版面及宏/加权平均的完整 PRF；含所有种子和分区")
    write("confusion_counts.csv", matrix_rows, "所有分区混淆计数，长表，每行一个真实/预测组合")
    write("prediction_file_index.csv", prediction_files, "323 个完整分区逐篇预测 CSV 的路径、方法、种子与文章数")
    word = ROOT / "artifacts/word_evaluation"
    write("word_metrics.csv", read_csv(word / "metrics.csv"), "594 条词相似度/类比指标；含 p 值、共同词表和 OOV 生成状态")
    write("word_model_configurations.csv", ({"model": m["id"], "origin": m["origin"],
          "method": m["method"], "classes": m["classes"], "seed": m["seed"],
          "uses": json.dumps(m["uses"], ensure_ascii=False, separators=(",", ":")), **flat_params(m["spec"])} for m in models),
          "99 条词表示模型的配置与来源；重训/控制不增加独立重复数")
    relation_rows, evaluation_files = [], []
    for m in models:
        directory = word / "evaluations" / m["id"]
        for benchmark in ["CA_translated", "CA8_morphological", "CA8_semantic"]:
            relation_rows.extend({"model": m["id"], "method": m["method"], "classes": m["classes"],
                                  "seed": m["seed"], "benchmark": benchmark, **row}
                                 for row in read_csv(directory / f"{benchmark}_section_metrics.csv"))
        for path in sorted(directory.glob("*.csv")):
            evaluation_files.append({"model": m["id"], "method": m["method"], "kind": path.stem,
                                     "path_from_project_root": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size})
    write("analogy_relations.csv", relation_rows, "所有模型逐关系类比；保留全部题、覆盖、OOV 与条件/总体准确率")
    write("evaluation_file_index.csv", evaluation_files, "原始逐词对、逐类比题等既有 CSV 的路径索引，不重复复制大文件")
    score_files = []
    for path in sorted(word.rglob("sentence_scores_*.csv")):
        with path.open(encoding="utf-8-sig", newline="") as stream:
            count = sum(1 for _ in csv.DictReader(stream))
        score_files.append({"model_or_run": path.parent.name, "part": path.stem.removeprefix("sentence_scores_"),
                            "rows": count, "path_from_project_root": path.relative_to(ROOT).as_posix()})
    write("sentence_score_file_index.csv", score_files, "421,011 条逐句评分所在既有 CSV；重复核对不视为独立样本")
    for original, name, description in [
        (word / "presentation_selection.csv", "presentation_selection.csv", "全部图表选择输入与规则，包含近乎常量和保留绘图的比较"),
        (word / "word2vec_factors.csv", "word2vec_factors.csv", "Word2Vec 单因素全部取值及内在/分类指标"),
        (word / "fasttext_factors.csv", "fasttext_factors.csv", "Gensim FastText 单因素全部取值及内在/分类指标"),
        (ROOT / "reports/补充实验数值.csv", "supplemental_runs.csv", "所有补充运行的数值，不筛除低分配置"),
    ]:
        write(name, read_csv(original), description)

    # Also preserve each original report table in a normalized CSV. These
    # snapshots retain display precision; machine-native exports above retain
    # the underlying full precision. Paragraphs remain in the source documents.
    displayed = []
    snapshots = {}
    for source in sources:
        if source.parent.name != "reports":
            continue
        title, headers, table_id, row_id = "", None, 0, 0
        for line in source.read_text(encoding="utf-8").splitlines():
            if re.match(r"^#{1,6} ", line):
                title = line.lstrip("# ")
            if not line.startswith("|"):
                headers = None
                continue
            cells = [cell.strip() for cell in re.split(r"(?<!\\)\|", line)[1:-1]]
            if headers is None:
                headers, table_id, row_id = cells, table_id + 1, 0
                continue
            if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            assert len(cells) == len(headers), (source, title)
            row_id += 1
            snapshots.setdefault((source.name, table_id), []).append(dict(zip(headers, cells)))
            for field, value in zip(headers, cells):
                displayed.append({"source_report": source.name, "section": title, "table": table_id,
                                  "row": row_id, "field": field, "display_value": value})
    write("historical_report_tables.csv", displayed, "原报告所有表格的长表快照；展示精度，完整精度优先使用上列机器导出")
    for report, table_id, name, description in [
        ("数据质量报告.md", 1, "monthly_data_quality.csv", "各月文章、控制字符与清洗后规模"),
        ("数据质量报告.md", 3, "january_label_distribution.csv", "一月各版面在训练/验证/测试中的文章数"),
        ("各版面清洗损失.md", 1, "cleaning_by_page.csv", "六个月逐版面的 72 行清洗损失及互斥排除原因"),
    ]:
        write(name, snapshots[(report, table_id)], description)
    write("source_fingerprints.csv", ({"source": p.relative_to(ROOT).as_posix(),
          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in sources), "编制时源文档校验值，用于追溯而非论文结果")
    # The inventory describes every export. It is intentionally not its own
    # checksum entry, avoiding a self-referential hash.
    index_path = DATA / "csv_index.csv"
    with index_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "rows", "columns", "description", "sha256"])
        writer.writeheader()
        writer.writerows(inventory)
    assert set(row["run"] for row in read_csv(DATA / "run_configurations.csv")) == set(jobs)
    assert len(relation_rows) == 1089
    assert sum(int(row["rows"]) for row in score_files) == 421011
    return {"inventory": inventory, "classification_metric_partitions": partition_count,
            "analogy_relation_rows": len(relation_rows), "sentence_scores": 421011}
