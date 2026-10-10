"""Independent data, split, metric and fitted-model audit of all current runs."""
from __future__ import annotations

import collections
import csv
import json
import math
from pathlib import Path

import numpy as np

from gensim_lab.common import ARTIFACTS, PROCESSED, REPORTS, ROOT, file_hash, fingerprint, markdown_table, read_json, stamp, write_json
from gensim_lab.data import load_docs, parse_file, simhash
from gensim_lab.integrity import artifact_lock, cache_problem, ensure_dataset, seal_run, validate_splits
from gensim_lab.models import Bundle, metrics
from gensim_lab.reporting import records


def same_numbers(actual, expected, path=""):
    if isinstance(expected, dict):
        for key, value in expected.items():
            same_numbers(actual[key], value, path + "/" + str(key))
    elif isinstance(expected, list):
        assert len(actual) == len(expected), path
        for i, value in enumerate(expected):
            same_numbers(actual[i], value, path + "/" + str(i))
    elif isinstance(expected, (int, float)):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12), (path, actual, expected)
    else:
        assert actual == expected, (path, actual, expected)


def exhaustive_near_pairs(docs):
    """All-pairs Hamming search independent of the production four-block index."""
    values = np.array([int(d["simhash"], 16) for d in docs], dtype=np.uint64)
    table = np.array([i.bit_count() for i in range(256)], dtype=np.uint8)
    result = []
    for i in range(len(docs) - 1):
        bits = np.bitwise_xor(values[i], values[i + 1:])
        distances = table[bits.view(np.uint8).reshape(-1, 8)].sum(axis=1)
        for offset in np.flatnonzero(distances <= 3):
            j = i + 1 + int(offset)
            result.append([docs[i]["id"], docs[j]["id"], int(distances[offset])])
        if i and i % 4000 == 0:
            print(f"ALL-PAIRS {i}/{len(docs)}", flush=True)
    return result


def audit():
    ensure_dataset()
    docs = load_docs()
    lookup = {d["id"]: d for d in docs}
    manifest = read_json(PROCESSED / "manifest.json")
    raw = {}
    for month in manifest["months"]:
        path = ROOT / "data" / "raw" / (month["month"] + ".txt")
        assert file_hash(path) == month["sha256"]
        parsed, stats, _ = parse_file(path)
        assert stats["raw_articles"] == month["raw_articles"]
        raw.update({d["id"]: d for d in parsed})
    excluded = read_json(PROCESSED / "excluded.json")
    omitted = {d["id"] for d in excluded}
    assert len(raw) == manifest["raw_total"] and len(lookup) == manifest["kept_total"]
    assert omitted.isdisjoint(lookup) and omitted | set(lookup) == set(raw)
    hashes = collections.defaultdict(list)
    for d in raw.values():
        hashes[d["content_hash"]].append(d)
    for d in docs:
        for key in ["tokens", "sentences", "text", "content_hash", "page_label", "title"]:
            assert d[key] == raw[d["id"]][key], (d["id"], key)
        assert d["tokens"] == sum(d["sentences"], []), d["id"]
        assert len(d["tokens"]) >= manifest["parameters"]["min_tokens"]
        original = hashes[d["content_hash"]]
        assert len({x["page_label"] for x in original}) == 1
        assert d["id"] == min(x["id"] for x in original)
        assert f"{simhash(d['text']):016x}" == d["simhash"], d["id"]
    for row in excluded:
        group = sorted(hashes[raw[row["id"]]["content_hash"]], key=lambda x: x["id"])
        reason = row["reason"]
        assert (len({x["page_label"] for x in group}) > 1 if reason == "conflicting_exact_duplicate"
                else len(raw[row["id"]]["tokens"]) < manifest["parameters"]["min_tokens"] if reason == "short_article"
                else row["id"] != group[0]["id"]), row
    print("RAW/CLEAN article content and exclusion reasons verified", flush=True)
    pairs = exhaustive_near_pairs(docs)
    saved_pairs = read_json(PROCESSED / "near_duplicate_pairs.json")
    assert sorted(pairs) == sorted(saved_pairs), (pairs, saved_pairs)
    # Independently compare connected components, including isolated articles.
    adjacent = collections.defaultdict(set)
    for a, b, _ in pairs:
        adjacent[a].add(b)
        adjacent[b].add(a)
    seen = set()
    for d in docs:
        if d["id"] in seen:
            continue
        component, queue = set(), [d["id"]]
        while queue:
            node = queue.pop()
            if node in component:
                continue
            component.add(node)
            queue.extend(adjacent[node] - component)
        assert {lookup[i]["group"] for i in component} == {min(component)}
        seen.update(component)
    print("SIMHASH exhaustive all-pairs and connected components verified", flush=True)

    rows = records()
    run_results, frequency_cache = [], {}
    for r in rows:
        spec, directory = r["spec"], r["directory"]
        assert directory.name == fingerprint(spec)[:20]
        validate_splits(spec)
        checked_parts = []
        for part in ["validation", "test"]:
            if not (directory / f"metrics_{part}.json").exists():
                continue
            assert cache_problem(directory, spec, part) is None, (r["run"], part, cache_problem(directory, spec, part))
            with (directory / f"predictions_{part}.csv").open(encoding="utf-8-sig") as f:
                predicted = list(csv.DictReader(f))
            computed = metrics([lookup[x["id"]]["page_label"] for x in predicted],
                               [x["predicted_label"] for x in predicted],
                               [f"{i:02d}" for i in range(1, spec["classes"] + 1)])
            same_numbers(read_json(directory / f"metrics_{part}.json"), computed)
            checked_parts.append({"part": part, "articles": len(predicted)})
        if "source_run" not in spec:
            bundle = Bundle.load(directory)
            assert bundle.spec == spec, r["run"]
            key = fingerprint(spec["splits"]["train"])
            if key not in frequency_cache:
                frequency_cache[key] = (
                    collections.Counter(t for i in spec["splits"]["train"] for t in set(lookup[i]["tokens"])),
                    set(t for i in spec["splits"]["train"] for t in lookup[i]["tokens"]),
                )
            document_frequency, vocabulary = frequency_cache[key]
            if bundle.representation is not None:
                rep = bundle.representation
                if spec["method"] == "doc2vec":
                    from gensim_lab.content_seed import ContentSeedRepresentation
                    assert isinstance(rep, ContentSeedRepresentation)
                assert rep.fit_ids == sorted(spec["splits"]["train"])
                assert rep.dictionary.num_docs == len(spec["splits"]["train"])
                assert set(rep.dictionary.token2id).issubset(vocabulary)
                for word, token in rep.dictionary.token2id.items():
                    assert rep.dictionary.dfs[token] == document_frequency[word]
                    assert math.isclose(rep.tfidf.idfs[token], math.log2(rep.dictionary.num_docs / document_frequency[word]), abs_tol=1e-12)
                if hasattr(rep.model, "wv"):
                    assert set(rep.model.wv.key_to_index).issubset(vocabulary)
            if bundle.native is not None:
                assert set(bundle.native.get_words()).issubset(vocabulary | {"</s>"})
                training = [lookup[i] for i in sorted(spec["splits"]["train"])]
                if "label_permutation_seed" in spec:
                    from gensim_lab.supplemental_worker import permute_training_labels
                    training = permute_training_labels(training, spec["label_permutation_seed"])
                expected_input = "".join(f"__label__p{d['page_label']} {' '.join(d['tokens'])}\n" for d in training)
                assert (directory / "native_train.txt").read_text(encoding="utf-8") == expected_input
            del bundle
        seal_run(directory, spec)
        run_results.append({"run": r["run"], "method": r["method"], "parts": checked_parts, "passed": True})
        if len(run_results) % 20 == 0:
            print(f"MODEL/METRIC {len(run_results)}/{len(rows)}", flush=True)
    result = {"checked_at": stamp(), "raw_articles": len(raw), "clean_articles": len(docs),
              "all_pairs_compared": len(docs) * (len(docs) - 1) // 2,
              "near_duplicate_pairs": pairs, "run_checks": run_results,
              "checks": {
                  "raw_to_clean_content_and_exclusion_trace": True,
                  "sentence_tokens_match_document_tokens": True,
                  "all_saved_simhash_values_recomputed": True,
                  "all_pairs_matches_production_near_groups": True,
                  "run_specs_and_group_isolation": True,
                  "all_prediction_truths_and_metrics_recomputed": True,
                  "loaded_bundle_matches_spec": True,
                  "train_only_dictionary_idf_and_embedding_vocabulary": True,
                  "native_training_input_exactly_matches_train_split": True,
                  "completed_artifacts_checksummed": True,
              }}
    write_json(ARTIFACTS / "deep_audit.json", result)
    md = "# 数据与实验深度核查\n\n"
    md += f"检查于 {result['checked_at']}。原始 {len(raw):,} 篇、保留 {len(docs):,} 篇；当前 {len(rows)} 个运行均通过。核查独立于原报告的汇总逻辑，逐篇核对真实标签和指标，并加载每个训练模型检查训练词典与 IDF。\n\n"
    md += markdown_table(["核查", "结果"], [[key, "通过"] for key in result["checks"]])
    md += f"\n\n对全部文章比较了 {result['all_pairs_compared']:,} 对 SimHash 汉明距离；直接全对比较与生产四块候选索引均得到 {len(pairs)} 对距离不大于 3 的文章，连接分组一致。该结果核验的是指定 SimHash 定义下的完整性，不证明汉明距离大于 3 的文章没有语义上的重复。\n\n"
    md += "保存的标签、逐类 Precision/Recall/F1、混淆矩阵、Accuracy 和 Macro-F1 均从逐篇预测重新计算；词典的文档频率和 log2 IDF 用训练集重新计算。没有使用测试集修改原模型。新建 SHA-256 完整性记录用于后续缓存复用检查。\n\n"
    md += "[逐运行机器记录](../artifacts/deep_audit.json)、[核查脚本](../tools/deep_audit.py)。重跑：`conda run -n Gensim python tools/deep_audit.py`。\n"
    (REPORTS / "数据与实验深度核查.md").write_text(md, encoding="utf-8")
    print(f"Deep audit passed: {len(rows)} runs; {result['all_pairs_compared']:,} document pairs", flush=True)


if __name__ == "__main__":
    with artifact_lock():
        audit()
