from __future__ import annotations

import collections
import hashlib
import json
import math
import re
import unicodedata
import zipfile
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from .common import PROCESSED, REPORTS, ROOT, config, file_hash, fingerprint, markdown_table, read_json, stamp, write_json

LINE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})/m\s*(.*)$")


def parse_body(body):
    """Strip POS/entity annotations while retaining original tokenization and sentence boundaries."""
    sentences, current, tokens = [], [], []
    controls = sum(unicodedata.category(c) == "Cc" for c in body)
    for raw in body.split():
        if "/" not in raw:
            raise ValueError(f"Token without POS suffix: {raw!r}")
        word, pos = raw.rsplit("/", 1)
        pos = pos.split("]", 1)[0]
        word = re.sub(r"^\[+", "", word)
        word = re.sub(r"\][A-Za-z]+$", "", word)
        word = unicodedata.normalize("NFKC", word)
        word = "".join(c for c in word if unicodedata.category(c) != "Cc")
        if pos == "w" or not word or all(unicodedata.category(c)[0] in "PS" for c in word):
            if current and any(c in "。！？!?；;" for c in word):
                sentences.append(current)
                current = []
            continue
        tokens.append(word)
        current.append(word)
        if len(current) >= 1000:
            sentences.append(current)
            current = []
    if current:
        sentences.append(current)
    return tokens, sentences, controls


def parse_file(path):
    articles, stats, anomalies = {}, collections.Counter(), []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            stats["blank_lines"] += 1
            continue
        stats["nonempty_lines"] += 1
        match = LINE.match(line)
        if not match:
            raise ValueError(f"Unrecognized line {path}:{lineno}: {line[:100]}")
        date, page, article, paragraph, body = match.groups()
        key = f"{date}-{page}-{article}"
        if key not in articles:
            articles[key] = {
                "id": key, "date": date, "month": date[:6], "page_label": page,
                "title": "", "tokens": [], "sentences": [], "source_file": Path(path).name,
                "quality_flags": [], "_paragraphs": {},
            }
        doc = articles[key]
        seen = doc["_paragraphs"]
        if paragraph in seen:
            reason = "repeated_paragraph_id" if body != seen[paragraph] else "identical_paragraph"
            doc["quality_flags"].append(reason)
            anomalies.append({"id": key, "line": lineno, "paragraph": paragraph, "reason": reason})
        seen[paragraph] = body
        tokens, sentences, controls = parse_body(body)
        stats["control_characters"] += controls
        if controls:
            doc["quality_flags"].append("control_character")
        if paragraph == "001" and not doc["title"]:
            doc["title"] = "".join(tokens)
        doc["tokens"].extend(tokens)
        doc["sentences"].extend(sentences)
    for doc in articles.values():
        paragraphs = doc.pop("_paragraphs")
        numbers = sorted(map(int, paragraphs))
        if numbers and numbers != list(range(1, max(numbers) + 1)):
            doc["quality_flags"].append("paragraph_gap")
        if "001" not in paragraphs:
            doc["quality_flags"].append("missing_first_paragraph")
        doc["text"] = "".join(doc["tokens"])
        doc["content_hash"] = hashlib.sha256(doc["text"].encode()).hexdigest()
    stats["raw_articles"] = len(articles)
    return list(articles.values()), dict(stats), anomalies


def simhash(text):
    grams = collections.Counter(text[i:i + 3] for i in range(max(1, len(text) - 2)))
    vals = np.array([int.from_bytes(hashlib.blake2b(g.encode(), digest_size=8).digest(), "little") for g in grams], dtype=np.uint64)
    weights = np.array([1 + math.log(n) for n in grams.values()])
    bits = ((vals[:, None] >> np.arange(64, dtype=np.uint64)) & 1).astype(np.float64)
    signs = (2 * bits - 1).T @ weights
    return sum(1 << int(i) for i in np.flatnonzero(signs >= 0))


def near_groups(docs, distance=3):
    if distance != 3:
        raise ValueError("The four-block index guarantees completeness only for distance=3")
    parent = list(range(len(docs)))
    buckets = collections.defaultdict(list)
    hashes, pairs = [], []

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, doc in enumerate(docs):
        h = simhash(doc["text"])
        candidates = set()
        for part in range(4):
            candidates.update(buckets[(part, (h >> (part * 16)) & 65535)])
        for j in candidates:
            if (h ^ hashes[j]).bit_count() <= distance:
                a, b = find(i), find(j)
                parent[max(a, b)] = min(a, b)
                pairs.append([docs[j]["id"], doc["id"], (h ^ hashes[j]).bit_count()])
        hashes.append(h)
        for part in range(4):
            buckets[(part, (h >> (part * 16)) & 65535)].append(i)
    for i, doc in enumerate(docs):
        doc["group"] = docs[find(i)]["id"]
        doc["simhash"] = f"{hashes[i]:016x}"
    return pairs


def january_split(docs, seed=42):
    jan = [d for d in docs if d["month"] == "199801"]
    labels = [d["page_label"] for d in jan]
    groups = [d["group"] for d in jan]
    cv = StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=seed)
    result = {"train": [], "validation": [], "test": []}
    for fold, (_, held) in enumerate(cv.split(np.zeros(len(jan)), labels, groups)):
        name = "train" if fold < 14 else "validation" if fold < 17 else "test"
        result[name].extend(jan[int(i)]["id"] for i in held)
    for name in result:
        result[name].sort()
    return result


def prepare(force=False):
    cfg = config()
    inputs = sorted((ROOT / "data" / "raw").glob("1998??.txt"))
    raw_hashes = {p.name: file_hash(p) for p in inputs}
    data_key = fingerprint({"raw": raw_hashes, "min_tokens": cfg["min_tokens"], "distance": cfg["simhash_distance"], "split_seed": cfg["split_seed"], "parser_sha256": file_hash(Path(__file__))})
    if not force and (PROCESSED / "manifest.json").exists() and read_json(PROCESSED / "manifest.json")["fingerprint"] == data_key:
        print("Prepared dataset cache hit", flush=True)
        return
    if len(inputs) != 6:
        raise ValueError("Expected six monthly text files")
    all_docs, months, anomalies = [], [], []
    for path in inputs:
        docs, stats, issues = parse_file(path)
        all_docs.extend(docs)
        anomalies.extend(issues)
        months.append({"month": path.stem, "sha256": raw_hashes[path.name], **stats})
        print(f"Parsed {path.name}: {len(docs)} articles", flush=True)
    duplicate_map = collections.defaultdict(list)
    for d in all_docs:
        duplicate_map[d["content_hash"]].append(d)
    excluded, kept = [], []
    for group in duplicate_map.values():
        group.sort(key=lambda d: d["id"])
        conflict = len({d["page_label"] for d in group}) > 1
        for i, d in enumerate(group):
            reason = "conflicting_exact_duplicate" if conflict else "short_article" if len(d["tokens"]) < cfg["min_tokens"] else "exact_duplicate" if i else None
            if reason:
                excluded.append({"id": d["id"], "month": d["month"], "page_label": d["page_label"], "reason": reason, "token_count": len(d["tokens"])})
            else:
                kept.append(d)
    kept.sort(key=lambda d: d["id"])
    print(f"Building near-duplicate groups for {len(kept)} articles", flush=True)
    pairs = near_groups(kept, cfg["simhash_distance"])
    splits = january_split(kept, cfg["split_seed"])
    index = {d["id"]: d for d in kept}
    groups = {part: {index[x]["group"] for x in ids} for part, ids in splits.items()}
    assert not (groups["train"] & groups["validation"] or groups["train"] & groups["test"] or groups["test"] & groups["validation"])
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for month in months:
        values = [d for d in kept if d["month"] == month["month"]]
        month["kept_articles"] = len(values)
        month["page_counts"] = dict(sorted(collections.Counter(d["page_label"] for d in values).items()))
        with (PROCESSED / f"{month['month']}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
            for d in values:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    split_counts = {p: dict(sorted(collections.Counter(index[x]["page_label"] for x in ids).items())) for p, ids in splits.items()}
    zip_matches = None
    archive = ROOT / "data" / "raw" / "PFR1998.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            zip_matches = all(hashlib.sha256(z.read(p.name)).hexdigest() == raw_hashes[p.name] for p in inputs)
    manifest = {"fingerprint": data_key, "created": stamp(), "months": months, "raw_total": len(all_docs), "kept_total": len(kept), "exclusion_counts": dict(collections.Counter(x["reason"] for x in excluded)), "near_duplicate_pairs": len(pairs), "near_duplicate_groups": len({d["group"] for d in kept}), "split_counts": split_counts, "zip_matches": zip_matches, "parameters": {k: cfg[k] for k in ["min_tokens", "simhash_distance", "split_seed"]}}
    write_json(PROCESSED / "manifest.json", manifest)
    write_json(PROCESSED / "splits.json", splits)
    write_json(PROCESSED / "excluded.json", excluded)
    write_json(PROCESSED / "anomalies.json", anomalies)
    write_json(PROCESSED / "near_duplicate_pairs.json", pairs)
    REPORTS.mkdir(exist_ok=True)
    md = "# 数据质量报告\n\n" + f"清洗前 {len(all_docs):,} 篇，清洗后 {len(kept):,} 篇。压缩包与原文本一致：{zip_matches}。数据指纹 `{data_key}`。\n\n"
    md += markdown_table(["月份", "原始文章", "保留文章", "非空行", "控制字符"], [[m["month"], m["raw_articles"], m["kept_articles"], m["nonempty_lines"], m["control_characters"]] for m in months])
    md += "\n\n## 排除原因\n\n" + markdown_table(["原因", "数量"], sorted(manifest["exclusion_counts"].items()))
    md += f"\n\n检测到 {len(pairs)} 对近重复文章，归组后划分，三个一月分区的组交集为空。不同内容的重复段落编号按原行顺序保留；清除控制字符，不猜测补写丢失的汉字。\n\n## 一月划分\n\n"
    md += markdown_table(["版面", "训练", "验证", "测试"], [[f"{i:02d}", *[split_counts[p].get(f"{i:02d}", 0) for p in ["train", "validation", "test"]]] for i in range(1, 13)])
    md += "\n\n机器记录：[清单](../data/processed/manifest.json)、[排除记录](../data/processed/excluded.json)、[异常段落](../data/processed/anomalies.json)、[近重复文章](../data/processed/near_duplicate_pairs.json)、[划分](../data/processed/splits.json)。\n"
    (REPORTS / "数据质量报告.md").write_text(md, encoding="utf-8")
    print(f"Prepared {len(kept)} articles; split sizes { {k: len(v) for k,v in splits.items()} }", flush=True)


def load_docs(ids=None, months=None):
    wanted = set(ids) if ids is not None else None
    result = []
    for path in sorted(PROCESSED.glob("1998??.jsonl")):
        if months is not None and path.stem not in months:
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                if wanted is None or d["id"] in wanted:
                    result.append(d)
    result.sort(key=lambda d: d["id"])
    if wanted is not None and len(result) != len(wanted):
        raise ValueError("Some requested article IDs are absent")
    return result
