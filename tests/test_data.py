import collections

import pytest

from gensim_lab.data import january_split, near_groups, parse_body, parse_file, simhash


def test_nested_entities_and_slashes():
    words, sentences, controls = parse_body("[中央/n 人民/n 广播/vn 电台/n]nt [中国/ns 政府/n]nt http://test/n 。/w 体育/n ！/w")
    assert words == ["中央", "人民", "广播", "电台", "中国", "政府", "http://test", "体育"]
    assert sentences[-1] == ["体育"]
    assert controls == 0


def test_normalization_controls_and_paragraph_conflict(tmp_path):
    f = tmp_path / "199801.txt"
    f.write_text("19980101-01-001-001/m  Ａ/n 闭\x16/v 。/w\n19980101-01-001-001/m  保留/v 两行/n\n19980101-01-002-002/m  无/r 标题/n\n", encoding="utf-8")
    docs, stats, issues = parse_file(f)
    assert docs[0]["tokens"] == ["A", "闭", "保留", "两行"]
    assert stats["control_characters"] == 1
    assert issues[0]["reason"] == "repeated_paragraph_id"
    assert "missing_first_paragraph" in docs[1]["quality_flags"]


def test_simhash_and_group_completeness():
    text = "这是用于测试近重复新闻文章的中文内容" * 20
    docs = [{"id": "a", "text": text}, {"id": "b", "text": text}, {"id": "c", "text": "独立的体育篮球报道" * 20}]
    assert simhash(text) == simhash(text)
    pairs = near_groups(docs)
    assert docs[0]["group"] == docs[1]["group"]
    assert ["a", "b", 0] in pairs


def test_stratified_group_split_deterministic():
    docs = [{"id": f"{label}-{i:03d}", "month": "199801", "page_label": f"{label:02d}", "group": f"{label}-{i//2:03d}"} for label in [1, 2] for i in range(80)]
    split = january_split(docs)
    assert split == january_split(docs)
    assert sum(map(len, split.values())) == len(docs)
    lookup = {d["id"]: d for d in docs}
    groups = [{lookup[x]["group"] for x in ids} for ids in split.values()]
    assert not groups[0] & groups[1]
    assert not groups[1] & groups[2]
    assert not groups[0] & groups[2]


def test_near_group_threshold_and_transitive_component(monkeypatch):
    # A-B and B-C are within 3 bits; A-C is 6 bits. The whole component must stay together.
    import gensim_lab.data as data
    a = 0x9A72BCD018EF4563
    b = a ^ (1 << 0) ^ (1 << 17) ^ (1 << 34)
    c = b ^ (1 << 1) ^ (1 << 18) ^ (1 << 49)
    unrelated = a ^ 0xffffffffffffffff
    known = {"a": a, "b": b, "c": c, "d": unrelated}
    monkeypatch.setattr(data, "simhash", lambda text: known[text])
    docs = [{"id": name, "text": name} for name in known]
    pairs = data.near_groups(docs)
    assert ["a", "b", 3] in pairs and ["b", "c", 3] in pairs
    assert docs[0]["group"] == docs[1]["group"] == docs[2]["group"]
    assert docs[3]["group"] != docs[0]["group"]


def test_split_groups_can_contain_multiple_labels():
    docs = [{"id": f"{i:03d}-{label}", "month": "199801", "page_label": f"{label:02d}", "group": f"shared-{i:03d}"} for i in range(80) for label in [1, 2]]
    split = january_split(docs)
    lookup = {d["id"]: d for d in docs}
    assignment = {}
    for part, ids in split.items():
        for doc_id in ids:
            group = lookup[doc_id]["group"]
            if group in assignment:
                assert assignment[group] == part
            assignment[group] = part
    assert len(assignment) == 80
