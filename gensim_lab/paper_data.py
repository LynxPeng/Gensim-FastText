"""Streaming original-paper datasets, with exact counts and source fingerprints."""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json

RAW = ROOT / "data/paper_reproduction/raw"
PREPARED = ROOT / "data/paper_reproduction/prepared"
PAPER_ARTIFACTS = ROOT / "artifacts/paper_reproduction"
PAPER_CONFIG = ROOT / "configs/paper_reproduction.yaml"
ASCII_WHITESPACE = re.compile(r"[ \t\n\r\v\f\x00]+")


def ascii_words(text):
    # fastText Dictionary::readWord deliberately does not split Unicode spaces.
    return [word for word in ASCII_WHITESPACE.split(text) if word]


def settings():
    return yaml.safe_load(PAPER_CONFIG.read_text(encoding="utf-8"))


def normalize_classification(line):
    # Port the archived author's normalize_text pipeline; commas are features.
    text = line.translate(str.maketrans("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"))
    text = text.replace("'", " ' ").replace('"', "").replace(".", " . ").replace("<br />", " ")
    for mark in ",()!?":
        text = text.replace(mark, f" {mark} ")
    text = text.replace(";", " ").replace(":", " ")
    fields = ascii_words(text)
    if not fields or not fields[0].isdigit():
        raise ValueError("Classification row must begin with an integer label")
    return [fields[0]], fields[1:]


def parse_labeled(line):
    labels, words = [], []
    for token in ascii_words(line):
        if token.startswith("__label__"):
            labels.append(token[len("__label__"):])
        else:
            words.append(token)
    return labels, words


def rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        for index, line in enumerate(stream):
            labels, words = parse_labeled(line)
            yield index, labels, words


def label_scan(path, *, known_labels=None, known_words=None, filtered_path=None):
    counts = collections.Counter()
    n = tokens = empty = eligible = 0
    digest = hashlib.sha256()
    output = Path(filtered_path).open("w", encoding="utf-8", newline="\n") if filtered_path else None
    try:
        for index, labels, words in rows(path):
            retained = [label for label in labels if known_labels is None or label in known_labels]
            counts.update(retained)
            tokens += len(words)
            empty += not bool(words)
            digest.update((" ".join(words) + "\n").encode())
            has_text = bool(words) if known_words is None else any(word in known_words for word in words)
            if retained and has_text:
                eligible += 1
                if output:
                    output.write(" ".join("__label__" + label for label in retained) + " " + " ".join(words) + "\n")
            n = index + 1
            if n % 1000000 == 0:
                print(f"Scan {Path(path).name}: {n:,} rows, {eligible:,} eligible", flush=True)
    finally:
        if output:
            output.close()
    return {"rows": n, "eligible_rows": eligible, "tokens": tokens, "empty_texts": empty,
            "label_counts": dict(sorted(counts.items())), "text_stream_sha256": digest.hexdigest()}


def source_ready(name):
    return (RAW / name / "source_manifest.json").exists()


def verify_source(name):
    source = RAW / name
    record = read_json(source / "source_manifest.json")
    entries = record.get("files")
    if isinstance(entries, list):
        paths = [(ROOT / entry["path"], entry) for entry in entries]
    elif isinstance(entries, dict):
        paths = [(source / filename, entry) for filename, entry in entries.items()]
    else:
        paths = [(ROOT / entry["path"], entry) for entry in record.values()
                 if isinstance(entry, dict) and "path" in entry]
    if not paths:
        raise ValueError(f"No checksummed original files: {name}")
    checked = []
    for path, entry in paths:
        if not path.resolve().is_relative_to(source.resolve()):
            raise ValueError("Source manifest path escapes its dataset directory")
        if not path.exists() or path.stat().st_size != entry["bytes"] or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Original data integrity failure: {path}")
        checked.append({"path": path.relative_to(ROOT).as_posix(), "sha256": entry["sha256"], "bytes": entry["bytes"]})
    write_json(PAPER_ARTIFACTS / f"source_check_{name}.json", {"checked": stamp(), "files": checked})
    return record


def prepare_classification(name, *, smoke=False):
    parameters = settings()["classification"]["datasets"][name]
    source = RAW / name
    if not source_ready(name):
        raise FileNotFoundError(f"Original dataset download is incomplete: {name}")
    destination = PREPARED / ("smoke" if smoke else "full") / name
    destination.mkdir(parents=True, exist_ok=True)
    source_record = verify_source(name)
    spec = {"dataset": name, "source": source_record, "smoke": smoke,
            "preparation_sha256": file_hash(Path(__file__))}
    manifest_path = destination / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["fingerprint"] == fingerprint(spec) and all(
            (destination / part["file"]).exists() and file_hash(destination / part["file"]) == part["sha256"]
            for part in manifest["partitions"].values()):
            return manifest
    partitions = {}
    for part in ["train", "test"]:
        path = destination / f"{part}.txt"
        counts, count, token_count = collections.Counter(), 0, 0
        limit = 400 if smoke else None
        # Smoke uses early and late rows so class-blocked CSVs remain multiclass.
        stride = max(1, parameters[part] // 400) if smoke else 1
        with (source / f"{part}.csv").open(encoding="utf-8", newline="") as input_stream, path.open("w", encoding="utf-8", newline="\n") as output:
            raw_count = 0
            for raw_index, line in enumerate(input_stream):
                raw_count += 1
                if raw_index % stride:
                    continue
                labels, words = normalize_classification(line)
                if len(labels) != 1 or not 1 <= int(labels[0]) <= parameters["classes"]:
                    raise ValueError(f"Unexpected label in {name}: {labels}")
                output.write(f"__label__{labels[0]} {' '.join(words)}\n")
                counts.update(labels)
                token_count += len(words)
                count += 1
        if raw_count != parameters[part]:
            raise ValueError(f"Original row count differs: {name}/{part}: {raw_count} != {parameters[part]}")
        partitions[part] = {"file": path.name, "sha256": file_hash(path), "rows": count,
                            "original_rows": raw_count, "tokens": token_count, "label_counts": dict(counts)}
        print(f"Prepared {name}/{part}: {count:,} rows, {token_count:,} tokens", flush=True)
    labels = sorted(partitions["train"]["label_counts"], key=int)
    if len(labels) != parameters["classes"]:
        raise ValueError("Missing training classes")
    manifest = {"created": stamp(), "fingerprint": fingerprint(spec), "spec": spec,
                "directory": destination.relative_to(ROOT).as_posix(), "labels": labels,
                "partitions": partitions, "protocol": "original train/test, author-published learning rate; no test-driven tuning"}
    write_json(manifest_path, manifest)
    return manifest


def prepare_tags(*, smoke=False):
    source = RAW / "yfcc100m"
    if not source_ready("yfcc100m"):
        raise FileNotFoundError("YFCC download/extraction has not completed")
    # Match the actual archive names rather than assuming split suffixes.
    candidates = [p for p in source.iterdir() if p.is_file() and p.name in {"train", "test", "valid"}]
    names = {}
    for part, synonyms in {"train": ["train"], "validation": ["valid", "val", "dev"], "test": ["test"]}.items():
        found = [p for p in candidates if any(word in p.stem.lower() for word in synonyms)]
        if len(found) != 1:
            raise ValueError(f"Cannot identify YFCC {part}: {[p.name for p in found]}")
        names[part] = found[0]
    destination = PREPARED / ("smoke" if smoke else "full") / "yfcc100m"
    destination.mkdir(parents=True, exist_ok=True)
    vocabulary_path = PAPER_ARTIFACTS / "yfcc_vocabulary.json"
    if not vocabulary_path.exists():
        raise FileNotFoundError("Build the full original-frequency vocabulary with tools/probe_yfcc_vocabulary.py first")
    spec = {"source": verify_source("yfcc100m"), "smoke": smoke,
            "vocabulary_sha256": file_hash(vocabulary_path), "preparation_sha256": file_hash(Path(__file__))}
    cached = destination / "manifest.json"
    if cached.exists():
        manifest = read_json(cached)
        if manifest["fingerprint"] == fingerprint(spec) and all(
                partition_path(manifest, part).exists() and file_hash(partition_path(manifest, part)) == entry["sha256"]
                for part, entry in manifest["partitions"].items()):
            return manifest
    vocabulary = read_json(vocabulary_path)
    known_labels, known_words = set(vocabulary["labels"]), set(vocabulary["words"]) - {"</s>"}
    if vocabulary["label_count"] != settings()["tags"]["labels"]:
        raise ValueError(f"Original-frequency label count differs: {vocabulary['label_count']}")
    partitions = {}
    for part, path in names.items():
        if smoke:
            output = destination / f"{part}.txt"
            with path.open(encoding="utf-8") as source_stream, output.open("w", encoding="utf-8") as dest:
                for index, line in enumerate(source_stream):
                    if index == 400:
                        break
                    labels, words = parse_labeled(line)
                    labels = [label for label in labels if label in known_labels]
                    dest.write(" ".join([*("__label__" + label for label in labels), *words]) + "\n")
            path = output
        filtered_path = destination / f"{part}_eligible.txt" if part != "train" else None
        result = label_scan(path, known_labels=known_labels, known_words=known_words, filtered_path=filtered_path)
        expected_raw = settings()["tags"]["raw_test"] if part == "test" else settings()["tags"][part]
        if not smoke and result["rows"] != expected_raw:
            raise ValueError(f"YFCC {part} does not match original size: {result['rows']}")
        result["original_rows"] = result["rows"]
        if filtered_path:
            path = filtered_path
            result["rows"] = result["eligible_rows"]
        if not smoke and part == "test" and result["rows"] != settings()["tags"]["test"]:
            raise ValueError(f"YFCC eligible test denominator differs from paper: {result['rows']}")
        if not smoke and part == "validation" and result["rows"] != settings()["tags"]["effective_validation"]:
            raise ValueError("YFCC effective validation count differs from the audited parser")
        result.update(path=path.relative_to(ROOT).as_posix(), sha256=file_hash(path))
        partitions[part] = result
        if smoke and part == "train":
            known_labels = set(result["label_counts"])
    label_counts = partitions["train"]["label_counts"] if smoke else vocabulary["labels"]
    labels = sorted(label_counts, key=lambda label: (-label_counts[label], label))
    if not smoke and len(labels) != settings()["tags"]["labels"]:
        raise ValueError(f"YFCC label vocabulary differs: {len(labels)}")
    manifest = {"created": stamp(), "fingerprint": fingerprint(spec), "spec": spec,
                "directory": destination.relative_to(ROOT).as_posix(), "labels": labels,
                "partitions": partitions, "training_label_counts": label_counts,
                "vocabulary_sha256": file_hash(vocabulary_path),
                "protocol": "original full training data; evaluation requires retained tags and retained text, minCount/minCountLabel 100; no resampling"}
    write_json(cached, manifest)
    return manifest


def partition_path(manifest, part):
    entry = manifest["partitions"][part]
    return ROOT / entry["path"] if "path" in entry else ROOT / manifest["directory"] / entry["file"]


class TextCorpus:
    """Labels never enter Gensim's unsupervised corpus."""
    def __init__(self, path):
        self.path = Path(path)

    def __iter__(self):
        for _, _, words in rows(self.path):
            yield words


def unlabeled_corpus(manifest):
    directory = ROOT / manifest["directory"]
    path = directory / "gensim_train.txt"
    receipt = directory / "gensim_corpus.json"
    if receipt.exists() and path.exists():
        previous = read_json(receipt)
        if previous["data_fingerprint"] == manifest["fingerprint"] and file_hash(path) == previous["sha256"]:
            return path
    temporary = path.with_suffix(".txt.part")
    count = tokens = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        for index, _, words in rows(partition_path(manifest, "train")):
            stream.write(" ".join(words) + "\n")
            count += 1
            tokens += len(words)
            if count % 1000000 == 0:
                print(f"Unlabeled training corpus: {count:,} documents", flush=True)
    if count != manifest["partitions"]["train"]["rows"]:
        raise ValueError("Corpus row count changed")
    temporary.replace(path)
    write_json(receipt, {"data_fingerprint": manifest["fingerprint"], "rows": count,
                         "tokens": tokens, "sha256": file_hash(path)})
    return path
