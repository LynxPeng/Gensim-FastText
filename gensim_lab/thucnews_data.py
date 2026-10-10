"""THUCNews archive verification and deterministic document sampling."""
from __future__ import annotations

import collections
import csv
import hashlib
import html
import json
import math
import random
import re
import time
import unicodedata
import zipfile
from pathlib import Path, PurePosixPath

import yaml
from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json

RAW = ROOT / "data/thucnews/raw"
PREPARED = ROOT / "data/thucnews/prepared"
CONFIG = ROOT / "configs/thucnews.yaml"
CATEGORIES = {"财经", "彩票", "房产", "股票", "家居", "教育", "科技", "社会", "时尚", "时政", "体育", "星座", "游戏", "娱乐"}


def settings():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def decoded_name(info):
    name = info.filename
    if not info.flag_bits & 0x800:
        try:
            name = name.encode("cp437").decode("utf-8")
        except (UnicodeError, LookupError):
            pass
    return name


def article_category(info):
    name = decoded_name(info)
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe archive member: {name}")
    if info.is_dir() or "__MACOSX" in path.parts or path.name.startswith("._") or path.suffix != ".txt":
        return None
    if path.parent.name not in CATEGORIES:
        raise ValueError(f"Unrecognized THUCNews category: {name}")
    return path.parent.name


def fetch():
    source = settings()["source"]
    RAW.mkdir(parents=True, exist_ok=True)
    archive, partial = RAW / "THUCNews.zip", RAW / "THUCNews.zip.part"
    if not archive.exists():
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > source["bytes"]:
            raise ValueError("Partial archive exceeds the declared file size")
        if offset < source["bytes"]:
            import requests
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
                headers["If-Range"] = '"' + source["etag"] + '"'
            with requests.get(source["url"], headers=headers, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                if response.headers.get("ETag", "").strip('"') != source["etag"]:
                    raise ValueError("THUCNews server object changed; preserve the partial file for review")
                if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                    raise ValueError("Server did not honor the resumable download range")
                last = time.monotonic()
                with partial.open("ab" if offset else "wb") as stream:
                    for chunk in response.iter_content(4 * 1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if time.monotonic() - last >= 10:
                            print(f"Download {offset}/{source['bytes']} bytes", flush=True)
                            last = time.monotonic()
        if partial.stat().st_size != source["bytes"]:
            raise ValueError("THUCNews download is incomplete")
        partial.replace(archive)
    if archive.stat().st_size != source["bytes"]:
        raise ValueError("THUCNews archive size differs from the pinned public object")
    digest = file_hash(archive)
    manifest_path = RAW / "source_manifest.json"
    if manifest_path.exists():
        previous = read_json(manifest_path)
        if previous["sha256"] != digest:
            raise ValueError("Local THUCNews archive changed since acquisition")
        return previous
    counts, ignored, raw_bytes = collections.Counter(), 0, 0
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            category = article_category(info)
            if category is None:
                ignored += 1
            else:
                counts[category] += 1
                raw_bytes += info.file_size
        total_members = len(zipped.infolist())
    if set(counts) != CATEGORIES:
        raise ValueError("The archive does not contain the expected 14 categories")
    manifest = {"retrieved": stamp(), "url": source["url"], "etag": source["etag"],
                "bytes": archive.stat().st_size, "sha256": digest, "zip_members": total_members,
                "ignored_members": ignored, "article_files": sum(counts.values()),
                "article_bytes": raw_bytes, "categories": dict(sorted(counts.items())),
                "filename_encoding": "UTF-8 bytes without ZIP UTF-8 flag; repair cp437 decoding",
                "integrity": "Archive SHA-256 and central-directory size checks; every selected member is CRC-checked by ZipFile.read",
                "description_source": "https://github.com/thunlp/THUCTC"}
    write_json(manifest_path, manifest)
    print(f"Archive verified: {manifest['article_files']} article files, 14 categories", flush=True)
    return manifest


def quotas(counts, total):
    if total > sum(counts.values()) or total < 0:
        raise ValueError("Requested sample is larger than the available corpus")
    denominator = sum(counts.values())
    exact = {name: total * value / denominator for name, value in counts.items()}
    result = {name: math.floor(value) for name, value in exact.items()}
    order = sorted(counts, key=lambda name: (-(exact[name] - result[name]), name))
    for name in order[:total - sum(result.values())]:
        result[name] += 1
    assert sum(result.values()) == total and all(result[name] <= counts[name] for name in result)
    return result


def clean(text):
    text = unicodedata.normalize("NFKC", html.unescape(text))
    text = re.sub(r"<[^>\n]{1,200}>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = "".join(c for c in text if c in "\n\t" or unicodedata.category(c)[0] != "C")
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def tokenize(text, tokenizer, parameters):
    sentences = []
    for fragment in re.split(r"(?<=[。！？!?；;])|\n+", text):
        tokens = [t for t in tokenizer.cut(fragment, HMM=parameters["tokenizer_hmm"])
                  if re.search(r"[\u3400-\u9fffA-Za-z0-9]", t) and not any(c.isspace() for c in t)]
        width = parameters["maximum_sentence_tokens"]
        sentences.extend(tokens[i:i + width] for i in range(0, len(tokens), width) if tokens[i:i + width])
    return sentences


def prepare(source, *, documents=None):
    import jieba
    configuration = settings()["sample"].copy()
    if documents is None and configuration.get("mode") == "full":
        from .thucnews_full_data import prepare_full
        return prepare_full(source, configuration)
    if documents is not None:
        configuration["documents"] = documents
        configuration["holdout_documents"] = max(1, documents // 10)
        configuration["mode"] = "sample"
    dictionary = Path(jieba.__file__).parent / "dict.txt"
    spec = {"archive_sha256": source["sha256"], "sample": configuration,
            "data_code_sha256": file_hash(Path(__file__)), "jieba_version": jieba.__version__,
            "jieba_dictionary_sha256": file_hash(dictionary)}
    directory = PREPARED / fingerprint(spec)[:20]
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest["spec"] == spec and all(file_hash(directory / name) == digest for name, digest in manifest["files"].items()):
            print(f"Reuse prepared sample: {directory}", flush=True)
            return manifest
        raise ValueError("Prepared THUCNews cache failed its integrity checks")
    directory.mkdir(parents=True, exist_ok=True)
    wanted = quotas(source["categories"], configuration["documents"])
    holdout = quotas(wanted, configuration["holdout_documents"])
    tokenizer = jieba.Tokenizer()
    tokenizer.tmp_dir = str(directory)
    tokenizer.initialize()
    seen, offsets, excluded, statistics = set(), [], [], collections.defaultdict(collections.Counter)
    temporary = directory / "documents.unsorted.jsonl"
    all_accepted = 0
    with zipfile.ZipFile(RAW / "THUCNews.zip") as zipped, temporary.open("wb") as output:
        by_category = collections.defaultdict(list)
        for info in zipped.infolist():
            category = article_category(info)
            if category is not None:
                by_category[category].append(info)
        for category in sorted(by_category):
            candidates = sorted(by_category[category], key=decoded_name)
            random.Random(f"{configuration['seed']}:{category}").shuffle(candidates)
            accepted = 0
            for info in candidates:
                if accepted == wanted[category]:
                    break
                name = decoded_name(info)
                raw = zipped.read(info)  # zipfile verifies this member's stored CRC.
                try:
                    text = clean(raw.decode("utf-8-sig"))
                except UnicodeDecodeError:
                    excluded.append({"member": name, "category": category, "reason": "invalid_utf8"})
                    continue
                digest = hashlib.sha256(text.encode()).hexdigest()
                if digest in seen:
                    excluded.append({"member": name, "category": category, "reason": "duplicate_clean_text"})
                    continue
                sentences = tokenize(text, tokenizer, configuration)
                tokens = sum(map(len, sentences))
                if tokens < configuration["minimum_tokens"]:
                    excluded.append({"member": name, "category": category, "reason": "short_document"})
                    continue
                seen.add(digest)
                part = "holdout" if accepted < holdout[category] else "train"
                identifier = hashlib.sha256(name.encode()).hexdigest()[:24]
                record = {"id": identifier, "category": category, "member": name, "part": part,
                          "text_sha256": digest, "raw_sha256": hashlib.sha256(raw).hexdigest(),
                          "zip_crc32": f"{info.CRC:08x}", "tokens": tokens, "sentences": sentences}
                offset = output.tell()
                output.write((json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                offsets.append((identifier, offset))
                statistics[part]["documents"] += 1
                statistics[part]["tokens"] += tokens
                statistics[part]["sentences"] += len(sentences)
                statistics[category][part] += 1
                accepted += 1
                all_accepted += 1
                if all_accepted % 1000 == 0:
                    print(f"Prepare {all_accepted}/{configuration['documents']} documents", flush=True)
            if accepted != wanted[category]:
                raise ValueError(f"Insufficient eligible articles in {category}: {accepted}/{wanted[category]}")
    streams = {part: (directory / f"{part}.txt").open("w", encoding="utf-8", newline="\n") for part in ["train", "holdout"]}
    metadata_streams = {part: (directory / f"{part}_sentences.csv").open("w", encoding="utf-8-sig", newline="") for part in streams}
    writers = {part: csv.writer(stream) for part, stream in metadata_streams.items()}
    for writer in writers.values():
        writer.writerow(["document_id", "sentence_index", "tokens"])
    try:
        with temporary.open("rb") as source_stream, (directory / "documents.jsonl").open("wb") as ordered:
            for identifier, offset in sorted(offsets):
                source_stream.seek(offset)
                line = source_stream.readline()
                ordered.write(line)
                doc = json.loads(line)
                for index, sentence in enumerate(doc["sentences"]):
                    streams[doc["part"]].write(" ".join(sentence) + "\n")
                    writers[doc["part"]].writerow([identifier, index, len(sentence)])
    finally:
        for stream in list(streams.values()) + list(metadata_streams.values()):
            stream.close()
    from .evaluation_support import save_csv
    save_csv(directory / "exclusions.csv", excluded, ["member", "category", "reason"])
    files = {name: file_hash(directory / name) for name in ["documents.jsonl", "train.txt", "holdout.txt", "train_sentences.csv", "holdout_sentences.csv", "exclusions.csv"]}
    manifest = {"prepared": stamp(), "spec": spec, "directory": directory.relative_to(ROOT).as_posix(),
                "source": source, "categories": {category: dict(statistics[category]) for category in sorted(wanted)},
                "partitions": {part: dict(statistics[part]) for part in ["train", "holdout"]},
                "excluded": dict(collections.Counter(row["reason"] for row in excluded)), "files": files,
                "fingerprint": fingerprint({"spec": spec, "files": files}),
                "integrity": "All accepted documents have distinct normalized-text hashes across both partitions; near-duplicate grouping is not applied",
                "context": "Sentence boundaries, maximum 256 tokens; no contexts cross documents",
                "order": "Documents ordered by SHA-256 of archive member name, interleaving categories"}
    assert sum(p["documents"] for p in manifest["partitions"].values()) == configuration["documents"]
    assert manifest["partitions"]["holdout"]["documents"] == configuration["holdout_documents"]
    write_json(manifest_path, manifest)
    write_json(PREPARED / "latest.json", {"manifest": manifest_path.relative_to(ROOT).as_posix()})
    print(json.dumps(manifest["partitions"], ensure_ascii=False), flush=True)
    return manifest
