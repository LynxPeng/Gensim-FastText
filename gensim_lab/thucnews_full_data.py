"""Bounded, resumable full-corpus preprocessing with a fixed held-out set."""
from __future__ import annotations

import collections
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import time
import zipfile

from .common import ROOT, file_hash, fingerprint, read_json, stamp, write_json
from .thucnews_data import PREPARED, RAW, article_category, clean, decoded_name, tokenize

TOKENIZER = None
PARAMETERS = None


def initialize_tokenizer(directory, parameters):
    import jieba
    global TOKENIZER, PARAMETERS
    PARAMETERS = parameters
    TOKENIZER = jieba.Tokenizer()
    TOKENIZER.tmp_dir = directory
    TOKENIZER.cache_file = f"jieba.worker.{os.getpid()}.cache"
    TOKENIZER.initialize()


def process_chunk(items):
    result = []
    for index, name, category, crc, raw in items:
        base = {"input_index": index, "member": name, "category": category}
        try:
            text = clean(raw.decode("utf-8-sig"))
        except UnicodeDecodeError:
            result.append({**base, "reason": "invalid_utf8"})
            continue
        sentences = tokenize(text, TOKENIZER, PARAMETERS)
        tokens = sum(map(len, sentences))
        if tokens < PARAMETERS["minimum_tokens"]:
            result.append({**base, "reason": "short_document"})
            continue
        result.append({**base, "id": hashlib.sha256(name.encode()).hexdigest()[:24],
                       "text_sha256": hashlib.sha256(text.encode()).hexdigest(),
                       "raw_sha256": hashlib.sha256(raw).hexdigest(), "zip_crc32": f"{crc:08x}",
                       "tokens": tokens, "sentences": sentences})
    return result


def ordered_chunks(executor, iterable, capacity):
    """Bound in-flight payloads and emit in input order, independent of worker speed."""
    iterator = iter(iterable)
    pending = collections.deque(executor.submit(process_chunk, chunk)
                                for chunk in itertools.islice(iterator, capacity))
    while pending:
        yield pending.popleft().result()
        next_chunk = next(iterator, None)
        if next_chunk is not None:
            pending.append(executor.submit(process_chunk, next_chunk))


def assign_partition(document, heldout, heldout_hashes, seen):
    digest, identifier = document["text_sha256"], document["id"]
    if digest in heldout_hashes and identifier not in heldout:
        return None  # Prioritize the exact reference article over duplicate copies.
    if digest in seen:
        return None
    if identifier in heldout:
        reference = heldout[identifier]
        assert digest == reference["text_sha256"] and document["sentences"] == reference["sentences"]
        return "holdout"
    return "train"


def prepare_full(source, configuration):
    import jieba
    reference_path = ROOT / configuration["holdout_reference"]
    reference = read_json(reference_path)
    reference_dir = ROOT / reference["directory"]
    assert reference["source"]["sha256"] == source["sha256"]
    assert all(file_hash(reference_dir / name) == digest for name, digest in reference["files"].items())
    heldout = {}
    with (reference_dir / "documents.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            document = json.loads(line)
            if document["part"] == "holdout":
                heldout[document["id"]] = document
    assert len(heldout) == configuration["holdout_documents"]
    heldout_hashes = {d["text_sha256"] for d in heldout.values()}
    dictionary = Path(jieba.__file__).parent / "dict.txt"
    spec = {"archive_sha256": source["sha256"], "sample": configuration,
            "data_code_sha256": file_hash(ROOT / "gensim_lab/thucnews_data.py"),
            "full_data_code_sha256": file_hash(Path(__file__)),
            "reference_manifest_sha256": file_hash(reference_path),
            "jieba_version": jieba.__version__, "jieba_dictionary_sha256": file_hash(dictionary)}
    assert spec["jieba_dictionary_sha256"] == reference["spec"]["jieba_dictionary_sha256"]
    directory = PREPARED / fingerprint(spec)[:20]
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        assert manifest["spec"] == spec
        assert all(file_hash(directory / name) == digest for name, digest in manifest["files"].items())
        print(f"Reuse prepared full corpus: {directory}", flush=True)
        return manifest
    spool, exclusions, checkpoint = directory / "documents.unsorted.jsonl", directory / "exclusions.csv", directory / "prepare_checkpoint.json"
    seen, offsets, statistics = set(), [], collections.defaultdict(collections.Counter)
    excluded = collections.Counter()
    processed = 0
    if checkpoint.exists():
        state = read_json(checkpoint)
        assert state["spec_fingerprint"] == fingerprint(spec)
        for path, size in [(spool, state["spool_bytes"]), (exclusions, state["exclusions_bytes"])]:
            assert path.resolve().is_relative_to(directory.resolve()) and path.stat().st_size >= size
            with path.open("r+b") as stream:
                stream.truncate(size)
        with spool.open("rb") as stream:
            while (position := stream.tell()) < state["spool_bytes"]:
                document = json.loads(stream.readline())
                assert document["text_sha256"] not in seen
                seen.add(document["text_sha256"])
                offsets.append((document["id"], position))
                statistics[document["part"]].update(documents=1, tokens=document["tokens"], sentences=len(document["sentences"]))
                statistics[document["category"]][document["part"]] += 1
        excluded.update(state["excluded"])
        processed = state["processed"]
        print(f"Resume full preparation after {processed} articles", flush=True)
    with zipfile.ZipFile(RAW / "THUCNews.zip") as archive:
        candidates = sorted((i for i in archive.infolist() if article_category(i)), key=lambda i: i.header_offset)
        assert len(candidates) == source["article_files"]
        def raw_chunks():
            for start in range(processed, len(candidates), 128):
                yield [(index, decoded_name(info), article_category(info), info.CRC, archive.read(info))
                       for index, info in enumerate(candidates[start:start + 128], start)]
        workers = configuration["preprocessing_workers"]
        last_status = time.monotonic()
        with spool.open("ab" if processed else "wb") as output, exclusions.open("ab" if processed else "wb") as rejected:
            if not processed:
                rejected.write(b"input_index,member,category,reason\n")
            with ProcessPoolExecutor(max_workers=workers, initializer=initialize_tokenizer,
                                     initargs=(str(directory), configuration)) as executor:
                for chunk in ordered_chunks(executor, raw_chunks(), workers * 2):
                    for document in chunk:
                        if "reason" not in document:
                            part = assign_partition(document, heldout, heldout_hashes, seen)
                            if part is None:
                                document["reason"] = "duplicate_clean_text"
                            else:
                                document["part"] = part
                                seen.add(document["text_sha256"])
                                offsets.append((document["id"], output.tell()))
                                output.write((json.dumps(document, ensure_ascii=False, separators=(",", ":")) + "\n").encode())
                                statistics[part].update(documents=1, tokens=document["tokens"], sentences=len(document["sentences"]))
                                statistics[document["category"]][part] += 1
                        if "reason" in document:
                            excluded[document["reason"]] += 1
                            # Member names contain no comma/newline in the pinned archive.
                            assert not any(c in document["member"] for c in ',\r\n')
                            rejected.write(f"{document['input_index']},{document['member']},{document['category']},{document['reason']}\n".encode())
                        processed = document["input_index"] + 1
                    output.flush()
                    rejected.flush()
                    state = {"spec_fingerprint": fingerprint(spec), "processed": processed,
                             "spool_bytes": output.tell(), "exclusions_bytes": rejected.tell(), "excluded": dict(excluded)}
                    write_json(checkpoint, state)
                    if time.monotonic() - last_status >= 10:
                        progress = {"status": "preparing", "phase": "full_corpus_tokenization", "processed": processed,
                                    "total": len(candidates), "accepted": len(seen), "excluded": dict(excluded),
                                    "directory": directory.relative_to(ROOT).as_posix(), "updated": stamp()}
                        write_json(ROOT / "artifacts/thucnews/progress.json", progress)
                        print(f"Full preparation {processed}/{len(candidates)}: accepted {len(seen)}, excluded {sum(excluded.values())}", flush=True)
                        last_status = time.monotonic()
    assert processed == source["article_files"]
    assert statistics["holdout"]["documents"] == len(heldout)
    assert len(seen) + sum(excluded.values()) == processed
    streams = {part: (directory / f"{part}.txt").open("w", encoding="utf-8", newline="\n") for part in ["train", "holdout"]}
    meta_streams = {part: (directory / f"{part}_sentences.csv").open("w", encoding="utf-8-sig", newline="") for part in streams}
    writers = {part: csv.writer(stream) for part, stream in meta_streams.items()}
    for writer in writers.values():
        writer.writerow(["document_id", "sentence_index", "tokens"])
    print("Write the complete corpus in fixed document hash order", flush=True)
    try:
        with spool.open("rb") as input_stream, (directory / "documents.jsonl").open("wb") as ordered:
            for identifier, position in sorted(offsets):
                input_stream.seek(position)
                line = input_stream.readline()
                ordered.write(line)
                document = json.loads(line)
                for index, sentence in enumerate(document["sentences"]):
                    streams[document["part"]].write(" ".join(sentence) + "\n")
                    writers[document["part"]].writerow([identifier, index, len(sentence)])
    finally:
        for stream in list(streams.values()) + list(meta_streams.values()):
            stream.close()
    for name in ["holdout.txt", "holdout_sentences.csv"]:
        assert file_hash(directory / name) == file_hash(reference_dir / name), "The fixed held-out set changed"
    files = {name: file_hash(directory / name) for name in ["documents.jsonl", "train.txt", "holdout.txt", "train_sentences.csv", "holdout_sentences.csv", "exclusions.csv"]}
    manifest = {"prepared": stamp(), "spec": spec, "directory": directory.relative_to(ROOT).as_posix(),
                "source": source, "processed_articles": processed, "eligible_articles": len(seen),
                "categories": {c: dict(statistics[c]) for c in sorted(source["categories"])},
                "partitions": {p: dict(statistics[p]) for p in ["train", "holdout"]},
                "excluded": dict(excluded), "files": files, "fingerprint": fingerprint({"spec": spec, "files": files}),
                "integrity": "All raw articles traversed and CRC-checked; all accepted clean-text hashes unique; exact reference holdout retained; near-duplicate grouping not applied",
                "context": "Sentence boundaries, maximum 256 tokens; no contexts cross documents",
                "order": "Documents ordered by SHA-256 of archive member name, interleaving categories"}
    write_json(manifest_path, manifest)
    write_json(PREPARED / "latest.json", {"manifest": manifest_path.relative_to(ROOT).as_posix()})
    print(json.dumps(manifest["partitions"], ensure_ascii=False), flush=True)
    return manifest
