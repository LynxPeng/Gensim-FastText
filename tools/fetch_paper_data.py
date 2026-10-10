"""Fetch verified AG News CSVs and inspect the official public YFCC archive."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import sys
import time
import zipfile
import tarfile

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/paper_reproduction/raw"
AUDIT = ROOT / "artifacts/paper_reproduction"
AG_MD5 = {"train": "b1a00f826fdfbd249f79597b59e1dc12", "test": "d52ea96a97a2d943681189a97654912d"}


def fetch_ag():
    directory = RAW / "ag_news"
    directory.mkdir(parents=True, exist_ok=True)
    records = {}
    for part, md5 in AG_MD5.items():
        url = f"https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/{part}.csv"
        target = directory / f"{part}.csv"
        if not target.exists() or hashlib.md5(target.read_bytes()).hexdigest() != md5:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                temporary = target.with_suffix(".csv.part")
                with temporary.open("wb") as stream:
                    for chunk in response.iter_content(1024 * 1024):
                        stream.write(chunk)
                if hashlib.md5(temporary.read_bytes()).hexdigest() != md5:
                    raise ValueError(f"AG News {part} differs from the torchtext documented MD5")
                temporary.replace(target)
        records[part] = {"path": target.relative_to(ROOT).as_posix(), "url": url,
                         "md5": md5, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                         "bytes": target.stat().st_size,
                         "checksum_reference": "https://docs.pytorch.org/text/stable/_modules/torchtext/datasets/ag_news.html"}
        print(json.dumps({part: records[part]}, ensure_ascii=False), flush=True)
    (directory / "source_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_zip():
    # Public shared download route: no user account or stored credentials.
    url = "https://fb-public.app.box.com/index.php?rm=box_download_shared_file&shared_name=htfdbrvycvroebv9ecaezaztocbcnsdn&file_id=f_108767604211"
    record = {"public_share": "https://fb-public.box.com/s/htfdbrvycvroebv9ecaezaztocbcnsdn",
              "download_route": url, "declared_archive_bytes": 7334065320}
    try:
        with requests.get(url, headers={"Range": "bytes=7333999784-7334065319"},
                          stream=True, timeout=(15, 45)) as response:
            record.update(status=response.status_code,
                          content_type=response.headers.get("Content-Type"),
                          content_range=response.headers.get("Content-Range"),
                          accepts_ranges=response.headers.get("Accept-Ranges"))
            data = next(response.iter_content(65536), b"")
            if response.status_code == 206 and data:
                (AUDIT / "yfcc_zip_tail.bin").write_bytes(data)
                record.update(tail_bytes=len(data), tail_sha256=hashlib.sha256(data).hexdigest())
            elif "html" in (record["content_type"] or ""):
                (AUDIT / "yfcc_download_response.html").write_bytes(data)
                record["title"] = re.findall(r"<title>(.*?)</title>", data.decode("utf-8", errors="replace"), re.S)[:1]
    except requests.RequestException as error:
        record["error"] = str(error)
    (AUDIT / "yfcc_zip_probe.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


def download(url, target, expected_bytes=None):
    """Resume a public download without trusting an ignored Range header."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.exists() and (expected_bytes is None or target.stat().st_size == expected_bytes):
        return
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(url, headers=headers, stream=True, timeout=(20, 90)) as response:
        response.raise_for_status()
        if offset and (response.status_code != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
            raise ValueError("Server ignored/misapplied resume Range; partial file preserved")
        if not offset and response.status_code == 206:
            raise ValueError("Unexpected partial response for a full-file request")
        if "html" in response.headers.get("Content-Type", ""):
            raise ValueError("Download returned HTML instead of an archive")
        started, last_print, done = time.monotonic(), time.monotonic(), offset
        with partial.open("ab" if offset else "wb") as stream:
            for chunk in response.iter_content(4 * 1024 * 1024):
                stream.write(chunk)
                done += len(chunk)
                if time.monotonic() - last_print > 20:
                    rate = (done - offset) / max(0.001, time.monotonic() - started) / 2**20
                    print(f"{target.name}: {done / 2**30:.3f} GiB, {rate:.2f} MiB/s", flush=True)
                    last_print = time.monotonic()
        if expected_bytes is not None and done != expected_bytes:
            raise ValueError(f"Incomplete archive: {done} != {expected_bytes}")
        partial.replace(target)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_yfcc():
    directory = RAW / "yfcc100m"
    target = directory / "YFCC100M.zip"
    url = "https://fb-public.app.box.com/index.php?rm=box_download_shared_file&shared_name=htfdbrvycvroebv9ecaezaztocbcnsdn&file_id=f_108767604211"
    download(url, target, 7334065320)
    manifest_path = directory / "source_manifest.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    previous_files = {Path(entry["path"]).name: entry for entry in previous.get("files", [])}
    records = {"public_share": "https://fb-public.box.com/s/htfdbrvycvroebv9ecaezaztocbcnsdn",
               "archive_bytes": target.stat().st_size, "archive_sha256": sha256(target), "files": []}
    with zipfile.ZipFile(target) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            filename = Path(member.filename).name
            print(f"YFCC archive member: {member.filename}, {member.file_size} bytes", flush=True)
            destination = directory / filename
            entry = previous_files.get(filename)
            if (entry and destination.exists() and destination.stat().st_size == member.file_size
                    and sha256(destination) == entry["sha256"] and entry["crc32"] == member.CRC):
                records["files"].append(entry)
                continue
            temporary = destination.with_suffix(destination.suffix + ".part")
            with archive.open(member) as source, temporary.open("wb") as output:
                while chunk := source.read(4 * 1024 * 1024):
                    output.write(chunk)
            # zipfile checks each decompressed member's CRC while reading it.
            temporary.replace(destination)
            records["files"].append({"member": member.filename, "path": destination.relative_to(ROOT).as_posix(),
                                      "bytes": member.file_size, "crc32": member.CRC,
                                      "sha256": sha256(destination)})
    (directory / "source_manifest.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print("YFCC full archive downloaded and extracted with member CRC verification", flush=True)


def fetch_classification():
    """Try original author's archives, then the fast-ai NLP public mirror."""
    fetch_ag()
    protocol = (AUDIT / "classification-results.sh").read_text(encoding="utf-8")
    ids = re.findall(r"(0B\w+)\s+#\s*(\w+)", protocol)
    for google_id, name in ids:
        if name == "ag_news":
            continue
        directory = RAW / name
        if (directory / "source_manifest.json").exists():
            continue
        archive = directory / f"{name}_csv.tar.gz"
        urls = [f"https://drive.usercontent.google.com/download?id={google_id}&export=download&confirm=t",
                f"https://s3.amazonaws.com/fast-ai-nlp/{name}_csv.tgz"]
        failures = []
        for source_index, url in enumerate(urls):
            try:
                # A partial stream must never be resumed from a different mirror.
                archive = directory / (f"{name}_csv.tar.gz" if source_index == 0 else f"{name}_csv.mirror.tar.gz")
                download(url, archive)
                with tarfile.open(archive) as packed:
                    chosen = [member for member in packed.getmembers()
                              if member.isfile() and Path(member.name).name in {"train.csv", "test.csv", "classes.txt"}]
                    if not {"train.csv", "test.csv"}.issubset({Path(m.name).name for m in chosen}):
                        raise ValueError("Archive lacks original CSV train/test members")
                    for member in chosen:
                        output = directory / Path(member.name).name
                        with packed.extractfile(member) as source, output.open("wb") as dest:
                            while chunk := source.read(4 * 1024 * 1024):
                                dest.write(chunk)
                record = {"url": url, "original_author_archive_id": google_id,
                          "archive_sha256": sha256(archive), "archive_bytes": archive.stat().st_size,
                          "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
                                    for p in directory.glob("*.csv")}, "failed_sources": failures}
                (directory / "source_manifest.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"{name}: complete CSV archive extracted", flush=True)
                break
            except (requests.RequestException, ValueError, tarfile.TarError) as error:
                failures.append({"url": url, "error": str(error)})
                print(f"{name}: source unavailable: {error}", flush=True)
        else:
            (AUDIT / f"{name}_download_failure.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
            raise RuntimeError(f"All public sources failed for {name}; see saved availability evidence")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["ag", "yfcc-probe", "all", "classification", "yfcc-full"], default="all")
    args = parser.parse_args()
    AUDIT.mkdir(parents=True, exist_ok=True)
    if args.stage == "ag":
        fetch_ag()
    if args.stage == "yfcc-probe":
        probe_zip()
    if args.stage in {"classification", "all"}:
        fetch_classification()
    if args.stage in {"yfcc-full", "all"}:
        fetch_yfcc()
