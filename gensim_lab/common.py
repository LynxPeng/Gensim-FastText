from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import psutil
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"


def fingerprint(value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config():
    return yaml.safe_load((ROOT / "configs" / "experiments.yaml").read_text(encoding="utf-8"))


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def environment():
    packages = ["gensim", "fasttext-wheel", "numpy", "scipy", "scikit-learn", "jieba", "matplotlib", "pandas", "psutil"]
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "packages": {p: version(p) for p in packages},
        "logical_cpus": os.cpu_count(), "ram_gib": psutil.virtual_memory().total / 2**30,
        "hash_seed": os.environ.get("PYTHONHASHSEED"),
    }


def code_hash():
    names = ["data.py", "models.py", "worker.py", "common.py", "diagnostics.py"]
    return fingerprint({n: file_hash(ROOT / "gensim_lab" / n) for n in names})


class PeakMemory:
    def __enter__(self):
        self.peak = psutil.Process().memory_info().rss
        self.stop = threading.Event()
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        process = psutil.Process()
        while not self.stop.wait(0.05):
            self.peak = max(self.peak, process.memory_info().rss)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()
        self.elapsed = time.perf_counter() - self.started
        self.peak = max(self.peak, psutil.Process().memory_info().rss)


def markdown_table(headers, rows):
    def cell(x):
        return str(x).replace("|", "\\|").replace("\n", " ")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])
