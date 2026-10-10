from pathlib import Path
import json
import sys
from gensim_lab.common import write_json

sys.stdout.reconfigure(encoding="utf-8")
root = Path(__file__).resolve().parents[1]
results = {}
for name in ["valid", "test", "train"]:
    path = root / "data/paper_reproduction/raw/yfcc100m" / name
    lines = 0
    with path.open("rb") as stream:
        first = stream.readline()
        lines += 1
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            lines += chunk.count(b"\n")
    results[name] = {"lines": lines, "bytes": path.stat().st_size, "first_line_bytes": len(first)}
    print(json.dumps({name: results[name]}), flush=True)
write_json(root / "artifacts/paper_reproduction/yfcc_raw_counts.json", results)
