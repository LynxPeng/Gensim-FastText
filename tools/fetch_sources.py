"""Archive exact upstream sources and wheel provenance for source reading."""
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import distribution
from pathlib import Path
import hashlib
import shutil
import requests
import gensim

from gensim_lab.common import ROOT, file_hash, stamp, write_json

base = ROOT / "references" / "sources"
groot = Path(gensim.__file__).parent
records = []
gfiles = ["models/fasttext.py", "models/fasttext_inner.pyx", "models/word2vec.py", "models/word2vec_inner.pyx", "models/doc2vec.py", "models/keyedvectors.py", "models/tfidfmodel.py", "corpora/dictionary.py", "utils.py"]
for name in gfiles:
    target = base / "gensim-4.4.0" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(groot / name, target)
    records.append({"path": str(target.relative_to(ROOT)).replace("\\", "/"), "version": "4.4.0", "origin": "installed PyPI wheel", "url": f"https://github.com/piskvorky/gensim/blob/4.4.0/gensim/{name}", "sha256": file_hash(target)})
license_file = next((p for p in distribution("gensim").files if p.name == "COPYING"), None)
if license_file:
    shutil.copyfile(distribution("gensim").locate_file(license_file), base / "gensim-4.4.0" / "LICENSE")

def verify_gensim(record):
    url = record["url"].replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    record["upstream_url"] = url
    record["upstream_sha256"] = hashlib.sha256(response.content).hexdigest()
    local = (ROOT / record["path"]).read_bytes()
    record["exact_bytes_match"] = record["sha256"] == record["upstream_sha256"]
    record["local_crlf_lines"] = local.count(b"\r\n")
    record["matches_upstream_tag"] = local.replace(b"\r\n", b"\n") == response.content.replace(b"\r\n", b"\n")
    record["comparison"] = "Exact text after normalizing CRLF/LF line endings"
    if not record["matches_upstream_tag"]:
        raise AssertionError(f"Installed source differs from upstream tag: {record['path']}")
    return record
with ThreadPoolExecutor(max_workers=4) as executor:
    records = list(executor.map(verify_gensim, records))

native_files = ["src/dictionary.cc", "src/dictionary.h", "src/fasttext.cc", "src/model.cc", "src/loss.cc", "src/args.cc", "src/densematrix.cc", "src/vector.cc", "python/fasttext_module/fasttext/FastText.py", "python/fasttext_module/fasttext/pybind/fasttext_pybind.cc", "LICENSE"]
def fetch(name):
    url = f"https://raw.githubusercontent.com/facebookresearch/fastText/v0.9.2/{name}"
    target = base / "fastText-0.9.2" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        target.write_bytes(response.content)
    return {"path": str(target.relative_to(ROOT)).replace("\\", "/"), "version": "0.9.2", "origin": "upstream tag v0.9.2", "url": url, "sha256": file_hash(target)}
with ThreadPoolExecutor(max_workers=4) as executor:
    records.extend(executor.map(fetch, native_files))

wheel = requests.get("https://pypi.org/pypi/fasttext-wheel/0.9.2/json", timeout=60)
wheel.raise_for_status()
windows = [x for x in wheel.json()["urls"] if "cp312-cp312-win_amd64" in x["filename"]]
write_json(ROOT / "references" / "source_manifest.json", {"created": stamp(), "sources": records, "windows_distribution": {"project": "fasttext-wheel", "version": "0.9.2", "pypi": "https://pypi.org/project/fasttext-wheel/0.9.2/", "wheel": windows, "note": "Third-party Windows binary distribution. Source reading uses upstream v0.9.2; this does not prove the binary is byte-identical to an upstream build."}})
print(f"Archived {len(records)} source files; Windows wheel provenance saved.")
