"""Check public original-data links; archive small upstream protocol files only."""
from pathlib import Path
import hashlib
import json
import re
import sys

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/paper_reproduction"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8")

sources = {
    "fasttext_readme": "https://raw.githubusercontent.com/facebookresearch/fastText/v0.9.2/README.md",
    "classification_protocol": "https://raw.githubusercontent.com/facebookresearch/fastText/v0.9.2/classification-results.sh",
    "ag_news_train": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv",
    "ag_news_test": "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv",
    "yfcc_original": "https://fb-public.box.com/s/htfdbrvycvroebv9ecaezaztocbcnsdn",
}
records = {}
for name, url in sources.items():
    try:
        is_ag = name.startswith("ag_news")
        response = requests.request("HEAD" if is_ag else "GET", url, timeout=(15, 30),
                                    allow_redirects=True)
        record = {"url": url, "final_url": response.url, "status": response.status_code,
                  "content_type": response.headers.get("Content-Type"),
                  "content_length": response.headers.get("Content-Length")}
        if not is_ag:
            content = response.content
            filename = {"fasttext_readme": "upstream_README.md",
                        "classification_protocol": "classification-results.sh",
                        "yfcc_original": "yfcc_public_page.html"}[name]
            (OUT / filename).write_bytes(content)
            record.update(bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                          archive=str((OUT / filename).relative_to(ROOT)))
            if name == "yfcc_original":
                record["page_title"] = re.findall(r"<title>(.*?)</title>", response.text, re.S)[:1]
                record["file_ids"] = sorted(set(re.findall(r'"(?:fileID|fileId|file_id)"\s*:\s*"?(\d+)', response.text)))
        records[name] = record
    except requests.RequestException as error:
        records[name] = {"url": url, "error": str(error)}
    print(json.dumps({name: records[name]}, ensure_ascii=False), flush=True)
(OUT / "data_availability.json").write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
