"""Record hardware and the installed experiment APIs without changing backends."""
from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import inspect
import io
import json
import platform
import subprocess
from datetime import datetime

from gensim.models import Doc2Vec, FastText, LsiModel, Word2Vec
from sklearn.svm import LinearSVC
from sklearn.utils import get_tags

from gensim_lab.common import ARTIFACTS, write_json


def main():
    command = [
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total,compute_cap",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(command, capture_output=True, text=True, timeout=20)
        hardware = {
            "command": command, "returncode": process.returncode,
            "stdout": process.stdout.strip(), "stderr": process.stderr.strip(),
            "gpus": [dict(zip(
                ["name", "driver_version", "memory_total_mib", "compute_capability"],
                [value.strip() for value in row],
            )) for row in csv.reader(io.StringIO(process.stdout))] if process.returncode == 0 else [],
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        hardware = {"command": command, "error": str(error), "gpus": []}

    classes = [Word2Vec, FastText, Doc2Vec, LsiModel, LinearSVC]
    result = {
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in
                     ["gensim", "fasttext-wheel", "numpy", "scipy", "scikit-learn"]},
        "hardware": hardware,
        "optional_module_installed": {name: importlib.util.find_spec(name) is not None
                                      for name in ["torch", "cupy", "cuml"]},
        "constructor_signatures": {cls.__name__: str(inspect.signature(cls)) for cls in classes},
        "linear_svc_array_api_support": get_tags(LinearSVC()).array_api_support,
        "decision": "Keep the fixed Gensim/fastText/LinearSVC experiment backends on CPU.",
        "scope": "Hardware/API audit only; no CUDA kernel benchmark or CUDA Toolkit verification.",
    }
    write_json(ARTIFACTS / "cuda_audit.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
