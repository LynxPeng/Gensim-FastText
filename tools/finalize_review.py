"""Run the strengthened verification gates and publish the review artifacts."""
import re
import subprocess
import sys

from gensim_lab.common import ARTIFACTS, ROOT, code_hash, file_hash, stamp, write_json
from gensim_lab.runner import worker_environment

steps = [
    ["-m", "gensim_lab.supplemental", "--stage", "report"],
    ["-m", "gensim_lab", "evidence"],
    ["tools/export_native.py"],
    ["tools/check_saved_predictions.py"],
    ["tools/deep_audit.py"],
    ["-m", "gensim_lab.word_checks"],
    ["-m", "gensim_lab", "report"],
    ["tools/check_markdown.py"],
    ["-m", "pytest", "--basetemp", ".artifact-work/pytest-audit-temp", "-q"],
]
result = {"started": stamp(), "status": "running", "completed_steps": [], "core_training_code": code_hash(),
          "review_code_sha256": {str(path.relative_to(ROOT)): file_hash(path) for path in
                                   [ROOT / "gensim_lab" / name for name in
                                    ["runner.py", "cli.py", "integrity.py", "content_seed.py", "supplemental_worker.py", "supplemental.py", "supplemental_report.py", "reporting.py", "evidence.py", "array_checks.py"]]
                                   + [ROOT / "tools" / name for name in
                                      ["export_native.py", "check_saved_predictions.py", "check_doc2vec_identity.py", "check_cli_predictions.py", "deep_audit.py", "check_markdown.py", "finalize_review.py", "run_all.py"]]}}
for n, args in enumerate(steps):
    result["current"] = args
    write_json(ARTIFACTS / "review_pipeline.json", result)
    print("REVIEW STAGE", n + 1, " ".join(args), flush=True)
    run = subprocess.run([sys.executable, *args], cwd=ROOT, env=worker_environment(), capture_output=True, text=True, encoding="utf-8")
    (ARTIFACTS / f"review_stage_{n+1}.log").write_text(run.stdout + run.stderr, encoding="utf-8")
    print(run.stdout[-2200:].strip(), flush=True)
    if run.returncode:
        result.update(status="failed", exit_code=run.returncode)
        write_json(ARTIFACTS / "review_pipeline.json", result)
        raise RuntimeError(run.stderr or run.stdout)
    result["completed_steps"].append(args)
    if "pytest" in args:
        match = re.search(r"(\d+) passed", run.stdout)
        result["tests_passed"] = int(match.group(1)) if match else None
result.update(status="complete", completed=stamp())
result.pop("current", None)
write_json(ARTIFACTS / "review_pipeline.json", result)
print("Review verification and Markdown publication completed", flush=True)
