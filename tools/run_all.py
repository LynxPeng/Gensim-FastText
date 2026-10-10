"""Resume the complete approved experiment order using fingerprinted caches."""
import os
import subprocess
import sys
from gensim_lab.common import ARTIFACTS, ROOT, stamp, write_json

steps = [
    ["prepare"],
    ["run", "--profile", "smoke"],
    ["run", "--profile", "full", "--seeds", "42"],
    ["run", "--profile", "ablations"],
    ["run", "--profile", "full", "--seeds", "42,2026,7"],
    ["run", "--profile", "temporal", "--seeds", "42,2026,7", "--finalize"],
    ["run", "--profile", "supplemental"],
]
done = []
for step in steps:
    write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": step, "status": "running"})
    print("STAGE", " ".join(step), flush=True)
    result = subprocess.run([sys.executable, "-m", "gensim_lab", *step], cwd=ROOT)
    if result.returncode:
        write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": step, "status": "failed", "exit_code": result.returncode})
        raise SystemExit(result.returncode)
    done.append(step)
write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": ["finalize_review"], "status": "running"})
print("STAGE word_evaluation", flush=True)
result = subprocess.run([sys.executable, "-m", "gensim_lab.word_evaluation", "--stage", "all"], cwd=ROOT)
if result.returncode:
    write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": ["word_evaluation"], "status": "failed", "exit_code": result.returncode})
    raise SystemExit(result.returncode)
done += [["word_evaluation"]]
for args in [["-m", "gensim_lab.word_fasttext_score"], ["-m", "gensim_lab.word_evaluation", "--stage", "report", "--export"]]:
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode:
        write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": args, "status": "failed", "exit_code": result.returncode})
        raise SystemExit(result.returncode)
    done += [args]
result = subprocess.run([sys.executable, "tools/finalize_review.py"], cwd=ROOT)
if result.returncode:
    write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "current": ["finalize_review"], "status": "failed", "exit_code": result.returncode})
    raise SystemExit(result.returncode)
done += [["finalize_review"]]
write_json(ARTIFACTS / "pipeline.json", {"updated": stamp(), "completed_steps": done, "status": "complete"})
