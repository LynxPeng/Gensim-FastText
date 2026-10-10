import subprocess
import sys
from importlib.metadata import version
from gensim_lab.common import ROOT, ARTIFACTS, environment, stamp, write_json

out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, encoding="utf-8")
lines = [line for line in out.splitlines() if not line.startswith(("-e ", "#")) and not line.lower().startswith("gensim-advanced-lab")]
lines = [f"{line.split(' @ ', 1)[0]}=={version(line.split(' @ ', 1)[0])}" if " @ " in line else line for line in lines]
(ROOT / "requirements.lock.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
result = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, encoding="utf-8")
write_json(ARTIFACTS / "environment.json", {"created": stamp(), **environment(), "pip_check": {"exit_code": result.returncode, "output": result.stdout + result.stderr}, "frozen_packages": lines})
print(result.stdout)
raise SystemExit(result.returncode)
