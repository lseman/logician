import importlib
import builtins
import subprocess

import json
builtins._coding_config = {"default_cwd": "."}
builtins._safe_json = lambda x: json.dumps(x)
def mock_run_cmd(cmd, cwd=None, timeout=None, venv_path=None):
    proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return {"exit_code": proc.returncode, "stdout": proc.stdout + proc.stderr}
builtins._run_cmd = mock_run_cmd

auto_format = getattr(importlib.import_module("skills.01_coding.50_quality"), "auto_format")

from pathlib import Path
dummy = Path("dummy.py")
dummy.write_text("def   hello (   ):    \n    print(   'hello world'   )\n")

print("Before:\n=====\n" + dummy.read_text() + "=====\n")
res = auto_format("dummy.py")
print("Response:", res)
print("After:\n=====\n" + dummy.read_text() + "=====\n")
dummy.unlink()
