import importlib.util
import sys
import os

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

edit_block = load_module("edit_block", "skills/02_coding/65_edit_block.py")
advanced_mining = load_module("advanced_mining", "skills/01_timeseries/80_advanced_mining.py")

import json

print("edit_block loaded!", "apply_edit_block" in dir(edit_block))
print("advanced_mining loaded!", "discover_motifs" in dir(advanced_mining))

# Test edit block logic
test_file = "dummy_file.py"
with open(test_file, "w") as f:
    f.write("def foo():\n    print('bar')\n")

blocks = """<<<< SEARCH
def foo():
    print('bar')
====
def foo():
    print('baz')
>>>> REPLACE"""

res = edit_block.apply_edit_block(test_file, blocks)
print("edit block result:", res)

with open(test_file, "r") as f:
    print("file content:", f.read())

os.remove(test_file)
