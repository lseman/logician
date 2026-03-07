from pathlib import Path
import importlib
apply_edit_block = getattr(importlib.import_module("skills.01_coding.65_edit_block"), "apply_edit_block")

def main():
    dummy = Path("dummy.py")
    dummy.write_text("def hello():\n    print('hello world')\n\n    # Some comment\n    return True\n")
    
    # Try exact match
    b1 = '''<<<< SEARCH
    print('hello world')

    # Some comment
====
    print('hello fuzzy')

    # Replaced comment
>>>> REPLACE'''
    
    r1 = apply_edit_block("dummy.py", b1)
    print("Exact match result:", r1)
    print("Content after r1:\n", dummy.read_text())

    # Try fuzzy match (bad indentation)
    b2 = '''<<<< SEARCH
def hello():
  print('hello fuzzy')
  
  # Replaced comment
====
def hello():
    print('hello fuzzy 2')
>>>> REPLACE'''
    
    r2 = apply_edit_block("dummy.py", b2)
    print("Fuzzy match result:", r2)
    print("Content after r2:\n", dummy.read_text())
    
    dummy.unlink()

if __name__ == "__main__":
    main()
