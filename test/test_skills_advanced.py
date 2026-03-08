import importlib
import tempfile
import unittest
from pathlib import Path


class SkillAdvancedBehaviorTests(unittest.TestCase):
    def test_apply_edit_block_replaces_exact_block(self) -> None:
        edit_block = importlib.import_module("skills.01_coding.65_edit_block")
        apply_edit_block = getattr(edit_block, "apply_edit_block")

        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "dummy.py"
            target.write_text(
                "def foo():\n"
                "    print('bar')\n",
                encoding="utf-8",
            )

            block = (
                "<<<< SEARCH\n"
                "def foo():\n"
                "    print('bar')\n"
                "====\n"
                "def foo():\n"
                "    print('baz')\n"
                ">>>> REPLACE"
            )
            raw = apply_edit_block(str(target), block)
            self.assertIn('"status":"ok"', raw.replace(" ", ""))
            updated = target.read_text(encoding="utf-8")
            self.assertIn("print('baz')", updated)
            self.assertNotIn("print('bar')", updated)


if __name__ == "__main__":
    unittest.main()
