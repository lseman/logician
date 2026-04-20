import importlib
import unittest


class SkillImportsSmokeTests(unittest.TestCase):
    def test_import_core_skill_modules(self) -> None:
        edit_block_mod = importlib.import_module(
            "skills.coding.edit_block.scripts.edit_block"
        )
        think_mod = importlib.import_module(
            "skills.global.think.scripts.think"
        )

        self.assertTrue(hasattr(edit_block_mod, "apply_edit_block"))
        self.assertTrue(hasattr(think_mod, "think"))


if __name__ == "__main__":
    unittest.main()
