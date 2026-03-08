import importlib
import unittest


class SkillImportsSmokeTests(unittest.TestCase):
    def test_import_core_skill_modules(self) -> None:
        edit_block_mod = importlib.import_module("skills.01_coding.65_edit_block")
        advanced_mining_mod = importlib.import_module(
            "skills.02_timeseries.80_advanced_mining"
        )

        self.assertTrue(hasattr(edit_block_mod, "apply_edit_block"))
        self.assertTrue(hasattr(advanced_mining_mod, "discover_motifs"))


if __name__ == "__main__":
    unittest.main()
