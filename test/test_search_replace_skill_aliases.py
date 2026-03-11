import importlib
import tempfile
import unittest
from pathlib import Path


class SearchReplaceSkillAliasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = importlib.import_module(
            "skills.coding.search_replace.search_replace"
        )

    def test_find_in_file_literal_reports_line_and_match(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "demo.txt"
            target.write_text(
                "alpha\nbeta token here\ngamma token here too\n",
                encoding="utf-8",
            )

            raw = self.mod.find_in_file(str(target), "token", max_results=10)

            self.assertIn('"status":"ok"', raw.replace(" ", ""))
            self.assertIn('"count":2', raw.replace(" ", ""))
            self.assertIn('"line":2', raw.replace(" ", ""))
            self.assertIn('"match":"token"', raw.replace(" ", ""))

    def test_sed_read_supports_line_spec(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "demo.py"
            target.write_text(
                "one\n"
                "two\n"
                "three\n"
                "four\n",
                encoding="utf-8",
            )

            raw = self.mod.sed_read(str(target), line_spec="2,3p")

            compact = raw.replace(" ", "")
            self.assertIn('"status":"ok"', compact)
            self.assertIn('"selected_lines":"2-3"', compact)
            self.assertIn("two", raw)
            self.assertIn("three", raw)
            self.assertNotIn('"selected_lines":"1-4"', compact)

    def test_sed_replace_previews_and_applies(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / "demo.txt"
            target.write_text("foo 1\nfoo 2\n", encoding="utf-8")

            preview = self.mod.sed_replace(str(target), "s/foo/bar/g")
            self.assertIn('"written":false', preview.replace(" ", "").lower())
            self.assertEqual(target.read_text(encoding="utf-8"), "foo 1\nfoo 2\n")

            applied = self.mod.sed_replace(
                str(target),
                "s/foo/bar/g",
                preview_only=False,
            )
            self.assertIn('"written":true', applied.replace(" ", "").lower())
            self.assertEqual(target.read_text(encoding="utf-8"), "bar 1\nbar 2\n")

    def test_find_path_alias_delegates_to_fd_find(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "src").mkdir()
            (root / "src" / "config.toml").write_text("x=1\n", encoding="utf-8")

            raw = self.mod.find_path(
                pattern="config",
                directory=str(root),
                extension="toml",
                max_results=10,
            )

            compact = raw.replace(" ", "")
            self.assertIn('"status":"ok"', compact)
            self.assertIn("config.toml", raw)


if __name__ == "__main__":
    unittest.main()
