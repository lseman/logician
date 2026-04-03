from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.tools.core import run_python
from src.tools.core.FileEditTool import edit_file, write_file
from src.tools.core.FileReadTool import read_file
from src.tools.text_normalization import normalize_text_payload


class TextNormalizationTests(unittest.TestCase):
    def test_normalize_python_payload_decodes_outer_newlines_only(self) -> None:
        raw = 'def demo():\\n    print("line1\\\\nline2")\\n'

        normalized, meta = normalize_text_payload(raw, language_hint="python")

        self.assertEqual(
            normalized,
            'def demo():\n    print("line1\\\\nline2")\n',
        )
        self.assertIn("linebreaks_decoded", ",".join(meta.get("transformations", [])))

    def test_normalize_python_payload_unwraps_outer_string_literal(self) -> None:
        raw = '"def demo():\\n    return \\"ok\\"\\n"'

        normalized, _ = normalize_text_payload(raw, language_hint="python")

        self.assertEqual(normalized, 'def demo():\n    return "ok"\n')

    def test_write_file_normalizes_escaped_multiline_python(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "example.py"

            result = write_file(
                str(path),
                'def demo():\\n    print("line1\\\\nline2")\\n',
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(
                path.read_text(encoding="utf-8"),
                'def demo():\n    print("line1\\\\nline2")\n',
            )

    def test_write_file_append_mode_appends_instead_of_replacing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "append.txt"
            path.write_text("alpha\n", encoding="utf-8")
            read_result = read_file(str(path))
            self.assertEqual(read_result["status"], "ok")

            result = write_file(str(path), "beta\n", mode="a")

            self.assertEqual(result["status"], "ok")
            self.assertEqual(path.read_text(encoding="utf-8"), "alpha\nbeta\n")

    def test_edit_file_preserves_crlf_style(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "windows.py"
            path.write_bytes(b"def demo():\r\n    return 1\r\n")
            read_result = read_file(str(path))
            self.assertEqual(read_result["status"], "ok")

            result = edit_file(
                str(path),
                "def demo():\\n    return 1\\n",
                "def demo():\\n    return 2\\n",
            )

            self.assertEqual(result["status"], "ok")
            self.assertEqual(path.read_bytes(), b"def demo():\r\n    return 2\r\n")

    def test_run_python_normalizes_outer_newlines_without_breaking_inner_strings(self) -> None:
        payload = 'print("line1\\\\nline2")\\nprint("ok")\\n'

        class _Runtime:
            def venv_path(self) -> str | None:
                return None

            def resolve_cwd(self, cwd: str | None) -> str | None:
                return cwd

        with patch("builtins.coding_runtime", _Runtime(), create=True):
            result = run_python(payload)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["stdout"], "line1\\nline2\nok\n")


if __name__ == "__main__":
    unittest.main()
