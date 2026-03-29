"""Tests for libcst-based file editing functions."""
import tempfile
import unittest
from pathlib import Path

from src.tools.core.files_libcst import (
    edit_file_libcst,
    replace_docstring,
    replace_decorators,
    find_function_by_name,
    find_class_by_name,
)


class TestEditFileLibcst(unittest.TestCase):
    """Tests for edit_file_libcst function."""

    def test_edit_simple_expression(self):
        """Test editing a simple expression."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    return x + y\n")
            f.flush()
            result = edit_file_libcst(f.name, "x + y", "x * 2")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["edits_applied"], 1)
        with open(f.name) as f:
            content = f.read()
        self.assertIn("x * 2", content)
        Path(f.name).unlink()

    def test_edit_preserves_whitespace(self):
        """Test that whitespace is preserved during edit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            # Create file with specific indentation
            f.write("def foo():\n    # This is a comment\n    return x\n")
            f.flush()
            result = edit_file_libcst(f.name, "return x", "return x + 1")

        self.assertEqual(result["status"], "ok")
        with open(f.name) as f:
            content = f.read()
        # Verify comment is still there
        self.assertIn("This is a comment", content)
        Path(f.name).unlink()

    def test_edit_not_found(self):
        """Test when pattern is not found."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    return x\n")
            f.flush()
            result = edit_file_libcst(f.name, "return y", "return z")

        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result.get("error", ""))
        Path(f.name).unlink()

    def test_edit_function_definition(self):
        """Test editing an entire function definition."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def old_func():\n    pass\n")
            f.flush()
            result = edit_file_libcst(
                f.name,
                "def old_func():",
                "def new_func():",
            )

        self.assertEqual(result["status"], "ok")
        with open(f.name) as f:
            content = f.read()
        self.assertIn("def new_func():", content)
        Path(f.name).unlink()


class TestReplaceDocstring(unittest.TestCase):
    """Tests for replace_docstring function."""

    def test_replace_function_docstring(self):
        """Test replacing a function's docstring."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''def foo():
    """Old docstring."""
    pass
'''
            )
            f.flush()
            result = replace_docstring(
                f.name, "foo", """
        New docstring.
        Args:
            x: The input value
    """
            )

        self.assertEqual(result["status"], "ok")
        with open(f.name) as f:
            content = f.read()
        self.assertIn('"""New docstring."""', content)
        Path(f.name).unlink()

    def test_replace_class_docstring(self):
        """Test replacing a class's docstring."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''class MyClass:
    """Old class docstring."""
    pass
'''
            )
            f.flush()
            result = replace_docstring(
                f.name, "MyClass", """
        New class docstring.
    """
            )

        self.assertEqual(result["status"], "ok")
        with open(f.name) as f:
            content = f.read()
        self.assertIn('"""New class docstring."""', content)
        Path(f.name).unlink()


class TestReplaceDecorators(unittest.TestCase):
    """Tests for replace_decorators function."""

    def test_replace_decorators(self):
        """Test replacing function decorators."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''@old_decorator
def foo():
    pass
'''
            )
            f.flush()
            result = replace_decorators(
                f.name, "foo", ["@property", "@staticmethod"]
            )

        self.assertEqual(result["status"], "ok")
        with open(f.name) as f:
            content = f.read()
        self.assertIn("@property", content)
        self.assertIn("@staticmethod", content)
        Path(f.name).unlink()


class TestFindFunctionByName(unittest.TestCase):
    """Tests for find_function_by_name function."""

    def test_find_function(self):
        """Test finding a function by name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''def foo(x, y):
    """A function."""
    return x + y
'''
            )
            f.flush()
            result = find_function_by_name(f.name, "foo")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["function_name"], "foo")
        self.assertIn("line", result)
        Path(f.name).unlink()

    def test_find_function_not_found(self):
        """Test finding a function that doesn't exist."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write("def foo():\n    pass\n")
            f.flush()
            result = find_function_by_name(f.name, "nonexistent")

        self.assertEqual(result["status"], "error")
        self.assertIn("not found", result.get("error", ""))
        Path(f.name).unlink()


class TestFindClassByName(unittest.TestCase):
    """Tests for find_class_by_name function."""

    def test_find_class(self):
        """Test finding a class by name."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(
                '''class MyClass:
    """A class."""
    pass
'''
            )
            f.flush()
            result = find_class_by_name(f.name, "MyClass")

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["class_name"], "MyClass")
        self.assertIn("line", result)
        Path(f.name).unlink()


if __name__ == "__main__":
    unittest.main()
