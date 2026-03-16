import sys
import unittest
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parents[1]
if str(AGENT_ROOT) not in sys.path:
    sys.path.insert(0, str(AGENT_ROOT))

from src.tools.parser import parse_tool_calls


class ToolParserTests(unittest.TestCase):
    def test_parses_standard_tool_call_json(self) -> None:
        text = (
            '{"tool_call":{"name":"rg_search","arguments":{"pattern":"TODO","max_results":5}}}'
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "rg_search")
        self.assertEqual(calls[0].arguments["pattern"], "TODO")

    def test_parses_anthropic_tool_use_block(self) -> None:
        text = (
            '{"type":"tool_use","name":"rg_search","input":{"pattern":"TODO","directory":"."}}'
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "rg_search")
        self.assertEqual(calls[0].arguments["directory"], ".")

    def test_parses_anthropic_content_wrapper(self) -> None:
        text = (
            '{"id":"msg_1","content":[{"type":"text","text":"ok"},'
            '{"type":"tool_use","name":"fd_find","input":{"pattern":"config"}}]}'
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fd_find")
        self.assertEqual(calls[0].arguments["pattern"], "config")

    def test_parses_tool_call_tag_wrapper(self) -> None:
        text = """
<assistant_response>
<tool_call>
{"name":"rg_search","arguments":{"pattern":"TODO","directory":"."}}
</tool_call>
</assistant_response>
""".strip()
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "rg_search")
        self.assertEqual(calls[0].arguments["directory"], ".")

    def test_parses_jinja_function_style_tool_call(self) -> None:
        text = "{{ tool_call(name='fd_find', arguments={'pattern':'config', 'directory':'.'}) }}"
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fd_find")
        self.assertEqual(calls[0].arguments["pattern"], "config")

    def test_parses_plain_tool_key_json_block(self) -> None:
        text = """```json
{
  "tool": "fd_find",
  "arguments": {
    "pattern": "rust-cli/src",
    "file_type": "d"
  }
}
```"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fd_find")
        self.assertEqual(
            calls[0].arguments,
            {"pattern": "rust-cli/src", "file_type": "d"},
        )

    def test_parses_bash_fence_as_tool_call_when_execution_intent_is_explicit(self) -> None:
        text = """I will now execute this plan.

```bash
ls -R rust-cli/src
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(calls[0].arguments, {"command": "ls -R rust-cli/src"})

    def test_parses_bash_fence_with_execution_heading_and_curly_apostrophe(self) -> None:
        text = """Execution

I’ll read the source files in parallel to understand the current implementation:

```bash
cat rust-cli/src/main.rs rust-cli/src/app.rs
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "bash")
        self.assertEqual(
            calls[0].arguments,
            {"command": "cat rust-cli/src/main.rs rust-cli/src/app.rs"},
        )

    def test_parses_bash_fence_with_direct_tool_invocation_syntax(self) -> None:
        text = """I’ll analyze the project structure now.

```bash
fd_find "src", extension="rs", directory="rust-cli"
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fd_find")
        self.assertEqual(
            calls[0].arguments,
            {"pattern": "src", "extension": "rs", "directory": "rust-cli"},
        )

    def test_parses_bash_fence_with_parenthesized_direct_tool_invocation(self) -> None:
        text = """Let me run the tool directly.

```bash
fd_find("src", extension="rs", directory="rust-cli")
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].name, "fd_find")
        self.assertEqual(
            calls[0].arguments,
            {"pattern": "src", "extension": "rs", "directory": "rust-cli"},
        )

    def test_does_not_parse_bash_fence_without_execution_intent(self) -> None:
        text = """Here is an example command:

```bash
ls -R rust-cli/src
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(calls, [])

    def test_does_not_parse_bash_fence_for_example_even_with_first_person_language(self) -> None:
        text = """I’ll show an example command:

```bash
cat rust-cli/src/main.rs
```
"""
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(calls, [])


    def test_parses_batched_inline_toon_multiple_reads(self) -> None:
        # Model emits multiple name:/arguments: entries under one tool_call: header
        text = (
            'tool_call: name: read_file arguments: path: "./rust-cli/src/main.rs"'
            ' name: read_file arguments: path: "./rust-cli/src/app.rs"'
            ' name: read_file arguments: path: "./rust-cli/src/ui.rs"'
            ' name: read_file arguments: path: "./rust-cli/src/bridge.rs"'
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 4)
        paths = [c.arguments.get("path") for c in calls]
        self.assertIn("./rust-cli/src/main.rs", paths)
        self.assertIn("./rust-cli/src/app.rs", paths)
        self.assertIn("./rust-cli/src/ui.rs", paths)
        self.assertIn("./rust-cli/src/bridge.rs", paths)

    def test_parses_batched_inline_toon_last_path_on_next_line(self) -> None:
        # Last path wraps to next line (common model output artifact)
        text = (
            'tool_call: name: read_file arguments: path: "./rust-cli/src/main.rs"'
            ' name: read_file arguments: path:\n"./rust-cli/src/bridge.rs"'
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 2)
        paths = [c.arguments.get("path") for c in calls]
        self.assertIn("./rust-cli/src/main.rs", paths)
        self.assertIn("./rust-cli/src/bridge.rs", paths)

    def test_parses_separate_inline_toon_blocks(self) -> None:
        # Each call has its own tool_call: prefix — existing behaviour preserved
        text = (
            "tool_call: name: read_file arguments: path: a.py\n"
            "tool_call: name: read_file arguments: path: b.py"
        )
        calls = parse_tool_calls(text, use_toon=False)
        self.assertEqual(len(calls), 2)
        paths = [c.arguments.get("path") for c in calls]
        self.assertIn("a.py", paths)
        self.assertIn("b.py", paths)


if __name__ == "__main__":
    unittest.main()
