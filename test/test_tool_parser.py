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


if __name__ == "__main__":
    unittest.main()
