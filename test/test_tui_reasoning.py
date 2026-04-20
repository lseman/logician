from rich.console import Console
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel

console = Console()
response = "<think>\nThinking through the user's problem...\nThis looks like a simple addition.\nBut I'll just put the answer here without closing the tag: The answer to 2+2 is 4."

import re

think_pattern = re.compile(r"<think>(.*?)(?:</think>|$)", re.DOTALL | re.IGNORECASE)
match = think_pattern.search(response)

if match:
    think_text = match.group(1).strip()
    rest_text = response[:match.start()].strip() + "\n\n" + response[match.end():].strip()
    rest_text = rest_text.strip()

    p = Panel(
        RichMarkdown(think_text, code_theme="github-dark"),
        title="[dim italic]Reasoning Process[/dim italic]",
        border_style="dim white"
    )
    console.print(p)
    console.print()
    if rest_text:
        console.print(RichMarkdown(rest_text, code_theme="github-dark"))
else:
    console.print(RichMarkdown(response, code_theme="github-dark"))
