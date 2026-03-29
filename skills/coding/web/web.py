from __future__ import annotations

import json
import urllib.request
from skills.coding.bootstrap.runtime_access import tool
import urllib.error

__skill__ = {
    "name": "Web",
    "description": "Use for fetching technical documentation, URLs, and web context for coding tasks.",
    "aliases": ["docs lookup", "documentation", "api docs", "external reference"],
    "triggers": [
        "look up the docs",
        "fetch this url",
        "read the package documentation",
        "search the web for this error",
    ],
    "preferred_tools": ["fetch_url", "web_search"],
    "example_queries": [
        "fetch the Python docs page for pathlib",
        "search for this stack trace online",
        "read the API reference for this package",
    ],
    "when_not_to_use": [
        "the answer is already available in the repo or only local files need to be inspected"
    ],
    "next_skills": ["file_ops", "shell", "explore"],
    "preferred_sequence": ["web_search", "fetch_url", "file_ops"],
    "entry_criteria": [
        "The task depends on external documentation, an exact URL, or current package reference material.",
        "Local repo inspection is insufficient to answer the implementation question.",
    ],
    "decision_rules": [
        "Prefer exact URLs when the source is already known.",
        "Search first when the problem is broad and source selection matters.",
        "Bring back only the technical facts needed for the next local edit or command.",
    ],
    "workflow": [
        "Prefer exact URLs when the source is known.",
        "Use search when the problem is broad or source selection matters.",
        "Bring back only the relevant technical facts, then continue locally.",
    ],
    "failure_recovery": [
        "If a page is noisy or badly rendered, switch to a better source or a richer web skill instead of over-trusting the extracted text.",
        "If external docs conflict with the repo, verify the local code path before editing.",
    ],
    "exit_criteria": [
        "The needed external fact has been reduced to a short actionable takeaway.",
        "The task is ready to continue locally in file_ops, shell, or explore.",
    ],
    "anti_patterns": [
        "Copying large chunks of documentation into context.",
        "Using external docs when the repo already contains the answer.",
    ],
}

# Fallback in case _safe_json is not injected by the agent environment
if "_safe_json" not in globals():
    def _safe_json(obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})

@tool
def fetch_url(url: str, timeout: int = 15) -> str:
    """Use when: Read documentation, articles, or text from a web page.

    Triggers: fetch url, read web page, download docs, scraping.
    Avoid when: The site requires complex JavaScript rendering or login.
    Inputs:
      url (str, required): The URL to retrieve (e.g. "https://docs.python.org/3/").
      timeout (int, optional): Timeout in seconds (default 15).
    Returns: JSON with webpage content (converted to basic text/markdown) or error.
    Side effects: Makes a network request.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLMAgent/1.0)"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            html = response.read().decode("utf-8", errors="replace")
            
        import re
        
        # Remove head, style, script
        body = re.sub(r"<(head|style|script)[^>]*>.*?</\1>", "", html, flags=re.IGNORECASE|re.DOTALL)
        # Extract body 
        body_match = re.search(r"<body[^>]*>(.*?)</body>", body, flags=re.IGNORECASE|re.DOTALL)
        if body_match:
            body = body_match.group(1)
            
        # Convert links to markdown
        body = re.sub(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', r"[\2](\1)", body, flags=re.IGNORECASE|re.DOTALL)
        
        # Replace block tags with newlines
        body = re.sub(r"<(p|br|div|h1|h2|h3|h4|h5|h6|li|tr|/table)[^>]*>", "\n", body, flags=re.IGNORECASE)
        
        # Remove remaining tags
        text = re.sub(r"<[^>]+>", " ", body)
        
        # Unescape HTML entities
        import html as html_lib
        text = html_lib.unescape(text)
        
        # Collapse whitespace
        lines = [line.strip() for line in text.splitlines()]
        text = "\n".join(line for line in lines if line)
        
        if len(text) > 40000:
             text = text[:40000] + "\n...[truncated]"
             
        # Use the global _safe_json injected, or our fallback
        return globals().get("_safe_json", _safe_json)({"status": "ok", "url": url, "content": text})
    except urllib.error.URLError as e:
        return globals().get("_safe_json", _safe_json)({"status": "error", "error": f"Failed to fetch: {e.reason}"})
    except Exception as e:
        return globals().get("_safe_json", _safe_json)({"status": "error", "error": str(e)})


__tools__ = [fetch_url]
