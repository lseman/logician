from __future__ import annotations

if "llm" not in globals():

    class _NoOpLLM:
        def tool(self, func=None, *, name=None, description=None):
            return func if func is not None else (lambda f: f)

    llm = _NoOpLLM()

import json
import urllib.request
import urllib.error

# Fallback in case _safe_json is not injected by the agent environment
if "_safe_json" not in globals():
    def _safe_json(obj) -> str:
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return json.dumps({"status": "error", "error": repr(obj)})

@llm.tool(description="Fetch a URL and return its text content (markdown or plain text).")
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
