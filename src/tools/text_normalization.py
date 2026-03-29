from __future__ import annotations

import ast
import json
from typing import Any


def normalize_text_payload(
    text: Any,
    *,
    language_hint: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Normalize agent-provided text payloads conservatively.

    The goal is to fix transport artifacts without corrupting legitimate source
    text, especially string literals that intentionally contain escape sequences.
    """
    if not isinstance(text, str):
        return str(text), {"transformations": ["non_string_input"]}
    if text == "":
        return text, {"transformations": []}

    meta: dict[str, Any] = {"transformations": [], "warnings": []}
    content = text

    if content.startswith("\ufeff"):
        content = content[1:]
        meta["transformations"].append("bom_stripped")

    content, fence_meta = strip_outer_code_fences(content)
    if fence_meta["stripped"]:
        meta["transformations"].append("fences_stripped")
        meta["fence"] = fence_meta

    candidates: list[tuple[str, str]] = [("raw", content)]

    unwrapped = _unwrap_outer_string_literal(content)
    if unwrapped is not None and unwrapped != content:
        candidates.append(("outer_string_unwrapped", unwrapped))

    expanded_candidates = list(candidates)
    for label, candidate in candidates:
        if _looks_like_escaped_multiline_payload(candidate):
            decoded = _decode_linebreak_escapes_outside_strings(candidate)
            if decoded != candidate:
                expanded_candidates.append((f"{label}_linebreaks_decoded", decoded))

    chosen_label, chosen = _choose_best_candidate(
        expanded_candidates,
        language_hint=language_hint,
    )

    chosen = normalize_text_for_matching(chosen)
    if chosen_label != "raw":
        meta["transformations"].append(chosen_label)
    if meta["transformations"]:
        meta["summary"] = f"Applied: {', '.join(meta['transformations'])}"
    return chosen, meta


def strip_outer_code_fences(content: str) -> tuple[str, dict[str, Any]]:
    """Strip a single outer fenced block if the entire payload is fenced."""
    stripped = content.strip()
    if not stripped.startswith(("```", "~~~")):
        return content, {"stripped": False}

    lines = content.splitlines(keepends=True)
    if len(lines) < 2:
        return content, {"stripped": False}

    first = lines[0].strip()
    last = lines[-1].strip()
    fence = "```" if first.startswith("```") else "~~~"
    if not last.startswith(fence):
        return content, {"stripped": False}

    language = first.split(maxsplit=1)[1] if len(first.split()) > 1 else None
    return "".join(lines[1:-1]), {
        "stripped": True,
        "type": fence,
        "language": language,
    }


def normalize_text_for_matching(text: str) -> str:
    if not isinstance(text, str):
        return text
    if text.startswith("\ufeff"):
        text = text[1:]
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _unwrap_outer_string_literal(text: str) -> str | None:
    stripped = text.strip()
    if len(stripped) < 2:
        return None
    if stripped[0] not in ('"', "'"):
        return None

    if stripped[0] == '"' and stripped[-1] == '"':
        try:
            decoded = json.loads(stripped)
            if isinstance(decoded, str):
                return decoded
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    try:
        decoded = ast.literal_eval(stripped)
        if isinstance(decoded, str):
            return decoded
    except (SyntaxError, ValueError, TypeError):
        return None
    return None


def _looks_like_escaped_multiline_payload(text: str) -> bool:
    if "\n" in text or "\r" in text:
        return False
    return any(seq in text for seq in ("\\n", "\\r", "\\t"))


def _decode_linebreak_escapes_outside_strings(text: str) -> str:
    """Decode \\n/\\r/\\t only when they appear outside quoted strings."""
    out: list[str] = []
    i = 0
    quote: str | None = None
    triple = False
    n = len(text)

    while i < n:
        if quote is None:
            if text.startswith("'''", i) or text.startswith('"""', i):
                quote = text[i]
                triple = True
                out.append(text[i : i + 3])
                i += 3
                continue
            ch = text[i]
            if ch in {"'", '"'}:
                quote = ch
                triple = False
                out.append(ch)
                i += 1
                continue
            if ch == "\\" and i + 1 < n:
                esc = text[i + 1]
                if esc == "n":
                    out.append("\n")
                    i += 2
                    continue
                if esc == "r":
                    out.append("\r")
                    i += 2
                    continue
                if esc == "t":
                    out.append("\t")
                    i += 2
                    continue
            out.append(ch)
            i += 1
            continue

        if triple:
            closing = quote * 3
            if text.startswith(closing, i):
                out.append(closing)
                i += 3
                quote = None
                triple = False
                continue
            out.append(text[i])
            i += 1
            continue

        ch = text[i]
        if ch == "\\" and i + 1 < n:
            out.append(text[i : i + 2])
            i += 2
            continue
        out.append(ch)
        i += 1
        if ch == quote:
            quote = None

    return "".join(out)


def _choose_best_candidate(
    candidates: list[tuple[str, str]],
    *,
    language_hint: str | None = None,
) -> tuple[str, str]:
    best_label, best_text = candidates[0]
    best_score = _score_candidate(best_text, language_hint=language_hint)
    for label, text in candidates[1:]:
        score = _score_candidate(text, language_hint=language_hint)
        if score > best_score:
            best_label, best_text, best_score = label, text, score
    return best_label, best_text


def _score_candidate(text: str, *, language_hint: str | None = None) -> int:
    score = 0
    if "\n" in text:
        score += 10
    score -= text.count("\\n") * 2
    score -= text.count("\\r") * 2
    score -= text.count("\\t")

    if (language_hint or "").strip().lower() == "python":
        stripped = text.strip()
        if not stripped:
            return score
        try:
            ast.parse(text)
        except SyntaxError as exc:
            score -= 5
            msg = str(exc).lower()
            if "unterminated string" in msg or "eol while scanning string literal" in msg:
                score -= 20
        else:
            score += 50
    return score
