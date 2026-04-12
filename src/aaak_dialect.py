from __future__ import annotations

import re
from typing import Any

_EMOTION_SIGNALS: dict[str, str] = {
    "decided": "determ",
    "because": "determ",
    "instead": "determ",
    "prefer": "convict",
    "switched": "decision",
    "chose": "decision",
    "worried": "anx",
    "excited": "excite",
    "frustrated": "frust",
    "confused": "confuse",
    "love": "love",
    "hate": "rage",
    "hope": "hope",
    "fear": "fear",
    "trust": "trust",
    "happy": "joy",
    "sad": "grief",
    "surprised": "surprise",
    "grateful": "grat",
    "curious": "curious",
    "wonder": "wonder",
    "anxious": "anx",
    "relieved": "relief",
    "satisfied": "satis",
}

_FLAG_SIGNALS: dict[str, str] = {
    "decided": "DECISION",
    "chose": "DECISION",
    "switched": "DECISION",
    "migrated": "DECISION",
    "replaced": "DECISION",
    "because": "DECISION",
    "founded": "ORIGIN",
    "created": "ORIGIN",
    "started": "ORIGIN",
    "born": "ORIGIN",
    "launched": "ORIGIN",
    "core": "CORE",
    "fundamental": "CORE",
    "principle": "CORE",
    "belief": "CORE",
    "turning point": "PIVOT",
    "realized": "PIVOT",
    "breakthrough": "PIVOT",
    "api": "TECHNICAL",
    "database": "TECHNICAL",
    "architecture": "TECHNICAL",
    "deploy": "TECHNICAL",
    "algorithm": "TECHNICAL",
}

_STOP_WORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "to",
    "of",
    "in",
    "for",
    "on",
    "with",
    "at",
    "by",
    "from",
    "as",
    "into",
    "about",
    "between",
    "through",
    "during",
    "so",
    "very",
    "just",
    "now",
    "and",
    "but",
    "or",
    "if",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "i",
    "we",
    "you",
    "he",
    "she",
    "they",
    "me",
    "him",
    "her",
    "us",
    "them",
    "my",
    "your",
    "his",
    "our",
    "their",
    "also",
    "much",
    "many",
    "like",
    "because",
    "since",
    "get",
    "got",
    "use",
    "used",
    "using",
    "make",
    "made",
    "thing",
    "things",
    "way",
    "well",
    "really",
    "want",
    "need",
}


class AAAKDialect:
    def __init__(self) -> None:
        self.entity_codes: dict[str, str] = {}

    def _detect_emotions(self, text: str) -> list[str]:
        text_lower = text.lower()
        detected: list[str] = []
        for keyword, code in _EMOTION_SIGNALS.items():
            if keyword in text_lower and code not in detected:
                detected.append(code)
        return detected[:3]

    def _detect_flags(self, text: str) -> list[str]:
        text_lower = text.lower()
        detected: list[str] = []
        for keyword, flag in _FLAG_SIGNALS.items():
            if keyword in text_lower and flag not in detected:
                detected.append(flag)
        return detected[:3]

    def _extract_topics(self, text: str, max_topics: int = 3) -> list[str]:
        words = re.findall(r"[a-zA-Z][a-zA-Z_-]{2,}", text)
        freq: dict[str, int] = {}
        for w in words:
            word = w.lower()
            if word in _STOP_WORDS or len(word) < 3:
                continue
            freq[word] = freq.get(word, 0) + 1
            if w[0].isupper():
                freq[word] += 1
            if "_" in w or "-" in w or any(c.isupper() for c in w[1:]):
                freq[word] += 1
        ranked = sorted(freq.items(), key=lambda item: -item[1])
        return [word for word, _ in ranked[:max_topics]]

    def _extract_key_sentence(self, text: str) -> str:
        sentences = re.split(r"[.!?\n]+", text)
        candidates = [s.strip() for s in sentences if len(s.strip()) > 10]
        if not candidates:
            return ""
        keywords = {
            "decided",
            "because",
            "instead",
            "prefer",
            "switched",
            "chose",
            "realized",
            "important",
            "key",
            "critical",
            "discovered",
            "learned",
            "conclusion",
            "solution",
            "reason",
            "breakthrough",
            "insight",
        }
        scored: list[tuple[int, str]] = []
        for s in candidates:
            score = 0
            s_lower = s.lower()
            for term in keywords:
                if term in s_lower:
                    score += 2
            if len(s) < 80:
                score += 1
            if len(s) < 40:
                score += 1
            if len(s) > 150:
                score -= 2
            scored.append((score, s))
        scored.sort(key=lambda item: -item[0])
        best = scored[0][1]
        return best[:52] + "..." if len(best) > 55 else best

    def _detect_entities_in_text(self, text: str) -> list[str]:
        found: list[str] = []
        words = re.findall(r"[A-Za-z][a-z]+", text)
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w.lower() not in _STOP_WORDS:
                code = w[:3].upper()
                if code not in found:
                    found.append(code)
                if len(found) >= 3:
                    break
        return found or ["???"]

    def compress(self, text: str, metadata: dict[str, Any] | None = None) -> str:
        metadata = metadata or {}
        entities = self._detect_entities_in_text(text)
        entity_str = "+".join(entities[:3])
        topics = self._extract_topics(text)
        topic_str = "_".join(topics[:3]) if topics else "misc"
        quote = self._extract_key_sentence(text)
        quote_part = f'"{quote}"' if quote else ""
        emotions = self._detect_emotions(text)
        emotion_str = "+".join(emotions) if emotions else ""
        flags = self._detect_flags(text)
        flag_str = "+".join(flags) if flags else ""
        source = metadata.get("source_file", "?")
        session = metadata.get("session", "?")
        date = metadata.get("date", "?")
        title = metadata.get("title", "?")
        header = f"{session}|{date}|{source}|{title}" if any([session, date, source, title]) else ""
        parts = [f"0:{entity_str}", topic_str]
        if quote_part:
            parts.append(quote_part)
        if emotion_str:
            parts.append(emotion_str)
        if flag_str:
            parts.append(flag_str)
        body = "|".join(parts)
        return f"{header}\n{body}" if header else body


def aaak_available() -> bool:
    return True


def compress_text_to_aaak(text: str, metadata: dict[str, Any] | None = None) -> str:
    try:
        dialect = AAAKDialect()
        return dialect.compress(str(text or ""), metadata=metadata or {})
    except Exception:
        return str(text or "")
