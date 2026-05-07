"""Deterministic preprocessing before the first AI calls."""
from __future__ import annotations

from bs4 import BeautifulSoup


MIN_CONTENT_LENGTH = 200


def clean_source_items(raw_items: list[dict] | object) -> list[dict]:
    """Normalize source items, convert HTML to text, and drop weak entries."""
    cleaned = []
    for item in list(raw_items):
        title = _clean_text(str(item.get("title", "")))
        url = str(item.get("url", "")).strip()
        if not title or not url:
            continue

        source_value = item.get("source_name") or item.get("source") or "unknown"
        raw_content = (
            item.get("content")
            or item.get("body")
            or item.get("text")
            or item.get("snippet")
            or ""
        )
        normalized_content = extract_text(str(raw_content))
        if len(normalized_content) < MIN_CONTENT_LENGTH:
            continue

        raw_snippet = item.get("snippet") or normalized_content
        normalized_snippet = extract_text(str(raw_snippet))

        cleaned.append(
            {
                "title": title,
                "url": url,
                "source": _clean_text(str(source_value)),
                "source_name": _clean_text(str(source_value)),
                "source_url": str(item.get("source_url") or "").strip() or None,
                "source_api_url": str(item.get("source_api_url") or "").strip() or None,
                "published_at": item.get("published_at"),
                "snippet": normalized_snippet,
                "content": normalized_content,
                "description": item.get("description"),
                "metadata": item.get("metadata") if isinstance(item.get("metadata"), dict) else {},
            }
        )
    return cleaned


def extract_text(raw_html: str) -> str:
    """Convert HTML-heavy RSS content into readable plain text."""
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    return _clean_text(soup.get_text(separator=" ", strip=True))


def _clean_text(value: str) -> str:
    return " ".join(value.split())
