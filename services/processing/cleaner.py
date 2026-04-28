"""Детерминированная предобработка перед любым AI-вызовом."""
from __future__ import annotations

from html import unescape


def clean_source_items(raw_items: list[dict] | object) -> list[dict]:
    """Нормализовать источники и убрать элементы без title или URL."""
    cleaned = []
    for item in list(raw_items):
        title = _clean_text(str(item.get("title", "")))
        url = str(item.get("url", "")).strip()
        if not title or not url:
            continue

        cleaned.append(
            {
                "title": title,
                "url": url,
                "source": _clean_text(str(item.get("source", "unknown"))),
                "published_at": item.get("published_at"),
                "snippet": _clean_text(str(item.get("snippet", ""))),
            }
        )
    return cleaned


def _clean_text(value: str) -> str:
    return " ".join(unescape(value).split())
