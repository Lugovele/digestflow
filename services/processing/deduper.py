"""Простая детерминированная дедупликация источников для MVP."""
from __future__ import annotations


def dedupe_source_items(items: list[dict]) -> list[dict]:
    """Удалить дубли по нормализованному URL, сохраняя порядок первых уникальных items."""
    seen_urls = set()
    unique_items = []

    for item in items:
        url_key = normalize_url(item["url"])
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)
        unique_items.append(item)

    return unique_items


def normalize_url(url: str) -> str:
    """Нормализовать URL для детерминированной дедупликации."""
    return url.strip().rstrip("/").lower()
