"""Простая детерминированная дедупликация источников для MVP."""
from __future__ import annotations


def dedupe_source_items(items: list[dict]) -> list[dict]:
    """Удалить дубли сначала по URL, затем по нормализованному title."""
    seen_urls = set()
    seen_titles = set()
    unique_items = []

    for item in items:
        url_key = item["url"].rstrip("/").lower()
        title_key = item["title"].strip().lower()
        if url_key in seen_urls or title_key in seen_titles:
            continue
        seen_urls.add(url_key)
        seen_titles.add(title_key)
        unique_items.append(item)

    return unique_items
