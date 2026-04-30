"""Simple deterministic deduplication for the MVP processing layer."""
from __future__ import annotations


def dedupe_source_items(items: list[dict]) -> list[dict]:
    """Return only unique items, preserving the order of first unique entries."""
    unique_items, _metrics = dedupe_source_items_with_metrics(items)
    return unique_items


def dedupe_source_items_with_metrics(items: list[dict]) -> tuple[list[dict], dict]:
    """Remove duplicates by normalized URL and then by normalized title."""
    seen_urls = set()
    seen_titles = set()
    unique_items = []
    duplicate_urls_removed = 0
    duplicate_titles_removed = 0

    for item in items:
        url_key = normalize_url(item.get("url", ""))
        if url_key in seen_urls:
            duplicate_urls_removed += 1
            continue

        title_key = normalize_title(item.get("title", ""))
        if title_key and title_key in seen_titles:
            duplicate_titles_removed += 1
            continue

        seen_urls.add(url_key)
        if title_key:
            seen_titles.add(title_key)
        unique_items.append(item)

    metrics = {
        "input_count": len(items),
        "output_count": len(unique_items),
        "duplicate_urls_removed": duplicate_urls_removed,
        "duplicate_titles_removed": duplicate_titles_removed,
        "duplicates_removed": duplicate_urls_removed + duplicate_titles_removed,
    }
    return unique_items, metrics


def normalize_url(url: str) -> str:
    """Normalize URL for deterministic URL-based deduplication."""
    return str(url).strip().rstrip("/").lower()


def normalize_title(title: str) -> str:
    """Normalize title for simple title-based deduplication."""
    return " ".join(str(title).strip().lower().split())
