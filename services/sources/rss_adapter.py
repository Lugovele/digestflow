"""Minimal RSS adapter for the DigestFlow MVP.

This adapter fetches articles from a single RSS feed and maps them into the
pipeline contract. It is intentionally simple and is not production-ready
ingestion.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def fetch_rss_articles(feed_url: str, limit: int = 10) -> list[dict]:
    """Fetch RSS items and map them to the DigestFlow source item format."""
    feed, parse_error = _parse_feed(feed_url)
    if parse_error or feed is None:
        return []

    entries = getattr(feed, "entries", None) or []
    if not entries:
        return []

    feed_title = (getattr(feed.feed, "title", "") or "").strip()
    fallback_source_name = _get_fallback_source_name(feed_url)
    source_name = feed_title or fallback_source_name

    articles: list[dict] = []

    for entry in entries:
        skip_reason = _get_skip_reason(entry)
        if skip_reason:
            continue

        title = (_get_entry_value(entry, "title") or "").strip()
        url = (_get_entry_value(entry, "link", "url", "id") or "").strip()

        raw_summary = _get_entry_value(entry, "summary", "description", "content") or ""
        snippet = _strip_html(raw_summary)
        if len(snippet) > 300:
            snippet = snippet[:297].rstrip() + "..."

        published_at = None
        published_parsed = _get_entry_value(entry, "published_parsed")
        if published_parsed:
            try:
                published_at = datetime(
                    published_parsed.tm_year,
                    published_parsed.tm_mon,
                    published_parsed.tm_mday,
                    published_parsed.tm_hour,
                    published_parsed.tm_min,
                    published_parsed.tm_sec,
                    tzinfo=timezone.utc,
                )
            except Exception:
                published_at = None

        articles.append(
            {
                "title": title,
                "url": url,
                "source_name": source_name,
                "snippet": snippet,
                "published_at": published_at.isoformat() if published_at else None,
            }
        )

        if len(articles) >= limit:
            break

    return articles


def get_rss_debug_snapshot(feed_url: str, sample_size: int = 5) -> dict[str, Any]:
    """Return a lightweight debug snapshot for RSS parsing and filtering."""
    feed, parse_error = _parse_feed(feed_url)
    if parse_error:
        return {
            "feed_url": feed_url,
            "feed_title": "",
            "total_entries": 0,
            "skip_reason": parse_error,
            "bozo": None,
            "bozo_exception": None,
            "status": None,
            "href": None,
            "entries": [],
        }

    entries = getattr(feed, "entries", None) or []
    bozo_exception = getattr(feed, "bozo_exception", None)
    feed_skip_reason = None
    if getattr(feed, "bozo", False) and bozo_exception:
        feed_skip_reason = f"parse error: {bozo_exception}"
    elif not entries:
        feed_skip_reason = "empty feed"

    return {
        "feed_url": feed_url,
        "feed_title": (getattr(feed.feed, "title", "") or "").strip(),
        "total_entries": len(entries),
        "skip_reason": feed_skip_reason,
        "bozo": getattr(feed, "bozo", None),
        "bozo_exception": str(bozo_exception) if bozo_exception else None,
        "status": getattr(feed, "status", None),
        "href": getattr(feed, "href", None),
        "entries": [
            {
                "available_keys": _entry_available_keys(entry),
                "raw_title": _get_entry_value(entry, "title"),
                "raw_link": _get_entry_value(entry, "link"),
                "raw_id": _get_entry_value(entry, "id"),
                "raw_summary": _get_entry_value(entry, "summary"),
                "raw_description": _get_entry_value(entry, "description"),
                "raw_published": _get_entry_value(entry, "published"),
                "skip_reason": _get_skip_reason(entry),
            }
            for entry in entries[:sample_size]
        ],
    }


def _strip_html(text: str) -> str:
    """Remove basic HTML tags and normalize whitespace."""
    if not text:
        return ""

    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _is_valid_url(url: str) -> bool:
    parsed = urlparse(url)
    return bool(parsed.scheme and parsed.netloc)


def _parse_feed(feed_url: str):
    try:
        import feedparser

        feed_content = _fetch_feed_content(feed_url)
        if feed_content:
            return feedparser.parse(feed_content), None

        return feedparser.parse(feed_url), None
    except Exception:
        return None, "parse error"


def _fetch_feed_content(feed_url: str) -> bytes | None:
    local_path = _resolve_local_feed_path(feed_url)
    if local_path is not None:
        try:
            return local_path.read_bytes()
        except Exception:
            return None

    request = Request(
        feed_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DigestFlowRSS/0.1)",
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            return response.read()
    except Exception:
        return None


def _resolve_local_feed_path(feed_url: str) -> Path | None:
    parsed = urlparse(feed_url)

    if parsed.scheme == "file":
        raw_path = parsed.path
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{parsed.path}"
        return Path(raw_path)

    if parsed.scheme in {"http", "https"}:
        return None

    candidate = Path(feed_url)
    return candidate if candidate.exists() and candidate.is_file() else None


def _get_fallback_source_name(feed_url: str) -> str:
    parsed = urlparse(feed_url)
    if parsed.scheme in {"http", "https"}:
        return parsed.netloc or "Unknown source"

    local_path = _resolve_local_feed_path(feed_url)
    if local_path is not None:
        return local_path.stem or "Local RSS"

    return parsed.netloc or "Unknown source"


def _get_skip_reason(entry) -> str | None:
    title = (_get_entry_value(entry, "title") or "").strip()
    if not title:
        return "missing title"

    raw_url = (_get_entry_value(entry, "link", "url", "id") or "").strip()
    if not raw_url:
        return "missing url"

    if not _is_valid_url(raw_url):
        return "invalid url"

    return None


def _get_entry_value(entry, *names: str):
    for name in names:
        if isinstance(entry, dict) and name in entry:
            value = entry.get(name)
            if name == "content" and isinstance(value, list) and value:
                first_item = value[0]
                if isinstance(first_item, dict):
                    return first_item.get("value", "")
            return value

        if hasattr(entry, name):
            value = getattr(entry, name)
            if name == "content" and isinstance(value, list) and value:
                first_item = value[0]
                if isinstance(first_item, dict):
                    return first_item.get("value", "")
            return value

    return None


def _entry_available_keys(entry) -> list[str]:
    if isinstance(entry, dict):
        return sorted(str(key) for key in entry.keys())

    keys_method = getattr(entry, "keys", None)
    if callable(keys_method):
        try:
            return sorted(str(key) for key in keys_method())
        except Exception:
            return []

    return sorted(str(key) for key in vars(entry).keys()) if hasattr(entry, "__dict__") else []
