"""Source ingestion helpers for RSS and dev.to sources."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup


DEVTO_HOST = "dev.to"
DEVTO_API_ROOT = "https://dev.to/api/articles"


@dataclass(frozen=True)
class NormalizedSource:
    original_url: str
    normalized_url: str
    source_type: str
    platform: str
    metadata: dict[str, Any]


def fetch_rss_articles(source_url: str, limit: int = 10) -> list[dict]:
    """Fetch source items from RSS or supported dev.to sources."""
    normalized_source = normalize_source_url(source_url)

    if normalized_source.source_type in {"dev_to_tag", "dev_to_api_list"}:
        return _fetch_dev_to_articles_from_list_source(normalized_source, limit=limit)

    if normalized_source.source_type == "dev_to_article":
        article = _fetch_dev_to_single_article(normalized_source)
        return [article] if article else []

    return _fetch_rss_feed_articles(normalized_source, limit=limit)


def get_rss_debug_snapshot(source_url: str, sample_size: int = 5) -> dict[str, Any]:
    """Return a debug snapshot for RSS parsing and supported source normalization."""
    normalized_source = normalize_source_url(source_url)

    if normalized_source.source_type in {"dev_to_tag", "dev_to_api_list", "dev_to_article"}:
        return {
            "feed_url": source_url,
            "normalized_url": normalized_source.normalized_url,
            "feed_title": "",
            "total_entries": 0,
            "skip_reason": None,
            "bozo": None,
            "bozo_exception": None,
            "status": None,
            "href": normalized_source.normalized_url,
            "source_type": normalized_source.source_type,
            "platform": normalized_source.platform,
            "entries": [],
        }

    feed, parse_error = _parse_feed(normalized_source.normalized_url)
    if parse_error:
        return {
            "feed_url": source_url,
            "normalized_url": normalized_source.normalized_url,
            "feed_title": "",
            "total_entries": 0,
            "skip_reason": parse_error,
            "bozo": None,
            "bozo_exception": None,
            "status": None,
            "href": None,
            "source_type": normalized_source.source_type,
            "platform": normalized_source.platform,
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
        "feed_url": source_url,
        "normalized_url": normalized_source.normalized_url,
        "feed_title": (getattr(feed.feed, "title", "") or "").strip(),
        "total_entries": len(entries),
        "skip_reason": feed_skip_reason,
        "bozo": getattr(feed, "bozo", None),
        "bozo_exception": str(bozo_exception) if bozo_exception else None,
        "status": getattr(feed, "status", None),
        "href": getattr(feed, "href", None),
        "source_type": normalized_source.source_type,
        "platform": normalized_source.platform,
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


def normalize_source_url(url: str) -> NormalizedSource:
    """Normalize a human-facing source URL into an internal fetchable source."""
    parsed = urlparse(url)
    host = _normalized_host(parsed.netloc)
    path = parsed.path.strip("/")
    query = parse_qs(parsed.query)

    if host == DEVTO_HOST:
        path_parts = [part for part in path.split("/") if part]

        if len(path_parts) >= 2 and path_parts[0] == "t":
            tag = path_parts[1]
            return NormalizedSource(
                original_url=url,
                normalized_url=build_dev_to_api_url(tag),
                source_type="dev_to_tag",
                platform="dev.to",
                metadata={"tag": tag},
            )

        if path_parts[:2] == ["api", "articles"]:
            tag = (query.get("tag") or [""])[0].strip()
            metadata: dict[str, Any] = {}
            if tag:
                metadata["tag"] = tag
            return NormalizedSource(
                original_url=url,
                normalized_url=url,
                source_type="dev_to_api_list",
                platform="dev.to",
                metadata=metadata,
            )

        if len(path_parts) >= 2 and path_parts[0] not in {"api", "t"}:
            return NormalizedSource(
                original_url=url,
                normalized_url=url,
                source_type="dev_to_article",
                platform="dev.to",
                metadata={"author": path_parts[0], "slug": path_parts[1]},
            )

    return NormalizedSource(
        original_url=url,
        normalized_url=url,
        source_type="rss_feed",
        platform=host or "unknown",
        metadata={},
    )


def detect_source_type(url: str) -> str:
    return normalize_source_url(url).source_type


def build_dev_to_api_url(tag: str) -> str:
    return f"{DEVTO_API_ROOT}?tag={tag}"


def fetch_dev_to_article_list(api_url: str) -> list[dict]:
    payload = _fetch_json(api_url)
    return payload if isinstance(payload, list) else []


def fetch_dev_to_article_content(article_id_or_url: int | str) -> dict[str, Any] | None:
    if isinstance(article_id_or_url, int) or str(article_id_or_url).isdigit():
        article_id = int(article_id_or_url)
        detail_url = f"{DEVTO_API_ROOT}/{article_id}"
        payload = _fetch_json(detail_url)
        if isinstance(payload, dict):
            return {
                "title": str(payload.get("title", "")).strip(),
                "url": str(payload.get("url", "")).strip(),
                "description": str(payload.get("description") or "").strip() or None,
                "content": _extract_dev_to_text(payload),
                "published_at": payload.get("published_at"),
                "metadata": {
                    "devto_id": payload.get("id", article_id),
                    "tag_list": payload.get("tag_list") or [],
                    "positive_reactions_count": payload.get("positive_reactions_count"),
                    "comments_count": payload.get("comments_count"),
                    "public_reactions_count": payload.get("public_reactions_count"),
                    "reading_time_minutes": payload.get("reading_time_minutes"),
                    "cover_image": payload.get("cover_image"),
                },
            }

    article_url = str(article_id_or_url).strip()
    if not article_url:
        return None

    html = _fetch_url_text(article_url)
    if not html:
        return None

    return {
        "title": _extract_html_title(html),
        "url": article_url,
        "description": None,
        "content": _extract_html_text(html),
        "published_at": None,
        "metadata": {},
    }


def _fetch_dev_to_articles_from_list_source(
    normalized_source: NormalizedSource,
    limit: int,
) -> list[dict[str, Any]]:
    article_list = fetch_dev_to_article_list(normalized_source.normalized_url)
    candidates: list[dict[str, Any]] = []

    for item in article_list[:limit]:
        if not isinstance(item, dict):
            continue

        title = str(item.get("title", "")).strip()
        article_url = str(item.get("url", "")).strip()
        article_id = item.get("id")
        detail = fetch_dev_to_article_content(article_id or article_url)

        content = ""
        detail_title = title
        description = str(item.get("description") or "").strip() or None
        detail_metadata: dict[str, Any] = {}
        detail_published_at = item.get("published_at")

        if detail:
            content = str(detail.get("content", "")).strip()
            detail_title = str(detail.get("title") or title).strip() or title
            article_url = str(detail.get("url") or article_url).strip()
            description = detail.get("description") or description
            detail_metadata = detail.get("metadata", {}) if isinstance(detail.get("metadata"), dict) else {}
            detail_published_at = detail.get("published_at") or detail_published_at

        metadata = {
            "platform": "dev.to",
            "source_type": normalized_source.source_type,
            "devto_id": article_id,
            "tag_list": item.get("tag_list") or detail_metadata.get("tag_list") or [],
            "positive_reactions_count": item.get("positive_reactions_count"),
            "comments_count": item.get("comments_count"),
            "public_reactions_count": item.get("public_reactions_count"),
            "published_at": detail_published_at,
            "content_unavailable": not bool(content),
        }
        metadata.update(detail_metadata)

        snippet = description or _build_snippet(content)
        candidates.append(
            {
                "title": detail_title or title,
                "url": article_url,
                "source_url": normalized_source.original_url,
                "source_api_url": normalized_source.normalized_url,
                "source_name": "DEV Community",
                "source_type": normalized_source.source_type,
                "platform": "dev.to",
                "content": content,
                "snippet": snippet,
                "description": description,
                "published_at": detail_published_at,
                "metadata": metadata,
            }
        )

    return candidates


def _fetch_dev_to_single_article(normalized_source: NormalizedSource) -> dict[str, Any] | None:
    detail = fetch_dev_to_article_content(normalized_source.normalized_url)
    if detail is None:
        return None

    content = str(detail.get("content", "")).strip()
    return {
        "title": str(detail.get("title", "")).strip(),
        "url": normalized_source.normalized_url,
        "source_url": normalized_source.original_url,
        "source_api_url": None,
        "source_name": "DEV Community",
        "source_type": normalized_source.source_type,
        "platform": "dev.to",
        "content": content,
        "snippet": str(detail.get("description") or _build_snippet(content)).strip(),
        "description": detail.get("description"),
        "published_at": detail.get("published_at"),
        "metadata": {
            "platform": "dev.to",
            "source_type": normalized_source.source_type,
            "content_unavailable": not bool(content),
            **(detail.get("metadata", {}) if isinstance(detail.get("metadata"), dict) else {}),
        },
    }


def _fetch_rss_feed_articles(normalized_source: NormalizedSource, limit: int = 10) -> list[dict]:
    """Fetch RSS items and map them to the DigestFlow source item format."""
    feed, parse_error = _parse_feed(normalized_source.normalized_url)
    if parse_error or feed is None:
        return []

    entries = getattr(feed, "entries", None) or []
    if not entries:
        return []

    feed_title = (getattr(feed.feed, "title", "") or "").strip()
    fallback_source_name = _get_fallback_source_name(normalized_source.normalized_url)
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
                "source_url": normalized_source.original_url,
                "source_api_url": None,
                "source_name": source_name,
                "source_type": normalized_source.source_type,
                "platform": normalized_source.platform,
                "snippet": snippet,
                "content": snippet,
                "description": None,
                "metadata": {
                    "platform": normalized_source.platform,
                    "source_type": normalized_source.source_type,
                    "content_unavailable": not bool(snippet),
                    "published_at": published_at.isoformat() if published_at else None,
                },
                "published_at": published_at.isoformat() if published_at else None,
            }
        )

        if len(articles) >= limit:
            break

    return articles


def _extract_dev_to_text(payload: dict[str, Any]) -> str:
    body_markdown = str(payload.get("body_markdown") or "").strip()
    if body_markdown:
        return _clean_text(body_markdown)

    body_html = str(payload.get("body_html") or "").strip()
    if body_html:
        return _extract_html_text(body_html)

    description = str(payload.get("description") or "").strip()
    return _clean_text(description)


def _extract_html_text(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find("article")
    text = article.get_text(separator=" ", strip=True) if article else soup.get_text(separator=" ", strip=True)
    return _clean_text(text)


def _extract_html_title(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        return _clean_text(str(og_title["content"]))
    if soup.title and soup.title.string:
        return _clean_text(soup.title.string)
    return ""


def _build_snippet(content: str, limit: int = 280) -> str:
    normalized = _clean_text(content)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit - 3].rstrip() + "..."


def _clean_text(value: str) -> str:
    return " ".join(value.split())


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
    return _fetch_url_bytes(feed_url)


def _fetch_url_bytes(
    source_url: str,
    accept_header: str = "application/rss+xml, application/xml, text/xml, */*",
) -> bytes | None:
    local_path = _resolve_local_feed_path(source_url)
    if local_path is not None:
        try:
            return local_path.read_bytes()
        except Exception:
            return None

    request = Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DigestFlowRSS/0.1)",
            "Accept": accept_header,
        },
    )
    try:
        with urlopen(request, timeout=15) as response:
            return response.read()
    except Exception:
        return None


def _fetch_json(source_url: str) -> Any:
    content = _fetch_url_bytes(source_url, accept_header="application/json, text/json, */*")
    if not content:
        return None
    try:
        return json.loads(content.decode("utf-8"))
    except Exception:
        return None


def _fetch_url_text(source_url: str) -> str:
    content = _fetch_url_bytes(source_url, accept_header="text/html, application/xhtml+xml, */*")
    if not content:
        return ""
    return content.decode("utf-8", errors="ignore")


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


def _normalized_host(netloc: str) -> str:
    host = (netloc or "").lower()
    return host[4:] if host.startswith("www.") else host
