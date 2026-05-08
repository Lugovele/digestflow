"""Deterministic source URL classification for ingestion strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse


DEVTO_HOST = "dev.to"
DEVTO_API_ROOT = "https://dev.to/api/articles"
BLOG_INDEX_SEGMENTS = {"blog", "news", "articles", "posts", "insights", "updates"}
RSS_PATH_SUFFIXES = (
    ".xml",
    ".rss",
    "/feed",
    "/feed/",
    "/rss",
    "/rss/",
    "/rss.xml",
    "/atom.xml",
)


@dataclass(frozen=True)
class NormalizedSource:
    original_url: str
    normalized_url: str
    source_type: str
    platform: str
    detection_reason: str
    metadata: dict[str, Any]


def detect_source_type(url: str) -> str:
    return classify_source_url(url).source_type


def classify_source_url(url: str) -> NormalizedSource:
    parsed = urlparse(url)
    host = _normalized_host(parsed.netloc)
    path = parsed.path or "/"
    normalized_path = path.rstrip("/") or "/"
    query = parse_qs(parsed.query)
    metadata: dict[str, Any] = {}

    if host == DEVTO_HOST:
        path_parts = [part for part in normalized_path.strip("/").split("/") if part]

        if len(path_parts) >= 2 and path_parts[0] == "t":
            tag = path_parts[1].strip()
            metadata["tag"] = tag
            return NormalizedSource(
                original_url=url,
                normalized_url=build_dev_to_api_url(tag),
                source_type="devto_tag",
                platform="dev.to",
                detection_reason="matched dev.to topic pattern",
                metadata=metadata,
            )

        if path_parts[:2] == ["api", "articles"]:
            tag = (query.get("tag") or [""])[0].strip()
            if tag:
                metadata["tag"] = tag
            metadata["input_variant"] = "api_list"
            return NormalizedSource(
                original_url=url,
                normalized_url=url,
                source_type="devto_tag",
                platform="dev.to",
                detection_reason="matched dev.to API articles endpoint",
                metadata=metadata,
            )

        if len(path_parts) >= 2 and path_parts[0] not in {"api", "t"}:
            metadata["author"] = path_parts[0]
            metadata["slug"] = path_parts[1]
            return NormalizedSource(
                original_url=url,
                normalized_url=url,
                source_type="devto_article",
                platform="dev.to",
                detection_reason="matched dev.to article path",
                metadata=metadata,
            )

    if _looks_like_rss(parsed, normalized_path, query):
        return NormalizedSource(
            original_url=url,
            normalized_url=url,
            source_type="rss_feed",
            platform=host or "unknown",
            detection_reason="matched RSS/XML URL pattern",
            metadata=metadata,
        )

    path_parts = [part for part in normalized_path.strip("/").split("/") if part]
    first_segment = path_parts[0].lower() if path_parts else ""

    if first_segment in BLOG_INDEX_SEGMENTS:
        metadata["path_segment"] = first_segment
        return NormalizedSource(
            original_url=url,
            normalized_url=url,
            source_type="blog_index",
            platform=host or "unknown",
            detection_reason="matched blog or news index path",
            metadata=metadata,
        )

    if normalized_path == "/":
        return NormalizedSource(
            original_url=url,
            normalized_url=url,
            source_type="publication",
            platform=host or "unknown",
            detection_reason="matched publication homepage pattern",
            metadata=metadata,
        )

    return NormalizedSource(
        original_url=url,
        normalized_url=url,
        source_type="generic_html",
        platform=host or "unknown",
        detection_reason="defaulted to generic HTML page",
        metadata=metadata,
    )


def build_dev_to_api_url(tag: str) -> str:
    return f"{DEVTO_API_ROOT}?tag={tag}"


def _looks_like_rss(parsed, normalized_path: str, query: dict[str, list[str]]) -> bool:
    if parsed.scheme == "file":
        return normalized_path.lower().endswith((".xml", ".rss"))

    path_lower = normalized_path.lower()
    if any(path_lower.endswith(suffix) for suffix in RSS_PATH_SUFFIXES):
        return True

    feed_param = (query.get("feed") or [""])[0].strip().lower()
    if feed_param in {"rss", "atom", "xml"}:
        return True

    format_param = (query.get("format") or [""])[0].strip().lower()
    return format_param in {"rss", "atom", "xml"}


def _normalized_host(netloc: str) -> str:
    host = (netloc or "").lower()
    return host[4:] if host.startswith("www.") else host
