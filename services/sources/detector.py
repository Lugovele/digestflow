"""Deterministic source URL classification for ingestion strategy selection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
import re


DEVTO_HOST = "dev.to"
DEVTO_API_ROOT = "https://dev.to/api/articles"
DEVTO_FEED_ROOT = "https://dev.to/feed"
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
TRACKING_QUERY_PREFIXES = ("utm_",)
TRACKING_QUERY_NAMES = {
    "fbclid",
    "gclid",
    "mc_cid",
    "mc_eid",
    "ref",
    "ref_src",
    "source",
    "src",
    "igshid",
}


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
    canonical_url = normalize_source_input_url(url)
    parsed = urlparse(canonical_url)
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
                original_url=canonical_url,
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
                original_url=canonical_url,
                normalized_url=build_dev_to_api_url(tag) if tag else canonical_url,
                source_type="devto_tag",
                platform="dev.to",
                detection_reason="matched dev.to API articles endpoint",
                metadata=metadata,
            )

        if len(path_parts) == 1 and path_parts[0] not in {"api", "t"}:
            author = path_parts[0].strip()
            metadata["author"] = author
            return NormalizedSource(
                original_url=canonical_url,
                normalized_url=build_dev_to_author_feed_url(author),
                source_type="devto_author",
                platform="dev.to",
                detection_reason="matched dev.to author profile",
                metadata=metadata,
            )

        if len(path_parts) >= 2 and path_parts[0] not in {"api", "t"}:
            metadata["author"] = path_parts[0]
            metadata["slug"] = path_parts[1]
            return NormalizedSource(
                original_url=canonical_url,
                normalized_url=canonical_url,
                source_type="devto_article",
                platform="dev.to",
                detection_reason="matched dev.to article path",
                metadata=metadata,
            )

    if _looks_like_rss(parsed, normalized_path, query):
        return NormalizedSource(
            original_url=canonical_url,
            normalized_url=canonical_url,
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
            original_url=canonical_url,
            normalized_url=canonical_url,
            source_type="blog_index",
            platform=host or "unknown",
            detection_reason="matched blog or news index path",
            metadata=metadata,
        )

    if normalized_path == "/":
        return NormalizedSource(
            original_url=canonical_url,
            normalized_url=canonical_url,
            source_type="publication",
            platform=host or "unknown",
            detection_reason="matched publication homepage pattern",
            metadata=metadata,
        )

    return NormalizedSource(
        original_url=canonical_url,
        normalized_url=canonical_url,
        source_type="generic_html",
        platform=host or "unknown",
        detection_reason="defaulted to generic HTML page",
        metadata=metadata,
    )


def build_dev_to_api_url(tag: str) -> str:
    return f"{DEVTO_API_ROOT}?tag={tag}"


def build_dev_to_author_feed_url(author: str) -> str:
    return f"{DEVTO_FEED_ROOT}/{author}"


def normalize_source_input_url(url: str) -> str:
    raw_url = str(url or "").strip()
    if not raw_url:
        return raw_url

    if raw_url.startswith("file://"):
        parsed = urlparse(raw_url)
        path = _normalize_path(parsed.path or "/")
        return urlunparse(("file", parsed.netloc, path, "", "", ""))

    if re.match(r"^[a-zA-Z]:[\\/]", raw_url):
        return raw_url

    if "://" not in raw_url and not raw_url.startswith("//"):
        return raw_url

    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "https").lower()
    host = _normalized_host(parsed.netloc)
    path = _normalize_path(parsed.path or "/")
    query = _normalize_query_for_source(host, path, parse_qs(parsed.query))
    normalized = urlunparse((scheme, host, path, "", query, ""))
    return normalized.rstrip("/") if path != "/" and normalized.endswith("/") else normalized


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


def _normalize_path(path: str) -> str:
    collapsed = re.sub(r"/{2,}", "/", path or "/")
    if not collapsed.startswith("/"):
        collapsed = f"/{collapsed}"
    return collapsed.rstrip("/") or "/"


def _normalize_query_for_source(host: str, path: str, query: dict[str, list[str]]) -> str:
    if host == DEVTO_HOST:
        path_parts = [part for part in path.strip("/").split("/") if part]
        if path_parts[:2] == ["api", "articles"]:
            tag = (query.get("tag") or [""])[0].strip()
            return f"tag={tag}" if tag else ""
        return ""

    feed_param = (query.get("feed") or [""])[0].strip().lower()
    format_param = (query.get("format") or [""])[0].strip().lower()
    if feed_param in {"rss", "atom", "xml"}:
        return f"feed={feed_param}"
    if format_param in {"rss", "atom", "xml"}:
        return f"format={format_param}"

    normalized_query: list[tuple[str, str]] = []
    for key in sorted(query):
        normalized_key = str(key or "").strip()
        if not normalized_key:
            continue
        lowered_key = normalized_key.lower()
        if lowered_key.startswith(TRACKING_QUERY_PREFIXES) or lowered_key in TRACKING_QUERY_NAMES:
            continue
        values = query.get(key) or [""]
        for raw_value in values:
            cleaned_value = str(raw_value or "").strip()
            if not cleaned_value:
                continue
            normalized_query.append((normalized_key, cleaned_value))

    return urlencode(normalized_query, doseq=True)
    return ""
