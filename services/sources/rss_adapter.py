"""Source ingestion helpers for RSS and dev.to sources."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urlparse, urlunparse
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, HTTPSHandler, ProxyHandler, Request, build_opener

from bs4 import BeautifulSoup
from bs4.element import Tag

from services.sources.detector import (
    DEVTO_API_ROOT,
    NormalizedSource,
    build_dev_to_api_url,
    classify_source_url,
    detect_source_type,
)


logger = logging.getLogger(__name__)
WEAK_EXTRACTION_LENGTH = 200
HTML_CONTENT_PREVIEW_LENGTH = 200


def fetch_rss_articles(source_url: str, limit: int = 10) -> list[dict]:
    """Fetch source items from RSS or supported dev.to sources."""
    normalized_source = normalize_source_url(source_url)
    logger.info(
        "Detected source type for %s -> %s (%s)",
        source_url,
        normalized_source.source_type,
        normalized_source.detection_reason,
    )

    if normalized_source.source_type == "devto_tag":
        return _fetch_dev_to_articles_from_list_source(normalized_source, limit=limit)

    if normalized_source.source_type == "devto_article":
        article = _fetch_dev_to_single_article(normalized_source)
        return [article] if article else []

    if normalized_source.source_type in {"generic_html", "blog_index", "publication"}:
        article = fetch_generic_web_article(normalized_source.normalized_url)
        return [article] if article else []

    return _fetch_rss_feed_articles(normalized_source, limit=limit)


def get_rss_debug_snapshot(source_url: str, sample_size: int = 5) -> dict[str, Any]:
    """Return a debug snapshot for RSS parsing and supported source normalization."""
    normalized_source = normalize_source_url(source_url)

    if normalized_source.source_type in {"devto_tag", "devto_article"}:
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
            "detection_reason": normalized_source.detection_reason,
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
            "detection_reason": normalized_source.detection_reason,
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
        "detection_reason": normalized_source.detection_reason,
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
    return classify_source_url(url)


def fetch_dev_to_article_list(api_url: str) -> list[dict]:
    payload = _fetch_json(api_url)
    return payload if isinstance(payload, list) else []


def fetch_dev_to_article_content(article_id_or_url: int | str) -> dict[str, Any] | None:
    if isinstance(article_id_or_url, int) or str(article_id_or_url).isdigit():
        article_id = int(article_id_or_url)
        detail_url = f"{DEVTO_API_ROOT}/{article_id}"
        payload = _fetch_json(detail_url)
        if isinstance(payload, dict):
            final_content_source = "metadata_summary"
            if str(payload.get("body_markdown") or "").strip() or str(payload.get("body_html") or "").strip():
                final_content_source = "full_article_api"
            content_diagnostics = _extract_dev_to_content_diagnostics(payload)
            return {
                "title": str(payload.get("title", "")).strip(),
                "url": str(payload.get("url", "")).strip(),
                "description": str(payload.get("description") or "").strip() or None,
                "content": content_diagnostics["content"],
                "published_at": payload.get("published_at"),
                "metadata": {
                    "devto_id": payload.get("id", article_id),
                    "tag_list": payload.get("tag_list") or [],
                    "positive_reactions_count": payload.get("positive_reactions_count"),
                    "comments_count": payload.get("comments_count"),
                    "public_reactions_count": payload.get("public_reactions_count"),
                    "reading_time_minutes": payload.get("reading_time_minutes"),
                    "cover_image": payload.get("cover_image"),
                    "final_content_source": final_content_source,
                    "headings": content_diagnostics["headings"],
                    "raw_html_heading_count": content_diagnostics["raw_html_heading_count"],
                    "extracted_heading_count": content_diagnostics["extracted_heading_count"],
                    "heading_extraction_strategy": content_diagnostics["heading_extraction_strategy"],
                    "sample_detected_headings": content_diagnostics["sample_detected_headings"],
                },
            }
        return None

    article_url = str(article_id_or_url).strip()
    if not article_url:
        return None

    html = _fetch_url_text(article_url)
    if not html:
        return None

    extraction = _extract_html_content_diagnostics(html)

    return {
        "title": _extract_html_title(html),
        "url": article_url,
        "description": None,
        "content": extraction["content"],
        "published_at": None,
        "metadata": {
            "extraction_method": extraction["extraction_method"],
            "extracted_content_length": extraction["extracted_content_length"],
            "extraction_warning": extraction["extraction_warning"],
            "extraction_candidates": extraction["extraction_candidates"],
            "headings": extraction["headings"],
            "raw_html_heading_count": extraction["raw_html_heading_count"],
            "extracted_heading_count": extraction["extracted_heading_count"],
            "heading_extraction_strategy": extraction["heading_extraction_strategy"],
            "sample_detected_headings": extraction["sample_detected_headings"],
        },
    }


def fetch_generic_web_article(source_url: str) -> dict[str, Any] | None:
    return inspect_generic_web_article(source_url).get("article")


def inspect_generic_web_article(source_url: str) -> dict[str, Any]:
    article_url = str(source_url or "").strip()
    if not article_url:
        return {
            "article": None,
            "diagnostics": {
                "normalized_url": "",
                "source_type": "",
                "fetch_status": None,
                "fetch_failure_reason": "empty source url",
                "content_type": "",
                "final_fetch_url": "",
                "title": "",
                "extraction_strategy": "empty_source_url",
                "usable_text_length": 0,
                "rejection_reason": "empty source url",
            },
        }

    normalized_source = normalize_source_url(article_url)
    fetch_result = _fetch_url_response(
        normalized_source.normalized_url,
        accept_header="text/html, application/xhtml+xml, */*",
    )
    html = fetch_result["content"].decode("utf-8", errors="ignore") if fetch_result["content"] else ""
    title = _extract_html_title(html) if html else ""
    diagnostics: dict[str, Any] = {
        "normalized_url": normalized_source.normalized_url,
        "source_type": normalized_source.source_type,
        "fetch_status": fetch_result["status"],
        "fetch_failure_reason": fetch_result["fetch_failure_reason"],
        "content_type": fetch_result["content_type"],
        "final_fetch_url": fetch_result["final_url"],
        "title": title,
        "extraction_strategy": "fetch_failed",
        "usable_text_length": 0,
        "rejection_reason": "",
    }

    blocked_reason = _detect_blocked_article_fetch(
        fetch_result["status"],
        fetch_result["content_type"],
        html,
    )
    if blocked_reason:
        reader_fallback = _fetch_reader_fallback_article(normalized_source.original_url)
        diagnostics["blocked_fetch_reason"] = blocked_reason
        diagnostics["reader_fallback_attempted"] = True
        if reader_fallback is not None:
            content = str(reader_fallback.get("content") or "").strip()
            title = str(reader_fallback.get("title") or "").strip()
            diagnostics.update(
                {
                    "title": title,
                    "extraction_strategy": "reader_markdown_fallback",
                    "usable_text_length": len(content),
                    "reader_fallback_used": True,
                    "rejection_reason": "",
                }
            )
            return {
                "article": {
                    "title": title or _get_fallback_source_name(normalized_source.normalized_url),
                    "url": normalized_source.original_url,
                    "source_url": normalized_source.original_url,
                    "source_api_url": None,
                    "source_name": title or _get_fallback_source_name(normalized_source.normalized_url),
                    "source_type": "web_article",
                    "platform": normalized_source.platform,
                    "content": content,
                    "snippet": _build_snippet(content),
                    "description": None,
                    "published_at": None,
                    "metadata": {
                        "platform": normalized_source.platform,
                        "source_type": "web_article",
                        "content_unavailable": False,
                        "detection_reason": normalized_source.detection_reason,
                        "extraction_method": "reader_markdown_fallback",
                        "extracted_content_length": len(content),
                        "extraction_warning": "primary fetch was blocked; reader fallback used",
                        "extraction_candidates": [],
                        "headings": [],
                        "raw_html_heading_count": 0,
                        "extracted_heading_count": 0,
                        "heading_extraction_strategy": "reader_markdown_fallback",
                        "sample_detected_headings": [],
                    },
                },
                "diagnostics": diagnostics,
            }

    if not html:
        diagnostics["rejection_reason"] = fetch_result["fetch_failure_reason"] or "html fetch failed"
        return {"article": None, "diagnostics": diagnostics}

    extraction = _extract_html_content_diagnostics(html)
    content = str(extraction.get("content") or "").strip()
    diagnostics.update(
        {
            "extraction_strategy": extraction["extraction_method"],
            "usable_text_length": int(extraction.get("extracted_content_length") or 0),
            "extraction_warning": extraction.get("extraction_warning"),
            "content_preview": extraction.get("content_preview") or "",
        }
    )

    if not content:
        diagnostics["rejection_reason"] = "no readable article text was extracted"
        return {"article": None, "diagnostics": diagnostics}

    if not _looks_like_useful_generic_article(title, extraction):
        diagnostics["rejection_reason"] = _build_generic_article_rejection_reason(title, extraction)
        return {"article": None, "diagnostics": diagnostics}

    source_name = title or _get_fallback_source_name(normalized_source.normalized_url)
    article = {
        "title": title or source_name,
        "url": normalized_source.original_url,
        "source_url": normalized_source.original_url,
        "source_api_url": None,
        "source_name": source_name,
        "source_type": "web_article",
        "platform": normalized_source.platform,
        "content": content,
        "snippet": _build_snippet(content),
        "description": None,
        "published_at": None,
        "metadata": {
            "platform": normalized_source.platform,
            "source_type": "web_article",
            "content_unavailable": False,
            "detection_reason": normalized_source.detection_reason,
            "extraction_method": extraction["extraction_method"],
            "extracted_content_length": extraction["extracted_content_length"],
            "extraction_warning": extraction["extraction_warning"],
            "extraction_candidates": extraction["extraction_candidates"],
            "headings": extraction["headings"],
            "raw_html_heading_count": extraction["raw_html_heading_count"],
            "extracted_heading_count": extraction["extracted_heading_count"],
            "heading_extraction_strategy": extraction["heading_extraction_strategy"],
            "sample_detected_headings": extraction["sample_detected_headings"],
        },
    }
    diagnostics["rejection_reason"] = ""
    return {"article": article, "diagnostics": diagnostics}


def _looks_like_useful_generic_article(title: str, extraction: dict[str, Any]) -> bool:
    content = _clean_text(str(extraction.get("content") or ""))
    content_length = len(content)
    if not content or not str(title or "").strip():
        return False

    extraction_method = str(extraction.get("extraction_method") or "")
    heading_count = int(extraction.get("extracted_heading_count") or 0)
    paragraph_like_count = content.count(". ") + content.count("! ") + content.count("? ")

    if content_length >= WEAK_EXTRACTION_LENGTH:
        return True

    if extraction_method != "fallback_text" and content_length >= 120:
        return True

    if extraction_method == "fallback_text" and heading_count >= 2 and content_length >= 120:
        return True

    if extraction_method == "fallback_text" and heading_count >= 1 and content_length >= 180 and paragraph_like_count >= 3:
        return True

    if extraction_method != "fallback_text" and content_length >= 140 and paragraph_like_count >= 2:
        return True

    return False


def _build_generic_article_rejection_reason(title: str, extraction: dict[str, Any]) -> str:
    content = _clean_text(str(extraction.get("content") or ""))
    content_length = len(content)
    extraction_method = str(extraction.get("extraction_method") or "")
    heading_count = int(extraction.get("extracted_heading_count") or 0)
    paragraph_like_count = content.count(". ") + content.count("! ") + content.count("? ")

    if not str(title or "").strip():
        return "page title was missing"
    if not content:
        return "no readable article text was extracted"
    if content_length < 120:
        return f"usable text was too short ({content_length} chars)"
    if extraction_method == "fallback_text" and heading_count == 0 and paragraph_like_count < 2:
        return "page content looked too weak or unstructured"
    return "page content did not look article-like enough"


def _detect_blocked_article_fetch(status: int | None, content_type: str, html: str) -> str:
    normalized_html = str(html or "").lower()
    normalized_type = str(content_type or "").lower()
    if status not in {401, 403}:
        return ""
    if "text/html" not in normalized_type:
        return ""
    if "just a moment" in normalized_html:
        return "bot protection interstitial"
    if "cf-browser-verification" in normalized_html or "cf-chl-" in normalized_html:
        return "cloudflare challenge"
    return "blocked html response"


def _fetch_reader_fallback_article(source_url: str) -> dict[str, str] | None:
    for candidate_url in _build_reader_fallback_source_urls(source_url):
        reader_url = f"https://r.jina.ai/http://{candidate_url}"
        result = _fetch_url_response(reader_url, accept_header="text/plain, text/markdown, */*")
        if int(result.get("status") or 0) != 200 or not result.get("content"):
            continue

        payload = result["content"].decode("utf-8", errors="ignore")
        title, content = _parse_reader_markdown_payload(payload)
        if not title or not content:
            continue

        if len(content) < 120:
            continue

        if title.strip().lower() == "just a moment...":
            continue

        return {"title": title, "content": content}
    return None


def _build_reader_fallback_source_urls(source_url: str) -> list[str]:
    raw_source = str(source_url or "").strip()
    if not raw_source:
        return []

    parsed = urlparse(raw_source)
    candidates = [raw_source]
    if parsed.scheme in {"http", "https"} and parsed.netloc and not parsed.netloc.lower().startswith("www."):
        www_url = urlunparse(
            (
                parsed.scheme,
                f"www.{parsed.netloc}",
                parsed.path,
                parsed.params,
                parsed.query,
                parsed.fragment,
            )
        )
        candidates.append(www_url)
    return candidates


def _parse_reader_markdown_payload(payload: str) -> tuple[str, str]:
    raw_text = str(payload or "").strip()
    if not raw_text:
        return "", ""

    title = ""
    title_match = re.search(r"^Title:\s*(.+)$", raw_text, re.MULTILINE)
    if title_match:
        title = _clean_text(title_match.group(1))

    content_match = re.search(r"Markdown Content:\s*(.*)$", raw_text, re.DOTALL)
    if not content_match:
        return title, ""

    markdown = content_match.group(1).strip()
    content = _clean_reader_markdown(markdown)
    return title, content


def _clean_reader_markdown(markdown: str) -> str:
    text = str(markdown or "")
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[*-]\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    lines = []
    for line in text.splitlines():
        normalized = _clean_text(line)
        if not normalized:
            continue
        if normalized.startswith("URL Source:"):
            continue
        if normalized.startswith("Published Time:"):
            continue
        lines.append(normalized)
    return _clean_text(" ".join(lines))


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
            "detection_reason": normalized_source.detection_reason,
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
            "detection_reason": normalized_source.detection_reason,
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

        extraction = _extract_rss_article_content(url, snippet)
        content = extraction["content"]

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
                "content": content,
                "description": None,
                "metadata": {
                    "platform": normalized_source.platform,
                    "source_type": normalized_source.source_type,
                    "content_unavailable": not bool(content),
                    "published_at": published_at.isoformat() if published_at else None,
                    "detection_reason": normalized_source.detection_reason,
                    "extraction_method": extraction["extraction_method"],
                    "extracted_content_length": extraction["extracted_content_length"],
                    "extraction_warning": extraction["extraction_warning"],
                    "extraction_candidates": extraction["extraction_candidates"],
                    "final_content_source": extraction["final_content_source"],
                    "rss_summary_length": extraction["rss_summary_length"],
                    "html_extracted_content_length": extraction["html_extracted_content_length"],
                },
                "published_at": published_at.isoformat() if published_at else None,
            }
        )

        if len(articles) >= limit:
            break

    return articles


def _extract_rss_article_content(article_url: str, rss_summary: str) -> dict[str, Any]:
    normalized_summary = _clean_text(rss_summary)
    summary_length = len(normalized_summary)
    html = _fetch_url_text(article_url)

    if not html:
        logger.info(
            "RSS article extraction for %s -> summary_len=%s html_len=0 method=rss_summary_fallback final_source=rss_summary",
            article_url,
            summary_length,
        )
        return {
            "content": normalized_summary,
            "extraction_method": "rss_summary_fallback",
            "extracted_content_length": summary_length,
            "extraction_warning": "html fetch failed; RSS summary used",
            "extraction_candidates": [],
            "final_content_source": "rss_summary",
            "rss_summary_length": summary_length,
            "html_extracted_content_length": 0,
        }

    html_extraction = _extract_html_content_diagnostics(html)
    html_content = html_extraction["content"]
    html_content_length = int(html_extraction["extracted_content_length"])

    if html_content:
        logger.info(
            "RSS article extraction for %s -> summary_len=%s html_len=%s method=%s final_source=html_article_body",
            article_url,
            summary_length,
            html_content_length,
            html_extraction["extraction_method"],
        )
        return {
            "content": html_content,
            "extraction_method": str(html_extraction["extraction_method"] or "fallback_text"),
            "extracted_content_length": html_content_length,
            "extraction_warning": html_extraction["extraction_warning"],
            "extraction_candidates": html_extraction["extraction_candidates"],
            "final_content_source": "html_article_body",
            "rss_summary_length": summary_length,
            "html_extracted_content_length": html_content_length,
        }

    warning = html_extraction["extraction_warning"] or "no article container found; RSS summary used"
    if normalized_summary:
        warning = f"{warning}; RSS summary used"
    logger.info(
        "RSS article extraction for %s -> summary_len=%s html_len=%s method=rss_summary_fallback final_source=rss_summary",
        article_url,
        summary_length,
        html_content_length,
    )
    return {
        "content": normalized_summary,
        "extraction_method": "rss_summary_fallback",
        "extracted_content_length": summary_length,
        "extraction_warning": warning,
        "extraction_candidates": html_extraction["extraction_candidates"],
        "final_content_source": "rss_summary" if normalized_summary else "none",
        "rss_summary_length": summary_length,
        "html_extracted_content_length": html_content_length,
    }


def _extract_dev_to_content_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    body_markdown = str(payload.get("body_markdown") or "").strip()
    body_html = str(payload.get("body_html") or "").strip()
    raw_html_headings = _extract_html_headings(body_html) if body_html else []

    if body_markdown:
        markdown_headings = _extract_markdown_headings(body_markdown)
        return {
            "content": _clean_text(body_markdown),
            "headings": markdown_headings,
            "raw_html_heading_count": len(raw_html_headings),
            "extracted_heading_count": len(markdown_headings),
            "heading_extraction_strategy": "markdown_headings" if markdown_headings else "markdown_without_headings",
            "sample_detected_headings": markdown_headings[:5],
        }

    if body_html:
        extraction = _extract_html_content_diagnostics(body_html)
        return {
            "content": extraction["content"],
            "headings": extraction["headings"],
            "raw_html_heading_count": extraction["raw_html_heading_count"],
            "extracted_heading_count": extraction["extracted_heading_count"],
            "heading_extraction_strategy": extraction["heading_extraction_strategy"],
            "sample_detected_headings": extraction["sample_detected_headings"],
        }

    description = str(payload.get("description") or "").strip()
    return {
        "content": _clean_text(description),
        "headings": [],
        "raw_html_heading_count": 0,
        "extracted_heading_count": 0,
        "heading_extraction_strategy": "none",
        "sample_detected_headings": [],
    }


def _extract_dev_to_text(payload: dict[str, Any]) -> str:
    return str(_extract_dev_to_content_diagnostics(payload)["content"])


def _extract_html_text(html: str) -> str:
    return _extract_html_content_diagnostics(html)["content"]


def _extract_html_content_diagnostics(html: str) -> dict[str, Any]:
    if not html:
        return {
            "content": "",
            "extraction_method": "empty_html",
            "extracted_content_length": 0,
            "extraction_warning": "no html content was fetched",
            "content_preview": "",
            "extraction_candidates": [],
            "headings": [],
            "raw_html_heading_count": 0,
            "extracted_heading_count": 0,
            "heading_extraction_strategy": "none",
            "sample_detected_headings": [],
        }

    soup = BeautifulSoup(html, "html.parser")
    raw_html_headings = _extract_heading_texts(soup)
    _remove_boilerplate_nodes(soup)

    candidate_builders = [
        ("article_tag", lambda root: root.find("article")),
        ("main_article", lambda root: root.select_one("main article")),
        ("role_main_article", lambda root: root.select_one('[role="main"] article')),
        ("role_main", lambda root: root.select_one('[role="main"]')),
        ("main_tag", lambda root: root.find("main")),
    ]
    candidate_builders.extend(
        [
            (f"common_selector:{selector}", lambda root, selector=selector: root.select_one(selector))
            for selector in (
                ".article-content",
                ".post-content",
                ".entry-content",
                ".article-body",
                ".post-body",
                ".story-body",
                ".content__body",
                ".blog-post",
                ".blog-content",
                ".c-article-body",
                ".ms-rtestate-field",
                ".layout-content",
            )
        ]
    )

    candidates: list[dict[str, Any]] = []
    candidate_diagnostics: list[dict[str, Any]] = []
    for method, builder in candidate_builders:
        try:
            node = builder(soup)
        except Exception:
            node = None
        if node is None:
            candidate_diagnostics.append(
                {
                    "selector": method,
                    "found": False,
                    "text_length": 0,
                    "text_preview": "",
                    "rejection_reason": "not found",
                }
            )
            continue
        text = _extract_readable_block_text(node)
        text_length = len(text)
        text_preview = text[:HTML_CONTENT_PREVIEW_LENGTH]
        if text:
            candidates.append(
                {
                    "method": method,
                    "node": node,
                    "text": text,
                    "text_length": text_length,
                    "text_preview": text_preview,
                    "score": _score_extracted_block(text),
                }
            )
            candidate_diagnostics.append(
                {
                    "selector": method,
                    "found": True,
                    "text_length": text_length,
                    "text_preview": text_preview,
                    "rejection_reason": None,
                }
            )
        else:
            candidate_diagnostics.append(
                {
                    "selector": method,
                    "found": True,
                    "text_length": 0,
                    "text_preview": "",
                    "rejection_reason": "no readable text extracted",
                }
            )

    body_text = _extract_readable_block_text(soup.body or soup)
    if body_text:
        candidates.append(
            {
                "method": "fallback_text",
                "node": soup.body or soup,
                "text": body_text,
                "text_length": len(body_text),
                "text_preview": body_text[:HTML_CONTENT_PREVIEW_LENGTH],
                "score": _score_extracted_block(body_text) - 50,
            }
        )
        candidate_diagnostics.append(
            {
                "selector": "fallback_text",
                "found": True,
                "text_length": len(body_text),
                "text_preview": body_text[:HTML_CONTENT_PREVIEW_LENGTH],
                "rejection_reason": None,
            }
        )
    else:
        candidate_diagnostics.append(
            {
                "selector": "fallback_text",
                "found": False,
                "text_length": 0,
                "text_preview": "",
                "rejection_reason": "no readable text extracted",
            }
        )

    if not candidates:
        return {
            "content": "",
            "extraction_method": "no_candidate_text",
            "extracted_content_length": 0,
            "extraction_warning": "no readable article text was extracted",
            "content_preview": "",
            "extraction_candidates": candidate_diagnostics,
            "headings": [],
            "raw_html_heading_count": len(raw_html_headings),
            "extracted_heading_count": 0,
            "heading_extraction_strategy": "none",
            "sample_detected_headings": [],
        }

    best_candidate = max(candidates, key=lambda candidate: candidate["score"])
    content = _clean_text(best_candidate["text"])
    extracted_headings = _extract_heading_texts(best_candidate.get("node"))
    extracted_content_length = len(content)
    extraction_warning = None
    if extracted_content_length < WEAK_EXTRACTION_LENGTH:
        extraction_warning = "extracted content is very short"

    diagnostics_by_selector = {item["selector"]: item for item in candidate_diagnostics}
    for candidate in candidates:
        item = diagnostics_by_selector.get(candidate["method"])
        if item is None:
            continue
        if candidate["method"] == best_candidate["method"]:
            item["rejection_reason"] = None
            continue
        if candidate["text_length"] < WEAK_EXTRACTION_LENGTH:
            item["rejection_reason"] = "too short"
        else:
            item["rejection_reason"] = "low article-like score"

    return {
        "content": content,
        "extraction_method": best_candidate["method"],
        "extracted_content_length": extracted_content_length,
        "extraction_warning": extraction_warning,
        "content_preview": content[:HTML_CONTENT_PREVIEW_LENGTH],
        "extraction_candidates": candidate_diagnostics,
        "headings": extracted_headings,
        "raw_html_heading_count": len(raw_html_headings),
        "extracted_heading_count": len(extracted_headings),
        "heading_extraction_strategy": "html_headings" if extracted_headings else "none",
        "sample_detected_headings": extracted_headings[:5],
    }


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


def _remove_boilerplate_nodes(soup: BeautifulSoup) -> None:
    for tag_name in ("script", "style", "nav", "header", "footer", "aside", "noscript", "svg"):
        for node in soup.find_all(tag_name):
            if isinstance(node, Tag):
                node.decompose()

    for node in soup.find_all("form"):
        if isinstance(node, Tag) and _is_boilerplate_form(node):
            node.decompose()

    boilerplate_markers = (
        "cookie",
        "subscribe",
        "newsletter",
        "share",
        "social",
        "menu",
        "nav",
        "footer",
        "header",
        "related",
        "recommend",
        "promo",
        "advert",
    )
    for node in list(soup.find_all(True)):
        if not isinstance(node, Tag):
            continue
        if node.name in {"html", "body", "main", "article"}:
            continue
        attrs = getattr(node, "attrs", None)
        if not isinstance(attrs, dict):
            continue

        class_values = attrs.get("class", [])
        if isinstance(class_values, str):
            classes = class_values
        elif isinstance(class_values, (list, tuple)):
            classes = " ".join(str(value) for value in class_values if value)
        else:
            classes = ""

        node_id = str(attrs.get("id") or "")
        aria_label = str(attrs.get("aria-label") or "")
        marker_text = " ".join((classes, node_id, aria_label)).lower()
        if any(marker in marker_text for marker in boilerplate_markers):
            node.decompose()


def _is_boilerplate_form(node: Tag) -> bool:
    if node.find("article") or node.find("main"):
        return False

    content_like_selector = node.select_one(
        ".ms-rtestate-field, .layout-content, .article-content, .post-content, .entry-content, .article-body, .post-body, .story-body, .content__body, .blog-post, .blog-content, .c-article-body"
    )
    if content_like_selector is not None:
        return False

    paragraph_count = len(node.find_all("p"))
    readable_text = _clean_text(node.get_text(separator=" ", strip=True))
    if paragraph_count >= 3 and len(readable_text) >= 300:
        return False

    return True


def _extract_readable_block_text(node) -> str:
    if node is None:
        return ""

    parts: list[str] = []
    for element in node.find_all(
        ["p", "h1", "h2", "h3", "h4", "blockquote", "li"],
        recursive=True,
    ):
        text = _clean_text(element.get_text(separator=" ", strip=True))
        if text:
            parts.append(text)

    if not parts:
        return _clean_text(node.get_text(separator=" ", strip=True))

    return _clean_text(" ".join(parts))


def _extract_markdown_headings(markdown: str) -> list[str]:
    headings: list[str] = []
    for line in str(markdown).splitlines():
        match = re.match(r"^\s{0,3}#{1,6}\s+(.*\S)\s*$", line)
        if not match:
            continue
        heading = _clean_text(match.group(1).strip("# ").strip())
        if heading and heading not in headings:
            headings.append(heading)
    return headings[:12]


def _extract_html_headings(html: str) -> list[str]:
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    return _extract_heading_texts(soup)


def _extract_heading_texts(node) -> list[str]:
    if node is None:
        return []

    headings: list[str] = []
    for element in node.find_all(["h1", "h2", "h3", "h4", "h5", "h6"], recursive=True):
        text = _clean_text(element.get_text(separator=" ", strip=True))
        if text and text not in headings:
            headings.append(text)
    return headings[:12]


def _score_extracted_block(text: str) -> int:
    normalized = _clean_text(text)
    if not normalized:
        return 0
    paragraph_like_bonus = normalized.count(". ") * 5
    return len(normalized) + paragraph_like_bonus


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


def _fetch_url_response(
    source_url: str,
    accept_header: str = "application/rss+xml, application/xml, text/xml, */*",
) -> dict[str, Any]:
    local_path = _resolve_local_feed_path(source_url)
    if local_path is not None:
        try:
            return {
                "content": local_path.read_bytes(),
                "status": 200,
                "content_type": "",
                "final_url": str(local_path),
                "fetch_failure_reason": "",
            }
        except Exception as exc:
            return {
                "content": b"",
                "status": None,
                "content_type": "",
                "final_url": "",
                "fetch_failure_reason": str(exc),
            }

    request = Request(
        source_url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; DigestFlowRSS/0.1)",
            "Accept": accept_header,
        },
    )
    try:
        opener = build_opener(ProxyHandler({}), HTTPRedirectHandler(), HTTPSHandler())
        with opener.open(request, timeout=15) as response:
            headers = getattr(response, "headers", None)
            if headers is None and hasattr(response, "info"):
                try:
                    headers = response.info()
                except Exception:
                    headers = None
            content_type = ""
            if headers is not None and hasattr(headers, "get"):
                content_type = str(headers.get("Content-Type") or "")
            final_url = source_url
            if hasattr(response, "geturl"):
                try:
                    final_url = response.geturl()
                except Exception:
                    final_url = source_url
            status = getattr(response, "status", None)
            if status is None and hasattr(response, "getcode"):
                try:
                    status = response.getcode()
                except Exception:
                    status = None
            return {
                "content": response.read(),
                "status": status,
                "content_type": content_type,
                "final_url": final_url,
                "fetch_failure_reason": "",
            }
    except HTTPError as exc:
        return {
            "content": b"",
            "status": exc.code,
            "content_type": str(exc.headers.get("Content-Type") or "") if exc.headers else "",
            "final_url": exc.geturl(),
            "fetch_failure_reason": f"http {exc.code}",
        }
    except URLError as exc:
        reason = getattr(exc, "reason", exc)
        return {
            "content": b"",
            "status": None,
            "content_type": "",
            "final_url": source_url,
            "fetch_failure_reason": str(reason),
        }
    except Exception as exc:
        return {
            "content": b"",
            "status": None,
            "content_type": "",
            "final_url": source_url,
            "fetch_failure_reason": str(exc),
        }


def _fetch_url_bytes(
    source_url: str,
    accept_header: str = "application/rss+xml, application/xml, text/xml, */*",
) -> bytes | None:
    result = _fetch_url_response(source_url, accept_header=accept_header)
    return result["content"] or None


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
