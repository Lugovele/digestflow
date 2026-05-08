"""Deterministic preprocessing before the first AI calls."""
from __future__ import annotations

from bs4 import BeautifulSoup


MIN_CONTENT_LENGTH = 200
MIN_RICH_SUMMARY_LENGTH = 120
CONTENT_PREVIEW_LENGTH = 200


def clean_source_items(raw_items: list[dict] | object) -> list[dict]:
    """Normalize source items, convert HTML to text, and drop weak entries."""
    cleaned, _ = clean_source_items_with_diagnostics(raw_items)
    return cleaned


def clean_source_items_with_diagnostics(raw_items: list[dict] | object) -> tuple[list[dict], list[dict]]:
    """Normalize items and return deterministic rejection diagnostics."""
    cleaned = []
    rejections = []
    for item in list(raw_items):
        if not isinstance(item, dict):
            rejections.append(_build_rejection({}, "invalid article structure"))
            continue
        title = _clean_text(str(item.get("title", "")))
        url = str(item.get("url", "")).strip()
        source_name = _clean_text(str(item.get("source_name") or item.get("source") or "unknown"))
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        raw_content = (
            item.get("content")
            or item.get("body")
            or item.get("text")
            or item.get("snippet")
            or ""
        )
        normalized_content = extract_text(str(raw_content))
        content_tier, final_content_source = _classify_content_tier(
            item=item,
            metadata=metadata,
            normalized_content=normalized_content,
        )
        metadata = {
            **metadata,
            "content_tier": content_tier,
            "final_content_source": final_content_source,
            "content_length": len(normalized_content),
        }
        if not title:
            rejections.append(
                _build_rejection(
                    {**item, "metadata": metadata},
                    "missing title",
                    source_name=source_name,
                    extracted_content=normalized_content,
                )
            )
            continue
        if not url:
            rejections.append(
                _build_rejection(
                    {**item, "metadata": metadata},
                    "missing url",
                    source_name=source_name,
                    extracted_content=normalized_content,
                )
            )
            continue

        source_value = item.get("source_name") or item.get("source") or "unknown"
        if content_tier == "missing_content":
            rejections.append(
                _build_rejection(
                    {**item, "metadata": metadata},
                    "missing extracted content",
                    source_name=source_name,
                    extracted_content=normalized_content,
                )
            )
            continue
        if content_tier == "weak_snippet":
            rejections.append(
                _build_rejection(
                    {**item, "metadata": metadata},
                    "content too short",
                    source_name=source_name,
                    extracted_content=normalized_content,
                )
            )
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
                "metadata": metadata,
            }
        )
    return cleaned, rejections


def extract_text(raw_html: str) -> str:
    """Convert HTML-heavy RSS content into readable plain text."""
    if not raw_html:
        return ""

    soup = BeautifulSoup(raw_html, "html.parser")
    return _clean_text(soup.get_text(separator=" ", strip=True))


def _clean_text(value: str) -> str:
    return " ".join(value.split())


def _classify_content_tier(
    *,
    item: dict,
    metadata: dict,
    normalized_content: str,
) -> tuple[str, str]:
    content_length = len(normalized_content)
    final_content_source = str(metadata.get("final_content_source") or "").strip()
    extraction_method = str(metadata.get("extraction_method") or "").strip()
    normalized_snippet = extract_text(str(item.get("snippet") or ""))
    raw_primary_content = str(item.get("content") or item.get("body") or item.get("text") or "").strip()

    if not final_content_source:
        if extraction_method == "rss_summary_fallback":
            final_content_source = "rss_summary"
        elif extraction_method and extraction_method not in {"empty_html", "no_candidate_text"}:
            final_content_source = "html_article_body"
        elif raw_primary_content and normalized_content != normalized_snippet and content_length >= MIN_CONTENT_LENGTH:
            final_content_source = "full_content"
        elif str(item.get("snippet") or "").strip() and not str(item.get("content") or item.get("body") or item.get("text") or "").strip():
            final_content_source = "rss_summary"
        else:
            final_content_source = "direct_content"

    if content_length <= 0:
        return "missing_content", final_content_source

    if final_content_source in {"html_article_body", "full_article_api", "full_content"}:
        if content_length >= MIN_CONTENT_LENGTH:
            return "full_article", final_content_source
        return "weak_snippet", final_content_source

    if final_content_source in {"rss_summary", "metadata_summary", "direct_content"}:
        if content_length >= MIN_RICH_SUMMARY_LENGTH:
            return "rich_summary", final_content_source
        return "weak_snippet", final_content_source

    if content_length >= MIN_CONTENT_LENGTH:
        return "full_article", final_content_source
    if content_length >= MIN_RICH_SUMMARY_LENGTH:
        return "rich_summary", final_content_source
    return "weak_snippet", final_content_source


def _build_rejection(
    item: dict,
    reason: str,
    source_name: str | None = None,
    extracted_content: str = "",
) -> dict:
    normalized_preview = _clean_text(extracted_content)[:CONTENT_PREVIEW_LENGTH]
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    return {
        "title": _clean_text(str(item.get("title", ""))) or "-",
        "url": str(item.get("url", "")).strip() or "-",
        "source_name": source_name or _clean_text(str(item.get("source_name") or item.get("source") or "unknown")),
        "reason": reason or "cleaning rule failed",
        "content_tier": metadata.get("content_tier"),
        "final_content_source": metadata.get("final_content_source"),
        "content_length": len(extracted_content or ""),
        "content_preview": normalized_preview,
        "extraction_method": metadata.get("extraction_method"),
        "extraction_warning": metadata.get("extraction_warning"),
        "extraction_candidates": metadata.get("extraction_candidates") if isinstance(metadata.get("extraction_candidates"), list) else [],
    }
