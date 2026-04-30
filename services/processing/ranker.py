"""Simple deterministic ranking for the MVP processing layer."""
from __future__ import annotations

import re
from typing import Iterable


RANKING_KEYWORDS = ("reduced", "increase", "improved", "cut", "growth")
MAX_QUALITY_SCORE = 10


def rank_source_items(
    items: list[dict],
    *,
    keywords: Iterable[str] | None = None,
    excluded_keywords: Iterable[str] | None = None,
    top_n: int = 3,
    min_quality_score: float = 0.0,
) -> tuple[list[dict], list[dict]]:
    """Score items deterministically and return selected items plus ranking metadata."""
    normalized_keywords = [term for term in (_normalize_term(k) for k in (keywords or [])) if term]
    normalized_excluded = [
        term for term in (_normalize_term(k) for k in (excluded_keywords or [])) if term
    ]

    scored_items: list[tuple[int, float, int, dict]] = []
    for index, item in enumerate(items):
        raw_score, quality_score = _score_item(item, normalized_keywords, normalized_excluded)
        scored_items.append((raw_score, quality_score, index, item))

    scored_items.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))

    ranking_scores = [
        {
            "url": item.get("url", ""),
            "score": raw_score,
            "quality_score": quality_score,
        }
        for raw_score, quality_score, _, item in scored_items
    ]

    filtered_items = [
        item
        for raw_score, quality_score, _, item in scored_items
        if quality_score >= min_quality_score
    ]
    if not filtered_items:
        filtered_items = [item for _, _, _, item in scored_items]

    selected_items = filtered_items[:top_n]
    return selected_items, ranking_scores


def _score_item(
    item: dict,
    normalized_keywords: list[str],
    normalized_excluded: list[str],
) -> tuple[int, float]:
    title = str(item.get("title", ""))
    snippet = str(item.get("snippet", ""))
    source_name = str(item.get("source_name") or item.get("source") or "")
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    text_blob = f"{title_lower} {snippet_lower}"
    source_lower = source_name.lower()

    score = 0

    keyword_hits = sum(1 for keyword in normalized_keywords if keyword in text_blob)
    score += min(keyword_hits, 2) * 2

    excluded_hits = sum(1 for keyword in normalized_excluded if keyword in text_blob)
    if excluded_hits:
        score -= min(excluded_hits, 2) * 3

    if re.search(r"\d|%", snippet):
        score += 2
    if any(keyword in snippet_lower for keyword in RANKING_KEYWORDS):
        score += 2
    if len(snippet) > 120:
        score += 1
    if "research" in source_lower or "report" in source_lower:
        score += 1

    normalized_score = round(max(score, 0) / MAX_QUALITY_SCORE, 2)
    return score, min(1.0, normalized_score)


def _normalize_term(value: str) -> str:
    return " ".join(str(value).strip().lower().split())
