"""Simple deterministic ranking for the MVP processing layer."""
from __future__ import annotations

import re
from typing import Iterable


RANKING_KEYWORDS = ("reduced", "increase", "improved", "cut", "growth")
DEFAULT_MIN_QUALITY_SCORE = 0.4
MAX_QUALITY_SCORE = 10


def rank_source_items(
    items: list[dict],
    *,
    keywords: Iterable[str] | None = None,
    excluded_keywords: Iterable[str] | None = None,
    top_n: int = 3,
    min_quality_score: float = DEFAULT_MIN_QUALITY_SCORE,
) -> tuple[list[dict], list[dict]]:
    """Score items deterministically and return selected items plus ranking metadata."""
    normalized_keywords = [term for term in (_normalize_term(k) for k in (keywords or [])) if term]
    normalized_excluded = [
        term for term in (_normalize_term(k) for k in (excluded_keywords or [])) if term
    ]

    scored_items: list[tuple[int, float, int, dict, list[str]]] = []
    for index, item in enumerate(items):
        raw_score, quality_score, quality_reasons = _score_item(
            item,
            normalized_keywords,
            normalized_excluded,
        )
        scored_items.append((raw_score, quality_score, index, item, quality_reasons))

    scored_items.sort(key=lambda entry: (-entry[0], -entry[1], entry[2]))

    ranking_scores = [
        {
            "article_id": item.get("id") or item.get("article_id"),
            "title": item.get("title", ""),
            "url": item.get("url", ""),
            "source_name": item.get("source_name") or item.get("source") or "",
            "score": raw_score,
            "quality_score": quality_score,
            "quality_reasons": quality_reasons,
        }
        for raw_score, quality_score, _, item, quality_reasons in scored_items
    ]

    filtered_items = [
        item
        for raw_score, quality_score, _, item, _quality_reasons in scored_items
        if quality_score >= min_quality_score
    ]
    selected_items = filtered_items[:top_n]
    return selected_items, ranking_scores


def _score_item(
    item: dict,
    normalized_keywords: list[str],
    normalized_excluded: list[str],
) -> tuple[int, float, list[str]]:
    title = str(item.get("title", ""))
    snippet = str(item.get("snippet", ""))
    source_name = str(item.get("source_name") or item.get("source") or "")
    title_lower = title.lower()
    snippet_lower = snippet.lower()
    text_blob = f"{title_lower} {snippet_lower}"
    source_lower = source_name.lower()

    score = 0
    reasons: list[str] = []

    keyword_hits = sum(1 for keyword in normalized_keywords if keyword in text_blob)
    score += min(keyword_hits, 2) * 2
    if keyword_hits:
        reasons.append("strong relevance to topic")
    elif normalized_keywords:
        reasons.append("weak relevance to topic")

    excluded_hits = sum(1 for keyword in normalized_excluded if keyword in text_blob)
    if excluded_hits:
        score -= min(excluded_hits, 2) * 3
        reasons.append("too narrow for selected topic")

    if re.search(r"\d|%", snippet):
        score += 2
    if any(keyword in snippet_lower for keyword in RANKING_KEYWORDS):
        score += 2
    if re.search(r"\d|%", snippet) or any(keyword in snippet_lower for keyword in RANKING_KEYWORDS):
        reasons.append("good technical/practical article")

    if len(snippet) > 120:
        score += 1
    else:
        reasons.append("insufficient evidence/detail")

    if "research" in source_lower or "report" in source_lower:
        score += 1
        reasons.append("high novelty or strategic value")

    if _looks_promotional(title_lower, snippet_lower):
        reasons.append("promotional/product announcement")
    if _looks_opinionated(title_lower, snippet_lower):
        reasons.append("mostly personal opinion")
    if score <= 2:
        reasons.append("low practical value")
    if score <= 1:
        reasons.append("low novelty")

    normalized_score = round(max(score, 0) / MAX_QUALITY_SCORE, 2)
    return score, min(1.0, normalized_score), _unique_reasons(reasons)


def _looks_promotional(title_lower: str, snippet_lower: str) -> bool:
    promo_terms = (
        "launch",
        "released",
        "release",
        "introducing",
        "update",
        "version",
        "v7",
        "announcement",
    )
    text_blob = f"{title_lower} {snippet_lower}"
    return any(term in text_blob for term in promo_terms)


def _looks_opinionated(title_lower: str, snippet_lower: str) -> bool:
    opinion_terms = (
        "i read",
        "i tried",
        "my take",
        "my thoughts",
        "lessons",
        "opinion",
        "what i learned",
    )
    text_blob = f"{title_lower} {snippet_lower}"
    return any(term in text_blob for term in opinion_terms)


def _unique_reasons(reasons: list[str]) -> list[str]:
    unique: list[str] = []
    for reason in reasons:
        if reason and reason not in unique:
            unique.append(reason)
    return unique
def _normalize_term(value: str) -> str:
    return " ".join(str(value).strip().lower().split())
