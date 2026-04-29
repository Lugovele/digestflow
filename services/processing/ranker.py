"""Simple deterministic ranking for the early MVP pipeline."""
from __future__ import annotations

import re


RANKING_KEYWORDS = ("reduced", "increase", "improved", "cut", "growth")


def rank_source_items(items: list[dict], top_n: int = 3) -> tuple[list[dict], list[dict]]:
    """Score items deterministically and return sorted items plus score metadata."""
    scored_items: list[tuple[int, int, dict]] = []
    ranking_scores: list[dict] = []

    for index, item in enumerate(items):
        score = _score_item(item)
        scored_items.append((score, index, item))
        ranking_scores.append({"url": item.get("url", ""), "score": score})

    scored_items.sort(key=lambda entry: (-entry[0], entry[1]))
    selected_items = [item for _, _, item in scored_items[:top_n]]

    ranking_scores.sort(key=lambda entry: (-entry["score"], entry["url"]))
    return selected_items, ranking_scores


def _score_item(item: dict) -> int:
    snippet = str(item.get("snippet", ""))
    source_name = str(item.get("source_name") or item.get("source") or "")
    snippet_lower = snippet.lower()
    source_lower = source_name.lower()

    score = 0
    if re.search(r"\d|%", snippet):
        score += 2
    if any(keyword in snippet_lower for keyword in RANKING_KEYWORDS):
        score += 2
    if len(snippet) > 120:
        score += 1
    if "research" in source_lower or "report" in source_lower:
        score += 1

    return score
