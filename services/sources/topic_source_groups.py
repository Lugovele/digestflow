"""Helpers for separating manual, pinned, and temporary topic sources."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable, Sequence

from apps.topics.models import TopicSource, TopicSourceOrigin
from services.sources.detector import classify_source_url


@dataclass(frozen=True)
class TopicSourceGroups:
    manual_saved_sources: tuple[TopicSource, ...]
    pinned_research_sources: tuple[TopicSource, ...]
    new_research_sources: tuple[TopicSource, ...]


def is_manual_saved_source(source: TopicSource) -> bool:
    return str(source.origin or "") == TopicSourceOrigin.MANUAL


def is_pinned_research_source(source: TopicSource) -> bool:
    return str(source.origin or "") == TopicSourceOrigin.DISCOVERED and bool(getattr(source, "is_pinned", False))


def is_new_research_source(source: TopicSource) -> bool:
    return str(source.origin or "") == TopicSourceOrigin.DISCOVERED and not bool(getattr(source, "is_pinned", False))


def split_topic_sources(sources: Iterable[TopicSource]) -> TopicSourceGroups:
    manual_saved_sources: list[TopicSource] = []
    pinned_research_sources: list[TopicSource] = []
    new_research_sources: list[TopicSource] = []

    for source in sources:
        if is_manual_saved_source(source):
            manual_saved_sources.append(source)
        elif is_pinned_research_source(source):
            pinned_research_sources.append(source)
        elif is_new_research_source(source):
            new_research_sources.append(source)

    return TopicSourceGroups(
        manual_saved_sources=tuple(manual_saved_sources),
        pinned_research_sources=tuple(pinned_research_sources),
        new_research_sources=tuple(new_research_sources),
    )


def filter_new_source_candidates(
    candidate_records: Sequence[Any],
    existing_sources: Iterable[TopicSource],
) -> list[Any]:
    excluded_normalized_urls = {
        str(source.normalized_url or "").strip()
        for source in existing_sources
        if (is_manual_saved_source(source) or is_pinned_research_source(source))
        and str(source.normalized_url or "").strip()
    }

    filtered_candidates: list[Any] = []
    for candidate in candidate_records:
        source_url = _get_candidate_url(candidate)
        if not source_url:
            continue
        normalized_url = _get_candidate_normalized_url(candidate)
        if not normalized_url:
            normalized_url = classify_source_url(source_url).normalized_url
        if normalized_url in excluded_normalized_urls:
            continue
        filtered_candidates.append(candidate)

    return filtered_candidates


def _get_candidate_url(candidate: Any) -> str:
    if hasattr(candidate, "url"):
        return str(getattr(candidate, "url") or "").strip()
    if isinstance(candidate, Mapping):
        return str(candidate.get("url") or "").strip()
    try:
        return str(candidate["url"] or "").strip()
    except (TypeError, KeyError, IndexError):
        return ""


def _get_candidate_normalized_url(candidate: Any) -> str:
    if hasattr(candidate, "normalized_url"):
        return str(getattr(candidate, "normalized_url") or "").strip()
    if isinstance(candidate, Mapping):
        return str(candidate.get("normalized_url") or "").strip()
    try:
        return str(candidate["normalized_url"] or "").strip()
    except (TypeError, KeyError, IndexError):
        return ""
