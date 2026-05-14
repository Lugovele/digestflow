"""Deterministic topic-to-source discovery for the lightweight source-selection MVP."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence
from urllib.parse import urlparse

from apps.topics.models import TopicSourceMode
from services.sources.detector import classify_source_url
from services.sources.rss_adapter import fetch_dev_to_article_list, get_rss_debug_snapshot


@dataclass(frozen=True)
class SourceCandidateTemplate:
    url: str
    title: str
    description: str
    quality_estimate: str


@dataclass(frozen=True)
class CuratedSourceSeed:
    url: str
    title: str = ""
    description: str = ""
    quality_estimate: str = "curated"
    is_manual: bool = False
    default_selected: bool = True


@dataclass(frozen=True)
class TopicSourceDiscoveryRequest:
    topic: str
    focus_terms: Sequence[str] = ()
    source_mode: str = TopicSourceMode.DISCOVERY_ONLY
    manual_source_url: str = ""
    manual_source_urls: Sequence[str] = ()
    curated_sources: Sequence[CuratedSourceSeed] = ()
    limit: int = 10


TOPIC_SOURCE_TEMPLATES: list[tuple[set[str], list[SourceCandidateTemplate]]] = [
    (
        {"ai agents", "agent", "agents", "agentic", "multi-agent", "mcp"},
        [
            SourceCandidateTemplate(
                url="https://dev.to/t/ai",
                title="DEV Community / #ai",
                description="Broad AI engineering stream with regular agent, tooling, and workflow posts.",
                quality_estimate="high",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/python",
                title="DEV Community / #python",
                description="Implementation-heavy Python posts for agent tooling, automation, and orchestration.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/devops",
                title="DEV Community / #devops",
                description="Deployment and infrastructure writing for agents, MCP servers, and operational workflows.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/security",
                title="DEV Community / #security",
                description="Security and authorization topics useful for agent permissions and MCP governance.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/cloud",
                title="DEV Community / #cloud",
                description="Cloud deployment and platform posts that often intersect with agent runtime topics.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/testing",
                title="DEV Community / #testing",
                description="Testing and local-validation posts that help evaluate agent workflow reliability.",
                quality_estimate="medium",
            ),
        ],
    ),
    (
        {"python", "automation", "workflow"},
        [
            SourceCandidateTemplate(
                url="https://dev.to/t/python",
                title="DEV Community / #python",
                description="Python implementation posts covering scripts, tooling, and automation patterns.",
                quality_estimate="high",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/devops",
                title="DEV Community / #devops",
                description="Operational automation, deployment workflows, and engineering platform practices.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/testing",
                title="DEV Community / #testing",
                description="Testing-oriented posts for workflow validation and automation reliability.",
                quality_estimate="medium",
            ),
            SourceCandidateTemplate(
                url="https://dev.to/t/ai",
                title="DEV Community / #ai",
                description="AI-adjacent workflow posts, helpful when automation overlaps with LLM features.",
                quality_estimate="medium",
            ),
        ],
    ),
]

DEFAULT_SOURCE_TEMPLATES: list[SourceCandidateTemplate] = []


def discover_sources(topic: str, manual_source_url: str = "", limit: int = 10) -> list[dict[str, Any]]:
    """Return deterministic candidate sources for a topic."""
    request = TopicSourceDiscoveryRequest(
        topic=topic,
        source_mode=TopicSourceMode.HYBRID,
        manual_source_url=manual_source_url,
        manual_source_urls=(manual_source_url,) if manual_source_url else (),
        limit=limit,
    )
    return resolve_source_candidates(request)


def resolve_source_candidates(request: TopicSourceDiscoveryRequest) -> list[dict[str, Any]]:
    """Return mode-aware source candidates for a topic discovery request."""
    normalized_topic = _build_topic_discovery_blob(request.topic, request.focus_terms)
    limit = max(1, int(request.limit or 10))
    mode = _normalize_source_mode(request.source_mode)
    candidates: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    manual_source_urls = _normalize_manual_source_urls(request.manual_source_urls, request.manual_source_url)
    if manual_source_urls:
        manual_candidates = [_build_manual_candidate(source_url) for source_url in manual_source_urls]
        _extend_candidates(candidates, seen_urls, manual_candidates, limit)

    if mode in {TopicSourceMode.CURATED_ONLY, TopicSourceMode.HYBRID} and len(candidates) < limit:
        curated_candidates = _build_curated_candidates(
            curated_sources=request.curated_sources,
        )
        _extend_candidates(candidates, seen_urls, curated_candidates, limit)

    if mode in {TopicSourceMode.DISCOVERY_ONLY, TopicSourceMode.HYBRID} and len(candidates) < limit:
        discovered_candidates = _build_discovered_candidates(normalized_topic)
        _extend_candidates(candidates, seen_urls, discovered_candidates, limit)

    return candidates[:limit]


def _normalize_manual_source_urls(
    manual_source_urls: Sequence[str],
    manual_source_url: str,
) -> list[str]:
    normalized_urls: list[str] = []
    seen: set[str] = set()
    combined = list(manual_source_urls or ())
    if manual_source_url:
        combined.append(manual_source_url)
    for raw_url in combined:
        source_url = str(raw_url or "").strip()
        if not source_url or source_url in seen:
            continue
        seen.add(source_url)
        normalized_urls.append(source_url)
    return normalized_urls


def _build_discovered_candidates(normalized_topic: str) -> Iterable[dict[str, Any]]:
    normalized_topic = str(normalized_topic or "").strip().lower()
    for template in _select_templates(normalized_topic):
        yield _build_candidate_from_template(template)


def _build_topic_discovery_blob(topic: str, focus_terms: Sequence[str]) -> str:
    parts = [str(topic or "").strip().lower()]
    parts.extend(str(term or "").strip().lower() for term in (focus_terms or ()) if str(term or "").strip())
    return " ".join(part for part in parts if part).strip()


def _build_curated_candidates(
    *,
    curated_sources: Sequence[CuratedSourceSeed],
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seed in curated_sources:
        source_url = str(seed.url or "").strip()
        if not source_url:
            continue
        candidates.append(
            _build_curated_candidate(
                url=source_url,
                title=seed.title,
                description=seed.description,
                quality_estimate=seed.quality_estimate,
                is_manual=seed.is_manual,
                default_selected=seed.default_selected,
            )
        )
    return candidates


def _extend_candidates(
    candidates: list[dict[str, Any]],
    seen_urls: set[str],
    new_candidates: Iterable[dict[str, Any]],
    limit: int,
) -> None:
    for candidate in new_candidates:
        candidate_key = str(candidate.get("normalized_url") or candidate.get("url") or "").strip()
        if not candidate_key or candidate_key in seen_urls:
            continue
        seen_urls.add(candidate_key)
        candidates.append(candidate)
        if len(candidates) >= limit:
            break


def _normalize_source_mode(source_mode: str) -> str:
    normalized_mode = str(source_mode or "").strip()
    valid_modes = {choice for choice, _label in TopicSourceMode.choices}
    if normalized_mode in valid_modes:
        return normalized_mode
    return TopicSourceMode.DISCOVERY_ONLY


def _select_templates(normalized_topic: str) -> list[SourceCandidateTemplate]:
    matched_templates: list[SourceCandidateTemplate] = []
    for keywords, templates in TOPIC_SOURCE_TEMPLATES:
        if any(keyword in normalized_topic for keyword in keywords):
            matched_templates.extend(templates)

    if matched_templates:
        return matched_templates
    return DEFAULT_SOURCE_TEMPLATES


def _build_manual_candidate(source_url: str) -> dict[str, Any]:
    candidate = _build_curated_candidate(
        url=source_url,
        title=f"Manual source / {_fallback_title_from_url(source_url)}",
        description="User-provided source URL. Review and select it like any other candidate source.",
        quality_estimate="manual",
        is_manual=True,
        default_selected=True,
    )
    return candidate


def _build_curated_candidate(
    *,
    url: str,
    title: str,
    description: str,
    quality_estimate: str,
    is_manual: bool,
    default_selected: bool,
) -> dict[str, Any]:
    normalized_source = classify_source_url(url)
    candidate = _build_candidate_record(
        url=normalized_source.original_url,
        normalized_url=normalized_source.normalized_url,
        title=title or f"Curated source / {_fallback_title_from_url(normalized_source.original_url)}",
        description=description or "Previously saved source candidate for this topic.",
        quality_estimate=quality_estimate,
        source_type=normalized_source.source_type,
        platform=normalized_source.platform,
        detection_reason=normalized_source.detection_reason,
    )
    candidate["is_manual"] = is_manual
    candidate["default_selected"] = default_selected
    candidate["candidate_origin"] = "curated"
    return candidate


def _build_candidate_from_template(template: SourceCandidateTemplate) -> dict[str, Any]:
    normalized_source = classify_source_url(template.url)
    candidate = _build_candidate_record(
        url=normalized_source.original_url,
        normalized_url=normalized_source.normalized_url,
        title=template.title,
        description=template.description,
        quality_estimate=template.quality_estimate,
        source_type=normalized_source.source_type,
        platform=normalized_source.platform,
        detection_reason=normalized_source.detection_reason,
    )
    candidate["candidate_origin"] = "discovered"
    return candidate


def _build_candidate_record(
    *,
    url: str,
    normalized_url: str,
    title: str,
    description: str,
    quality_estimate: str,
    source_type: str,
    platform: str,
    detection_reason: str,
) -> dict[str, Any]:
    recent_article_count = _detect_recent_article_count(url, source_type)
    return {
        "url": url,
        "normalized_url": normalized_url,
        "title": title,
        "description": description,
        "source_type": source_type,
        "platform": platform,
        "detection_reason": detection_reason,
        "recent_article_count": recent_article_count,
        "has_recent_article_count": recent_article_count is not None,
        "quality_estimate": quality_estimate,
        "default_selected": quality_estimate in {"high", "manual"},
    }


def _detect_recent_article_count(url: str, source_type: str) -> int | None:
    if source_type == "devto_tag":
        source = classify_source_url(url)
        payload = fetch_dev_to_article_list(source.normalized_url)
        if isinstance(payload, list):
            return len(payload[:30])
        return None

    if source_type == "rss_feed":
        snapshot = get_rss_debug_snapshot(url)
        total_entries = snapshot.get("total_entries")
        if isinstance(total_entries, int):
            return total_entries
    return None


def _fallback_title_from_url(url: str) -> str:
    parsed = urlparse(url)
    host = parsed.netloc or parsed.path or url
    return host.replace("www.", "").strip("/") or url
