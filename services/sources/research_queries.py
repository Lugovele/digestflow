"""Deterministic research query planning for future source discovery work."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Sequence
import re


_TECHNICAL_INDICATORS = {
    "agent",
    "agents",
    "ai",
    "api",
    "automation",
    "developer",
    "development",
    "django",
    "engineering",
    "javascript",
    "llm",
    "make",
    "n8n",
    "python",
    "workflow",
    "workflows",
    "zapier",
}

_GENERAL_MODIFIERS: dict[str, tuple[str, str]] = {
    "official_guidelines": ("official guidelines", "Look for authoritative guidance and standards."),
    "evidence_based": ("evidence based", "Look for evidence-oriented coverage and grounded advice."),
    "expert_advice": ("expert advice", "Look for practical guidance from credible experts."),
    "organization_resources": ("organization resources", "Look for organization or institution resource pages."),
}

_TECHNICAL_MODIFIERS: dict[str, tuple[str, str]] = {
    "implementation_guide": ("implementation guide", "Look for hands-on implementation guidance."),
    "case_study": ("case study", "Look for concrete examples and outcome-focused writeups."),
    "engineering_blog": ("engineering blog", "Look for practitioner blog posts with technical detail."),
    "best_practices": ("best practices", "Look for durable practices and operational guidance."),
}


class ResearchQueryIntent(StrEnum):
    OFFICIAL_GUIDELINES = "official_guidelines"
    EVIDENCE_BASED = "evidence_based"
    EXPERT_ADVICE = "expert_advice"
    ORGANIZATION_RESOURCES = "organization_resources"
    IMPLEMENTATION_GUIDE = "implementation_guide"
    CASE_STUDY = "case_study"
    ENGINEERING_BLOG = "engineering_blog"
    BEST_PRACTICES = "best_practices"


@dataclass(frozen=True)
class ResearchQueryItem:
    intent: ResearchQueryIntent
    query: str
    reason: str
    source_type_hint: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResearchQueryPlan:
    topic_name: str
    topic_keywords: tuple[str, ...]
    topic_domain: str
    query_items: tuple[ResearchQueryItem, ...]
    diagnostics: dict[str, Any]


def build_research_query_plan(topic) -> ResearchQueryPlan:
    topic_name = str(getattr(topic, "name", "") or "").strip()
    topic_keywords = _normalize_keywords(getattr(topic, "keywords", ()) or ())
    topic_domain, domain_diagnostics = _detect_topic_domain(topic_name, topic_keywords)
    query_specs = _build_query_specs(topic_name, topic_keywords, topic_domain)
    query_items = _build_query_items(
        topic_name=topic_name,
        topic_keywords=topic_keywords,
        topic_domain=topic_domain,
        query_specs=query_specs,
    )

    diagnostics = {
        "topic_domain": topic_domain,
        "domain_diagnostics": domain_diagnostics,
        "query_count": len(query_items),
        "topic_keyword_count": len(topic_keywords),
        "used_topic_keywords": _collect_used_keywords(query_items, topic_keywords),
    }

    return ResearchQueryPlan(
        topic_name=topic_name,
        topic_keywords=tuple(topic_keywords),
        topic_domain=topic_domain,
        query_items=tuple(query_items),
        diagnostics=diagnostics,
    )


def _normalize_keywords(raw_keywords: Iterable[str]) -> list[str]:
    if isinstance(raw_keywords, str):
        raw_values: list[str] = [raw_keywords]
    else:
        raw_values = list(raw_keywords)

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_value in raw_values:
        for value in _split_keyword_value(raw_value):
            key = value.casefold()
            if not value or key in seen:
                continue
            seen.add(key)
            normalized.append(value)
    return normalized


def _split_keyword_value(raw_value: Any) -> list[str]:
    value = str(raw_value or "").strip()
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _detect_topic_domain(topic_name: str, topic_keywords: Sequence[str]) -> tuple[str, dict[str, Any]]:
    corpus = " ".join([topic_name, *topic_keywords]).casefold()
    matched_indicators = sorted(indicator for indicator in _TECHNICAL_INDICATORS if indicator in corpus)
    if matched_indicators:
        return "technical", {"matched_indicators": matched_indicators}
    return "general", {"matched_indicators": []}


def _build_query_specs(
    topic_name: str,
    topic_keywords: Sequence[str],
    topic_domain: str,
) -> list[tuple[ResearchQueryIntent, str, str, str]]:
    if topic_domain == "technical":
        modifier_map = _TECHNICAL_MODIFIERS
        source_type_hint = "technical_web"
    else:
        modifier_map = _GENERAL_MODIFIERS
        source_type_hint = "general_web"

    keyword_groups = _build_keyword_groups(topic_keywords, max_groups=len(modifier_map))
    specs: list[tuple[ResearchQueryIntent, str, str, str]] = []
    for index, (intent_name, (modifier, reason)) in enumerate(modifier_map.items()):
        intent = ResearchQueryIntent(intent_name)
        keyword_group = keyword_groups[index] if index < len(keyword_groups) else ()
        specs.append((intent, modifier, reason, _render_query(topic_name, keyword_group, modifier)))

    return specs


def _build_query_items(
    *,
    topic_name: str,
    topic_keywords: Sequence[str],
    topic_domain: str,
    query_specs: Sequence[tuple[ResearchQueryIntent, str, str, str]],
) -> list[ResearchQueryItem]:
    seen_queries: set[str] = set()
    items: list[ResearchQueryItem] = []
    for intent, modifier, reason, query in query_specs:
        normalized_query = query.casefold()
        if not normalized_query or normalized_query in seen_queries:
            continue
        seen_queries.add(normalized_query)
        items.append(
            ResearchQueryItem(
                intent=intent,
                query=query,
                reason=reason,
                source_type_hint="technical_web" if topic_domain == "technical" else "general_web",
                diagnostics={
                    "topic_name": topic_name,
                    "topic_keywords": list(topic_keywords),
                    "modifier": modifier,
                    "query_word_count": len(query.split()),
                },
            )
        )
    return items


def _build_keyword_groups(topic_keywords: Sequence[str], *, max_groups: int) -> list[tuple[str, ...]]:
    if not topic_keywords:
        return [()] * max_groups

    groups: list[tuple[str, ...]] = []
    keywords = list(topic_keywords)
    for index in range(max_groups):
        primary = keywords[index % len(keywords)]
        group = [primary]
        secondary_index = index + 1
        if secondary_index < len(keywords):
            group.append(keywords[secondary_index])
        groups.append(tuple(group))
    return groups


def _render_query(topic_name: str, keyword_group: Sequence[str], modifier: str) -> str:
    parts = [topic_name.strip()]
    parts.extend(keyword.strip() for keyword in keyword_group if str(keyword or "").strip())
    parts.append(modifier.strip())
    query = " ".join(part for part in parts if part).strip()
    return re.sub(r"\s+", " ", query)


def _collect_used_keywords(query_items: Sequence[ResearchQueryItem], topic_keywords: Sequence[str]) -> list[str]:
    used: list[str] = []
    for keyword in topic_keywords:
        lowered_keyword = keyword.casefold()
        if any(lowered_keyword in item.query.casefold() for item in query_items):
            used.append(keyword)
    return used
