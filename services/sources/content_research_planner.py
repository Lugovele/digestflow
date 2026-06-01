"""AI-assisted planning for post-worthy source discovery queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from apps.ai.client import OpenAIClient
from django.conf import settings
from django.utils import timezone
from services.sources.query_history_summary import (
    build_query_history_summary,
    render_query_history_summary_for_prompt,
)

MAX_FINAL_QUERY_COUNT = 6
MIN_QUERY_WORD_COUNT = 3
PLACEHOLDER_API_KEYS = {"", "sk-your-key"}
PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "content_research_planner.md"
_KNOWN_QUERY_TERM_CORRECTIONS: tuple[tuple[str, str], ...] = (
    ("bitcion", "Bitcoin"),
)
_GENERIC_RETAIL_QUERY_NEEDLES: tuple[str, ...] = (
    "retail investor",
    "new investors",
    "trading strategies",
    "trading strategy",
    "trading approaches",
    "for beginners",
    "best trading strategies",
    "ultimate guide",
    "price prediction",
    "live price",
)
_SURFACE_KEY_QUERY_TERMS: dict[str, str] = {
    "etf_flows_report": "ETF flows weekly report",
    "etf_flow_data_market_report": "spot ETF fund flows analysis",
    "institutional_demand_report": "treasury holdings institutional demand",
    "institutional_flows_report": "institutional fund flows report",
    "funding_open_interest_report": "funding rates open interest report",
    "funding_rates_analysis": "funding rates analysis",
    "open_interest_futures_positioning": "open interest futures positioning",
    "derivatives_positioning_market_structure": "derivatives positioning market structure",
    "market_structure_report": "market structure report",
    "market_structure_research_paper": "market structure research paper",
    "research_paper": "research paper",
    "on_chain_exchange_reserves_analysis": "on-chain exchange reserves analysis",
    "on_chain_weekly_report": "on-chain weekly report",
    "on_chain_analysis": "on-chain analysis recent report",
    "analyst_report": "analyst report market outlook",
    "volatility_market_structure_report": "volatility market structure report",
    "volatility_drawdown_risk_analysis": "volatility drawdown risk analysis",
}


@dataclass(frozen=True)
class ContentResearchPlannerResult:
    planner_status: str
    fallback_used: bool
    final_queries: tuple[str, ...]
    error_message: str = ""
    topic_interpretation: str = ""
    content_research_goal: str = ""
    source_selection_criteria: dict[str, Any] = field(default_factory=dict)
    content_tension_opportunities: tuple[dict[str, str], ...] = ()
    search_angles: tuple[dict[str, str], ...] = ()
    query_history_summary: dict[str, Any] = field(default_factory=dict)
    prompt: str = ""
    raw_response_text: str = ""

    @property
    def diagnostics(self) -> dict[str, Any]:
        return {
            "planner_status": self.planner_status,
            "fallback_used": self.fallback_used,
            "error_message": self.error_message,
            "topic_interpretation": self.topic_interpretation,
            "content_research_goal": self.content_research_goal,
            "source_selection_criteria": dict(self.source_selection_criteria),
            "content_tension_opportunities": [dict(item) for item in self.content_tension_opportunities],
            "search_angles": [dict(item) for item in self.search_angles],
            "query_history_summary": dict(self.query_history_summary),
            "final_queries": list(self.final_queries),
            "prompt": self.prompt,
            "raw_response_text": self.raw_response_text,
        }


def create_content_research_plan(topic) -> ContentResearchPlannerResult:
    topic_title = _get_topic_title(topic)
    topic_keywords = _normalize_keywords(getattr(topic, "keywords", ()) or ())
    query_history_summary = build_query_history_summary(topic)
    prompt = build_content_research_planner_prompt(
        topic_title,
        topic_keywords,
        query_history_summary=query_history_summary,
    )

    if _should_use_fallback():
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            query_history_summary=query_history_summary,
            prompt=prompt,
            error_message="OPENAI_API_KEY is missing or uses the local placeholder.",
        )

    try:
        response = OpenAIClient().generate_text(
            prompt=prompt,
            max_output_tokens=1200,
            json_mode=True,
        )
        return _build_result_from_ai_response(
            topic_title=topic_title,
            topic_keywords=topic_keywords,
            query_history_summary=query_history_summary,
            prompt=prompt,
            response_text=response.text,
        )
    except Exception as exc:  # noqa: BLE001 - deterministic fallback keeps discovery usable offline
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            query_history_summary=query_history_summary,
            prompt=prompt,
            error_message=f"AI content research planning failed: {exc}",
        )


def build_content_research_planner_prompt(
    topic_title: str,
    topic_keywords: Sequence[str],
    *,
    query_history_summary: dict[str, Any] | None = None,
) -> str:
    current_date = timezone.localdate()
    normalized_title = str(topic_title or "").strip()
    normalized_keywords = [keyword for keyword in _normalize_keywords(topic_keywords) if keyword]
    keywords_text = ", ".join(normalized_keywords) if normalized_keywords else "(none)"
    template = _load_prompt_template()
    return _render_prompt_template(
        template,
        topic_title=normalized_title,
        topic_keywords=keywords_text,
        query_history_summary=render_query_history_summary_for_prompt(query_history_summary),
        max_final_query_count=str(MAX_FINAL_QUERY_COUNT),
        current_date=current_date.isoformat(),
        current_year=str(current_date.year),
    )


def _build_result_from_ai_response(
    *,
    topic_title: str,
    topic_keywords: Sequence[str],
    query_history_summary: dict[str, Any],
    prompt: str,
    response_text: str,
) -> ContentResearchPlannerResult:
    try:
        payload = _parse_planner_payload(response_text)
    except ValueError as exc:
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            query_history_summary=query_history_summary,
            prompt=prompt,
            raw_response_text=response_text,
            error_message=str(exc),
        )

    cleaned_queries = _clean_queries(
        payload.get("queries"),
        topic_title=topic_title,
        topic_keywords=topic_keywords,
        query_history_summary=query_history_summary,
    )
    if not cleaned_queries:
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            query_history_summary=query_history_summary,
            prompt=prompt,
            raw_response_text=response_text,
            error_message="AI planner returned no usable queries.",
        )

    aligned_search_angles = _align_search_angles_with_final_queries(
        cleaned_queries,
        payload.get("search_angles"),
    )

    return ContentResearchPlannerResult(
        planner_status="ai_planned",
        fallback_used=False,
        final_queries=tuple(cleaned_queries),
        topic_interpretation=_clean_text(payload.get("topic_interpretation")),
        content_research_goal=_clean_text(payload.get("content_research_goal")),
        source_selection_criteria=_clean_selection_criteria(payload.get("source_selection_criteria")),
        content_tension_opportunities=tuple(_clean_named_pairs(payload.get("content_tension_opportunities"), "tension", "why_it_matters")),
        search_angles=tuple(aligned_search_angles),
        query_history_summary=dict(query_history_summary),
        prompt=prompt,
        raw_response_text=response_text,
    )


def _build_fallback_result(
    topic_title: str,
    topic_keywords: Sequence[str],
    *,
    query_history_summary: dict[str, Any],
    prompt: str,
    error_message: str,
    raw_response_text: str = "",
) -> ContentResearchPlannerResult:
    final_queries = tuple(
        _build_fallback_queries(
            topic_title,
            topic_keywords,
            query_history_summary=query_history_summary,
        )
    )
    return ContentResearchPlannerResult(
        planner_status="fallback_used",
        fallback_used=True,
        final_queries=final_queries,
        error_message=error_message,
        topic_interpretation=f"Content research around {topic_title}.".strip(),
        content_research_goal="Find fresh, practical, post-worthy materials for the topic digest and follow-up post.",
        source_selection_criteria={
            "must_be_relevant_to": [topic_title] if topic_title else [],
            "preferred_material_types": [
                "recent review",
                "practical case",
                "expert opinion",
                "research report",
            ],
            "freshness_signals": ["recent examples", "current trends", "new reports"],
            "post_value_signals": [
                "practical outcomes",
                "trade-offs",
                "conflicting opinions",
            ],
            "relevance_boundary": "Stay close to the topic title and supporting keywords.",
        },
        content_tension_opportunities=(
            {
                "tension": "Compare practical wins against risks or trade-offs.",
                "why_it_matters": "Useful posts usually need contrast, not only summary.",
            },
        ),
        search_angles=(
            {"angle": "recent examples", "purpose": "Surface current practice and fresh evidence."},
            {"angle": "expert opinion", "purpose": "Find opinionated sources that support a stronger post."},
        ),
        query_history_summary=dict(query_history_summary),
        prompt=prompt,
        raw_response_text=raw_response_text,
    )


def _parse_planner_payload(response_text: str) -> dict[str, Any]:
    candidate = _extract_json_candidate(response_text)
    if not candidate:
        raise ValueError("AI planner returned invalid JSON.")
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"AI planner returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("AI planner response must be a JSON object.")
    for key in (
        "topic_interpretation",
        "content_research_goal",
        "source_selection_criteria",
        "content_tension_opportunities",
        "search_angles",
        "queries",
    ):
        if key not in payload:
            raise ValueError(f"AI planner response is missing required field: {key}")
    if not isinstance(payload.get("queries"), list):
        raise ValueError("AI planner queries field must be a list.")
    return payload


def _extract_json_candidate(response_text: str) -> str:
    text = str(response_text or "").strip()
    if not text:
        return ""
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return ""
    return text[first_brace : last_brace + 1]


def _clean_queries(
    raw_queries: Any,
    *,
    topic_title: str,
    topic_keywords: Sequence[str],
    query_history_summary: dict[str, Any] | None = None,
) -> list[str]:
    if not isinstance(raw_queries, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    topic_token_set = set(_tokenize(topic_title))
    keyword_token_set = set(token for keyword in topic_keywords for token in _tokenize(keyword))
    all_context_tokens = topic_token_set | keyword_token_set

    for raw_query in raw_queries:
        query = re.sub(r"\s+", " ", str(raw_query or "").strip())
        if not query:
            continue
        query = _polish_query_text(query, topic_title=topic_title)
        query = _rewrite_stale_year_query(query)
        if not query:
            continue
        normalized_key = query.casefold()
        if normalized_key in seen:
            continue
        if _is_query_too_short_or_generic(query, all_context_tokens):
            continue
        seen.add(normalized_key)
        cleaned.append(query)
        if len(cleaned) >= MAX_FINAL_QUERY_COUNT:
            break
    return _apply_quality_material_guidance(
        cleaned,
        topic_title=topic_title,
        topic_keywords=topic_keywords,
        query_history_summary=query_history_summary,
    )


def _is_query_too_short_or_generic(query: str, context_tokens: set[str]) -> bool:
    tokens = _tokenize(query)
    if len(tokens) < MIN_QUERY_WORD_COUNT:
        return True
    if context_tokens and set(tokens).issubset(context_tokens) and len(tokens) <= max(len(context_tokens), 3):
        return True
    return False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text or "").casefold())


def _rewrite_stale_year_query(query: str) -> str:
    current_year = timezone.localdate().year
    stale_years = [int(match.group(0)) for match in re.finditer(r"\b20\d{2}\b", query) if int(match.group(0)) < current_year]
    if not stale_years:
        return re.sub(r"\s+", " ", query).strip()

    rewritten = re.sub(
        r"\b(?:in|during|for|from)\s+(20\d{2})\b",
        lambda match: "" if int(match.group(1)) < current_year else match.group(0),
        query,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"\b(20\d{2})\b",
        lambda match: "" if int(match.group(1)) < current_year else match.group(0),
        rewritten,
    )
    rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,:-")
    if rewritten and not _contains_freshness_signal(rewritten):
        rewritten = f"{rewritten} latest"
    return re.sub(r"\s+", " ", rewritten).strip()


def _contains_freshness_signal(query: str) -> bool:
    lowered = str(query or "").casefold()
    freshness_needles = (
        "latest",
        "current",
        "recent",
        "this month",
        "now",
        str(timezone.localdate().year),
    )
    return any(needle in lowered for needle in freshness_needles)


def _build_fallback_queries(
    topic_title: str,
    topic_keywords: Sequence[str],
    *,
    query_history_summary: dict[str, Any] | None = None,
) -> list[str]:
    title = str(topic_title or "").strip()
    keyword_context = ""
    if topic_keywords:
        keyword_context = topic_keywords[0]
    base_topic = f"{title} {keyword_context}".strip() if keyword_context and keyword_context.casefold() not in title.casefold() else title
    patterns = (
        "recent examples",
        "expert opinion",
        "case study",
        "research report",
        "practical guide",
        "trends",
    )
    queries: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        query = re.sub(r"\s+", " ", f"{base_topic} {pattern}".strip())
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return _apply_quality_material_guidance(
        queries[:MAX_FINAL_QUERY_COUNT],
        topic_title=topic_title,
        topic_keywords=topic_keywords,
        query_history_summary=query_history_summary,
    )


def _apply_quality_material_guidance(
    queries: Sequence[str],
    *,
    topic_title: str,
    topic_keywords: Sequence[str],
    query_history_summary: dict[str, Any] | None,
) -> list[str]:
    preferred_terms = _extract_preferred_material_terms(query_history_summary)
    normalized_queries = [re.sub(r"\s+", " ", str(query or "").strip()) for query in queries if str(query or "").strip()]
    if not preferred_terms:
        return _apply_search_surface_memory(
            normalized_queries[:MAX_FINAL_QUERY_COUNT],
            topic_title=topic_title,
            topic_keywords=topic_keywords,
            query_history_summary=query_history_summary,
        )

    current_material_query_count = sum(1 for query in normalized_queries if _query_uses_preferred_material_term(query, preferred_terms))
    target_material_query_count = min(3, len(preferred_terms), MAX_FINAL_QUERY_COUNT)
    if current_material_query_count >= target_material_query_count:
        return _apply_search_surface_memory(
            normalized_queries[:MAX_FINAL_QUERY_COUNT],
            topic_title=topic_title,
            topic_keywords=topic_keywords,
            query_history_summary=query_history_summary,
        )

    base_topic = _build_guided_query_topic_base(topic_title, topic_keywords)
    guided_queries = _build_guided_queries_for_terms(base_topic, preferred_terms)

    merged: list[str] = []
    seen: set[str] = set()
    for query in guided_queries:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(query)
    for query in normalized_queries:
        key = query.casefold()
        if key in seen:
            continue
        seen.add(key)
        merged.append(query)
        if len(merged) >= MAX_FINAL_QUERY_COUNT:
            break
    return _apply_search_surface_memory(
        merged[:MAX_FINAL_QUERY_COUNT],
        topic_title=topic_title,
        topic_keywords=topic_keywords,
        query_history_summary=query_history_summary,
    )


def _apply_search_surface_memory(
    queries: Sequence[str],
    *,
    topic_title: str,
    topic_keywords: Sequence[str],
    query_history_summary: dict[str, Any] | None,
) -> list[str]:
    preferred_terms = _extract_preferred_material_terms(query_history_summary)
    memory = (
        query_history_summary.get("search_surface_memory")
        if isinstance(query_history_summary, dict) and isinstance(query_history_summary.get("search_surface_memory"), dict)
        else {}
    )
    if not memory:
        return _limit_generic_retail_queries(
            queries,
            preferred_terms=preferred_terms,
            topic_title=topic_title,
            topic_keywords=topic_keywords,
        )

    recent_query_keys = {
        re.sub(r"\s+", " ", str(item or "").strip()).casefold()
        for item in (query_history_summary.get("recent_query_texts") or [] if isinstance(query_history_summary, dict) else [])
        if str(item or "").strip()
    }
    avoided_surfaces = {
        str(item or "").strip()
        for item in memory.get("avoided_surfaces") or []
        if str(item or "").strip()
    }
    preferred_surfaces = [
        str(item or "").strip()
        for item in memory.get("preferred_surfaces") or []
        if str(item or "").strip()
    ]
    underexplored_surfaces = [
        str(item or "").strip()
        for item in memory.get("underexplored_surfaces") or []
        if str(item or "").strip()
    ]

    base_topic = _build_guided_query_topic_base(topic_title, topic_keywords)
    injected_queries = _build_guided_queries_for_surface_keys(
        base_topic,
        [*preferred_surfaces, *underexplored_surfaces],
    )
    preferred_surface_set = set(preferred_surfaces) | set(underexplored_surfaces)

    primary: list[str] = []
    fallback: list[str] = []
    seen_queries: set[str] = set()
    seen_surfaces: set[str] = set()
    for query in [*injected_queries, *queries]:
        normalized_query = re.sub(r"\s+", " ", str(query or "").strip())
        if not normalized_query:
            continue
        query_key = normalized_query.casefold()
        if query_key in seen_queries:
            continue
        surface_key = _surface_key_for_query_text(normalized_query)
        is_repeat = query_key in recent_query_keys
        is_avoided = bool(surface_key and surface_key in avoided_surfaces)
        is_duplicate_surface = bool(surface_key and surface_key in seen_surfaces and surface_key in preferred_surface_set)
        target_bucket = fallback if is_repeat or is_avoided or is_duplicate_surface else primary
        target_bucket.append(normalized_query)
        seen_queries.add(query_key)
        if surface_key:
            seen_surfaces.add(surface_key)

    merged = [*primary, *fallback]
    return _limit_generic_retail_queries(
        merged[:MAX_FINAL_QUERY_COUNT],
        preferred_terms=preferred_terms,
        topic_title=topic_title,
        topic_keywords=topic_keywords,
    )


def _build_guided_queries_for_terms(base_topic: str, preferred_terms: Sequence[str]) -> list[str]:
    guided_queries: list[str] = []
    for term in preferred_terms:
        guided_query = re.sub(r"\s+", " ", f"{base_topic} {term} latest".strip())
        guided_query = guided_query.strip()
        if not guided_query:
            continue
        guided_queries.append(guided_query)
    return guided_queries


def _build_guided_queries_for_surface_keys(base_topic: str, surface_keys: Sequence[str]) -> list[str]:
    guided_queries: list[str] = []
    seen: set[str] = set()
    for surface_key in surface_keys:
        term = _SURFACE_KEY_QUERY_TERMS.get(str(surface_key or "").strip())
        if not term:
            continue
        query = re.sub(r"\s+", " ", f"{base_topic} {term}".strip())
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        guided_queries.append(query)
    return guided_queries


def _limit_generic_retail_queries(
    queries: Sequence[str],
    *,
    preferred_terms: Sequence[str],
    topic_title: str,
    topic_keywords: Sequence[str],
) -> list[str]:
    limited: list[str] = []
    generic_retail_count = 0
    seen: set[str] = set()
    for query in queries:
        normalized_query = re.sub(r"\s+", " ", str(query or "").strip())
        if not normalized_query:
            continue
        key = normalized_query.casefold()
        if key in seen:
            continue
        is_generic_retail = _is_generic_retail_query(normalized_query)
        if is_generic_retail and generic_retail_count >= 1:
            continue
        seen.add(key)
        limited.append(normalized_query)
        if is_generic_retail:
            generic_retail_count += 1

    if len(limited) >= MAX_FINAL_QUERY_COUNT:
        return limited[:MAX_FINAL_QUERY_COUNT]

    base_topic = _build_guided_query_topic_base(topic_title, topic_keywords)
    for guided_query in _build_guided_queries_for_terms(base_topic, preferred_terms):
        key = guided_query.casefold()
        if key in seen:
            continue
        seen.add(key)
        limited.append(guided_query)
        if len(limited) >= MAX_FINAL_QUERY_COUNT:
            break
    return limited[:MAX_FINAL_QUERY_COUNT]


def _build_guided_query_topic_base(topic_title: str, topic_keywords: Sequence[str]) -> str:
    title = _normalize_topic_phrase(topic_title)
    if title and "market" in title.casefold() and "analysis" not in title.casefold():
        title = f"{title} analysis"
    if title and "market" in title.casefold():
        return title
    if not topic_keywords:
        return title
    primary_keyword = str(topic_keywords[0] or "").strip()
    if not primary_keyword:
        return title
    if primary_keyword.casefold() in title.casefold():
        return title
    return f"{title} {primary_keyword}".strip()


def _extract_preferred_material_terms(query_history_summary: dict[str, Any] | None) -> list[str]:
    if not isinstance(query_history_summary, dict):
        return []

    suggestions: list[str] = []
    seen: set[str] = set()
    surface_memory = query_history_summary.get("search_surface_memory") if isinstance(query_history_summary.get("search_surface_memory"), dict) else {}
    for surface_key in [*(surface_memory.get("preferred_surfaces") or []), *(surface_memory.get("underexplored_surfaces") or [])]:
        candidate = _SURFACE_KEY_QUERY_TERMS.get(str(surface_key or "").strip(), "")
        key = candidate.casefold()
        if not candidate or key in seen:
            continue
        seen.add(key)
        suggestions.append(candidate)
        if len(suggestions) >= 5:
            return suggestions
    for item in query_history_summary.get("quality_guidance") or []:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        match = re.search(r"Use query terms such as (.+?)(?:\.|$)", cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        for part in match.group(1).split(","):
            candidate = re.sub(r"\s+", " ", part.strip(" .")).strip()
            key = candidate.casefold()
            if not candidate or key in seen:
                continue
            seen.add(key)
            suggestions.append(candidate)
    if suggestions:
        return suggestions[:5]

    fallback_terms_by_material_type = {
        "institutional / analyst report": ("analyst report", "research report"),
        "market data / flow analysis": ("ETF flows", "institutional flows", "funding rates", "open interest"),
        "on-chain analysis": ("on-chain analysis", "exchange reserves"),
        "market structure analysis": ("market structure", "liquidity"),
        "research paper": ("research paper",),
    }
    for item in query_history_summary.get("preferred_material_types_found") or []:
        material_type = str(item.get("material_type") or "").strip()
        for term in fallback_terms_by_material_type.get(material_type, ()):
            key = term.casefold()
            if key in seen:
                continue
            seen.add(key)
            suggestions.append(term)
            if len(suggestions) >= 5:
                return suggestions
    return suggestions


def _query_uses_preferred_material_term(query: str, preferred_terms: Sequence[str]) -> bool:
    normalized_query = str(query or "").casefold()
    return any(str(term or "").casefold() in normalized_query for term in preferred_terms)


def _surface_key_for_query_text(query: str) -> str:
    normalized_query = str(query or "").casefold()
    for surface_key, term in _SURFACE_KEY_QUERY_TERMS.items():
        if term.casefold() in normalized_query:
            return surface_key
    if "exchange reserves" in normalized_query:
        return "on_chain_exchange_reserves_analysis"
    if "on-chain" in normalized_query or "on chain" in normalized_query:
        return "on_chain_analysis"
    if "analyst report" in normalized_query:
        return "analyst_report"
    if "market structure" in normalized_query and "research paper" in normalized_query:
        return "market_structure_research_paper"
    if "research paper" in normalized_query:
        return "research_paper"
    if "market structure" in normalized_query:
        return "market_structure_report"
    if "funding rates" in normalized_query and "open interest" in normalized_query:
        return "funding_open_interest_report"
    if "funding rates" in normalized_query:
        return "funding_rates_analysis"
    if "open interest" in normalized_query:
        return "open_interest_futures_positioning"
    if "institutional flows" in normalized_query or "fund flows" in normalized_query:
        return "institutional_flows_report"
    if "treasury holdings" in normalized_query or "institutional demand" in normalized_query:
        return "institutional_demand_report"
    if "etf flows" in normalized_query or "spot etf" in normalized_query:
        return "etf_flows_report"
    return ""


def _align_search_angles_with_final_queries(
    final_queries: Sequence[str],
    raw_search_angles: Any,
) -> list[dict[str, str]]:
    cleaned_angles = _clean_named_pairs(raw_search_angles, "angle", "purpose")
    aligned: list[dict[str, str]] = []
    for index, query in enumerate(final_queries):
        inferred = _infer_query_metadata_from_text(query)
        if index < len(cleaned_angles):
            candidate = cleaned_angles[index]
            angle = str(candidate.get("angle") or "").strip()
            purpose = str(candidate.get("purpose") or "").strip()
            if angle and purpose and _query_metadata_matches_query(query, angle=angle, purpose=purpose):
                aligned.append({"angle": angle, "purpose": purpose})
                continue
        aligned.append(inferred)
    return aligned


def _query_metadata_matches_query(query: str, *, angle: str, purpose: str) -> bool:
    normalized_query = str(query or "").casefold()
    normalized_angle = str(angle or "").casefold()
    normalized_purpose = str(purpose or "").casefold()

    expected_needles = _material_metadata_needles_for_query(query)
    if expected_needles:
        return any(
            needle in normalized_angle or needle in normalized_purpose
            for needle in expected_needles
        )
    if "retail" in normalized_query and ("retail" in normalized_angle or "retail" in normalized_purpose):
        return True
    if "volatility" in normalized_query and ("volatility" in normalized_angle or "volatility" in normalized_purpose):
        return True
    if "analyst" in normalized_query and ("analyst" in normalized_angle or "analyst" in normalized_purpose):
        return True
    return bool(normalized_angle and normalized_purpose)


def _material_metadata_needles_for_query(query: str) -> tuple[str, ...]:
    normalized_query = str(query or "").casefold()
    if "etf flows" in normalized_query:
        return ("etf", "flow")
    if "institutional flows" in normalized_query:
        return ("institutional", "flow")
    if "funding rates" in normalized_query or "open interest" in normalized_query:
        return ("derivatives", "funding", "open interest", "market structure")
    if "research paper" in normalized_query:
        return ("research", "paper", "evidence")
    if "on-chain analysis" in normalized_query:
        return ("on-chain", "network", "data")
    if "analyst report" in normalized_query:
        return ("analyst", "report")
    if "market structure" in normalized_query:
        return ("market structure", "liquidity", "derivatives")
    return ()


def _infer_query_metadata_from_text(query: str) -> dict[str, str]:
    normalized_query = str(query or "").casefold()
    if "etf flows" in normalized_query:
        return {
            "angle": "ETF flows / fund flows",
            "purpose": "Track spot ETF inflows, outflows, and their market impact.",
        }
    if "institutional flows" in normalized_query:
        return {
            "angle": "institutional flows",
            "purpose": "Track institutional positioning, treasury activity, and fund-flow signals.",
        }
    if "funding rates" in normalized_query or "open interest" in normalized_query:
        return {
            "angle": "derivatives / market structure",
            "purpose": "Track derivatives positioning, funding rates, open interest, and market stress.",
        }
    if "research paper" in normalized_query:
        return {
            "angle": "research evidence",
            "purpose": "Find research-backed evidence, papers, or empirical analysis.",
        }
    if "on-chain analysis" in normalized_query:
        return {
            "angle": "on-chain analysis",
            "purpose": "Use on-chain or network data that can support a grounded market narrative.",
        }
    if "analyst report" in normalized_query:
        return {
            "angle": "analyst report",
            "purpose": "Collect analyst viewpoints, scenario framing, and research-backed outlooks.",
        }
    if "market structure" in normalized_query:
        return {
            "angle": "market structure",
            "purpose": "Look for liquidity, positioning, and market-structure signals shaping current behavior.",
        }
    if "retail" in normalized_query:
        return {
            "angle": "retail behavior",
            "purpose": "Surface current retail participation, behavior, or sentiment patterns.",
        }
    if "volatility" in normalized_query:
        return {
            "angle": "volatility and risk",
            "purpose": "Focus on current volatility, downside risk, and risk-management framing.",
        }
    return {
        "angle": "fresh evidence",
        "purpose": "Find fresh, practical, post-worthy materials with specific evidence or examples.",
    }


def _polish_query_text(query: str, *, topic_title: str) -> str:
    polished = re.sub(r"\s+", " ", str(query or "").strip())
    if not polished:
        return ""
    for typo, canonical in _KNOWN_QUERY_TERM_CORRECTIONS:
        polished = re.sub(rf"\b{re.escape(typo)}\b", canonical, polished, flags=re.IGNORECASE)

    normalized_topic = _normalize_topic_phrase(topic_title)
    if normalized_topic:
        prefix_pattern = rf"^{re.escape(normalized_topic)}\s+"
        remainder = re.sub(prefix_pattern, "", polished, count=1, flags=re.IGNORECASE)
        if remainder != polished and _contains_canonical_topic_signal(remainder, normalized_topic):
            polished = remainder

    polished = re.sub(r"\s+", " ", polished).strip(" ,:-")
    return polished


def _normalize_topic_phrase(topic_title: str) -> str:
    normalized = re.sub(r"\s+", " ", str(topic_title or "").strip())
    for typo, canonical in _KNOWN_QUERY_TERM_CORRECTIONS:
        normalized = re.sub(rf"\b{re.escape(typo)}\b", canonical, normalized, flags=re.IGNORECASE)
    return normalized


def _contains_canonical_topic_signal(query: str, normalized_topic: str) -> bool:
    normalized_query = str(query or "").casefold()
    normalized_topic_value = str(normalized_topic or "").casefold().strip()
    if not normalized_topic_value:
        return False
    if normalized_topic_value in normalized_query:
        return True

    topic_tokens = [token for token in _tokenize(normalized_topic_value) if len(token) >= 5]
    if not topic_tokens:
        topic_tokens = [token for token in _tokenize(normalized_topic_value) if len(token) >= 3]
    if not topic_tokens:
        return False
    return any(token in normalized_query for token in topic_tokens)


def _is_generic_retail_query(query: str) -> bool:
    normalized_query = str(query or "").casefold()
    return any(needle in normalized_query for needle in _GENERIC_RETAIL_QUERY_NEEDLES)


def _normalize_keywords(raw_keywords: Iterable[str]) -> list[str]:
    if isinstance(raw_keywords, str):
        raw_values: list[str] = [raw_keywords]
    else:
        raw_values = list(raw_keywords)
    seen: set[str] = set()
    normalized: list[str] = []
    for raw_value in raw_values:
        value = str(raw_value or "").strip()
        if not value:
            continue
        for part in value.split(","):
            cleaned = part.strip()
            if not cleaned:
                continue
            key = cleaned.casefold()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(cleaned)
    return normalized


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _clean_selection_criteria(raw_value: Any) -> dict[str, Any]:
    if not isinstance(raw_value, dict):
        return {
            "must_be_relevant_to": [],
            "preferred_material_types": [],
            "freshness_signals": [],
            "post_value_signals": [],
            "relevance_boundary": "",
        }
    return {
        "must_be_relevant_to": _clean_string_list(raw_value.get("must_be_relevant_to")),
        "preferred_material_types": _clean_string_list(raw_value.get("preferred_material_types")),
        "freshness_signals": _clean_string_list(raw_value.get("freshness_signals")),
        "post_value_signals": _clean_string_list(raw_value.get("post_value_signals")),
        "relevance_boundary": _clean_text(raw_value.get("relevance_boundary")),
    }


def _clean_string_list(raw_value: Any) -> list[str]:
    if not isinstance(raw_value, list):
        return []
    seen: set[str] = set()
    cleaned: list[str] = []
    for item in raw_value:
        value = _clean_text(item)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(value)
    return cleaned


def _clean_named_pairs(raw_value: Any, first_key: str, second_key: str) -> list[dict[str, str]]:
    if not isinstance(raw_value, list):
        return []
    cleaned: list[dict[str, str]] = []
    for item in raw_value:
        if not isinstance(item, dict):
            continue
        first_value = _clean_text(item.get(first_key))
        second_value = _clean_text(item.get(second_key))
        if not first_value and not second_value:
            continue
        cleaned.append({first_key: first_value, second_key: second_value})
    return cleaned


def _get_topic_title(topic) -> str:
    return str(getattr(topic, "title", "") or getattr(topic, "name", "") or "").strip()


def _should_use_fallback() -> bool:
    api_key = str(getattr(settings, "OPENAI_API_KEY", "") or "").strip()
    return api_key in PLACEHOLDER_API_KEYS


def _load_prompt_template() -> str:
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def _render_prompt_template(template: str, **context: str) -> str:
    rendered = str(template)
    for key, value in context.items():
        rendered = rendered.replace(f"{{{{ {key} }}}}", str(value))
    return rendered
