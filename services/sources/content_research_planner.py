"""AI-assisted planning for post-worthy source discovery queries."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from apps.ai.client import OpenAIClient
from django.conf import settings

MAX_FINAL_QUERY_COUNT = 6
MIN_QUERY_WORD_COUNT = 3
PLACEHOLDER_API_KEYS = {"", "sk-your-key"}
PROMPT_TEMPLATE_PATH = Path(__file__).resolve().parent / "prompts" / "content_research_planner.md"


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
            "final_queries": list(self.final_queries),
            "prompt": self.prompt,
            "raw_response_text": self.raw_response_text,
        }


def create_content_research_plan(topic) -> ContentResearchPlannerResult:
    topic_title = _get_topic_title(topic)
    topic_keywords = _normalize_keywords(getattr(topic, "keywords", ()) or ())
    prompt = build_content_research_planner_prompt(topic_title, topic_keywords)

    if _should_use_fallback():
        return _build_fallback_result(
            topic_title,
            topic_keywords,
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
            prompt=prompt,
            response_text=response.text,
        )
    except Exception as exc:  # noqa: BLE001 - deterministic fallback keeps discovery usable offline
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            prompt=prompt,
            error_message=f"AI content research planning failed: {exc}",
        )


def build_content_research_planner_prompt(topic_title: str, topic_keywords: Sequence[str]) -> str:
    normalized_title = str(topic_title or "").strip()
    normalized_keywords = [keyword for keyword in _normalize_keywords(topic_keywords) if keyword]
    keywords_text = ", ".join(normalized_keywords) if normalized_keywords else "(none)"
    template = _load_prompt_template()
    return _render_prompt_template(
        template,
        topic_title=normalized_title,
        topic_keywords=keywords_text,
        max_final_query_count=str(MAX_FINAL_QUERY_COUNT),
    )


def _build_result_from_ai_response(
    *,
    topic_title: str,
    topic_keywords: Sequence[str],
    prompt: str,
    response_text: str,
) -> ContentResearchPlannerResult:
    try:
        payload = _parse_planner_payload(response_text)
    except ValueError as exc:
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            prompt=prompt,
            raw_response_text=response_text,
            error_message=str(exc),
        )

    cleaned_queries = _clean_queries(payload.get("queries"), topic_title=topic_title, topic_keywords=topic_keywords)
    if not cleaned_queries:
        return _build_fallback_result(
            topic_title,
            topic_keywords,
            prompt=prompt,
            raw_response_text=response_text,
            error_message="AI planner returned no usable queries.",
        )

    return ContentResearchPlannerResult(
        planner_status="ai_planned",
        fallback_used=False,
        final_queries=tuple(cleaned_queries),
        topic_interpretation=_clean_text(payload.get("topic_interpretation")),
        content_research_goal=_clean_text(payload.get("content_research_goal")),
        source_selection_criteria=_clean_selection_criteria(payload.get("source_selection_criteria")),
        content_tension_opportunities=tuple(_clean_named_pairs(payload.get("content_tension_opportunities"), "tension", "why_it_matters")),
        search_angles=tuple(_clean_named_pairs(payload.get("search_angles"), "angle", "purpose")),
        prompt=prompt,
        raw_response_text=response_text,
    )


def _build_fallback_result(
    topic_title: str,
    topic_keywords: Sequence[str],
    *,
    prompt: str,
    error_message: str,
    raw_response_text: str = "",
) -> ContentResearchPlannerResult:
    final_queries = tuple(_build_fallback_queries(topic_title, topic_keywords))
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


def _clean_queries(raw_queries: Any, *, topic_title: str, topic_keywords: Sequence[str]) -> list[str]:
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
        normalized_key = query.casefold()
        if normalized_key in seen:
            continue
        if _is_query_too_short_or_generic(query, all_context_tokens):
            continue
        seen.add(normalized_key)
        cleaned.append(query)
        if len(cleaned) >= MAX_FINAL_QUERY_COUNT:
            break
    return cleaned


def _is_query_too_short_or_generic(query: str, context_tokens: set[str]) -> bool:
    tokens = _tokenize(query)
    if len(tokens) < MIN_QUERY_WORD_COUNT:
        return True
    if context_tokens and set(tokens).issubset(context_tokens) and len(tokens) <= max(len(context_tokens), 3):
        return True
    return False


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text or "").casefold())


def _build_fallback_queries(topic_title: str, topic_keywords: Sequence[str]) -> list[str]:
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
    return queries[:MAX_FINAL_QUERY_COUNT]


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
