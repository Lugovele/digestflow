from __future__ import annotations

from typing import Iterable

from django.conf import settings

from apps.ai.client import OpenAIClient
from apps.topics.focus import clean_focus_terms, is_meaningful_focus_term


MAX_FOCUS_SUGGESTIONS = 6

_FOCUS_TEMPLATE_GROUPS: list[tuple[set[str], tuple[str, ...]]] = [
    (
        {"agent", "agents", "agentic", "mcp"},
        (
            "AI agents",
            "LLM agents",
            "agent workflows",
            "autonomous agents",
            "multi-agent systems",
            "AI automation",
        ),
    ),
    (
        {"ai", "llm", "artificial intelligence"},
        (
            "AI applications",
            "AI tools",
            "AI automation",
            "LLM applications",
        ),
    ),
    (
        {"travel", "trip", "trips", "vacation", "vacations", "tourism"},
        (
            "travel planning",
            "budget travel",
            "family travel",
            "travel destinations",
            "travel tips",
            "digital nomad travel",
        ),
    ),
    (
        {"education", "learning", "classroom", "teacher", "student", "students"},
        (
            "student learning",
            "classroom learning",
            "teaching strategies",
            "education technology",
            "curriculum planning",
        ),
    ),
    (
        {"health", "wellness", "fitness", "nutrition", "medical"},
        (
            "healthy habits",
            "preventive health",
            "daily wellness",
            "health education",
            "health routines",
        ),
    ),
    (
        {"finance", "money", "investing", "investment", "budgeting", "personal finance"},
        (
            "personal finance",
            "budgeting strategies",
            "investment planning",
            "saving money",
            "financial goals",
        ),
    ),
    (
        {"business", "startup", "company", "leadership", "management"},
        (
            "business strategy",
            "team operations",
            "business growth",
            "management practices",
            "customer experience",
        ),
    ),
    (
        {"parenting", "baby", "babies", "infant", "infants", "newborn", "newborns"},
        (
            "parenting routines",
            "child development",
            "baby care",
            "family routines",
            "newborn care",
        ),
    ),
    (
        {"workflow", "automation", "operations", "ops", "process"},
        (
            "workflow automation",
            "process automation",
            "operational workflows",
            "automation platforms",
        ),
    ),
    (
        {"developer", "engineering", "software", "coding", "python", "dev"},
        (
            "developer tools",
            "software workflows",
            "engineering automation",
            "technical implementation",
        ),
    ),
    (
        {"business", "small business", "startup", "smb", "operations"},
        (
            "business automation",
            "team productivity",
            "operational efficiency",
            "small business AI",
        ),
    ),
    (
        {"baby", "babies", "infant", "infants", "newborn", "newborns", "sleep", "sleeping", "bedtime", "nap", "naps"},
        (
            "infant sleep",
            "baby bedtime",
            "sleep regression",
            "newborn sleep",
            "baby sleep schedule",
            "wake windows",
            "baby sleep training",
        ),
    ),
]

_COMBINATION_PATTERNS: list[tuple[set[str], set[str], tuple[str, ...]]] = [
    (
        {"ai", "llm", "mcp", "agent", "agents", "agentic"},
        {"education", "learning", "classroom", "teacher", "student", "students"},
        (
            "AI tutors",
            "AI learning tools",
            "education technology",
            "personalized learning",
            "classroom AI",
        ),
    ),
    (
        {"ai", "llm", "mcp", "agent", "agents", "agentic"},
        {"workflow", "automation", "operations", "ops", "process"},
        (
            "AI automation",
            "agent workflows",
            "workflow automation",
            "operational AI",
        ),
    ),
]

def generate_focus_suggestions(topic_name: str, existing_terms: Iterable[str] | None = None) -> list[str]:
    normalized_topic = " ".join(str(topic_name or "").strip().split())
    if not normalized_topic:
        return []

    existing_normalized = {str(term or "").casefold() for term in (existing_terms or []) if str(term or "").strip()}
    topic_lower = normalized_topic.casefold()
    topic_tokens = _tokenize(topic_lower)

    candidates = _generate_ai_focus_candidates(normalized_topic)
    if not candidates:
        candidates = []

        for required_a, required_b, suggestions in _COMBINATION_PATTERNS:
            if topic_tokens & required_a and topic_tokens & required_b:
                candidates.extend(suggestions)

        for keywords, suggestions in _FOCUS_TEMPLATE_GROUPS:
            if topic_tokens & keywords:
                candidates.extend(suggestions)

        if not candidates:
            candidates.extend(_build_generic_focus(topic_lower))

    cleaned_candidates = clean_focus_terms(candidates)
    suggestions: list[str] = []
    seen: set[str] = set()

    for suggestion in cleaned_candidates:
        lowered = suggestion.casefold()
        if lowered == topic_lower and len(topic_tokens) <= 1:
            continue
        if lowered in existing_normalized or lowered in seen:
            continue
        if not is_meaningful_focus_term(suggestion):
            continue
        seen.add(lowered)
        suggestions.append(suggestion)
        if len(suggestions) >= MAX_FOCUS_SUGGESTIONS:
            break

    return suggestions


def should_seed_focus_terms(topic_name: str, current_terms: Iterable[str] | None, *, focus_initialized: bool) -> bool:
    if focus_initialized:
        return False

    cleaned_terms = clean_focus_terms(list(current_terms or []))
    if not cleaned_terms:
        return True

    if len(cleaned_terms) == 1 and cleaned_terms[0].casefold() == str(topic_name or "").strip().casefold():
        return True

    return False


def _tokenize(value: str) -> set[str]:
    parts = [part for part in value.replace("/", " ").replace("-", " ").split() if part]
    token_set = set(parts)
    token_set.add(value)
    return token_set


def _build_generic_focus(topic_lower: str) -> tuple[str, ...]:
    topic_tokens = [part for part in topic_lower.replace("/", " ").replace("-", " ").split() if part]

    if {"baby", "babies", "infant", "infants", "newborn", "newborns"} & set(topic_tokens):
        return (
            "baby sleep",
            "infant sleep",
            "newborn sleep",
        )

    if "sleeping" in topic_tokens or "sleep" in topic_tokens:
        return (
            "sleep habits",
            "bedtime routines",
            "sleep schedule",
        )

    if len(topic_tokens) == 1:
        root = topic_tokens[0]
        return tuple(
            clean_focus_terms(
                [
                    f"{root} planning",
                    f"{root} tips",
                    f"{root} ideas",
                    f"{root} guide",
                    f"{root} strategies",
                ]
            )[:MAX_FOCUS_SUGGESTIONS]
        )

    normalized_variant = _normalize_topic_variant(topic_lower)
    anchored_terms = [
        normalized_variant,
        f"{normalized_variant} tips",
        f"{normalized_variant} planning",
        f"{normalized_variant} guide",
    ]
    return tuple(clean_focus_terms(anchored_terms)[:MAX_FOCUS_SUGGESTIONS])


def _normalize_topic_variant(topic_lower: str) -> str:
    replacements = {
        "sleeping": "sleep",
        "travelling": "travel",
        "traveling": "travel",
    }
    parts = [replacements.get(part, part) for part in topic_lower.split()]
    return " ".join(parts).strip()


def _generate_ai_focus_candidates(topic_name: str) -> list[str]:
    if _should_use_ai_focus_generation() is False:
        return []

    prompt = (
        "Generate 4 to 8 short focus phrases for a content discovery topic.\n"
        "Use only the current topic as context.\n"
        "Return a JSON object with a single key focus_terms containing an array of strings.\n"
        "Rules:\n"
        "- stay tightly grounded in the topic domain\n"
        "- prefer search-friendly phrases\n"
        "- avoid generic filler like best practices, trends, tools, workflows unless clearly topic-specific\n"
        "- do not repeat the topic unchanged unless necessary\n"
        f"Topic: {topic_name}"
    )

    try:
        response = OpenAIClient().generate_text(prompt, max_output_tokens=300, json_mode=True)
    except Exception:
        return []

    return _parse_ai_focus_response(response.text)


def _should_use_ai_focus_generation() -> bool:
    api_key = str(getattr(settings, "OPENAI_API_KEY", "") or "").strip()
    return bool(api_key) and api_key != "sk-your-key"


def _parse_ai_focus_response(response_text: str) -> list[str]:
    import json

    try:
        payload = json.loads(response_text)
    except Exception:
        return []

    raw_terms = payload.get("focus_terms", []) if isinstance(payload, dict) else []
    if not isinstance(raw_terms, list):
        return []
    return [str(term).strip() for term in raw_terms if str(term).strip()]
