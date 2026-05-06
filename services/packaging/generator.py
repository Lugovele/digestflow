"""First working LinkedIn packaging stage for the MVP."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.db import transaction

from apps.ai.client import OpenAIClient, estimate_cost_usd
from apps.digests.models import Digest
from apps.packaging.models import ContentPackage
from services.ai import build_prompt
from services.packaging.validators import (
    ContentPackageValidationError,
    validate_content_package_payload,
)

logger = logging.getLogger(__name__)

DEFAULT_AUTHOR_PROFILE = {
    "role": "AI Automation Specialist",
    "background": "Builds and improves workflow systems.",
    "focus": "workflow design, validation, reusable systems",
    "voice": "analytical",
    "style_constraints": [
        "avoid generic marketing language",
        "focus on systems, not tools",
        "connect facts into insights",
    ],
}


@dataclass(frozen=True)
class PackagingGenerationResult:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    provider: str
    is_mock: bool
    fallback_reason: str
    tokens: dict[str, int | None] | None
    estimated_cost_usd: float | None


def generate_content_package_for_digest(
    digest: Digest,
    author_profile: dict[str, Any] | None = None,
) -> tuple[ContentPackage, dict[str, Any]]:
    """Generate and save a ContentPackage for a ready Digest."""
    _debug(digest.run.id, "INFO", f"digest loaded -> {digest.id}")
    _debug(digest.run.id, "INFO", f"digest title -> {digest.title}")

    generation = _generate_packaging_payload(digest, author_profile=author_profile)

    _debug(digest.run.id, "INFO", f"provider -> {generation.provider}")
    _debug(digest.run.id, "INFO", f"is_mock -> {generation.is_mock}")
    if generation.fallback_reason:
        _debug(digest.run.id, "INFO", f"fallback_reason -> {generation.fallback_reason}")
    if generation.tokens and generation.tokens.get("total_tokens") is not None:
        _debug(digest.run.id, "INFO", f"tokens -> total: {generation.tokens['total_tokens']}")
    if generation.estimated_cost_usd is not None:
        _debug(digest.run.id, "INFO", f"estimated cost -> ${generation.estimated_cost_usd:.6f}")

    payload = generation.payload
    validation_report = _build_validation_report(payload)

    with transaction.atomic():
        ContentPackage.objects.filter(digest=digest).delete()
        content_package = ContentPackage.objects.create(
            digest=digest,
            post_text=payload["post_text"],
            hook_variants=payload["hook_variants"],
            cta_variants=payload["cta_variants"],
            hashtags=payload["hashtags"],
            carousel_outline=payload.get("carousel_outline", []),
            validation_report=validation_report,
        )

    _debug(digest.run.id, "OK", f"package saved -> {content_package.id}")

    debug_info = {
        "prompt": generation.prompt,
        "response_text": generation.response_text,
        "provider": generation.provider,
        "is_mock": generation.is_mock,
        "fallback_reason": generation.fallback_reason,
        "validation_report": validation_report,
        "tokens": generation.tokens,
        "estimated_cost_usd": generation.estimated_cost_usd,
    }
    return content_package, debug_info


def _generate_packaging_payload(
    digest: Digest,
    author_profile: dict[str, Any] | None = None,
) -> PackagingGenerationResult:
    profile = _normalize_author_profile(author_profile)
    articles = digest.get_articles()
    if not articles:
        logger.warning("[DigestRun %s] packaging received no digest articles", digest.run.id)

    fallback_reason = ""
    provider = "openai"
    is_mock = False
    tokens: dict[str, int | None] | None = None
    estimated_cost: float | None = None

    if _should_use_mock():
        payload = _build_mock_payload(digest, articles)
        prompt = build_post_prompt(digest, articles, profile)
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        provider = "mock"
        is_mock = True
        fallback_reason = "OPENAI_API_KEY не задан или содержит placeholder."
    else:
        try:
            payload, prompt, response_text, tokens = _generate_payload_via_llm(digest, articles, profile)
            estimated_cost = estimate_cost_usd(
                tokens.get("prompt_tokens") if tokens else None,
                tokens.get("completion_tokens") if tokens else None,
            )
            validate_content_package_payload(payload)
            return PackagingGenerationResult(
                prompt=prompt,
                response_text=response_text,
                payload=payload,
                provider=provider,
                is_mock=is_mock,
                fallback_reason=fallback_reason,
                tokens=tokens,
                estimated_cost_usd=estimated_cost,
            )
        except Exception as exc:  # noqa: BLE001 - explicit fallback for the MVP stage
            payload = _build_mock_payload(digest, articles)
            prompt = build_post_prompt(digest, articles, profile)
            response_text = json.dumps(payload, ensure_ascii=False, indent=2)
            provider = "mock"
            is_mock = True
            fallback_reason = (
                "Fallback на mock из-за ошибки реального AI call или невалидного JSON: "
                f"{exc}. Raw response: <empty>"
            )

    validate_content_package_payload(payload)

    return PackagingGenerationResult(
        prompt=prompt,
        response_text=response_text,
        payload=payload,
        provider=provider,
        is_mock=is_mock,
        fallback_reason=fallback_reason,
        tokens=tokens,
        estimated_cost_usd=estimated_cost,
    )


def generate_post_from_articles(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> dict[str, Any]:
    """Mode 1: build one post from all article analyses."""
    prompt = build_post_prompt(digest, articles, author_profile)
    if not articles:
        return _build_safe_fallback_post(digest)

    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=900,
        json_mode=True,
    )
    payload = _parse_json_response(response.text.strip())
    return payload


def generate_carousel_from_articles(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    """Mode 2: build one slide per article plus CTA slide."""
    prompt = build_carousel_prompt(digest, articles, author_profile)
    if not articles:
        return [
            {
                "slide": 1,
                "title": digest.title,
                "bullets": ["No digest articles available."],
            }
        ]

    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=900,
        json_mode=True,
    )
    payload = _parse_json_response(response.text.strip())
    slides = payload.get("slides", [])
    return _normalize_carousel_slides(slides)


def build_post_prompt(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> str:
    """Build prompt for single-post mode from digest articles."""
    return build_prompt(
        "linkedin/generate_post_from_articles.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        articles=_format_list_for_prompt(articles),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
        style_constraint_1=author_profile["style_constraints"][0],
        style_constraint_2=author_profile["style_constraints"][1],
        style_constraint_3=author_profile["style_constraints"][2],
    )


def build_carousel_prompt(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> str:
    """Build prompt for carousel mode from digest articles."""
    return build_prompt(
        "linkedin/generate_carousel_from_articles.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        articles=_format_list_for_prompt(articles),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
    )


def _generate_payload_via_llm(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    if not articles:
        prompt = build_post_prompt(digest, articles, author_profile)
        payload = _build_safe_fallback_post(digest)
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        return payload, prompt, response_text, None

    post_payload = generate_post_from_articles(digest, articles, author_profile)
    carousel_outline = generate_carousel_from_articles(digest, articles, author_profile)
    payload = {
        "post_text": post_payload["post_text"],
        "hook_variants": post_payload["hook_variants"],
        "cta_variants": post_payload["cta_variants"],
        "hashtags": post_payload["hashtags"],
        "carousel_outline": carousel_outline,
        "quality_checks": post_payload["quality_checks"],
    }
    prompt = build_post_prompt(digest, articles, author_profile)
    response_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return payload, prompt, response_text, None


def _build_validation_report(payload: dict[str, Any]) -> dict[str, Any]:
    quality_checks = payload.get("quality_checks", {})
    return {
        "status": "valid",
        "post_text_length": len(payload["post_text"]),
        "hook_variants_count": len(payload["hook_variants"]),
        "cta_variants_count": len(payload["cta_variants"]),
        "hashtags_count": len(payload["hashtags"]),
        "carousel_outline_count": len(payload.get("carousel_outline", [])),
        "quality_checks": quality_checks,
    }


def _format_list_for_prompt(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False, indent=2)


def _normalize_author_profile(author_profile: dict[str, Any] | None) -> dict[str, Any]:
    profile = dict(DEFAULT_AUTHOR_PROFILE)
    if not author_profile:
        return profile

    for key in ("role", "background", "focus", "voice"):
        value = author_profile.get(key)
        if value:
            profile[key] = str(value)

    constraints = author_profile.get("style_constraints")
    if isinstance(constraints, list):
        normalized = [str(item) for item in constraints if item]
        if normalized:
            while len(normalized) < 3:
                normalized.append(DEFAULT_AUTHOR_PROFILE["style_constraints"][len(normalized)])
            profile["style_constraints"] = normalized[:3]

    return profile


def _parse_json_response(response_text: str) -> dict[str, Any]:
    normalized = _extract_json_candidate(response_text)
    if not normalized:
        raise ContentPackageValidationError("Ответ модели пустой или не содержит JSON-объект.")

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise ContentPackageValidationError(
            f"Ответ модели не является валидным JSON: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ContentPackageValidationError("ContentPackage payload должен быть JSON-объектом.")
    return payload


def _normalize_carousel_slides(slides: Any) -> list[dict[str, Any]]:
    if not isinstance(slides, list):
        return []

    normalized = []
    for index, slide in enumerate(slides, start=1):
        if not isinstance(slide, dict):
            continue
        title = str(slide.get("title", "")).strip()
        content = str(slide.get("content", "")).strip()
        bullets = [line.strip() for line in content.split("\n") if line.strip()]
        if not title or not bullets:
            continue
        normalized.append(
            {
                "slide": index,
                "title": title,
                "bullets": bullets[:3],
            }
        )
    return normalized


def _should_use_mock() -> bool:
    api_key = settings.OPENAI_API_KEY.strip()
    return not api_key or api_key == "sk-your-key"


def _build_mock_payload(digest: Digest, articles: list[dict[str, Any]]) -> dict[str, Any]:
    if not articles:
        return _build_safe_fallback_post(digest)

    post_text = _build_mock_post_text_from_articles(articles)
    hooks = _build_mock_hooks_from_articles(articles)
    ctas = [
        "What still breaks?",
        "Where does it fail?",
        "Are you fixing or just speeding up?",
    ]
    hashtags = ["#AI", "#Workflows", "#Operations"]
    carousel_outline = _build_mock_carousel_from_articles(articles)

    return {
        "post_text": post_text,
        "hook_variants": hooks,
        "cta_variants": ctas,
        "hashtags": hashtags,
        "carousel_outline": carousel_outline,
        "quality_checks": {
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
            "linkedin_ready": True,
        },
    }


def _build_mock_post_text_from_articles(articles: list[dict[str, Any]]) -> str:
    if not articles:
        return "No digest articles were available."

    pattern = _build_cross_article_pattern(articles)
    top_article = articles[0]
    second_article = articles[1] if len(articles) > 1 else articles[0]
    body = [
        pattern,
        "",
        f"One article points to: {top_article['summary']}",
        f"Another article adds tension: {second_article['summary']}",
        "",
        "The pattern is not in one summary. It shows up across the set.",
    ]
    return "\n".join(body)[:1250]


def _build_mock_hooks_from_articles(articles: list[dict[str, Any]]) -> list[str]:
    if not articles:
        return ["AI makes things faster. But not better."] * 3

    pattern = _build_cross_article_pattern(articles)
    return [
        pattern,
        "The bottleneck does not disappear. It moves.",
        "Different articles. Same workflow problem.",
    ]


def _build_mock_carousel_from_articles(articles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    slides = []
    for index, article in enumerate(articles, start=1):
        bullets = [str(point).strip() for point in article.get("key_points", []) if str(point).strip()]
        slides.append(
            {
                "slide": index,
                "title": f"Article {index}",
                "bullets": [article.get("summary", "Failed to extract")] + bullets[:2],
            }
        )

    slides.append(
        {
            "slide": len(slides) + 1,
            "title": "CTA",
            "bullets": ["What still breaks?", "Where does it fail?"],
        }
    )
    return slides


def _build_cross_article_pattern(articles: list[dict[str, Any]]) -> str:
    content_types = {str(article.get("content_type", "unknown")).strip() for article in articles}
    if "tutorial" in content_types:
        return "The pattern is clearer when articles show the workflow step by step."
    if "opinion" in content_types:
        return "The strongest pattern is where different takes still point to the same tension."
    return "Across the articles, the same workflow tension keeps showing up."


def _build_safe_fallback_post(digest: Digest) -> dict[str, Any]:
    return {
        "post_text": f"{digest.title}\n\nNo digest articles were available.",
        "hook_variants": [
            "No article pattern was available.",
            "The source set was too thin to shape a post.",
            "There was not enough article structure to build from.",
        ],
        "cta_variants": [
            "What still breaks?",
            "Where does it fail?",
            "What data would help here?",
        ],
        "hashtags": ["#AI", "#Workflows"],
        "carousel_outline": [
            {
                "slide": 1,
                "title": digest.title,
                "bullets": ["No digest articles were available."],
            }
        ],
        "quality_checks": {
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": False,
            "linkedin_ready": True,
        },
    }


def _extract_json_candidate(response_text: str) -> str:
    text = response_text.strip()
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


def _debug(run_id: int, level: str, message: str) -> None:
    logger.info("[DigestRun %s] %s: %s", run_id, level, message)
