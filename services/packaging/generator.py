"""First working LinkedIn packaging stage for the MVP."""
from __future__ import annotations

import json
import logging
import re
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

_HASHTAG_TOKEN_RE = re.compile(r"^#?[A-Za-z0-9][A-Za-z0-9_-]*$")
_HASHTAG_SPLIT_RE = re.compile(r"[\s,]+")
_HASHTAG_SENTENCE_PUNCT_RE = re.compile(r"[.!?;:]")
_HASHTAG_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "about",
    "for",
    "from",
    "in",
    "into",
    "is",
    "of",
    "on",
    "or",
    "the",
    "this",
    "that",
    "to",
    "with",
    "your",
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
    quality_gate: dict[str, Any] | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_reasons: list[str] | None = None
    repair_quality_gate: dict[str, Any] | None = None
    repair_prompt: str = ""
    repair_response_text: str = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt: str = ""
    post_brief_tokens: dict[str, int | None] | None = None
    brief_alignment: dict[str, Any] | None = None


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

    payload = _normalize_linkedin_post_payload(generation.payload)
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
        "quality_gate": generation.quality_gate or {},
        "repair_attempted": generation.repair_attempted,
        "repair_succeeded": generation.repair_succeeded,
        "repair_reasons": generation.repair_reasons or [],
        "repair_quality_gate": generation.repair_quality_gate or {},
        "repair_prompt": generation.repair_prompt,
        "repair_response_text": generation.repair_response_text,
        "post_brief": generation.post_brief,
        "post_brief_prompt": generation.post_brief_prompt,
        "post_brief_tokens": generation.post_brief_tokens,
        "brief_alignment": generation.brief_alignment or {},
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
    quality_gate: dict[str, Any] | None = None
    repair_attempted = False
    repair_succeeded = False
    repair_reasons: list[str] = []
    repair_quality_gate: dict[str, Any] | None = None
    repair_prompt = ""
    repair_response_text = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt = ""
    post_brief_tokens: dict[str, int | None] | None = None
    brief_alignment: dict[str, Any] | None = None
    prompt = build_post_prompt(digest, articles, profile)
    response_text = ""

    if _should_use_mock():
        payload = _build_mock_payload(digest, articles)
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        provider = "mock"
        is_mock = True
        fallback_reason = "OPENAI_API_KEY не задан или содержит placeholder."
    else:
        try:
            try:
                post_brief, post_brief_prompt, _brief_response_text, post_brief_tokens = _generate_post_brief_via_llm(
                    digest,
                    articles,
                    profile,
                )
            except Exception as exc:
                raise ContentPackageValidationError(
                    f"LinkedIn post brief generation/validation failed: {exc}"
                ) from exc

            payload, prompt, response_text, tokens = _generate_payload_via_llm(
                digest,
                articles,
                profile,
                post_brief=post_brief,
            )
            payload = _normalize_linkedin_post_payload(payload)
            estimated_cost = estimate_cost_usd(
                tokens.get("prompt_tokens") if tokens else None,
                tokens.get("completion_tokens") if tokens else None,
            )
            validate_content_package_payload(payload)
            quality_gate = _evaluate_linkedin_post_quality(payload)
            brief_alignment = _evaluate_post_brief_alignment(payload, post_brief)
            repair_reasons = _combined_repair_reasons(quality_gate, brief_alignment)

            if _quality_gate_requires_repair(quality_gate) or _brief_alignment_requires_repair(brief_alignment):
                repair_attempted = True
                repair_report = {
                    "status": "retry",
                    "reasons": repair_reasons,
                    "quality_gate": quality_gate,
                    "brief_alignment": brief_alignment,
                }
                repaired_payload, repair_prompt, repair_response_text, _repair_tokens = _repair_packaging_payload_via_llm(
                    digest,
                    articles,
                    profile,
                    weak_payload=payload,
                    quality_report=repair_report,
                    post_brief=post_brief,
                )
                repaired_payload = _normalize_linkedin_post_payload(repaired_payload)
                repaired_payload["post_text"] = _split_long_post_paragraphs(repaired_payload["post_text"])
                validate_content_package_payload(repaired_payload)
                repair_quality_gate = _evaluate_linkedin_post_quality(repaired_payload)
                repair_brief_alignment = _evaluate_post_brief_alignment(repaired_payload, post_brief)
                if _quality_gate_requires_repair(repair_quality_gate) or _brief_alignment_requires_repair(repair_brief_alignment):
                    raise ContentPackageValidationError(
                        "LinkedIn post quality repair did not resolve retry-trigger issues: "
                        f"{_combined_repair_reasons(repair_quality_gate, repair_brief_alignment)}"
                    )
                payload = repaired_payload
                brief_alignment = repair_brief_alignment
                response_text = repair_response_text
                repair_succeeded = True
            return PackagingGenerationResult(
                prompt=prompt,
                response_text=response_text,
                payload=payload,
                provider=provider,
                is_mock=is_mock,
                fallback_reason=fallback_reason,
                tokens=tokens,
                estimated_cost_usd=estimated_cost,
                quality_gate=quality_gate,
                repair_attempted=repair_attempted,
                repair_succeeded=repair_succeeded,
                repair_reasons=repair_reasons,
                repair_quality_gate=repair_quality_gate,
                repair_prompt=repair_prompt,
                repair_response_text=repair_response_text,
                post_brief=post_brief,
                post_brief_prompt=post_brief_prompt,
                post_brief_tokens=post_brief_tokens,
                brief_alignment=brief_alignment,
            )
        except Exception as exc:  # noqa: BLE001 - explicit fallback for the MVP stage
            payload = _build_mock_payload(digest, articles)
            if post_brief is not None:
                prompt = build_post_prompt(digest, articles, profile, post_brief=post_brief)
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
        quality_gate=quality_gate,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        repair_reasons=repair_reasons,
        repair_quality_gate=repair_quality_gate,
        repair_prompt=repair_prompt,
        repair_response_text=repair_response_text,
        post_brief=post_brief,
        post_brief_prompt=post_brief_prompt,
        post_brief_tokens=post_brief_tokens,
        brief_alignment=brief_alignment,
    )


def generate_post_from_articles(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mode 1: build one post from all article analyses."""
    prompt = build_post_prompt(digest, articles, author_profile, post_brief=post_brief)
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
                "bullets": ["No post draft articles available."],
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
    post_brief: dict[str, Any] | None = None,
) -> str:
    """Build prompt for single-post mode from digest articles."""
    return build_prompt(
        "linkedin/generate_post_from_articles.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        articles=_format_list_for_prompt(articles),
        post_brief=_format_list_for_prompt(post_brief or {}),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
        style_constraint_1=author_profile["style_constraints"][0],
        style_constraint_2=author_profile["style_constraints"][1],
        style_constraint_3=author_profile["style_constraints"][2],
    )


def build_post_brief_prompt(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> str:
    """Build prompt for an internal editorial brief from digest articles."""
    return build_prompt(
        "linkedin/generate_post_brief_from_articles.txt",
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


_POST_BRIEF_STRING_FIELDS = [
    "target_reader",
    "reader_pain_or_mistake",
    "hook_type",
    "sharp_claim",
    "credibility_basis",
    "tension",
    "pattern_interrupt",
    "human_angle",
    "practical_takeaway",
    "ending_reframe",
    "suggested_hook_direction",
    "avoid_angle",
]

_POST_BRIEF_HOOK_TYPES = {"personal_action", "reader_pain", "counterintuitive_fact"}


def _validate_post_brief_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize an internal LinkedIn post brief payload."""
    if not isinstance(payload, dict):
        raise ContentPackageValidationError("Post brief payload must be a JSON object.")

    normalized: dict[str, Any] = {}
    for field_name in _POST_BRIEF_STRING_FIELDS:
        if field_name not in payload:
            raise ContentPackageValidationError(f"Post brief payload is missing required field: {field_name}")
        value = str(payload.get(field_name) or "").strip()
        if not value:
            raise ContentPackageValidationError(f"Post brief field must be a non-empty string: {field_name}")
        normalized[field_name] = value

    if normalized["hook_type"] not in _POST_BRIEF_HOOK_TYPES:
        raise ContentPackageValidationError(
            "Post brief hook_type must be one of: personal_action, reader_pain, counterintuitive_fact."
        )

    if "evidence_points" not in payload:
        raise ContentPackageValidationError("Post brief payload is missing required field: evidence_points")

    raw_evidence_points = payload.get("evidence_points")
    if not isinstance(raw_evidence_points, list):
        raise ContentPackageValidationError("Post brief evidence_points must be a list.")

    evidence_points = [
        str(item).strip()
        for item in raw_evidence_points
        if isinstance(item, str) and str(item).strip()
    ]
    if len(evidence_points) < 2:
        raise ContentPackageValidationError("Post brief evidence_points must include at least 2 non-empty strings.")

    if "concrete_details" not in payload:
        raise ContentPackageValidationError("Post brief payload is missing required field: concrete_details")

    raw_concrete_details = payload.get("concrete_details")
    if not isinstance(raw_concrete_details, list):
        raise ContentPackageValidationError("Post brief concrete_details must be a list.")

    concrete_details = [
        str(item).strip()
        for item in raw_concrete_details
        if isinstance(item, str) and str(item).strip()
    ]

    return {
        "target_reader": normalized["target_reader"],
        "reader_pain_or_mistake": normalized["reader_pain_or_mistake"],
        "hook_type": normalized["hook_type"],
        "sharp_claim": normalized["sharp_claim"],
        "credibility_basis": normalized["credibility_basis"],
        "tension": normalized["tension"],
        "pattern_interrupt": normalized["pattern_interrupt"],
        "evidence_points": evidence_points[:4],
        "concrete_details": concrete_details[:6],
        "human_angle": normalized["human_angle"],
        "practical_takeaway": normalized["practical_takeaway"],
        "ending_reframe": normalized["ending_reframe"],
        "suggested_hook_direction": normalized["suggested_hook_direction"],
        "avoid_angle": normalized["avoid_angle"],
    }


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


def build_post_repair_prompt(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
    weak_payload: dict[str, Any],
    quality_report: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
) -> str:
    """Build prompt for one-pass quality repair of a structurally valid post payload."""
    return build_prompt(
        "linkedin/repair_post_quality.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        articles=_format_list_for_prompt(articles),
        post_brief=_format_list_for_prompt(post_brief or {}),
        weak_payload=_format_list_for_prompt(weak_payload),
        quality_reasons=_format_list_for_prompt(quality_report.get("reasons", [])),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
        style_constraint_1=author_profile["style_constraints"][0],
        style_constraint_2=author_profile["style_constraints"][1],
        style_constraint_3=author_profile["style_constraints"][2],
    )


def _generate_payload_via_llm(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    if not articles:
        prompt = build_post_prompt(digest, articles, author_profile, post_brief=post_brief)
        payload = _build_safe_fallback_post(digest)
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        return payload, prompt, response_text, None

    post_payload = generate_post_from_articles(digest, articles, author_profile, post_brief=post_brief)
    carousel_outline = generate_carousel_from_articles(digest, articles, author_profile)
    payload = {
        "post_text": post_payload["post_text"],
        "hook_variants": post_payload["hook_variants"],
        "cta_variants": post_payload["cta_variants"],
        "hashtags": post_payload["hashtags"],
        "carousel_outline": carousel_outline,
        "quality_checks": post_payload["quality_checks"],
    }
    prompt = build_post_prompt(digest, articles, author_profile, post_brief=post_brief)
    response_text = json.dumps(payload, ensure_ascii=False, indent=2)
    return payload, prompt, response_text, None


def _generate_post_brief_via_llm(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    prompt = build_post_brief_prompt(digest, articles, author_profile)
    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=700,
        json_mode=True,
    )
    response_text = response.text.strip()
    payload = _parse_json_response(response_text)
    post_brief = _validate_post_brief_payload(payload)
    return post_brief, prompt, response_text, response.usage


def _repair_packaging_payload_via_llm(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
    *,
    weak_payload: dict[str, Any],
    quality_report: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    prompt = build_post_repair_prompt(
        digest,
        articles,
        author_profile,
        weak_payload,
        quality_report,
        post_brief=post_brief,
    )
    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=900,
        json_mode=True,
    )
    payload = _parse_json_response(response.text.strip())
    return payload, prompt, response.text.strip(), response.usage


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


_BROAD_OPENING_PHRASES = [
    "in the landscape of",
    "in today's world",
    "as businesses",
    "the future of",
    "in the digital landscape",
]

_WEAK_GENERIC_OPENING_PHRASES = [
    "authentic storytelling is essential",
    "effective personal branding",
    "personal branding is essential",
    "many professionals struggle",
    "success in personal branding",
    "building a personal brand",
    "a strong personal brand",
]

_BANNED_LINKEDIN_PHRASES = [
    "resonate",
    "cohesive",
    "holistic",
    "systemic alignment",
    "professional authority",
    "elevate your brand",
    "unlock potential",
    "leverage",
    "landscape",
    "seamless",
    "paramount",
    "game changer",
]

_ARTICLE_RECAP_PHRASES = [
    "one article",
    "another article",
    "this article highlights",
    "the sources suggest",
]

_VAGUE_ABSTRACT_TERMS = [
    "authentic",
    "authenticity",
    "storytelling",
    "visibility",
    "growth",
    "journey",
    "narrative",
    "resilience",
    "trust",
    "engagement",
    "audience",
    "outcomes",
    "development",
    "personal brand",
    "brand to thrive",
    "polished outcomes",
    "true experiences",
    "open narrative",
]

_DIAGNOSTIC_PATTERNS = [
    "if your",
    "if people",
    "look at",
    "the test is",
    "a useful check",
    "check whether",
    "ask yourself",
]

_POST_TEXT_URL_RE = re.compile(r"(https?://|www\.)", re.IGNORECASE)

_POST_TEXT_CTA_PHRASES = [
    "what do you think?",
    "comment below",
    "let me know",
    "share your thoughts",
    "follow me",
    "follow for",
    "follow my",
    "follow this page",
    "subscribe",
]

_ALIGNMENT_STOPWORDS = {
    "a",
    "an",
    "and",
    "angle",
    "avoid",
    "about",
    "as",
    "acknowledging",
    "brand",
    "branding",
    "broad",
    "for",
    "from",
    "generalizations",
    "generic",
    "in",
    "is",
    "it",
    "not",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
    "without",
    "write",
}


def _evaluate_linkedin_post_quality(payload: dict[str, Any]) -> dict[str, Any]:
    post_text = str(payload.get("post_text") or "").strip()
    folded_text = post_text.casefold()
    first_line = _first_non_empty_line(post_text).casefold()
    reasons: list[str] = []
    warnings: list[str] = []

    for phrase in _BROAD_OPENING_PHRASES:
        if first_line.startswith(phrase):
            reasons.append(f"broad_opening:{phrase}")

    for phrase in _WEAK_GENERIC_OPENING_PHRASES:
        if first_line.startswith(phrase):
            reasons.append(f"weak_generic_opening:{phrase}")

    for phrase in _BANNED_LINKEDIN_PHRASES:
        if phrase in folded_text:
            reasons.append(f"banned_phrase:{phrase}")

    for phrase in _ARTICLE_RECAP_PHRASES:
        if phrase in folded_text:
            reasons.append(f"article_recap:{phrase}")

    diagnostic_patterns = _matching_phrases(folded_text, _DIAGNOSTIC_PATTERNS)
    vague_terms = _matching_phrases(folded_text, _VAGUE_ABSTRACT_TERMS)
    if len(vague_terms) >= 5:
        if diagnostic_patterns:
            warnings.append("vague_language_density")
        else:
            reasons.append("vague_language_density")

    if not diagnostic_patterns:
        warnings.append("missing_concrete_diagnostic")

    if post_text.endswith("?"):
        reasons.append("cta_question_ending")

    if 1200 < len(post_text) <= 1300:
        reasons.append("soft_length_limit")

    long_paragraphs = [
        index
        for index, paragraph in enumerate(_split_non_empty_paragraphs(post_text), start=1)
        if len(paragraph) > 450
    ]
    if long_paragraphs:
        reasons.append("long_paragraph")

    return {
        "status": "retry" if reasons else "pass",
        "reasons": reasons,
        "warnings": warnings,
    }


def _evaluate_post_brief_alignment(
    payload: dict[str, Any],
    post_brief: dict[str, Any] | None,
) -> dict[str, Any]:
    """Conservative deterministic checks that final post text follows the validated brief."""
    if not post_brief:
        return {
            "checked": False,
            "passed": True,
            "issues": [],
            "warnings": [],
            "reason": "missing_post_brief",
        }

    post_text = str(payload.get("post_text") or "").strip()
    normalized_post = _normalize_alignment_text(post_text)
    issues: list[str] = []
    warnings: list[str] = []

    if not post_text:
        issues.append("missing_post_text")

    if _POST_TEXT_URL_RE.search(post_text):
        issues.append("url_in_post_text")

    if post_text.endswith("?"):
        issues.append("post_text_ends_with_question")

    for phrase in _POST_TEXT_CTA_PHRASES:
        if phrase in normalized_post:
            issues.append(f"cta_phrase_in_post_text:{phrase}")

    avoid_angle = str(post_brief.get("avoid_angle") or "")
    if _contains_meaningful_fragment(normalized_post, avoid_angle, min_words=2):
        issues.append("avoid_angle_in_post_text")

    concrete_details = [
        str(item).strip()
        for item in post_brief.get("concrete_details", [])
        if isinstance(item, str) and str(item).strip()
    ]
    if concrete_details and not any(
        _contains_meaningful_fragment(normalized_post, detail, min_words=2)
        for detail in concrete_details
    ):
        issues.append("missing_concrete_detail")

    opening = _first_third(post_text)
    normalized_opening = _normalize_alignment_text(opening)
    if not any(
        _contains_meaningful_fragment(normalized_opening, post_brief.get(field_name, ""), min_words=3)
        for field_name in ["sharp_claim", "tension"]
    ):
        warnings.append("opening_may_not_reflect_brief")

    if not any(
        _contains_meaningful_fragment(normalized_post, post_brief.get(field_name, ""), min_words=3)
        for field_name in ["ending_reframe", "practical_takeaway"]
    ):
        warnings.append("ending_may_not_reflect_brief")

    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
    }


def _brief_alignment_requires_repair(report: dict[str, Any] | None) -> bool:
    return bool(report and report.get("checked") and report.get("issues"))


def _combined_repair_reasons(
    quality_report: dict[str, Any] | None,
    brief_alignment: dict[str, Any] | None,
) -> list[str]:
    reasons = list((quality_report or {}).get("reasons", []))
    reasons.extend(
        f"brief_alignment:{issue}"
        for issue in (brief_alignment or {}).get("issues", [])
    )
    return reasons


def _normalize_alignment_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9#?]+", " ", str(value).casefold())).strip()


def _meaningful_words(value: str) -> list[str]:
    return [
        word
        for word in _normalize_alignment_text(value).split()
        if len(word) > 2 and word not in _ALIGNMENT_STOPWORDS
    ]


def _contains_meaningful_fragment(text: str, value: Any, *, min_words: int = 3) -> bool:
    words = _meaningful_words(str(value or ""))
    if len(words) < min_words:
        return False

    normalized_value = " ".join(words)
    if normalized_value and normalized_value in text:
        return True

    window_size = min(len(words), max(min_words, 2))
    for index in range(0, len(words) - window_size + 1):
        fragment = " ".join(words[index : index + window_size])
        if fragment in text:
            return True
    return False


def _first_third(text: str) -> str:
    words = str(text or "").split()
    if not words:
        return ""
    end = max(1, len(words) // 3)
    return " ".join(words[:end])


def _matching_phrases(text: str, phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def _quality_gate_requires_repair(report: dict[str, Any]) -> bool:
    return str(report.get("status") or "") == "retry" and bool(report.get("reasons"))


def _first_non_empty_line(text: str) -> str:
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _split_non_empty_paragraphs(text: str) -> list[str]:
    return [
        " ".join(paragraph.split())
        for paragraph in re.split(r"\n\s*\n", str(text or ""))
        if paragraph.strip()
    ]


def _split_long_post_paragraphs(post_text: str, max_chars: int = 450) -> str:
    paragraphs = _split_non_empty_paragraphs(post_text)
    normalized_paragraphs: list[str] = []
    for paragraph in paragraphs:
        if len(paragraph) <= max_chars:
            normalized_paragraphs.append(paragraph)
            continue

        current = ""
        sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", paragraph) if sentence.strip()]
        for sentence in sentences or [paragraph]:
            candidate = f"{current} {sentence}".strip()
            if current and len(candidate) > max_chars:
                normalized_paragraphs.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            normalized_paragraphs.append(current)

    return "\n\n".join(normalized_paragraphs).strip()


def normalize_linkedin_hashtags(post_text: str) -> str:
    lines = str(post_text or "").splitlines()
    last_line_index = _find_last_non_empty_line_index(lines)
    if last_line_index is None:
        return str(post_text or "")

    trailing_tags = _extract_hashtag_line_tags(lines[last_line_index])
    if not trailing_tags:
        return str(post_text or "")

    lines[last_line_index] = " ".join(trailing_tags)
    return "\n".join(lines).strip()


def _normalize_linkedin_post_payload(payload: dict[str, Any]) -> dict[str, Any]:
    post_text = normalize_linkedin_hashtags(str(payload.get("post_text") or "").strip())
    trailing_tags = _extract_trailing_hashtags(post_text)
    hashtags = _normalize_hashtag_values(payload.get("hashtags", []))
    if trailing_tags:
        hashtags = _merge_hashtag_lists(trailing_tags, hashtags)
    if not hashtags:
        hashtags = [str(item).strip() for item in payload.get("hashtags", []) if str(item).strip()]

    return {
        "post_text": post_text,
        "hook_variants": payload.get("hook_variants", []),
        "cta_variants": payload.get("cta_variants", []),
        "hashtags": hashtags,
        "quality_checks": payload.get("quality_checks", {}),
    }


def _find_last_non_empty_line_index(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if str(lines[index]).strip():
            return index
    return None


def _extract_trailing_hashtags(post_text: str) -> list[str]:
    lines = str(post_text or "").splitlines()
    last_line_index = _find_last_non_empty_line_index(lines)
    if last_line_index is None:
        return []
    return _extract_hashtag_line_tags(lines[last_line_index])


def _extract_hashtag_line_tags(line: str) -> list[str]:
    raw_line = " ".join(str(line or "").strip().split())
    if not raw_line or _HASHTAG_SENTENCE_PUNCT_RE.search(raw_line):
        return []

    raw_tokens = [token for token in _HASHTAG_SPLIT_RE.split(raw_line) if token]
    if len(raw_tokens) < 2:
        return []
    if any(len(token.lstrip("#")) > 32 for token in raw_tokens):
        return []

    lowered_tokens = [token.lstrip("#").casefold() for token in raw_tokens]
    if sum(1 for token in lowered_tokens if token in _HASHTAG_STOPWORDS) >= 2:
        return []

    normalized = _normalize_hashtag_values(raw_tokens)
    if len(normalized) < 2:
        return []
    return normalized


def _normalize_hashtag_values(values: Any) -> list[str]:
    if isinstance(values, str):
        raw_values = [values]
    elif isinstance(values, list):
        raw_values = values
    else:
        raw_values = []

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for token in _HASHTAG_SPLIT_RE.split(str(raw_value or "").replace(",", " ")):
            cleaned = str(token or "").strip()
            if not cleaned:
                continue
            if not _HASHTAG_TOKEN_RE.match(cleaned):
                continue
            hashtag = f"#{cleaned.lstrip('#')}"
            dedupe_key = hashtag.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(hashtag)
    return normalized


def _merge_hashtag_lists(*hashtag_lists: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for hashtag_list in hashtag_lists:
        for hashtag in hashtag_list:
            cleaned = str(hashtag or "").strip()
            if not cleaned:
                continue
            dedupe_key = cleaned.casefold()
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            normalized.append(cleaned)
    return normalized


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
        return "No post draft articles were available."

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
        "post_text": f"{digest.title}\n\nNo post draft articles were available.",
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
                "bullets": ["No post draft articles were available."],
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
