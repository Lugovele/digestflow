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
    repair_delta: dict[str, Any] | None = None
    repair_prompt: str = ""
    repair_response_text: str = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt: str = ""
    post_brief_tokens: dict[str, int | None] | None = None
    brief_alignment: dict[str, Any] | None = None
    post_mechanics: dict[str, Any] | None = None
    editorial_review: dict[str, Any] | None = None
    editorial_review_prompt: str = ""
    editorial_review_response_text: str = ""
    editorial_review_tokens: dict[str, int | None] | None = None
    editorial_review_error: str = ""
    editorial_review_used_for_repair: bool = False
    editorial_review_triggered_repair: bool = False
    editorial_repair_reasons: list[str] | None = None
    concrete_detail_diagnostics: dict[str, Any] | None = None
    banned_phrase_diagnostics: dict[str, Any] | None = None


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
        "repair_delta": generation.repair_delta or {},
        "repair_prompt": generation.repair_prompt,
        "repair_response_text": generation.repair_response_text,
        "post_brief": generation.post_brief,
        "post_brief_prompt": generation.post_brief_prompt,
        "post_brief_tokens": generation.post_brief_tokens,
        "brief_alignment": generation.brief_alignment or {},
        "post_mechanics": generation.post_mechanics or {},
        "editorial_review": generation.editorial_review or {},
        "editorial_review_prompt": generation.editorial_review_prompt,
        "editorial_review_response_text": generation.editorial_review_response_text,
        "editorial_review_tokens": generation.editorial_review_tokens,
        "editorial_review_error": generation.editorial_review_error,
        "editorial_review_used_for_repair": generation.editorial_review_used_for_repair,
        "editorial_review_triggered_repair": generation.editorial_review_triggered_repair,
        "editorial_repair_reasons": generation.editorial_repair_reasons or [],
        "editorial_review_repair_threshold": {"min_score": 7},
        "concrete_detail_diagnostics": generation.concrete_detail_diagnostics or {},
        "banned_phrase_diagnostics": generation.banned_phrase_diagnostics or {},
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
    repair_delta: dict[str, Any] | None = None
    repair_prompt = ""
    repair_response_text = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt = ""
    post_brief_tokens: dict[str, int | None] | None = None
    brief_alignment: dict[str, Any] | None = None
    post_mechanics: dict[str, Any] | None = None
    editorial_review: dict[str, Any] | None = None
    editorial_review_prompt = ""
    editorial_review_response_text = ""
    editorial_review_tokens: dict[str, int | None] | None = None
    editorial_review_error = ""
    editorial_review_used_for_repair = False
    editorial_review_triggered_repair = False
    editorial_repair_reasons: list[str] = []
    concrete_detail_diagnostics: dict[str, Any] | None = None
    banned_phrase_diagnostics: dict[str, Any] | None = None
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
            repairable_payload_issues = _collect_repairable_payload_issues(payload)
            if not repairable_payload_issues:
                validate_content_package_payload(payload)
            quality_gate = _evaluate_linkedin_post_quality(payload)
            brief_alignment = _evaluate_post_brief_alignment(payload, post_brief)
            post_mechanics = _evaluate_linkedin_post_mechanics(payload, post_brief)
            concrete_detail_diagnostics = _build_concrete_detail_diagnostics(
                post_brief,
                payload,
                initial_alignment=brief_alignment,
            )
            repair_reasons = repairable_payload_issues + _combined_repair_reasons(
                quality_gate,
                brief_alignment,
                post_mechanics,
            )
            banned_phrase_diagnostics = _build_banned_phrase_diagnostics(
                repair_reasons,
                payload,
            )

            deterministic_repair_required = (
                repairable_payload_issues
                or _quality_gate_requires_repair(quality_gate)
                or _brief_alignment_requires_repair(brief_alignment)
                or _post_mechanics_requires_repair(post_mechanics)
            )
            if deterministic_repair_required:
                repair_attempted = True
                if getattr(settings, "PACKAGING_EDITORIAL_REVIEW_ENABLED", True):
                    try:
                        (
                            editorial_review,
                            editorial_review_prompt,
                            editorial_review_response_text,
                            editorial_review_tokens,
                        ) = _generate_editorial_review_via_llm(
                            digest,
                            payload,
                            profile,
                            post_brief,
                            quality_gate,
                            brief_alignment,
                            post_mechanics,
                            repair_delta=None,
                            repair_reasons=repair_reasons,
                        )
                        editorial_review_used_for_repair = True
                    except Exception as exc:  # noqa: BLE001 - editorial guidance is diagnostic-only
                        editorial_review_error = str(exc)
                repair_report = {
                    "status": "retry",
                    "reasons": repair_reasons,
                    "quality_gate": quality_gate,
                    "brief_alignment": brief_alignment,
                    "post_mechanics": post_mechanics,
                }
                repaired_payload, repair_prompt, repair_response_text, _repair_tokens = _repair_packaging_payload_via_llm(
                    digest,
                    articles,
                    profile,
                    weak_payload=payload,
                    quality_report=repair_report,
                    post_brief=post_brief,
                    editorial_review=editorial_review,
                    editorial_review_error=editorial_review_error,
                )
                repaired_payload = _normalize_linkedin_post_payload(repaired_payload)
                repaired_payload["post_text"] = _split_long_post_paragraphs(repaired_payload["post_text"])
                validate_content_package_payload(repaired_payload)
                repair_quality_gate = _evaluate_linkedin_post_quality(repaired_payload)
                repair_brief_alignment = _evaluate_post_brief_alignment(repaired_payload, post_brief)
                repair_post_mechanics = _evaluate_linkedin_post_mechanics(repaired_payload, post_brief)
                repair_delta = _evaluate_repair_rewrite_delta(payload, repaired_payload, repair_reasons)
                concrete_detail_diagnostics = _build_concrete_detail_diagnostics(
                    post_brief,
                    payload,
                    initial_alignment=brief_alignment,
                    repaired_payload=repaired_payload,
                    repair_alignment=repair_brief_alignment,
                    repair_attempted=True,
                )
                banned_phrase_diagnostics = _build_banned_phrase_diagnostics(
                    repair_reasons,
                    payload,
                    repaired_payload=repaired_payload,
                    repair_attempted=True,
                )
                if (
                    _quality_gate_requires_repair(repair_quality_gate)
                    or _brief_alignment_requires_repair(repair_brief_alignment)
                    or _post_mechanics_requires_repair(repair_post_mechanics)
                    or _repair_delta_requires_failure(repair_delta)
                ):
                    unresolved_reasons = _combined_repair_reasons(
                        repair_quality_gate,
                        repair_brief_alignment,
                        repair_post_mechanics,
                    )
                    unresolved_reasons.extend(
                        f"repair_delta:{issue}"
                        for issue in repair_delta.get("issues", [])
                    )
                    raise ContentPackageValidationError(
                        "LinkedIn post quality repair did not resolve retry-trigger issues: "
                        f"{unresolved_reasons}"
                    )
                payload = repaired_payload
                brief_alignment = repair_brief_alignment
                post_mechanics = repair_post_mechanics
                response_text = repair_response_text
                repair_succeeded = True
            if getattr(settings, "PACKAGING_EDITORIAL_REVIEW_ENABLED", True) and not repair_attempted:
                try:
                    (
                        editorial_review,
                        editorial_review_prompt,
                        editorial_review_response_text,
                        editorial_review_tokens,
                    ) = _generate_editorial_review_via_llm(
                        digest,
                        payload,
                        profile,
                        post_brief,
                        quality_gate,
                        brief_alignment,
                        post_mechanics,
                        repair_delta=repair_delta,
                        repair_reasons=repair_reasons,
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostic-only review must not block saving
                    editorial_review_error = str(exc)
                if _editorial_review_requires_repair(editorial_review):
                    editorial_review_triggered_repair = True
                    editorial_review_used_for_repair = True
                    editorial_repair_reasons = _editorial_review_repair_reasons(editorial_review)
                    repair_reasons = editorial_repair_reasons
                    repair_attempted = True
                    repair_report = {
                        "status": "retry",
                        "reasons": repair_reasons,
                        "quality_gate": quality_gate,
                        "brief_alignment": brief_alignment,
                        "post_mechanics": post_mechanics,
                        "editorial_review": editorial_review,
                    }
                    repaired_payload, repair_prompt, repair_response_text, _repair_tokens = _repair_packaging_payload_via_llm(
                        digest,
                        articles,
                        profile,
                        weak_payload=payload,
                        quality_report=repair_report,
                        post_brief=post_brief,
                        editorial_review=editorial_review,
                        editorial_review_error=editorial_review_error,
                    )
                    repaired_payload = _normalize_linkedin_post_payload(repaired_payload)
                    repaired_payload["post_text"] = _split_long_post_paragraphs(repaired_payload["post_text"])
                    validate_content_package_payload(repaired_payload)
                    repair_quality_gate = _evaluate_linkedin_post_quality(repaired_payload)
                    repair_brief_alignment = _evaluate_post_brief_alignment(repaired_payload, post_brief)
                    repair_post_mechanics = _evaluate_linkedin_post_mechanics(repaired_payload, post_brief)
                    repair_delta = _evaluate_repair_rewrite_delta(payload, repaired_payload, repair_reasons)
                    concrete_detail_diagnostics = _build_concrete_detail_diagnostics(
                        post_brief,
                        payload,
                        initial_alignment=brief_alignment,
                        repaired_payload=repaired_payload,
                        repair_alignment=repair_brief_alignment,
                        repair_attempted=True,
                    )
                    banned_phrase_diagnostics = _build_banned_phrase_diagnostics(
                        repair_reasons,
                        payload,
                        repaired_payload=repaired_payload,
                        repair_attempted=True,
                    )
                    if (
                        _quality_gate_requires_repair(repair_quality_gate)
                        or _brief_alignment_requires_repair(repair_brief_alignment)
                        or _post_mechanics_requires_repair(repair_post_mechanics)
                        or _repair_delta_requires_failure(repair_delta)
                    ):
                        unresolved_reasons = _combined_repair_reasons(
                            repair_quality_gate,
                            repair_brief_alignment,
                            repair_post_mechanics,
                        )
                        unresolved_reasons.extend(
                            f"repair_delta:{issue}"
                            for issue in repair_delta.get("issues", [])
                        )
                        raise ContentPackageValidationError(
                            "LinkedIn post editorial repair did not produce a valid package: "
                            f"{unresolved_reasons}"
                        )
                    payload = repaired_payload
                    brief_alignment = repair_brief_alignment
                    post_mechanics = repair_post_mechanics
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
                repair_delta=repair_delta,
                repair_prompt=repair_prompt,
                repair_response_text=repair_response_text,
                post_brief=post_brief,
                post_brief_prompt=post_brief_prompt,
                post_brief_tokens=post_brief_tokens,
                brief_alignment=brief_alignment,
                post_mechanics=post_mechanics,
                editorial_review=editorial_review,
                editorial_review_prompt=editorial_review_prompt,
                editorial_review_response_text=editorial_review_response_text,
                editorial_review_tokens=editorial_review_tokens,
                editorial_review_error=editorial_review_error,
                editorial_review_used_for_repair=editorial_review_used_for_repair,
                editorial_review_triggered_repair=editorial_review_triggered_repair,
                editorial_repair_reasons=editorial_repair_reasons,
                concrete_detail_diagnostics=concrete_detail_diagnostics,
                banned_phrase_diagnostics=banned_phrase_diagnostics,
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
        repair_delta=repair_delta,
        repair_prompt=repair_prompt,
        repair_response_text=repair_response_text,
        post_brief=post_brief,
        post_brief_prompt=post_brief_prompt,
        post_brief_tokens=post_brief_tokens,
        brief_alignment=brief_alignment,
        post_mechanics=post_mechanics,
        editorial_review=editorial_review,
        editorial_review_prompt=editorial_review_prompt,
        editorial_review_response_text=editorial_review_response_text,
        editorial_review_tokens=editorial_review_tokens,
        editorial_review_error=editorial_review_error,
        editorial_review_used_for_repair=editorial_review_used_for_repair,
        editorial_review_triggered_repair=editorial_review_triggered_repair,
        editorial_repair_reasons=editorial_repair_reasons,
        concrete_detail_diagnostics=concrete_detail_diagnostics,
        banned_phrase_diagnostics=banned_phrase_diagnostics,
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
    editorial_review: dict[str, Any] | None = None,
    editorial_review_error: str = "",
) -> str:
    """Build prompt for one-pass quality repair of a structurally valid post payload."""
    quality_reasons = list(quality_report.get("reasons", []))
    blocked_phrases = _extract_banned_phrases_from_repair_reasons(quality_reasons)
    editorial_review_payload = editorial_review or {}
    return build_prompt(
        "linkedin/repair_post_quality.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        articles=_format_list_for_prompt(articles),
        post_brief=_format_list_for_prompt(post_brief or {}),
        editorial_review=_format_list_for_prompt(editorial_review_payload),
        editorial_review_issues=_format_list_for_prompt(editorial_review_payload.get("issues", [])),
        editorial_repair_instructions=_format_list_for_prompt(
            editorial_review_payload.get("repair_instructions", [])
        ),
        editorial_review_score=editorial_review_payload.get("score", ""),
        editorial_review_error=editorial_review_error,
        weak_payload=_format_list_for_prompt(weak_payload),
        quality_reasons=_format_list_for_prompt(quality_reasons),
        blocked_phrases=_format_list_for_prompt(blocked_phrases),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
        style_constraint_1=author_profile["style_constraints"][0],
        style_constraint_2=author_profile["style_constraints"][1],
        style_constraint_3=author_profile["style_constraints"][2],
    )


def build_editorial_review_prompt(
    digest: Digest,
    payload: dict[str, Any],
    author_profile: dict[str, Any],
    post_brief: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
    brief_alignment: dict[str, Any] | None,
    post_mechanics: dict[str, Any] | None,
    repair_delta: dict[str, Any] | None = None,
    repair_reasons: list[str] | None = None,
) -> str:
    """Build prompt for diagnostic-only editorial review of the final post package."""
    return build_prompt(
        "linkedin/review_post_editorial_quality.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        payload=_format_list_for_prompt(payload),
        post_brief=_format_list_for_prompt(post_brief or {}),
        quality_gate=_format_list_for_prompt(quality_gate or {}),
        brief_alignment=_format_list_for_prompt(brief_alignment or {}),
        post_mechanics=_format_list_for_prompt(post_mechanics or {}),
        repair_delta=_format_list_for_prompt(repair_delta or {}),
        repair_reasons=_format_list_for_prompt(repair_reasons or []),
        author_role=author_profile["role"],
        author_background=author_profile["background"],
        author_focus=author_profile["focus"],
        author_voice=author_profile["voice"],
        style_constraint_1=author_profile["style_constraints"][0],
        style_constraint_2=author_profile["style_constraints"][1],
        style_constraint_3=author_profile["style_constraints"][2],
    )


_EDITORIAL_REVIEW_ALLOWED_ISSUES = {
    "too_generic",
    "weak_hook",
    "low_reader_value",
    "not_enough_point_of_view",
    "too_abstract",
    "weak_ending",
    "not_linkedin_native",
    "sounds_like_corporate_blog",
    "missing_practical_diagnostic",
    "unclear_reader_value",
}


def _validate_editorial_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize diagnostic-only editorial review payload."""
    if not isinstance(payload, dict):
        raise ContentPackageValidationError("Editorial review payload must be a JSON object.")

    if "passed" not in payload:
        raise ContentPackageValidationError("Editorial review payload is missing required field: passed")
    if not isinstance(payload.get("passed"), bool):
        raise ContentPackageValidationError("Editorial review passed must be a boolean.")

    if "score" not in payload:
        raise ContentPackageValidationError("Editorial review payload is missing required field: score")
    score = payload.get("score")
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ContentPackageValidationError("Editorial review score must be a number from 1 to 10.")
    if score < 1 or score > 10:
        raise ContentPackageValidationError("Editorial review score must be from 1 to 10.")

    normalized_lists: dict[str, list[str]] = {}
    for field_name in ["issues", "strengths", "repair_instructions"]:
        if field_name not in payload:
            raise ContentPackageValidationError(
                f"Editorial review payload is missing required field: {field_name}"
            )
        raw_items = payload.get(field_name)
        if not isinstance(raw_items, list):
            raise ContentPackageValidationError(f"Editorial review {field_name} must be a list.")
        normalized_lists[field_name] = [
            str(item).strip()
            for item in raw_items
            if isinstance(item, str) and str(item).strip()
        ]

    normalized_lists["issues"] = [
        issue
        for issue in normalized_lists["issues"]
        if issue in _EDITORIAL_REVIEW_ALLOWED_ISSUES
    ]

    return {
        "passed": payload["passed"],
        "score": score,
        "issues": normalized_lists["issues"],
        "strengths": normalized_lists["strengths"],
        "repair_instructions": normalized_lists["repair_instructions"],
    }


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


def _generate_editorial_review_via_llm(
    digest: Digest,
    payload: dict[str, Any],
    author_profile: dict[str, Any],
    post_brief: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
    brief_alignment: dict[str, Any] | None,
    post_mechanics: dict[str, Any] | None,
    repair_delta: dict[str, Any] | None = None,
    repair_reasons: list[str] | None = None,
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    prompt = build_editorial_review_prompt(
        digest,
        payload,
        author_profile,
        post_brief,
        quality_gate,
        brief_alignment,
        post_mechanics,
        repair_delta=repair_delta,
        repair_reasons=repair_reasons,
    )
    response = OpenAIClient().generate_text(
        prompt=prompt,
        max_output_tokens=500,
        json_mode=True,
    )
    response_text = response.text.strip()
    review_payload = _parse_json_response(response_text)
    editorial_review = _validate_editorial_review_payload(review_payload)
    return editorial_review, prompt, response_text, response.usage


def _repair_packaging_payload_via_llm(
    digest: Digest,
    articles: list[dict[str, Any]],
    author_profile: dict[str, Any],
    *,
    weak_payload: dict[str, Any],
    quality_report: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
    editorial_review: dict[str, Any] | None = None,
    editorial_review_error: str = "",
) -> tuple[dict[str, Any], str, str, dict[str, int | None] | None]:
    prompt = build_post_repair_prompt(
        digest,
        articles,
        author_profile,
        weak_payload,
        quality_report,
        post_brief=post_brief,
        editorial_review=editorial_review,
        editorial_review_error=editorial_review_error,
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

_MECHANICS_GENERIC_OPENINGS = [
    "in today's world",
    "in the digital landscape",
    "as businesses",
    "many professionals",
    "personal branding is essential",
    "building a personal brand",
    "effective personal branding",
]

_PATTERN_INTERRUPT_SIGNALS = [
    "but",
    "except",
    "the problem is",
    "the mistake is",
    "the real issue",
    "what changes",
    "what most people miss",
    "the counterintuitive part",
    "the uncomfortable part",
    "not because",
]

_SPECIFICITY_OPERATIONAL_PHRASES = [
    "handoff",
    "workflow",
    "validation",
    "source",
    "metric",
    "review",
    "decision",
    "constraint",
    "cost",
]

_MECHANICS_GENERIC_TERMS = [
    "authentic",
    "resonate",
    "elevate",
    "leverage",
    "holistic",
    "seamless",
    "landscape",
    "essential",
    "unlock",
    "potential",
    "powerful",
    "meaningful",
]

_WEAK_ENDING_PHRASES = [
    "start today",
    "take action",
    "build your brand",
    "unlock your potential",
    "elevate your brand",
    "make it happen",
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

_AVOID_ANGLE_BROAD_TOPIC_WORDS = {
    "ai",
    "automation",
    "workflow",
    "linkedin",
    "content",
    "strategy",
    "personal",
    "branding",
    "brand",
    "post",
    "posts",
    "audience",
    "business",
    "professional",
    "professionals",
    "system",
    "systems",
    "tool",
    "tools",
}

_AVOID_ANGLE_LOW_SIGNAL_WORDS = {
    "advice",
    "angle",
    "avoid",
    "broad",
    "generic",
    "generalizations",
}

_CONCRETE_DETAIL_LOW_SIGNAL_WORDS = {
    "advice",
    "angle",
    "approach",
    "brand",
    "branding",
    "content",
    "effective",
    "generic",
    "personal",
    "professional",
    "professionals",
    "strategy",
    "strategies",
    "useful",
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


def _collect_repairable_payload_issues(payload: dict[str, Any]) -> list[str]:
    post_text = (payload or {}).get("post_text")
    if isinstance(post_text, str) and len(post_text) > 1300:
        return ["post_text_too_long"]
    return []


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
    avoid_angle_match = _find_avoid_angle_match(post_text, avoid_angle)
    details: dict[str, Any] = {}
    if avoid_angle_match:
        issues.append("avoid_angle_in_post_text")
        details["avoid_angle_match"] = avoid_angle_match

    concrete_details = [
        str(item).strip()
        for item in post_brief.get("concrete_details", [])
        if isinstance(item, str) and str(item).strip()
    ]
    concrete_detail_match = _find_concrete_detail_match(post_text, concrete_details)
    if concrete_details and not concrete_detail_match:
        issues.append("missing_concrete_detail")
    elif concrete_detail_match:
        details["concrete_detail_match"] = concrete_detail_match

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
        "details": details,
    }


def _build_concrete_detail_diagnostics(
    post_brief: dict[str, Any] | None,
    initial_payload: dict[str, Any] | None,
    *,
    initial_alignment: dict[str, Any] | None = None,
    repaired_payload: dict[str, Any] | None = None,
    repair_alignment: dict[str, Any] | None = None,
    repair_attempted: bool = False,
) -> dict[str, Any]:
    """Build debug-only diagnostics for concrete detail matching."""
    if not post_brief:
        return {}

    required_details = [
        str(item).strip()
        for item in post_brief.get("concrete_details", [])
        if isinstance(item, str) and str(item).strip()
    ]
    if not required_details:
        return {
            "required_details": [],
            "initial_match": None,
            "repair_match": None,
            "missing_after_repair": False,
        }

    initial_text = str((initial_payload or {}).get("post_text") or "").strip()
    repair_text = str((repaired_payload or {}).get("post_text") or "").strip()
    initial_match = _find_concrete_detail_match(initial_text, required_details)
    repair_match = _find_concrete_detail_match(repair_text, required_details) if repaired_payload else None

    missing_after_repair = False
    if repair_attempted:
        if repair_alignment is not None:
            missing_after_repair = "missing_concrete_detail" in repair_alignment.get("issues", [])
        else:
            missing_after_repair = repair_match is None

    return {
        "required_details": required_details,
        "initial_match": initial_match,
        "repair_match": repair_match,
        "missing_after_repair": missing_after_repair,
        "initial_missing": "missing_concrete_detail" in (initial_alignment or {}).get("issues", []),
        "post_text_excerpt": _debug_text_excerpt(initial_text),
        "repair_text_excerpt": _debug_text_excerpt(repair_text) if repaired_payload else "",
    }


def _build_banned_phrase_diagnostics(
    repair_reasons: list[str],
    initial_payload: dict[str, Any] | None,
    *,
    repaired_payload: dict[str, Any] | None = None,
    repair_attempted: bool = False,
) -> dict[str, Any]:
    """Build debug-only diagnostics for banned phrase repair regressions."""
    banned_phrases = _extract_banned_phrases_from_repair_reasons(repair_reasons)
    if not banned_phrases:
        return {
            "banned_phrases": [],
            "initial_matches": [],
            "repair_matches": [],
            "regressed_after_repair": False,
        }

    initial_matches = _find_banned_phrase_payload_matches(initial_payload or {}, banned_phrases)
    repair_matches = (
        _find_banned_phrase_payload_matches(repaired_payload or {}, banned_phrases)
        if repaired_payload is not None
        else []
    )

    return {
        "banned_phrases": banned_phrases,
        "initial_matches": initial_matches,
        "repair_matches": repair_matches,
        "regressed_after_repair": bool(repair_attempted and repair_matches),
    }


def _find_banned_phrase_payload_matches(payload: Any, banned_phrases: list[str]) -> list[dict[str, str]]:
    matches: list[dict[str, str]] = []

    def visit(value: Any, path: str) -> None:
        if isinstance(value, str):
            folded_value = value.casefold()
            for phrase in banned_phrases:
                if phrase.casefold() in folded_value:
                    matches.append(
                        {
                            "phrase": phrase,
                            "field": path,
                            "matched_text": phrase,
                            "excerpt": _debug_text_excerpt(value),
                        }
                    )
        elif isinstance(value, list):
            for index, item in enumerate(value):
                visit(item, f"{path}[{index}]")
        elif isinstance(value, dict):
            for key, item in value.items():
                next_path = str(key) if not path else f"{path}.{key}"
                visit(item, next_path)

    visit(payload, "")
    return matches


def _debug_text_excerpt(text: str, limit: int = 240) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}..."


def _evaluate_linkedin_post_mechanics(
    payload: dict[str, Any],
    post_brief: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate deterministic LinkedIn-native mechanics without judging strategy."""
    post_text = str(payload.get("post_text") or "").strip()
    normalized_post = _normalize_alignment_text(post_text)
    first_line = _first_non_empty_line(post_text)
    normalized_first_line = _normalize_alignment_text(first_line)
    first_line_word_count = len(first_line.split())
    issues: list[str] = []
    warnings: list[str] = []

    if not post_text:
        issues.append("missing_post_text")

    if _POST_TEXT_URL_RE.search(post_text):
        issues.append("url_in_post_text")

    if post_text.endswith("?"):
        issues.append("post_text_ends_with_question")

    if any(phrase in normalized_post for phrase in _POST_TEXT_CTA_PHRASES):
        issues.append("cta_in_post_text")

    if any(
        normalized_first_line.startswith(_normalize_alignment_text(phrase))
        for phrase in _MECHANICS_GENERIC_OPENINGS
    ):
        issues.append("generic_opening")

    if post_text and first_line_word_count < 5:
        warnings.append("hook_may_be_too_short")
    if first_line_word_count > 18:
        warnings.append("hook_may_be_too_long")

    has_pattern_interrupt_signal = any(
        signal in normalized_post for signal in _PATTERN_INTERRUPT_SIGNALS
    )
    if not has_pattern_interrupt_signal:
        warnings.append("missing_pattern_interrupt_signal")

    concrete_detail_count = _count_concrete_detail_signals(post_text, post_brief)
    if concrete_detail_count == 0:
        warnings.append("low_specificity")

    if _has_weak_ending(post_text):
        warnings.append("weak_ending")

    generic_language_count = sum(
        1 for term in _MECHANICS_GENERIC_TERMS if term in normalized_post
    )
    if generic_language_count >= 3:
        warnings.append("high_generic_language_density")

    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "signals": {
            "first_line_word_count": first_line_word_count,
            "has_pattern_interrupt_signal": has_pattern_interrupt_signal,
            "concrete_detail_count": concrete_detail_count,
            "generic_language_count": generic_language_count,
        },
    }


def _brief_alignment_requires_repair(report: dict[str, Any] | None) -> bool:
    return bool(report and report.get("checked") and report.get("issues"))


def _post_mechanics_requires_repair(report: dict[str, Any] | None) -> bool:
    return bool(report and report.get("checked") and report.get("issues"))


def _repair_delta_requires_failure(report: dict[str, Any] | None) -> bool:
    return bool(report and report.get("checked") and report.get("issues"))


def _combined_repair_reasons(
    quality_report: dict[str, Any] | None,
    brief_alignment: dict[str, Any] | None,
    post_mechanics: dict[str, Any] | None = None,
) -> list[str]:
    reasons = list((quality_report or {}).get("reasons", []))
    reasons.extend(
        f"brief_alignment:{issue}"
        for issue in (brief_alignment or {}).get("issues", [])
    )
    reasons.extend(
        f"post_mechanics:{issue}"
        for issue in (post_mechanics or {}).get("issues", [])
    )
    return reasons


def _evaluate_repair_rewrite_delta(
    weak_payload: dict[str, Any],
    repaired_payload: dict[str, Any],
    repair_reasons: list[str],
) -> dict[str, Any]:
    weak_text = str((weak_payload or {}).get("post_text") or "").strip()
    repaired_text = str((repaired_payload or {}).get("post_text") or "").strip()
    weak_tokens = _rewrite_delta_tokens(weak_text)
    repaired_tokens = _rewrite_delta_tokens(repaired_text)
    weak_sentences = _rewrite_delta_sentences(weak_text)
    repaired_sentences = _rewrite_delta_sentences(repaired_text)
    weak_sentence_count = len(weak_sentences)
    repaired_sentence_count = len(repaired_sentences)
    shared_sentence_count = len(set(weak_sentences) & set(repaired_sentences))
    shared_sentence_ratio = (
        shared_sentence_count / weak_sentence_count if weak_sentence_count else 0.0
    )
    shared_bigram_ratio = _shared_bigram_ratio(weak_tokens, repaired_tokens)
    issues: list[str] = []
    warnings: list[str] = []

    if not weak_text or not repaired_text:
        issues.append("missing_repair_text")
    elif _normalize_rewrite_delta_text(weak_text) == _normalize_rewrite_delta_text(repaired_text):
        issues.append("repair_text_too_similar")
    elif weak_sentence_count >= 3 and shared_sentence_ratio >= 0.6:
        issues.append("repair_text_too_similar")
    elif len(weak_tokens) >= 40 and shared_bigram_ratio >= 0.75:
        issues.append("repair_text_too_similar")
    elif len(weak_tokens) >= 25 and shared_bigram_ratio >= 0.6:
        warnings.append("repair_text_overlap_high")

    return {
        "checked": True,
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "signals": {
            "weak_word_count": len(weak_tokens),
            "repaired_word_count": len(repaired_tokens),
            "shared_sentence_count": shared_sentence_count,
            "weak_sentence_count": weak_sentence_count,
            "repaired_sentence_count": repaired_sentence_count,
            "shared_sentence_ratio": round(shared_sentence_ratio, 4),
            "shared_bigram_ratio": round(shared_bigram_ratio, 4),
            "repair_reasons_count": len(repair_reasons or []),
        },
    }


def _normalize_rewrite_delta_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", str(value).casefold())).strip()


def _rewrite_delta_tokens(value: str) -> list[str]:
    normalized = _normalize_rewrite_delta_text(value)
    return normalized.split() if normalized else []


def _rewrite_delta_sentences(value: str) -> list[str]:
    chunks = [
        _normalize_rewrite_delta_text(chunk)
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", str(value or ""))
    ]
    return [chunk for chunk in chunks if chunk]


def _shared_bigram_ratio(weak_tokens: list[str], repaired_tokens: list[str]) -> float:
    weak_bigrams = _token_bigrams(weak_tokens)
    if not weak_bigrams:
        return 0.0
    repaired_bigrams = _token_bigrams(repaired_tokens)
    if not repaired_bigrams:
        return 0.0
    return len(weak_bigrams & repaired_bigrams) / len(weak_bigrams)


def _token_bigrams(tokens: list[str]) -> set[tuple[str, str]]:
    return set(zip(tokens, tokens[1:]))


def _extract_banned_phrases_from_repair_reasons(reasons: list[str]) -> list[str]:
    blocked_phrases: list[str] = []
    seen: set[str] = set()
    prefix = "banned_phrase:"
    for reason in reasons:
        if not isinstance(reason, str) or not reason.startswith(prefix):
            continue
        phrase = reason.removeprefix(prefix).strip()
        if not phrase:
            continue
        key = phrase.casefold()
        if key in seen:
            continue
        seen.add(key)
        blocked_phrases.append(phrase)
    return blocked_phrases


def _normalize_alignment_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9#?]+", " ", str(value).casefold())).strip()


def _find_avoid_angle_match(post_text: str, avoid_angle: str) -> dict[str, Any] | None:
    normalized_post = _normalize_alignment_text(post_text)
    normalized_avoid_angle = _normalize_alignment_text(avoid_angle)
    if not normalized_post or not normalized_avoid_angle:
        return None

    candidate_words = [
        word
        for word in normalized_avoid_angle.split()
        if len(word) > 2 and word not in _ALIGNMENT_STOPWORDS
    ]
    specific_words = [
        word
        for word in candidate_words
        if word not in _AVOID_ANGLE_BROAD_TOPIC_WORDS
        and word not in _AVOID_ANGLE_LOW_SIGNAL_WORDS
    ]
    if len(specific_words) < 2:
        return None

    if normalized_avoid_angle in normalized_post:
        return {
            "matched": True,
            "matched_fragment": normalized_avoid_angle,
            "match_type": "exact_phrase",
        }

    generic_terms = set(_BANNED_LINKEDIN_PHRASES) | set(_VAGUE_ABSTRACT_TERMS)
    for window_size in range(min(len(candidate_words), 5), 1, -1):
        for index in range(0, len(candidate_words) - window_size + 1):
            window = candidate_words[index : index + window_size]
            window_specific_words = [
                word
                for word in window
                if word not in _AVOID_ANGLE_BROAD_TOPIC_WORDS
                and word not in _AVOID_ANGLE_LOW_SIGNAL_WORDS
            ]
            if len(window_specific_words) < 2:
                continue
            if window_size == 2 and not any(word in generic_terms for word in window):
                continue
            fragment = " ".join(window)
            if fragment in normalized_post:
                return {
                    "matched": True,
                    "matched_fragment": fragment,
                    "match_type": "meaningful_fragment",
                }
    return None


def _find_concrete_detail_match(
    post_text: str,
    concrete_details: list[str],
) -> dict[str, Any] | None:
    normalized_post = _normalize_alignment_text(post_text)
    if not normalized_post:
        return None

    post_numbers = set(_number_like_tokens(post_text))
    for raw_detail in concrete_details:
        detail = str(raw_detail or "").strip()
        normalized_detail = _normalize_alignment_text(detail)
        if not normalized_detail:
            continue

        if normalized_detail in normalized_post:
            return {
                "matched": True,
                "matched_detail": detail,
                "matched_fragment": normalized_detail,
                "match_type": "exact_phrase",
            }

        detail_numbers = set(_number_like_tokens(detail))
        number_overlap = sorted(detail_numbers & post_numbers)
        if number_overlap:
            return {
                "matched": True,
                "matched_detail": detail,
                "matched_fragment": number_overlap[0],
                "match_type": "number_overlap",
            }

        specific_words = _concrete_detail_specific_words(detail)
        if len(specific_words) < 2:
            continue

        for window_size in range(min(len(specific_words), 5), 1, -1):
            for index in range(0, len(specific_words) - window_size + 1):
                fragment = " ".join(specific_words[index : index + window_size])
                if fragment in normalized_post:
                    return {
                        "matched": True,
                        "matched_detail": detail,
                        "matched_fragment": fragment,
                        "match_type": "meaningful_fragment",
                    }

        post_words = set(normalized_post.split())
        overlapping_words = [word for word in specific_words if word in post_words]
        required_overlap = min(3, len(specific_words))
        if len(overlapping_words) >= required_overlap:
            return {
                "matched": True,
                "matched_detail": detail,
                "matched_fragment": " ".join(overlapping_words[:required_overlap]),
                "match_type": "meaningful_fragment",
            }
    return None


def _concrete_detail_specific_words(detail: str) -> list[str]:
    return [
        word
        for word in _meaningful_words(detail)
        if word not in _AVOID_ANGLE_BROAD_TOPIC_WORDS
        and word not in _CONCRETE_DETAIL_LOW_SIGNAL_WORDS
    ]


def _number_like_tokens(value: str) -> list[str]:
    return re.findall(r"(?<!\w)\d+(?:\.\d+)?%?(?!\w)", str(value or ""))


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


def _count_concrete_detail_signals(
    post_text: str,
    post_brief: dict[str, Any] | None = None,
) -> int:
    count = 0
    normalized_post = _normalize_alignment_text(post_text)
    count += len(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(post_text or "")))
    count += sum(1 for phrase in _SPECIFICITY_OPERATIONAL_PHRASES if phrase in normalized_post)

    concrete_details = [
        detail
        for detail in (post_brief or {}).get("concrete_details", [])
        if isinstance(detail, str) and detail.strip()
    ]
    if _find_concrete_detail_match(post_text, concrete_details):
        count += 1

    return count


def _has_weak_ending(post_text: str) -> bool:
    last_sentence = _last_sentence(post_text)
    if not last_sentence:
        return False
    normalized_last_sentence = _normalize_alignment_text(last_sentence)
    return any(phrase in normalized_last_sentence for phrase in _WEAK_ENDING_PHRASES)


def _last_sentence(text: str) -> str:
    sentences = [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", str(text or "")) if sentence.strip()]
    return sentences[-1] if sentences else _first_non_empty_line(text)


def _matching_phrases(text: str, phrases: list[str]) -> list[str]:
    return [phrase for phrase in phrases if phrase in text]


def _quality_gate_requires_repair(report: dict[str, Any]) -> bool:
    return str(report.get("status") or "") == "retry" and bool(report.get("reasons"))


def _editorial_review_requires_repair(editorial_review: dict[str, Any] | None) -> bool:
    if not editorial_review:
        return False
    if editorial_review.get("passed") is False:
        return True
    return _editorial_review_score_below_threshold(editorial_review.get("score"))


def _editorial_review_score_below_threshold(score: Any, min_score: int = 7) -> bool:
    if isinstance(score, bool):
        return False
    return isinstance(score, (int, float)) and score < min_score


def _editorial_review_repair_reasons(editorial_review: dict[str, Any] | None) -> list[str]:
    if not editorial_review:
        return []

    reasons: list[str] = []
    if editorial_review.get("passed") is False:
        reasons.append("editorial_review:failed")

    if _editorial_review_score_below_threshold(editorial_review.get("score")):
        reasons.append("editorial_review:score_below_threshold")

    seen = set(reasons)
    for issue in editorial_review.get("issues", []):
        if not isinstance(issue, str):
            continue
        normalized_issue = re.sub(r"[^a-z0-9_:-]+", "_", issue.strip().casefold()).strip("_")
        if not normalized_issue:
            continue
        reason = f"editorial_review:{normalized_issue}"
        if reason in seen:
            continue
        seen.add(reason)
        reasons.append(reason)
    return reasons


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
