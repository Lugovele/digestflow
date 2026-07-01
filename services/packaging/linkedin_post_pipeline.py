"""Deterministic contracts for the clean LinkedIn posting pipeline.

This module defines typed stage boundaries only. It does not call models, render
prompts, call providers, or connect to the current packaging runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


ALLOWED_EVIDENCE_TYPES = {
    "fact",
    "pattern",
    "example",
    "contrast",
    "warning",
    "practical_point",
}

ALLOWED_SPECIFICITY_LEVELS = {"high", "medium", "low"}

ALLOWED_BEST_USE_VALUES = {
    "hook",
    "tension",
    "proof",
    "practical_point",
    "ending",
    "background_only",
}

REQUIRED_QUALITY_CHECKS = {
    "uses_only_provided_facts",
    "has_clear_point_of_view",
    "linkedin_ready",
}


class LinkedInPostPipelineContractError(ValueError):
    """Raised when a LinkedIn posting pipeline contract is invalid."""


@dataclass(frozen=True)
class SelectedArticle:
    source_index: int
    title: str
    url: str
    summary: str
    key_points: list[str]
    source_name: str = ""
    published_at: str = ""
    content_type: str = ""
    confidence: float | None = None


@dataclass(frozen=True)
class PipelineInput:
    digest_id: int
    topic_name: str
    digest_title: str
    articles: list[SelectedArticle]
    author_profile: dict[str, Any]


@dataclass(frozen=True)
class ArticleEvidence:
    evidence_id: str
    source_index: int
    source_title: str
    evidence_text: str
    evidence_type: str
    specificity_level: str
    source_limitations: str


@dataclass(frozen=True)
class ArticleEvidencePack:
    items: list[ArticleEvidence]
    usable_count: int
    rejected_count: int
    rejection_reasons: list[str] = field(default_factory=list)
    coverage_notes: str = ""


@dataclass(frozen=True)
class ContextualEvidence:
    evidence_id: str
    evidence_text: str
    what_it_says: str
    supports_argument: str
    best_use_in_post: str
    do_not_use_for: str
    risk_of_misuse: str


@dataclass(frozen=True)
class ContextualEvidencePack:
    items: list[ContextualEvidence]
    main_candidate_evidence_ids: list[str]
    background_evidence_ids: list[str]
    risks: list[str]


@dataclass(frozen=True)
class AngleDecision:
    controlling_angle: str
    reader_problem: str
    author_position: str
    main_tension: str
    supporting_evidence_ids: list[str]
    angle_to_avoid: list[str]


@dataclass(frozen=True)
class BriefEvidenceUse:
    evidence_id: str
    evidence_text: str
    role_in_post: str


@dataclass(frozen=True)
class PostBrief:
    opening_direction: str
    pattern_interrupt: str
    core_point: str
    evidence_to_use: list[BriefEvidenceUse]
    practical_point: str
    ending_direction: str
    cta_direction: str


@dataclass(frozen=True)
class FinalPostPayload:
    post_text: str
    hook_variants: list[str]
    cta_variants: list[str]
    hashtags: list[str]
    quality_checks: dict[str, bool]
    carousel_outline: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class QualityReviewResult:
    scores: dict[str, int]
    total_score: int
    pass_result: bool
    failed_criteria: list[str]
    automatic_fail_reason: str = ""


@dataclass(frozen=True)
class TargetedRepairPlan:
    failed_criterion: str
    repair_scope: str
    repair_instruction: str
    preserve: list[str]
    avoid: list[str]


def build_pipeline_input_from_digest(
    digest: Any,
    author_profile: dict[str, Any] | None = None,
) -> PipelineInput:
    if author_profile is None:
        normalized_author_profile: dict[str, Any] = {}
    elif isinstance(author_profile, dict):
        normalized_author_profile = dict(author_profile)
    else:
        raise LinkedInPostPipelineContractError("author_profile must be a dictionary.")

    raw_articles = digest.get_articles()
    if not isinstance(raw_articles, list):
        raise LinkedInPostPipelineContractError("digest.get_articles() must return a list.")

    articles: list[SelectedArticle] = []
    for source_index, article in enumerate(raw_articles):
        if not isinstance(article, dict):
            raise LinkedInPostPipelineContractError(
                f"digest.get_articles()[{source_index}] must be a dictionary."
            )
        articles.append(
            SelectedArticle(
                source_index=source_index,
                title=article.get("title", ""),
                url=article.get("url", ""),
                summary=article.get("summary", ""),
                key_points=article.get("key_points", []),
                source_name=str(article.get("source_name") or "").strip(),
                published_at=str(article.get("published_at") or "").strip(),
                content_type=str(article.get("content_type") or "").strip(),
                confidence=article.get("confidence"),
            )
        )

    run = getattr(digest, "run", None)
    topic = getattr(run, "topic", None)

    pipeline_input = PipelineInput(
        digest_id=getattr(digest, "id", None),
        topic_name=str(getattr(topic, "name", "")).strip(),
        digest_title=str(getattr(digest, "title", "")).strip(),
        articles=articles,
        author_profile=normalized_author_profile,
    )
    validate_pipeline_input(pipeline_input)
    return pipeline_input


def validate_pipeline_input(pipeline_input: PipelineInput) -> None:
    if not isinstance(pipeline_input, PipelineInput):
        raise LinkedInPostPipelineContractError(
            "pipeline_input must be a PipelineInput."
        )
    if not isinstance(pipeline_input.digest_id, int):
        raise LinkedInPostPipelineContractError("PipelineInput.digest_id must be an int.")
    _require_non_empty_string(pipeline_input.topic_name, "PipelineInput.topic_name")
    _require_non_empty_string(pipeline_input.digest_title, "PipelineInput.digest_title")
    validate_selected_articles(pipeline_input.articles)
    if not isinstance(pipeline_input.author_profile, dict):
        raise LinkedInPostPipelineContractError(
            "PipelineInput.author_profile must be a dictionary."
        )


def validate_selected_articles(articles: list[SelectedArticle]) -> None:
    if not isinstance(articles, list):
        raise LinkedInPostPipelineContractError("articles must be a list.")

    for expected_index, article in enumerate(articles):
        if not isinstance(article, SelectedArticle):
            raise LinkedInPostPipelineContractError(
                f"articles[{expected_index}] must be a SelectedArticle."
            )
        if article.source_index != expected_index:
            raise LinkedInPostPipelineContractError(
                "source_index must be zero-based and match the article position "
                "in the selected article list."
            )
        _require_non_empty_string(article.title, f"articles[{expected_index}].title")
        _require_non_empty_string(article.url, f"articles[{expected_index}].url")
        _require_non_empty_string(article.summary, f"articles[{expected_index}].summary")
        _require_string_list(article.key_points, f"articles[{expected_index}].key_points")


def validate_article_evidence_pack(pack: ArticleEvidencePack) -> None:
    if not isinstance(pack, ArticleEvidencePack):
        raise LinkedInPostPipelineContractError("pack must be an ArticleEvidencePack.")
    if not isinstance(pack.items, list):
        raise LinkedInPostPipelineContractError("ArticleEvidencePack.items must be a list.")
    if pack.usable_count != len(pack.items):
        raise LinkedInPostPipelineContractError(
            "ArticleEvidencePack.usable_count must match the number of evidence items."
        )
    if pack.rejected_count < 0:
        raise LinkedInPostPipelineContractError(
            "ArticleEvidencePack.rejected_count must not be negative."
        )
    _require_string_list(pack.rejection_reasons, "ArticleEvidencePack.rejection_reasons")

    for item_index, item in enumerate(pack.items):
        if not isinstance(item, ArticleEvidence):
            raise LinkedInPostPipelineContractError(
                f"ArticleEvidencePack.items[{item_index}] must be an ArticleEvidence."
            )
        _require_non_empty_string(item.evidence_id, f"items[{item_index}].evidence_id")
        if item.source_index < 0:
            raise LinkedInPostPipelineContractError(
                f"items[{item_index}].source_index must not be negative."
            )
        _require_non_empty_string(item.source_title, f"items[{item_index}].source_title")
        _require_non_empty_string(item.evidence_text, f"items[{item_index}].evidence_text")
        _require_choice(
            item.evidence_type,
            ALLOWED_EVIDENCE_TYPES,
            f"items[{item_index}].evidence_type",
        )
        _require_choice(
            item.specificity_level,
            ALLOWED_SPECIFICITY_LEVELS,
            f"items[{item_index}].specificity_level",
        )
        _require_non_empty_string(
            item.source_limitations,
            f"items[{item_index}].source_limitations",
        )


def validate_contextual_evidence_pack(pack: ContextualEvidencePack) -> None:
    if not isinstance(pack, ContextualEvidencePack):
        raise LinkedInPostPipelineContractError("pack must be a ContextualEvidencePack.")
    if not isinstance(pack.items, list):
        raise LinkedInPostPipelineContractError("ContextualEvidencePack.items must be a list.")

    evidence_ids: set[str] = set()
    for item_index, item in enumerate(pack.items):
        if not isinstance(item, ContextualEvidence):
            raise LinkedInPostPipelineContractError(
                f"ContextualEvidencePack.items[{item_index}] must be a ContextualEvidence."
            )
        _require_non_empty_string(item.evidence_id, f"items[{item_index}].evidence_id")
        _require_non_empty_string(item.evidence_text, f"items[{item_index}].evidence_text")
        _require_non_empty_string(item.what_it_says, f"items[{item_index}].what_it_says")
        _require_non_empty_string(
            item.supports_argument,
            f"items[{item_index}].supports_argument",
        )
        _require_choice(
            item.best_use_in_post,
            ALLOWED_BEST_USE_VALUES,
            f"items[{item_index}].best_use_in_post",
        )
        _require_non_empty_string(item.do_not_use_for, f"items[{item_index}].do_not_use_for")
        _require_non_empty_string(
            item.risk_of_misuse,
            f"items[{item_index}].risk_of_misuse",
        )
        evidence_ids.add(item.evidence_id)

    _require_existing_ids(
        pack.main_candidate_evidence_ids,
        evidence_ids,
        "ContextualEvidencePack.main_candidate_evidence_ids",
    )
    _require_existing_ids(
        pack.background_evidence_ids,
        evidence_ids,
        "ContextualEvidencePack.background_evidence_ids",
    )
    _require_string_list(pack.risks, "ContextualEvidencePack.risks")


def validate_angle_decision(decision: AngleDecision) -> None:
    if not isinstance(decision, AngleDecision):
        raise LinkedInPostPipelineContractError("decision must be an AngleDecision.")

    _require_non_empty_string(decision.controlling_angle, "AngleDecision.controlling_angle")
    _require_non_empty_string(decision.reader_problem, "AngleDecision.reader_problem")
    _require_non_empty_string(decision.author_position, "AngleDecision.author_position")
    _require_non_empty_string(decision.main_tension, "AngleDecision.main_tension")
    _require_string_list(decision.supporting_evidence_ids, "AngleDecision.supporting_evidence_ids")
    _require_string_list(decision.angle_to_avoid, "AngleDecision.angle_to_avoid")


def validate_post_brief(brief: PostBrief) -> None:
    if not isinstance(brief, PostBrief):
        raise LinkedInPostPipelineContractError("brief must be a PostBrief.")

    _require_non_empty_string(brief.opening_direction, "PostBrief.opening_direction")
    _require_non_empty_string(brief.pattern_interrupt, "PostBrief.pattern_interrupt")
    _require_non_empty_string(brief.core_point, "PostBrief.core_point")
    _require_non_empty_string(brief.practical_point, "PostBrief.practical_point")
    _require_non_empty_string(brief.ending_direction, "PostBrief.ending_direction")
    _require_non_empty_string(brief.cta_direction, "PostBrief.cta_direction")

    if not isinstance(brief.evidence_to_use, list) or not brief.evidence_to_use:
        raise LinkedInPostPipelineContractError(
            "PostBrief.evidence_to_use must be a non-empty list."
        )
    for item_index, item in enumerate(brief.evidence_to_use):
        if not isinstance(item, BriefEvidenceUse):
            raise LinkedInPostPipelineContractError(
                f"PostBrief.evidence_to_use[{item_index}] must be a BriefEvidenceUse."
            )
        _require_non_empty_string(
            item.evidence_id,
            f"PostBrief.evidence_to_use[{item_index}].evidence_id",
        )
        _require_non_empty_string(
            item.evidence_text,
            f"PostBrief.evidence_to_use[{item_index}].evidence_text",
        )
        _require_non_empty_string(
            item.role_in_post,
            f"PostBrief.evidence_to_use[{item_index}].role_in_post",
        )


def validate_final_post_payload(payload: FinalPostPayload) -> None:
    if not isinstance(payload, FinalPostPayload):
        raise LinkedInPostPipelineContractError("payload must be a FinalPostPayload.")

    _require_non_empty_string(payload.post_text, "FinalPostPayload.post_text")
    if len(payload.post_text) > 1300:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.post_text must not exceed 1300 characters."
        )
    _require_string_list(
        payload.hook_variants,
        "FinalPostPayload.hook_variants",
        min_items=3,
    )
    _require_string_list(
        payload.cta_variants,
        "FinalPostPayload.cta_variants",
        min_items=3,
    )
    _require_string_list(payload.hashtags, "FinalPostPayload.hashtags", min_items=1)
    if not isinstance(payload.carousel_outline, list):
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.carousel_outline must be a list."
        )

    quality_checks = payload.quality_checks
    if not isinstance(quality_checks, dict):
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.quality_checks must be a dictionary."
        )
    missing_checks = sorted(REQUIRED_QUALITY_CHECKS - quality_checks.keys())
    if missing_checks:
        raise LinkedInPostPipelineContractError(
            f"FinalPostPayload.quality_checks is missing keys: {missing_checks}."
        )
    for key in REQUIRED_QUALITY_CHECKS:
        if not isinstance(quality_checks[key], bool):
            raise LinkedInPostPipelineContractError(
                f"FinalPostPayload.quality_checks.{key} must be a boolean."
            )


def final_post_payload_to_dict(payload: FinalPostPayload) -> dict[str, Any]:
    validate_final_post_payload(payload)
    return {
        "post_text": payload.post_text,
        "hook_variants": list(payload.hook_variants),
        "cta_variants": list(payload.cta_variants),
        "hashtags": list(payload.hashtags),
        "quality_checks": dict(payload.quality_checks),
        "carousel_outline": list(payload.carousel_outline),
    }


def _require_non_empty_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise LinkedInPostPipelineContractError(f"{field_name} must be a non-empty string.")


def _require_string_list(value: Any, field_name: str, min_items: int = 0) -> None:
    if not isinstance(value, list):
        raise LinkedInPostPipelineContractError(f"{field_name} must be a list.")
    if len(value) < min_items:
        raise LinkedInPostPipelineContractError(
            f"{field_name} must contain at least {min_items} items."
        )
    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise LinkedInPostPipelineContractError(
                f"{field_name}[{item_index}] must be a non-empty string."
            )


def _require_choice(value: Any, allowed_values: set[str], field_name: str) -> None:
    if value not in allowed_values:
        allowed = ", ".join(sorted(allowed_values))
        raise LinkedInPostPipelineContractError(
            f"{field_name} must be one of: {allowed}."
        )


def _require_existing_ids(value: Any, allowed_ids: set[str], field_name: str) -> None:
    _require_string_list(value, field_name)
    unknown_ids = sorted(set(value) - allowed_ids)
    if unknown_ids:
        raise LinkedInPostPipelineContractError(
            f"{field_name} contains unknown evidence IDs: {unknown_ids}."
        )


__all__ = [
    "ALLOWED_BEST_USE_VALUES",
    "ALLOWED_EVIDENCE_TYPES",
    "ALLOWED_SPECIFICITY_LEVELS",
    "REQUIRED_QUALITY_CHECKS",
    "AngleDecision",
    "ArticleEvidence",
    "ArticleEvidencePack",
    "BriefEvidenceUse",
    "ContextualEvidence",
    "ContextualEvidencePack",
    "FinalPostPayload",
    "LinkedInPostPipelineContractError",
    "PipelineInput",
    "PostBrief",
    "QualityReviewResult",
    "SelectedArticle",
    "TargetedRepairPlan",
    "build_pipeline_input_from_digest",
    "final_post_payload_to_dict",
    "validate_angle_decision",
    "validate_article_evidence_pack",
    "validate_contextual_evidence_pack",
    "validate_final_post_payload",
    "validate_pipeline_input",
    "validate_post_brief",
    "validate_selected_articles",
]
