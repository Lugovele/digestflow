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
    source_index: int
    source_title: str
    evidence_text: str
    evidence_type: str
    specificity_level: str
    source_limitations: str
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


def build_article_evidence_pack_from_pipeline_input(
    pipeline_input: PipelineInput,
) -> ArticleEvidencePack:
    validate_pipeline_input(pipeline_input)

    items: list[ArticleEvidence] = []
    source_limitations = (
        "Evidence is derived from digest summaries and key points, "
        "not full article extraction."
    )
    for article in pipeline_input.articles:
        items.append(
            ArticleEvidence(
                evidence_id=f"a{article.source_index}-summary",
                source_index=article.source_index,
                source_title=article.title,
                evidence_text=article.summary,
                evidence_type="pattern",
                specificity_level="medium",
                source_limitations=source_limitations,
            )
        )
        for key_point_index, key_point in enumerate(article.key_points):
            items.append(
                ArticleEvidence(
                    evidence_id=f"a{article.source_index}-kp{key_point_index}",
                    source_index=article.source_index,
                    source_title=article.title,
                    evidence_text=key_point,
                    evidence_type="practical_point",
                    specificity_level="medium",
                    source_limitations=source_limitations,
                )
            )

    article_evidence_pack = ArticleEvidencePack(
        items=items,
        usable_count=len(items),
        rejected_count=0,
    )
    validate_article_evidence_pack_for_pipeline_input(
        pipeline_input,
        article_evidence_pack,
    )
    return article_evidence_pack


def build_contextual_evidence_pack_from_article_evidence_pack(
    article_evidence_pack: ArticleEvidencePack,
) -> ContextualEvidencePack:
    validate_article_evidence_pack(article_evidence_pack)

    items: list[ContextualEvidence] = []
    main_candidate_evidence_ids: list[str] = []
    background_evidence_ids: list[str] = []
    risks: list[str] = []

    for evidence in article_evidence_pack.items:
        best_use = _best_use_for_article_evidence(evidence)
        evidence_risks = _risks_for_article_evidence(evidence)
        risk_of_misuse = "; ".join(evidence_risks)

        items.append(
            ContextualEvidence(
                evidence_id=evidence.evidence_id,
                source_index=evidence.source_index,
                source_title=evidence.source_title,
                evidence_text=evidence.evidence_text,
                evidence_type=evidence.evidence_type,
                specificity_level=evidence.specificity_level,
                source_limitations=evidence.source_limitations,
                what_it_says=f"Evidence says: {evidence.evidence_text}",
                supports_argument=(
                    f"May support a {best_use} role in a later LinkedIn post."
                    if best_use != "background_only"
                    else "Provides background context only."
                ),
                best_use_in_post=best_use,
                do_not_use_for=(
                    "Do not use as main proof or the controlling angle."
                    if best_use == "background_only"
                    else "Do not use as a standalone claim beyond the source evidence."
                ),
                risk_of_misuse=risk_of_misuse,
            )
        )

        if best_use == "background_only":
            background_evidence_ids.append(evidence.evidence_id)
        else:
            main_candidate_evidence_ids.append(evidence.evidence_id)
        for risk in evidence_risks:
            if risk not in risks:
                risks.append(risk)

    contextual_evidence_pack = ContextualEvidencePack(
        items=items,
        main_candidate_evidence_ids=main_candidate_evidence_ids,
        background_evidence_ids=background_evidence_ids,
        risks=risks,
    )
    validate_contextual_evidence_pack_for_article_evidence(
        article_evidence_pack,
        contextual_evidence_pack,
    )
    return contextual_evidence_pack


def build_angle_decision_from_contextual_evidence_pack(
    contextual_evidence_pack: ContextualEvidencePack,
) -> AngleDecision:
    validate_contextual_evidence_pack(contextual_evidence_pack)

    if not contextual_evidence_pack.main_candidate_evidence_ids:
        raise LinkedInPostPipelineContractError(
            "ContextualEvidencePack.main_candidate_evidence_ids must contain at least one item."
        )

    items_by_id = {item.evidence_id: item for item in contextual_evidence_pack.items}
    selected_items = _select_angle_supporting_evidence(
        contextual_evidence_pack,
        items_by_id,
    )
    selected_ids = [item.evidence_id for item in selected_items]
    selected_roles = _unique_preserving_order(
        [item.best_use_in_post for item in selected_items]
    )
    role_summary = ", ".join(selected_roles)
    primary_evidence_text = selected_items[0].evidence_text

    angle_decision = AngleDecision(
        controlling_angle=(
            "Use the selected contextual evidence to keep one source-grounded "
            f"angle focused on {role_summary}, starting from: {primary_evidence_text}"
        ),
        reader_problem=(
            "The reader may collapse separate source signals into one broad claim "
            "unless the post separates what each selected evidence item supports."
        ),
        author_position=(
            "Separate signals, keep claims attributed to the selected evidence, "
            "and avoid unsupported conclusions or investment advice."
        ),
        main_tension=(
            "The selected evidence can support a useful post angle, but its limits "
            "must stay visible."
        ),
        supporting_evidence_ids=selected_ids,
        angle_to_avoid=_build_angle_to_avoid(
            contextual_evidence_pack,
            selected_items,
        ),
    )
    validate_angle_decision_for_contextual_evidence(
        contextual_evidence_pack,
        angle_decision,
    )
    return angle_decision


def build_post_brief_from_angle_decision(
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
) -> PostBrief:
    validate_angle_decision_for_contextual_evidence(
        contextual_evidence_pack,
        angle_decision,
    )

    items_by_id = {item.evidence_id: item for item in contextual_evidence_pack.items}
    selected_items = [
        items_by_id[evidence_id]
        for evidence_id in angle_decision.supporting_evidence_ids
    ]
    practical_item = next(
        (
            item
            for item in selected_items
            if item.best_use_in_post == "practical_point"
        ),
        None,
    )

    post_brief = PostBrief(
        opening_direction=(
            "Open with the controlling angle as a planning direction: "
            f"{angle_decision.controlling_angle}"
        ),
        pattern_interrupt=(
            "Use this tension to interrupt the expected framing: "
            f"{angle_decision.main_tension}"
        ),
        core_point=(
            "Frame the core point as the author's position: "
            f"{angle_decision.author_position}"
        ),
        evidence_to_use=[
            BriefEvidenceUse(
                evidence_id=item.evidence_id,
                evidence_text=item.evidence_text,
                role_in_post=_brief_role_for_contextual_evidence(item),
            )
            for item in selected_items
        ],
        practical_point=(
            "Use this practical evidence to shape the reader takeaway: "
            f"{practical_item.evidence_text}"
            if practical_item is not None
            else (
                "Keep the practical point source-grounded and limited to the "
                "selected supporting evidence."
            )
        ),
        ending_direction=(
            "End by returning to the author's position without adding new claims: "
            f"{angle_decision.author_position}"
        ),
        cta_direction=(
            "Ask the reader to consider the reader problem: "
            f"{angle_decision.reader_problem}"
        ),
    )
    validate_post_brief_for_angle_decision(
        contextual_evidence_pack,
        angle_decision,
        post_brief,
    )
    return post_brief


def build_final_post_payload_from_post_brief(
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
    post_brief: PostBrief,
) -> FinalPostPayload:
    validate_post_brief_for_angle_decision(
        contextual_evidence_pack,
        angle_decision,
        post_brief,
    )

    for item in post_brief.evidence_to_use:
        if _contains_url(item.evidence_text):
            raise LinkedInPostPipelineContractError(
                "PostBrief.evidence_to_use.evidence_text must not contain URLs "
                "when building the FinalPostPayload scaffold."
            )

    evidence_lines = [
        f"- {item.evidence_text}"
        for item in post_brief.evidence_to_use
    ]
    post_text = "\n".join(
        [
            "[Scaffold only - not production copy]",
            f"Controlling angle: {angle_decision.controlling_angle}",
            f"Core point direction: {post_brief.core_point}",
            "Selected evidence:",
            *evidence_lines,
            f"Practical direction: {post_brief.practical_point}",
            f"CTA direction: {post_brief.cta_direction}",
        ]
    )
    if len(post_text) > 1300:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload scaffold post_text would exceed 1300 characters."
        )

    final_post_payload = FinalPostPayload(
        post_text=post_text,
        hook_variants=[
            f"Scaffold hook from opening: {post_brief.opening_direction}",
            f"Scaffold hook from core point: {post_brief.core_point}",
            f"Scaffold hook from angle: {angle_decision.controlling_angle}",
        ],
        cta_variants=[
            f"Scaffold CTA from brief: {post_brief.cta_direction}",
            f"Scaffold CTA from reader problem: {angle_decision.reader_problem}",
            "What should the reader check against the selected evidence?",
        ],
        hashtags=["#PostFlow"],
        quality_checks={
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
            "linkedin_ready": False,
        },
        carousel_outline=[],
    )
    validate_final_post_payload(final_post_payload)
    validate_final_post_payload_for_post_brief(
        contextual_evidence_pack,
        angle_decision,
        post_brief,
        final_post_payload,
    )
    return final_post_payload


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


def validate_article_evidence_pack_for_pipeline_input(
    pipeline_input: PipelineInput,
    article_evidence_pack: ArticleEvidencePack,
) -> None:
    validate_pipeline_input(pipeline_input)
    validate_article_evidence_pack(article_evidence_pack)

    article_indexes = {article.source_index for article in pipeline_input.articles}
    _require_unique_ids(
        [item.evidence_id for item in article_evidence_pack.items],
        "ArticleEvidencePack.items.evidence_id",
    )
    for item in article_evidence_pack.items:
        if item.source_index not in article_indexes:
            raise LinkedInPostPipelineContractError(
                "ArticleEvidence.source_index must reference an existing selected article."
            )


def validate_contextual_evidence_pack(pack: ContextualEvidencePack) -> None:
    if not isinstance(pack, ContextualEvidencePack):
        raise LinkedInPostPipelineContractError("pack must be a ContextualEvidencePack.")
    if not isinstance(pack.items, list):
        raise LinkedInPostPipelineContractError("ContextualEvidencePack.items must be a list.")

    for item_index, item in enumerate(pack.items):
        if not isinstance(item, ContextualEvidence):
            raise LinkedInPostPipelineContractError(
                f"ContextualEvidencePack.items[{item_index}] must be a ContextualEvidence."
            )
        _require_non_empty_string(item.evidence_id, f"items[{item_index}].evidence_id")
        if not isinstance(item.source_index, int) or item.source_index < 0:
            raise LinkedInPostPipelineContractError(
                f"items[{item_index}].source_index must be a non-negative integer."
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
    evidence_ids = _require_unique_ids(
        [item.evidence_id for item in pack.items],
        "ContextualEvidencePack.items.evidence_id",
    )

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
    _require_unique_ids(
        pack.main_candidate_evidence_ids,
        "ContextualEvidencePack.main_candidate_evidence_ids",
    )
    _require_unique_ids(
        pack.background_evidence_ids,
        "ContextualEvidencePack.background_evidence_ids",
    )
    main_candidate_ids = set(pack.main_candidate_evidence_ids)
    background_ids = set(pack.background_evidence_ids)
    overlapping_ids = sorted(main_candidate_ids & background_ids)
    if overlapping_ids:
        raise LinkedInPostPipelineContractError(
            "ContextualEvidencePack main and background evidence IDs must be disjoint: "
            f"{overlapping_ids}."
        )
    background_only_ids = {
        item.evidence_id
        for item in pack.items
        if item.best_use_in_post == "background_only"
    }
    expected_main_candidate_ids = evidence_ids - background_only_ids
    if main_candidate_ids != expected_main_candidate_ids:
        raise LinkedInPostPipelineContractError(
            "ContextualEvidencePack.main_candidate_evidence_ids must exactly match "
            "contextual evidence that is not background_only."
        )
    if background_ids != background_only_ids:
        raise LinkedInPostPipelineContractError(
            "ContextualEvidencePack.background_evidence_ids must exactly match "
            "contextual evidence where best_use_in_post is background_only."
        )
    _require_string_list(pack.risks, "ContextualEvidencePack.risks")


def validate_contextual_evidence_pack_for_article_evidence(
    article_evidence_pack: ArticleEvidencePack,
    contextual_evidence_pack: ContextualEvidencePack,
) -> None:
    validate_article_evidence_pack(article_evidence_pack)
    validate_contextual_evidence_pack(contextual_evidence_pack)

    article_evidence_ids = _require_unique_ids(
        [item.evidence_id for item in article_evidence_pack.items],
        "ArticleEvidencePack.items.evidence_id",
    )
    contextual_evidence_ids = _require_unique_ids(
        [item.evidence_id for item in contextual_evidence_pack.items],
        "ContextualEvidencePack.items.evidence_id",
    )
    if contextual_evidence_ids != article_evidence_ids:
        missing_ids = sorted(article_evidence_ids - contextual_evidence_ids)
        extra_ids = sorted(contextual_evidence_ids - article_evidence_ids)
        raise LinkedInPostPipelineContractError(
            "ContextualEvidencePack.items.evidence_id must exactly match "
            "ArticleEvidencePack.items.evidence_id. "
            f"Missing: {missing_ids}. Extra: {extra_ids}."
        )
    _require_existing_ids(
        contextual_evidence_pack.main_candidate_evidence_ids,
        article_evidence_ids,
        "ContextualEvidencePack.main_candidate_evidence_ids",
    )
    _require_existing_ids(
        contextual_evidence_pack.background_evidence_ids,
        article_evidence_ids,
        "ContextualEvidencePack.background_evidence_ids",
    )
    article_evidence_by_id = {
        item.evidence_id: item for item in article_evidence_pack.items
    }
    for item in contextual_evidence_pack.items:
        article_evidence = article_evidence_by_id[item.evidence_id]
        _require_matching_contextual_evidence_field(
            item.source_index,
            article_evidence.source_index,
            item.evidence_id,
            "source_index",
        )
        _require_matching_contextual_evidence_field(
            item.source_title,
            article_evidence.source_title,
            item.evidence_id,
            "source_title",
        )
        _require_matching_contextual_evidence_field(
            item.evidence_text,
            article_evidence.evidence_text,
            item.evidence_id,
            "evidence_text",
        )
        _require_matching_contextual_evidence_field(
            item.evidence_type,
            article_evidence.evidence_type,
            item.evidence_id,
            "evidence_type",
        )
        _require_matching_contextual_evidence_field(
            item.specificity_level,
            article_evidence.specificity_level,
            item.evidence_id,
            "specificity_level",
        )
        _require_matching_contextual_evidence_field(
            item.source_limitations,
            article_evidence.source_limitations,
            item.evidence_id,
            "source_limitations",
        )


def validate_angle_decision(decision: AngleDecision) -> None:
    if not isinstance(decision, AngleDecision):
        raise LinkedInPostPipelineContractError("decision must be an AngleDecision.")

    _require_non_empty_string(decision.controlling_angle, "AngleDecision.controlling_angle")
    _require_non_empty_string(decision.reader_problem, "AngleDecision.reader_problem")
    _require_non_empty_string(decision.author_position, "AngleDecision.author_position")
    _require_non_empty_string(decision.main_tension, "AngleDecision.main_tension")
    _require_string_list(decision.supporting_evidence_ids, "AngleDecision.supporting_evidence_ids")
    _require_string_list(decision.angle_to_avoid, "AngleDecision.angle_to_avoid")


def validate_angle_decision_for_contextual_evidence(
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
) -> None:
    validate_contextual_evidence_pack(contextual_evidence_pack)
    validate_angle_decision(angle_decision)

    if not 1 <= len(angle_decision.supporting_evidence_ids) <= 3:
        raise LinkedInPostPipelineContractError(
            "AngleDecision.supporting_evidence_ids must contain 1 to 3 IDs."
        )
    _require_unique_ids(
        angle_decision.supporting_evidence_ids,
        "AngleDecision.supporting_evidence_ids",
    )
    _require_existing_ids(
        angle_decision.supporting_evidence_ids,
        set(contextual_evidence_pack.main_candidate_evidence_ids),
        "AngleDecision.supporting_evidence_ids",
    )


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


def validate_post_brief_for_angle_decision(
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
    post_brief: PostBrief,
) -> None:
    validate_angle_decision_for_contextual_evidence(contextual_evidence_pack, angle_decision)
    validate_post_brief(post_brief)

    contextual_evidence_ids = _require_unique_ids(
        [item.evidence_id for item in contextual_evidence_pack.items],
        "ContextualEvidencePack.items.evidence_id",
    )
    supporting_evidence_ids = set(angle_decision.supporting_evidence_ids)
    approved_evidence_ids = contextual_evidence_ids & supporting_evidence_ids
    brief_evidence_ids = [item.evidence_id for item in post_brief.evidence_to_use]
    _require_unique_ids(
        brief_evidence_ids,
        "PostBrief.evidence_to_use.evidence_id",
    )

    _require_existing_ids(
        brief_evidence_ids,
        approved_evidence_ids,
        "PostBrief.evidence_to_use.evidence_id",
    )
    if brief_evidence_ids != angle_decision.supporting_evidence_ids:
        raise LinkedInPostPipelineContractError(
            "PostBrief.evidence_to_use.evidence_id must exactly match "
            "AngleDecision.supporting_evidence_ids in order."
        )

    contextual_evidence_by_id = {
        item.evidence_id: item
        for item in contextual_evidence_pack.items
    }
    for item in post_brief.evidence_to_use:
        contextual_evidence = contextual_evidence_by_id[item.evidence_id]
        if item.evidence_text != contextual_evidence.evidence_text:
            raise LinkedInPostPipelineContractError(
                "PostBrief.evidence_to_use.evidence_text must preserve "
                f"ContextualEvidence.evidence_text for evidence_id {item.evidence_id}."
            )


def validate_linkedin_post_stage_relationships(
    pipeline_input: PipelineInput,
    article_evidence_pack: ArticleEvidencePack,
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
    post_brief: PostBrief,
) -> None:
    validate_article_evidence_pack_for_pipeline_input(
        pipeline_input,
        article_evidence_pack,
    )
    validate_contextual_evidence_pack_for_article_evidence(
        article_evidence_pack,
        contextual_evidence_pack,
    )
    validate_angle_decision_for_contextual_evidence(
        contextual_evidence_pack,
        angle_decision,
    )
    validate_post_brief_for_angle_decision(
        contextual_evidence_pack,
        angle_decision,
        post_brief,
    )


def validate_final_post_payload_for_post_brief(
    contextual_evidence_pack: ContextualEvidencePack,
    angle_decision: AngleDecision,
    post_brief: PostBrief,
    final_post_payload: FinalPostPayload,
) -> None:
    validate_post_brief_for_angle_decision(
        contextual_evidence_pack,
        angle_decision,
        post_brief,
    )
    validate_final_post_payload(final_post_payload)

    if "[Scaffold only - not production copy]" not in final_post_payload.post_text:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.post_text must include the scaffold marker."
        )
    if angle_decision.controlling_angle not in final_post_payload.post_text:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.post_text must include AngleDecision.controlling_angle."
        )
    for item in post_brief.evidence_to_use:
        if item.evidence_text not in final_post_payload.post_text:
            raise LinkedInPostPipelineContractError(
                "FinalPostPayload.post_text must include selected evidence text "
                f"for evidence_id {item.evidence_id}."
            )
    if _contains_url(final_post_payload.post_text):
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.post_text must not contain URLs."
        )
    if final_post_payload.quality_checks.get("linkedin_ready") is not False:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.quality_checks.linkedin_ready must be False for the scaffold."
        )
    if final_post_payload.carousel_outline != []:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.carousel_outline must be empty for the scaffold."
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


def _require_unique_ids(values: list[str], field_name: str) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        raise LinkedInPostPipelineContractError(
            f"{field_name} contains duplicate IDs: {sorted(duplicates)}."
        )
    return seen


def _require_matching_contextual_evidence_field(
    contextual_value: Any,
    article_value: Any,
    evidence_id: str,
    field_name: str,
) -> None:
    if contextual_value != article_value:
        raise LinkedInPostPipelineContractError(
            "ContextualEvidence must preserve ArticleEvidence "
            f"{field_name} for evidence_id {evidence_id}."
        )


def _select_angle_supporting_evidence(
    contextual_evidence_pack: ContextualEvidencePack,
    items_by_id: dict[str, ContextualEvidence],
) -> list[ContextualEvidence]:
    main_items = [
        items_by_id[evidence_id]
        for evidence_id in contextual_evidence_pack.main_candidate_evidence_ids
    ]
    selected_items: list[ContextualEvidence] = []
    selected_ids: set[str] = set()
    selected_source_indexes: set[int] = set()

    for role in ["tension", "proof", "practical_point"]:
        role_candidates = [
            item
            for item in main_items
            if item.best_use_in_post == role and item.evidence_id not in selected_ids
        ]
        if not role_candidates:
            continue
        source_diverse_candidate = next(
            (
                item
                for item in role_candidates
                if item.source_index not in selected_source_indexes
            ),
            role_candidates[0],
        )
        selected_items.append(source_diverse_candidate)
        selected_ids.add(source_diverse_candidate.evidence_id)
        selected_source_indexes.add(source_diverse_candidate.source_index)
        if len(selected_items) == 3:
            return selected_items

    for item in main_items:
        if item.evidence_id in selected_ids or item.source_index in selected_source_indexes:
            continue
        selected_items.append(item)
        selected_ids.add(item.evidence_id)
        selected_source_indexes.add(item.source_index)
        if len(selected_items) == 3:
            return selected_items

    for item in main_items:
        if item.evidence_id in selected_ids:
            continue
        selected_items.append(item)
        selected_ids.add(item.evidence_id)
        if len(selected_items) == 3:
            return selected_items

    return selected_items


def _build_angle_to_avoid(
    contextual_evidence_pack: ContextualEvidencePack,
    selected_items: list[ContextualEvidence],
) -> list[str]:
    values: list[str] = []
    values.extend(contextual_evidence_pack.risks)
    for item in contextual_evidence_pack.items:
        values.append(item.do_not_use_for)
        values.append(item.risk_of_misuse)
    for item in selected_items:
        values.append(f"Do not overstate evidence from {item.evidence_id}.")
    return _unique_preserving_order(values)


def _brief_role_for_contextual_evidence(item: ContextualEvidence) -> str:
    return (
        f"Use this evidence as {item.best_use_in_post} support. "
        f"{item.supports_argument}"
    )


def _contains_url(value: str) -> bool:
    lowered = value.lower()
    return "http://" in lowered or "https://" in lowered or "www." in lowered


def _unique_preserving_order(values: list[str]) -> list[str]:
    unique_values: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


def _best_use_for_article_evidence(evidence: ArticleEvidence) -> str:
    if evidence.specificity_level == "low":
        return "background_only"
    if evidence.evidence_type in {"contrast", "warning"}:
        return "tension"
    if evidence.evidence_type in {"example", "fact", "pattern"}:
        return "proof"
    if evidence.evidence_type == "practical_point":
        return "practical_point"
    return "background_only"


def _risks_for_article_evidence(evidence: ArticleEvidence) -> list[str]:
    risks: list[str] = []
    if evidence.specificity_level == "low":
        risks.append("low-specificity evidence should not become main proof")
    elif evidence.evidence_type == "pattern":
        risks.append("generic summary")
    elif evidence.evidence_type in {"contrast", "warning"}:
        risks.append("overstated contrast")
    elif evidence.evidence_type == "practical_point":
        risks.append("unsupported prescription risk")
    else:
        risks.append("evidence may be overgeneralized")

    if "digest summaries" in evidence.source_limitations.lower():
        risks.append("source-term drift")
    return risks


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
    "build_angle_decision_from_contextual_evidence_pack",
    "build_article_evidence_pack_from_pipeline_input",
    "build_contextual_evidence_pack_from_article_evidence_pack",
    "build_final_post_payload_from_post_brief",
    "build_pipeline_input_from_digest",
    "build_post_brief_from_angle_decision",
    "final_post_payload_to_dict",
    "validate_angle_decision",
    "validate_angle_decision_for_contextual_evidence",
    "validate_article_evidence_pack",
    "validate_article_evidence_pack_for_pipeline_input",
    "validate_contextual_evidence_pack",
    "validate_contextual_evidence_pack_for_article_evidence",
    "validate_final_post_payload",
    "validate_final_post_payload_for_post_brief",
    "validate_linkedin_post_stage_relationships",
    "validate_pipeline_input",
    "validate_post_brief",
    "validate_post_brief_for_angle_decision",
    "validate_selected_articles",
]
