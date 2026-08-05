"""Deterministic contracts for the clean LinkedIn posting pipeline.

This module defines typed stage boundaries only. It does not call models, render
prompts, call providers, or connect to the current packaging runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT,
    FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT,
    FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT,
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
    REQUIRED_QUALITY_CHECKS as FINAL_POST_PAYLOAD_REQUIRED_QUALITY_CHECKS,
    build_final_post_payload_constraints,
)


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

REQUIRED_QUALITY_CHECKS = set(FINAL_POST_PAYLOAD_REQUIRED_QUALITY_CHECKS)

AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED = "allowed_not_required"

AUTHORIAL_PERSONAL_PRESENCE_REQUIRED = "explicit_author_owned_statement_required"
AUTHORIAL_PERSONAL_PRESENCE_ALLOWED = "author_owned_statement_allowed"
AUTHORIAL_PERSONAL_PRESENCE_EDITORIAL_ONLY = "editorial_stance_only"
AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS = (
    AUTHORIAL_PERSONAL_PRESENCE_REQUIRED,
    AUTHORIAL_PERSONAL_PRESENCE_ALLOWED,
    AUTHORIAL_PERSONAL_PRESENCE_EDITORIAL_ONLY,
)

AUTHORIAL_FORBIDDEN_AUTHOR_CLAIMS = (
    "personal experience",
    "professional authority",
    "direct market exposure",
    "client or customer stories",
    "invented emotional reaction",
    "biographical claims",
)


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
class EvidenceRelationship:
    relationship_type: str
    left_label: str
    right_label: str
    left_evidence_ids: list[str]
    right_evidence_ids: list[str]
    supporting_evidence_ids: list[str]
    qualifier: str
    thesis: str
    reader_problem: str
    author_position: str
    main_tension: str
    score: int
    priority: int


@dataclass(frozen=True)
class EditorialSynthesisResult:
    status: str
    dominant_relationship: EvidenceRelationship | None
    candidate_relationships: list[EvidenceRelationship]
    selected_evidence_ids: list[str]
    non_ready_reason: str = ""
    non_ready_detail: str = ""


@dataclass(frozen=True)
class AuthorialVoiceDirective:
    authorial_observation: str
    rejected_reading: str
    why_distinction_matters: str
    personal_presence_requirement: str
    first_person_policy: str
    forbidden_author_claims: tuple[str, ...]


@dataclass(frozen=True)
class AngleDecision:
    controlling_angle: str
    reader_problem: str
    author_position: str
    main_tension: str
    supporting_evidence_ids: list[str]
    angle_to_avoid: list[str]
    authorial_voice_directive: AuthorialVoiceDirective


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
    editorial_synthesis = build_editorial_synthesis_result_for_selected_items(
        selected_items,
    )
    validate_editorial_synthesis_result(editorial_synthesis)
    if editorial_synthesis.status != EDITORIAL_SYNTHESIS_STATUS_READY:
        raise LinkedInPostPipelineContractError(
            "Selected evidence is not ready for an editorial angle: "
            f"{editorial_synthesis.non_ready_reason}. "
            f"{editorial_synthesis.non_ready_detail}"
        )
    relationship = editorial_synthesis.dominant_relationship
    if relationship is None:
        raise LinkedInPostPipelineContractError(
            "Selected evidence is not ready for an editorial angle: "
            "NO_SUPPORTED_RELATIONSHIP."
        )
    controlling_angle, reader_problem, author_position, main_tension = (
        relationship.thesis,
        relationship.reader_problem,
        relationship.author_position,
        relationship.main_tension,
    )

    angle_decision = AngleDecision(
        controlling_angle=controlling_angle,
        reader_problem=reader_problem,
        author_position=author_position,
        main_tension=main_tension,
        supporting_evidence_ids=selected_ids,
        angle_to_avoid=_build_angle_to_avoid(
            contextual_evidence_pack,
            selected_items,
        ),
        authorial_voice_directive=build_authorial_voice_directive_from_evidence_relationship(
            relationship
        ),
    )
    validate_angle_decision_for_contextual_evidence(
        contextual_evidence_pack,
        angle_decision,
    )
    return angle_decision


def build_authorial_voice_directive_from_evidence_relationship(
    relationship: EvidenceRelationship,
) -> AuthorialVoiceDirective:
    validate_evidence_relationship(relationship)

    directive = AuthorialVoiceDirective(
        authorial_observation=(
            "The author notices that "
            f"{relationship.left_label} and {relationship.right_label} should not be "
            "collapsed into one easy conclusion."
        ),
        rejected_reading=(
            "Reject treating "
            f"{relationship.left_label} as proof that {relationship.right_label} "
            "has been resolved."
        ),
        why_distinction_matters=(
            "The distinction matters because "
            f"{relationship.reader_problem[:1].lower()}{relationship.reader_problem[1:]}"
        ),
        personal_presence_requirement=AUTHORIAL_PERSONAL_PRESENCE_REQUIRED,
        first_person_policy=AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED,
        forbidden_author_claims=AUTHORIAL_FORBIDDEN_AUTHOR_CLAIMS,
    )
    validate_authorial_voice_directive(directive)
    return directive


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
            and not _is_market_projection_or_positioning(item)
        ),
        None,
    )
    reader_takeaway = _build_reader_takeaway_for_selected_items(
        selected_items,
        angle_decision,
    )

    post_brief = PostBrief(
        opening_direction=(
            "Open with this editorial angle: "
            f"{angle_decision.controlling_angle}"
        ),
        pattern_interrupt=(
            "Interrupt the expected framing with this tension: "
            f"{angle_decision.main_tension}"
        ),
        core_point=angle_decision.author_position,
        evidence_to_use=[
            BriefEvidenceUse(
                evidence_id=item.evidence_id,
                evidence_text=item.evidence_text,
                role_in_post=_brief_role_for_contextual_evidence(item),
            )
            for item in selected_items
        ],
        practical_point=(
            "Anchor the reader takeaway in this practical point: "
            f"{practical_item.evidence_text}"
            if practical_item is not None
            else reader_takeaway
        ),
        ending_direction=(
            "End by returning to this author position: "
            f"{angle_decision.author_position}"
        ),
        cta_direction=(
            "Ask the reader to reconsider this reader problem: "
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

    evidence_references = [
        f"{item.evidence_id} ({_compact_role_label(item.role_in_post)})"
        for item in post_brief.evidence_to_use
    ]
    post_text = "\n".join(
        [
            "[Scaffold only - not production copy]",
            f"Controlling angle: {angle_decision.controlling_angle}",
            f"Core point direction: {post_brief.core_point}",
            f"Selected evidence references: {', '.join(evidence_references)}",
            "Practical direction: Use the selected evidence references to keep the takeaway source-grounded.",
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
    validate_authorial_voice_directive(decision.authorial_voice_directive)


def validate_authorial_voice_directive(
    directive: AuthorialVoiceDirective,
) -> None:
    if not isinstance(directive, AuthorialVoiceDirective):
        raise LinkedInPostPipelineContractError(
            "AngleDecision.authorial_voice_directive must be an AuthorialVoiceDirective."
        )
    _require_non_empty_string(
        directive.authorial_observation,
        "AngleDecision.authorial_voice_directive.authorial_observation",
    )
    _require_non_empty_string(
        directive.rejected_reading,
        "AngleDecision.authorial_voice_directive.rejected_reading",
    )
    _require_non_empty_string(
        directive.why_distinction_matters,
        "AngleDecision.authorial_voice_directive.why_distinction_matters",
    )
    _require_non_empty_string(
        directive.personal_presence_requirement,
        "AngleDecision.authorial_voice_directive.personal_presence_requirement",
    )
    if (
        directive.personal_presence_requirement
        not in AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS
    ):
        raise LinkedInPostPipelineContractError(
            "AngleDecision.authorial_voice_directive.personal_presence_requirement "
            "must be one of "
            f"{', '.join(AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS)}."
        )
    _require_non_empty_string(
        directive.first_person_policy,
        "AngleDecision.authorial_voice_directive.first_person_policy",
    )
    if directive.first_person_policy != AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED:
        raise LinkedInPostPipelineContractError(
            "AngleDecision.authorial_voice_directive.first_person_policy "
            f"must be {AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED}."
        )
    _require_unique_non_empty_string_tuple(
        directive.forbidden_author_claims,
        "AngleDecision.authorial_voice_directive.forbidden_author_claims",
    )
    if directive.forbidden_author_claims != AUTHORIAL_FORBIDDEN_AUTHOR_CLAIMS:
        raise LinkedInPostPipelineContractError(
            "AngleDecision.authorial_voice_directive.forbidden_author_claims "
            "must match the canonical forbidden author claims."
        )


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
    selected_reference_ids = _extract_selected_evidence_reference_ids(
        final_post_payload.post_text
    )
    for item in post_brief.evidence_to_use:
        if item.evidence_id not in selected_reference_ids:
            raise LinkedInPostPipelineContractError(
                "FinalPostPayload.post_text must include selected evidence ID "
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
    if len(payload.post_text) > FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS:
        raise LinkedInPostPipelineContractError(
            "FinalPostPayload.post_text must not exceed "
            f"{FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS} characters."
        )
    _require_string_list(
        payload.hook_variants,
        "FinalPostPayload.hook_variants",
        min_items=FINAL_POST_PAYLOAD_HOOK_VARIANTS_MIN_COUNT,
    )
    _require_string_list(
        payload.cta_variants,
        "FinalPostPayload.cta_variants",
        min_items=FINAL_POST_PAYLOAD_CTA_VARIANTS_MIN_COUNT,
    )
    _require_string_list(
        payload.hashtags,
        "FinalPostPayload.hashtags",
        min_items=FINAL_POST_PAYLOAD_HASHTAGS_MIN_COUNT,
    )
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


def _require_unique_non_empty_string_list(
    value: Any,
    field_name: str,
    min_items: int = 1,
) -> None:
    _require_string_list(value, field_name, min_items=min_items)
    if len(set(value)) != len(value):
        raise LinkedInPostPipelineContractError(
            f"{field_name} must not contain duplicate strings."
        )


def _require_unique_non_empty_string_tuple(
    value: Any,
    field_name: str,
    min_items: int = 1,
) -> None:
    if not isinstance(value, tuple):
        raise LinkedInPostPipelineContractError(f"{field_name} must be a tuple.")
    if len(value) < min_items:
        raise LinkedInPostPipelineContractError(
            f"{field_name} must contain at least {min_items} items."
        )
    for item_index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise LinkedInPostPipelineContractError(
                f"{field_name}[{item_index}] must be a non-empty string."
            )
    if len(set(value)) != len(value):
        raise LinkedInPostPipelineContractError(
            f"{field_name} must not contain duplicate strings."
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


EDITORIAL_SIGNAL_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "adoption interest",
        (
            "adoption",
            "consumer",
            "integrated",
            "leverage",
            "accessibility",
            "own crypto",
            "owners",
            "users",
        ),
    ),
    (
        "security confidence",
        (
            "security",
            "confidence",
            "trust",
            "volatility",
            "concern",
        ),
    ),
    (
        "market forecast",
        (
            "forecast",
            "projected",
            "projection",
            "cagr",
            "growth",
            "grow",
        ),
    ),
    (
        "price positioning",
        (
            "bottom",
            "downside",
            "bear",
            "pessimistic",
            "trader",
        ),
    ),
    (
        "regulatory clarity",
        (
            "regulation",
            "regulatory",
        ),
    ),
    (
        "technology reliability",
        (
            "technological advancements",
            "transaction efficiency",
            "efficiency and security",
        ),
    ),
    (
        "workplace expectations",
        (
            "remote",
            "hybrid",
            "workplace",
            "expectations",
            "communication",
            "isolation",
            "mental health",
        ),
    ),
    (
        "inclusion",
        (
            "diversity",
            "inclusion",
            "inclusive",
        ),
    ),
    (
        "pricing strategy",
        (
            "pricing strategy",
            "value-based pricing",
            "product value",
            "customer perceptions",
            "profitability",
        ),
    ),
    (
        "freelance client positioning",
        (
            "freelance",
            "freelancers",
            "client acquisition",
            "international clients",
            "specialization",
            "personal brand",
        ),
    ),
    (
        "education impact",
        (
            "education",
            "students",
            "teens",
            "school",
            "learning",
        ),
    ),
    (
        "work decisions",
        (
            "decisions",
            "tradeoffs",
            "lessons",
            "visible process",
        ),
    ),
    (
        "surface outcomes",
        (
            "finished outcomes",
            "polished output",
            "surface-level",
        ),
    ),
)


EDITORIAL_SYNTHESIS_STATUS_READY = "READY"
EDITORIAL_SYNTHESIS_STATUS_NON_READY = "NON_READY"

EDITORIAL_SYNTHESIS_REASON_INSUFFICIENT_SELECTED_EVIDENCE = (
    "INSUFFICIENT_SELECTED_EVIDENCE"
)
EDITORIAL_SYNTHESIS_REASON_NO_SUPPORTED_RELATIONSHIP = "NO_SUPPORTED_RELATIONSHIP"
EDITORIAL_SYNTHESIS_REASON_ONLY_SIGNAL_INVENTORY = "ONLY_SIGNAL_INVENTORY"
EDITORIAL_SYNTHESIS_REASON_AMBIGUOUS_DOMINANT_RELATIONSHIP = (
    "AMBIGUOUS_DOMINANT_RELATIONSHIP"
)
EDITORIAL_SYNTHESIS_REASON_UNSAFE_CAUSAL_SYNTHESIS = "UNSAFE_CAUSAL_SYNTHESIS"
EDITORIAL_SYNTHESIS_NON_READY_REASONS = {
    EDITORIAL_SYNTHESIS_REASON_INSUFFICIENT_SELECTED_EVIDENCE,
    EDITORIAL_SYNTHESIS_REASON_NO_SUPPORTED_RELATIONSHIP,
    EDITORIAL_SYNTHESIS_REASON_ONLY_SIGNAL_INVENTORY,
    EDITORIAL_SYNTHESIS_REASON_AMBIGUOUS_DOMINANT_RELATIONSHIP,
    EDITORIAL_SYNTHESIS_REASON_UNSAFE_CAUSAL_SYNTHESIS,
}

RELATIONSHIP_PRIORITY: dict[str, int] = {
    "growth_vs_constraint": 1,
    "interest_vs_confidence": 2,
    "forecast_vs_current_condition": 3,
    "expectation_vs_behavior": 4,
    "adoption_vs_infrastructure": 5,
}

METRIC_OR_QUALIFIER_TERMS = (
    "%",
    "projected",
    "forecast",
    "likely",
    "may",
    "suggest",
    "risk",
    "concern",
)


def build_editorial_synthesis_result_for_selected_items(
    selected_items: list[ContextualEvidence],
) -> EditorialSynthesisResult:
    selected_evidence_ids = [item.evidence_id for item in selected_items]
    if not selected_items:
        return _non_ready_editorial_synthesis_result(
            selected_evidence_ids,
            EDITORIAL_SYNTHESIS_REASON_INSUFFICIENT_SELECTED_EVIDENCE,
            "No selected evidence items were provided.",
        )

    signal_evidence_ids = _editorial_signal_evidence_ids_for_selected_items(
        selected_items
    )
    signal_labels = list(signal_evidence_ids)
    if not signal_labels:
        return _non_ready_editorial_synthesis_result(
            selected_evidence_ids,
            EDITORIAL_SYNTHESIS_REASON_NO_SUPPORTED_RELATIONSHIP,
            "Selected evidence did not contain recognized editorial signals.",
        )

    raw_candidates = _build_evidence_relationship_candidates(
        selected_items,
        signal_evidence_ids,
    )
    candidates, rejected_reasons = _validated_relationship_candidates(raw_candidates)
    if not candidates:
        if EDITORIAL_SYNTHESIS_REASON_UNSAFE_CAUSAL_SYNTHESIS in rejected_reasons:
            reason = EDITORIAL_SYNTHESIS_REASON_UNSAFE_CAUSAL_SYNTHESIS
        else:
            reason = (
                EDITORIAL_SYNTHESIS_REASON_INSUFFICIENT_SELECTED_EVIDENCE
                if len(selected_items) == 1
                else EDITORIAL_SYNTHESIS_REASON_ONLY_SIGNAL_INVENTORY
            )
        return _non_ready_editorial_synthesis_result(
            selected_evidence_ids,
            reason,
            "Recognized signals did not form an approved evidence relationship.",
        )

    sorted_candidates = sorted(
        candidates,
        key=lambda relationship: (-relationship.score, relationship.priority),
    )
    dominant_relationship = sorted_candidates[0]
    tied = [
        relationship
        for relationship in sorted_candidates[1:]
        if relationship.score == dominant_relationship.score
        and relationship.priority == dominant_relationship.priority
        and relationship.thesis != dominant_relationship.thesis
    ]
    if tied:
        return EditorialSynthesisResult(
            status=EDITORIAL_SYNTHESIS_STATUS_NON_READY,
            dominant_relationship=None,
            candidate_relationships=sorted_candidates,
            selected_evidence_ids=selected_evidence_ids,
            non_ready_reason=EDITORIAL_SYNTHESIS_REASON_AMBIGUOUS_DOMINANT_RELATIONSHIP,
            non_ready_detail="Multiple incompatible relationships tied for dominance.",
        )

    result = EditorialSynthesisResult(
        status=EDITORIAL_SYNTHESIS_STATUS_READY,
        dominant_relationship=dominant_relationship,
        candidate_relationships=sorted_candidates,
        selected_evidence_ids=selected_evidence_ids,
    )
    validate_editorial_synthesis_result(result)
    return result


def _build_reader_takeaway_for_selected_items(
    selected_items: list[ContextualEvidence],
    angle_decision: AngleDecision,
) -> str:
    synthesis_result = build_editorial_synthesis_result_for_selected_items(
        selected_items
    )
    if (
        synthesis_result.status == EDITORIAL_SYNTHESIS_STATUS_READY
        and synthesis_result.dominant_relationship is not None
    ):
        relationship = synthesis_result.dominant_relationship
        return _practical_point_for_relationship(relationship)
    return (
        "The reader takeaway should follow the author position: "
        f"{angle_decision.author_position}"
    )


def _build_evidence_relationship_candidates(
    selected_items: list[ContextualEvidence],
    signal_evidence_ids: dict[str, list[str]],
) -> list[EvidenceRelationship]:
    topic_label = _topic_label_for_selected_items(selected_items)
    candidates: list[EvidenceRelationship] = []

    candidates.extend(
        _build_growth_vs_constraint_candidates(
            selected_items,
            signal_evidence_ids,
            topic_label,
        )
    )
    candidates.extend(
        _build_interest_vs_confidence_candidates(
            selected_items,
            signal_evidence_ids,
            topic_label,
        )
    )
    candidates.extend(
        _build_forecast_vs_current_condition_candidates(
            selected_items,
            signal_evidence_ids,
            topic_label,
        )
    )
    candidates.extend(
        _build_expectation_vs_behavior_candidates(
            selected_items,
            signal_evidence_ids,
            topic_label,
        )
    )
    return candidates


def _build_growth_vs_constraint_candidates(
    selected_items: list[ContextualEvidence],
    signal_evidence_ids: dict[str, list[str]],
    topic_label: str,
) -> list[EvidenceRelationship]:
    growth_ids = _ids_for_labels(
        signal_evidence_ids,
        ["adoption interest", "market forecast"],
    )
    constraint_ids = _ids_for_labels(
        signal_evidence_ids,
        ["security confidence", "price positioning"],
    )
    if not growth_ids or not constraint_ids:
        return []

    thesis = (
        f"The stronger {topic_label} angle is not that adoption or forecasts "
        "settle the market story, but that visible growth evidence remains "
        "conditional on confidence and risk evidence."
    )
    relationship = EvidenceRelationship(
        relationship_type="growth_vs_constraint",
        left_label="growth evidence",
        right_label="confidence and risk evidence",
        left_evidence_ids=growth_ids,
        right_evidence_ids=constraint_ids,
        supporting_evidence_ids=_unique_preserving_order(growth_ids + constraint_ids),
        qualifier=_qualifier_for_evidence_ids(selected_items, growth_ids + constraint_ids),
        thesis=thesis,
        reader_problem=(
            "Readers may treat ownership, investment intent, or market forecasts "
            "as one clean growth story before checking the confidence and risk evidence."
        ),
        author_position=(
            f"The author position is that {topic_label} growth should be read as "
            "conditional: adoption and forecasts matter, but security concerns, "
            "volatility, and positioning risk still shape what the evidence can support."
        ),
        main_tension=(
            "The tension is between visible growth evidence and unresolved "
            "confidence or risk evidence."
        ),
        score=_relationship_score(selected_items, growth_ids, constraint_ids),
        priority=RELATIONSHIP_PRIORITY["growth_vs_constraint"],
    )
    return [relationship]


def _build_interest_vs_confidence_candidates(
    selected_items: list[ContextualEvidence],
    signal_evidence_ids: dict[str, list[str]],
    topic_label: str,
) -> list[EvidenceRelationship]:
    interest_ids = _ids_for_labels(signal_evidence_ids, ["adoption interest"])
    confidence_ids = _ids_for_labels(signal_evidence_ids, ["security confidence"])
    if not interest_ids or not confidence_ids:
        return []

    relationship = EvidenceRelationship(
        relationship_type="interest_vs_confidence",
        left_label="participation interest",
        right_label="confidence barrier",
        left_evidence_ids=interest_ids,
        right_evidence_ids=confidence_ids,
        supporting_evidence_ids=_unique_preserving_order(interest_ids + confidence_ids),
        qualifier=_qualifier_for_evidence_ids(
            selected_items,
            interest_ids + confidence_ids,
        ),
        thesis=(
            f"The stronger {topic_label} angle is that participation interest "
            "can rise before confidence catches up."
        ),
        reader_problem=(
            "Readers may treat participation interest as proof of broad trust "
            "before checking the confidence barrier."
        ),
        author_position=(
            f"The author position is that {topic_label} participation should not "
            "be confused with durable confidence."
        ),
        main_tension=(
            "The tension is between visible participation interest and unresolved "
            "confidence barriers."
        ),
        score=_relationship_score(selected_items, interest_ids, confidence_ids),
        priority=RELATIONSHIP_PRIORITY["interest_vs_confidence"],
    )
    return [relationship]


def _build_forecast_vs_current_condition_candidates(
    selected_items: list[ContextualEvidence],
    signal_evidence_ids: dict[str, list[str]],
    topic_label: str,
) -> list[EvidenceRelationship]:
    forecast_ids = _ids_for_labels(signal_evidence_ids, ["market forecast"])
    condition_ids = _ids_for_labels(signal_evidence_ids, ["price positioning"])
    if not forecast_ids or not condition_ids:
        return []

    relationship = EvidenceRelationship(
        relationship_type="forecast_vs_current_condition",
        left_label="long-range forecast",
        right_label="current positioning risk",
        left_evidence_ids=forecast_ids,
        right_evidence_ids=condition_ids,
        supporting_evidence_ids=_unique_preserving_order(forecast_ids + condition_ids),
        qualifier=_qualifier_for_evidence_ids(
            selected_items,
            forecast_ids + condition_ids,
        ),
        thesis=(
            f"The stronger {topic_label} angle is that a long-range forecast does "
            "not erase the qualified short-term positioning evidence."
        ),
        reader_problem=(
            "Readers may treat a projected growth path as a current-market answer "
            "even when current positioning remains qualified."
        ),
        author_position=(
            f"The author position is that {topic_label} forecasts and current "
            "positioning evidence answer different questions."
        ),
        main_tension=(
            "The tension is between projected long-range growth and qualified "
            "current positioning evidence."
        ),
        score=_relationship_score(selected_items, forecast_ids, condition_ids),
        priority=RELATIONSHIP_PRIORITY["forecast_vs_current_condition"],
    )
    return [relationship]


def _build_expectation_vs_behavior_candidates(
    selected_items: list[ContextualEvidence],
    signal_evidence_ids: dict[str, list[str]],
    topic_label: str,
) -> list[EvidenceRelationship]:
    candidates: list[EvidenceRelationship] = []
    relationship_pairs = [
        (
            "workplace expectations",
            "inclusion",
            "workplace rules",
            "inclusive work design",
            (
                "Workplace expectations are not durable on their own; they need "
                "inclusive work design to become usable."
            ),
        ),
        (
            "pricing strategy",
            "freelance client positioning",
            "pricing strategy",
            "client positioning",
            (
                "Freelance pricing works best as a positioning choice, not just "
                "as a number on an offer."
            ),
        ),
        (
            "education impact",
            "adoption interest",
            "education impact",
            "learner adoption",
            (
                "Education impact becomes more concrete when learner adoption is "
                "visible, but adoption alone does not explain educational value."
            ),
        ),
        (
            "work decisions",
            "surface outcomes",
            "visible decisions",
            "surface polish",
            (
                "proof of judgment comes from visible decisions, not surface polish."
            ),
        ),
        (
            "technology reliability",
            "security confidence",
            "technology reliability",
            "security confidence",
            (
                "technology reliability matters alongside confidence, not just "
                "as an efficiency claim."
            ),
        ),
    ]
    for left_signal, right_signal, left_label, right_label, thesis in relationship_pairs:
        left_ids = _ids_for_labels(signal_evidence_ids, [left_signal])
        right_ids = _ids_for_labels(signal_evidence_ids, [right_signal])
        if not left_ids or not right_ids:
            continue
        relationship = EvidenceRelationship(
            relationship_type="expectation_vs_behavior",
            left_label=left_label,
            right_label=right_label,
            left_evidence_ids=left_ids,
            right_evidence_ids=right_ids,
            supporting_evidence_ids=_unique_preserving_order(left_ids + right_ids),
            qualifier=_qualifier_for_evidence_ids(selected_items, left_ids + right_ids),
            thesis=f"The stronger {topic_label} angle is that {thesis}",
            reader_problem=(
                f"Readers may treat {left_label} as the whole story before "
                f"checking {right_label}."
            ),
            author_position=(
                f"The author position is that {topic_label} needs both "
                f"{left_label} and {right_label} to become a useful takeaway."
            ),
            main_tension=(
                f"The tension is between {left_label} and {right_label}."
            ),
            score=_relationship_score(selected_items, left_ids, right_ids),
            priority=RELATIONSHIP_PRIORITY["expectation_vs_behavior"],
        )
        candidates.append(relationship)
    return candidates


def _validated_relationship_candidates(
    relationships: list[EvidenceRelationship],
) -> tuple[list[EvidenceRelationship], list[str]]:
    valid_relationships: list[EvidenceRelationship] = []
    rejected_reasons: list[str] = []
    for relationship in relationships:
        try:
            validate_evidence_relationship(relationship)
        except LinkedInPostPipelineContractError as exc:
            if "unsupported causal strengthening" in str(exc):
                rejected_reasons.append(
                    EDITORIAL_SYNTHESIS_REASON_UNSAFE_CAUSAL_SYNTHESIS
                )
            continue
        valid_relationships.append(relationship)
    return valid_relationships, _unique_preserving_order(rejected_reasons)


def validate_evidence_relationship(relationship: EvidenceRelationship) -> None:
    _require_non_empty_string(
        relationship.relationship_type,
        "EvidenceRelationship.relationship_type",
    )
    _require_non_empty_string(relationship.left_label, "EvidenceRelationship.left_label")
    _require_non_empty_string(
        relationship.right_label,
        "EvidenceRelationship.right_label",
    )
    _require_non_empty_string(relationship.thesis, "EvidenceRelationship.thesis")
    _require_non_empty_string(
        relationship.reader_problem,
        "EvidenceRelationship.reader_problem",
    )
    _require_non_empty_string(
        relationship.author_position,
        "EvidenceRelationship.author_position",
    )
    _require_non_empty_string(
        relationship.main_tension,
        "EvidenceRelationship.main_tension",
    )
    _require_unique_non_empty_string_list(
        relationship.left_evidence_ids,
        "EvidenceRelationship.left_evidence_ids",
    )
    _require_unique_non_empty_string_list(
        relationship.right_evidence_ids,
        "EvidenceRelationship.right_evidence_ids",
    )
    _require_unique_non_empty_string_list(
        relationship.supporting_evidence_ids,
        "EvidenceRelationship.supporting_evidence_ids",
    )
    if not set(relationship.left_evidence_ids).issubset(
        set(relationship.supporting_evidence_ids)
    ):
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship.left_evidence_ids must be included in supporting_evidence_ids."
        )
    if not set(relationship.right_evidence_ids).issubset(
        set(relationship.supporting_evidence_ids)
    ):
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship.right_evidence_ids must be included in supporting_evidence_ids."
        )
    if (
        len(set(relationship.supporting_evidence_ids)) < 2
        and set(relationship.left_evidence_ids) != set(relationship.right_evidence_ids)
    ):
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship must use at least two selected evidence items unless one item supports both sides."
        )
    if _contains_unsupported_causal_strengthening(
        "\n".join(
            [
                relationship.thesis,
                relationship.reader_problem,
                relationship.author_position,
                relationship.main_tension,
            ]
        )
    ):
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship must not introduce unsupported causal strengthening."
        )
    if relationship.score < 0:
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship.score must be non-negative."
        )
    if relationship.priority <= 0:
        raise LinkedInPostPipelineContractError(
            "EvidenceRelationship.priority must be positive."
        )


def validate_editorial_synthesis_result(result: EditorialSynthesisResult) -> None:
    if result.status not in {
        EDITORIAL_SYNTHESIS_STATUS_READY,
        EDITORIAL_SYNTHESIS_STATUS_NON_READY,
    }:
        raise LinkedInPostPipelineContractError(
            "EditorialSynthesisResult.status must be READY or NON_READY."
        )
    for relationship in result.candidate_relationships:
        validate_evidence_relationship(relationship)
    if result.status == EDITORIAL_SYNTHESIS_STATUS_READY:
        _require_unique_non_empty_string_list(
            result.selected_evidence_ids,
            "EditorialSynthesisResult.selected_evidence_ids",
        )
        if result.non_ready_reason or result.non_ready_detail:
            raise LinkedInPostPipelineContractError(
                "Ready EditorialSynthesisResult must not include non-ready fields."
            )
        if result.dominant_relationship is None:
            raise LinkedInPostPipelineContractError(
                "Ready EditorialSynthesisResult must include a dominant relationship."
            )
        validate_evidence_relationship(result.dominant_relationship)
        selected_evidence_ids = set(result.selected_evidence_ids)
        _require_relationship_ids_subset_selected(
            result.dominant_relationship,
            selected_evidence_ids,
            "EditorialSynthesisResult.dominant_relationship",
        )
        if result.dominant_relationship not in result.candidate_relationships:
            raise LinkedInPostPipelineContractError(
                "Ready EditorialSynthesisResult dominant relationship must be included in candidate_relationships."
            )
        for index, relationship in enumerate(result.candidate_relationships):
            _require_relationship_ids_subset_selected(
                relationship,
                selected_evidence_ids,
                f"EditorialSynthesisResult.candidate_relationships[{index}]",
            )
        return
    _require_string_list(
        result.selected_evidence_ids,
        "EditorialSynthesisResult.selected_evidence_ids",
    )
    if len(set(result.selected_evidence_ids)) != len(result.selected_evidence_ids):
        raise LinkedInPostPipelineContractError(
            "EditorialSynthesisResult.selected_evidence_ids must not contain duplicate strings."
        )
    if result.dominant_relationship is not None:
        raise LinkedInPostPipelineContractError(
            "Non-ready EditorialSynthesisResult must not include a dominant relationship."
        )
    selected_evidence_ids = set(result.selected_evidence_ids)
    for index, relationship in enumerate(result.candidate_relationships):
        _require_relationship_ids_subset_selected(
            relationship,
            selected_evidence_ids,
            f"EditorialSynthesisResult.candidate_relationships[{index}]",
        )
    _require_non_empty_string(
        result.non_ready_reason,
        "EditorialSynthesisResult.non_ready_reason",
    )
    if result.non_ready_reason not in EDITORIAL_SYNTHESIS_NON_READY_REASONS:
        allowed = ", ".join(sorted(EDITORIAL_SYNTHESIS_NON_READY_REASONS))
        raise LinkedInPostPipelineContractError(
            "EditorialSynthesisResult.non_ready_reason must be one of: "
            f"{allowed}."
        )


def _require_relationship_ids_subset_selected(
    relationship: EvidenceRelationship,
    selected_evidence_ids: set[str],
    field_name: str,
) -> None:
    relationship_ids = set(
        relationship.left_evidence_ids
        + relationship.right_evidence_ids
        + relationship.supporting_evidence_ids
    )
    unknown_ids = sorted(relationship_ids - selected_evidence_ids)
    if unknown_ids:
        raise LinkedInPostPipelineContractError(
            f"{field_name} contains evidence IDs outside selected_evidence_ids: "
            f"{unknown_ids}."
        )


def _non_ready_editorial_synthesis_result(
    selected_evidence_ids: list[str],
    reason: str,
    detail: str,
) -> EditorialSynthesisResult:
    return EditorialSynthesisResult(
        status=EDITORIAL_SYNTHESIS_STATUS_NON_READY,
        dominant_relationship=None,
        candidate_relationships=[],
        selected_evidence_ids=selected_evidence_ids,
        non_ready_reason=reason,
        non_ready_detail=detail,
    )


def _editorial_signal_evidence_ids_for_selected_items(
    selected_items: list[ContextualEvidence],
) -> dict[str, list[str]]:
    signal_evidence_ids: dict[str, list[str]] = {}
    has_market_context = any(
        _contains_market_context(
            _source_grounded_editorial_text_blob_for_evidence_item(item)
        )
        for item in selected_items
    )
    for item in selected_items:
        item_text = _source_grounded_editorial_text_blob_for_evidence_item(item)
        for label, keywords in EDITORIAL_SIGNAL_KEYWORDS:
            if _is_market_specific_signal_label(label) and not has_market_context:
                continue
            if any(keyword in item_text for keyword in keywords):
                signal_evidence_ids.setdefault(label, []).append(item.evidence_id)
    return {
        label: _unique_preserving_order(evidence_ids)
        for label, evidence_ids in signal_evidence_ids.items()
    }


def _ids_for_labels(
    signal_evidence_ids: dict[str, list[str]],
    labels: list[str],
) -> list[str]:
    evidence_ids: list[str] = []
    for label in labels:
        evidence_ids.extend(signal_evidence_ids.get(label, []))
    return _unique_preserving_order(evidence_ids)


def _relationship_score(
    selected_items: list[ContextualEvidence],
    left_evidence_ids: list[str],
    right_evidence_ids: list[str],
) -> int:
    supporting_ids = _unique_preserving_order(left_evidence_ids + right_evidence_ids)
    items_by_id = {item.evidence_id: item for item in selected_items}
    score = len(supporting_ids) * 4
    if set(left_evidence_ids).intersection(right_evidence_ids):
        score += 3
    source_indexes = {
        items_by_id[evidence_id].source_index
        for evidence_id in supporting_ids
        if evidence_id in items_by_id
    }
    if len(source_indexes) > 1:
        score += 2
    for evidence_id in supporting_ids:
        item = items_by_id.get(evidence_id)
        if item is not None and _is_metric_or_qualified_item(item):
            score += 1
    return score


def _is_metric_or_qualified_item(item: ContextualEvidence) -> bool:
    item_text = _source_evidence_support_text_blob_for_evidence_item(item)
    return any(term in item_text for term in METRIC_OR_QUALIFIER_TERMS)


def _qualifier_for_evidence_ids(
    selected_items: list[ContextualEvidence],
    evidence_ids: list[str],
) -> str:
    values: list[str] = []
    items_by_id = {item.evidence_id: item for item in selected_items}
    for evidence_id in evidence_ids:
        item = items_by_id.get(evidence_id)
        if item is None:
            continue
        item_text = _source_evidence_support_text_blob_for_evidence_item(item)
        for qualifier in ["projected", "likely", "may", "suggest", "concern", "risk"]:
            if qualifier in item_text and qualifier not in values:
                values.append(qualifier)
    return ", ".join(values)


def _practical_point_for_relationship(relationship: EvidenceRelationship) -> str:
    if relationship.relationship_type == "growth_vs_constraint":
        return (
            "Before treating the market story as settled, check whether growth "
            "evidence is being qualified by confidence or risk evidence."
        )
    if relationship.relationship_type == "forecast_vs_current_condition":
        return (
            "Use the forecast as long-range context and keep current positioning "
            "qualified."
        )
    return (
        f"Turn the post toward the relationship between {relationship.left_label} "
        f"and {relationship.right_label}."
    )


def _contains_unsupported_causal_strengthening(text: str) -> bool:
    lowered = text.lower()
    forbidden_phrases = [
        "causes",
        "caused",
        "drives",
        "driven by",
        "guarantees",
        "will lead to",
        "leads to",
        "lead to",
        "will create",
        "will prevent",
        "prevents downturns",
        "prevents declines",
        "creates stability",
        "led to stability",
    ]
    return any(phrase in lowered for phrase in forbidden_phrases)


def _topic_label_for_selected_items(selected_items: list[ContextualEvidence]) -> str:
    combined_text = " ".join(
        _editorial_text_blob_for_evidence_item(item) for item in selected_items
    )
    if any(term in combined_text for term in ("bitcoin", "crypto", "cryptocurrency")):
        return "crypto-market"
    if any(term in combined_text for term in ("remote", "hybrid", "workplace")):
        return "workplace"
    if any(term in combined_text for term in ("education", "students", "teens")):
        return "education"
    if "freelanc" in combined_text:
        return "freelancing"
    return "reader"


def _is_market_specific_signal_label(label: str) -> bool:
    return label in {
        "security confidence",
        "market forecast",
        "price positioning",
    }


def _contains_market_context(text: str) -> bool:
    return any(
        term in text
        for term in (
            "bitcoin",
            "crypto",
            "cryptocurrency",
            "finance",
            "financial",
            "investment",
            "investor",
            "trading",
            "trader",
        )
    )


def _is_market_projection_or_positioning(item: ContextualEvidence | ArticleEvidence) -> bool:
    item_text = _source_evidence_support_text_blob_for_evidence_item(item)
    explicit_projection_terms = (
        "forecast",
        "forecasted",
        "projected",
        "projection",
        "cagr",
        "compound annual growth",
    )
    explicit_positioning_terms = (
        "bottomed",
        "downside",
        "bear case",
        "bull case",
        "pessimistic trader",
        "trader sentiment",
    )
    if any(term in item_text for term in explicit_projection_terms):
        return True
    if any(term in item_text for term in explicit_positioning_terms):
        return True
    has_market_context = any(
        term in item_text
        for term in (
            "bitcoin",
            "crypto",
            "cryptocurrency",
            "finance",
            "financial",
            "investment",
            "investor",
            "market",
            "stock",
            "trading",
            "trader",
        )
    )
    return has_market_context and "price" in item_text


def _editorial_text_blob_for_evidence_item(
    item: ContextualEvidence | ArticleEvidence,
) -> str:
    values = [item.source_title, item.evidence_text, item.evidence_type]
    if isinstance(item, ContextualEvidence):
        values.extend(
            [
                item.what_it_says,
                item.supports_argument,
            ]
        )
    return " ".join(values).lower()


def _source_grounded_editorial_text_blob_for_evidence_item(
    item: ContextualEvidence | ArticleEvidence,
) -> str:
    values = [item.source_title, item.evidence_text]
    if isinstance(item, ContextualEvidence):
        values.append(item.what_it_says)
    return " ".join(values).lower()


def _source_evidence_support_text_blob_for_evidence_item(
    item: ContextualEvidence | ArticleEvidence,
) -> str:
    values = [item.source_title, item.evidence_text]
    if isinstance(item, ContextualEvidence):
        values.extend([item.what_it_says, item.supports_argument])
    return " ".join(values).lower()


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
    best_use = item.best_use_in_post
    supports_argument = item.supports_argument
    if best_use == "practical_point" and _is_market_projection_or_positioning(item):
        best_use = "proof"
        supports_argument = (
            "Treat market projection or positioning evidence as context for the "
            "thesis, not as a reader instruction."
        )
    return (
        f"Use this evidence as {best_use} support. "
        f"{supports_argument}"
    )


def _compact_role_label(role_in_post: str) -> str:
    role = role_in_post.strip()
    prefix = "Use this evidence as "
    if role.startswith(prefix):
        role = role[len(prefix) :]
    role = role.split(".", 1)[0]
    role = role.replace(" support", "").strip()
    return role or "support"


def _extract_selected_evidence_reference_ids(post_text: str) -> set[str]:
    prefix = "Selected evidence references:"
    for line in post_text.splitlines():
        stripped_line = line.strip()
        if not stripped_line.startswith(prefix):
            continue

        raw_references = stripped_line[len(prefix) :].strip()
        if not raw_references:
            return set()

        reference_ids: set[str] = set()
        for reference in raw_references.split(","):
            reference_label = reference.strip()
            evidence_id = reference_label.split(" (", 1)[0].strip()
            if evidence_id:
                reference_ids.add(evidence_id)
        return reference_ids

    raise LinkedInPostPipelineContractError(
        "FinalPostPayload.post_text must include the selected evidence references line."
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
    "AUTHORIAL_FIRST_PERSON_ALLOWED_NOT_REQUIRED",
    "AUTHORIAL_FORBIDDEN_AUTHOR_CLAIMS",
    "AUTHORIAL_PERSONAL_PRESENCE_ALLOWED",
    "AUTHORIAL_PERSONAL_PRESENCE_EDITORIAL_ONLY",
    "AUTHORIAL_PERSONAL_PRESENCE_REQUIRED",
    "AUTHORIAL_PERSONAL_PRESENCE_REQUIREMENTS",
    "REQUIRED_QUALITY_CHECKS",
    "AngleDecision",
    "ArticleEvidence",
    "ArticleEvidencePack",
    "AuthorialVoiceDirective",
    "BriefEvidenceUse",
    "ContextualEvidence",
    "ContextualEvidencePack",
    "EditorialSynthesisResult",
    "EvidenceRelationship",
    "FinalPostPayload",
    "LinkedInPostPipelineContractError",
    "PipelineInput",
    "PostBrief",
    "QualityReviewResult",
    "SelectedArticle",
    "TargetedRepairPlan",
    "build_angle_decision_from_contextual_evidence_pack",
    "build_article_evidence_pack_from_pipeline_input",
    "build_authorial_voice_directive_from_evidence_relationship",
    "build_contextual_evidence_pack_from_article_evidence_pack",
    "build_editorial_synthesis_result_for_selected_items",
    "build_final_post_payload_from_post_brief",
    "build_final_post_payload_constraints",
    "build_pipeline_input_from_digest",
    "build_post_brief_from_angle_decision",
    "final_post_payload_to_dict",
    "validate_angle_decision",
    "validate_angle_decision_for_contextual_evidence",
    "validate_article_evidence_pack",
    "validate_article_evidence_pack_for_pipeline_input",
    "validate_authorial_voice_directive",
    "validate_contextual_evidence_pack",
    "validate_contextual_evidence_pack_for_article_evidence",
    "validate_editorial_synthesis_result",
    "validate_evidence_relationship",
    "validate_final_post_payload",
    "validate_final_post_payload_for_post_brief",
    "validate_linkedin_post_stage_relationships",
    "validate_pipeline_input",
    "validate_post_brief",
    "validate_post_brief_for_angle_decision",
    "validate_selected_articles",
]
