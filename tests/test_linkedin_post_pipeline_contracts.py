from django.test import SimpleTestCase

from services.packaging.linkedin_post_pipeline import (
    AngleDecision,
    ArticleEvidence,
    ArticleEvidencePack,
    BriefEvidenceUse,
    ContextualEvidence,
    ContextualEvidencePack,
    FinalPostPayload,
    LinkedInPostPipelineContractError,
    PipelineInput,
    PostBrief,
    SelectedArticle,
    final_post_payload_to_dict,
    validate_angle_decision,
    validate_article_evidence_pack,
    validate_contextual_evidence_pack,
    validate_final_post_payload,
    validate_pipeline_input,
    validate_post_brief,
    validate_selected_articles,
)
from services.packaging.validators import validate_content_package_payload


def make_selected_article(
    source_index: int = 0,
    summary: str = "The article explains why visible work examples matter.",
) -> SelectedArticle:
    return SelectedArticle(
        source_index=source_index,
        title=f"Article {source_index}",
        url=f"https://example.com/article-{source_index}",
        summary=summary,
        key_points=["Show decisions, tradeoffs, and lessons."],
        source_name="Example Source",
        content_type="article",
        confidence=0.8,
    )


def make_pipeline_input() -> PipelineInput:
    return PipelineInput(
        digest_id=130,
        topic_name="Personal Branding",
        digest_title="Digest for Personal Branding",
        articles=[make_selected_article()],
        author_profile={"role": "AI Automation Specialist"},
    )


def make_article_evidence(
    evidence_type: str = "contrast",
    specificity_level: str = "medium",
) -> ArticleEvidence:
    return ArticleEvidence(
        evidence_id="e1",
        source_index=0,
        source_title="Article 0",
        evidence_text="Recent posts can show decisions and tradeoffs, not just finished outcomes.",
        evidence_type=evidence_type,
        specificity_level=specificity_level,
        source_limitations="No named case or metric.",
    )


def make_contextual_evidence(best_use_in_post: str = "tension") -> ContextualEvidence:
    return ContextualEvidence(
        evidence_id="e1",
        evidence_text="Recent posts can show decisions and tradeoffs, not just finished outcomes.",
        what_it_says="Visible process helps people evaluate judgment.",
        supports_argument="The post can argue that public proof needs more than polished output.",
        best_use_in_post=best_use_in_post,
        do_not_use_for="Do not turn this into generic personal branding advice.",
        risk_of_misuse="Could drift into surface-level branding language.",
    )


def make_final_post_payload(
    post_text: str = "A polished profile is weak proof.",
    hook_variants: list[str] | None = None,
    quality_checks: dict[str, bool] | None = None,
) -> FinalPostPayload:
    return FinalPostPayload(
        post_text=post_text,
        hook_variants=hook_variants
        if hook_variants is not None
        else [
            "A polished profile is weak proof.",
            "Finished work can hide the important part.",
            "Your last ten posts may reveal less than you think.",
        ],
        cta_variants=[
            "What kind of proof do you look for?",
            "What makes expertise feel real to you?",
            "Which matters more: the result or the decision behind it?",
        ],
        hashtags=["#PersonalBranding"],
        quality_checks=quality_checks
        if quality_checks is not None
        else {
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
            "linkedin_ready": True,
        },
        carousel_outline=[],
    )


class LinkedInPostPipelineContractTests(SimpleTestCase):
    def test_validate_pipeline_input_accepts_valid_input(self) -> None:
        validate_pipeline_input(make_pipeline_input())

    def test_validate_pipeline_input_rejects_non_dict_author_profile(self) -> None:
        pipeline_input = PipelineInput(
            digest_id=130,
            topic_name="Personal Branding",
            digest_title="Digest for Personal Branding",
            articles=[make_selected_article()],
            author_profile=[],  # type: ignore[arg-type]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_pipeline_input(pipeline_input)

    def test_validate_selected_articles_rejects_non_sequential_source_index(self) -> None:
        articles = [
            make_selected_article(source_index=0),
            make_selected_article(source_index=3),
        ]

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_selected_articles(articles)

    def test_validate_selected_articles_rejects_empty_summary(self) -> None:
        article = make_selected_article(summary=" ")

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_selected_articles([article])

    def test_validate_article_evidence_pack_accepts_valid_pack(self) -> None:
        pack = ArticleEvidencePack(
            items=[make_article_evidence()],
            usable_count=1,
            rejected_count=0,
        )

        validate_article_evidence_pack(pack)

    def test_validate_article_evidence_pack_rejects_unknown_evidence_type(self) -> None:
        pack = ArticleEvidencePack(
            items=[make_article_evidence(evidence_type="unsupported")],
            usable_count=1,
            rejected_count=0,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_article_evidence_pack(pack)

    def test_validate_article_evidence_pack_rejects_unknown_specificity_level(self) -> None:
        pack = ArticleEvidencePack(
            items=[make_article_evidence(specificity_level="extreme")],
            usable_count=1,
            rejected_count=0,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_article_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_accepts_valid_pack(self) -> None:
        pack = ContextualEvidencePack(
            items=[make_contextual_evidence()],
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=[],
            risks=["generic summary"],
        )

        validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_unknown_candidate_id(self) -> None:
        pack = ContextualEvidencePack(
            items=[make_contextual_evidence()],
            main_candidate_evidence_ids=["missing"],
            background_evidence_ids=[],
            risks=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_invalid_best_use(self) -> None:
        pack = ContextualEvidencePack(
            items=[make_contextual_evidence(best_use_in_post="headline")],
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=[],
            risks=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_angle_decision_accepts_valid_decision(self) -> None:
        decision = AngleDecision(
            controlling_angle="A polished profile is weak proof without visible decisions.",
            reader_problem="The reader shows finished work but not the thinking behind it.",
            author_position="Proof of judgment matters more than surface polish.",
            main_tension="Finished outcomes can hide how the person actually works.",
            supporting_evidence_ids=["e1"],
            angle_to_avoid=["generic personal branding advice"],
        )

        validate_angle_decision(decision)

    def test_validate_post_brief_accepts_valid_brief(self) -> None:
        brief = PostBrief(
            opening_direction="Start with polished outcomes versus visible judgment.",
            pattern_interrupt="The profile is not the real proof.",
            core_point="Recent content should show decisions, tradeoffs, and lessons.",
            evidence_to_use=[
                BriefEvidenceUse(
                    evidence_id="e1",
                    evidence_text="Recent posts can show decisions and tradeoffs.",
                    role_in_post="Use as the concrete proof point.",
                )
            ],
            practical_point="Review the last ten posts for decisions or lessons.",
            ending_direction="End by reframing proof as visible judgment.",
            cta_direction="Ask what proof makes expertise feel real.",
        )

        validate_post_brief(brief)

    def test_validate_post_brief_rejects_empty_evidence_to_use(self) -> None:
        brief = PostBrief(
            opening_direction="Start with polished outcomes versus visible judgment.",
            pattern_interrupt="The profile is not the real proof.",
            core_point="Recent content should show decisions, tradeoffs, and lessons.",
            evidence_to_use=[],
            practical_point="Review the last ten posts for decisions or lessons.",
            ending_direction="End by reframing proof as visible judgment.",
            cta_direction="Ask what proof makes expertise feel real.",
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief(brief)

    def test_validate_final_post_payload_accepts_valid_payload(self) -> None:
        validate_final_post_payload(make_final_post_payload())

    def test_validate_final_post_payload_rejects_overlength_post(self) -> None:
        payload = make_final_post_payload(post_text="x" * 1301)

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload(payload)

    def test_validate_final_post_payload_rejects_too_few_hooks(self) -> None:
        payload = make_final_post_payload(
            hook_variants=[
                "A polished profile is weak proof.",
                "Finished work can hide the important part.",
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload(payload)

    def test_validate_final_post_payload_rejects_missing_quality_check(self) -> None:
        payload = make_final_post_payload(
            quality_checks={
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
            }
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload(payload)

    def test_final_post_payload_to_dict_returns_validator_compatible_shape(self) -> None:
        payload = make_final_post_payload()

        result = final_post_payload_to_dict(payload)

        self.assertEqual(result["post_text"], payload.post_text)
        self.assertEqual(result["hook_variants"], payload.hook_variants)
        self.assertEqual(result["cta_variants"], payload.cta_variants)
        self.assertEqual(result["hashtags"], payload.hashtags)
        self.assertEqual(result["quality_checks"], payload.quality_checks)
        self.assertEqual(result["carousel_outline"], payload.carousel_outline)

    def test_final_post_payload_to_dict_passes_content_package_validator(self) -> None:
        payload = make_final_post_payload()

        result = final_post_payload_to_dict(payload)

        validate_content_package_payload(result)
