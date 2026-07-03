from django.test import SimpleTestCase

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic
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
    build_angle_decision_from_contextual_evidence_pack,
    build_article_evidence_pack_from_pipeline_input,
    build_contextual_evidence_pack_from_article_evidence_pack,
    build_final_post_payload_from_post_brief,
    build_pipeline_input_from_digest,
    build_post_brief_from_angle_decision,
    final_post_payload_to_dict,
    validate_angle_decision,
    validate_angle_decision_for_contextual_evidence,
    validate_article_evidence_pack,
    validate_article_evidence_pack_for_pipeline_input,
    validate_contextual_evidence_pack,
    validate_contextual_evidence_pack_for_article_evidence,
    validate_final_post_payload,
    validate_final_post_payload_for_post_brief,
    validate_linkedin_post_stage_relationships,
    validate_pipeline_input,
    validate_post_brief,
    validate_post_brief_for_angle_decision,
    validate_selected_articles,
)
from services.packaging.validators import validate_content_package_payload


class FakeTopic:
    name = "Personal Branding"


class FakeRun:
    topic = FakeTopic()


class FakeDigest:
    def __init__(self, articles: list[dict]) -> None:
        self.id = 130
        self.title = "Digest for Personal Branding"
        self.run = FakeRun()
        self.get_articles_calls = 0
        self.payload = {"articles": [{"title": "Raw payload should not be used"}]}
        self._articles = articles

    def get_articles(self) -> list[dict]:
        self.get_articles_calls += 1
        return self._articles


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


def make_pipeline_input_with_two_articles() -> PipelineInput:
    return PipelineInput(
        digest_id=130,
        topic_name="Personal Branding",
        digest_title="Digest for Personal Branding",
        articles=[
            SelectedArticle(
                source_index=0,
                title="First article",
                url="https://example.com/article-0",
                summary="First summary.",
                key_points=["First key point.", "Second key point."],
            ),
            SelectedArticle(
                source_index=1,
                title="Second article",
                url="https://example.com/article-1",
                summary="Second summary.",
                key_points=["Third key point."],
            ),
        ],
        author_profile={},
    )


def make_article_evidence(
    evidence_id: str = "e1",
    source_index: int = 0,
    evidence_type: str = "contrast",
    specificity_level: str = "medium",
) -> ArticleEvidence:
    return ArticleEvidence(
        evidence_id=evidence_id,
        source_index=source_index,
        source_title="Article 0",
        evidence_text="Recent posts can show decisions and tradeoffs, not just finished outcomes.",
        evidence_type=evidence_type,
        specificity_level=specificity_level,
        source_limitations="No named case or metric.",
    )


def make_contextual_evidence(
    evidence_id: str = "e1",
    best_use_in_post: str = "tension",
    source_index: int = 0,
    source_title: str = "Article 0",
    evidence_type: str = "contrast",
    specificity_level: str = "medium",
    source_limitations: str = "No named case or metric.",
    evidence_text: str = "Recent posts can show decisions and tradeoffs, not just finished outcomes.",
    do_not_use_for: str = "Do not turn this into generic personal branding advice.",
    risk_of_misuse: str = "Could drift into surface-level branding language.",
) -> ContextualEvidence:
    return ContextualEvidence(
        evidence_id=evidence_id,
        source_index=source_index,
        source_title=source_title,
        evidence_text=evidence_text,
        evidence_type=evidence_type,
        specificity_level=specificity_level,
        source_limitations=source_limitations,
        what_it_says="Visible process helps people evaluate judgment.",
        supports_argument="The post can argue that public proof needs more than polished output.",
        best_use_in_post=best_use_in_post,
        do_not_use_for=do_not_use_for,
        risk_of_misuse=risk_of_misuse,
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


def make_article_evidence_pack(
    items: list[ArticleEvidence] | None = None,
) -> ArticleEvidencePack:
    evidence_items = items if items is not None else [make_article_evidence()]
    return ArticleEvidencePack(
        items=evidence_items,
        usable_count=len(evidence_items),
        rejected_count=0,
    )


def make_contextual_evidence_pack(
    items: list[ContextualEvidence] | None = None,
    main_candidate_evidence_ids: list[str] | None = None,
    background_evidence_ids: list[str] | None = None,
) -> ContextualEvidencePack:
    contextual_items = items if items is not None else [make_contextual_evidence()]
    return ContextualEvidencePack(
        items=contextual_items,
        main_candidate_evidence_ids=main_candidate_evidence_ids
        if main_candidate_evidence_ids is not None
        else ["e1"],
        background_evidence_ids=background_evidence_ids
        if background_evidence_ids is not None
        else [],
        risks=["generic summary"],
    )


def make_angle_decision(supporting_evidence_ids: list[str] | None = None) -> AngleDecision:
    return AngleDecision(
        controlling_angle="A polished profile is weak proof without visible decisions.",
        reader_problem="The reader shows finished work but not the thinking behind it.",
        author_position="Proof of judgment matters more than surface polish.",
        main_tension="Finished outcomes can hide how the person actually works.",
        supporting_evidence_ids=supporting_evidence_ids
        if supporting_evidence_ids is not None
        else ["e1"],
        angle_to_avoid=["generic personal branding advice"],
    )


def make_post_brief(evidence_ids: list[str] | None = None) -> PostBrief:
    ids = evidence_ids if evidence_ids is not None else ["e1"]
    return PostBrief(
        opening_direction="Start with polished outcomes versus visible judgment.",
        pattern_interrupt="The profile is not the real proof.",
        core_point="Recent content should show decisions, tradeoffs, and lessons.",
        evidence_to_use=[
            BriefEvidenceUse(
                evidence_id=evidence_id,
                evidence_text=(
                    "Recent posts can show decisions and tradeoffs, "
                    "not just finished outcomes."
                ),
                role_in_post="Use as the concrete proof point.",
            )
            for evidence_id in ids
        ],
        practical_point="Review the last ten posts for decisions or lessons.",
        ending_direction="End by reframing proof as visible judgment.",
        cta_direction="Ask what proof makes expertise feel real.",
    )


class LinkedInPostPipelineContractTests(SimpleTestCase):
    def test_build_pipeline_input_from_digest_uses_digest_articles_helper(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article from helper",
                    "summary": "Helper summary.",
                    "key_points": ["Helper point."],
                    "content_type": "article",
                    "confidence": 0.9,
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(digest)

        self.assertEqual(digest.get_articles_calls, 1)
        self.assertEqual(pipeline_input.articles[0].title, "Article from helper")

    def test_build_pipeline_input_from_digest_works_with_real_digest_get_articles(self) -> None:
        topic = Topic(name="Personal Branding")
        run = DigestRun(topic=topic)
        digest = Digest(
            id=130,
            run=run,
            title="Digest for Personal Branding",
            payload={
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "Article from real Digest",
                        "summary": "Real Digest summary.",
                        "key_points": ["Real Digest point."],
                        "content_type": "analysis",
                        "confidence": 0.73,
                    }
                ]
            },
        )

        pipeline_input = build_pipeline_input_from_digest(digest)

        self.assertEqual(pipeline_input.digest_id, 130)
        self.assertEqual(pipeline_input.topic_name, "Personal Branding")
        self.assertEqual(pipeline_input.articles[0].title, "Article from real Digest")
        self.assertEqual(pipeline_input.articles[0].source_index, 0)

    def test_build_pipeline_input_from_digest_assigns_zero_based_source_indexes(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "First article",
                    "summary": "First summary.",
                    "key_points": ["First point."],
                },
                {
                    "url": "https://example.com/article-2",
                    "title": "Second article",
                    "summary": "Second summary.",
                    "key_points": ["Second point."],
                },
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(digest)

        self.assertEqual([article.source_index for article in pipeline_input.articles], [0, 1])

    def test_build_pipeline_input_from_digest_preserves_available_article_metadata(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Metadata article",
                    "summary": "Metadata summary.",
                    "key_points": ["Metadata point."],
                    "source_name": "Example Source",
                    "published_at": "2026-06-01",
                    "content_type": "analysis",
                    "confidence": 0.74,
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(
            digest,
            author_profile={"role": "AI Automation Specialist"},
        )
        article = pipeline_input.articles[0]

        self.assertEqual(pipeline_input.digest_id, 130)
        self.assertEqual(pipeline_input.digest_title, "Digest for Personal Branding")
        self.assertEqual(pipeline_input.topic_name, "Personal Branding")
        self.assertEqual(article.url, "https://example.com/article-1")
        self.assertEqual(article.title, "Metadata article")
        self.assertEqual(article.summary, "Metadata summary.")
        self.assertEqual(article.key_points, ["Metadata point."])
        self.assertEqual(article.content_type, "analysis")
        self.assertEqual(article.confidence, 0.74)
        self.assertEqual(article.source_name, "Example Source")
        self.assertEqual(article.published_at, "2026-06-01")

    def test_build_pipeline_input_from_digest_defaults_none_optional_metadata_to_empty_strings(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Metadata article",
                    "summary": "Metadata summary.",
                    "key_points": ["Metadata point."],
                    "source_name": None,
                    "published_at": None,
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(digest)
        article = pipeline_input.articles[0]

        self.assertEqual(article.source_name, "")
        self.assertEqual(article.published_at, "")

    def test_build_pipeline_input_from_digest_accepts_author_profile_none(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(digest, author_profile=None)

        self.assertEqual(pipeline_input.author_profile, {})

    def test_build_pipeline_input_from_digest_accepts_partial_author_profile(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(
            digest,
            author_profile={"role": "AI Automation Specialist"},
        )

        self.assertEqual(pipeline_input.author_profile, {"role": "AI Automation Specialist"})

    def test_build_pipeline_input_from_digest_rejects_non_dict_author_profile(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest, author_profile=[])  # type: ignore[arg-type]

    def test_build_pipeline_input_from_digest_fails_validation_for_invalid_article(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": " ",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_missing_url(self) -> None:
        digest = FakeDigest(
            [
                {
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_blank_url(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": " ",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_non_string_url(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": 123,
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_non_string_title(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": 123,
                    "summary": "Summary.",
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_non_string_summary(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": 123,
                    "key_points": ["Point."],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_allows_missing_key_points_as_empty_list(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                }
            ]
        )

        pipeline_input = build_pipeline_input_from_digest(digest)

        self.assertEqual(pipeline_input.articles[0].key_points, [])

    def test_build_pipeline_input_from_digest_rejects_non_list_key_points(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": "Point.",
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_non_string_key_point_items(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": [123],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_pipeline_input_from_digest_rejects_empty_key_point_items(self) -> None:
        digest = FakeDigest(
            [
                {
                    "url": "https://example.com/article-1",
                    "title": "Article",
                    "summary": "Summary.",
                    "key_points": [" "],
                }
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_pipeline_input_from_digest(digest)

    def test_build_article_evidence_pack_from_pipeline_input_creates_pack(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(make_pipeline_input())

        self.assertIsInstance(pack, ArticleEvidencePack)
        self.assertEqual(pack.usable_count, len(pack.items))
        self.assertEqual(pack.rejected_count, 0)

    def test_build_article_evidence_pack_from_pipeline_input_uses_stable_evidence_ids(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(
            make_pipeline_input_with_two_articles()
        )

        self.assertEqual(
            [item.evidence_id for item in pack.items],
            [
                "a0-summary",
                "a0-kp0",
                "a0-kp1",
                "a1-summary",
                "a1-kp0",
            ],
        )

    def test_build_article_evidence_pack_from_pipeline_input_sets_source_indexes(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(
            make_pipeline_input_with_two_articles()
        )

        self.assertEqual(
            [(item.evidence_id, item.source_index) for item in pack.items],
            [
                ("a0-summary", 0),
                ("a0-kp0", 0),
                ("a0-kp1", 0),
                ("a1-summary", 1),
                ("a1-kp0", 1),
            ],
        )

    def test_build_article_evidence_pack_from_pipeline_input_preserves_source_title(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(
            make_pipeline_input_with_two_articles()
        )

        self.assertEqual(
            [(item.evidence_id, item.source_title) for item in pack.items],
            [
                ("a0-summary", "First article"),
                ("a0-kp0", "First article"),
                ("a0-kp1", "First article"),
                ("a1-summary", "Second article"),
                ("a1-kp0", "Second article"),
            ],
        )

    def test_build_article_evidence_pack_from_pipeline_input_uses_only_summary_and_key_points(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(
            make_pipeline_input_with_two_articles()
        )

        self.assertEqual(
            [item.evidence_text for item in pack.items],
            [
                "First summary.",
                "First key point.",
                "Second key point.",
                "Second summary.",
                "Third key point.",
            ],
        )

    def test_build_article_evidence_pack_from_pipeline_input_sets_evidence_types(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(make_pipeline_input())

        self.assertEqual(pack.items[0].evidence_type, "pattern")
        self.assertEqual(pack.items[0].specificity_level, "medium")
        self.assertEqual(pack.items[1].evidence_type, "practical_point")
        self.assertEqual(pack.items[1].specificity_level, "medium")

    def test_build_article_evidence_pack_from_pipeline_input_discloses_source_limitations(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(make_pipeline_input())

        self.assertTrue(
            all("digest summaries and key points" in item.source_limitations for item in pack.items)
        )
        self.assertTrue(
            all("not full article extraction" in item.source_limitations for item in pack.items)
        )

    def test_build_article_evidence_pack_from_pipeline_input_passes_relationship_validation(self) -> None:
        pipeline_input = make_pipeline_input_with_two_articles()

        pack = build_article_evidence_pack_from_pipeline_input(pipeline_input)

        validate_article_evidence_pack_for_pipeline_input(pipeline_input, pack)

    def test_build_article_evidence_pack_from_pipeline_input_rejects_invalid_pipeline_input(self) -> None:
        pipeline_input = PipelineInput(
            digest_id=130,
            topic_name="Personal Branding",
            digest_title="Digest for Personal Branding",
            articles=[make_selected_article(summary=" ")],
            author_profile={},
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_article_evidence_pack_from_pipeline_input(pipeline_input)

    def test_build_article_evidence_pack_from_pipeline_input_does_not_include_later_stage_fields(self) -> None:
        pack = build_article_evidence_pack_from_pipeline_input(make_pipeline_input())

        self.assertFalse(hasattr(pack, "post_text"))
        self.assertFalse(hasattr(pack, "angle_decision"))
        self.assertFalse(hasattr(pack, "author_position"))
        self.assertFalse(hasattr(pack, "post_brief"))
        self.assertFalse(hasattr(pack, "quality_review"))
        self.assertFalse(hasattr(pack, "repair_instruction"))

    def test_build_contextual_evidence_pack_from_article_evidence_pack_creates_one_item_per_evidence_item(self) -> None:
        article_pack = make_article_evidence_pack(
            items=[
                make_article_evidence(evidence_id="e1"),
                make_article_evidence(evidence_id="e2", evidence_type="practical_point"),
            ]
        )

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )

        self.assertEqual([item.evidence_id for item in contextual_pack.items], ["e1", "e2"])

    def test_build_contextual_evidence_pack_from_article_evidence_pack_preserves_evidence_text(self) -> None:
        article_pack = make_article_evidence_pack()

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )

        self.assertEqual(
            contextual_pack.items[0].evidence_text,
            article_pack.items[0].evidence_text,
        )

    def test_build_contextual_evidence_pack_from_article_evidence_pack_preserves_source_metadata(self) -> None:
        article_pack = make_article_evidence_pack(
            items=[
                ArticleEvidence(
                    evidence_id="e1",
                    source_index=2,
                    source_title="Source title",
                    evidence_text="A concrete source point.",
                    evidence_type="fact",
                    specificity_level="high",
                    source_limitations="Derived from digest summaries.",
                )
            ]
        )

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )
        item = contextual_pack.items[0]

        self.assertEqual(item.source_index, 2)
        self.assertEqual(item.source_title, "Source title")
        self.assertEqual(item.evidence_type, "fact")
        self.assertEqual(item.specificity_level, "high")
        self.assertEqual(item.source_limitations, "Derived from digest summaries.")

    def test_build_contextual_evidence_pack_from_article_evidence_pack_maps_roles(self) -> None:
        article_pack = make_article_evidence_pack(
            items=[
                make_article_evidence(evidence_id="contrast", evidence_type="contrast"),
                make_article_evidence(evidence_id="warning", evidence_type="warning"),
                make_article_evidence(evidence_id="example", evidence_type="example"),
                make_article_evidence(evidence_id="fact", evidence_type="fact"),
                make_article_evidence(
                    evidence_id="practical", evidence_type="practical_point"
                ),
                make_article_evidence(evidence_id="pattern", evidence_type="pattern"),
            ]
        )

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )

        self.assertEqual(
            {item.evidence_id: item.best_use_in_post for item in contextual_pack.items},
            {
                "contrast": "tension",
                "warning": "tension",
                "example": "proof",
                "fact": "proof",
                "practical": "practical_point",
                "pattern": "proof",
            },
        )

    def test_build_contextual_evidence_pack_from_article_evidence_pack_maps_low_specificity_to_background(self) -> None:
        article_pack = make_article_evidence_pack(
            items=[
                make_article_evidence(
                    evidence_id="low",
                    evidence_type="fact",
                    specificity_level="low",
                )
            ]
        )

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )

        self.assertEqual(contextual_pack.items[0].best_use_in_post, "background_only")
        self.assertEqual(contextual_pack.background_evidence_ids, ["low"])
        self.assertNotIn("low", contextual_pack.main_candidate_evidence_ids)

    def test_build_contextual_evidence_pack_from_article_evidence_pack_passes_relationship_validation(self) -> None:
        article_pack = make_article_evidence_pack()

        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )

        validate_contextual_evidence_pack_for_article_evidence(
            article_pack,
            contextual_pack,
        )

    def test_build_contextual_evidence_pack_from_article_evidence_pack_rejects_invalid_article_pack(self) -> None:
        article_pack = ArticleEvidencePack(
            items=[make_article_evidence()],
            usable_count=99,
            rejected_count=0,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_contextual_evidence_pack_from_article_evidence_pack(article_pack)

    def test_build_contextual_evidence_pack_from_article_evidence_pack_does_not_include_later_stage_fields(self) -> None:
        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            make_article_evidence_pack()
        )

        self.assertFalse(hasattr(contextual_pack, "angle_decision"))
        self.assertFalse(hasattr(contextual_pack, "post_brief"))
        self.assertFalse(hasattr(contextual_pack, "post_text"))
        self.assertFalse(hasattr(contextual_pack, "quality_review"))
        self.assertFalse(hasattr(contextual_pack, "repair_instruction"))

    def test_build_angle_decision_from_contextual_evidence_pack_returns_valid_decision(self) -> None:
        contextual_pack = make_contextual_evidence_pack()

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        validate_angle_decision_for_contextual_evidence(
            contextual_pack,
            angle_decision,
        )

    def test_build_angle_decision_from_contextual_evidence_pack_uses_only_main_candidates(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="main", best_use_in_post="proof"),
                make_contextual_evidence(
                    evidence_id="background",
                    best_use_in_post="background_only",
                ),
            ],
            main_candidate_evidence_ids=["main"],
            background_evidence_ids=["background"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertEqual(angle_decision.supporting_evidence_ids, ["main"])

    def test_build_angle_decision_from_contextual_evidence_pack_never_selects_background_evidence(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="tension", best_use_in_post="tension"),
                make_contextual_evidence(
                    evidence_id="background",
                    best_use_in_post="background_only",
                ),
            ],
            main_candidate_evidence_ids=["tension"],
            background_evidence_ids=["background"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertNotIn("background", angle_decision.supporting_evidence_ids)

    def test_build_angle_decision_from_contextual_evidence_pack_fails_without_main_candidates(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="background",
                    best_use_in_post="background_only",
                ),
            ],
            main_candidate_evidence_ids=[],
            background_evidence_ids=["background"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_angle_decision_from_contextual_evidence_pack(contextual_pack)

    def test_build_angle_decision_from_contextual_evidence_pack_prefers_source_diversity(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="tension",
                    best_use_in_post="tension",
                    source_index=0,
                ),
                make_contextual_evidence(
                    evidence_id="proof-same-source",
                    best_use_in_post="proof",
                    source_index=0,
                ),
                make_contextual_evidence(
                    evidence_id="proof-new-source",
                    best_use_in_post="proof",
                    source_index=1,
                    source_title="Article 1",
                ),
            ],
            main_candidate_evidence_ids=[
                "tension",
                "proof-same-source",
                "proof-new-source",
            ],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertEqual(
            angle_decision.supporting_evidence_ids[:2],
            ["tension", "proof-new-source"],
        )

    def test_build_angle_decision_from_contextual_evidence_pack_prefers_role_diversity(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="tension", best_use_in_post="tension"),
                make_contextual_evidence(
                    evidence_id="proof",
                    best_use_in_post="proof",
                    source_index=1,
                    source_title="Article 1",
                ),
                make_contextual_evidence(
                    evidence_id="practical",
                    best_use_in_post="practical_point",
                    source_index=2,
                    source_title="Article 2",
                ),
            ],
            main_candidate_evidence_ids=["tension", "proof", "practical"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertEqual(
            angle_decision.supporting_evidence_ids,
            ["tension", "proof", "practical"],
        )

    def test_build_angle_decision_from_contextual_evidence_pack_caps_supporting_evidence_at_three(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="tension", best_use_in_post="tension"),
                make_contextual_evidence(
                    evidence_id="proof",
                    best_use_in_post="proof",
                    source_index=1,
                    source_title="Article 1",
                ),
                make_contextual_evidence(
                    evidence_id="practical",
                    best_use_in_post="practical_point",
                    source_index=2,
                    source_title="Article 2",
                ),
                make_contextual_evidence(
                    evidence_id="hook",
                    best_use_in_post="hook",
                    source_index=3,
                    source_title="Article 3",
                ),
                make_contextual_evidence(
                    evidence_id="ending",
                    best_use_in_post="ending",
                    source_index=4,
                    source_title="Article 4",
                ),
            ],
            main_candidate_evidence_ids=[
                "tension",
                "proof",
                "practical",
                "hook",
                "ending",
            ],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertLessEqual(len(angle_decision.supporting_evidence_ids), 3)

    def test_build_angle_decision_from_contextual_evidence_pack_supports_non_priority_main_roles(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="hook", best_use_in_post="hook"),
                make_contextual_evidence(
                    evidence_id="ending",
                    best_use_in_post="ending",
                    source_index=1,
                    source_title="Article 1",
                ),
            ],
            main_candidate_evidence_ids=["hook", "ending"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertEqual(angle_decision.supporting_evidence_ids, ["hook", "ending"])

    def test_build_angle_decision_from_contextual_evidence_pack_falls_back_when_source_diversity_is_impossible(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="tension", best_use_in_post="tension"),
                make_contextual_evidence(evidence_id="proof", best_use_in_post="proof"),
                make_contextual_evidence(
                    evidence_id="practical",
                    best_use_in_post="practical_point",
                ),
            ],
            main_candidate_evidence_ids=["tension", "proof", "practical"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertEqual(
            angle_decision.supporting_evidence_ids,
            ["tension", "proof", "practical"],
        )

    def test_build_angle_decision_from_contextual_evidence_pack_flows_risks_into_angle_to_avoid(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="e1",
                    do_not_use_for="Do not turn this into a price prediction.",
                    risk_of_misuse="Could imply investment advice.",
                )
            ],
            main_candidate_evidence_ids=["e1"],
        )

        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        self.assertIn("generic summary", angle_decision.angle_to_avoid)
        self.assertIn(
            "Do not turn this into a price prediction.",
            angle_decision.angle_to_avoid,
        )
        self.assertIn("Could imply investment advice.", angle_decision.angle_to_avoid)

    def test_build_angle_decision_from_contextual_evidence_pack_does_not_include_later_stage_fields(self) -> None:
        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            make_contextual_evidence_pack()
        )

        self.assertFalse(hasattr(angle_decision, "post_brief"))
        self.assertFalse(hasattr(angle_decision, "post_text"))
        self.assertFalse(hasattr(angle_decision, "quality_review"))
        self.assertFalse(hasattr(angle_decision, "repair_instruction"))
        self.assertFalse(hasattr(angle_decision, "core_opinion"))

    def test_build_post_brief_from_angle_decision_returns_valid_brief(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        validate_post_brief_for_angle_decision(
            contextual_pack,
            angle_decision,
            post_brief,
        )

    def test_build_post_brief_from_angle_decision_uses_only_supporting_evidence_ids(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="selected"),
                make_contextual_evidence(evidence_id="not-selected", source_index=1),
            ],
            main_candidate_evidence_ids=["selected", "not-selected"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["selected"])

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        self.assertEqual(
            [item.evidence_id for item in post_brief.evidence_to_use],
            ["selected"],
        )

    def test_build_post_brief_from_angle_decision_preserves_exact_evidence_id_and_text(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="e1",
                    evidence_text="First exact evidence text.",
                ),
                make_contextual_evidence(
                    evidence_id="e2",
                    evidence_text="Second exact evidence text.",
                    source_index=1,
                ),
            ],
            main_candidate_evidence_ids=["e1", "e2"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["e2", "e1"])

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        self.assertEqual(
            [(item.evidence_id, item.evidence_text) for item in post_brief.evidence_to_use],
            [
                ("e2", "Second exact evidence text."),
                ("e1", "First exact evidence text."),
            ],
        )

    def test_build_post_brief_from_angle_decision_preserves_supporting_evidence_order(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="first"),
                make_contextual_evidence(evidence_id="second", source_index=1),
                make_contextual_evidence(evidence_id="third", source_index=2),
            ],
            main_candidate_evidence_ids=["first", "second", "third"],
        )
        angle_decision = make_angle_decision(
            supporting_evidence_ids=["third", "first", "second"]
        )

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        self.assertEqual(
            [item.evidence_id for item in post_brief.evidence_to_use],
            ["third", "first", "second"],
        )

    def test_build_post_brief_from_angle_decision_assigns_non_empty_roles(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        self.assertTrue(post_brief.evidence_to_use[0].role_in_post.strip())

    def test_build_post_brief_from_angle_decision_rejects_unknown_supporting_evidence(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision(supporting_evidence_ids=["missing"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_post_brief_from_angle_decision(
                contextual_pack,
                angle_decision,
            )

    def test_build_post_brief_from_angle_decision_staged_chain_accepts_generated_brief(self) -> None:
        pipeline_input = make_pipeline_input()
        article_pack = build_article_evidence_pack_from_pipeline_input(pipeline_input)
        contextual_pack = build_contextual_evidence_pack_from_article_evidence_pack(
            article_pack
        )
        angle_decision = build_angle_decision_from_contextual_evidence_pack(
            contextual_pack
        )

        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        validate_linkedin_post_stage_relationships(
            pipeline_input,
            article_pack,
            contextual_pack,
            angle_decision,
            post_brief,
        )

    def test_build_post_brief_from_angle_decision_does_not_include_final_stage_fields(self) -> None:
        post_brief = build_post_brief_from_angle_decision(
            make_contextual_evidence_pack(),
            make_angle_decision(),
        )

        self.assertFalse(hasattr(post_brief, "post_text"))
        self.assertFalse(hasattr(post_brief, "hook_variants"))
        self.assertFalse(hasattr(post_brief, "hashtags"))
        self.assertFalse(hasattr(post_brief, "quality_checks"))
        self.assertFalse(hasattr(post_brief, "repair_instruction"))
        self.assertFalse(hasattr(post_brief, "package_payload"))
        self.assertFalse(hasattr(post_brief, "core_opinion"))

    def test_build_final_post_payload_from_post_brief_returns_valid_payload(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        validate_final_post_payload(payload)
        validate_final_post_payload_for_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
            payload,
        )

    def test_build_final_post_payload_from_post_brief_dict_passes_content_package_validator(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        validate_content_package_payload(final_post_payload_to_dict(payload))

    def test_build_final_post_payload_from_post_brief_sets_scaffold_post_text(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertTrue(payload.post_text.strip())
        self.assertLessEqual(len(payload.post_text), 1300)
        self.assertIn("[Scaffold only - not production copy]", payload.post_text)

    def test_build_final_post_payload_from_post_brief_post_text_contains_no_urls(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertNotIn("http://", payload.post_text.lower())
        self.assertNotIn("https://", payload.post_text.lower())
        self.assertNotIn("www.", payload.post_text.lower())

    def test_build_final_post_payload_from_post_brief_includes_selected_evidence_ids(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="e1",
                    evidence_text="Exact selected evidence for final scaffold.",
                )
            ],
            main_candidate_evidence_ids=["e1"],
        )
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertIn("e1", payload.post_text)
        self.assertNotIn("Exact selected evidence for final scaffold.", payload.post_text)

    def test_build_final_post_payload_from_post_brief_includes_controlling_angle(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertIn(angle_decision.controlling_angle, payload.post_text)

    def test_build_final_post_payload_from_post_brief_does_not_copy_url_bearing_evidence_text(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="e1",
                    evidence_text="Evidence text points to https://example.com.",
                )
            ],
            main_candidate_evidence_ids=["e1"],
        )
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertIn("e1", payload.post_text)
        self.assertNotIn("https://example.com", payload.post_text)

    def test_build_final_post_payload_from_post_brief_rejects_overlength_scaffold(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = AngleDecision(
            controlling_angle="x" * 1200,
            reader_problem="The reader shows finished work but not the thinking behind it.",
            author_position="Proof of judgment matters more than surface polish.",
            main_tension="Finished outcomes can hide how the person actually works.",
            supporting_evidence_ids=["e1"],
            angle_to_avoid=["generic personal branding advice"],
        )
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            build_final_post_payload_from_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
            )

    def test_build_final_post_payload_from_post_brief_sets_conservative_quality_checks(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertIs(payload.quality_checks["linkedin_ready"], False)
        self.assertIs(payload.quality_checks["uses_only_provided_facts"], True)
        self.assertIs(payload.quality_checks["has_clear_point_of_view"], True)

    def test_build_final_post_payload_from_post_brief_has_required_lists(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertGreaterEqual(len(payload.hook_variants), 3)
        self.assertGreaterEqual(len(payload.cta_variants), 3)
        self.assertGreaterEqual(len(payload.hashtags), 1)
        self.assertEqual(payload.carousel_outline, [])

    def test_final_post_payload_relationship_rejects_missing_selected_evidence_id(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = FinalPostPayload(
            post_text=(
                "[Scaffold only - not production copy]\n"
                f"Controlling angle: {angle_decision.controlling_angle}\n"
                "Scaffold text without the selected evidence."
            ),
            hook_variants=[
                "Scaffold hook one.",
                "Scaffold hook two.",
                "Scaffold hook three.",
            ],
            cta_variants=[
                "Scaffold CTA one?",
                "Scaffold CTA two?",
                "Scaffold CTA three?",
            ],
            hashtags=["#PostFlow"],
            quality_checks={
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
                "linkedin_ready": False,
            },
            carousel_outline=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload,
            )

    def test_final_post_payload_relationship_rejects_prefix_matched_evidence_id(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = FinalPostPayload(
            post_text=(
                "[Scaffold only - not production copy]\n"
                f"Controlling angle: {angle_decision.controlling_angle}\n"
                "Selected evidence references: e10 (proof)\n"
                "Scaffold text with a prefix-colliding evidence reference."
            ),
            hook_variants=[
                "Scaffold hook one.",
                "Scaffold hook two.",
                "Scaffold hook three.",
            ],
            cta_variants=[
                "Scaffold CTA one?",
                "Scaffold CTA two?",
                "Scaffold CTA three?",
            ],
            hashtags=["#PostFlow"],
            quality_checks={
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
                "linkedin_ready": False,
            },
            carousel_outline=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload,
            )

    def test_final_post_payload_relationship_rejects_missing_scaffold_marker(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )
        payload_without_marker = FinalPostPayload(
            post_text=payload.post_text.replace(
                "[Scaffold only - not production copy]",
                "Scaffold text",
            ),
            hook_variants=payload.hook_variants,
            cta_variants=payload.cta_variants,
            hashtags=payload.hashtags,
            quality_checks=payload.quality_checks,
            carousel_outline=payload.carousel_outline,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload_without_marker,
            )

    def test_final_post_payload_relationship_rejects_missing_controlling_angle(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )
        payload_without_angle = FinalPostPayload(
            post_text=payload.post_text.replace(
                angle_decision.controlling_angle,
                "Controlling angle removed.",
            ),
            hook_variants=payload.hook_variants,
            cta_variants=payload.cta_variants,
            hashtags=payload.hashtags,
            quality_checks=payload.quality_checks,
            carousel_outline=payload.carousel_outline,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload_without_angle,
            )

    def test_final_post_payload_relationship_rejects_urls_in_post_text(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )
        payload_with_url = FinalPostPayload(
            post_text=f"{payload.post_text}\nhttps://example.com",
            hook_variants=payload.hook_variants,
            cta_variants=payload.cta_variants,
            hashtags=payload.hashtags,
            quality_checks=payload.quality_checks,
            carousel_outline=payload.carousel_outline,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload_with_url,
            )

    def test_final_post_payload_relationship_rejects_linkedin_ready_true(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )
        payload_ready = FinalPostPayload(
            post_text=payload.post_text,
            hook_variants=payload.hook_variants,
            cta_variants=payload.cta_variants,
            hashtags=payload.hashtags,
            quality_checks={
                **payload.quality_checks,
                "linkedin_ready": True,
            },
            carousel_outline=payload.carousel_outline,
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload_ready,
            )

    def test_final_post_payload_relationship_rejects_carousel_outline(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )
        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )
        payload_with_carousel = FinalPostPayload(
            post_text=payload.post_text,
            hook_variants=payload.hook_variants,
            cta_variants=payload.cta_variants,
            hashtags=payload.hashtags,
            quality_checks=payload.quality_checks,
            carousel_outline=[{"slide": "Not part of this scaffold."}],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_final_post_payload_for_post_brief(
                contextual_pack,
                angle_decision,
                post_brief,
                payload_with_carousel,
            )

    def test_build_final_post_payload_from_post_brief_does_not_include_runtime_or_repair_fields(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        post_brief = build_post_brief_from_angle_decision(
            contextual_pack,
            angle_decision,
        )

        payload = build_final_post_payload_from_post_brief(
            contextual_pack,
            angle_decision,
            post_brief,
        )

        self.assertFalse(hasattr(payload, "content_package"))
        self.assertFalse(hasattr(payload, "package_payload"))
        self.assertFalse(hasattr(payload, "debug_info"))
        self.assertFalse(hasattr(payload, "quality_review"))
        self.assertFalse(hasattr(payload, "repair_instruction"))
        self.assertFalse(hasattr(payload, "repair_output"))

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

    def test_article_evidence_relationship_rejects_source_index_outside_pipeline_input(self) -> None:
        pipeline_input = make_pipeline_input()
        pack = make_article_evidence_pack(
            items=[make_article_evidence(source_index=7)]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_article_evidence_pack_for_pipeline_input(pipeline_input, pack)

    def test_article_evidence_relationship_rejects_duplicate_evidence_id(self) -> None:
        pipeline_input = make_pipeline_input()
        pack = make_article_evidence_pack(
            items=[
                make_article_evidence(evidence_id="e1"),
                make_article_evidence(evidence_id="e1"),
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_article_evidence_pack_for_pipeline_input(pipeline_input, pack)

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

    def test_validate_contextual_evidence_pack_rejects_duplicate_item_evidence_id(self) -> None:
        pack = ContextualEvidencePack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e1"),
            ],
            main_candidate_evidence_ids=["e1"],
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

    def test_validate_contextual_evidence_pack_rejects_negative_source_index(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_index=-1)]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_empty_source_title(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_title=" ")]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_invalid_evidence_type(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_type="unsupported")]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_invalid_specificity_level(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(specificity_level="extreme")]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_non_string_source_limitations(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    source_limitations=None,  # type: ignore[arg-type]
                )
            ]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_duplicate_main_candidate_ids(self) -> None:
        pack = make_contextual_evidence_pack(
            main_candidate_evidence_ids=["e1", "e1"],
            background_evidence_ids=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_duplicate_background_ids(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(best_use_in_post="background_only")],
            main_candidate_evidence_ids=[],
            background_evidence_ids=["e1", "e1"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_blank_source_limitations(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_limitations=" ")]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_background_only_in_main_candidates(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(best_use_in_post="background_only")],
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_background_only_missing_from_background_ids(self) -> None:
        pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(best_use_in_post="background_only")],
            main_candidate_evidence_ids=[],
            background_evidence_ids=[],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_non_background_item_in_background_ids(self) -> None:
        pack = make_contextual_evidence_pack(
            main_candidate_evidence_ids=[],
            background_evidence_ids=["e1"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_validate_contextual_evidence_pack_rejects_candidate_background_overlap(self) -> None:
        pack = make_contextual_evidence_pack(
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=["e1"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack(pack)

    def test_contextual_evidence_relationship_rejects_unknown_evidence_id(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_id="missing")],
            main_candidate_evidence_ids=["missing"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_duplicate_evidence_id(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e1"),
            ],
            main_candidate_evidence_ids=["e1"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_missing_contextual_evidence(self) -> None:
        article_pack = make_article_evidence_pack(
            items=[
                make_article_evidence(evidence_id="e1"),
                make_article_evidence(evidence_id="e2", evidence_type="fact"),
            ]
        )
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_id="e1")],
            main_candidate_evidence_ids=["e1"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_source_index(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_index=1)],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_source_title(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_title="Different article")],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_evidence_text(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[
                ContextualEvidence(
                    evidence_id="e1",
                    source_index=0,
                    source_title="Article 0",
                    evidence_text="Different evidence text.",
                    evidence_type="contrast",
                    specificity_level="medium",
                    source_limitations="No named case or metric.",
                    what_it_says="Visible process helps people evaluate judgment.",
                    supports_argument=(
                        "The post can argue that public proof needs more than polished output."
                    ),
                    best_use_in_post="tension",
                    do_not_use_for="Do not turn this into generic personal branding advice.",
                    risk_of_misuse="Could drift into surface-level branding language.",
                )
            ],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_evidence_type(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_type="fact")],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_specificity_level(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(specificity_level="high")],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_mismatched_source_limitations(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(source_limitations="Different limitation.")],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_unknown_candidate_evidence_id(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_id="e1")],
            main_candidate_evidence_ids=["missing"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

    def test_contextual_evidence_relationship_rejects_unknown_background_evidence_id(self) -> None:
        article_pack = make_article_evidence_pack()
        contextual_pack = make_contextual_evidence_pack(
            items=[make_contextual_evidence(evidence_id="e1")],
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=["missing"],
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_contextual_evidence_pack_for_article_evidence(
                article_pack,
                contextual_pack,
            )

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

    def test_angle_decision_relationship_rejects_unknown_supporting_evidence_id(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision(supporting_evidence_ids=["missing"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_angle_decision_for_contextual_evidence(
                contextual_pack,
                angle_decision,
            )

    def test_angle_decision_relationship_rejects_empty_supporting_evidence_ids(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision(supporting_evidence_ids=[])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_angle_decision_for_contextual_evidence(
                contextual_pack,
                angle_decision,
            )

    def test_angle_decision_relationship_rejects_duplicate_supporting_evidence_ids(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision(supporting_evidence_ids=["e1", "e1"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_angle_decision_for_contextual_evidence(
                contextual_pack,
                angle_decision,
            )

    def test_angle_decision_relationship_rejects_more_than_three_supporting_evidence_ids(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e2", source_index=1),
                make_contextual_evidence(evidence_id="e3", source_index=2),
                make_contextual_evidence(evidence_id="e4", source_index=3),
            ],
            main_candidate_evidence_ids=["e1", "e2", "e3", "e4"],
        )
        angle_decision = make_angle_decision(
            supporting_evidence_ids=["e1", "e2", "e3", "e4"]
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_angle_decision_for_contextual_evidence(
                contextual_pack,
                angle_decision,
            )

    def test_angle_decision_relationship_rejects_background_supporting_evidence_id(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="main", best_use_in_post="proof"),
                make_contextual_evidence(
                    evidence_id="background",
                    best_use_in_post="background_only",
                ),
            ],
            main_candidate_evidence_ids=["main"],
            background_evidence_ids=["background"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["background"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_angle_decision_for_contextual_evidence(
                contextual_pack,
                angle_decision,
            )

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

    def test_post_brief_relationship_rejects_evidence_not_selected_by_angle(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e2"),
            ],
            main_candidate_evidence_ids=["e1", "e2"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["e1"])
        brief = make_post_brief(evidence_ids=["e2"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief_for_angle_decision(
                contextual_pack,
                angle_decision,
                brief,
            )

    def test_post_brief_relationship_rejects_missing_selected_evidence(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e2", source_index=1),
            ],
            main_candidate_evidence_ids=["e1", "e2"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["e1", "e2"])
        brief = make_post_brief(evidence_ids=["e1"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief_for_angle_decision(
                contextual_pack,
                angle_decision,
                brief,
            )

    def test_post_brief_relationship_rejects_reordered_selected_evidence(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(evidence_id="e1"),
                make_contextual_evidence(evidence_id="e2", source_index=1),
            ],
            main_candidate_evidence_ids=["e1", "e2"],
        )
        angle_decision = make_angle_decision(supporting_evidence_ids=["e1", "e2"])
        brief = make_post_brief(evidence_ids=["e2", "e1"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief_for_angle_decision(
                contextual_pack,
                angle_decision,
                brief,
            )

    def test_post_brief_relationship_rejects_duplicate_selected_evidence(self) -> None:
        contextual_pack = make_contextual_evidence_pack()
        angle_decision = make_angle_decision()
        brief = make_post_brief(evidence_ids=["e1", "e1"])

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief_for_angle_decision(
                contextual_pack,
                angle_decision,
                brief,
            )

    def test_post_brief_relationship_rejects_mismatched_evidence_text(self) -> None:
        contextual_pack = make_contextual_evidence_pack(
            items=[
                make_contextual_evidence(
                    evidence_id="e1",
                    evidence_text="Original contextual evidence text.",
                )
            ],
            main_candidate_evidence_ids=["e1"],
        )
        angle_decision = make_angle_decision()
        brief = PostBrief(
            opening_direction="Start with polished outcomes versus visible judgment.",
            pattern_interrupt="The profile is not the real proof.",
            core_point="Recent content should show decisions, tradeoffs, and lessons.",
            evidence_to_use=[
                BriefEvidenceUse(
                    evidence_id="e1",
                    evidence_text="Changed evidence text.",
                    role_in_post="Use as the concrete proof point.",
                )
            ],
            practical_point="Review the last ten posts for decisions or lessons.",
            ending_direction="End by reframing proof as visible judgment.",
            cta_direction="Ask what proof makes expertise feel real.",
        )

        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_post_brief_for_angle_decision(
                contextual_pack,
                angle_decision,
                brief,
            )

    def test_linkedin_post_stage_relationships_accepts_valid_staged_chain(self) -> None:
        validate_linkedin_post_stage_relationships(
            make_pipeline_input(),
            make_article_evidence_pack(),
            make_contextual_evidence_pack(),
            make_angle_decision(),
            make_post_brief(),
        )

    def test_linkedin_post_stage_relationships_rejects_broken_staged_chain(self) -> None:
        with self.assertRaises(LinkedInPostPipelineContractError):
            validate_linkedin_post_stage_relationships(
                make_pipeline_input(),
                make_article_evidence_pack(),
                make_contextual_evidence_pack(),
                make_angle_decision(supporting_evidence_ids=["missing"]),
                make_post_brief(),
            )

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
