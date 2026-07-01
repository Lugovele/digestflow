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
    build_article_evidence_pack_from_pipeline_input,
    build_pipeline_input_from_digest,
    final_post_payload_to_dict,
    validate_angle_decision,
    validate_angle_decision_for_contextual_evidence,
    validate_article_evidence_pack,
    validate_article_evidence_pack_for_pipeline_input,
    validate_contextual_evidence_pack,
    validate_contextual_evidence_pack_for_article_evidence,
    validate_final_post_payload,
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
) -> ContextualEvidence:
    return ContextualEvidence(
        evidence_id=evidence_id,
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
                evidence_text="Recent posts can show decisions and tradeoffs.",
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

    def test_validate_contextual_evidence_pack_rejects_invalid_best_use(self) -> None:
        pack = ContextualEvidencePack(
            items=[make_contextual_evidence(best_use_in_post="headline")],
            main_candidate_evidence_ids=["e1"],
            background_evidence_ids=[],
            risks=[],
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
