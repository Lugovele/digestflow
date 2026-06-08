from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.topics.models import Topic
from apps.digests.models import Digest, DigestRun
from services.packaging import generate_content_package_for_digest
from services.packaging.generator import PackagingGenerationResult, normalize_linkedin_hashtags


@override_settings(OPENAI_API_KEY="sk-your-key")
class PackagingArticlesOnlyTests(TestCase):
    def test_normalize_linkedin_hashtags_prefixes_trailing_keyword_line(self) -> None:
        post_text = (
            "Share your personal branding strategies in the comments!\n\n"
            "PersonalBranding Authority Storytelling BrandLag VisualIdentity"
        )

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your personal branding strategies in the comments!\n\n"
            "#PersonalBranding #Authority #Storytelling #BrandLag #VisualIdentity",
        )

    def test_normalize_linkedin_hashtags_prefixes_comma_separated_keywords(self) -> None:
        post_text = "Share your thoughts.\n\nPersonalBranding, Authority, Storytelling"

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )

    def test_normalize_linkedin_hashtags_preserves_existing_tags_and_deduplicates(self) -> None:
        post_text = "Share your thoughts.\n\n#PersonalBranding Authority Storytelling #Authority"

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(
            normalized,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )

    def test_normalize_linkedin_hashtags_leaves_normal_prose_unchanged(self) -> None:
        post_text = "This is a normal final sentence about personal branding."

        normalized = normalize_linkedin_hashtags(post_text)

        self.assertEqual(normalized, post_text)

    def test_packaging_uses_digest_get_articles_not_legacy_digest_fields(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-user")
        topic = Topic.objects.create(
            user=user,
            name="Workflow systems",
            keywords=["workflow"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Workflow systems",
            payload={
                "title": "Digest for Workflow systems",
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "Workflow redesign article",
                        "summary": "The article argues that workflow redesign matters before adding AI.",
                        "key_points": [
                            "Review time dropped after teams changed the handoff.",
                            "The model helped only after validation became clearer.",
                        ],
                        "content_type": "opinion",
                        "confidence": 0.9,
                    }
                ],
            },
            quality_score=0.0,
        )

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertTrue(digest.has_articles())
        self.assertIn("workflow redesign matters", content_package.post_text.lower())
        self.assertEqual(debug_info["provider"], "mock")

    def test_packaging_uses_digest_helper_as_article_source(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-helper-user")
        topic = Topic.objects.create(
            user=user,
            name="Helper topic",
            keywords=["helper"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Helper topic",
            payload={"version": 1, "title": "Digest for Helper topic", "articles": []},
            quality_score=0.0,
        )

        helper_articles = [
            {
                "url": "https://example.com/helper-article",
                "title": "Helper article title",
                "summary": "The article argues that workflow fixes come before AI.",
                "key_points": ["The team changed the workflow before the model step."],
                "content_type": "opinion",
                "confidence": 0.8,
            }
        ]

        with patch.object(Digest, "get_articles", return_value=helper_articles):
            content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertIn("workflow fixes come before ai", content_package.post_text.lower())

    def test_packaging_returns_safe_fallback_when_articles_are_missing(self) -> None:
        user = get_user_model().objects.create_user(username="packaging-fallback-user")
        topic = Topic.objects.create(
            user=user,
            name="Sparse topic",
            keywords=["sparse"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_GENERATING_DIGEST,
            metrics={"digest_stage": {"status": "completed", "articles_count": 0}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Sparse topic",
            payload={"title": "Digest for Sparse topic", "articles": []},
            quality_score=0.0,
        )

        content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertFalse(digest.has_articles())
        self.assertIn("No post draft articles were available.", content_package.post_text)

    @patch("services.packaging.generator._generate_packaging_payload")
    def test_packaging_saves_normalized_linkedin_hashtags_in_post_text_and_hashtag_list(
        self,
        mock_generate_packaging_payload,
    ) -> None:
        user = get_user_model().objects.create_user(username="packaging-normalizer-user")
        topic = Topic.objects.create(
            user=user,
            name="Hashtag normalization",
            keywords=["linkedin"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PACKAGING,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Hashtag normalization",
            payload={"title": "Digest for Hashtag normalization", "articles": [{"title": "One", "summary": "Two"}]},
            quality_score=0.0,
        )
        mock_generate_packaging_payload.return_value = PackagingGenerationResult(
            prompt="prompt",
            response_text="{}",
            payload={
                "post_text": "Share your thoughts.\n\n#PersonalBranding Authority Storytelling Authority",
                "hook_variants": ["Opening one", "Opening two", "Opening three"],
                "cta_variants": ["Closing one", "Closing two", "Closing three"],
                "hashtags": ["PersonalBranding", "Authority", "Storytelling", "Authority"],
                "carousel_outline": [],
                "quality_checks": {
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                    "linkedin_ready": True,
                },
            },
            provider="mock",
            is_mock=True,
            fallback_reason="",
            tokens=None,
            estimated_cost_usd=None,
        )

        content_package, _debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(
            content_package.post_text,
            "Share your thoughts.\n\n#PersonalBranding #Authority #Storytelling",
        )
        self.assertEqual(
            content_package.hashtags,
            ["#PersonalBranding", "#Authority", "#Storytelling"],
        )

    @patch("services.packaging.generator._generate_packaging_payload")
    def test_packaging_strips_extra_model_keys_before_saving(
        self,
        mock_generate_packaging_payload,
    ) -> None:
        user = get_user_model().objects.create_user(username="packaging-extra-keys-user")
        topic = Topic.objects.create(
            user=user,
            name="Extra key normalization",
            keywords=["linkedin"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PACKAGING,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for Extra key normalization",
            payload={"title": "Digest for Extra key normalization", "articles": [{"title": "One", "summary": "Two"}]},
            quality_score=0.0,
        )
        mock_generate_packaging_payload.return_value = PackagingGenerationResult(
            prompt="prompt",
            response_text="{}",
            payload={
                "post_text": "A useful post body.",
                "hook_variants": ["Opening one", "Opening two", "Opening three"],
                "cta_variants": ["Closing one", "Closing two", "Closing three"],
                "hashtags": ["#Workflow"],
                "quality_checks": {
                    "uses_only_provided_facts": True,
                    "has_clear_point_of_view": True,
                    "linkedin_ready": True,
                },
                "carousel_outline": [{"slide": 1, "title": "Extra", "bullets": ["Drop me"]}],
                "title": "Extra title",
                "metadata": {"source": "model"},
                "slides": [{"title": "Slide"}],
            },
            provider="openai",
            is_mock=False,
            fallback_reason="",
            tokens=None,
            estimated_cost_usd=None,
        )

        content_package, debug_info = generate_content_package_for_digest(digest)

        self.assertEqual(content_package.post_text, "A useful post body.")
        self.assertEqual(content_package.hook_variants, ["Opening one", "Opening two", "Opening three"])
        self.assertEqual(content_package.cta_variants, ["Closing one", "Closing two", "Closing three"])
        self.assertEqual(content_package.hashtags, ["#Workflow"])
        self.assertEqual(content_package.carousel_outline, [])
        self.assertEqual(debug_info["validation_report"]["carousel_outline_count"], 0)
