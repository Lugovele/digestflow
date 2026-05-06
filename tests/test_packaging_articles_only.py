from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.topics.models import Topic
from apps.digests.models import Digest, DigestRun
from services.packaging import generate_content_package_for_digest


@override_settings(OPENAI_API_KEY="sk-your-key")
class PackagingArticlesOnlyTests(TestCase):
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
        self.assertIn("No digest articles were available.", content_package.post_text)
