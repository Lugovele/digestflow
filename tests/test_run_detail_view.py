from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.digests.models import Digest, DigestRun
from apps.topics.models import Topic


class RunDetailViewTests(TestCase):
    def test_run_detail_renders_only_per_article_digest_view(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user")
        topic = Topic.objects.create(
            user=user,
            name="AI workflows",
            keywords=["AI workflows"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            metrics={
                "digest_stage": {
                    "status": "completed",
                    "articles_count": 2,
                }
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI workflows",
            payload={
                "title": "Digest for AI workflows",
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "First article title",
                        "summary": "Article one summary",
                        "key_points": ["Point one", "Point two"],
                        "content_type": "news",
                        "confidence": 0.8,
                    },
                    {
                        "url": "https://example.com/article-2",
                        "title": "Second article title",
                        "summary": "Article two summary",
                        "key_points": ["Point three", "Point four"],
                        "content_type": "opinion",
                        "confidence": 0.6,
                    },
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(run.digest.has_articles())
        self.assertTrue(response.context["has_digest_articles"])
        self.assertEqual(
            response.context["digest_payload"]["title"],
            "Digest for AI workflows",
        )
        self.assertEqual(len(response.context["digest_payload"]["articles"]), 2)
        self.assertContains(response, 'href="https://example.com/article-1"', html=False)
        self.assertContains(response, "First article title")
        self.assertContains(response, "Article one summary")
        self.assertNotContains(response, "Legacy compatibility view")
        self.assertNotContains(response, "Sources:")

    def test_run_detail_reads_articles_from_digest_helpers(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-v1")
        topic = Topic.objects.create(
            user=user,
            name="AI operations",
            keywords=["AI operations"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for AI operations",
            payload={
                "title": "Digest for AI operations",
                "articles": [
                    {
                        "url": "https://example.com/article-v1",
                        "title": "Linked article title",
                        "summary": "Article summary",
                        "key_points": ["Point one"],
                        "content_type": "news",
                        "confidence": 0.5,
                    }
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(
            response.context["digest_payload"]["articles"][0]["url"],
            digest.get_articles()[0]["url"],
        )
        self.assertEqual(
            response.context["digest_payload"]["articles"][0]["summary"],
            digest.get_articles()[0]["summary"],
        )
        self.assertEqual(response.context["digest_payload"]["articles"][0]["domain"], "example.com")
        self.assertEqual(response.context["digest_payload"]["articles"][0]["title"], "Linked article title")
        self.assertEqual(response.context["digest_payload"]["articles"][0]["link_label"], "Linked article title")

    def test_run_detail_shows_fallback_when_article_url_is_missing(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-empty-url")
        topic = Topic.objects.create(
            user=user,
            name="AI delivery",
            keywords=["AI delivery"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            metrics={"digest_stage": {"status": "completed", "articles_count": 1}},
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI delivery",
            payload={
                "title": "Digest for AI delivery",
                "articles": [
                    {
                        "url": "",
                        "title": "",
                        "summary": "Article without source url",
                        "key_points": ["Point one"],
                        "content_type": "news",
                        "confidence": 0.4,
                    }
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "No source available")
