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
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.8,
                    "min_actual_quality_score": 0.6,
                    "average_quality_score": 0.7,
                    "articles_above_quality_threshold": 2,
                    "selected_for_prompt": 2,
                    "ranking_scores": [
                        {
                            "title": "First article title",
                            "url": "https://example.com/article-1",
                            "source_name": "Example Source",
                            "score": 7,
                            "quality_score": 0.8,
                            "quality_reasons": ["strong relevance to topic"],
                        },
                        {
                            "title": "Second article title",
                            "url": "https://example.com/article-2",
                            "source_name": "Example Source",
                            "score": 6,
                            "quality_score": 0.6,
                            "quality_reasons": ["good technical/practical article"],
                        },
                    ],
                },
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
        self.assertContains(response, "Digest generated successfully.")
        self.assertContains(response, "Selected articles")
        self.assertContains(response, "All ranked articles")
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
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 1,
                    "ranking_scores": [
                        {
                            "title": "Linked article title",
                            "url": "https://example.com/article-v1",
                            "source_name": "Example Source",
                            "score": 5,
                            "quality_score": 0.5,
                            "quality_reasons": ["strong relevance to topic"],
                        }
                    ],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
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
        self.assertEqual(response.context["ranked_articles"][0]["title"], "Linked article title")

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
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 1,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
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

    def test_run_detail_shows_insufficient_quality_message_without_digest_or_package(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-insufficient")
        topic = Topic.objects.create(
            user=user,
            name="Broad AI source",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            result_message="Not enough high-quality articles for a full digest.",
            error_message=(
                "Недостаточно качественных статей для полноценного дайджеста. "
                "Источник обработан, но найденные материалы слишком слабые или разрозненные."
            ),
            metrics={
                "ranking_stage": {
                    "status": "insufficient_quality",
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.3,
                    "min_actual_quality_score": 0.0,
                    "average_quality_score": 0.08,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 0,
                    "rejected_low_quality_count": 3,
                    "ranking_scores": [
                        {
                            "title": "Weak article one",
                            "url": "https://example.com/weak-1",
                            "source_name": "Example Blog",
                            "score": 3,
                            "quality_score": 0.3,
                            "quality_reasons": [
                                "too narrow for selected topic",
                                "low practical value",
                            ],
                        },
                        {
                            "title": "Weak article two",
                            "url": "https://example.com/weak-2",
                            "source_name": "Example Blog",
                            "score": 2,
                            "quality_score": 0.2,
                            "quality_reasons": [
                                "weak relevance to topic",
                                "insufficient evidence/detail",
                            ],
                        },
                    ],
                    "top_rejected_articles": [
                        {
                            "title": "Weak article one",
                            "url": "https://example.com/weak-1",
                            "quality_score": 0.3,
                        },
                        {
                            "title": "Weak article two",
                            "url": "https://example.com/weak-2",
                            "quality_score": 0.2,
                        },
                    ],
                    "insufficient_quality_message": (
                        "Недостаточно качественных статей для полноценного дайджеста. "
                        "Источник обработан, но найденные материалы слишком слабые или разрозненные."
                    ),
                },
                "digest_stage": {"status": "skipped", "reason": "insufficient_quality"},
                "packaging_stage": {"status": "skipped", "reason": "insufficient_quality"},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))
        response_text = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["is_insufficient_quality"])
        self.assertEqual(
            response.context["insufficient_quality_message"],
            "Not enough high-quality articles for a full digest.",
        )
        self.assertContains(response, "Not enough high-quality articles for a full digest.")
        self.assertEqual(
            response_text.count("Not enough high-quality articles for a full digest."),
            1,
        )
        self.assertLess(
            response_text.index("Not enough high-quality articles for a full digest."),
            response_text.index(run.error_message),
        )
        self.assertContains(response, "Ranking diagnostics")
        self.assertContains(response, "All ranked articles")
        self.assertContains(response, "Weak article one")
        self.assertContains(response, "https://example.com/weak-1")
        self.assertContains(response, "Digest не был создан из-за недостаточного качества")
        self.assertContains(
            response,
            "LinkedIn post и упаковка не были созданы, потому что статей выше порога оказалось слишком мало.",
        )
        self.assertContains(response, "articles_above_quality_threshold:</strong> 1", html=False)

    def test_run_detail_shows_zero_values_as_zero(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-zero")
        topic = Topic.objects.create(
            user=user,
            name="Zero diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.2,
                    "min_actual_quality_score": 0.0,
                    "average_quality_score": 0.08,
                    "articles_above_quality_threshold": 0,
                    "selected_for_prompt": 0,
                    "rejected_low_quality_count": 3,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "skipped", "tokens": {"total": 0}},
                "packaging_stage": {"status": "skipped", "tokens": {"total": 0}, "estimated_cost_usd": 0},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "articles_above_quality_threshold:</strong> 0", html=False)
        self.assertContains(response, "selected_for_prompt:</strong> 0", html=False)
        self.assertContains(response, "total_tokens:</strong> 0", html=False)
        self.assertContains(response, "total_estimated_cost:</strong> 0", html=False)

    def test_run_detail_hides_result_message_block_when_empty(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-no-result-message")
        topic = Topic.objects.create(
            user=user,
            name="No result message",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_FAILED,
            result_message="",
            error_message="Technical failure",
            metrics={},
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "<strong>Result:</strong>", html=False)
        self.assertContains(response, "Technical failure")
