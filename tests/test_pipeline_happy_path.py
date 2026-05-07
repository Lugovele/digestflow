import json
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.digests.models import DigestRun
from apps.sources.models import Article
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

LONG_RSS_SNIPPET_1 = (
    "A content team cut prep from 6 hours to 2.5 hours, but editors still checked "
    "every claim before publishing. The workflow got faster at the start, yet the review "
    "step stayed manual, and the team still spent time cleaning up unsupported details "
    "before anything was ready to ship."
)

LONG_RSS_SNIPPET_2 = (
    "A support team got triage 28% faster with structured forms, but bad labels still "
    "broke routing. The queue moved quicker at intake, yet handoffs still failed later, "
    "and operators had to step back in to correct tickets that landed with the wrong team."
)

LONG_RSS_SNIPPET_3 = (
    "An ops team cut review time by 35% after redesigning the workflow first and adding "
    "AI later. They changed the handoff before the model step, made validation clearer, "
    "and stopped losing time in the same back-and-forth review loop."
)


@override_settings(OPENAI_API_KEY="sk-your-key")
class DigestPipelineHappyPathTests(TestCase):
    def test_run_digest_pipeline_completes_end_to_end_with_mock_ai(self):
        user = get_user_model().objects.create_user(
            username="pipeline-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="AI automation",
            keywords=["AI automation", "workflow automation"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "demo",
                "source": "integration_test",
            },
        )

        raw_items = get_demo_articles_for_topic(topic.name)

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()
        digest = run.digest
        content_package = digest.content_package

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)

        self.assertEqual(Article.objects.filter(topic=topic).count(), 3)
        self.assertEqual(digest.run_id, run.id)
        self.assertTrue(digest.title)
        self.assertEqual(digest.get_payload_version(), 1)
        self.assertEqual(digest.get_payload_title(), digest.title)
        self.assertTrue(digest.has_articles())
        self.assertEqual(len(digest.get_articles()), 3)
        self.assertEqual(len(digest.payload["articles"]), 3)
        self.assertTrue(digest.get_articles()[0]["title"])

        self.assertEqual(content_package.digest_id, digest.id)
        self.assertTrue(content_package.post_text)
        self.assertEqual(len(content_package.hook_variants), 3)
        self.assertEqual(len(content_package.cta_variants), 3)
        self.assertGreaterEqual(len(content_package.hashtags), 1)
        self.assertEqual(content_package.validation_report.get("status"), "valid")

        source_stage = run.metrics.get("source_stage", {})
        ranking_stage = run.metrics.get("ranking_stage", {})
        digest_stage = run.metrics.get("digest_stage", {})
        packaging_stage = run.metrics.get("packaging_stage", {})

        self.assertEqual(source_stage.get("articles_count"), 4)
        self.assertEqual(source_stage.get("article_links_extracted"), 4)
        self.assertEqual(source_stage.get("article_contents_fetched"), 4)
        self.assertEqual(source_stage.get("content_unavailable_count"), 0)
        self.assertEqual(source_stage.get("articles_after_dedupe"), 3)
        self.assertEqual(source_stage.get("duplicates_removed"), 1)
        self.assertEqual(source_stage.get("saved_articles_count"), 3)
        self.assertEqual(len(source_stage.get("article_ids", [])), 3)

        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertEqual(ranking_stage.get("max_quality_score"), 0.7)
        self.assertEqual(ranking_stage.get("min_actual_quality_score"), 0.4)
        self.assertEqual(ranking_stage.get("average_quality_score"), 0.53)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 3)
        self.assertEqual(ranking_stage.get("rejected_low_quality_count"), 0)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 3)
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["title"])
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["quality_reasons"])
        self.assertEqual(digest_stage.get("provider"), "mock")
        self.assertTrue(digest_stage.get("is_mock"))
        self.assertEqual(digest_stage.get("status"), "completed")
        self.assertEqual(digest_stage.get("articles_count"), 3)
        self.assertNotIn("article_analyses_count", digest_stage)
        self.assertNotIn("article_analyses", digest_stage)
        self.assertNotIn("key_points_count", digest_stage)
        self.assertNotIn("sources_count", digest_stage)
        self.assertEqual(packaging_stage.get("status"), "completed")
        self.assertEqual(packaging_stage.get("provider"), "mock")
        self.assertTrue(packaging_stage.get("is_mock"))

    def test_run_digest_pipeline_completes_with_local_rss_items_and_no_error_message(self):
        user = get_user_model().objects.create_user(
            username="rss-pipeline-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="DigestFlow AI",
            keywords=["AI", "automation"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "rss_url_override",
                "source": "integration_test",
                "rss_url": "tests/fixtures/sample_feed.xml",
            },
        )

        raw_items = fetch_rss_articles(str(Path("tests/fixtures/sample_feed.xml")))
        json.dumps(raw_items[0])

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertTrue(hasattr(run, "digest"))
        self.assertTrue(hasattr(run.digest, "content_package"))
        article = Article.objects.filter(topic=topic).order_by("id").first()
        self.assertIsNotNone(article)
        self.assertIsInstance(article.published_at, datetime)
        self.assertIsInstance(article.raw_payload.get("published_at"), str)

    def test_run_digest_pipeline_completes_with_snippet_only_rss_items(self):
        user = get_user_model().objects.create_user(
            username="snippet-rss-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Workflow automation",
            keywords=["workflow", "automation"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "rss_items",
                "source": "integration_test",
            },
        )
        raw_items = [
            {
                "title": "Briefing workflow cuts prep time",
                "url": "https://example.com/rss-1",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_1,
                "published_at": "2026-05-05T10:00:00+00:00",
            },
            {
                "title": "Support triage gets faster with forms",
                "url": "https://example.com/rss-2",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_2,
                "published_at": "2026-05-05T11:00:00+00:00",
            },
            {
                "title": "Workflow redesign shortens review cycles",
                "url": "https://example.com/rss-3",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_3,
                "published_at": "2026-05-05T12:00:00+00:00",
            },
        ]

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertTrue(hasattr(run, "digest"))
        self.assertTrue(hasattr(run.digest, "content_package"))
        source_stage = run.metrics.get("source_stage", {})
        self.assertEqual(source_stage.get("raw_items_count"), 3)
        self.assertEqual(source_stage.get("article_contents_fetched"), 3)
        self.assertEqual(source_stage.get("articles_after_cleaning"), 3)
        self.assertEqual(source_stage.get("removed_during_cleaning"), 0)

    @patch("services.sources.rss_adapter.fetch_dev_to_article_content")
    @patch("services.sources.rss_adapter.fetch_dev_to_article_list")
    def test_run_digest_pipeline_marks_insufficient_quality_when_only_one_article_survives(
        self,
        mock_fetch_dev_to_article_list,
        mock_fetch_dev_to_article_content,
    ):
        user = get_user_model().objects.create_user(
            username="devto-tag-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Dev.to AI",
            keywords=["ai"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "source_url", "source": "integration_test"},
        )
        mock_fetch_dev_to_article_list.return_value = [
            {
                "id": 1,
                "title": "Full content article",
                "url": "https://dev.to/alice/full-content-article",
                "description": "List description 1",
                "published_at": "2026-05-05T10:00:00Z",
            },
            {
                "id": 2,
                "title": "Missing content article",
                "url": "https://dev.to/bob/missing-content-article",
                "description": "List description 2",
                "published_at": "2026-05-05T11:00:00Z",
            },
        ]
        mock_fetch_dev_to_article_content.side_effect = [
            {
                "title": "Full content article",
                "url": "https://dev.to/alice/full-content-article",
                "description": "List description 1",
                "content": LONG_RSS_SNIPPET_1,
                "published_at": "2026-05-05T10:00:00Z",
                "metadata": {"reading_time_minutes": 5},
            },
            None,
        ]

        raw_items = fetch_rss_articles("https://dev.to/t/ai")
        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_INSUFFICIENT_QUALITY)
        self.assertFalse(hasattr(run, "digest"))
        source_stage = run.metrics.get("source_stage", {})
        ranking_stage = run.metrics.get("ranking_stage", {})
        self.assertEqual(source_stage.get("normalized_source_type"), "dev_to_tag")
        self.assertEqual(source_stage.get("raw_items_count"), 2)
        self.assertEqual(source_stage.get("article_links_extracted"), 2)
        self.assertEqual(source_stage.get("article_contents_fetched"), 1)
        self.assertEqual(source_stage.get("content_unavailable_count"), 1)
        self.assertEqual(source_stage.get("articles_after_cleaning"), 1)
        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertEqual(ranking_stage.get("max_quality_score"), 0.2)
        self.assertEqual(ranking_stage.get("min_actual_quality_score"), 0.2)
        self.assertEqual(ranking_stage.get("average_quality_score"), 0.2)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 0)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 0)
        self.assertEqual(ranking_stage.get("status"), "insufficient_quality")
        self.assertTrue(ranking_stage.get("insufficient_quality"))
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["quality_reasons"])
        self.assertEqual(run.metrics.get("digest_stage", {}).get("status"), "skipped")
        self.assertEqual(run.metrics.get("packaging_stage", {}).get("status"), "skipped")
        self.assertIn("Недостаточно качественных статей", run.error_message)
        self.assertEqual(Article.objects.filter(topic=topic).count(), 1)
        stored_article = Article.objects.get(topic=topic)
        self.assertEqual(stored_article.url, "https://dev.to/alice/full-content-article")
        self.assertEqual(stored_article.raw_payload.get("source_url"), "https://dev.to/t/ai")
        self.assertEqual(stored_article.raw_payload.get("source_api_url"), "https://dev.to/api/articles?tag=ai")

    def test_run_digest_pipeline_stops_when_all_articles_are_below_quality_threshold(self):
        user = get_user_model().objects.create_user(
            username="low-quality-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Broad AI source",
            keywords=["ai"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )
        raw_items = [
            {
                "title": "General AI note",
                "url": "https://example.com/weak-1",
                "source_name": "Example Blog",
                "snippet": (
                    "This article gives a broad note about AI tools in general terms without "
                    "specific mechanisms, measurable outcomes, or a concrete workflow example. "
                    "It stays abstract from start to finish and does not explain what changed in practice."
                ),
            },
            {
                "title": "Another AI note",
                "url": "https://example.com/weak-2",
                "source_name": "Example Blog",
                "snippet": (
                    "Another broad AI update repeats general claims about productivity and future "
                    "impact without offering numbers, process details, or grounded evidence. "
                    "The piece stays vague and disconnected from any specific operating context."
                ),
            },
            {
                "title": "Third AI note",
                "url": "https://example.com/weak-3",
                "source_name": "Example Blog",
                "snippet": (
                    "A third article talks about AI momentum at a high level and gestures toward "
                    "change, but it still avoids examples, tools, or measurable operational detail. "
                    "The result reads like a generic overview rather than a strong source."
                ),
            },
        ]

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_INSUFFICIENT_QUALITY)
        self.assertFalse(hasattr(run, "digest"))
        self.assertEqual(Article.objects.filter(topic=topic).count(), 3)
        ranking_stage = run.metrics.get("ranking_stage", {})
        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertEqual(ranking_stage.get("max_quality_score"), 0.3)
        self.assertEqual(ranking_stage.get("min_actual_quality_score"), 0.3)
        self.assertEqual(ranking_stage.get("average_quality_score"), 0.3)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 0)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 0)
        self.assertEqual(ranking_stage.get("rejected_low_quality_count"), 3)
        self.assertEqual(len(ranking_stage.get("top_rejected_articles", [])), 3)
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["title"])
        self.assertIn("Недостаточно качественных статей", run.error_message)
