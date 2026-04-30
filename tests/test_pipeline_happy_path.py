import json
from pathlib import Path
from datetime import datetime

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.digests.models import DigestRun
from apps.sources.models import Article
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles


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
        self.assertTrue(digest.summary)
        self.assertEqual(len(digest.key_points), 3)
        self.assertEqual(len(digest.sources), 3)

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
        self.assertEqual(source_stage.get("articles_after_dedupe"), 3)
        self.assertEqual(source_stage.get("duplicates_removed"), 1)
        self.assertEqual(source_stage.get("saved_articles_count"), 3)
        self.assertEqual(len(source_stage.get("article_ids", [])), 3)

        self.assertEqual(ranking_stage.get("selected_for_prompt"), 3)
        self.assertEqual(digest_stage.get("provider"), "mock")
        self.assertTrue(digest_stage.get("is_mock"))
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
