from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic


@override_settings(OPENAI_API_KEY="sk-your-key")
class DigestPipelineFailureTests(TestCase):
    def test_run_digest_pipeline_marks_partial_failed_when_packaging_stage_crashes(self):
        user = get_user_model().objects.create_user(
            username="pipeline-failure-user",
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
                "source": "integration_test_failure",
            },
        )
        raw_items = get_demo_articles_for_topic(topic.name)

        with patch(
            "services.pipeline.run_pipeline.generate_content_package_for_digest",
            side_effect=RuntimeError("Simulated packaging failure"),
        ):
            result = run_digest_pipeline(run.id, raw_items)

        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_PARTIAL_FAILED)
        self.assertIn("Packaging stage failed", run.error_message)
        self.assertIn("Simulated packaging failure", run.error_message)
        self.assertIsNotNone(run.finished_at)

        digest = run.digest
        self.assertIsNotNone(digest)
        self.assertFalse(hasattr(digest, "content_package"))

        self.assertIn("digest_stage", run.metrics)
        packaging_stage = run.metrics.get("packaging_stage", {})
        if packaging_stage:
            self.assertEqual(packaging_stage.get("status"), "failed")
            self.assertEqual(packaging_stage.get("digest_id"), digest.id)
            self.assertIn("Simulated packaging failure", packaging_stage.get("error", ""))

    def test_run_digest_pipeline_marks_failed_when_source_stage_returns_no_articles(self):
        user = get_user_model().objects.create_user(
            username="pipeline-empty-source-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Empty source topic",
            keywords=["empty source"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={
                "mode": "demo",
                "source": "integration_test_empty_source",
            },
        )

        result = run_digest_pipeline(run.id, raw_items=[])

        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_FAILED)
        self.assertIn("Source stage returned no articles", run.error_message)
        self.assertIsNotNone(run.finished_at)
        self.assertFalse(hasattr(run, "digest"))

        self.assertFalse(run.metrics.get("digest_stage"))
        self.assertFalse(run.metrics.get("packaging_stage"))
