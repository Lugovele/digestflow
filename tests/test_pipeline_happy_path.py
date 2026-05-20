import json
from pathlib import Path
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.digests.models import Digest, DigestRun, UsedArticle
from apps.digests import result_messages
from apps.sources.models import Article
from apps.topics.models import Topic
from services.pipeline.run_pipeline import _resolve_quality_threshold, _resolve_source_mode, run_digest_pipeline
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
        self.assertEqual(run.result_message, result_messages.COMPLETED)
        self.assertEqual(run.source_mode, Topic.SOURCE_MODE_AUTOMATIC)
        self.assertEqual(run.quality_threshold_used, 0.4)
        self.assertIsNotNone(run.started_at)
        self.assertIsNotNone(run.finished_at)
        self.assertEqual(run.input_snapshot.get("topic_id"), topic.id)
        self.assertEqual(run.input_snapshot.get("topic_name"), topic.name)
        self.assertEqual(run.input_snapshot.get("source_mode"), Topic.SOURCE_MODE_AUTOMATIC)
        self.assertEqual(run.input_snapshot.get("default_quality_threshold"), 0.4)
        self.assertEqual(run.input_snapshot.get("mode"), "demo")
        self.assertEqual(run.input_snapshot.get("source"), "integration_test")

        self.assertEqual(Article.objects.filter(topic=topic).count(), 3)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), 3)
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
        self.assertIsNone(source_stage.get("source_url"))
        self.assertEqual(source_stage.get("detected_source_type"), "raw_items")
        self.assertIsNone(source_stage.get("detection_reason"))
        self.assertEqual(source_stage.get("article_links_extracted"), 4)
        self.assertEqual(source_stage.get("article_contents_fetched"), 4)
        self.assertEqual(source_stage.get("content_unavailable_count"), 0)
        self.assertEqual(source_stage.get("articles_after_dedupe"), 3)
        self.assertEqual(source_stage.get("duplicates_removed"), 1)
        self.assertEqual(source_stage.get("saved_articles_count"), 3)
        self.assertEqual(len(source_stage.get("article_ids", [])), 3)
        self.assertEqual(source_stage.get("cleaning_rejections"), [])
        used_stage = run.metrics.get("used_articles_stage", {})
        self.assertEqual(used_stage.get("status"), "completed")
        self.assertEqual(used_stage.get("used_article_count"), 3)
        self.assertEqual(len(used_stage.get("used_article_ids", [])), 3)
        used_articles = list(UsedArticle.objects.filter(topic=topic).order_by("id"))
        self.assertTrue(all(article.digest_run_id == run.id for article in used_articles))
        self.assertTrue(all(article.user_id == user.id for article in used_articles))
        self.assertTrue(all(article.use_count == 1 for article in used_articles))
        self.assertTrue(all(article.first_used_in_run_id == run.id for article in used_articles))
        self.assertTrue(all(article.last_used_in_run_id == run.id for article in used_articles))
        self.assertTrue(all(article.first_used_at is not None for article in used_articles))
        self.assertTrue(all(article.last_used_at is not None for article in used_articles))

        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertGreaterEqual(ranking_stage.get("max_quality_score"), 0.8)
        self.assertGreaterEqual(ranking_stage.get("min_actual_quality_score"), 0.6)
        self.assertGreaterEqual(ranking_stage.get("average_quality_score"), 0.6)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 3)
        self.assertEqual(ranking_stage.get("rejected_low_quality_count"), 0)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 3)
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["title"])
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["quality_reasons"])
        self.assertIn("topic_relevance_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("topic_specificity_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("topic_specificity_reason", ranking_stage.get("ranking_scores")[0])
        self.assertIn("specificity_signals", ranking_stage.get("ranking_scores")[0])
        self.assertIn("generic_topic_signals", ranking_stage.get("ranking_scores")[0])
        self.assertIn("evidence_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("practical_value_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("novelty_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("final_quality_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("primary_article_type", ranking_stage.get("ranking_scores")[0])
        self.assertIn("secondary_article_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("weighted_secondary_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("dominant_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("supporting_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("weak_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("article_type", ranking_stage.get("ranking_scores")[0])
        self.assertIn("classification_signal_summary", ranking_stage.get("ranking_scores")[0])
        self.assertIn("dominant_theme_reason", ranking_stage.get("ranking_scores")[0])
        self.assertIn("primary_type_override_reason", ranking_stage.get("ranking_scores")[0])
        self.assertIn("heading_diagnostics", ranking_stage.get("ranking_scores")[0])
        self.assertIn("diversity_penalty", ranking_stage.get("ranking_scores")[0])
        self.assertIn("similarity_reasons", ranking_stage.get("ranking_scores")[0])
        self.assertIn("diversity_adjusted_score", ranking_stage.get("ranking_scores")[0])
        weighted_tags = ranking_stage.get("ranking_scores")[0].get("weighted_secondary_tags", {})
        if weighted_tags:
            sample_payload = next(iter(weighted_tags.values()))
            self.assertIn("title_matches", sample_payload)
            self.assertIn("intro_matches", sample_payload)
            self.assertIn("heading_matches", sample_payload)
            self.assertIn("body_match_count", sample_payload)
            self.assertIn("editorial_weight", sample_payload)
            self.assertIn("body_weight_component", sample_payload)
            self.assertIn("body_saturation_applied", sample_payload)
            self.assertIn("heading_weight_component", sample_payload)
            self.assertIn("heading_boost_capped", sample_payload)
            self.assertIn("dominant_signal_sources", sample_payload)
            self.assertIn("centrality_reason", sample_payload)
        heading_diagnostics = ranking_stage.get("ranking_scores")[0].get("heading_diagnostics", {})
        self.assertIn("detected_headings", heading_diagnostics)
        self.assertIn("normalized_headings", heading_diagnostics)
        self.assertIn("heading_count", heading_diagnostics)
        self.assertIn("heading_source", heading_diagnostics)
        self.assertIn("raw_html_heading_count", heading_diagnostics)
        self.assertIn("extracted_heading_count", heading_diagnostics)
        self.assertIn("heading_extraction_strategy", heading_diagnostics)
        self.assertIn("sample_detected_headings", heading_diagnostics)
        self.assertIn("matched_heading_tags", heading_diagnostics)
        self.assertIn("rejection_reasons", ranking_stage.get("ranking_scores")[0])
        self.assertIn("diagnostic_warnings", ranking_stage.get("ranking_scores")[0])
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

        with patch("services.sources.rss_adapter._fetch_url_text", return_value=""):
            raw_items = fetch_rss_articles(str(Path("tests/fixtures/sample_feed.xml")))
        json.dumps(raw_items[0])

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertEqual(run.result_message, result_messages.COMPLETED)
        self.assertTrue(hasattr(run, "digest"))
        self.assertTrue(hasattr(run.digest, "content_package"))
        article = Article.objects.filter(topic=topic).order_by("id").first()
        self.assertIsNotNone(article)
        self.assertIsInstance(article.published_at, datetime)
        self.assertIsInstance(article.raw_payload.get("published_at"), str)

    def test_run_digest_pipeline_uses_topic_default_quality_threshold(self):
        user = get_user_model().objects.create_user(
            username="topic-threshold-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="AI workflows",
            keywords=["AI automation", "workflow automation"],
            excluded_keywords=[],
            default_quality_threshold=0.2,
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "demo", "source": "integration_test"},
        )

        result = run_digest_pipeline(run.id, get_demo_articles_for_topic(topic.name))
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.quality_threshold_used, 0.2)
        self.assertEqual(run.metrics.get("ranking_stage", {}).get("quality_threshold"), 0.2)

    def test_resolve_quality_threshold_uses_default_fallback_when_run_is_missing(self):
        self.assertEqual(_resolve_quality_threshold(None), 0.4)

    def test_run_digest_pipeline_copies_topic_source_mode_to_run_snapshot(self):
        user = get_user_model().objects.create_user(
            username="source-mode-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="AI source mode",
            keywords=["AI automation", "workflow automation"],
            excluded_keywords=[],
            source_mode=Topic.SOURCE_MODE_HYBRID,
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "demo", "source": "integration_test"},
        )

        run_digest_pipeline(run.id, get_demo_articles_for_topic(topic.name))
        run.refresh_from_db()

        self.assertEqual(run.source_mode, Topic.SOURCE_MODE_HYBRID)
        self.assertEqual(run.input_snapshot.get("source_mode"), Topic.SOURCE_MODE_HYBRID)

    def test_run_digest_pipeline_stores_topic_configuration_in_input_snapshot(self):
        user = get_user_model().objects.create_user(
            username="snapshot-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Snapshot topic",
            keywords=["AI automation"],
            excluded_keywords=[],
            source_mode=Topic.SOURCE_MODE_CUSTOM_ONLY,
            default_quality_threshold=0.35,
            source_url="https://dev.to/t/ai",
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "manual", "custom_flag": True},
        )

        run_digest_pipeline(run.id, get_demo_articles_for_topic(topic.name))
        run.refresh_from_db()

        self.assertEqual(run.input_snapshot.get("topic_id"), topic.id)
        self.assertEqual(run.input_snapshot.get("topic_name"), "Snapshot topic")
        self.assertEqual(run.input_snapshot.get("source_mode"), Topic.SOURCE_MODE_CUSTOM_ONLY)
        self.assertEqual(run.input_snapshot.get("default_quality_threshold"), 0.35)
        self.assertEqual(run.input_snapshot.get("source_url"), "https://dev.to/t/ai")
        self.assertEqual(run.input_snapshot.get("mode"), "manual")
        self.assertTrue(run.input_snapshot.get("custom_flag"))

    def test_resolve_source_mode_keeps_existing_value_without_topic(self):
        run = SimpleNamespace(topic=None, source_mode=Topic.SOURCE_MODE_CUSTOM_ONLY)
        self.assertEqual(_resolve_source_mode(run), Topic.SOURCE_MODE_CUSTOM_ONLY)

    def test_resolve_source_mode_uses_automatic_default_without_topic_or_value(self):
        run = SimpleNamespace(topic=None, source_mode="")
        self.assertEqual(_resolve_source_mode(run), Topic.SOURCE_MODE_AUTOMATIC)

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
        self.assertEqual(source_stage.get("cleaning_rejections"), [])

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
        self.assertEqual(run.result_message, result_messages.INSUFFICIENT_QUALITY)
        self.assertFalse(hasattr(run, "digest"))
        source_stage = run.metrics.get("source_stage", {})
        ranking_stage = run.metrics.get("ranking_stage", {})
        self.assertEqual(source_stage.get("source_url"), "https://dev.to/t/ai")
        self.assertEqual(source_stage.get("detected_source_type"), "devto_tag")
        self.assertEqual(source_stage.get("detection_reason"), "matched dev.to topic pattern")
        self.assertEqual(source_stage.get("normalized_source_type"), "devto_tag")
        self.assertEqual(source_stage.get("raw_items_count"), 2)
        self.assertEqual(source_stage.get("article_links_extracted"), 2)
        self.assertEqual(source_stage.get("article_contents_fetched"), 1)
        self.assertEqual(source_stage.get("content_unavailable_count"), 1)
        self.assertEqual(source_stage.get("articles_after_cleaning"), 1)
        self.assertEqual(source_stage.get("removed_during_cleaning"), 1)
        self.assertEqual(source_stage.get("full_article_count"), 1)
        self.assertEqual(source_stage.get("rich_summary_count"), 0)
        self.assertEqual(source_stage.get("weak_snippet_count"), 1)
        self.assertEqual(source_stage.get("missing_content_count"), 0)
        self.assertEqual(
            source_stage.get("cleaning_rejections"),
            [
                {
                    "title": "Missing content article",
                    "url": "https://dev.to/bob/missing-content-article",
                    "source_name": "DEV Community",
                    "reason": "content too short",
                    "content_tier": "weak_snippet",
                    "final_content_source": "rss_summary",
                    "content_length": len("List description 2"),
                    "content_preview": "List description 2",
                    "extraction_method": None,
                    "extraction_warning": None,
                    "extraction_candidates": [],
                }
            ],
        )
        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertEqual(ranking_stage.get("max_quality_score"), 0.4)
        self.assertEqual(ranking_stage.get("min_actual_quality_score"), 0.4)
        self.assertEqual(ranking_stage.get("average_quality_score"), 0.4)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 1)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 1)
        self.assertEqual(ranking_stage.get("status"), "insufficient_quality")
        self.assertTrue(ranking_stage.get("insufficient_quality"))
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["quality_reasons"])
        self.assertIn("topic_relevance_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("topic_specificity_score", ranking_stage.get("ranking_scores")[0])
        self.assertIn("primary_article_type", ranking_stage.get("ranking_scores")[0])
        self.assertIn("secondary_article_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("weighted_secondary_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("dominant_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("supporting_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("weak_tags", ranking_stage.get("ranking_scores")[0])
        self.assertIn("classification_signal_summary", ranking_stage.get("ranking_scores")[0])
        self.assertIn("dominant_theme_reason", ranking_stage.get("ranking_scores")[0])
        self.assertIn("primary_type_override_reason", ranking_stage.get("ranking_scores")[0])
        self.assertIn("rejection_reasons", ranking_stage.get("ranking_scores")[0])
        self.assertIn("heading_diagnostics", ranking_stage.get("ranking_scores")[0])
        self.assertEqual(run.metrics.get("digest_stage", {}).get("status"), "skipped")
        self.assertEqual(run.metrics.get("packaging_stage", {}).get("status"), "skipped")
        self.assertFalse(run.metrics.get("used_articles_stage"))
        self.assertIn("Недостаточно качественных статей", run.error_message)
        self.assertEqual(Article.objects.filter(topic=topic).count(), 1)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), 0)
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
        self.assertEqual(run.result_message, result_messages.INSUFFICIENT_QUALITY)
        self.assertFalse(hasattr(run, "digest"))
        self.assertEqual(Article.objects.filter(topic=topic).count(), 3)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), 0)
        ranking_stage = run.metrics.get("ranking_stage", {})
        self.assertEqual(ranking_stage.get("quality_threshold"), 0.4)
        self.assertLess(ranking_stage.get("max_quality_score"), 0.4)
        self.assertGreaterEqual(ranking_stage.get("max_quality_score"), 0.3)
        self.assertLess(ranking_stage.get("min_actual_quality_score"), 0.4)
        self.assertGreaterEqual(ranking_stage.get("min_actual_quality_score"), 0.2)
        self.assertLess(ranking_stage.get("average_quality_score"), 0.4)
        self.assertGreaterEqual(ranking_stage.get("average_quality_score"), 0.2)
        self.assertEqual(ranking_stage.get("articles_above_quality_threshold"), 0)
        self.assertEqual(ranking_stage.get("selected_for_prompt"), 0)
        self.assertEqual(ranking_stage.get("rejected_low_quality_count"), 3)
        self.assertEqual(len(ranking_stage.get("top_rejected_articles", [])), 3)
        self.assertTrue(ranking_stage.get("ranking_scores")[0]["title"])
        self.assertIn("topic_relevance_score", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("topic_specificity_score", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("primary_article_type", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("secondary_article_tags", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("weighted_secondary_tags", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("dominant_tags", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("supporting_tags", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("weak_tags", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("classification_signal_summary", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("dominant_theme_reason", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("primary_type_override_reason", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("rejection_reasons", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("heading_diagnostics", ranking_stage.get("top_rejected_articles")[0])
        self.assertIn("Недостаточно качественных статей", run.error_message)
    def test_run_digest_pipeline_records_cleaning_rejections_in_source_metrics(self):
        user = get_user_model().objects.create_user(
            username="cleaning-diagnostics-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Cleaning diagnostics",
            keywords=["workflow"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )
        raw_items = [
            {
                "title": "Valid workflow article",
                "url": "https://example.com/valid",
                "source_name": "Example Source",
                "snippet": LONG_RSS_SNIPPET_1,
            },
            {
                "title": "Missing content article",
                "url": "https://example.com/no-content",
                "source_name": "Example Source",
                "content": "",
                "snippet": "",
                "metadata": {
                    "extraction_method": "rss_summary_fallback",
                    "extraction_warning": "html fetch failed; RSS summary used",
                    "extraction_candidates": [],
                },
            },
            {
                "title": "Tiny article",
                "url": "https://example.com/tiny",
                "source_name": "Example Source",
                "content": "<p>Too short.</p>",
                "metadata": {
                    "extraction_method": "article_tag",
                    "extraction_warning": "extracted content is very short",
                    "extraction_candidates": [],
                },
            },
        ]

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        source_stage = run.metrics.get("source_stage", {})
        self.assertEqual(source_stage.get("articles_after_cleaning"), 1)
        self.assertEqual(source_stage.get("removed_during_cleaning"), 2)
        self.assertEqual(source_stage.get("full_article_count"), 0)
        self.assertEqual(source_stage.get("rich_summary_count"), 1)
        self.assertEqual(source_stage.get("weak_snippet_count"), 1)
        self.assertEqual(source_stage.get("missing_content_count"), 1)
        self.assertEqual(
            source_stage.get("cleaning_rejections"),
            [
                {
                    "title": "Missing content article",
                    "url": "https://example.com/no-content",
                    "source_name": "Example Source",
                    "reason": "missing extracted content",
                    "content_tier": "missing_content",
                    "final_content_source": "rss_summary",
                    "content_length": 0,
                    "content_preview": "",
                    "extraction_method": "rss_summary_fallback",
                    "extraction_warning": "html fetch failed; RSS summary used",
                    "extraction_candidates": [],
                },
                {
                    "title": "Tiny article",
                    "url": "https://example.com/tiny",
                    "source_name": "Example Source",
                    "reason": "content too short",
                    "content_tier": "weak_snippet",
                    "final_content_source": "html_article_body",
                    "content_length": len("Too short."),
                    "content_preview": "Too short.",
                    "extraction_method": "article_tag",
                    "extraction_warning": "extracted content is very short",
                    "extraction_candidates": [],
                },
            ],
        )

    def test_run_digest_pipeline_populates_result_message_for_source_stage_failure(self):
        user = get_user_model().objects.create_user(
            username="source-failure-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Empty source topic",
            keywords=["ai"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )

        result = run_digest_pipeline(run.id, [])
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_FAILED)
        self.assertEqual(run.result_message, result_messages.SOURCE_NO_USABLE_ARTICLES)
        self.assertEqual(run.error_message, "Source stage returned no articles.")

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    @patch("services.pipeline.run_pipeline.rank_source_items")
    def test_successful_digest_records_only_selected_for_prompt_articles(
        self,
        mock_rank_source_items,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        user = get_user_model().objects.create_user(
            username="used-article-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Used article selection",
            keywords=["workflow"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )
        raw_items = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_1},
            {"title": "Selected two", "url": "https://example.com/selected-2", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_2},
            {"title": "Selected one duplicate", "url": "https://example.com/selected-1", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_1},
            {"title": "Rejected four", "url": "https://example.com/rejected-4", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_3},
        ]
        selected_items = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "source_name": "Example", "source_url": "https://example.com/feed"},
            {"title": "Selected two", "url": "https://example.com/selected-2", "source_name": "Example", "source_url": "https://example.com/feed"},
            {"title": "Selected one duplicate", "url": "https://example.com/selected-1", "source_name": "Example", "source_url": "https://example.com/feed"},
        ]
        ranking_scores = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "quality_score": 0.9, "quality_reasons": ["strong relevance"], "rejection_reasons": []},
            {"title": "Selected two", "url": "https://example.com/selected-2", "quality_score": 0.8, "quality_reasons": ["strong relevance"], "rejection_reasons": []},
            {"title": "Rejected four", "url": "https://example.com/rejected-4", "quality_score": 0.2, "quality_reasons": ["weak"], "rejection_reasons": ["low quality"]},
        ]
        mock_rank_source_items.return_value = (selected_items, ranking_scores)
        mock_save_articles_for_topic.return_value = []

        digest = Digest.objects.create(
            run=run,
            title="Digest for Used article selection",
            payload={
                "title": "Digest for Used article selection",
                "articles": [
                    {"url": "https://example.com/selected-1", "title": "Selected one", "summary": "Summary"},
                    {"url": "https://example.com/selected-2", "title": "Selected two", "summary": "Summary"},
                ],
            },
            quality_score=0.8,
        )
        mock_generate_digest_for_run.return_value = (
            digest,
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )

        result = run_digest_pipeline(run.id, raw_items)
        run.refresh_from_db()

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        used_articles = list(UsedArticle.objects.filter(topic=topic).order_by("normalized_url"))
        self.assertEqual(len(used_articles), 2)
        self.assertEqual(
            [article.normalized_url for article in used_articles],
            ["https://example.com/selected-1", "https://example.com/selected-2"],
        )
        self.assertTrue(all(article.digest_run_id == run.id for article in used_articles))
        self.assertTrue(all(article.user_id == user.id for article in used_articles))
        self.assertTrue(all(article.source_url == "https://example.com/feed" for article in used_articles))
        self.assertTrue(all(article.use_count == 1 for article in used_articles))
        self.assertTrue(all(article.first_used_in_run_id == run.id for article in used_articles))
        self.assertTrue(all(article.last_used_in_run_id == run.id for article in used_articles))
        self.assertFalse(UsedArticle.objects.filter(topic=topic, normalized_url="https://example.com/rejected-4").exists())

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    @patch("services.pipeline.run_pipeline.rank_source_items")
    def test_successful_digest_repeat_use_updates_topic_history_without_duplicate_rows(
        self,
        mock_rank_source_items,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        user = get_user_model().objects.create_user(
            username="used-article-repeat-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Used article repeat visibility",
            keywords=["workflow"],
            excluded_keywords=[],
        )
        first_run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )
        second_run = DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )
        raw_items = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_1},
            {"title": "Selected two", "url": "https://example.com/selected-2", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_2},
            {"title": "Rejected four", "url": "https://example.com/rejected-4", "source_name": "Example", "snippet": LONG_RSS_SNIPPET_3},
        ]
        selected_items = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "source_name": "Example", "source_url": "https://example.com/feed"},
            {"title": "Selected two", "url": "https://example.com/selected-2", "source_name": "Example", "source_url": "https://example.com/feed"},
        ]
        ranking_scores = [
            {"title": "Selected one", "url": "https://example.com/selected-1", "quality_score": 0.9, "quality_reasons": ["strong relevance"], "rejection_reasons": []},
            {"title": "Selected two", "url": "https://example.com/selected-2", "quality_score": 0.85, "quality_reasons": ["strong relevance"], "rejection_reasons": []},
            {"title": "Rejected four", "url": "https://example.com/rejected-4", "quality_score": 0.2, "quality_reasons": ["weak"], "rejection_reasons": ["low quality"]},
        ]
        mock_rank_source_items.return_value = (selected_items, ranking_scores)
        mock_save_articles_for_topic.return_value = []

        first_digest = Digest.objects.create(
            run=first_run,
            title="First digest",
            payload={"title": "First digest", "articles": [
                {"url": "https://example.com/selected-1", "title": "Selected one", "summary": "Summary"},
                {"url": "https://example.com/selected-2", "title": "Selected two", "summary": "Summary"},
            ]},
            quality_score=0.8,
        )
        second_digest = Digest.objects.create(
            run=second_run,
            title="Second digest",
            payload={"title": "Second digest", "articles": [
                {"url": "https://example.com/selected-1", "title": "Selected one", "summary": "Summary"},
                {"url": "https://example.com/selected-2", "title": "Selected two", "summary": "Summary"},
            ]},
            quality_score=0.8,
        )
        mock_generate_digest_for_run.side_effect = [
            (first_digest, {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None}),
            (second_digest, {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None}),
        ]
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )

        first_result = run_digest_pipeline(first_run.id, raw_items)
        history_after_first = UsedArticle.objects.get(topic=topic, normalized_url="https://example.com/selected-1")
        first_used_at = history_after_first.first_used_at
        last_used_at = history_after_first.last_used_at

        second_result = run_digest_pipeline(second_run.id, raw_items)
        topic_history = UsedArticle.objects.get(topic=topic, normalized_url="https://example.com/selected-1")

        self.assertEqual(first_result.id, first_run.id)
        self.assertEqual(second_result.id, second_run.id)
        self.assertEqual(UsedArticle.objects.filter(topic=topic, normalized_url="https://example.com/selected-1").count(), 1)
        self.assertEqual(topic_history.use_count, 2)
        self.assertEqual(topic_history.first_used_in_run_id, first_run.id)
        self.assertEqual(topic_history.last_used_in_run_id, second_run.id)
        self.assertEqual(topic_history.digest_run_id, second_run.id)
        self.assertEqual(topic_history.first_used_at, first_used_at)
        self.assertGreaterEqual(topic_history.last_used_at, last_used_at)
        self.assertFalse(UsedArticle.objects.filter(topic=topic, normalized_url="https://example.com/rejected-4").exists())
