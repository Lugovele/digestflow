from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings

from apps.digests.models import Digest, DigestRun, UsedArticle
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline


LONG_SNIPPET_1 = (
    "A content team cut prep from 6 hours to 2.5 hours, but editors still checked "
    "every claim before publishing. The workflow got faster at the start, yet the review "
    "step stayed manual, and the team still spent time cleaning up unsupported details "
    "before anything was ready to ship."
)
LONG_SNIPPET_2 = (
    "A support team got triage 28% faster with structured forms, but bad labels still "
    "broke routing. The queue moved quicker at intake, yet handoffs still failed later, "
    "and operators had to step back in to correct tickets that landed with the wrong team."
)
LONG_SNIPPET_3 = (
    "An ops team cut review time by 35% after redesigning the workflow first and adding "
    "AI later. They changed the handoff before the model step, made validation clearer, "
    "and stopped losing time in the same back-and-forth review loop."
)


@override_settings(OPENAI_API_KEY="sk-your-key")
class UsedArticleExclusionTests(TestCase):
    def _make_topic(self, username: str, name: str) -> Topic:
        user = get_user_model().objects.create_user(username=username, password="not-used-in-test")
        return Topic.objects.create(
            user=user,
            name=name,
            keywords=["workflow", "automation"],
            excluded_keywords=[],
            default_quality_threshold=0.2,
        )

    def _build_raw_items(self) -> list[dict]:
        return [
            {
                "title": "Briefing workflow cuts prep time",
                "url": "https://example.com/used-article",
                "source_name": "Example Source",
                "snippet": LONG_SNIPPET_1,
                "source_url": "https://example.com/feed",
            },
            {
                "title": "Support triage gets faster with forms",
                "url": "https://example.com/fresh-article-1",
                "source_name": "Example Source",
                "snippet": LONG_SNIPPET_2,
                "source_url": "https://example.com/feed",
            },
            {
                "title": "Workflow redesign shortens review cycles",
                "url": "https://example.com/fresh-article-2",
                "source_name": "Example Source",
                "snippet": LONG_SNIPPET_3,
                "source_url": "https://example.com/feed",
            },
        ]

    def _make_digest_side_effect(self, selected_payloads: list[list[dict]]):
        def _side_effect(run: DigestRun, selected_items: list[dict]):
            selected_payloads.append(selected_items)
            digest = Digest.objects.create(
                run=run,
                title=f"Digest for {run.topic.name}",
                payload={
                    "title": f"Digest for {run.topic.name}",
                    "articles": [
                        {
                            "url": item["url"],
                            "title": item["title"],
                            "summary": "Summary",
                        }
                        for item in selected_items
                    ],
                },
                quality_score=0.8,
            )
            return digest, {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None}

        return _side_effect

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    def test_same_topic_used_article_is_excluded_before_selected_for_prompt(
        self,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        topic = self._make_topic("used-article-exclusion-user", "Used article exclusion")
        run = DigestRun.objects.create(topic=topic, input_snapshot={"mode": "raw_items", "source": "integration_test"})
        UsedArticle.objects.create(
            user=topic.user,
            topic=topic,
            digest_run=run,
            first_used_in_run=run,
            last_used_in_run=run,
            normalized_url="https://example.com/used-article",
            article_url="https://example.com/used-article",
            title="Briefing workflow cuts prep time",
            source_url="https://example.com/feed",
            use_count=1,
        )
        selected_payloads: list[list[dict]] = []
        mock_generate_digest_for_run.side_effect = self._make_digest_side_effect(selected_payloads)
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )
        mock_save_articles_for_topic.return_value = []

        result = run_digest_pipeline(run.id, self._build_raw_items())
        run.refresh_from_db()

        self.assertEqual(result.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.metrics["ranking_stage"]["articles_excluded_as_used"], 1)
        self.assertEqual(run.metrics["ranking_stage"]["articles_remaining_after_used_filter"], 2)
        self.assertEqual(run.metrics["ranking_stage"]["used_article_count_for_topic"], 1)
        self.assertTrue(run.metrics["ranking_stage"]["used_article_filter_enabled"])
        self.assertEqual(len(selected_payloads[0]), 2)
        self.assertEqual(
            {item["url"] for item in selected_payloads[0]},
            {
                "https://example.com/fresh-article-1",
                "https://example.com/fresh-article-2",
            },
        )
        self.assertFalse(any(item["url"] == "https://example.com/used-article" for item in selected_payloads[0]))
        self.assertEqual(UsedArticle.objects.filter(topic=topic, normalized_url="https://example.com/used-article").count(), 1)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), 3)

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    def test_used_article_from_other_topic_is_not_excluded(
        self,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        topic = self._make_topic("used-article-exclusion-current", "Current topic")
        other_topic = self._make_topic("used-article-exclusion-other", "Other topic")
        other_run = DigestRun.objects.create(topic=other_topic, input_snapshot={"mode": "raw_items", "source": "integration_test"})
        UsedArticle.objects.create(
            user=other_topic.user,
            topic=other_topic,
            digest_run=other_run,
            first_used_in_run=other_run,
            last_used_in_run=other_run,
            normalized_url="https://example.com/used-article",
            article_url="https://example.com/used-article",
            title="Used in other topic",
            source_url="https://example.com/feed",
            use_count=1,
        )
        run = DigestRun.objects.create(topic=topic, input_snapshot={"mode": "raw_items", "source": "integration_test"})
        selected_payloads: list[list[dict]] = []
        mock_generate_digest_for_run.side_effect = self._make_digest_side_effect(selected_payloads)
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )
        mock_save_articles_for_topic.return_value = []

        result = run_digest_pipeline(run.id, self._build_raw_items())
        run.refresh_from_db()

        self.assertEqual(result.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.metrics["ranking_stage"]["articles_excluded_as_used"], 0)
        self.assertEqual(run.metrics["ranking_stage"]["used_article_count_for_topic"], 0)
        self.assertTrue(any(item["url"] == "https://example.com/used-article" for item in selected_payloads[0]))

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    def test_when_all_ranked_articles_are_used_pipeline_does_not_reuse_them(
        self,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        topic = self._make_topic("used-article-exclusion-all-used", "All used topic")
        run = DigestRun.objects.create(topic=topic, input_snapshot={"mode": "raw_items", "source": "integration_test"})
        for url, title in [
            ("https://example.com/used-article", "Used workflow article"),
            ("https://example.com/fresh-article-1", "Fresh workflow article one"),
            ("https://example.com/fresh-article-2", "Fresh workflow article two"),
        ]:
            UsedArticle.objects.create(
                user=topic.user,
                topic=topic,
                digest_run=run,
                first_used_in_run=run,
                last_used_in_run=run,
                normalized_url=url,
                article_url=url,
                title=title,
                source_url="https://example.com/feed",
                use_count=1,
            )
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )
        mock_save_articles_for_topic.return_value = []

        result = run_digest_pipeline(run.id, self._build_raw_items())
        run.refresh_from_db()

        self.assertEqual(result.status, DigestRun.STATUS_INSUFFICIENT_QUALITY)
        self.assertEqual(run.metrics["ranking_stage"]["articles_excluded_as_used"], 3)
        self.assertEqual(run.metrics["ranking_stage"]["articles_remaining_after_used_filter"], 0)
        self.assertEqual(run.metrics["ranking_stage"]["selected_for_prompt"], 0)
        self.assertFalse(mock_generate_digest_for_run.called)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), 3)

    @patch("services.pipeline.run_pipeline.generate_content_package_for_digest")
    @patch("services.pipeline.run_pipeline.generate_digest_for_run")
    @patch("services.pipeline.run_pipeline.save_articles_for_topic")
    def test_without_used_article_rows_existing_success_behavior_remains(
        self,
        mock_save_articles_for_topic,
        mock_generate_digest_for_run,
        mock_generate_content_package_for_digest,
    ):
        topic = self._make_topic("used-article-exclusion-none", "No used history topic")
        run = DigestRun.objects.create(topic=topic, input_snapshot={"mode": "raw_items", "source": "integration_test"})
        selected_payloads: list[list[dict]] = []
        mock_generate_digest_for_run.side_effect = self._make_digest_side_effect(selected_payloads)
        mock_generate_content_package_for_digest.return_value = (
            SimpleNamespace(id=1),
            {"provider": "mock", "is_mock": True, "tokens": None, "estimated_cost_usd": None},
        )
        mock_save_articles_for_topic.return_value = []

        result = run_digest_pipeline(run.id, self._build_raw_items())
        run.refresh_from_db()

        self.assertEqual(result.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.metrics["ranking_stage"]["articles_excluded_as_used"], 0)
        self.assertEqual(run.metrics["ranking_stage"]["used_article_count_for_topic"], 0)
        self.assertGreaterEqual(run.metrics["ranking_stage"]["selected_for_prompt"], 2)
        self.assertEqual(UsedArticle.objects.filter(topic=topic).count(), len(selected_payloads[0]))
