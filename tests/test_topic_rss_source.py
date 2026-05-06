from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from apps.digests.forms import TopicInputForm
from apps.digests.models import DigestRun
from apps.sources.models import Article
from apps.topics.models import Topic

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


class TopicRssSourceTests(TestCase):
    def test_topic_can_store_source_url(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI Ops",
            source_url="https://example.com/feed.xml",
            keywords=["AI Ops"],
            excluded_keywords=[],
        )

        self.assertEqual(topic.source_url, "https://example.com/feed.xml")

    def test_topic_form_accepts_optional_source_url(self) -> None:
        form = TopicInputForm(
            data={
                "topic_name": "AI Ops",
                "source_url": "https://example.com/feed.xml",
            }
        )

        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["source_url"], "https://example.com/feed.xml")

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_quick_start_with_source_url_uses_rss_items(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        rss_items = [
            {
                "title": "AI Ops case",
                "url": "https://example.com/articles/1",
                "source_name": "Example Feed",
                "snippet": "Snippet",
                "published_at": "2026-05-05T10:00:00+00:00",
            }
        ]
        mock_fetch_rss_articles.return_value = rss_items

        response = self.client.post(
            reverse("create-topic-and-run"),
            data={
                "topic_name": "AI Ops",
                "source_url": "https://example.com/feed.xml",
            },
        )

        self.assertEqual(response.status_code, 302)
        topic = Topic.objects.get(name="AI Ops")
        run = DigestRun.objects.get(topic=topic)
        self.assertEqual(topic.source_url, "https://example.com/feed.xml")
        mock_fetch_rss_articles.assert_called_once_with("https://example.com/feed.xml")
        mock_run_digest_pipeline.assert_called_once_with(run.id, raw_items=rss_items)

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_existing_topic_run_with_source_url_uses_rss_items(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI Support",
            source_url="https://example.com/support.xml",
            keywords=["AI Support"],
            excluded_keywords=[],
        )
        rss_items = [
            {
                "title": "Support triage",
                "url": "https://example.com/articles/2",
                "source_name": "Support Feed",
                "snippet": "Snippet",
                "published_at": "2026-05-05T11:00:00+00:00",
            }
        ]
        mock_fetch_rss_articles.return_value = rss_items

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.get(topic=topic)
        mock_fetch_rss_articles.assert_called_once_with("https://example.com/support.xml")
        mock_run_digest_pipeline.assert_called_once_with(run.id, raw_items=rss_items)

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_empty_rss_marks_run_failed_with_clear_error(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty RSS",
            source_url="https://example.com/empty.xml",
            keywords=["Empty RSS"],
            excluded_keywords=[],
        )
        mock_fetch_rss_articles.return_value = []

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.get(topic=topic)
        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_FAILED)
        self.assertEqual(
            run.error_message,
            "RSS source returned no valid items: https://example.com/empty.xml",
        )
        mock_run_digest_pipeline.assert_not_called()

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.get_demo_articles_for_topic")
    def test_topic_without_source_url_uses_demo_articles_as_before(
        self,
        mock_get_demo_articles_for_topic,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Demo Topic",
            keywords=["Demo Topic"],
            excluded_keywords=[],
        )
        demo_items = [
            {
                "title": "Demo article",
                "url": "https://example.com/demo",
                "source_name": "Demo",
                "snippet": "Snippet",
                "published_at": None,
            }
        ]
        mock_get_demo_articles_for_topic.return_value = demo_items

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.get(topic=topic)
        mock_get_demo_articles_for_topic.assert_called_once_with("Demo Topic")
        mock_run_digest_pipeline.assert_called_once_with(run.id, raw_items=demo_items)

    def test_topic_list_shows_source_url(self) -> None:
        Topic.objects.create(
            user=self._get_ui_user(),
            name="Visible Feed",
            source_url="https://example.com/visible.xml",
            keywords=["Visible Feed"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, "https://example.com/visible.xml")
        self.assertContains(response, "Source RSS URL")

    @override_settings(OPENAI_API_KEY="sk-your-key")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_ui_run_with_source_url_completes_with_snippet_only_rss_items(
        self,
        mock_fetch_rss_articles,
    ) -> None:
        rss_items = [
            {
                "title": "Briefing workflow cuts prep time",
                "url": "https://example.com/articles/10",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_1,
                "published_at": "2026-05-05T10:00:00+00:00",
            },
            {
                "title": "Support triage gets faster with forms",
                "url": "https://example.com/articles/11",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_2,
                "published_at": "2026-05-05T11:00:00+00:00",
            },
            {
                "title": "Workflow redesign shortens review cycles",
                "url": "https://example.com/articles/12",
                "source_name": "DEV Community: Example",
                "snippet": LONG_RSS_SNIPPET_3,
                "published_at": "2026-05-05T12:00:00+00:00",
            },
        ]
        mock_fetch_rss_articles.return_value = rss_items

        response = self.client.post(
            reverse("create-topic-and-run"),
            data={
                "topic_name": "Snippet RSS Topic",
                "source_url": "https://dev.to/feed/example",
            },
        )

        self.assertEqual(response.status_code, 302)
        topic = Topic.objects.get(name="Snippet RSS Topic")
        run = DigestRun.objects.get(topic=topic)
        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertTrue(hasattr(run, "digest"))
        self.assertTrue(hasattr(run.digest, "content_package"))
        self.assertGreater(Article.objects.filter(topic=topic).count(), 0)

    def _get_ui_user(self):
        user_model = Topic._meta.get_field("user").remote_field.model
        return user_model.objects.create_user(username="tester")
