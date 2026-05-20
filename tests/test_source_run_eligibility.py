from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.digests.models import DigestRun
from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin


class SourceRunEligibilityTests(TestCase):
    def _create_topic(self, *, source_mode: str = TopicSourceMode.HYBRID) -> Topic:
        user = get_user_model().objects.create_user(
            username=f"run-eligibility-{source_mode}-{Topic.objects.count()}",
            password="pw",
        )
        return Topic.objects.create(
            user=user,
            name=f"Topic {Topic.objects.count() + 1}",
            source_mode=source_mode,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

    def _add_manual_source(self, topic: Topic, *, is_active: bool = True) -> TopicSource:
        return TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=is_active,
        )

    def _add_research_source(self, topic: Topic, *, is_active: bool = True, is_pinned: bool = False) -> TopicSource:
        slug = "kept-ai" if is_pinned else "new-ai"
        return TopicSource.objects.create(
            topic=topic,
            name="Research source",
            url=f"https://dev.to/t/{slug}",
            normalized_url=f"https://dev.to/api/articles?tag={slug}",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=is_pinned,
            is_active=is_active,
        )

    def _assert_run_disabled(self, response, message: str) -> None:
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run digest")
        self.assertContains(response, message)
        self.assertContains(response, "disabled", html=False)

    def _assert_run_enabled(self, response, message: str) -> None:
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Run digest")
        self.assertContains(response, message)
        self.assertNotContains(response, 'class="primary-cta" disabled', html=False)

    def test_my_sources_only_with_zero_active_manual_sources_cannot_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.CURATED_ONLY)
        self._add_manual_source(topic, is_active=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_disabled(response, "Select at least one my source before running this digest.")

    def test_my_sources_only_with_active_manual_source_can_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.CURATED_ONLY)
        self._add_manual_source(topic, is_active=True)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_enabled(response, "1 selected source will be used in the next digest run.")

    def test_research_only_with_zero_active_research_sources_cannot_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.DISCOVERY_ONLY)
        self._add_research_source(topic, is_active=False, is_pinned=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_disabled(response, "Find or keep at least one research source before running this digest.")

    def test_research_only_with_active_research_source_can_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.DISCOVERY_ONLY)
        self._add_research_source(topic, is_active=True, is_pinned=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_enabled(response, "1 selected source will be used in the next digest run.")

    def test_hybrid_with_only_manual_sources_cannot_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_manual_source(topic, is_active=True)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_disabled(response, "Find or keep at least one research source before running this digest.")

    def test_hybrid_with_only_research_sources_cannot_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_research_source(topic, is_active=True, is_pinned=True)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_disabled(response, "Select at least one my source before running this digest.")

    def test_hybrid_with_manual_and_research_sources_can_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_manual_source(topic, is_active=True)
        self._add_research_source(topic, is_active=True, is_pinned=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_enabled(response, "2 selected sources will be used in the next digest run.")

    def test_kept_research_sources_count_as_research_sources(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.DISCOVERY_ONLY)
        self._add_research_source(topic, is_active=True, is_pinned=True)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_enabled(response, "1 selected source will be used in the next digest run.")

    def test_new_unpinned_research_sources_count_as_research_sources(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.DISCOVERY_ONLY)
        self._add_research_source(topic, is_active=True, is_pinned=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_enabled(response, "1 selected source will be used in the next digest run.")

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_invalid_run_post_does_not_create_digest_run(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_manual_source(topic, is_active=True)

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)
        mock_fetch_rss_articles.assert_not_called()
        mock_run_digest_pipeline.assert_not_called()

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_invalid_run_post_preserves_source_state_without_workspace_mutation(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        manual_source = self._add_manual_source(topic, is_active=True)

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        topic.refresh_from_db()
        manual_source.refresh_from_db()
        self.assertEqual(topic.source_mode, TopicSourceMode.HYBRID)
        self.assertTrue(manual_source.is_active)
        self.assertEqual(manual_source.origin, TopicSourceOrigin.MANUAL)
        mock_fetch_rss_articles.assert_not_called()
        mock_run_digest_pipeline.assert_not_called()

    def test_ready_to_generate_message_explains_missing_source_category(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_research_source(topic, is_active=True, is_pinned=False)

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self._assert_run_disabled(response, "Select at least one my source before running this digest.")

    def test_topic_list_hybrid_with_only_manual_sources_disables_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_manual_source(topic, is_active=True)

        response = self.client.get(reverse("topic-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "my sources & research")
        self.assertContains(response, "1 my source")
        self.assertContains(response, "0 research sources")
        self.assertContains(response, "Needs a research source")
        self.assertContains(
            response,
            f'aria-label="Run disabled: Needs a research source"',
            html=False,
        )
        topic_row = response.content.decode("utf-8").split(topic.name, 1)[1].split("Delete topic", 1)[0]
        self.assertLess(topic_row.index("Needs a research source"), topic_row.index(">Run</button>"))

    def test_topic_list_hybrid_with_manual_and_research_sources_enables_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.HYBRID)
        self._add_manual_source(topic, is_active=True)
        self._add_research_source(topic, is_active=True, is_pinned=True)

        response = self.client.get(reverse("topic-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "1 my source")
        self.assertContains(response, "1 research source")
        topic_row = response.content.decode("utf-8").split(topic.name, 1)[1].split("Delete topic", 1)[0]
        self.assertIn(">Run</button>", topic_row)
        self.assertNotIn("Needs a research source", topic_row)
        self.assertNotIn("disabled", topic_row)
        self.assertIn('class="topic-run-hint" aria-hidden="true"></span>', topic_row)

    def test_topic_list_research_only_without_research_sources_disables_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.DISCOVERY_ONLY)

        response = self.client.get(reverse("topic-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "research only")
        self.assertContains(response, "0 research sources")
        self.assertContains(response, "Needs a research source")

    def test_topic_list_my_sources_only_with_manual_source_enables_run(self) -> None:
        topic = self._create_topic(source_mode=TopicSourceMode.CURATED_ONLY)
        self._add_manual_source(topic, is_active=True)

        response = self.client.get(reverse("topic-list"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "my sources only")
        self.assertContains(response, "1 my source")
        topic_row = response.content.decode("utf-8").split(topic.name, 1)[1].split("Delete topic", 1)[0]
        self.assertIn(">Run</button>", topic_row)
        self.assertNotIn("Needs a my source", topic_row)
        self.assertNotIn("disabled", topic_row)
