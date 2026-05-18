from unittest.mock import patch

from html import unescape
import re

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.digests.forms import TOPIC_NAME_REQUIRED_MESSAGE, TopicInputForm
from apps.digests import result_messages
from apps.digests.models import DigestRun
from apps.digests.views import _build_curated_source_seeds
from apps.sources.models import Article
from apps.topics.focus import (
    FOCUS_DUPLICATE_MESSAGE,
    FOCUS_NUMBER_ONLY_MESSAGE,
    FOCUS_TOO_SHORT_MESSAGE,
    FOCUS_VALIDATION_MESSAGE,
    is_meaningful_focus_term,
)
from apps.topics.focus_suggestions import generate_focus_suggestions
from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin

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
    def _assert_any_term_contains(self, terms: list[str], *needles: str) -> None:
        self.assertTrue(
            any(any(needle in term.casefold() for needle in needles) for term in terms),
            f"Expected one of {needles!r} in suggestions: {terms!r}",
        )

    def _assert_not_topic_echo(self, topic_name: str, terms: list[str]) -> None:
        normalized_topic = topic_name.casefold().strip()
        self.assertTrue(any(term.casefold() != normalized_topic for term in terms), terms)

    def test_topic_can_store_source_url(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI Ops",
            source_url="https://example.com/feed.xml",
            keywords=["AI Ops"],
            excluded_keywords=[],
        )

        self.assertEqual(topic.source_url, "https://example.com/feed.xml")

    def test_topic_form_accepts_single_source_url_for_topic_source_management(self) -> None:
        form = TopicInputForm(
            data={
                "topic_name": "AI Ops",
                "source_url": "https://example.com/feed.xml",
                "source_mode": TopicSourceMode.HYBRID,
            }
        )

        self.assertTrue(form.is_valid())
        self.assertEqual(form.cleaned_data["source_url"], "https://example.com/feed.xml")
        self.assertEqual(form.cleaned_data["source_mode"], TopicSourceMode.HYBRID)

    def test_topic_form_rejects_invalid_source_url(self) -> None:
        form = TopicInputForm(
            data={
                "topic_name": "AI Ops",
                "source_url": "not-a-url",
                "source_mode": TopicSourceMode.HYBRID,
            }
        )

        self.assertFalse(form.is_valid())
        self.assertIn("source_url", form.errors)

    def test_topic_form_required_message_is_english(self) -> None:
        form = TopicInputForm(data={"topic_name": "", "source_url": "", "source_mode": TopicSourceMode.HYBRID})

        self.assertFalse(form.is_valid())
        self.assertEqual(
            form.errors["topic_name"],
            [TOPIC_NAME_REQUIRED_MESSAGE],
        )

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
                "source_mode": TopicSourceMode.HYBRID,
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
        self.assertEqual(run.result_message, result_messages.SOURCE_NO_USABLE_ARTICLES)
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

        self.assertContains(response, "What do you want to explore?")
        self.assertNotContains(response, "Where to look")
        self.assertNotContains(response, "Source mode")
        self.assertNotContains(response, "Choose how this topic should find sources.")
        self.assertContains(response, 'value="hybrid"', html=False)
        self.assertContains(response, "Saved topics")
        self.assertContains(response, "Recent digests")
        self.assertContains(response, "Open Django admin")
        self.assertContains(response, "Find sources")
        self.assertContains(response, "Settings")
        self.assertContains(response, 'aria-label="Delete topic"', html=False)
        self.assertContains(response, "⋮⋮")
        self.assertNotContains(response, "Review sources")
        self.assertContains(response, "0 saved")
        self.assertNotContains(response, "Legacy source URL saved")
        self.assertContains(response, "Delete this topic?")
        self.assertNotContains(response, ">Delete<", html=False)
        self.assertContains(response, 'class="drag-handle"', html=False)
        self.assertContains(response, 'draggable="true"', html=False)

    def test_dashboard_recent_digests_show_human_readable_time_without_run_metadata(self) -> None:
        topic = Topic.objects.create(name="AI agents", source_mode=TopicSourceMode.HYBRID, user=self._get_ui_user())
        run = DigestRun.objects.create(topic=topic, source_mode=topic.source_mode, status=DigestRun.STATUS_COMPLETED)
        DigestRun.objects.filter(pk=run.pk).update(created_at=timezone.now() - timedelta(days=1))

        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, "Recent digests")
        self.assertContains(response, "AI agents")
        self.assertContains(response, "Yesterday")
        self.assertNotContains(response, f"Digest {run.id}")
        self.assertNotContains(response, DigestRun.STATUS_COMPLETED)

    def test_topic_workspace_renders_focus_chip_editor(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI Education",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation", "workflow automation"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Focus")
        self.assertContains(response, "AI automation")
        self.assertContains(response, "workflow automation")
        self.assertContains(response, 'data-focus-form', html=False)
        self.assertContains(response, 'data-focus-input', html=False)
        self.assertContains(response, 'data-focus-chip-list', html=False)

    def test_new_topic_gets_generated_focus_suggestions(self) -> None:
        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "AI Education",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic = Topic.objects.get(name="AI Education")
        topic.refresh_from_db()
        self.assertTrue(topic.focus_initialized)
        self.assertGreaterEqual(len(topic.keywords), 3)
        self.assertNotEqual(topic.keywords, ["AI Education"])
        self._assert_any_term_contains(topic.keywords, "ai", "llm")
        self._assert_any_term_contains(
            topic.keywords,
            "education",
            "learning",
            "classroom",
            "student",
            "teaching",
            "tutor",
        )
        self.assertTrue(any(" " in term for term in topic.keywords), topic.keywords)
        response_html = response.content.decode("utf-8")
        self.assertTrue(any(term in response_html for term in topic.keywords))

    def test_generated_focus_terms_pass_validation(self) -> None:
        self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "AI Education",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )
        topic = Topic.objects.get(name="AI Education")

        for term in topic.keywords:
            self.assertTrue(is_meaningful_focus_term(term), term)

    def test_baby_sleeping_gets_parenting_grounded_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("Baby sleeping")

        self.assertGreaterEqual(len(suggestions), 3)
        self.assertTrue(any("sleep" in term.casefold() for term in suggestions))
        self.assertTrue(any("baby" in term.casefold() or "infant" in term.casefold() or "newborn" in term.casefold() for term in suggestions))
        self.assertNotIn("industry tools", suggestions)
        self.assertNotIn("implementation patterns", suggestions)
        self.assertNotIn("practical workflows", suggestions)

    def test_travel_gets_more_than_topic_echo_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("travel")
        lowered = [term.casefold() for term in suggestions]

        self.assertGreater(len(suggestions), 1)
        self._assert_not_topic_echo("travel", suggestions)
        self.assertTrue(
            any(
                keyword in term
                for term in lowered
                for keyword in ("travel", "traveler", "tourism", "vacation", "trip")
            ),
            suggestions,
        )
        self.assertTrue(
            any(
                keyword in term
                for term in lowered
                for keyword in (
                    "family",
                    "cuisine",
                    "culture",
                    "cultural",
                    "safety",
                    "eco",
                    "hidden gems",
                    "destination",
                    "spot",
                    "local",
                )
            ),
            suggestions,
        )
        self.assertTrue(any(" " in term.strip() for term in suggestions), suggestions)
        self.assertGreaterEqual(len({term.casefold() for term in suggestions}), 3)
        self.assertNotIn("industry tools", suggestions)
        self.assertNotIn("implementation patterns", suggestions)
        self.assertNotIn("practical workflows", suggestions)

    def test_ai_agents_gets_grounded_technical_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("AI agents")

        self._assert_any_term_contains(suggestions, "agent")
        self._assert_any_term_contains(suggestions, "ai", "llm", "autonomous", "multi-agent", "automation")
        self.assertTrue(any(" " in term for term in suggestions), suggestions)
        self.assertGreaterEqual(len({term.casefold() for term in suggestions}), 3)
        self.assertNotIn("industry tools", suggestions)
        self.assertNotIn("implementation patterns", suggestions)
        self.assertNotIn("practical workflows", suggestions)

    def test_baby_sleeping_workspace_does_not_render_technical_focus_defaults(self) -> None:
        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Baby sleeping",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic = Topic.objects.get(name="Baby sleeping")
        topic.refresh_from_db()
        self.assertTrue(any("sleep" in term.casefold() for term in topic.keywords))
        self.assertNotContains(response, "industry tools")
        self.assertNotContains(response, "implementation patterns")
        self.assertNotContains(response, "practical workflows")
        self.assertNotContains(response, "DEV Community / #python")
        self.assertNotContains(response, "DEV Community / #devops")

    def test_existing_manual_focus_is_not_overwritten_by_generated_focus(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="travel",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["family travel", "solo travel"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["family travel", "solo travel"])
        self.assertContains(response, "family travel")
        self.assertContains(response, "solo travel")
        self.assertNotContains(response, "travel planning")

    def test_removed_focus_suggestions_do_not_reappear_automatically(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI Education",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI Education"],
            focus_initialized=False,
            excluded_keywords=[],
        )

        first_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
            },
        )

        self.assertEqual(first_response.status_code, 200)
        topic.refresh_from_db()
        initial_generated_terms = list(topic.keywords)
        self.assertTrue(initial_generated_terms)

        remove_response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": ""},
        )

        self.assertEqual(remove_response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, [])
        self.assertTrue(topic.focus_initialized)

        reopen_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
            },
        )

        self.assertEqual(reopen_response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, [])
        reopen_html = reopen_response.content.decode("utf-8")
        for term in initial_generated_terms:
            self.assertNotIn(term, reopen_html)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_generated_focus_terms_are_passed_to_discovery(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = []

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "AI Education",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        discovery_request = mock_resolve_source_candidates.call_args.args[0]
        self.assertGreaterEqual(len(discovery_request.focus_terms), 3)
        self._assert_any_term_contains(discovery_request.focus_terms, "ai", "llm")
        self._assert_any_term_contains(
            discovery_request.focus_terms,
            "education",
            "learning",
            "classroom",
            "student",
            "teaching",
            "tutor",
        )

    def test_topic_focus_terms_can_be_added_and_removed(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Focus topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation"],
            excluded_keywords=[],
        )

        add_response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "AI automation\nsmall business AI\nworkflow automation\nAI automation"},
        )

        self.assertEqual(add_response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["AI automation", "small business AI", "workflow automation"])
        self.assertContains(add_response, "small business AI")
        self.assertContains(add_response, "workflow automation")

        remove_response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "small business AI"},
        )

        self.assertEqual(remove_response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["small business AI"])
        self.assertContains(remove_response, "small business AI")
        self.assertNotContains(remove_response, "workflow automation")

    def test_focus_rejects_gibberish_term_and_does_not_persist_it(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Focus validation topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "AI automation\npumpumpum"},
        )

        self.assertEqual(response.status_code, 400)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["AI automation"])
        self.assertContains(
            response,
            FOCUS_TOO_SHORT_MESSAGE,
            status_code=400,
        )
        self.assertNotContains(response, 'data-focus-value="pumpumpum"', html=False, status_code=400)

    def test_focus_rejects_high_confidence_junk_examples(self) -> None:
        failing_terms = ["asdfasdf", "qwerty", "767ghjb;k", "pumpumpum", "фывафыва", "олололо"]
        user = self._get_ui_user()
        for term in failing_terms:
            topic = Topic.objects.create(
                user=user,
                name=f"High confidence junk {term}",
                source_mode=TopicSourceMode.HYBRID,
                keywords=["AI automation"],
                excluded_keywords=[],
            )

            response = self.client.post(
                reverse("update-topic-focus", args=[topic.id]),
                data={"focus_terms": f"AI automation\n{term}"},
            )

            self.assertEqual(response.status_code, 400)
            topic.refresh_from_db()
            self.assertEqual(topic.keywords, ["AI automation"])
            self.assertContains(response, FOCUS_TOO_SHORT_MESSAGE, status_code=400)
            self.assertNotContains(response, f'data-focus-value="{term}"', html=False, status_code=400)

    def test_focus_rejects_short_cyrillic_fragments(self) -> None:
        failing_terms = ["фш", "ао", "лпр"]
        user = self._get_ui_user()
        for term in failing_terms:
            topic = Topic.objects.create(
                user=user,
                name=f"Short Cyrillic junk {term}",
                source_mode=TopicSourceMode.HYBRID,
                keywords=["AI automation"],
                excluded_keywords=[],
            )

            response = self.client.post(
                reverse("update-topic-focus", args=[topic.id]),
                data={"focus_terms": f"AI automation\n{term}"},
            )

            self.assertEqual(response.status_code, 400)
            topic.refresh_from_db()
            self.assertEqual(topic.keywords, ["AI automation"])
            self.assertContains(response, FOCUS_TOO_SHORT_MESSAGE, status_code=400)
            self.assertNotContains(response, f'data-focus-value="{term}"', html=False, status_code=400)

    def test_focus_rejects_repeated_pattern_term(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Repeated pattern topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "asdfasdf"},
        )

        self.assertEqual(response.status_code, 400)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, [])
        self.assertContains(
            response,
            FOCUS_TOO_SHORT_MESSAGE,
            status_code=400,
        )
        self.assertNotContains(response, 'data-focus-value="asdfasdf"', html=False, status_code=400)

    def test_focus_accepts_meaningful_abbreviation(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Abbreviation topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "AI"},
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["AI"])
        self.assertContains(response, 'data-focus-value="AI"', html=False)

    def test_focus_accepts_normal_single_word_terms(self) -> None:
        accepted_terms = ["workflow", "automation", "observability", "vectorization", "LangChain", "MCP"]
        user = self._get_ui_user()
        for term in accepted_terms:
            topic = Topic.objects.create(
                user=user,
                name=f"Accepted focus {term}",
                source_mode=TopicSourceMode.HYBRID,
                keywords=[],
                excluded_keywords=[],
            )

            response = self.client.post(
                reverse("update-topic-focus", args=[topic.id]),
                data={"focus_terms": term},
            )

            self.assertEqual(response.status_code, 200)
            topic.refresh_from_db()
            self.assertEqual(topic.keywords, [term])
            self.assertContains(response, f'data-focus-value="{term}"', html=False)

    def test_focus_accepts_short_latin_abbreviations(self) -> None:
        accepted_terms = ["AI", "API", "CRM", "MCP"]
        user = self._get_ui_user()
        for term in accepted_terms:
            topic = Topic.objects.create(
                user=user,
                name=f"Short abbreviation {term}",
                source_mode=TopicSourceMode.HYBRID,
                keywords=[],
                excluded_keywords=[],
            )

            response = self.client.post(
                reverse("update-topic-focus", args=[topic.id]),
                data={"focus_terms": term},
            )

            self.assertEqual(response.status_code, 200)
            topic.refresh_from_db()
            self.assertEqual(topic.keywords, [term])
            self.assertContains(response, f'data-focus-value="{term}"', html=False)

    def test_focus_accepts_meaningful_phrase(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Phrase topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "workflow automation"},
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["workflow automation"])
        self.assertContains(response, 'data-focus-value="workflow automation"', html=False)

    def test_focus_accepts_contextual_numeric_hint(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Physical exercises for pregnant women",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "8 month"},
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["8 month"])
        self.assertContains(response, 'data-focus-value="8 month"', html=False)
        self.assertNotContains(response, f"<strong>{FOCUS_VALIDATION_MESSAGE}</strong>", html=False)

    def test_focus_accepts_short_contextual_phrases(self) -> None:
        accepted_terms = ["8 months", "third trimester", "low impact", "safe", "beginner", "no equipment"]
        user = self._get_ui_user()
        for term in accepted_terms:
            topic = Topic.objects.create(
                user=user,
                name=f"Contextual focus {term}",
                source_mode=TopicSourceMode.HYBRID,
                keywords=[],
                excluded_keywords=[],
            )

            response = self.client.post(
                reverse("update-topic-focus", args=[topic.id]),
                data={"focus_terms": term},
            )

            self.assertEqual(response.status_code, 200)
            topic.refresh_from_db()
            self.assertEqual(topic.keywords, [term])
            self.assertContains(response, f'data-focus-value="{term}"', html=False)
            self.assertNotContains(response, f"<strong>{FOCUS_VALIDATION_MESSAGE}</strong>", html=False)

    def test_focus_rejects_number_only_input_with_specific_message(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Numeric focus topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["third trimester"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "third trimester\n567"},
        )

        self.assertEqual(response.status_code, 400)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["third trimester"])
        self.assertContains(response, FOCUS_NUMBER_ONLY_MESSAGE, status_code=400)
        self.assertContains(response, 'value="567"', html=False, status_code=400)
        self.assertNotContains(response, 'data-focus-value="567"', html=False, status_code=400)

    def test_focus_error_renders_between_input_and_chips(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Focus message placement",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["third trimester"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "third trimester\n567"},
        )

        self.assertEqual(response.status_code, 400)
        html = response.content.decode("utf-8")
        input_index = html.index("data-focus-input")
        message_index = html.index(FOCUS_NUMBER_ONLY_MESSAGE)
        chips_index = html.index("data-focus-chip-list")
        self.assertLess(input_index, message_index)
        self.assertLess(message_index, chips_index)

    def test_focus_workspace_js_includes_duplicate_message(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Duplicate focus workspace",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["third trimester"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertContains(response, 'data-focus-feedback', html=False)
        self.assertContains(response, FOCUS_DUPLICATE_MESSAGE)

    def test_focus_duplicate_terms_are_deduped_after_normalization(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Deduped focus topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("update-topic-focus", args=[topic.id]),
            data={"focus_terms": "AI agents\nai agents\n  AI   agents  "},
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.keywords, ["AI agents"])
        self.assertEqual(response.content.decode("utf-8").count('data-focus-value="AI agents"'), 1)

    def test_saved_topics_dashboard_does_not_render_focus_terms(self) -> None:
        Topic.objects.create(
            user=self._get_ui_user(),
            name="Compact dashboard",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation", "workflow automation"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, "Compact dashboard")
        self.assertNotContains(response, "AI automation")
        self.assertNotContains(response, "workflow automation")

    def test_topic_list_form_disables_browser_native_validation(self) -> None:
        response = self.client.get(reverse("topic-list"))
        html = response.content.decode("utf-8")

        self.assertIn('action="/discover-sources/" novalidate', html)

    def test_topic_dashboard_delete_action_removes_topic(self) -> None:
        user = self._get_ui_user()
        keep_topic = Topic.objects.create(
            user=user,
            name="Keep me",
            keywords=["Keep me"],
            excluded_keywords=[],
        )
        delete_topic = Topic.objects.create(
            user=user,
            name="Delete me",
            keywords=["Delete me"],
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=delete_topic,
            name="Saved source",
            url="https://example.com/feed.xml",
            normalized_url="https://example.com/feed.xml",
            source_type="rss_feed",
            origin="manual",
            is_active=True,
        )

        response = self.client.post(reverse("delete-topic", args=[delete_topic.id]))

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("topic-list"))
        self.assertTrue(Topic.objects.filter(id=keep_topic.id).exists())
        self.assertFalse(Topic.objects.filter(id=delete_topic.id).exists())
        self.assertFalse(TopicSource.objects.filter(topic_id=delete_topic.id).exists())

        dashboard_response = self.client.get(reverse("topic-list"))
        self.assertContains(dashboard_response, "Keep me")
        self.assertNotContains(dashboard_response, "Delete me")

    def test_saved_topics_dashboard_renders_topics_in_saved_order(self) -> None:
        user = self._get_ui_user()
        first = Topic.objects.create(
            user=user,
            name="First topic",
            display_order=2,
            keywords=["First topic"],
            excluded_keywords=[],
        )
        second = Topic.objects.create(
            user=user,
            name="Second topic",
            display_order=1,
            keywords=["Second topic"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        html = response.content.decode("utf-8")
        self.assertLess(html.index("Second topic"), html.index("First topic"))
        self.assertContains(response, f'data-topic-id="{second.id}"', html=False)
        self.assertContains(response, f'data-topic-id="{first.id}"', html=False)

    def test_dashboard_sections_render_section_level_collapse_controls(self) -> None:
        user = self._get_ui_user()
        topic = Topic.objects.create(
            user=user,
            name="Collapsible topic",
            keywords=["Collapsible topic"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, 'id="saved-topics-section-body"', html=False)
        self.assertContains(response, 'data-collapse-key="saved-topics"', html=False)
        self.assertContains(response, 'aria-controls="saved-topics-section-body"', html=False)
        self.assertContains(response, 'id="recent-runs-section-body"', html=False)
        self.assertContains(response, 'data-collapse-key="recent-runs"', html=False)
        self.assertContains(response, 'aria-controls="recent-runs-section-body"', html=False)
        self.assertContains(response, 'aria-expanded="true"', html=False)
        self.assertContains(response, 'data-collapse-button', html=False)
        self.assertNotContains(response, f'aria-controls="topic-details-{topic.id}"', html=False)
        self.assertNotContains(response, f'id="topic-details-{topic.id}"', html=False)

    def test_newly_created_topic_appears_first_on_dashboard(self) -> None:
        user = self._get_ui_user()
        older = Topic.objects.create(
            user=user,
            name="Older topic",
            keywords=["Older topic"],
            excluded_keywords=[],
        )
        oldest = Topic.objects.create(
            user=user,
            name="Oldest topic",
            keywords=["Oldest topic"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Newest topic",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        newest = Topic.objects.get(user=user, name="Newest topic")
        older.refresh_from_db()
        oldest.refresh_from_db()
        self.assertEqual((newest.display_order, older.display_order, oldest.display_order), (1, 2, 3))

        dashboard_response = self.client.get(reverse("topic-list"))
        html = dashboard_response.content.decode("utf-8")
        self.assertLess(html.index("Newest topic"), html.index("Older topic"))
        self.assertLess(html.index("Older topic"), html.index("Oldest topic"))

    def test_saved_topics_dashboard_only_renders_ui_user_topics(self) -> None:
        user = self._get_ui_user()
        other_user_model = Topic._meta.get_field("user").remote_field.model
        other_user = other_user_model.objects.create_user(username="other-tester")
        Topic.objects.create(
            user=user,
            name="Visible topic",
            keywords=["Visible topic"],
            excluded_keywords=[],
        )
        Topic.objects.create(
            user=other_user,
            name="Hidden topic",
            keywords=["Hidden topic"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, "Visible topic")
        self.assertNotContains(response, "Hidden topic")

    def test_reorder_topics_persists_dashboard_order(self) -> None:
        user = self._get_ui_user()
        first = Topic.objects.create(
            user=user,
            name="Alpha",
            keywords=["Alpha"],
            excluded_keywords=[],
        )
        second = Topic.objects.create(
            user=user,
            name="Beta",
            keywords=["Beta"],
            excluded_keywords=[],
        )
        third = Topic.objects.create(
            user=user,
            name="Gamma",
            keywords=["Gamma"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("reorder-topics"),
            data={"topic_ids": [third.id, first.id, second.id]},
        )

        self.assertEqual(response.status_code, 200)
        self.assertJSONEqual(response.content.decode("utf-8"), {"ok": True})
        first.refresh_from_db()
        second.refresh_from_db()
        third.refresh_from_db()
        self.assertEqual((third.display_order, first.display_order, second.display_order), (1, 2, 3))

        dashboard_response = self.client.get(reverse("topic-list"))
        html = dashboard_response.content.decode("utf-8")
        self.assertLess(html.index("Gamma"), html.index("Alpha"))
        self.assertLess(html.index("Alpha"), html.index("Beta"))

    def test_reorder_topics_rejects_missing_or_foreign_topic_ids_without_changing_order(self) -> None:
        user = self._get_ui_user()
        other_user_model = Topic._meta.get_field("user").remote_field.model
        other_user = other_user_model.objects.create_user(username="outsider")
        first = Topic.objects.create(
            user=user,
            name="Alpha",
            display_order=1,
            keywords=["Alpha"],
            excluded_keywords=[],
        )
        second = Topic.objects.create(
            user=user,
            name="Beta",
            display_order=2,
            keywords=["Beta"],
            excluded_keywords=[],
        )
        foreign = Topic.objects.create(
            user=other_user,
            name="Foreign",
            display_order=1,
            keywords=["Foreign"],
            excluded_keywords=[],
        )

        response = self.client.post(
            reverse("reorder-topics"),
            data={"topic_ids": [second.id, foreign.id]},
        )

        self.assertEqual(response.status_code, 400)
        first.refresh_from_db()
        second.refresh_from_db()
        self.assertEqual((first.display_order, second.display_order), (1, 2))

    def test_deleting_topic_preserves_remaining_custom_order(self) -> None:
        user = self._get_ui_user()
        first = Topic.objects.create(
            user=user,
            name="Alpha",
            display_order=2,
            keywords=["Alpha"],
            excluded_keywords=[],
        )
        second = Topic.objects.create(
            user=user,
            name="Beta",
            display_order=1,
            keywords=["Beta"],
            excluded_keywords=[],
        )
        third = Topic.objects.create(
            user=user,
            name="Gamma",
            display_order=3,
            keywords=["Gamma"],
            excluded_keywords=[],
        )

        response = self.client.post(reverse("delete-topic", args=[first.id]))

        self.assertEqual(response.status_code, 302)
        dashboard_response = self.client.get(reverse("topic-list"))
        html = dashboard_response.content.decode("utf-8")
        self.assertLess(html.index("Beta"), html.index("Gamma"))
        self.assertNotContains(dashboard_response, "Alpha")

    def test_topic_list_empty_state_and_validation_copy_are_english(self) -> None:
        response = self.client.get(reverse("topic-list"))

        self.assertContains(response, "No topics yet. Enter a topic above to find sources and start a digest.")
        self.assertContains(response, "No runs yet.")
        self.assertNotContains(response, "РќСѓР¶РЅРѕ СѓРєР°Р·Р°С‚СЊ С‚РµРјСѓ")
        self.assertNotContains(response, "Р’РІРµРґРёС‚Рµ С‚РµРјСѓ")

    def test_create_topic_validation_error_is_english(self) -> None:
        response = self.client.post(
            reverse("create-topic-and-run"),
            data={"topic_name": "", "source_url": "", "source_mode": TopicSourceMode.HYBRID},
        )
        html = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, TOPIC_NAME_REQUIRED_MESSAGE, status_code=400)
        self.assertEqual(html.count(TOPIC_NAME_REQUIRED_MESSAGE), 1)
        self.assertNotContains(response, "РќСѓР¶РЅРѕ СѓРєР°Р·Р°С‚СЊ С‚РµРјСѓ", status_code=400)
        self.assertNotContains(response, "Р’РІРµРґРёС‚Рµ С‚РµРјСѓ", status_code=400)

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
                "topic_name": "Workflow operations",
                "source_url": "https://dev.to/feed/example",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 302)
        topic = Topic.objects.get(name="Workflow operations")
        run = DigestRun.objects.get(topic=topic)
        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.error_message, "")
        self.assertTrue(hasattr(run, "digest"))
        self.assertTrue(hasattr(run.digest, "content_package"))
        self.assertGreater(Article.objects.filter(topic=topic).count(), 0)
        self.assertEqual(topic.source_mode, TopicSourceMode.HYBRID)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_discover_sources_view_renders_candidate_sources_for_topic(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "detection_reason": "matched dev.to topic pattern",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "quality_estimate": "high",
                "default_selected": True,
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/python",
                "title": "DEV Community / #python",
                "description": "Python implementation posts.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "detection_reason": "matched dev.to topic pattern",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "quality_estimate": "medium",
                "default_selected": False,
                "candidate_origin": "curated",
            },
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "AI agents",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Back to topics")
        self.assertContains(response, "AI agents")
        self.assertNotContains(response, '<label for="id_topic_name">Topic</label>', html=False)
        self.assertNotContains(response, '<label for="id_source_mode">Where to look</label>', html=False)
        self.assertNotContains(response, "Topic name")
        self.assertNotContains(response, "Source mode")
        self.assertNotContains(response, "Choose how this topic should find sources.")
        self.assertContains(response, "Saved sources")
        self.assertContains(response, "New sources")
        self.assertContains(response, "Find sources")
        self.assertContains(response, "Use saved and new sources")
        self.assertContains(response, "Use saved sources only")
        self.assertContains(response, "Use new sources only")
        self.assertNotContains(response, ">Save<", html=False)
        self.assertNotContains(response, "Refresh source discovery")
        self.assertNotContains(response, "Saved topics")
        self.assertNotContains(response, "Recent digests")
        self.assertNotContains(response, "Topic settings")
        self.assertContains(response, "0 saved")
        self.assertContains(response, "1 new")
        self.assertContains(response, "Add a link and press Enter")
        self.assertNotContains(response, "Add source")
        self.assertNotContains(response, "Р’РІРµРґРёС‚Рµ URL")
        self.assertContains(response, "DEV Community / #ai")
        self.assertContains(response, "12 recent articles")
        self.assertNotContains(response, "Deduped")
        self.assertNotContains(response, "Temporary review set")
        self.assertNotContains(response, "Selected sources are saved to this topic and used for the run")
        self.assertNotContains(response, "Detection:")
        self.assertNotContains(response, "matched dev.to topic pattern")
        self.assertNotContains(response, "devto_tag")
        self.assertNotContains(response, "Sources saved for this topic.")
        self.assertNotContains(response, "Add a source")
        self.assertNotContains(response, "Pipeline actions")
        self.assertNotContains(response, "Use the selected sources to generate the next digest.")
        self.assertNotContains(response, "Discover new sources")
        self.assertNotContains(response, "Refresh suggestions")
        self.assertNotContains(response, "suggestions")
        self.assertNotContains(response, "Hybrid")
        self.assertNotContains(response, "Run pipeline")
        self.assertContains(response, "Run digest")
        topic = Topic.objects.get(name="AI agents")
        self.assertEqual(topic.sources.count(), 1)
        discovered_source = topic.sources.get()
        self.assertEqual(discovered_source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertTrue(discovered_source.is_active)
        self.assertEqual(topic.source_mode, TopicSourceMode.HYBRID)
        self.assertNotContains(response, "Legacy source")
        self.assertNotContains(response, "Curated")
        self.assertNotContains(response, "Valid")
        self.assertNotContains(response, "inactive")
        self.assertNotContains(response, "devto_tag")

        html = response.content.decode("utf-8")
        self.assertIn("saved &amp; new", html)
        self.assertNotIn("Saved + New", html)
        self.assertNotIn("Saved + new", html)
        self.assertIn('<h1 class="page-title">AI agents</h1>', html)
        self.assertNotIn("<h1>DigestFlow</h1>", html)
        self.assertNotIn('<h2 class="workflow-title">AI agents</h2>', html)
        self.assertLess(html.index('<h1 class="page-title">AI agents</h1>'), html.index("Saved sources"))
        self.assertLess(html.index("New sources"), html.index("Find sources"))
        self.assertLess(html.index("New sources"), html.index("Run digest"))
        self.assertIn("1 selected source will be used in the next digest run.", html)
        self.assertIn('name="topic_id" value="', html)
        self.assertIn('onchange="this.form.requestSubmit();"', html)
        self.assertEqual(html.count('class="primary-cta"'), 1)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_workspace_topic_configuration_autosaves_without_explicit_save_button(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI agents",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": "AI agent systems",
                "source_url": "",
                "source_mode": TopicSourceMode.CURATED_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.name, "AI agent systems")
        self.assertEqual(topic.source_mode, TopicSourceMode.CURATED_ONLY)
        self.assertEqual(
            Topic.objects.filter(user=topic.user, name="AI agent systems").count(),
            1,
        )
        self.assertContains(response, "AI agent systems")
        self.assertContains(response, "saved only")
        self.assertContains(response, "Use saved sources only")
        self.assertNotContains(response, ">Save<", html=False)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_curated_only_workspace_hides_discovery_section(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            }
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Curated AI",
                "source_url": "",
                "source_mode": TopicSourceMode.CURATED_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Curated AI")
        self.assertContains(response, "saved only")
        self.assertContains(response, "Saved sources")
        self.assertContains(response, "Run digest")
        self.assertContains(response, "Please select at least one source to run a new digest.")
        self.assertNotContains(response, "New sources")
        self.assertNotContains(response, "Find sources")
        self.assertNotContains(response, "No new sources yet.")
        self.assertNotContains(response, "DEV Community / #ai")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count('class="primary-cta"'), 1)
        self.assertIn('class="primary-cta" disabled', html)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_discovery_only_workspace_hides_saved_sources_section(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            }
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Discovery AI",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Discovery AI")
        self.assertContains(response, "new only")
        self.assertContains(response, "New sources")
        self.assertContains(response, "Find sources")
        self.assertContains(response, "Find additional sources for this topic.")
        self.assertContains(response, "DEV Community / #ai")
        self.assertContains(response, "Run digest")
        self.assertContains(response, "1 selected source will be used in the next digest run.")
        self.assertNotContains(response, "No new sources yet.")
        self.assertNotContains(response, "Saved sources")
        self.assertNotContains(response, "Add a link and press Enter")
        self.assertNotContains(response, "Add source")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count('class="primary-cta"'), 1)
        self.assertNotIn('class="primary-cta" disabled', html)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_empty_discovery_workspace_still_renders_run_digest_card_disabled(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = []

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Empty discovery topic",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Empty discovery topic")
        self.assertContains(response, "New sources")
        self.assertContains(response, "No new sources yet.")
        self.assertContains(response, "Find")
        self.assertNotContains(response, "Find additional sources for this topic.")
        self.assertNotContains(response, "No new sources were found for this topic yet.")
        self.assertContains(response, "Ready to generate")
        self.assertContains(response, "Please select at least one source to run a new digest.")
        self.assertContains(response, "Run digest")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count('class="primary-cta"'), 1)
        self.assertIn('class="primary-cta"', html)
        self.assertIn('data-run-source-count-button', html)
        self.assertIn('disabled', html)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_run_digest_card_uses_plural_selected_source_helper_text(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/python",
                "title": "DEV Community / #python",
                "description": "Broad Python engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            },
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Plural source count",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "2 selected sources will be used in the next digest run.")
        self.assertNotContains(response, 'class="primary-cta" disabled', html=False)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_saved_source_does_not_render_inside_new_sources_section(self, mock_resolve_source_candidates) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Separated sources",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Saved source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="manual",
            platform="dev.to",
            is_active=True,
        )
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "Saved source",
                "description": "Already saved on the topic.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/python",
                "title": "New source",
                "description": "Fresh discovery result.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "default_selected": False,
                "candidate_origin": "discovered",
            },
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Separated sources",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Saved sources")
        self.assertContains(response, "New sources")
        self.assertContains(response, "Saved source")
        self.assertContains(response, "New source")
        self.assertContains(response, "Fresh discovery result.")
        self.assertNotContains(response, "Already saved on the topic.")
        self.assertContains(response, 'type="hidden" name="selected_source_urls" value="https://dev.to/t/ai"', html=False)

        html = response.content.decode("utf-8")
        self.assertIn("https://dev.to/t/ai", html)

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_selected_sources_drive_digest_generation(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI agents",
            keywords=["AI agents"],
            excluded_keywords=[],
            source_mode=TopicSourceMode.CURATED_ONLY,
        )
        source_one = TopicSource.objects.create(
            topic=topic,
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            is_active=True,
        )
        source_two = TopicSource.objects.create(
            topic=topic,
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            is_active=True,
        )
        source_three = TopicSource.objects.create(
            topic=topic,
            url="https://dev.to/t/security",
            normalized_url="https://dev.to/api/articles?tag=security",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            is_active=True,
        )
        mock_fetch_rss_articles.side_effect = [
            [
                {
                    "title": "AI article",
                    "url": "https://example.com/ai-1",
                    "source_name": "DEV Community",
                    "snippet": "AI article snippet",
                }
            ],
            [
                {
                    "title": "Security article",
                    "url": "https://example.com/sec-1",
                    "source_name": "DEV Community",
                    "snippet": "Security article snippet",
                }
            ],
        ]

        response = self.client.post(
            reverse("run-with-selected-sources", args=[topic.id]),
            data={"selected_source_urls": [source_one.url, source_three.url]},
        )

        self.assertEqual(response.status_code, 302)
        topic.refresh_from_db()
        self.assertEqual(topic.source_mode, TopicSourceMode.CURATED_ONLY)
        source_one.refresh_from_db()
        source_two.refresh_from_db()
        source_three.refresh_from_db()
        self.assertTrue(source_one.is_active)
        self.assertTrue(source_two.is_active)
        self.assertTrue(source_three.is_active)
        self.assertEqual(mock_fetch_rss_articles.call_count, 2)
        mock_fetch_rss_articles.assert_any_call("https://dev.to/api/articles?tag=ai")
        mock_fetch_rss_articles.assert_any_call("https://dev.to/api/articles?tag=security")
        run = DigestRun.objects.get(topic=topic)
        self.assertEqual(
            run.input_snapshot.get("selected_source_urls"),
            ["https://dev.to/t/ai", "https://dev.to/t/security"],
        )
        mock_run_digest_pipeline.assert_called_once_with(
            run.id,
            raw_items=[
                {
                    "title": "AI article",
                    "url": "https://example.com/ai-1",
                    "source_name": "DEV Community",
                    "snippet": "AI article snippet",
                },
                {
                    "title": "Security article",
                    "url": "https://example.com/sec-1",
                    "source_name": "DEV Community",
                    "snippet": "Security article snippet",
                },
            ],
        )

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_selected_devto_source_run_does_not_fail_when_devto_items_use_numeric_ids(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI agents",
            keywords=["AI agents"],
            excluded_keywords=[],
            source_mode=TopicSourceMode.CURATED_ONLY,
        )
        source = TopicSource.objects.create(
            topic=topic,
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            is_active=True,
        )
        mock_fetch_rss_articles.return_value = [
            {
                "title": "Gateway article",
                "url": "https://dev.to/example/gateway-article",
                "source_name": "DEV Community",
                "snippet": "Gateway article snippet",
                "metadata": {"devto_id": 3630333},
            }
        ]

        response = self.client.post(
            reverse("run-with-selected-sources", args=[topic.id]),
            data={"selected_source_urls": [source.url]},
        )

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.get(topic=topic)
        self.assertEqual(run.input_snapshot.get("selected_source_urls"), ["https://dev.to/t/ai"])
        mock_fetch_rss_articles.assert_called_once_with("https://dev.to/api/articles?tag=ai")
        mock_run_digest_pipeline.assert_called_once()

    @patch("apps.digests.views.resolve_source_candidates")
    def test_selecting_no_sources_returns_clear_error(self, mock_resolve_source_candidates) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="MCP",
            keywords=["MCP"],
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            is_active=False,
        )
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "detection_reason": "matched dev.to topic pattern",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "quality_estimate": "high",
                "default_selected": False,
                "candidate_origin": "discovered",
            }
        ]

        response = self.client.post(reverse("run-with-selected-sources", args=[topic.id]), data={})

        self.assertEqual(response.status_code, 400)
        self.assertContains(response, "Select at least one source before generating the digest.", status_code=400)

    @patch("apps.digests.views.fetch_rss_articles")
    @patch("apps.digests.views.run_digest_pipeline")
    def test_manual_source_url_creates_persistent_topic_source(
        self,
        mock_run_digest_pipeline,
        mock_fetch_rss_articles,
    ) -> None:
        mock_fetch_rss_articles.return_value = [
            {
                "title": "Manual source article",
                "url": "https://example.com/article",
                "source_name": "Example Feed",
                "snippet": "Snippet",
            }
        ]

        response = self.client.post(
            reverse("create-topic-and-run"),
            data={
                "topic_name": "Persistent sources",
                "source_url": "https://example.com/feed.xml",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 302)
        topic = Topic.objects.get(name="Persistent sources")
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(source.url, "https://example.com/feed.xml")
        self.assertEqual(source.origin, "manual")
        self.assertTrue(source.is_active)

    @patch("apps.digests.views.fetch_rss_articles")
    @patch("apps.digests.views.run_digest_pipeline")
    def test_adding_one_manual_source_creates_persistent_topic_source(
        self,
        mock_run_digest_pipeline,
        mock_fetch_rss_articles,
    ) -> None:
        mock_fetch_rss_articles.return_value = [
            {
                "title": "Manual source article",
                "url": "https://example.com/article",
                "source_name": "Example Feed",
                "snippet": "Snippet",
            }
        ]

        response = self.client.post(
            reverse("create-topic-and-run"),
            data={
                "topic_name": "Multiple persistent sources",
                "source_url": "https://example.com/feed.xml",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 302)
        topic = Topic.objects.get(name="Multiple persistent sources")
        self.assertEqual(topic.sources.count(), 1)
        self.assertEqual(topic.sources.get().url, "https://example.com/feed.xml")

    @patch("apps.digests.views.fetch_rss_articles")
    def test_add_topic_source_persists_source_in_inventory(self, mock_fetch_rss_articles) -> None:
        mock_fetch_rss_articles.return_value = [
            {"title": "Feed item", "url": "https://example.com/articles/1", "snippet": "Snippet"}
        ]
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Deduped sources",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/feed.xml",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "Source added and saved for this topic. It will be used when generating the digest.",
        )
        self.assertContains(response, "https://example.com/feed.xml")

    def test_saved_sources_render_newest_source_first(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Newest first topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        older = TopicSource.objects.create(
            topic=topic,
            name="Older source",
            url="https://example.com/older",
            normalized_url="https://example.com/older",
            source_type="generic_html",
            origin="manual",
            is_active=True,
        )
        newer = TopicSource.objects.create(
            topic=topic,
            name="Newer source",
            url="https://example.com/newer",
            normalized_url="https://example.com/newer",
            source_type="generic_html",
            origin="manual",
            is_active=True,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        self.assertLess(html.index(newer.name), html.index(older.name))

    def test_add_topic_source_prevents_duplicate_normalized_url(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Duplicate source topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Saved source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="manual",
            platform="dev.to",
            is_active=True,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://dev.to/t/ai",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertContains(
            response,
            "This source has already been added to this topic. Please check the address or use another source.",
        )

    @patch("apps.digests.views.fetch_rss_articles")
    def test_add_topic_source_accepts_devto_author_profile_and_normalizes_duplicate_variants(
        self,
        mock_fetch_rss_articles,
    ) -> None:
        mock_fetch_rss_articles.return_value = [
            {"title": "Author post", "url": "https://dev.to/michael_rakutko/post", "snippet": "Snippet"}
        ]
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Author sources",
            source_mode=TopicSourceMode.HYBRID,
        )

        first_response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": " https://DEV.to/michael_rakutko/?ref=foo#about ",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(source.url, "https://dev.to/michael_rakutko")
        self.assertEqual(source.normalized_url, "https://dev.to/feed/michael_rakutko")
        self.assertEqual(source.source_type, "devto_author")
        self.assertNotContains(
            first_response,
            "Source added and saved for this topic. It will be used when generating the digest.",
        )

        duplicate_response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://dev.to/michael_rakutko/",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(duplicate_response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertContains(
            duplicate_response,
            "This source has already been added to this topic. Please check the address or use another source.",
        )

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_readable_web_article(self, mock_inspect_generic_web_article) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": {
                "title": "The science of safe and healthy baby sleep",
                "url": "https://www.bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
                "content": "A long readable article body about infant sleep, naps, bedtime routines, and wake windows.",
                "source_type": "web_article",
            },
            "diagnostics": {
                "normalized_url": "https://bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
                "source_type": "generic_html",
                "fetch_status": 200,
                "extraction_strategy": "article_tag",
                "usable_text_length": 96,
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Readable article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(source.url, "https://bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep")
        self.assertEqual(source.source_type, "generic_html")
        self.assertEqual(source.name, "The science of safe and healthy baby sleep")
        self.assertNotContains(
            response,
            "Source added and saved for this topic. It will be used when generating the digest.",
        )
        self.assertNotContains(response, "web_article")

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_stanford_style_article_url_with_meaningful_id_query_param(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": {
                "title": "Infant Sleep",
                "url": "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
                "content": "Readable article text about infant sleep, naps, bedtime routines, and wake windows.",
                "source_type": "web_article",
            },
            "diagnostics": {
                "normalized_url": "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
                "source_type": "generic_html",
                "fetch_status": 200,
                "extraction_strategy": "main_content",
                "usable_text_length": 84,
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Stanford article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        source = topic.sources.get()
        self.assertEqual(
            source.url,
            "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
        )
        self.assertEqual(
            source.normalized_url,
            "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
        )
        self.assertEqual(source.name, "Infant Sleep")

    @patch("services.sources.rss_adapter._fetch_url_response")
    def test_add_topic_source_accepts_hopkins_article_via_saved_source_path_when_primary_fetch_is_blocked(
        self,
        mock_fetch_url_response,
    ) -> None:
        challenge_html = """
        <!DOCTYPE html>
        <html lang="en-US">
          <head><title>Just a moment...</title></head>
          <body>
            <h1>Just a moment...</h1>
            <p>Checking your browser before accessing the site.</p>
          </body>
        </html>
        """
        reader_payload = """
Title: Infant Safe Sleep

URL Source: https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep

Published Time: 2025-05-08

Markdown Content:
According to the Centers for Disease Control and Prevention, each year there are about 3,400 sudden unexplained infant deaths (SUID).

A safe sleeping area — along with how you lay your baby down to sleep — can prevent SUID.

## Reducing the Risk for Sleep-Related Infant Deaths

* Babies should sleep on a firm, flat surface.
* Keep loose blankets, pillows, and toys out of the crib.
* Room-sharing without bed-sharing is recommended during the early months.
"""

        def fake_fetch(url: str, accept_header: str = ""):
            if url.startswith("https://r.jina.ai/http://"):
                return {
                    "content": reader_payload.encode("utf-8"),
                    "status": 200,
                    "content_type": "text/plain; charset=utf-8",
                    "final_url": url,
                    "fetch_failure_reason": "",
                }
            return {
                "content": challenge_html.encode("utf-8"),
                "status": 403,
                "content_type": "text/html; charset=UTF-8",
                "final_url": "https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep",
                "fetch_failure_reason": "http 403",
            }

        mock_fetch_url_response.side_effect = fake_fetch
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Hopkins article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(
            source.url,
            "https://hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep",
        )
        self.assertEqual(source.source_type, "generic_html")
        self.assertEqual(source.name, "Infant Safe Sleep")
        self.assertNotContains(response, "Please check the address or try another article.")

    @patch("services.sources.rss_adapter._fetch_url_response")
    def test_add_topic_source_accepts_healthychildren_article_via_saved_source_path(
        self,
        mock_fetch_url_response,
    ) -> None:
        healthychildren_html = """
        <html>
          <head>
            <title>Sleep - HealthyChildren.org</title>
          </head>
          <body class="v4master">
            <header>
              <nav>Home Ages and Stages Baby Toddler Teen Healthy Living Safety Tips</nav>
            </header>
            <div id="s4-bodyContainer">
              <section class="page-content">
                <div class="middle-col-container col-xs-12 col-sm-9 col-md-9 col-lg-9">
                  <div class="layout-content">
                    <h1>Sleep</h1>
                    <div id="ctl00_cphPageContent_PublishingPageContentField__ControlWrapper_RichHtmlField" class="ms-rtestate-field">
                      <p>Babies do not have regular sleep cycles until about 6 months of age. While newborns sleep about 16 to 17 hours per day, they may only sleep 1 or 2 hours at a time.</p>
                      <p>As babies get older, they need less sleep. However, different babies have different sleep needs. It is normal for a 6-month-old to wake up during the night but go back to sleep after a few minutes.</p>
                      <p>Babies can become overtired when they stay awake for too long, so bedtime routines and age-appropriate sleep windows can help parents settle them more easily.</p>
                    </div>
                  </div>
                </div>
                <aside class="article-rollup-container rollup-container">
                  <h2>Articles</h2>
                  <ul class="article-rollup rollup">
                    <li><a href="/English/ages-stages/baby/sleep/Pages/getting-your-baby-to-sleep.aspx">Getting Your Baby to Sleep</a></li>
                  </ul>
                </aside>
              </section>
            </div>
            <footer>About Us Contact Us Advertise</footer>
          </body>
        </html>
        """

        mock_fetch_url_response.return_value = {
            "content": healthychildren_html.encode("utf-8"),
            "status": 200,
            "content_type": "text/html; charset=utf-8",
            "final_url": "https://www.healthychildren.org/English/ages-stages/baby/sleep/Pages/default.aspx",
            "fetch_failure_reason": "",
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="HealthyChildren article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.healthychildren.org/English/ages-stages/baby/sleep/Pages/default.aspx",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(
            source.url,
            "https://healthychildren.org/English/ages-stages/baby/sleep/Pages/default.aspx",
        )
        self.assertEqual(source.source_type, "generic_html")
        self.assertEqual(source.name, "Sleep - HealthyChildren.org")
        self.assertNotContains(response, "Please check the address or try another article.")

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_lullaby_trust_style_parenting_article(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": {
                "title": "Baby sleep patterns | The Lullaby Trust",
                "url": "https://lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns",
                "content": (
                    "Parents and carers often worry about their babies' sleep and might try tips and hacks to get them to sleep longer, but these can actually be dangerous. "
                    "Babies have small stomachs and will wake often throughout the night to feed, and every baby is different."
                ),
                "source_type": "web_article",
            },
            "diagnostics": {
                "normalized_url": "https://lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns",
                "source_type": "generic_html",
                "fetch_status": 200,
                "extraction_strategy": "fallback_text",
                "usable_text_length": 276,
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Parenting article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns/",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "Source added and saved for this topic. It will be used when generating the digest.",
        )

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_saves_reachable_web_article_even_when_extraction_is_unverified(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": None,
            "diagnostics": {
                "normalized_url": "https://example.com/some-article",
                "source_type": "generic_html",
                "fetch_status": 200,
                "fetch_failure_reason": "",
                "extraction_strategy": "fallback_text",
                "usable_text_length": 48,
                "rejection_reason": "page content looked too weak or unstructured",
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Unsupported source topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/some-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        source = topic.sources.get()
        self.assertEqual(source.url, "https://example.com/some-article")
        html = unescape(response.content.decode("utf-8"))
        self.assertIn('"normalized_url": "https://example.com/some-article"', html)
        self.assertIn('"source_type": "generic_html"', html)
        self.assertIn('"rejection_reason": "page content looked too weak or unstructured"', html)
        self.assertNotIn('class="feedback feedback--error"', html)
        self.assertNotIn(
            "Source saved. We reached this URL, but article extraction has not been verified yet. Extraction will be checked during digest generation.",
            html,
        )
        self.assertNotIn('class="source-inline-feedback"', html)

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_prevents_duplicate_normalized_web_article_url(self, mock_inspect_generic_web_article) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": {
                "title": "Baby sleep schedules",
                "url": "https://example.com/articles/baby-sleep",
                "content": "Readable body text about naps, sleep regressions, and bedtime routines for infants.",
                "source_type": "web_article",
            },
            "diagnostics": {
                "normalized_url": "https://example.com/articles/baby-sleep",
                "source_type": "generic_html",
                "fetch_status": 200,
                "extraction_strategy": "article_tag",
                "usable_text_length": 79,
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Duplicate web article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        first_response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/articles/baby-sleep",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        duplicate_response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": " https://example.com/articles/baby-sleep/#top ",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(duplicate_response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertContains(
            duplicate_response,
            "This source has already been added to this topic. Please check the address or use another source.",
        )

    def test_add_topic_source_rejects_invalid_url(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Invalid source topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "not-a-url",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(
            response,
            "Please check the URL format.",
            status_code=400,
        )
        self.assertContains(response, 'value="not-a-url"', html=False, status_code=400)

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_valid_generic_web_article_even_when_fetch_fails(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": None,
            "diagnostics": {
                "normalized_url": "https://missing.example/article",
                "source_type": "generic_html",
                "fetch_status": None,
                "fetch_failure_reason": "Temporary failure in name resolution",
                "extraction_strategy": "fetch_failed",
                "usable_text_length": 0,
                "rejection_reason": "temporary failure in name resolution",
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Unreachable source topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://missing.example/article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(response, "We could not reach this URL.", status_code=200)
        self.assertContains(
            response,
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
            html=False,
            status_code=200,
        )

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_valid_generic_web_article_even_when_fetch_returns_404(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": None,
            "diagnostics": {
                "normalized_url": "https://example.com/missing-article",
                "source_type": "generic_html",
                "fetch_status": 404,
                "fetch_failure_reason": "http 404",
                "extraction_strategy": "fetch_failed",
                "usable_text_length": 0,
                "rejection_reason": "http 404",
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="404 source topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/missing-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(response, "This page returned 404/410.", status_code=200)

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_add_topic_source_accepts_spinning_babies_style_url_without_requiring_live_fetch(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": None,
            "diagnostics": {
                "normalized_url": "https://spinningbabies.com/pregnancy-birth/daily-activities",
                "source_type": "generic_html",
                "fetch_status": 403,
                "fetch_failure_reason": "blocked by anti-bot challenge",
                "extraction_strategy": "fetch_failed",
                "usable_text_length": 0,
                "rejection_reason": "blocked by anti-bot challenge",
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Spinning Babies topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://www.spinningbabies.com/pregnancy-birth/daily-activities/",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        saved_source = topic.sources.get()
        self.assertEqual(saved_source.normalized_url, "https://spinningbabies.com/pregnancy-birth/daily-activities")
        self.assertNotContains(response, "We could not reach this URL.", status_code=200)

    def test_add_source_feedback_renders_below_form_and_without_technical_strings(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Rendered feedback topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "not-a-url",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        html = response.content.decode("utf-8")
        form_marker = '<form method="post" action="{}" novalidate autocomplete="off" class="inline-add-form"'.format(
            reverse("add-topic-source", args=[topic.id])
        )
        controls_marker = '<div class="inline-add-controls">'
        feedback_marker = 'class="source-inline-feedback"'

        self.assertIn("Add a link and press Enter", html)
        self.assertNotIn("Р’РІРµРґРёС‚Рµ URL", html)
        self.assertNotIn("RSS feed detected", html)
        self.assertNotIn("matched RSS/XML URL pattern", html)
        self.assertNotIn("Add source", html)
        self.assertIn(form_marker, html)
        self.assertIn(controls_marker, html)
        self.assertIn(feedback_marker, html)
        self.assertNotIn('class="feedback feedback--error"', html)
        self.assertLess(html.index(form_marker), html.index(controls_marker))
        self.assertLess(html.index(controls_marker), html.index(feedback_marker))
        self.assertIn('autocomplete="off"', html)

    def test_saved_source_form_disables_browser_native_validation(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Saved source novalidate topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        add_source_action = reverse("add-topic-source", args=[topic.id])
        self.assertIn(f'action="{add_source_action}"', html)
        self.assertIn('class="inline-add-form"', html)
        self.assertIn("novalidate", html)
        self.assertIn('autocomplete="off"', html)
        self.assertIn(f'const workspaceUrl = "{reverse("topic-workspace", args=[topic.id])}";', html)
        self.assertIn('window.history.replaceState({}, "", workspaceUrl);', html)
        self.assertIn('data-preserve-scroll', html)
        self.assertIn('const storageKey = "digestflow:scroll-restore";', html)
        self.assertIn('const collapseStoragePrefix = "digestflow:collapse:";', html)

    def test_add_topic_source_rejects_unsupported_scheme(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Unsupported scheme topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "ftp://example.com/article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(response, "Use an http or https URL.", status_code=400)
        self.assertContains(response, 'value="ftp://example.com/article"', html=False, status_code=400)

    def test_failed_saved_source_post_preserves_submitted_url_only_on_that_response(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Saved source failed input topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        failed_response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "not-a-url",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )
        refreshed_response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(failed_response.status_code, 400)
        self.assertContains(failed_response, 'value="not-a-url"', html=False, status_code=400)
        self.assertNotContains(refreshed_response, 'value="not-a-url"', html=False)
        self.assertContains(
            refreshed_response,
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
            html=False,
        )

    def test_workspace_get_renders_empty_saved_source_input(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Saved source workspace get topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
            html=False,
            status_code=200,
        )

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_successful_saved_source_add_renders_empty_input(self, mock_inspect_generic_web_article) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": {
                "title": "Reachable article",
                "url": "https://example.com/reachable-article",
                "content": "Readable article content about sleep routines and parenting decisions.",
                "source_type": "web_article",
            },
            "diagnostics": {
                "normalized_url": "https://example.com/reachable-article",
                "source_type": "generic_html",
                "fetch_status": 200,
                "fetch_failure_reason": "",
                "extraction_strategy": "article_tag",
                "usable_text_length": 68,
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Saved source success input topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/reachable-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
            html=False,
            status_code=200,
        )

    @patch("apps.digests.views.inspect_generic_web_article")
    def test_reachable_but_extraction_unverified_source_saves_without_visible_warning(
        self,
        mock_inspect_generic_web_article,
    ) -> None:
        mock_inspect_generic_web_article.return_value = {
            "article": None,
            "diagnostics": {
                "normalized_url": "https://example.com/reachable-page",
                "source_type": "generic_html",
                "fetch_status": 200,
                "fetch_failure_reason": "",
                "extraction_strategy": "no_candidate_text",
                "usable_text_length": 0,
                "rejection_reason": "no readable article text was extracted",
            },
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Reachable warning topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/reachable-page",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        self.assertNotIn('class="feedback feedback--error"', html)
        self.assertNotIn(
            'class="source-inline-feedback"',
            html,
        )
        self.assertNotIn(
            "Source saved. We reached this URL, but article extraction has not been verified yet. Extraction will be checked during digest generation.",
            html,
        )

    def test_source_add_error_clears_when_input_is_edited(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Source feedback reset topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "not-a-url",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 400)
        html = response.content.decode("utf-8")
        self.assertIn('data-source-feedback', html)
        self.assertIn('data-source-feedback-input', html)
        self.assertIn('data-initial-value="not-a-url"', html)
        self.assertContains(response, "Please check the URL format.", status_code=400)

    @patch("apps.digests.views.fetch_rss_articles")
    def test_add_topic_source_accepts_valid_rss_feed_even_when_fetch_returns_no_items(self, mock_fetch_rss_articles) -> None:
        mock_fetch_rss_articles.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Unreadable feed topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/feed.xml",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "We could not read this RSS feed. Please check the URL and make sure it is a valid RSS or Atom feed.",
            status_code=200,
        )

    @patch("apps.digests.views.fetch_dev_to_article_content")
    def test_add_topic_source_accepts_devto_article_even_when_fetch_returns_no_content(
        self,
        mock_fetch_dev_to_article_content,
    ) -> None:
        mock_fetch_dev_to_article_content.return_value = None
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Missing article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://dev.to/michael_rakutko/some-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "We could not find content at this address. Please check the URL and try again.",
            status_code=200,
        )

    @patch("apps.digests.views.fetch_dev_to_article_content")
    def test_add_topic_source_accepts_valid_devto_article(self, mock_fetch_dev_to_article_content) -> None:
        mock_fetch_dev_to_article_content.return_value = {
            "title": "A real article",
            "url": "https://dev.to/michael_rakutko/some-article",
            "content": "Real content body",
        }
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Valid article topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://dev.to/michael_rakutko/some-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "Source added and saved for this topic. It will be used when generating the digest.",
        )

    @patch("apps.digests.views.fetch_rss_articles")
    def test_add_topic_source_accepts_devto_author_even_when_fetch_returns_no_articles(
        self,
        mock_fetch_rss_articles,
    ) -> None:
        mock_fetch_rss_articles.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty author topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://dev.to/michael_rakutko",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.count(), 1)
        self.assertNotContains(
            response,
            "This source does not seem to contain any articles yet. Please check the address or use another source.",
            status_code=200,
        )

    @patch("apps.digests.views.resolve_source_candidates")
    def test_discovered_candidates_do_not_create_persistent_topic_sources_until_selected(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "detection_reason": "matched dev.to topic pattern",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "quality_estimate": "high",
                "default_selected": True,
                "candidate_origin": "discovered",
            }
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Preview only",
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        topic = Topic.objects.get(name="Preview only")
        self.assertEqual(topic.sources.count(), 1)
        discovered_source = topic.sources.get()
        self.assertEqual(discovered_source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertTrue(discovered_source.is_active)
        self.assertNotContains(response, "No saved sources yet.")

    @patch("apps.digests.views.resolve_source_candidates")
    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_selected_discovered_candidate_stays_in_new_sources_and_does_not_become_saved_source(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
        mock_resolve_source_candidates,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Selected discovery",
            source_mode=TopicSourceMode.HYBRID,
        )
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/ai",
                "title": "DEV Community / #ai",
                "description": "Broad AI engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "detection_reason": "matched dev.to topic pattern",
                "recent_article_count": 12,
                "has_recent_article_count": True,
                "quality_estimate": "high",
                "default_selected": True,
                "candidate_origin": "discovered",
            }
        ]
        mock_fetch_rss_articles.return_value = [
            {
                "title": "AI article",
                "url": "https://example.com/ai-1",
                "source_name": "DEV Community",
                "snippet": "AI article snippet",
            }
        ]

        response = self.client.post(
            reverse("run-with-selected-sources", args=[topic.id]),
            data={"selected_source_urls": ["https://dev.to/t/ai"]},
        )

        self.assertEqual(response.status_code, 302)
        persisted_source = topic.sources.get()
        self.assertEqual(persisted_source.url, "https://dev.to/t/ai")
        self.assertEqual(persisted_source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertTrue(persisted_source.is_active)
        topic.refresh_from_db()
        self.assertEqual(topic.source_mode, TopicSourceMode.HYBRID)
        mock_fetch_rss_articles.assert_called_once_with("https://dev.to/api/articles?tag=ai")
        mock_run_digest_pipeline.assert_called_once()

        workspace_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(workspace_response.status_code, 200)
        workspace_html = workspace_response.content.decode("utf-8")
        self.assertIn("Saved sources", workspace_html)
        self.assertIn("New sources", workspace_html)
        self.assertIn("DEV Community / #ai", workspace_html)
        self.assertIn("1 selected source will be used in the next digest run.", workspace_html)
        saved_section = workspace_html.split("Saved sources", 1)[1].split("New sources", 1)[0]
        new_section = workspace_html.split("New sources", 1)[1]
        self.assertNotIn("DEV Community / #ai", saved_section)
        self.assertIn("DEV Community / #ai", new_section)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_new_sources_are_checked_by_default_after_discovery_and_persist_when_toggled(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/python",
                "title": "DEV Community / #python",
                "description": "Broad Python engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "default_selected": False,
                "candidate_origin": "discovered",
            }
        ]

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Python discovery",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic = Topic.objects.get(name="Python discovery")
        source = topic.sources.get()
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertTrue(source.is_active)
        self.assertContains(discovery_response, "1 selected source will be used in the next digest run.")
        self.assertNotContains(discovery_response, 'class="primary-cta" disabled', html=False)
        discovery_html = discovery_response.content.decode("utf-8")
        checkbox_pattern = re.compile(
            rf'<form method="post" action="{re.escape(reverse("toggle-topic-source", args=[topic.id, source.id]))}".*?<input[^>]*type="checkbox"[^>]*checked',
            re.DOTALL,
        )
        self.assertRegex(discovery_html, checkbox_pattern)
        self.assertContains(
            discovery_response,
            f'<input type="hidden" name="topic_id" value="{topic.id}">',
            html=False,
        )

        toggle_response = self.client.post(reverse("toggle-topic-source", args=[topic.id, source.id]))

        self.assertEqual(toggle_response.status_code, 302)
        self.assertEqual(toggle_response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertFalse(source.is_active)

        refreshed_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        refreshed_html = refreshed_response.content.decode("utf-8")
        self.assertEqual(refreshed_response.status_code, 200)
        self.assertIn("DEV Community / #python", refreshed_html)
        self.assertIn("Please select at least one source to run a new digest.", refreshed_html)
        self.assertNotIn("Saved sources", refreshed_html)
        self.assertIn("data-run-source-count-button", refreshed_html)
        self.assertIn("disabled", refreshed_html)
        new_section = refreshed_html.split("New sources", 1)[1]
        self.assertIn("DEV Community / #python", new_section)
        self.assertNotIn('checked', new_section.split('value="1"', 1)[1].split('>', 1)[0])

    @patch("apps.digests.views.resolve_source_candidates")
    def test_unchecked_discovered_source_post_matches_browser_shape_and_stays_unchecked_after_refresh(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/python",
                "title": "DEV Community / #python",
                "description": "Broad Python engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "default_selected": True,
                "candidate_origin": "discovered",
            }
        ]

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Python browser flow",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic = Topic.objects.get(name="Python browser flow")
        source = topic.sources.get()
        self.assertTrue(source.is_active)

        unchecked_post_response = self.client.post(
            reverse("toggle-topic-source", args=[topic.id, source.id]),
            data={},
        )

        self.assertEqual(unchecked_post_response.status_code, 302)
        self.assertEqual(unchecked_post_response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertFalse(source.is_active)

        refreshed_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        refreshed_html = refreshed_response.content.decode("utf-8")
        self.assertEqual(refreshed_response.status_code, 200)
        self.assertIn("New sources", refreshed_html)
        self.assertNotIn("Saved sources", refreshed_html)
        self.assertIn("DEV Community / #python", refreshed_html)
        self.assertIn("Please select at least one source to run a new digest.", refreshed_html)
        self.assertIn("data-run-source-count-button", refreshed_html)
        self.assertIn("disabled", refreshed_html)
        new_section = refreshed_html.split("New sources", 1)[1]
        self.assertIn("DEV Community / #python", new_section)
        self.assertNotIn('checked', new_section.split('value="1"', 1)[1].split('>', 1)[0])

    @patch("apps.digests.views.resolve_source_candidates")
    def test_rediscovery_keeps_existing_discovered_source_unchecked_instead_of_resetting_default_selection(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://dev.to/t/python",
                "title": "DEV Community / #python",
                "description": "Broad Python engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 8,
                "has_recent_article_count": True,
                "default_selected": False,
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/django",
                "title": "DEV Community / #django",
                "description": "Broad Django engineering stream.",
                "source_type": "devto_tag",
                "platform": "dev.to",
                "recent_article_count": 6,
                "has_recent_article_count": True,
                "default_selected": False,
                "candidate_origin": "discovered",
            },
        ]

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Python rediscovery flow",
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic = Topic.objects.get(name="Python rediscovery flow")
        discovered_sources = {source.url: source for source in topic.sources.order_by("id")}
        python_source = discovered_sources["https://dev.to/t/python"]
        django_source = discovered_sources["https://dev.to/t/django"]
        self.assertTrue(python_source.is_active)
        self.assertTrue(django_source.is_active)
        self.assertContains(discovery_response, "2 selected sources will be used in the next digest run.")

        unchecked_post_response = self.client.post(
            reverse("toggle-topic-source", args=[topic.id, python_source.id]),
            data={},
        )

        self.assertEqual(unchecked_post_response.status_code, 302)
        python_source.refresh_from_db()
        django_source.refresh_from_db()
        self.assertFalse(python_source.is_active)
        self.assertTrue(django_source.is_active)

        rediscovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
            },
        )

        self.assertEqual(rediscovery_response.status_code, 200)
        python_source.refresh_from_db()
        django_source.refresh_from_db()
        self.assertFalse(python_source.is_active)
        self.assertTrue(django_source.is_active)
        self.assertContains(rediscovery_response, "1 selected source will be used in the next digest run.")
        html = rediscovery_response.content.decode("utf-8")
        new_section = html.split("New sources", 1)[1]
        self.assertIn("DEV Community / #python", new_section)
        self.assertIn("DEV Community / #django", new_section)
        python_checkbox = new_section.split("DEV Community / #python", 1)[0].rsplit('value="1"', 1)[1].split('>', 1)[0]
        django_checkbox = new_section.split("DEV Community / #django", 1)[0].rsplit('value="1"', 1)[1].split('>', 1)[0]
        self.assertNotIn("checked", python_checkbox)
        self.assertIn("checked", django_checkbox)

    def test_topic_list_hides_legacy_source_when_persistent_topic_source_exists(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="No duplicate source display",
            source_url="https://dev.to/t/ai",
            keywords=["AI"],
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Saved dev.to",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="manual",
            platform="dev.to",
            is_active=True,
        )

        response = self.client.get(reverse("topic-list"))

        self.assertNotContains(response, "Legacy source URL saved")
        self.assertNotContains(response, "Legacy source: https://dev.to/t/ai")

    def test_saved_source_cards_hide_internal_metadata_labels(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Clean source cards",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="DEV Community / #ai",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin="curated",
            platform="dev.to",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=False,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "DEV Community / #ai")
        self.assertContains(response, "https://dev.to/t/ai")
        self.assertContains(response, 'type="checkbox"', html=False)
        self.assertContains(response, 'aria-label="Remove saved source DEV Community / #ai"', html=False)
        self.assertNotContains(response, "Disabled")
        self.assertNotContains(response, "Enable")
        self.assertNotContains(response, ">Remove<", html=False)
        self.assertNotContains(response, "Curated")
        self.assertNotContains(response, "devto_tag")
        self.assertNotContains(response, "Valid")
        self.assertNotContains(response, "inactive")

        html = response.content.decode("utf-8")
        self.assertIn('onchange="this.form.requestSubmit();"', html)
        self.assertIn('type="checkbox"', html)
        self.assertNotIn('checked', html.split('type="checkbox"', 1)[1].split('>', 1)[0])

    def test_saved_source_card_falls_back_to_domain_when_title_is_access_block_page(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Blocked title topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Unfortunately we are unable to give you access to our site at this time.",
            url="https://www.spinningbabies.com/pregnancy-birth/daily-activities/",
            normalized_url="https://spinningbabies.com/pregnancy-birth/daily-activities",
            source_type="generic_html",
            origin="manual",
            platform="web",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=True,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "spinningbabies.com")
        self.assertContains(response, "spinningbabies.com/pregnancy-birth/daily-activities")
        self.assertNotContains(response, "Unfortunately we are unable to give you access to our site at this time.")

    def test_saved_source_card_falls_back_to_domain_when_title_is_403_forbidden(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Forbidden title topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="403 Forbidden",
            url="https://example.com/protected-article",
            normalized_url="https://example.com/protected-article",
            source_type="generic_html",
            origin="manual",
            platform="web",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<strong>example.com</strong>", html=False)
        self.assertContains(response, "example.com/protected-article")
        self.assertNotContains(response, "403 Forbidden")

    def test_saved_source_card_falls_back_to_domain_when_title_is_anti_bot_wait_page(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Wait page title topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Just a moment...",
            url="https://example.org/blocked",
            normalized_url="https://example.org/blocked",
            source_type="generic_html",
            origin="manual",
            platform="web",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "<strong>example.org</strong>", html=False)
        self.assertContains(response, "example.org/blocked")
        self.assertNotContains(response, "Just a moment...")

    def test_saved_source_card_keeps_legitimate_article_title(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Legitimate title topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="The science of safe and healthy baby sleep",
            url="https://www.bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
            normalized_url="https://bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
            source_type="generic_html",
            origin="manual",
            platform="web",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "The science of safe and healthy baby sleep")
        self.assertContains(response, "bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep")

    def test_curated_source_seed_uses_safe_title_for_blocked_saved_source_name(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Curated seed title topic",
            source_mode=TopicSourceMode.HYBRID,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Unfortunately we are unable to give you access to our site at this time.",
            url="https://www.spinningbabies.com/pregnancy-birth/daily-activities/",
            normalized_url="https://spinningbabies.com/pregnancy-birth/daily-activities",
            source_type="generic_html",
            origin="manual",
            platform="web",
            validation_status=TopicSource.VALIDATION_VALID,
            is_active=True,
        )

        seeds = _build_curated_source_seeds(topic)

        self.assertEqual(len(seeds), 1)
        self.assertEqual(seeds[0].title, "spinningbabies.com")

    @patch("apps.digests.views.resolve_source_candidates")
    def test_can_disable_and_remove_topic_sources_from_review_ui(self, mock_resolve_source_candidates) -> None:
        user = self._get_ui_user()
        topic = Topic.objects.create(user=user, name="Managed sources", source_mode=TopicSourceMode.HYBRID)
        source = TopicSource.objects.create(
            topic=topic,
            name="Saved source",
            url="https://example.com/feed.xml",
            normalized_url="https://example.com/feed.xml",
            source_type="rss_feed",
            origin="manual",
            is_active=True,
        )
        mock_resolve_source_candidates.return_value = []

        toggle_response = self.client.post(reverse("toggle-topic-source", args=[topic.id, source.id]))
        self.assertEqual(toggle_response.status_code, 302)
        self.assertRedirects(
            toggle_response,
            reverse("topic-workspace", args=[topic.id]),
            fetch_redirect_response=False,
        )
        source.refresh_from_db()
        self.assertFalse(source.is_active)

        remove_response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))
        self.assertEqual(remove_response.status_code, 200)
        self.assertFalse(topic.sources.filter(id=source.id).exists())

    def _get_ui_user(self):
        user_model = Topic._meta.get_field("user").remote_field.model
        return user_model.objects.create_user(username="tester")


from datetime import timedelta
