from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse

from apps.digests.forms import TOPIC_NAME_REQUIRED_MESSAGE, TopicInputForm
from apps.digests import result_messages
from apps.digests.models import DigestRun
from apps.sources.models import Article
from apps.topics.focus import FOCUS_VALIDATION_MESSAGE, is_meaningful_focus_term
from apps.topics.focus_suggestions import generate_focus_suggestions
from apps.topics.models import Topic, TopicSource, TopicSourceMode

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

        self.assertContains(response, "Where to look")
        self.assertNotContains(response, "Source mode")
        self.assertNotContains(response, "Choose how this topic should find sources.")
        self.assertContains(response, 'value="hybrid"', html=False)
        self.assertContains(response, "Saved topics")
        self.assertContains(response, "Recent Digest Runs")
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
        self.assertIn("AI tutors", topic.keywords)
        self.assertIn("education technology", topic.keywords)
        self.assertIn("personalized learning", topic.keywords)
        self.assertNotEqual(topic.keywords, ["AI Education"])
        self.assertContains(response, "AI tutors")
        self.assertContains(response, "education technology")

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

        self.assertTrue(any("sleep" in term.casefold() for term in suggestions))
        self.assertTrue(any("baby" in term.casefold() or "infant" in term.casefold() or "newborn" in term.casefold() for term in suggestions))
        self.assertNotIn("industry tools", suggestions)
        self.assertNotIn("implementation patterns", suggestions)
        self.assertNotIn("practical workflows", suggestions)

    def test_travel_gets_more_than_topic_echo_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("travel")

        self.assertGreater(len(suggestions), 1)
        self.assertIn("travel planning", suggestions)
        self.assertIn("budget travel", suggestions)
        self.assertIn("travel destinations", suggestions)
        self.assertNotEqual(suggestions, ["travel"])
        self.assertNotIn("industry tools", suggestions)
        self.assertNotIn("implementation patterns", suggestions)
        self.assertNotIn("practical workflows", suggestions)

    def test_ai_agents_gets_grounded_technical_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("AI agents")

        self.assertIn("AI agents", suggestions)
        self.assertIn("LLM agents", suggestions)
        self.assertIn("agent workflows", suggestions)
        self.assertIn("multi-agent systems", suggestions)
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
        self.assertIn("AI tutors", topic.keywords)

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
        self.assertNotContains(reopen_response, "AI tutors")

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
        self.assertIn("AI tutors", discovery_request.focus_terms)
        self.assertIn("education technology", discovery_request.focus_terms)

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
            FOCUS_VALIDATION_MESSAGE,
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
            self.assertContains(response, FOCUS_VALIDATION_MESSAGE, status_code=400)
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
            self.assertContains(response, FOCUS_VALIDATION_MESSAGE, status_code=400)
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
            FOCUS_VALIDATION_MESSAGE,
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
        self.assertContains(response, "Topic")
        self.assertContains(response, "Where to look")
        self.assertNotContains(response, "Topic name")
        self.assertNotContains(response, "Source mode")
        self.assertNotContains(response, "Choose how this topic should find sources.")
        self.assertContains(response, "Saved sources")
        self.assertContains(response, "New sources")
        self.assertContains(response, "Find sources")
        self.assertNotContains(response, ">Save<", html=False)
        self.assertNotContains(response, "Refresh source discovery")
        self.assertNotContains(response, "Saved topics")
        self.assertNotContains(response, "Recent Digest Runs")
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
        self.assertContains(response, "Run")
        topic = Topic.objects.get(name="AI agents")
        self.assertEqual(topic.sources.count(), 0)
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
        self.assertLess(html.index("New sources"), html.index('class="pipeline-bar pipeline-bar--final"'))
        self.assertLess(html.index('class="pipeline-bar pipeline-bar--final"'), html.index(">Run<"))
        self.assertIn('name="topic_id" value="', html)
        self.assertIn('onchange="this.form.requestSubmit();"', html)
        self.assertEqual(html.count(">Run<"), 1)

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
        self.assertContains(response, "Run")
        self.assertNotContains(response, "New sources")
        self.assertNotContains(response, "Find sources")
        self.assertNotContains(response, "No new sources were found for this topic yet.")
        self.assertNotContains(response, "DEV Community / #ai")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count(">Run<"), 1)

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
        self.assertContains(response, "DEV Community / #ai")
        self.assertContains(response, "Run")
        self.assertNotContains(response, "Saved sources")
        self.assertNotContains(response, "Add a link and press Enter")
        self.assertNotContains(response, "Add source")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count(">Run<"), 1)

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
        self.assertEqual(topic.source_mode, Topic.SOURCE_MODE_CUSTOM_ONLY)
        source_one.refresh_from_db()
        source_two.refresh_from_db()
        source_three.refresh_from_db()
        self.assertTrue(source_one.is_active)
        self.assertFalse(source_two.is_active)
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

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_add_topic_source_accepts_readable_web_article(self, mock_fetch_generic_web_article) -> None:
        mock_fetch_generic_web_article.return_value = {
            "title": "The science of safe and healthy baby sleep",
            "url": "https://www.bbc.com/future/article/20220131-the-science-of-safe-and-healthy-baby-sleep",
            "content": "A long readable article body about infant sleep, naps, bedtime routines, and wake windows.",
            "source_type": "web_article",
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

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_add_topic_source_accepts_stanford_style_article_url_with_meaningful_id_query_param(
        self,
        mock_fetch_generic_web_article,
    ) -> None:
        mock_fetch_generic_web_article.return_value = {
            "title": "Infant Sleep",
            "url": "https://stanfordchildrens.org/en/topic/default?id=infant-sleep-90-P02237",
            "content": "Readable article text about infant sleep, naps, bedtime routines, and wake windows.",
            "source_type": "web_article",
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

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_add_topic_source_accepts_lullaby_trust_style_parenting_article(
        self,
        mock_fetch_generic_web_article,
    ) -> None:
        mock_fetch_generic_web_article.return_value = {
            "title": "Baby sleep patterns | The Lullaby Trust",
            "url": "https://lullabytrust.org.uk/baby-safety/being-a-parent-or-caregiver/baby-sleep-patterns",
            "content": (
                "Parents and carers often worry about their babies' sleep and might try tips and hacks to get them to sleep longer, but these can actually be dangerous. "
                "Babies have small stomachs and will wake often throughout the night to feed, and every baby is different."
            ),
            "source_type": "web_article",
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

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_add_topic_source_rejects_unreadable_web_article(self, mock_fetch_generic_web_article) -> None:
        mock_fetch_generic_web_article.return_value = None
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

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(
            response,
            "Please check the address or try another article.",
            status_code=400,
        )

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_add_topic_source_prevents_duplicate_normalized_web_article_url(self, mock_fetch_generic_web_article) -> None:
        mock_fetch_generic_web_article.return_value = {
            "title": "Baby sleep schedules",
            "url": "https://example.com/articles/baby-sleep",
            "content": "Readable body text about naps, sleep regressions, and bedtime routines for infants.",
            "source_type": "web_article",
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
            "Please check the address - it does not look like a valid URL. Make sure the link is correct and starts with http:// or https:// so it can be used for the digest.",
            status_code=400,
        )
        self.assertContains(response, 'value="not-a-url"', html=False, status_code=400)

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
        form_marker = '<form method="post" action="{}" class="inline-add-form"'.format(
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

    @patch("apps.digests.views.fetch_generic_web_article")
    def test_source_add_error_clears_when_input_is_edited(self, mock_fetch_generic_web_article) -> None:
        mock_fetch_generic_web_article.return_value = None
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Source feedback reset topic",
            source_mode=TopicSourceMode.HYBRID,
        )

        response = self.client.post(
            reverse("add-topic-source", args=[topic.id]),
            data={
                "source_url": "https://example.com/unreadable-article",
                "source_mode": TopicSourceMode.HYBRID,
            },
        )

        self.assertEqual(response.status_code, 400)
        html = response.content.decode("utf-8")
        self.assertIn('data-source-feedback', html)
        self.assertIn('data-source-feedback-input', html)
        self.assertIn('data-initial-value="https://example.com/unreadable-article"', html)

    @patch("apps.digests.views.fetch_rss_articles")
    def test_add_topic_source_rejects_unreadable_rss_feed(self, mock_fetch_rss_articles) -> None:
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

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(
            response,
            "We could not read this RSS feed. Please check the URL and make sure it is a valid RSS or Atom feed.",
            status_code=400,
        )

    @patch("apps.digests.views.fetch_dev_to_article_content")
    def test_add_topic_source_rejects_missing_devto_article(self, mock_fetch_dev_to_article_content) -> None:
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

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(
            response,
            "We could not find content at this address. Please check the URL and try again.",
            status_code=400,
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
    def test_add_topic_source_rejects_devto_author_without_articles(self, mock_fetch_rss_articles) -> None:
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

        self.assertEqual(response.status_code, 400)
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(
            response,
            "This source does not seem to contain any articles yet. Please check the address or use another source.",
            status_code=400,
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
        self.assertEqual(topic.sources.count(), 0)
        self.assertNotContains(response, "No saved sources yet.")

    @patch("apps.digests.views.resolve_source_candidates")
    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_selected_discovered_candidate_becomes_persistent_topic_source(
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
        self.assertEqual(persisted_source.origin, "discovered")
        self.assertTrue(persisted_source.is_active)
        mock_fetch_rss_articles.assert_called_once_with("https://dev.to/api/articles?tag=ai")
        mock_run_digest_pipeline.assert_called_once()

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
        self.assertEqual(toggle_response.status_code, 200)
        source.refresh_from_db()
        self.assertFalse(source.is_active)

        remove_response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))
        self.assertEqual(remove_response.status_code, 200)
        self.assertFalse(topic.sources.filter(id=source.id).exists())

    def _get_ui_user(self):
        user_model = Topic._meta.get_field("user").remote_field.model
        return user_model.objects.create_user(username="tester")


