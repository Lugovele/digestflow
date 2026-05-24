from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin
from services.sources.candidates import SourceCandidateInput
from services.sources.topic_source_groups import (
    filter_new_source_candidates,
    is_manual_saved_source,
    is_new_research_source,
    is_pinned_research_source,
    split_topic_sources,
)


@override_settings(
    SEARCH_PROVIDER_ENABLED=False,
    SEARCH_PROVIDER="",
    SEARCH_PROVIDER_API_KEY="",
)
class SourcePinningTests(TestCase):
    def _create_topic(self) -> Topic:
        user = get_user_model().objects.create_user(username="pinning-user", password="pw")
        return Topic.objects.create(
            user=user,
            name="AI automation",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["zapier", "n8n"],
            focus_initialized=True,
            excluded_keywords=[],
        )

    def test_manual_sources_are_classified_as_saved_sources(self) -> None:
        topic = self._create_topic()
        manual = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )

        groups = split_topic_sources(topic.sources.all())

        self.assertEqual(groups.manual_saved_sources, (manual,))
        self.assertTrue(is_manual_saved_source(manual))

    def test_discovered_unpinned_sources_are_classified_as_new_sources(self) -> None:
        topic = self._create_topic()
        discovered = TopicSource.objects.create(
            topic=topic,
            name="New research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        groups = split_topic_sources(topic.sources.all())

        self.assertEqual(groups.new_research_sources, (discovered,))
        self.assertTrue(is_new_research_source(discovered))

    def test_discovered_pinned_sources_are_classified_as_pinned_research_sources(self) -> None:
        topic = self._create_topic()
        discovered = TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        groups = split_topic_sources(topic.sources.all())

        self.assertEqual(groups.pinned_research_sources, (discovered,))
        self.assertTrue(is_pinned_research_source(discovered))

    def test_manual_and_pinned_urls_are_excluded_from_new_source_candidates(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Existing unpinned research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        filtered = filter_new_source_candidates(
            [
                {"url": "https://example.com/manual", "title": "Manual source duplicate"},
                {"url": "https://dev.to/t/python", "title": "Pinned source duplicate"},
                {"url": "https://dev.to/t/ai", "title": "Existing unpinned research source"},
                {"url": "https://dev.to/t/django", "title": "Fresh research source"},
            ],
            topic.sources.all(),
        )

        filtered_urls = {candidate["url"] for candidate in filtered}
        self.assertNotIn("https://example.com/manual", filtered_urls)
        self.assertNotIn("https://dev.to/t/python", filtered_urls)
        self.assertIn("https://dev.to/t/ai", filtered_urls)
        self.assertIn("https://dev.to/t/django", filtered_urls)

    def test_filter_new_source_candidates_supports_source_candidate_input_objects(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://saved-source-test.com/article",
            normalized_url="https://saved-source-test.com/article",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://pinned-source-test.com/article",
            normalized_url="https://pinned-source-test.com/article",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        manual_candidate = SourceCandidateInput(
            url="https://saved-source-test.com/article",
            title="Saved source duplicate",
        )
        pinned_candidate = SourceCandidateInput(
            url="https://pinned-source-test.com/article",
            title="Pinned source duplicate",
        )
        fresh_candidate = SourceCandidateInput(
            url="https://fresh-source-test.com/article",
            title="Fresh source",
        )

        filtered = filter_new_source_candidates(
            [manual_candidate, pinned_candidate, fresh_candidate],
            topic.sources.all(),
        )

        self.assertEqual(filtered, [fresh_candidate])
        self.assertIs(filtered[0], fresh_candidate)

    def test_filter_new_source_candidates_preserves_dict_support(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://saved-source-test.com/article",
            normalized_url="https://saved-source-test.com/article",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )

        fresh_candidate = {"url": "https://fresh-source-test.com/article", "title": "Fresh source"}

        filtered = filter_new_source_candidates(
            [
                {"url": "https://saved-source-test.com/article", "title": "Saved source duplicate"},
                fresh_candidate,
            ],
            topic.sources.all(),
        )

        self.assertEqual(filtered, [fresh_candidate])
        self.assertIs(filtered[0], fresh_candidate)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_research_refresh_pruning_removes_only_unpinned_discovered_sources(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        topic = self._create_topic()
        manual = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        pinned = TopicSource.objects.create(
            topic=topic,
            name="Pinned source",
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        stale_unpinned = TopicSource.objects.create(
            topic=topic,
            name="Stale new source",
            url="https://dev.to/t/old",
            normalized_url="https://dev.to/api/articles?tag=old",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://example.com/manual",
                "title": "Manual source duplicate",
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/python",
                "title": "Pinned source duplicate",
                "candidate_origin": "discovered",
            },
            {
                "url": "https://dev.to/t/fresh",
                "title": "Fresh discovered source",
                "candidate_origin": "discovered",
                "default_selected": True,
            },
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertTrue(topic.sources.filter(pk=manual.pk).exists())
        self.assertTrue(topic.sources.filter(pk=pinned.pk, is_pinned=True).exists())
        self.assertFalse(topic.sources.filter(pk=stale_unpinned.pk).exists())
        self.assertTrue(
            topic.sources.filter(
                url="https://dev.to/t/fresh",
                origin=TopicSourceOrigin.DISCOVERED,
                is_pinned=False,
            ).exists()
        )

    def test_pinning_flag_does_not_convert_source_to_manual_saved_source(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertTrue(source.is_pinned)
        self.assertNotEqual(source.origin, TopicSourceOrigin.MANUAL)

    def test_existing_origin_semantics_remain_stable(self) -> None:
        topic = self._create_topic()
        manual = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        discovered = TopicSource.objects.create(
            topic=topic,
            name="Discovered source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        self.assertEqual(manual.origin, TopicSourceOrigin.MANUAL)
        self.assertEqual(discovered.origin, TopicSourceOrigin.DISCOVERED)

    def test_workspace_renders_manual_pinned_and_new_sources_in_separate_sections(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="New research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        before_count = topic.sources.count()

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "My sources")
        self.assertContains(response, "1 my source")
        self.assertContains(response, "1 research source")
        self.assertContains(response, "my sources & research")
        self.assertNotContains(response, "<h3 class=\"section-heading\">Saved sources</h3>", html=False)
        self.assertContains(response, "Research sources")
        self.assertContains(response, 'class="section-heading">Research sources</h3>', html=False)
        self.assertContains(response, "What should research focus on?")
        self.assertContains(response, "Add a research angle and press Enter")
        self.assertContains(response, "Add a manual source link and press Enter")
        self.assertNotContains(response, "Add at least one research angle so DigestFlow knows what sources to look for.")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
        self.assertContains(response, "Find new sources")
        self.assertContains(response, "secondary-button--local", html=False)
        self.assertContains(response, "secondary-button--tertiary", html=False)
        self.assertContains(response, 'class="primary-cta"', html=False)
        self.assertNotContains(response, "Pinned research sources")
        self.assertNotContains(response, "New sources")
        self.assertContains(response, 'section-heading section-heading--subgroup">Kept sources · 1</h4>', html=False)
        self.assertContains(response, 'section-heading section-heading--subgroup">New suggestions · 1</h4>', html=False)
        self.assertNotContains(response, "Sources you chose to keep for future runs.")
        self.assertNotContains(response, "Fresh suggestions from research.")
        self.assertContains(response, ">Keep<", html=False)
        self.assertContains(response, ">Remove<", html=False)
        self.assertEqual(response.content.decode("utf-8").count(">Research sources<"), 1)
        self.assertNotContains(response, "Previously discovered source saved on this topic.")
        self.assertNotContains(response, "Recent articles unknown")

        html = response.content.decode("utf-8")
        saved_section = html.split("My sources", 1)[1].split("Research sources", 1)[0]
        research_section = html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertLess(research_section.index("Kept sources · 1"), research_section.index("Pinned research source"))
        self.assertLess(research_section.index("New suggestions · 1"), research_section.index("New research source"))

        self.assertIn("Manual source", saved_section)
        self.assertNotIn("Pinned research source", saved_section)
        self.assertNotIn("New research source", saved_section)

        self.assertIn("Pinned research source", research_section)
        self.assertIn("New research source", research_section)
        self.assertNotIn("Manual source", research_section)

        topic.refresh_from_db()
        self.assertEqual(topic.sources.count(), before_count)

    def test_workspace_without_focus_disables_find_research_sources_action(self) -> None:
        user = get_user_model().objects.create_user(username="no-focus-pinning-user", password="pw")
        topic = Topic.objects.create(
            user=user,
            name="AI automation",
            source_mode=TopicSourceMode.HYBRID,
            keywords=[],
            focus_initialized=True,
            excluded_keywords=[],
        )
        before_count = topic.sources.count()

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "What should research focus on?")
        self.assertContains(response, "Add a research angle and press Enter")
        self.assertContains(response, "Add at least one research angle so DigestFlow knows what sources to look for.")
        self.assertContains(response, "digestflow:focus-input-restore")
        self.assertContains(response, 'target: "research-angle-input"', html=False)
        self.assertContains(response, "textInput.focus();", html=False)
        self.assertContains(response, "Add at least one research angle before finding sources.")
        self.assertContains(response, "Find sources")
        self.assertContains(response, "Use my sources & research")
        self.assertContains(response, "Use my sources only")
        self.assertContains(response, "Use research sources only")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
        self.assertContains(response, "New suggestions · 0")
        self.assertNotContains(response, "Fresh suggestions from research.")
        self.assertContains(response, "No new suggestions yet.")
        self.assertNotContains(response, "No research sources yet.")
        self.assertContains(response, "secondary-button--local", html=False)
        self.assertContains(response, 'class="primary-cta"', html=False)
        self.assertContains(
            response,
            'Find sources</button>',
            html=False,
        )
        self.assertContains(response, "disabled", html=False)
        topic.refresh_from_db()
        self.assertEqual(topic.sources.count(), before_count)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_find_research_sources_creates_visible_new_sources(self, mock_resolve_source_candidates) -> None:
        topic = self._create_topic()
        manual = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=True,
        )
        pinned = TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://dev.to/t/python",
            normalized_url="https://dev.to/api/articles?tag=python",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        mock_resolve_source_candidates.return_value = [
            {
                "url": "https://example.com/manual",
                "title": "Manual source duplicate",
                "candidate_origin": "discovered",
                "default_selected": True,
            },
            {
                "url": "https://dev.to/t/python",
                "title": "Pinned source duplicate",
                "candidate_origin": "discovered",
                "default_selected": True,
            },
            {
                "url": "https://dev.to/t/ai",
                "title": "New research source",
                "candidate_origin": "discovered",
                "default_selected": True,
            },
        ]

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(topic.sources.filter(pk=manual.pk).exists())
        self.assertTrue(topic.sources.filter(pk=pinned.pk, is_pinned=True).exists())
        new_source = topic.sources.get(url="https://dev.to/t/ai")
        self.assertEqual(new_source.origin, TopicSourceOrigin.DISCOVERED)
        self.assertFalse(new_source.is_pinned)
        self.assertContains(response, "Find new sources")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
        self.assertContains(response, "Kept sources · 1")
        self.assertContains(response, "New suggestions · 1")
        self.assertNotContains(response, "Sources you chose to keep for future runs.")
        self.assertNotContains(response, "Fresh suggestions from research.")
        self.assertContains(response, "secondary-button--local", html=False)
        self.assertContains(response, "secondary-button--tertiary", html=False)
        self.assertNotContains(response, "Previously discovered source saved on this topic.")
        self.assertNotContains(response, "Recent articles unknown")

        html = response.content.decode("utf-8")
        research_section = html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn("New research source", research_section)
        self.assertIn(">Keep<", research_section)
        self.assertIn("Kept sources · 1", research_section)
        self.assertIn("Pinned research source", research_section)
        self.assertIn("New suggestions · 1", research_section)
        self.assertNotIn("Manual source duplicate", research_section)
        self.assertNotIn("Pinned source duplicate", research_section)

    def test_pin_and_unpin_actions_move_discovered_source_between_sections(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="New research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        initial_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        self.assertEqual(initial_response.status_code, 200)
        initial_html = initial_response.content.decode("utf-8")
        self.assertIn(">Keep<", initial_html)
        self.assertNotIn(">Remove<", initial_html.split("Pinned research source", 1)[0] if "Pinned research source" in initial_html else "")
        self.assertIn("New research source", initial_html.split("Research sources", 1)[1])
        self.assertNotIn("Pinned research sources", initial_html)

        pin_response = self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(pin_response.status_code, 302)
        self.assertEqual(pin_response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertTrue(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

        pinned_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        pinned_html = pinned_response.content.decode("utf-8")
        self.assertEqual(pinned_response.status_code, 200)
        self.assertIn("Research sources", pinned_html)
        research_section = pinned_html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn("New research source", research_section)
        self.assertIn(">Remove<", research_section)
        self.assertIn("Kept sources · 1", research_section)
        self.assertNotIn(">Keep<", research_section.split("New suggestions", 1)[0])

        unpin_response = self.client.post(reverse("unpin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(unpin_response.status_code, 302)
        self.assertEqual(unpin_response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertFalse(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

        unpinned_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        unpinned_html = unpinned_response.content.decode("utf-8")
        research_section = unpinned_html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn("New research source", research_section)
        self.assertIn(">Keep<", research_section)
        self.assertIn("New suggestions · 1", research_section)

    def test_new_active_and_inactive_sources_both_render_pin_action(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Active new research source",
            url="https://dev.to/t/active-ai",
            normalized_url="https://dev.to/api/articles?tag=active-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Inactive new research source",
            url="https://dev.to/t/inactive-ai",
            normalized_url="https://dev.to/api/articles?tag=inactive-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        research_section = html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn("Active new research source", research_section)
        self.assertIn("Inactive new research source", research_section)
        self.assertGreaterEqual(research_section.count(">Keep<"), 2)
        self.assertIn("New suggestions · 2", research_section)

    def test_pin_active_source_keeps_is_active_true_and_sets_is_pinned_true(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="Active new research source",
            url="https://dev.to/t/active-ai",
            normalized_url="https://dev.to/api/articles?tag=active-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

        response = self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertTrue(source.is_active)
        self.assertTrue(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

    def test_pin_inactive_source_keeps_is_active_false_and_sets_is_pinned_true(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="Inactive new research source",
            url="https://dev.to/t/inactive-ai",
            normalized_url="https://dev.to/api/articles?tag=inactive-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        )

        response = self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertFalse(source.is_active)
        self.assertTrue(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

    def test_pinned_active_and_inactive_sources_both_render_unpin_action(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Active pinned research source",
            url="https://dev.to/t/pinned-active-ai",
            normalized_url="https://dev.to/api/articles?tag=pinned-active-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        TopicSource.objects.create(
            topic=topic,
            name="Inactive pinned research source",
            url="https://dev.to/t/pinned-inactive-ai",
            normalized_url="https://dev.to/api/articles?tag=pinned-inactive-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=False,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        research_section = html.split("Research sources", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn("Active pinned research source", research_section)
        self.assertIn("Inactive pinned research source", research_section)
        self.assertGreaterEqual(research_section.count(">Remove<"), 2)
        self.assertIn("Kept sources · 2", research_section)

    def test_unpin_active_source_keeps_is_active_true_and_sets_is_pinned_false(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="Active pinned research source",
            url="https://dev.to/t/pinned-active-ai",
            normalized_url="https://dev.to/api/articles?tag=pinned-active-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        response = self.client.post(reverse("unpin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertTrue(source.is_active)
        self.assertFalse(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

    def test_unpin_inactive_source_keeps_is_active_false_and_sets_is_pinned_false(self) -> None:
        topic = self._create_topic()
        source = TopicSource.objects.create(
            topic=topic,
            name="Inactive pinned research source",
            url="https://dev.to/t/pinned-inactive-ai",
            normalized_url="https://dev.to/api/articles?tag=pinned-inactive-ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=False,
        )

        response = self.client.post(reverse("unpin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        self.assertEqual(response.headers["Location"], reverse("topic-workspace", args=[topic.id]))
        source.refresh_from_db()
        self.assertFalse(source.is_active)
        self.assertFalse(source.is_pinned)
        self.assertEqual(source.origin, TopicSourceOrigin.DISCOVERED)

    def test_manual_saved_sources_do_not_render_pin_or_unpin_actions(self) -> None:
        topic = self._create_topic()
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_pinned=False,
            is_active=False,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        html = response.content.decode("utf-8")
        saved_section = html.split("My sources", 1)[1].split("Research sources", 1)[0] if "Research sources" in html else html.split("My sources", 1)[1]
        self.assertIn("Manual source", saved_section)
        self.assertNotIn(">Keep<", saved_section)
        self.assertNotIn(">Remove from kept<", saved_section)
