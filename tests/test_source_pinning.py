from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
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
