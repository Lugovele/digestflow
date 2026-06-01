from unittest.mock import patch
from unittest.mock import MagicMock

from html import unescape
import json
import re
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone

from apps.digests.forms import TOPIC_NAME_REQUIRED_MESSAGE, TopicInputForm
from apps.digests import result_messages
from apps.digests.models import DigestRun, SourceDiscoveryHistory, SourceDiscoveryRun
from apps.digests.views import (
    _build_discovery_repair_plan,
    _build_curated_source_seeds,
    _select_repair_queries_for_next_round,
    _build_source_discovery_run_diagnostics,
    _upsert_and_build_source_candidates,
)
from apps.sources.models import Article
from services.sources.content_research_planner import ContentResearchPlannerResult
from services.sources.discovery_history import sync_topic_discovered_sources_into_history
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


@override_settings(
    SEARCH_PROVIDER_ENABLED=False,
    SEARCH_PROVIDER="",
    SEARCH_PROVIDER_API_KEY="",
)
class TopicRssSourceTests(TestCase):
    def _forced_fallback_planner_result(self) -> ContentResearchPlannerResult:
        return ContentResearchPlannerResult(
            planner_status="fallback_used",
            fallback_used=True,
            final_queries=(),
            error_message="Forced deterministic planner fallback for topic RSS source tests.",
        )

    def _assert_any_term_contains(self, terms: list[str], *needles: str) -> None:
        self.assertTrue(
            any(any(needle in term.casefold() for needle in needles) for term in terms),
            f"Expected one of {needles!r} in suggestions: {terms!r}",
        )

    def _assert_not_topic_echo(self, topic_name: str, terms: list[str]) -> None:
        normalized_topic = topic_name.casefold().strip()
        self.assertTrue(any(term.casefold() != normalized_topic for term in terms), terms)

    def _mock_serpapi_urlopen(self, mock_urlopen, organic_results: list[dict]) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"organic_results": organic_results}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response

    def _build_fake_discovery_cycle_round_result(
        self,
        *,
        topic: Topic,
        new_visible_candidates: list[dict],
        status: str = SourceDiscoveryRun.STATUS_COMPLETED,
        provider_result_count: int = 4,
        provider_error_count: int = 0,
        accepted_count: int | None = None,
        rejected_count: int = 0,
        known_or_duplicate_count: int = 0,
        quality_rejected_count: int = 0,
        reason_summary: str = "mixed_low_yield",
        query_count: int = 4,
        per_query_result_counts: list[dict] | None = None,
        weak_domains: list[dict] | None = None,
        weak_material_types: list[dict] | None = None,
        dominant_rejection_reasons: list[dict] | None = None,
    ) -> dict:
        accepted_total = len(new_visible_candidates) if accepted_count is None else int(accepted_count)
        query_rows = list(
            per_query_result_counts
            or [
                {
                    "intent": "analysis",
                    "query": f"{topic.name} market analysis latest",
                    "result_count": provider_result_count,
                }
            ]
        )
        run = SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=status,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=query_count,
            provider_result_count=provider_result_count,
            accepted_count=accepted_total,
            rejected_count=rejected_count,
            new_suggestions_count=len(new_visible_candidates),
            already_known_count=known_or_duplicate_count,
            diagnostics={
                "provider_name": "serpapi",
                "provider_error_count": provider_error_count,
                "raw_result_count": provider_result_count,
                "candidate_input_count": accepted_total + rejected_count,
                "query_count": query_count,
                "source_quality_feedback": {
                    "quality_rejected_count": quality_rejected_count,
                    "known_or_duplicate_count": known_or_duplicate_count,
                    "shown_count": len(new_visible_candidates),
                    "dominant_rejection_reasons": list(dominant_rejection_reasons or []),
                    "weak_domains": list(weak_domains or []),
                    "weak_material_types": list(weak_material_types or []),
                    "preferred_material_types_found": [],
                    "main_quality_issue": "",
                    "planner_quality_guidance": [],
                },
                "query_performance": [],
                "per_query_result_counts": query_rows,
                "provider_errors": (
                    [{"message": "SerpAPI returned an API error."}] if provider_error_count else []
                ),
            },
        )
        return {
            "display_candidate_records": list(new_visible_candidates),
            "new_visible_candidates": list(new_visible_candidates),
            "source_research_result": MagicMock(),
            "discovery_run": run,
            "execution_status": "failed" if provider_error_count else "completed",
            "provider_unavailable": status in {SourceDiscoveryRun.STATUS_FAILED, SourceDiscoveryRun.STATUS_BLOCKED}
            and provider_result_count == 0
            and provider_error_count > 0,
            "provider_error_count": provider_error_count,
            "accepted_count": accepted_total,
            "rejected_count": rejected_count,
            "known_or_duplicate_count": known_or_duplicate_count,
            "quality_rejected_count": quality_rejected_count,
            "returned_count": provider_result_count,
            "reason_summary": reason_summary,
        }

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
    def test_workspace_run_post_blocks_hybrid_topic_without_research_sources_and_preserves_state(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Hybrid topic without research",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["automation"],
            excluded_keywords=[],
        )
        manual_source = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        self.assertRedirects(response, reverse("topic-workspace", args=[topic.id]), fetch_redirect_response=False)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)
        manual_source.refresh_from_db()
        self.assertTrue(manual_source.is_active)
        self.assertEqual(topic.source_mode, TopicSourceMode.HYBRID)
        mock_fetch_rss_articles.assert_not_called()
        mock_run_digest_pipeline.assert_not_called()

        workspace_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        self.assertContains(
            workspace_response,
            "Find or keep at least one research source before running this digest.",
        )

    @patch("apps.digests.views.run_digest_pipeline")
    @patch("apps.digests.views.fetch_rss_articles")
    def test_workspace_run_post_uses_manual_and_research_sources_in_hybrid_mode(
        self,
        mock_fetch_rss_articles,
        mock_run_digest_pipeline,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Hybrid run topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["automation"],
            excluded_keywords=[],
        )
        manual_source = TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/manual",
            normalized_url="https://example.com/manual",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )
        kept_research_source = TopicSource.objects.create(
            topic=topic,
            name="Kept research source",
            url="https://dev.to/t/ai",
            normalized_url="https://dev.to/api/articles?tag=ai",
            source_type="devto_tag",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        mock_fetch_rss_articles.side_effect = [
            [{"title": "Manual article", "url": "https://example.com/articles/manual", "source_name": "Manual"}],
            [{"title": "Research article", "url": "https://example.com/articles/research", "source_name": "Research"}],
        ]

        response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.get(topic=topic)
        self.assertEqual(
            run.input_snapshot.get("selected_source_urls"),
            [manual_source.url, kept_research_source.url],
        )
        mock_fetch_rss_articles.assert_any_call(manual_source.normalized_url)
        mock_fetch_rss_articles.assert_any_call(kept_research_source.normalized_url)
        mock_run_digest_pipeline.assert_called_once()

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
        topic = Topic.objects.get(name="Visible Feed")
        self.assertContains(
            response,
            f'<a href="{reverse("topic-workspace", args=[topic.id])}" class="inline-link">Visible Feed</a>',
            html=False,
        )
        self.assertNotContains(response, ">Settings<", html=False)
        self.assertContains(response, 'aria-label="Delete topic"', html=False)
        self.assertContains(response, "⋮⋮")
        self.assertNotContains(response, "Review sources")
        self.assertContains(response, "0 my sources")
        self.assertNotContains(response, "Legacy source URL saved")
        self.assertContains(response, "Delete this topic?")
        self.assertNotContains(response, ">Delete<", html=False)
        self.assertContains(response, 'class="drag-handle"', html=False)
        self.assertContains(response, 'draggable="true"', html=False)

    def test_topic_title_link_opens_existing_workspace_page(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Workspace Link Topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["automation"],
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-list"))

        self.assertContains(
            response,
            f'<a href="{reverse("topic-workspace", args=[topic.id])}" class="inline-link">Workspace Link Topic</a>',
            html=False,
        )

        workspace_response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(workspace_response.status_code, 200)
        self.assertContains(workspace_response, "Workspace Link Topic")

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
        self.assertContains(response, "What should research focus on?")
        self.assertContains(response, "AI automation")
        self.assertContains(response, "workflow automation")
        self.assertContains(response, "Add a research angle and press Enter")
        self.assertNotContains(response, '<h3 class="section-heading">Focus</h3>', html=False)
        self.assertContains(response, 'data-focus-form', html=False)
        self.assertContains(response, 'data-focus-input', html=False)
        self.assertContains(response, 'data-focus-chip-list', html=False)
        self.assertNotContains(response, "Source discovery completed")
        self.assertNotContains(response, "No new research sources found")
        self.assertNotContains(response, "Source discovery did not run")
        self.assertNotContains(response, "Source discovery results")
        self.assertNotContains(response, "Show all suggestions")

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

    def test_n8n_existing_topic_echo_still_gets_contextual_focus_suggestions(self) -> None:
        suggestions = generate_focus_suggestions("n8n", existing_terms=["n8n"])
        lowered_suggestions = [term.casefold() for term in suggestions]
        signal_needles = (
            "automation",
            "integration",
            "integrations",
            "workflow",
            "workflows",
            "api",
            "custom node",
            "error handling",
            "use case",
            "use cases",
        )
        matched_signals = {
            needle
            for needle in signal_needles
            if any(needle in suggestion for suggestion in lowered_suggestions)
        }

        self.assertTrue(suggestions)
        self.assertGreaterEqual(len(set(lowered_suggestions)), 3)
        self.assertNotEqual(lowered_suggestions, ["n8n"])
        self.assertTrue(all("n8n" in suggestion for suggestion in lowered_suggestions), suggestions)
        self.assertGreaterEqual(len(matched_signals), 3, suggestions)

    @patch("apps.topics.focus_suggestions._generate_ai_focus_candidates", return_value=[])
    def test_n8n_deterministic_fallback_still_produces_product_specific_angles(self, _mock_ai_focus_candidates) -> None:
        suggestions = generate_focus_suggestions("n8n", existing_terms=["n8n"])

        self.assertIn("n8n integrations", suggestions)
        self.assertIn("n8n workflow templates", suggestions)
        self.assertIn("n8n self-hosting", suggestions)
        self.assertIn("n8n automation examples", suggestions)
        self.assertIn("n8n vs Zapier", suggestions)

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
                "run_research": "1",
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI agents",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI agents", "agent workflows"],
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
                "run_research": "1",
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
        self.assertContains(response, "My sources")
        self.assertContains(response, "Research sources")
        self.assertContains(response, "New suggestions")
        self.assertContains(response, "Find new sources")
        self.assertContains(response, "Use my sources & research")
        self.assertContains(response, "Use my sources only")
        self.assertContains(response, "Use research sources only")
        self.assertNotContains(response, ">Save<", html=False)
        self.assertNotContains(response, "Refresh source discovery")
        self.assertNotContains(response, "Saved topics")
        self.assertNotContains(response, "Recent digests")
        self.assertNotContains(response, "Topic settings")
        self.assertContains(response, "0 my sources")
        self.assertContains(response, "Add a manual source link and press Enter")
        self.assertNotContains(response, "Add source")
        self.assertNotContains(response, "Р’РІРµРґРёС‚Рµ URL")
        self.assertContains(response, "DEV Community / #ai")
        self.assertNotContains(response, "12 recent articles")
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
        self.assertNotContains(response, "Fresh suggestions from research.")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
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
        self.assertIn("my sources &amp; research", html)
        self.assertNotIn("Saved + New", html)
        self.assertNotIn("Saved + new", html)
        self.assertNotIn("saved &amp; new", html)
        self.assertNotIn("saved &amp; research", html)
        self.assertIn('<h1 class="page-title">AI agents</h1>', html)
        self.assertNotIn("<h1>DigestFlow</h1>", html)
        self.assertNotIn('<h2 class="workflow-title">AI agents</h2>', html)
        self.assertLess(html.index('<h1 class="page-title">AI agents</h1>'), html.index("My sources"))
        self.assertLess(html.index("Research sources"), html.index("Find new sources"))
        self.assertLess(html.index("Research sources"), html.index("Run digest"))
        self.assertIn("Select at least one my source before running this digest.", html)
        self.assertIn("0 my sources", html)
        self.assertIn("disabled", html)
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
        self.assertContains(response, "my sources only")
        self.assertContains(response, "Use my sources only")
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
        self.assertContains(response, "my sources only")
        self.assertContains(response, "My sources")
        self.assertContains(response, "Run digest")
        self.assertContains(response, "Select at least one my source before running this digest.")
        self.assertNotContains(response, "Research sources")
        self.assertNotContains(response, "Find sources")
        self.assertNotContains(response, "No new suggestions yet.")
        self.assertNotContains(response, "Source discovery completed")
        self.assertNotContains(response, "No new research sources found")
        self.assertNotContains(response, "Source discovery results")
        self.assertNotContains(response, "Show all suggestions")
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Discovery AI",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Discovery AI")
        self.assertContains(response, "research only")
        self.assertContains(response, "Research sources")
        self.assertContains(response, "New suggestions")
        self.assertContains(response, "Find new sources")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
        self.assertContains(response, "DEV Community / #ai")
        self.assertContains(response, "Run digest")
        self.assertContains(response, "1 selected source will be used in the next digest run.")
        self.assertNotContains(response, "No new suggestions yet.")
        self.assertNotContains(response, "My sources")
        self.assertNotContains(response, "Add a manual source link and press Enter")
        self.assertNotContains(response, "Add source")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count('class="primary-cta"'), 1)
        self.assertNotIn('class="primary-cta" disabled', html)

    @patch("apps.digests.views.resolve_source_candidates")
    def test_empty_discovery_workspace_still_renders_run_digest_card_disabled(self, mock_resolve_source_candidates) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty discovery topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Empty discovery topic")
        self.assertContains(response, "Research sources")
        self.assertContains(response, "New suggestions")
        self.assertContains(response, "No new suggestions yet.")
        self.assertContains(response, "Find sources")
        self.assertContains(response, "Check sources to use in the next digest. Keep useful ones for future runs.")
        self.assertNotContains(response, "No new sources were found for this topic yet.")
        self.assertContains(response, "Ready to generate")
        self.assertContains(response, "Find or keep at least one research source before running this digest.")
        self.assertContains(response, "Run digest")

        html = response.content.decode("utf-8")
        self.assertEqual(html.count('class="primary-cta"'), 1)
        self.assertIn('class="primary-cta"', html)
        self.assertIn('data-run-source-count-button', html)
        self.assertIn('disabled', html)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    @patch("apps.digests.views.resolve_source_candidates")
    def test_workspace_shows_research_provider_disabled_notice_for_discovery_topics(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Disabled research topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research is currently disabled")
        self.assertContains(
            response,
            "DigestFlow can still use your sources, but automatic research is turned off.",
        )
        self.assertContains(response, "Research unavailable")
        self.assertContains(response, 'Find sources</button>', html=False)
        self.assertContains(response, 'aria-disabled="true"', html=False)
        self.assertNotContains(response, "Missing settings:")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="",
    )
    @patch("apps.digests.views.resolve_source_candidates")
    def test_workspace_shows_research_provider_missing_config_notice(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Missing config topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research provider needs configuration")
        self.assertContains(
            response,
            "Automatic research is enabled, but the selected provider is missing required settings.",
        )
        self.assertContains(response, "Provider setup required")
        self.assertContains(response, 'aria-disabled="true"', html=False)
        self.assertContains(response, "Missing settings: SEARCH_PROVIDER_API_KEY")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="tavily",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views.resolve_source_candidates")
    def test_workspace_shows_research_provider_not_implemented_notice(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Not implemented research topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research provider is not connected yet")
        self.assertContains(
            response,
            "The selected provider is configured, but the real search adapter has not been implemented yet.",
        )
        self.assertContains(response, "Search adapter not connected")
        self.assertContains(response, 'aria-disabled="true"', html=False)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    @patch("apps.digests.views.resolve_source_candidates")
    def test_curated_only_workspace_does_not_show_research_provider_notice(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_name": "Manual workflow topic",
                "source_url": "",
                "source_mode": TopicSourceMode.CURATED_ONLY,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "Research is currently disabled")
        self.assertNotContains(response, "Research provider needs configuration")
        self.assertNotContains(response, "Research provider is not connected yet")
        self.assertNotContains(response, "Research unavailable")
        self.assertNotContains(response, "Provider setup required")
        self.assertNotContains(response, "Search adapter not connected")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="fake",
        SEARCH_PROVIDER_API_KEY="",
    )
    @patch("apps.digests.views.resolve_source_candidates")
    def test_workspace_keeps_find_sources_active_when_provider_is_ready(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.return_value = []
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Ready provider topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Find sources")
        self.assertNotContains(response, "Research is currently disabled")
        self.assertNotContains(response, "Research provider needs configuration")
        self.assertNotContains(response, "Research provider is not connected yet")
        self.assertNotContains(response, "Research unavailable")
        self.assertNotContains(response, "Provider setup required")
        self.assertNotContains(response, "Search adapter not connected")
        self.assertNotContains(response, 'aria-disabled="true"', html=False)

    def test_workspace_keeps_find_sources_label_before_any_discovered_research_sources(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Fresh research topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Find sources")
        self.assertNotContains(response, "Find new sources")
        self.assertContains(
            response,
            f'href="{reverse("topic-research-history", args=[topic.id])}"',
            html=False,
        )
        self.assertNotContains(response, "Source discovery details")
        self.assertNotContains(response, "Provider filter")

    def test_workspace_uses_find_new_sources_label_when_pinned_research_source_exists(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Pinned research topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://example.org/research/pinned-source",
            normalized_url="https://example.org/research/pinned-source",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Find new sources</button>', html=False)
        self.assertContains(
            response,
            f'href="{reverse("topic-research-history", args=[topic.id])}"',
            html=False,
        )
        self.assertNotContains(response, "Source discovery details")

    def test_workspace_keeps_find_sources_label_when_only_manual_sources_exist(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Manual sources topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.org/manual-source",
            normalized_url="https://example.org/manual-source",
            source_type="website",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Find sources")
        self.assertNotContains(response, "Find new sources")
        self.assertContains(
            response,
            f'href="{reverse("topic-research-history", args=[topic.id])}"',
            html=False,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_blocked_provider_with_existing_discovered_source_uses_find_new_sources_label(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Blocked provider existing research topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Existing research source",
            url="https://example.org/research/existing-source",
            normalized_url="https://example.org/research/existing-source",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research unavailable")
        self.assertContains(response, 'Find new sources</button>', html=False)
        self.assertContains(response, 'aria-disabled="true"', html=False)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.research_queries.create_content_research_plan")
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_creates_discovered_new_suggestions(
        self,
        mock_urlopen,
        mock_create_content_research_plan,
    ) -> None:
        original_query = "topic rss unique planner query"
        mock_create_content_research_plan.return_value = MagicMock(
            planner_status="ai_planned",
            fallback_used=False,
            final_queries=(original_query,),
            diagnostics={
                "planner_status": "ai_planned",
                "fallback_used": False,
                "final_queries": [original_query],
                "topic_interpretation": "AI automation topic interpretation",
                "content_research_goal": "Find fresh, practical materials.",
                "source_selection_criteria": {},
                "content_tension_opportunities": [],
                "search_angles": [],
            },
        )
        repaired_queries_seen: list[str] = []

        def _bounded_cycle_urlopen(request, timeout=None):
            query = parse_qs(urlparse(request.full_url).query).get("q", [""])[0]
            response = MagicMock()
            if query == original_query:
                response.read.return_value = json.dumps(
                    {
                        "organic_results": [
                            {
                                "position": 1,
                                "title": "AI automation guide",
                                "link": "https://example.com/ai-guide",
                                "snippet": "Practical guide for AI automation workflows.",
                                "source": "Example",
                            }
                        ]
                    }
                ).encode("utf-8")
            else:
                repaired_queries_seen.append(query)
                response.read.return_value = json.dumps({"organic_results": []}).encode("utf-8")
            context_manager = MagicMock()
            context_manager.__enter__.return_value = response
            return context_manager

        mock_urlopen.side_effect = _bounded_cycle_urlopen
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI automation guide")
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "Found 1 new source suggestion after 3 search rounds. DigestFlow could not reach the 6-source target with the current search strategy.",
        )
        self.assertContains(response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertNotContains(response, "Source discovery details")
        self.assertNotContains(response, "Provider filter")
        self.assertNotContains(response, "Angle reason")
        self.assertNotContains(response, "Previous discovery runs")
        self.assertNotContains(response, "implementation guide")
        topic.refresh_from_db()
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)
        discovery_runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(discovery_runs), 3)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.MANUAL).count(), 0)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)
        discovered_source = topic.sources.get(origin=TopicSourceOrigin.DISCOVERED)
        self.assertEqual(discovered_source.url, "https://example.com/ai-guide")
        self.assertTrue(discovered_source.is_active)
        self.assertFalse(discovered_source.is_pinned)
        discovery_run = discovery_runs[0]
        self.assertEqual(discovery_run.status, SourceDiscoveryRun.STATUS_COMPLETED)
        self.assertEqual(discovery_run.provider_name, "serpapi")
        self.assertEqual(discovery_run.new_suggestions_count, 1)
        self.assertEqual(discovery_run.diagnostics.get("planner_status"), "ai_planned")
        self.assertEqual(
            [item.get("query") for item in discovery_run.diagnostics.get("query_performance", [])],
            ["topic rss unique planner query"],
        )
        final_cycle = discovery_runs[-1].diagnostics.get("discovery_cycle") or {}
        self.assertEqual(final_cycle.get("decision"), "max_rounds_reached")
        self.assertEqual(final_cycle.get("round_count"), 3)
        self.assertEqual(final_cycle.get("max_immediate_rounds"), 3)
        self.assertEqual(final_cycle.get("accumulated_visible_suggestions"), 1)
        self.assertTrue(final_cycle.get("rounds", [])[1].get("used_repair_plan"))
        self.assertTrue(final_cycle.get("rounds", [])[2].get("used_repair_plan"))
        self.assertTrue(repaired_queries_seen)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url="https://example.com/ai-guide")
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_NEW_SHOWN)
        self.assertEqual(history_item.seen_count, 1)
        self.assertTrue(history_item.created_topic_source)
        self.assertEqual(history_item.topic_source_id, discovered_source.id)
        self.assertTrue(history_item.query_text)
        self.assertEqual(history_item.query_text, original_query)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.research_queries.create_content_research_plan")
    @patch("services.sources.serpapi_provider.urlopen")
    def test_partial_provider_failure_still_surfaces_successful_new_suggestions(
        self,
        mock_urlopen,
        mock_create_content_research_plan,
    ) -> None:
        successful_query = "bitcoin infrastructure case study"
        failed_query = "bitcoin impossible angle"
        repaired_queries_seen: list[str] = []
        mock_create_content_research_plan.return_value = MagicMock(
            planner_status="ai_planned",
            fallback_used=False,
            final_queries=(successful_query, failed_query),
            diagnostics={
                "planner_status": "ai_planned",
                "fallback_used": False,
                "final_queries": [successful_query, failed_query],
                "topic_interpretation": "Bitcoin infrastructure and operating practices.",
                "content_research_goal": "Find useful post-worthy bitcoin materials.",
                "source_selection_criteria": {},
                "content_tension_opportunities": [],
                "search_angles": [],
            },
        )

        def _mixed_urlopen(request, timeout=None):
            query = parse_qs(urlparse(request.full_url).query).get("q", [""])[0]
            if query == successful_query:
                response = MagicMock()
                response.read.return_value = json.dumps(
                    {
                        "organic_results": [
                            {
                                "position": 1,
                                "title": "Bitcoin infrastructure case study",
                                "link": "https://example.com/bitcoin-infrastructure-case-study",
                                "snippet": (
                                    "A detailed case study covering tradeoffs, operating constraints, "
                                    "energy strategy, and implementation lessons from a bitcoin operator."
                                ),
                                "source": "Example",
                                "date": "2026-05-20",
                            }
                        ]
                    }
                ).encode("utf-8")
                context_manager = MagicMock()
                context_manager.__enter__.return_value = response
                return context_manager
            if query == failed_query:
                response = MagicMock()
                response.read.return_value = json.dumps(
                    {"error": "Google hasn't returned any results for this query."}
                ).encode("utf-8")
                context_manager = MagicMock()
                context_manager.__enter__.return_value = response
                return context_manager
            if query not in {successful_query, failed_query}:
                self.assertLessEqual(len(query.split()), 8)
                self.assertNotEqual(query, successful_query)
                self.assertNotEqual(query, failed_query)
                repaired_queries_seen.append(query)
                response = MagicMock()
                response.read.return_value = json.dumps(
                    {"organic_results": []}
                ).encode("utf-8")
                context_manager = MagicMock()
                context_manager.__enter__.return_value = response
                return context_manager
            raise AssertionError(f"Unexpected query: {query!r}")

        mock_urlopen.side_effect = _mixed_urlopen
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="bitcoin",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Bitcoin infrastructure case study")
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(response, "Some searches could not be completed.")
        self.assertContains(response, "1 new source suggestion is still available after 3 search rounds.")

        discovered_source = TopicSource.objects.get(topic=topic, origin=TopicSourceOrigin.DISCOVERED)
        self.assertEqual(discovered_source.url, "https://example.com/bitcoin-infrastructure-case-study")
        self.assertTrue(discovered_source.is_active)
        self.assertFalse(discovered_source.is_pinned)

        discovery_runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(discovery_runs), 3)
        discovery_run = discovery_runs[0]
        self.assertEqual(discovery_run.status, SourceDiscoveryRun.STATUS_PARTIAL_FAILED)
        self.assertEqual(discovery_run.new_suggestions_count, 1)
        self.assertEqual(discovery_run.provider_result_count, 1)

        query_performance = discovery_run.diagnostics.get("query_performance", [])
        successful_row = next(item for item in query_performance if item.get("query") == successful_query)
        failed_row = next(item for item in query_performance if item.get("query") == failed_query)
        self.assertEqual(successful_row.get("status"), "useful")
        self.assertEqual(successful_row.get("visible_new_suggestions_count"), 1)
        self.assertEqual(successful_row.get("accepted_count"), 1)
        self.assertEqual(failed_row.get("status"), "partial_error")
        self.assertEqual(failed_row.get("returned_count"), 0)
        self.assertTrue(str(failed_row.get("error_message") or "").strip())
        final_cycle = discovery_runs[-1].diagnostics.get("discovery_cycle") or {}
        self.assertEqual(final_cycle.get("decision"), "max_rounds_reached")
        self.assertEqual(final_cycle.get("round_count"), 3)
        self.assertEqual(final_cycle.get("max_immediate_rounds"), 3)
        self.assertEqual(final_cycle.get("accumulated_visible_suggestions"), 1)
        self.assertTrue(final_cycle.get("rounds", [])[1].get("used_repair_plan"))
        self.assertTrue(final_cycle.get("rounds", [])[2].get("used_repair_plan"))
        self.assertTrue(repaired_queries_seen)
        self.assertEqual(
            [item.get("query") for item in (final_cycle.get("rounds", [])[1].get("repair_plan_usage") or {}).get("repair_queries_used", [])],
            repaired_queries_seen[:len((final_cycle.get("rounds", [])[1].get("repair_plan_usage") or {}).get("repair_queries_used", []))],
        )

        history_item = SourceDiscoveryHistory.objects.get(
            topic=topic,
            normalized_url="https://example.com/bitcoin-infrastructure-case-study",
        )
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_NEW_SHOWN)
        self.assertTrue(history_item.created_topic_source)
        self.assertEqual(history_item.topic_source_id, discovered_source.id)
        self.assertEqual(history_item.query_text, successful_query)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_recovers_from_zero_visible_first_round_to_target_reached_second_round(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="cycle runner manual verification",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_two_candidates = [
            {
                "title": f"Round two recovery source {index}",
                "url": f"https://example.com/cycle-recovery-{index}",
                "normalized_url": f"https://example.com/cycle-recovery-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 7)
        ]
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=5,
                accepted_count=0,
                rejected_count=4,
                quality_rejected_count=4,
                known_or_duplicate_count=1,
                reason_summary="zero_visible",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=round_two_candidates,
                provider_result_count=6,
                accepted_count=6,
                reason_summary="target_reached",
            ),
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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 2)
        self.assertContains(response, "Source discovery completed")
        self.assertContains(response, "Found 6 new source suggestions after 2 search rounds.")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 6)

        cycle_runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(cycle_runs), 2)
        cycle = cycle_runs[-1].diagnostics.get("discovery_cycle") or {}
        self.assertEqual(cycle.get("target_visible_new_suggestions"), 6)
        self.assertEqual(cycle.get("max_immediate_rounds"), 3)
        self.assertEqual(cycle.get("round_count"), 2)
        self.assertEqual(cycle.get("rounds_run"), 2)
        self.assertEqual(cycle.get("accumulated_visible_suggestions"), 6)
        self.assertEqual(cycle.get("decision"), "target_reached")
        self.assertEqual((cycle.get("repair_plan") or {}).get("strategy"), "stop")
        self.assertEqual((cycle.get("repair_plan") or {}).get("query_repair_plan"), [])
        self.assertEqual(len(cycle.get("rounds") or []), 2)
        self.assertEqual(cycle["rounds"][0].get("visible_new_suggestions"), 0)
        self.assertEqual(cycle["rounds"][0].get("reason_summary"), "zero_visible")
        self.assertEqual(cycle["rounds"][1].get("visible_new_suggestions"), 6)
        self.assertEqual(cycle["rounds"][1].get("reason_summary"), "target_reached")

        history_response = self.client.get(reverse("topic-research-history", args=[topic.id]))
        self.assertEqual(history_response.status_code, 200)
        self.assertContains(history_response, "Discovery cycle")
        self.assertContains(history_response, "Cycle round")
        self.assertContains(history_response, "Cycle target")
        copy_report = history_response.context["full_history_copy_report"]
        self.assertIn("Discovery cycle", copy_report)
        self.assertIn("rounds run: 2", copy_report)
        self.assertIn("accumulated visible suggestions: 6", copy_report)
        self.assertIn("decision: target_reached", copy_report)
        self.assertIn("Strategy repair", copy_report)
        self.assertIn("strategy: stop", copy_report)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_runs_second_round_when_first_round_is_below_target(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle retry topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_one_candidates = [
            {
                "title": "Round one source",
                "url": "https://example.com/round-1",
                "normalized_url": "https://example.com/round-1",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
        ]
        round_two_candidates = [
            {
                "title": f"Round two source {index}",
                "url": f"https://example.com/round-2-{index}",
                "normalized_url": f"https://example.com/round-2-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 6)
        ]
        round_one_query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "result_count": 0},
            {"query": "Bitcoin market analysis institutional flows latest", "result_count": 0},
            {"query": "Bitcoin market analysis funding rates latest", "result_count": 0},
            {"query": "Bitcoin market analysis open interest latest", "result_count": 0},
        ]
        captured_round_two_queries: list[str] = []

        def fake_round(*, round_index, query_plan_override=None, repair_usage=None, **kwargs):
            if round_index == 1:
                self.assertIsNone(query_plan_override)
                self.assertIsNone(repair_usage)
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=round_one_candidates,
                    accepted_count=1,
                    reason_summary="zero_visible",
                    per_query_result_counts=round_one_query_rows,
                )

            self.assertEqual(round_index, 2)
            self.assertIsNotNone(query_plan_override)
            self.assertIsNotNone(repair_usage)
            self.assertTrue(repair_usage.get("used_repair_plan"))
            self.assertEqual(repair_usage.get("repair_plan_source_round"), 1)
            captured_round_two_queries[:] = [item.query for item in query_plan_override.query_items]
            self.assertEqual(
                captured_round_two_queries,
                [
                    "Bitcoin ETF flows weekly report",
                    "Bitcoin treasury holdings institutional demand",
                    "Bitcoin funding rates open interest report",
                    "Bitcoin derivatives positioning market structure",
                ],
            )
            self.assertTrue(all(len(query.split()) <= 8 for query in captured_round_two_queries))
            self.assertTrue(all(query not in {row['query'] for row in round_one_query_rows} for query in captured_round_two_queries))
            return self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=round_two_candidates,
                accepted_count=5,
                reason_summary="target_reached",
                per_query_result_counts=[
                    {"query": query, "result_count": 1}
                    for query in captured_round_two_queries
                ],
            )

        mock_run_provider_discovery_round.side_effect = fake_round

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 2)
        self.assertContains(response, "Source discovery completed")
        self.assertContains(response, "Found 6 new source suggestions after 2 search rounds.")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 6)
        cycle_runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(cycle_runs), 2)
        for run in cycle_runs:
            cycle = run.diagnostics.get("discovery_cycle") or {}
            self.assertEqual(cycle.get("decision"), "target_reached")
            self.assertEqual(cycle.get("round_count"), 2)
            self.assertEqual(cycle.get("accumulated_visible_suggestions"), 6)
            self.assertEqual(len(cycle.get("rounds") or []), 2)
            self.assertEqual((cycle.get("cycle_diagnosis") or {}).get("primary_cause"), "target_reached")
        latest_cycle = cycle_runs[-1].diagnostics.get("discovery_cycle") or {}
        self.assertTrue(latest_cycle["rounds"][1].get("used_repair_plan"))
        self.assertEqual((latest_cycle["rounds"][1].get("repair_plan_usage") or {}).get("repair_plan_source_round"), 1)
        self.assertEqual(
            [item.get("query") for item in (latest_cycle["rounds"][1].get("repair_plan_usage") or {}).get("repair_queries_used", [])],
            captured_round_two_queries,
        )
        self.assertEqual(
            [item.get("query") for item in latest_cycle["rounds"][0].get("query_rows", [])],
            [row["query"] for row in round_one_query_rows],
        )
        self.assertEqual(
            [item.get("query") for item in latest_cycle["rounds"][1].get("query_rows", [])],
            captured_round_two_queries,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_stops_after_first_round_when_target_is_reached(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle stop topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_one_candidates = [
            {
                "title": f"Immediate source {index}",
                "url": f"https://example.com/immediate-{index}",
                "normalized_url": f"https://example.com/immediate-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 7)
        ]
        mock_run_provider_discovery_round.return_value = self._build_fake_discovery_cycle_round_result(
            topic=topic,
            new_visible_candidates=round_one_candidates,
            accepted_count=6,
            reason_summary="target_reached",
        )

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 1)
        self.assertContains(response, "Source discovery completed")
        self.assertContains(response, "Found 6 new source suggestions.")
        self.assertNotContains(response, "after 2 search rounds")
        run = SourceDiscoveryRun.objects.get(topic=topic)
        cycle = run.diagnostics.get("discovery_cycle") or {}
        self.assertEqual(cycle.get("decision"), "target_reached")
        self.assertEqual(cycle.get("round_count"), 1)
        self.assertEqual((cycle.get("cycle_diagnosis") or {}).get("primary_cause"), "target_reached")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_does_not_loop_when_provider_is_unavailable(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle provider unavailable topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        mock_run_provider_discovery_round.return_value = self._build_fake_discovery_cycle_round_result(
            topic=topic,
            new_visible_candidates=[],
            status=SourceDiscoveryRun.STATUS_FAILED,
            provider_result_count=0,
            provider_error_count=2,
            accepted_count=0,
            rejected_count=0,
            reason_summary="provider_error",
        )

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 1)
        self.assertContains(response, "Source search is temporarily unavailable")
        self.assertContains(response, "DigestFlow could not connect to the search provider. Please try again later.")
        run = SourceDiscoveryRun.objects.get(topic=topic)
        cycle = run.diagnostics.get("discovery_cycle") or {}
        self.assertEqual(cycle.get("decision"), "provider_unavailable")
        self.assertEqual(cycle.get("round_count"), 1)
        self.assertEqual((cycle.get("cycle_diagnosis") or {}).get("primary_cause"), "provider_unavailable")
        self.assertEqual((cycle.get("repair_plan") or {}).get("strategy"), "stop_provider_unavailable")
        self.assertEqual((cycle.get("repair_plan") or {}).get("query_repair_plan"), [])

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_counts_partial_provider_failure_suggestions_toward_target(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle partial provider failure topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_one_candidates = [
            {
                "title": f"Partial source {index}",
                "url": f"https://example.com/partial-{index}",
                "normalized_url": f"https://example.com/partial-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 5)
        ]
        round_two_candidates = [
            {
                "title": f"Recovery source {index}",
                "url": f"https://example.com/recovery-{index}",
                "normalized_url": f"https://example.com/recovery-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 3)
        ]
        round_one_query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis institutional flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis funding rates latest", "result_count": 1},
            {"query": "Bitcoin market analysis open interest latest", "result_count": 1},
        ]
        captured_round_two_queries: list[str] = []

        def fake_round(*, round_index, query_plan_override=None, repair_usage=None, **kwargs):
            if round_index == 1:
                self.assertIsNone(query_plan_override)
                self.assertIsNone(repair_usage)
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=round_one_candidates,
                    status=SourceDiscoveryRun.STATUS_PARTIAL_FAILED,
                    provider_result_count=4,
                    provider_error_count=1,
                    accepted_count=4,
                    reason_summary="provider_error",
                    per_query_result_counts=round_one_query_rows,
                )

            self.assertEqual(round_index, 2)
            self.assertIsNotNone(query_plan_override)
            self.assertTrue(repair_usage.get("used_repair_plan"))
            captured_round_two_queries[:] = [item.query for item in query_plan_override.query_items]
            self.assertTrue(captured_round_two_queries)
            self.assertTrue(all(len(query.split()) <= 8 for query in captured_round_two_queries))
            return self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=round_two_candidates,
                accepted_count=2,
                reason_summary="target_reached",
                per_query_result_counts=[
                    {"query": query, "result_count": 1}
                    for query in captured_round_two_queries
                ],
            )

        mock_run_provider_discovery_round.side_effect = fake_round

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 2)
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(response, "Some searches could not be completed. 6 new source suggestions are still available after 2 search rounds.")
        cycle_runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(cycle_runs), 2)
        cycle = cycle_runs[-1].diagnostics.get("discovery_cycle") or {}
        self.assertEqual(cycle.get("decision"), "target_reached")
        self.assertEqual(cycle.get("accumulated_visible_suggestions"), 6)
        self.assertEqual(cycle.get("rounds", [])[0].get("provider_error_count"), 1)
        self.assertIn("provider_partial_error", (cycle.get("rounds", [])[0].get("diagnosis") or {}).get("secondary_causes", []))
        repair_plan = cycle.get("rounds", [])[0].get("repair_plan_for_next_round") or {}
        self.assertIn(
            repair_plan.get("strategy"),
            {"mixed_repair", "recover_failed_search_area"},
        )
        repaired_queries = [item.get("new_query") for item in repair_plan.get("query_repair_plan") or [] if isinstance(item, dict)]
        self.assertTrue(repaired_queries)
        self.assertTrue(all(len(str(query).split()) <= 8 for query in repaired_queries))
        self.assertTrue(cycle.get("rounds", [])[1].get("used_repair_plan"))
        self.assertEqual(
            [item.get("query") for item in (cycle.get("rounds", [])[1].get("repair_plan_usage") or {}).get("repair_queries_used", [])],
            captured_round_two_queries,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_can_run_third_round_with_new_repair_plan_from_round_two(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle third round topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_one_candidates = [
            {
                "title": f"Round one source {index}",
                "url": f"https://example.com/cycle-third-round-1-{index}",
                "normalized_url": f"https://example.com/cycle-third-round-1-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 3)
        ]
        round_two_candidates = [
            {
                "title": f"Round two source {index}",
                "url": f"https://example.com/cycle-third-round-2-{index}",
                "normalized_url": f"https://example.com/cycle-third-round-2-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 3)
        ]
        round_three_candidates = [
            {
                "title": f"Round three source {index}",
                "url": f"https://example.com/cycle-third-round-3-{index}",
                "normalized_url": f"https://example.com/cycle-third-round-3-{index}",
                "source_type": "generic_html",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "default_selected": True,
            }
            for index in range(1, 3)
        ]
        round_one_query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis institutional flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis funding rates latest", "result_count": 1},
            {"query": "Bitcoin market analysis open interest latest", "result_count": 1},
        ]
        captured_round_two_queries: list[str] = []
        captured_round_three_queries: list[str] = []

        def fake_round(*, round_index, query_plan_override=None, repair_usage=None, **kwargs):
            if round_index == 1:
                self.assertIsNone(query_plan_override)
                self.assertIsNone(repair_usage)
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=round_one_candidates,
                    accepted_count=2,
                    reason_summary="mixed_low_yield",
                    per_query_result_counts=round_one_query_rows,
                )
            if round_index == 2:
                self.assertIsNotNone(query_plan_override)
                self.assertEqual((repair_usage or {}).get("repair_plan_source_round"), 1)
                captured_round_two_queries[:] = [item.query for item in query_plan_override.query_items]
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=round_two_candidates,
                    accepted_count=2,
                    reason_summary="mixed_low_yield",
                    per_query_result_counts=[
                        {"query": query, "result_count": 1}
                        for query in captured_round_two_queries
                    ],
                )

            self.assertEqual(round_index, 3)
            self.assertIsNotNone(query_plan_override)
            self.assertEqual((repair_usage or {}).get("repair_plan_source_round"), 2)
            captured_round_three_queries[:] = [item.query for item in query_plan_override.query_items]
            self.assertTrue(captured_round_three_queries)
            self.assertTrue(set(captured_round_three_queries).isdisjoint(set(captured_round_two_queries)))
            return self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=round_three_candidates,
                accepted_count=2,
                reason_summary="target_reached",
                per_query_result_counts=[
                    {"query": query, "result_count": 1}
                    for query in captured_round_three_queries
                ],
            )

        mock_run_provider_discovery_round.side_effect = fake_round

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 3)
        self.assertContains(response, "Source discovery completed")
        self.assertContains(response, "Found 6 new source suggestions after 3 search rounds.")
        latest_cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        self.assertEqual(latest_cycle.get("decision"), "target_reached")
        self.assertEqual(latest_cycle.get("round_count"), 3)
        self.assertEqual(latest_cycle.get("max_immediate_rounds"), 3)
        self.assertEqual(latest_cycle.get("accumulated_visible_suggestions"), 6)
        self.assertEqual((latest_cycle.get("rounds", [])[1].get("repair_plan_usage") or {}).get("repair_plan_source_round"), 1)
        self.assertEqual((latest_cycle.get("rounds", [])[2].get("repair_plan_usage") or {}).get("repair_plan_source_round"), 2)
        self.assertEqual(
            [item.get("query") for item in (latest_cycle.get("rounds", [])[2].get("repair_plan_usage") or {}).get("repair_queries_used", [])],
            captured_round_three_queries,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_stops_at_three_rounds_when_still_below_target(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle max rounds topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        round_one_query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis institutional flows latest", "result_count": 1},
            {"query": "Bitcoin market analysis funding rates latest", "result_count": 1},
            {"query": "Bitcoin market analysis open interest latest", "result_count": 1},
        ]
        captured_round_two_queries: list[str] = []
        captured_round_three_queries: list[str] = []

        def fake_round(*, round_index, query_plan_override=None, repair_usage=None, **kwargs):
            if round_index == 1:
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=[],
                    accepted_count=0,
                    reason_summary="mixed_low_yield",
                    per_query_result_counts=round_one_query_rows,
                )
            if round_index == 2:
                captured_round_two_queries[:] = [item.query for item in query_plan_override.query_items]
                return self._build_fake_discovery_cycle_round_result(
                    topic=topic,
                    new_visible_candidates=[],
                    accepted_count=0,
                    reason_summary="mixed_low_yield",
                    per_query_result_counts=[
                        {"query": query, "result_count": 0}
                        for query in captured_round_two_queries
                    ],
                )

            captured_round_three_queries[:] = [item.query for item in query_plan_override.query_items]
            return self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                accepted_count=0,
                reason_summary="mixed_low_yield",
                per_query_result_counts=[
                    {"query": query, "result_count": 0}
                    for query in captured_round_three_queries
                ],
            )

        mock_run_provider_discovery_round.side_effect = fake_round

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
        self.assertEqual(mock_run_provider_discovery_round.call_count, 3)
        latest_cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        self.assertEqual(latest_cycle.get("decision"), "max_rounds_reached")
        self.assertEqual(latest_cycle.get("round_count"), 3)
        self.assertEqual(latest_cycle.get("max_immediate_rounds"), 3)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_diagnoses_duplicate_heavy_rounds(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle duplicate-heavy topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=10,
                accepted_count=0,
                known_or_duplicate_count=7,
                quality_rejected_count=1,
                reason_summary="duplicate_heavy",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=8,
                accepted_count=0,
                known_or_duplicate_count=5,
                quality_rejected_count=1,
                reason_summary="duplicate_heavy",
            ),
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
        cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        round_diagnosis = cycle.get("rounds", [])[0].get("diagnosis") or {}
        self.assertEqual(round_diagnosis.get("primary_cause"), "duplicate_heavy")
        self.assertEqual(round_diagnosis.get("recommended_next_action"), "pivot_to_new_subangles")
        self.assertEqual((cycle.get("cycle_diagnosis") or {}).get("primary_cause"), "duplicate_heavy")
        repair_plan = cycle.get("rounds", [])[0].get("repair_plan_for_next_round") or {}
        self.assertEqual(repair_plan.get("strategy"), "pivot_exhausted_angle")
        self.assertTrue(repair_plan.get("query_repair_plan"))
        self.assertTrue(all(len(str(item.get("new_query") or "").split()) <= 8 for item in repair_plan.get("query_repair_plan") or []))

    def test_repair_plan_diversifies_duplicate_heavy_queries_across_distinct_surfaces(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="bitcion market",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        diagnosis = {
            "primary_cause": "duplicate_heavy",
            "secondary_causes": ["provider_partial_error"],
            "severity": "high",
            "explanation": "Duplicate-heavy results need a fresh adjacent surface.",
            "recommended_next_action": "pivot_to_new_subangles",
        }
        query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis institutional flows latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis funding rates latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis open interest latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis research paper latest", "status": "useful", "returned_count": 1},
        ]
        repair_plan = _build_discovery_repair_plan(
            topic=topic,
            diagnosis=diagnosis,
            rounds=[
                {
                    "run_id": 1,
                    "round_index": 1,
                    "returned_count": 10,
                    "visible_new_suggestions": 0,
                    "diagnosis": diagnosis,
                    "query_rows": query_rows,
                    "quality_feedback": {
                        "preferred_material_types_found": [
                            {"material_type": "market_data_flow_analysis", "label": "market data / flow analysis", "count": 2},
                            {"material_type": "research_paper", "label": "research paper", "count": 1},
                            {"material_type": "on_chain_analysis", "label": "on-chain analysis", "count": 1},
                        ]
                    },
                }
            ],
        )

        items = repair_plan.get("query_repair_plan") or []
        new_queries = [str(item.get("new_query") or "") for item in items]
        self.assertEqual(len(new_queries), len(set(new_queries)))
        self.assertEqual(
            {
                str(item.get("old_query") or ""): str(item.get("new_query") or "")
                for item in items
            },
            {
                "Bitcoin market analysis ETF flows latest": "Bitcoin ETF flows weekly report",
                "Bitcoin market analysis institutional flows latest": "Bitcoin treasury holdings institutional demand",
                "Bitcoin market analysis funding rates latest": "Bitcoin funding rates open interest report",
                "Bitcoin market analysis open interest latest": "Bitcoin derivatives positioning market structure",
                "Bitcoin market analysis research paper latest": "Bitcoin market structure research paper",
            },
        )
        self.assertTrue(all(len(query.split()) <= 8 for query in new_queries))
        self.assertTrue(all(query != str(item.get("old_query") or "") for query, item in zip(new_queries, items, strict=False)))
        self.assertTrue(all(str(item.get("surface_key") or "").strip() for item in items))

    def test_repair_plan_remains_generation_only_and_does_not_replace_query_rows(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="bitcion market",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        diagnosis = {
            "primary_cause": "mixed_low_yield",
            "secondary_causes": ["duplicate_heavy", "quality_heavy"],
            "severity": "medium",
            "explanation": "Mixed low-yield results need diversified repair planning.",
            "recommended_next_action": "reframe_search_strategy",
        }
        query_rows = [
            {"query": "Bitcoin market analysis ETF flows latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis institutional flows latest", "status": "useful", "returned_count": 1},
            {"query": "Bitcoin market analysis funding rates latest", "status": "useful", "returned_count": 1},
        ]
        repair_plan = _build_discovery_repair_plan(
            topic=topic,
            diagnosis=diagnosis,
            rounds=[
                {
                    "run_id": 1,
                    "round_index": 1,
                    "returned_count": 8,
                    "visible_new_suggestions": 0,
                    "diagnosis": diagnosis,
                    "query_rows": query_rows,
                    "quality_feedback": {
                        "preferred_material_types_found": [
                            {"material_type": "market_data_flow_analysis", "label": "market data / flow analysis", "count": 2},
                            {"material_type": "market_structure_analysis", "label": "market structure analysis", "count": 1},
                        ]
                    },
                }
            ],
        )

        repaired_queries = {str(item.get("new_query") or "") for item in repair_plan.get("query_repair_plan") or []}
        executed_queries = {str(item.get("query") or "") for item in query_rows}
        self.assertTrue(repaired_queries)
        self.assertTrue(repaired_queries.isdisjoint(executed_queries))
        self.assertTrue(
            {
                "avoid_duplicate_repaired_queries",
                "avoid_near_duplicate_repaired_queries",
                "require_query_surface_diversity",
            }.issubset(set((repair_plan.get("constraints") or {}).keys()))
        )

    def test_select_repair_queries_for_next_round_deduplicates_queries(self) -> None:
        selected = _select_repair_queries_for_next_round(
            repair_plan={
                "query_repair_plan": [
                    {
                        "old_query": "Bitcoin market analysis ETF flows latest",
                        "new_query": "Bitcoin ETF flows weekly report",
                        "action": "replace_query",
                        "semantic_shift_type": "adjacent_angle_shift",
                        "material_type": "report",
                        "angle": "ETF flows",
                        "surface_key": "etf_flows_report",
                        "diversity_reason": "Primary ETF surface.",
                        "repair_reason": "Compact ETF repair.",
                    },
                    {
                        "old_query": "Bitcoin market analysis institutional flows latest",
                        "new_query": "Bitcoin ETF flows weekly report",
                        "action": "replace_query",
                        "semantic_shift_type": "adjacent_angle_shift",
                        "material_type": "report",
                        "angle": "institutional flows",
                        "surface_key": "institutional_flows_report",
                        "diversity_reason": "Would duplicate ETF query and should be dropped.",
                        "repair_reason": "Duplicate repair should not survive selection.",
                    },
                    {
                        "old_query": "Bitcoin market analysis funding rates latest",
                        "new_query": "Bitcoin funding rates open interest report",
                        "action": "replace_query",
                        "semantic_shift_type": "adjacent_angle_shift",
                        "material_type": "market structure",
                        "angle": "derivatives / market structure",
                        "surface_key": "funding_open_interest_report",
                        "diversity_reason": "Distinct derivatives surface.",
                        "repair_reason": "Compact derivatives repair.",
                    },
                ]
            },
            prior_rounds=[],
            query_limit=4,
        )
        self.assertEqual(
            [item.get("query") for item in selected[0]],
            [
                "Bitcoin ETF flows weekly report",
                "Bitcoin funding rates open interest report",
            ],
        )
        self.assertIsNone(selected[1])

    def test_select_repair_queries_for_next_round_skips_used_queries_and_surfaces(self) -> None:
        selected, stop_reason = _select_repair_queries_for_next_round(
            repair_plan={
                "query_repair_plan": [
                    {
                        "old_query": "Bitcoin market analysis ETF flows latest",
                        "new_query": "Bitcoin ETF flows weekly report",
                        "action": "replace_query",
                        "semantic_shift_type": "adjacent_angle_shift",
                        "material_type": "report",
                        "angle": "ETF flows",
                        "surface_key": "etf_flows_report",
                        "diversity_reason": "Already used ETF surface.",
                        "repair_reason": "Would repeat a used repair.",
                    }
                ]
            },
            prior_rounds=[
                {
                    "query_rows": [{"query": "Bitcoin ETF flows weekly report"}],
                    "repair_plan_usage": {
                        "repair_queries_used": [
                            {
                                "query": "Bitcoin ETF flows weekly report",
                                "surface_key": "etf_flows_report",
                            }
                        ]
                    },
                }
            ],
            query_limit=4,
        )
        self.assertEqual(selected, [])
        self.assertEqual(stop_reason, "partial_target_not_reached_no_unused_surfaces")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_diagnoses_quality_heavy_rounds(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle quality-heavy topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        weak_material_types = [{"material_type": "beginner_seo_guide", "label": "beginner / SEO guide", "count": 6}]
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=10,
                accepted_count=0,
                quality_rejected_count=7,
                weak_material_types=weak_material_types,
                reason_summary="quality_heavy",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=8,
                accepted_count=0,
                quality_rejected_count=5,
                weak_material_types=weak_material_types,
                reason_summary="quality_heavy",
            ),
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
        cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        round_diagnosis = cycle.get("rounds", [])[0].get("diagnosis") or {}
        self.assertEqual(round_diagnosis.get("primary_cause"), "over_broad_query")
        self.assertEqual(round_diagnosis.get("recommended_next_action"), "narrow_by_material_type")
        self.assertIn((cycle.get("cycle_diagnosis") or {}).get("primary_cause"), {"over_broad_query", "quality_heavy"})
        repair_plan = cycle.get("rounds", [])[0].get("repair_plan_for_next_round") or {}
        self.assertEqual(repair_plan.get("strategy"), "narrow_by_material_type")
        self.assertTrue(any("report" in str(item.get("new_query") or "").casefold() or "paper" in str(item.get("new_query") or "").casefold() for item in repair_plan.get("query_repair_plan") or []))

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_diagnoses_zero_return_as_over_narrow_or_zero_return(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle zero-return topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        zero_query_rows = [
            {"intent": "report", "query": "bitcoin market structure funding rates open interest report", "result_count": 0},
            {"intent": "paper", "query": "bitcoin market structure funding rates open interest research paper", "result_count": 0},
            {"intent": "analysis", "query": "bitcoin market structure funding rates open interest latest analysis", "result_count": 0},
        ]
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=0,
                accepted_count=0,
                quality_rejected_count=0,
                known_or_duplicate_count=0,
                per_query_result_counts=zero_query_rows,
                reason_summary="zero_visible",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=0,
                accepted_count=0,
                quality_rejected_count=0,
                known_or_duplicate_count=0,
                per_query_result_counts=zero_query_rows,
                reason_summary="zero_visible",
            ),
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
        cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        round_diagnosis = cycle.get("rounds", [])[0].get("diagnosis") or {}
        self.assertIn(round_diagnosis.get("primary_cause"), {"zero_return", "over_narrow_query"})
        self.assertEqual(round_diagnosis.get("recommended_next_action"), "broaden_query")
        repair_plan = cycle.get("rounds", [])[0].get("repair_plan_for_next_round") or {}
        self.assertEqual(repair_plan.get("strategy"), "adjacent_scope_shift")
        self.assertTrue(all(len(str(item.get("new_query") or "").split()) <= 8 for item in repair_plan.get("query_repair_plan") or []))

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_diagnoses_stale_heavy_rounds(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle stale-heavy topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        stale_reasons = [{"reason": "stale source outside recency window", "count": 5}]
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=7,
                accepted_count=0,
                quality_rejected_count=5,
                dominant_rejection_reasons=stale_reasons,
                reason_summary="stale_heavy",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=6,
                accepted_count=0,
                quality_rejected_count=4,
                dominant_rejection_reasons=stale_reasons,
                reason_summary="stale_heavy",
            ),
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
        cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        round_diagnosis = cycle.get("rounds", [])[0].get("diagnosis") or {}
        self.assertEqual(round_diagnosis.get("primary_cause"), "stale_heavy")
        self.assertEqual(round_diagnosis.get("recommended_next_action"), "tighten_recency_or_use_current_terms")
        repair_plan = cycle.get("rounds", [])[0].get("repair_plan_for_next_round") or {}
        self.assertEqual(repair_plan.get("strategy"), "tighten_recency_or_current_terms")
        self.assertTrue(any(str(timezone.now().year) in str(item.get("new_query") or "") for item in repair_plan.get("query_repair_plan") or []))

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("apps.digests.views._run_provider_discovery_round")
    def test_discovery_cycle_aggregates_mixed_failure_diagnoses_into_copy_report(
        self,
        mock_run_provider_discovery_round,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle mixed diagnosis topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["bitcoin market"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        mock_run_provider_discovery_round.side_effect = [
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=10,
                accepted_count=0,
                known_or_duplicate_count=6,
                quality_rejected_count=1,
                reason_summary="duplicate_heavy",
            ),
            self._build_fake_discovery_cycle_round_result(
                topic=topic,
                new_visible_candidates=[],
                provider_result_count=9,
                accepted_count=0,
                quality_rejected_count=6,
                weak_material_types=[{"material_type": "beginner_seo_guide", "label": "beginner / SEO guide", "count": 6}],
                reason_summary="quality_heavy",
            ),
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
        history_response = self.client.get(reverse("topic-research-history", args=[topic.id]))
        self.assertContains(history_response, "Search diagnosis")
        self.assertContains(history_response, "Strategy repair")
        copy_report = history_response.context["full_history_copy_report"]
        self.assertIn("Search diagnosis", copy_report)
        self.assertIn("Strategy repair", copy_report)
        self.assertIn("primary cause:", copy_report)
        self.assertIn("secondary causes:", copy_report)
        self.assertIn("recommended next action:", copy_report)
        self.assertIn("query repair plan:", copy_report)
        cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        cycle_diagnosis = cycle.get("cycle_diagnosis") or {}
        self.assertIn(cycle_diagnosis.get("primary_cause"), {"duplicate_heavy", "quality_heavy", "over_broad_query", "mixed_low_yield"})
        self.assertTrue({"duplicate_heavy", "quality_heavy"}.intersection(set(cycle_diagnosis.get("secondary_causes") or [])) or cycle_diagnosis.get("primary_cause") in {"duplicate_heavy", "quality_heavy"})

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_records_quality_rejected_history_row(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation consulting services",
                    "link": "https://example.com/services/ai-automation",
                    "snippet": "Book a demo with our platform and contact sales to get started.",
                    "source": "Example",
                }
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "AI automation consulting services")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 0)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_COMMERCIAL_REJECTED)
        self.assertTrue(
            any(
                phrase in history_item.quality_rejection_reason.lower()
                for phrase in ("commercial", "product/demo/pricing")
            )
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_records_stale_history_row(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "2018 automation adoption report with methodology and limitations",
                    "link": "https://example.com/research/automation-report-2018",
                    "snippet": "2018 report with survey data, methodology, findings, and limitations.",
                    "source": "Example",
                    "date": "2018-04-03",
                }
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "2018 automation adoption report with methodology and limitations")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 0)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_STALE_REJECTED)
        self.assertEqual(history_item.freshness_status, "very_stale")
        self.assertEqual(history_item.detected_publication_year, 2018)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_excludes_duplicate_manual_url(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Manual duplicate topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.com/ai-guide",
            normalized_url="https://example.com/ai-guide",
            source_type="generic_html",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation guide",
                    "link": "https://example.com/ai-guide",
                    "snippet": "Practical guide for AI automation workflows.",
                    "source": "Example",
                }
            ],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.HYBRID,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.MANUAL).count(), 1)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 0)
        self.assertNotContains(response, "AI automation guide")
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url="https://example.com/ai-guide")
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SEEN)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN)
        self.assertFalse(history_item.created_topic_source)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_increments_history_seen_count_for_repeat_url(
        self,
        mock_urlopen,
    ) -> None:
        organic_results = [
            {
                "position": 1,
                "title": "Automation case study with implementation details",
                "link": "https://example.com/research/automation-case-study",
                "snippet": "Recent case study with lessons learned, implementation details, and tradeoffs.",
                "source": "Example",
            }
        ]
        self._mock_serpapi_urlopen(mock_urlopen, organic_results)
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        first_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )
        self.assertEqual(first_response.status_code, 200)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        first_seen_at = history_item.last_seen_at

        second_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(second_response.status_code, 200)
        self.assertEqual(SourceDiscoveryHistory.objects.filter(topic=topic).count(), 1)
        runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertEqual(len(runs), 6)
        self.assertTrue(all(((run.diagnostics.get("discovery_cycle") or {}).get("round_count") == 3) for run in runs))
        history_item.refresh_from_db()
        self.assertEqual(history_item.seen_count, 6)
        self.assertGreater(history_item.last_seen_at, first_seen_at)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_pinning_discovered_source_updates_history_status_to_kept(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Automation case study with implementation details",
                    "link": "https://example.com/research/automation-case-study",
                    "snippet": "Recent case study with lessons learned, implementation details, and tradeoffs.",
                    "source": "Example",
                }
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )
        source = topic.sources.get(origin=TopicSourceOrigin.DISCOVERED)

        response = self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        source.refresh_from_db()
        self.assertTrue(source.is_pinned)
        self.assertTrue(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)
        self.assertEqual(history_item.topic_source_id, source.id)

    def test_keep_after_remove_restores_kept_source_and_checked_state(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Keep after remove topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Recoverable suggestion",
            url="https://example.org/recoverable-suggestion",
            normalized_url="https://example.org/recoverable-suggestion",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )

        response = self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        source.refresh_from_db()
        self.assertTrue(source.is_pinned)
        self.assertTrue(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)
        workspace_response = self.client.get(reverse("topic-workspace", args=[topic.id]))
        html = workspace_response.content.decode("utf-8")
        kept_section = html.split("Kept sources", 1)[1].split("New suggestions", 1)[0]
        new_section = html.rsplit("New suggestions", 1)[1]
        self.assertIn("Recoverable suggestion", kept_section)
        self.assertNotIn("Recoverable suggestion", new_section)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_removing_discovered_source_returns_it_to_new_suggestions_unchecked(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Automation case study with implementation details",
                    "link": "https://example.com/research/automation-case-study",
                    "snippet": "Recent case study with lessons learned, implementation details, and tradeoffs.",
                    "source": "Example",
                }
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )
        source = topic.sources.get(origin=TopicSourceOrigin.DISCOVERED)
        self.client.post(reverse("pin-topic-source", args=[topic.id, source.id]))

        response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 200)
        source.refresh_from_db()
        self.assertFalse(source.is_pinned)
        self.assertFalse(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        self.assertEqual(history_item.topic_source_id, source.id)
        self.assertContains(response, "New suggestions")
        self.assertNotContains(response, "Kept sources · 1")
        html = response.content.decode("utf-8")
        new_section = html.rsplit("New suggestions", 1)[1]
        self.assertIn("Automation case study with implementation details", new_section)
        self.assertNotIn('checked', new_section.split('value="1"', 1)[1].split('>', 1)[0])

    def test_remove_from_kept_does_not_create_removed_history_state(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Remove keeps shown history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Kept source",
            url="https://example.org/kept-source",
            normalized_url="https://example.org/kept-source",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 200)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        self.assertNotEqual(history_item.status, SourceDiscoveryHistory.STATUS_REMOVED_BY_USER)
        removed_response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "removed"})
        self.assertNotContains(removed_response, "Kept source")

    def test_remove_from_shown_suggestion_keeps_source_in_new_suggestions_unchecked(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Remove shown suggestion topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Shown suggestion",
            url="https://example.org/shown-suggestion",
            normalized_url="https://example.org/shown-suggestion",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )

        response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 200)
        source.refresh_from_db()
        self.assertFalse(source.is_pinned)
        self.assertFalse(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        html = response.content.decode("utf-8")
        new_section = html.rsplit("New suggestions", 1)[1]
        self.assertIn("Shown suggestion", new_section)
        self.assertNotIn('checked', new_section.split('value="1"', 1)[1].split('>', 1)[0])

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_caps_visible_new_suggestions_and_shows_truncation_summary(
        self,
        mock_urlopen,
    ) -> None:
        organic_results = []
        for index in range(14):
            organic_results.append(
                {
                    "position": index + 1,
                    "title": f"Automation source {index + 1}",
                    "link": f"https://example.com/automation-{index + 1}",
                    "snippet": f"Automation workflow result {index + 1} for research discovery.",
                    "source": "Example",
                }
            )
        self._mock_serpapi_urlopen(mock_urlopen, organic_results)
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Automation discovery cap",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Source discovery completed")
        self.assertContains(response, "Found 14 new source suggestions.")
        self.assertContains(response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertContains(response, "Showing the first 12 suggestions. Refine the research focus to narrow results.")
        self.assertContains(response, "Show all suggestions")
        self.assertContains(response, "New suggestions · 14")
        html = response.content.decode("utf-8")
        new_section = html.split("New suggestions · 14", 1)[1].split("Ready to generate", 1)[0]
        self.assertEqual(new_section.count(">Keep</button>"), 12)
        topic.refresh_from_db()
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 14)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_can_expand_to_show_all_suggestions(
        self,
        mock_urlopen,
    ) -> None:
        organic_results = []
        for index in range(14):
            organic_results.append(
                {
                    "position": index + 1,
                    "title": f"Automation source {index + 1}",
                    "link": f"https://example.com/expand-{index + 1}",
                    "snippet": f"Automation workflow result {index + 1} for research discovery.",
                    "source": "Example",
                }
            )
        self._mock_serpapi_urlopen(mock_urlopen, organic_results)
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Expanded discovery cap",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )
        topic.refresh_from_db()

        response = self.client.get(
            reverse("topic-workspace", args=[topic.id]),
            data={"discovery_context": "1", "show_all_suggestions": "1"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Source discovery results")
        self.assertContains(response, "14 research suggestions available")
        self.assertNotContains(response, "Showing the first 12 suggestions. Refine the research focus to narrow results.")
        self.assertNotContains(response, "Show all suggestions")
        html = response.content.decode("utf-8")
        new_section = html.split("New suggestions · 14", 1)[1].split("Ready to generate", 1)[0]
        self.assertEqual(new_section.count(">Keep</button>"), 14)
        self.assertEqual(new_section.count('name="is_active"'), 14)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_excludes_duplicate_pinned_research_url_and_preserves_pin(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Pinned duplicate topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        pinned_source = TopicSource.objects.create(
            topic=topic,
            name="Pinned research source",
            url="https://example.com/ai-guide",
            normalized_url="https://example.com/ai-guide",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation guide",
                    "link": "https://example.com/ai-guide",
                    "snippet": "Practical guide for AI automation workflows.",
                    "source": "Example",
                }
            ],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)
        pinned_source.refresh_from_db()
        self.assertTrue(pinned_source.is_pinned)
        html = response.content.decode("utf-8")
        kept_section = html.split("Kept sources", 1)[1].split("New suggestions", 1)[0]
        new_section = html.rsplit("New suggestions", 1)[1]
        self.assertIn("Pinned research source", kept_section)
        self.assertNotIn("AI automation guide", new_section)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_refresh_prunes_only_inactive_unpinned_discovered_sources(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Refresh pruning topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        active_unpinned_source = TopicSource.objects.create(
            topic=topic,
            name="Active source",
            url="https://example.com/stale",
            normalized_url="https://example.com/stale",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        inactive_unpinned_source = TopicSource.objects.create(
            topic=topic,
            name="Inactive source",
            url="https://example.com/inactive",
            normalized_url="https://example.com/inactive",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        )
        pinned_source = TopicSource.objects.create(
            topic=topic,
            name="Pinned source",
            url="https://example.com/pinned",
            normalized_url="https://example.com/pinned",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Fresh automation source",
                    "link": "https://example.com/fresh",
                    "snippet": "Fresh provider-backed automation research result.",
                    "source": "Example",
                }
            ],
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(TopicSource.objects.filter(pk=active_unpinned_source.id, is_active=True).exists())
        self.assertFalse(TopicSource.objects.filter(pk=inactive_unpinned_source.id).exists())
        self.assertTrue(TopicSource.objects.filter(pk=pinned_source.id).exists())
        self.assertTrue(
            TopicSource.objects.filter(
                topic=topic,
                origin=TopicSourceOrigin.DISCOVERED,
                url="https://example.com/fresh",
            ).exists()
        )
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "Found 1 new source suggestion after 3 search rounds. DigestFlow could not reach the 6-source target with the current search strategy.",
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_keep_action_after_discovery_preserves_discovery_results_context(
        self,
        mock_urlopen,
    ) -> None:
        organic_results = []
        for index in range(14):
            organic_results.append(
                {
                    "position": index + 1,
                    "title": f"Automation source {index + 1}",
                    "link": f"https://example.com/keep-{index + 1}",
                    "snippet": f"Automation workflow result {index + 1} for research discovery.",
                    "source": "Example",
                }
            )
        self._mock_serpapi_urlopen(mock_urlopen, organic_results)
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Keep after discovery",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
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
                "run_research": "1",
            },
        )
        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        source_to_keep = topic.sources.filter(
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
        ).order_by("id").first()
        self.assertIsNotNone(source_to_keep)

        response = self.client.post(
            reverse("pin-topic-source", args=[topic.id, source_to_keep.id]),
            data={"discovery_context": "1"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        source_to_keep.refresh_from_db()
        self.assertTrue(source_to_keep.is_pinned)
        self.assertContains(response, "Source discovery results")
        self.assertContains(response, "12 of 13 suggestions shown")
        self.assertContains(response, "Show all suggestions")
        self.assertContains(response, "Kept sources · 1")
        self.assertContains(response, "New suggestions · 13")
        html = response.content.decode("utf-8")
        kept_section = html.split("Kept sources · 1", 1)[1].split("New suggestions · 13", 1)[0]
        new_section = html.split("New suggestions · 13", 1)[1].split("Ready to generate", 1)[0]
        self.assertIn(source_to_keep.name, kept_section)
        self.assertNotIn(
            reverse("pin-topic-source", args=[topic.id, source_to_keep.id]),
            new_section,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_blocked_real_provider_find_sources_does_not_fallback_to_template_suggestions(
        self,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI agents",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI agents", "agent workflows"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
        self.assertEqual(topic.sources.count(), 0)
        self.assertContains(response, "Source search is temporarily unavailable")
        self.assertContains(response, "DigestFlow could not connect to the search provider. Please try again later.")
        self.assertContains(response, "Research is currently disabled")
        self.assertNotContains(response, "DEV Community / #ai")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_blocked_provider_keeps_existing_new_suggestions(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Blocked provider keeps suggestions",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI agents", "agent workflows"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing discovered source",
            url="https://example.com/existing-blocked",
            normalized_url="https://example.com/existing-blocked",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

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
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.pk).exists())
        self.assertContains(response, "Source search is temporarily unavailable")
        self.assertContains(response, "Existing suggestions were kept.")
        self.assertContains(response, "Existing discovered source")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_failure_does_not_crash_or_create_unexpected_topic_sources(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Provider failure topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing discovered source",
            url="https://example.com/existing",
            normalized_url="https://example.com/existing",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        mock_urlopen.side_effect = HTTPError(
            url="https://serpapi.com/search.json",
            code=503,
            msg="service unavailable",
            hdrs=None,
            fp=None,
        )

        response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.id).exists())
        self.assertEqual(topic.sources.count(), 1)
        self.assertContains(response, "Existing discovered source")
        self.assertContains(response, "Source search is temporarily unavailable")
        self.assertContains(response, "DigestFlow could not connect to the search provider.")
        self.assertContains(response, "Existing suggestions were kept.")
        self.assertContains(response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertNotContains(response, "Source discovery details")
        self.assertNotContains(response, "Provider filter")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_with_empty_results_runs_cycle_and_reports_target_not_reached(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(mock_urlopen, [])
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty provider results",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "DigestFlow could not reach the 6-source target after 2 search rounds with the current search strategy.",
        )
        self.assertContains(response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertNotContains(response, "Source discovery details")
        self.assertNotContains(response, "Provider filter")
        self.assertNotContains(response, "Source discovery completed")
        topic = Topic.objects.get(name="Empty provider results")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 0)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.MANUAL).count(), 0)
        self.assertEqual(SourceDiscoveryRun.objects.filter(topic=topic).count(), 2)
        final_cycle = SourceDiscoveryRun.objects.filter(topic=topic).order_by("id").last().diagnostics.get("discovery_cycle") or {}
        self.assertEqual(final_cycle.get("decision"), "partial_target_not_reached_no_usable_repair_queries")
        self.assertEqual(final_cycle.get("round_count"), 2)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_empty_refresh_keeps_existing_new_suggestions(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(mock_urlopen, [])
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty refresh keeps suggestions",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing suggestion",
            url="https://example.com/existing-empty",
            normalized_url="https://example.com/existing-empty",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

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
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.pk).exists())
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "DigestFlow could not reach the 6-source target after 2 search rounds with the current search strategy.",
        )
        self.assertContains(response, "Existing suggestion")
        self.assertContains(response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertNotContains(response, "Source discovery details")
        self.assertNotContains(response, "Provider filter")
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    @patch("services.sources.research_queries.create_content_research_plan")
    def test_repeated_discovery_rotates_query_angle_across_cycle_rounds_and_clicks(
        self,
        mock_create_content_research_plan,
        mock_urlopen,
    ) -> None:
        mock_create_content_research_plan.return_value = self._forced_fallback_planner_result()
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Repeated discovery angle rotation",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation workflows"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing suggestion",
            url="https://example.com/existing-rotation",
            normalized_url="https://example.com/existing-rotation",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=existing_source,
            normalized_url="https://example.com/existing-rotation",
            url="https://example.com/existing-rotation",
            title="Existing suggestion",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
            first_seen_at=timezone.now(),
            last_seen_at=timezone.now(),
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Existing suggestion",
                    "link": "https://example.com/existing-rotation",
                    "snippet": "Recent survey data, methodology, findings, and implementation lessons.",
                    "source": "Example",
                    "date": "May 21, 2026",
                }
            ],
        )

        first_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )
        second_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertGreaterEqual(mock_urlopen.call_count, 16)
        self.assertLessEqual(mock_urlopen.call_count, 24)
        runs = list(SourceDiscoveryRun.objects.filter(topic=topic).order_by("id"))
        self.assertGreaterEqual(len(runs), 4)
        self.assertLessEqual(len(runs), 6)
        self.assertTrue(all((run.query_count or 0) > 0 for run in runs))
        self.assertTrue(all((run.query_count or 0) <= 4 for run in runs))
        self.assertEqual(runs[0].diagnostics.get("selected_query_angle_key"), "base")
        self.assertNotEqual(runs[0].diagnostics.get("selected_query_angle_key"), runs[1].diagnostics.get("selected_query_angle_key"))
        all_query_sets = [
            tuple(item.get("query") for item in run.diagnostics.get("per_query_result_counts", []))
            for run in runs
        ]
        self.assertGreater(len(set(all_query_sets)), 1)
        self.assertTrue(all(((run.diagnostics.get("discovery_cycle") or {}).get("round_count") or 0) <= 3 for run in runs))
        for run in runs:
            cycle = run.diagnostics.get("discovery_cycle") or {}
            self.assertLessEqual(int(cycle.get("max_immediate_rounds") or 0), 3)
            repair_usage = {}
            if int(cycle.get("round_index") or 0) > 1:
                round_items = cycle.get("rounds") or []
                for item in round_items:
                    if isinstance(item, dict) and int(item.get("round_index") or 0) == int(cycle.get("round_index") or 0):
                        repair_usage = item.get("repair_plan_usage") if isinstance(item.get("repair_plan_usage"), dict) else {}
                        break
            used_surface_keys = [
                str(item.get("surface_key") or "").strip()
                for item in (repair_usage.get("repair_queries_used") or [])
                if isinstance(item, dict) and str(item.get("surface_key") or "").strip()
            ]
            self.assertEqual(len(used_surface_keys), len(set(used_surface_keys)))
        self.assertContains(second_response, "Source discovery partially completed")
        self.assertContains(
            second_response,
            "DigestFlow could not reach the 6-source target after 3 search rounds with the current search strategy.",
        )
        self.assertContains(second_response, f'href="{reverse("topic-research-history", args=[topic.id])}"', html=False)
        self.assertNotContains(second_response, "Source discovery details")
        self.assertNotContains(second_response, "research report")
        self.assertNotContains(second_response, "Previous discovery runs")
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.pk).exists())

    def test_research_history_page_renders_human_readable_run_summary_and_query_details(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Research history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=4,
            provider_result_count=3,
            accepted_count=1,
            rejected_count=2,
            new_suggestions_count=1,
            already_known_count=1,
            diagnostics={
                "selected_query_angle_key": "research_report",
                "selected_query_angle_suffix": "research report",
                "selected_query_angle_reason": "Rotate toward research reports and evidence summaries.",
                "previous_discovery_run_count": 1,
                "duplicate_url_count": 2,
                "query_performance": [
                    {
                        "query": "Automation research report implementation guide",
                        "provider": "serpapi",
                        "angle": "research report",
                        "purpose": "Look for hands-on implementation guidance.",
                        "returned_count": 1,
                        "accepted_count": 1,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 1,
                        "status": "useful",
                    },
                    {
                        "query": "Automation research report case study",
                        "provider": "serpapi",
                        "angle": "research report",
                        "purpose": "Look for concrete examples and outcome-focused writeups.",
                        "returned_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 0,
                        "status": "no_visible_results",
                    },
                ],
                "source_quality_feedback": {
                    "quality_rejected_count": 4,
                    "known_or_duplicate_count": 2,
                    "shown_count": 1,
                    "dominant_rejection_reasons": [
                        {"reason": "not enough substantive signals", "count": 3},
                    ],
                    "weak_domains": [
                        {"domain": "quora.com", "count": 2, "reason": "social/profile/forum"},
                    ],
                    "weak_material_types": [
                        {"material_type": "beginner_seo_guide", "label": "beginner / SEO guide", "count": 3},
                        {"material_type": "price_prediction_live_price", "label": "price prediction / live price", "count": 1},
                    ],
                    "preferred_material_types_found": [
                        {"material_type": "market_data_flow_analysis", "label": "market data / flow analysis", "count": 1},
                    ],
                    "main_quality_issue": "beginner / SEO guide results dominate recent rejected candidates",
                    "planner_quality_guidance": [
                        "Broad beginner or SEO-style guide phrasing is producing weak pages. Avoid 'for beginners', 'ultimate guide', or generic strategy phrasing.",
                        "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest.",
                    ],
                },
                "query_history_summary": {
                    "history_available": True,
                    "recent_run_count": 2,
                    "malformed_run_count": 0,
                    "total_query_rows": 6,
                    "quality_guidance": [
                        "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest.",
                        "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest.",
                    ],
                    "planning_guidance": [
                        "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest.",
                        "Preferred material types already found include: market data / flow analysis. Lean harder into those patterns in the next run."
                    ],
                    "useful_angles": [{"angle": "market structure", "count": 1}],
                    "weak_angles": [{"angle": "retail behavior", "count": 1}],
                },
                "discovery_cycle": {
                    "cycle_id": "cycle-123",
                    "target_visible_new_suggestions": 6,
                    "max_immediate_rounds": 3,
                    "round_index": 2,
                    "round_count": 2,
                    "accumulated_visible_suggestions": 4,
                    "decision": "partial_target_not_reached",
                    "cycle_diagnosis": {
                        "primary_cause": "duplicate_heavy",
                        "secondary_causes": ["provider_partial_error", "quality_heavy"],
                        "severity": "high",
                        "explanation": "Most returned URLs were already known or rejected, and some provider queries failed.",
                        "recommended_next_action": "pivot_to_new_subangles",
                    },
                    "repair_plan": {
                        "strategy": "pivot_exhausted_angle",
                        "reason": "The previous round underperformed because duplicate-heavy results dominated the visible output.",
                        "constraints": {
                            "avoid_repeating_queries": True,
                            "avoid_verbatim_failed_queries": True,
                            "avoid_duplicate_repaired_queries": True,
                            "avoid_near_duplicate_repaired_queries": True,
                            "avoid_long_natural_language_queries": True,
                            "prefer_compact_search_grade_queries": True,
                            "require_semantic_distance_from_failed_query": True,
                            "require_query_surface_diversity": True,
                            "prefer_material_types": ["market data / flow analysis", "research paper"],
                            "avoid_material_types": ["beginner / SEO guide"],
                            "avoid_domains": ["quora.com"],
                        },
                        "query_repair_plan": [
                            {
                                "old_query": "Automation research report implementation guide",
                                "action": "replace_query",
                                "semantic_shift_type": "material_type_shift",
                                "repair_reason": "Shifted the duplicate-heavy report query toward a more compact evidence surface.",
                                "new_query": "Automation market structure report",
                                "angle": "market structure",
                                "material_type": "report",
                                "surface_key": "market_structure_report",
                                "diversity_reason": "Use a more compact adjacent market-structure surface.",
                            }
                        ],
                    },
                    "rounds": [
                        {
                            "run_id": 123,
                            "round_index": 1,
                            "visible_new_suggestions": 0,
                            "accepted_count": 0,
                            "quality_rejected_count": 3,
                            "known_or_duplicate_count": 2,
                            "provider_error_count": 1,
                            "returned_count": 5,
                            "reason_summary": "provider_error",
                            "diagnosis": {
                                "primary_cause": "provider_partial_error",
                                "secondary_causes": ["quality_heavy"],
                                "severity": "high",
                                "explanation": "Primary cause: Provider partial errors. 3 results were rejected by quality filters; 1 provider query failed.",
                                "recommended_next_action": "retry_or_rephrase_failed_queries",
                            },
                            "repair_plan_for_next_round": {
                                "strategy": "recover_failed_search_area",
                                "reason": "Some provider queries failed, so the next round should recover the failed search area with compact queries.",
                                "query_repair_plan": [
                                    {
                                        "old_query": "Automation research report implementation guide",
                                        "action": "replace_query",
                                        "semantic_shift_type": "query_compression",
                                        "repair_reason": "Preserve the failed search area, but change to a compact report-style query.",
                                        "new_query": "Automation market structure report",
                                        "angle": "market structure",
                                        "material_type": "report",
                                        "surface_key": "market_structure_report",
                                        "diversity_reason": "Use a more compact adjacent market-structure surface.",
                                    }
                                ],
                                "constraints": {},
                            },
                            "used_repair_plan": False,
                            "repair_plan_usage": {},
                        },
                        {
                            "run_id": 124,
                            "round_index": 2,
                            "visible_new_suggestions": 4,
                            "accepted_count": 1,
                            "quality_rejected_count": 1,
                            "known_or_duplicate_count": 0,
                            "provider_error_count": 0,
                            "returned_count": 4,
                            "reason_summary": "partial_target_not_reached",
                            "diagnosis": {
                                "primary_cause": "duplicate_heavy",
                                "secondary_causes": ["quality_heavy"],
                                "severity": "medium",
                                "explanation": "Primary cause: Duplicate-heavy results. 1 result was rejected by quality filters.",
                                "recommended_next_action": "pivot_to_new_subangles",
                            },
                            "repair_plan_for_next_round": {
                                "strategy": "pivot_exhausted_angle",
                                "reason": "Avoid the exhausted duplicate-heavy angle and pivot to a fresh adjacent surface.",
                                "query_repair_plan": [
                                    {
                                        "old_query": "Automation research report case study",
                                        "action": "replace_query",
                                        "semantic_shift_type": "adjacent_angle_shift",
                                        "repair_reason": "Pivoted the exhausted angle to a stronger adjacent evidence layer.",
                                        "new_query": "Automation ETF flows report",
                                        "angle": "adjacent angle",
                                        "material_type": "report",
                                        "surface_key": "etf_flows_report",
                                        "diversity_reason": "Choose a new ETF surface for the next round.",
                                    }
                                ],
                                "constraints": {},
                            },
                            "used_repair_plan": True,
                            "repair_plan_usage": {
                                "used_repair_plan": True,
                                "repair_plan_source_round": 1,
                                "strategy": "recover_failed_search_area",
                                "queries_used_count": 1,
                                "repair_queries_used": [
                                    {
                                        "query": "Automation market structure report",
                                        "old_query": "Automation research report implementation guide",
                                        "action": "replace_query",
                                        "semantic_shift_type": "query_compression",
                                        "material_type": "report",
                                    }
                                ],
                            },
                        },
                    ],
                },
                "per_query_result_counts": [
                    {
                        "intent": "implementation_guide",
                        "query": "Automation research report implementation guide",
                        "result_count": 1,
                    },
                    {
                        "intent": "case_study",
                        "query": "Automation research report case study",
                        "result_count": 0,
                    },
                ],
            },
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research history")
        self.assertContains(
            response,
            "Understand how this topic’s source research evolved over time. See the current research state, past discovery runs, and all seen sources.",
        )
        self.assertContains(response, "Current research state")
        self.assertContains(response, "Query performance")
        self.assertContains(response, "Source quality feedback")
        self.assertContains(response, "Copy full history")
        self.assertContains(response, 'id="copy-full-history-button"', html=False)
        self.assertContains(response, 'id="full-history-copy-payload"', html=False)
        copy_report = response.context["full_history_copy_report"]
        self.assertIn("Topic", copy_report)
        self.assertIn("Current research state", copy_report)
        self.assertIn("Query performance", copy_report)
        self.assertIn("Source quality feedback", copy_report)
        self.assertIn("Discovery runs", copy_report)
        self.assertIn("Seen sources", copy_report)
        self.assertIn("Planner history guidance", copy_report)
        self.assertIn("Discovery cycle", copy_report)
        self.assertIn("weak material types", copy_report.casefold())
        self.assertIn("planner quality guidance", copy_report.casefold())
        self.assertIn("quality guidance used for next run", copy_report.casefold())
        self.assertIn("target visible suggestions: 6", copy_report)
        self.assertIn("max immediate rounds: 3", copy_report)
        self.assertIn("stop reason: partial_target_not_reached", copy_report)
        self.assertIn("Round 1", copy_report)
        self.assertIn("ETF flows", copy_report)
        self.assertIn("Search diagnosis", copy_report)
        self.assertIn("Strategy repair", copy_report)
        self.assertIn("Repair plan used", copy_report)
        self.assertIn("query repair plan:", copy_report)
        self.assertIn("source round:", copy_report)
        self.assertIn("queries used:", copy_report)
        self.assertIn("primary cause: duplicate_heavy", copy_report)
        self.assertIn("recommended next action: pivot_to_new_subangles", copy_report)
        self.assertEqual(
            copy_report.count(
                "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest."
            ),
            2,
        )
        self.assertNotContains(response, copy_report)
        self.assertContains(response, "Discovery completed")
        self.assertNotContains(response, '<h2 class="history-run__title">completed</h2>', html=False)
        self.assertContains(response, "Discovery runs")
        self.assertContains(response, "Seen sources")
        self.assertContains(response, "See which queries were used, what they returned, and which search directions looked useful or weak.")
        self.assertContains(response, "Purpose / angle")
        self.assertContains(
            response,
            "Audit trail of past source-finding runs. You usually only need this when checking why discovery behaved a certain way.",
        )
        self.assertContains(response, "serpapi")
        self.assertContains(response, "Useful")
        self.assertContains(response, "No visible results")
        self.assertContains(response, "research report")
        self.assertContains(response, "3 URLs returned")
        self.assertContains(response, "1 visible new suggestions")
        self.assertContains(response, "completed run")
        self.assertNotContains(response, "1 passed, 2 rejected")
        self.assertNotContains(response, "3 known/duplicate")
        self.assertContains(response, "4 new source suggestions were found after 2 search rounds.")
        self.assertContains(response, "last 1 month")
        self.assertContains(response, "Last discovery run status")
        self.assertContains(response, "Last discovery cycle: partial target not reached (4 of 6 visible suggestions). — duplicate-heavy results.")
        self.assertContains(response, "Stage diagnostics")
        self.assertContains(
            response,
            "These counts describe different pipeline checks and may overlap; they are not an additive breakdown.",
        )
        self.assertContains(response, "Passed filtering")
        self.assertContains(response, "Rejected by filters")
        self.assertContains(response, "Already known or duplicate")
        self.assertContains(response, "Main issue:")
        self.assertContains(response, "beginner / SEO guide results dominate recent rejected candidates")
        self.assertContains(response, "Weak material types")
        self.assertContains(response, "Preferred material types found")
        self.assertContains(response, "Quality diagnostics")
        self.assertContains(response, "Discovery cycle")
        self.assertContains(response, "Cycle round:")
        self.assertContains(response, "2 of 2")
        self.assertContains(response, "Cycle target:")
        self.assertContains(response, "Max immediate rounds:")
        self.assertContains(response, "Rounds run:")
        self.assertContains(response, "Accumulated visible suggestions:")
        self.assertContains(response, "Cycle decision:")
        self.assertContains(response, "Partial target not reached")
        self.assertContains(response, "Search diagnosis")
        self.assertContains(response, "Primary cause:")
        self.assertContains(response, "Duplicate-heavy results")
        self.assertContains(response, "Recommended next action:")
        self.assertContains(response, "Pivot to new sub-angles")
        self.assertContains(response, "Strategy repair")
        self.assertContains(response, "Repair plan used")
        self.assertContains(response, "Strategy:")
        self.assertContains(response, "Changed queries:")
        self.assertContains(response, "Source round:")
        self.assertContains(response, "Queries used:")
        self.assertNotContains(response, "https://quora.com/answer")
        self.assertNotContains(response, "Visible new suggestions from this run")
        self.assertNotContains(response, "URLs returned by provider")
        self.assertContains(response, "<details class=\"history-run\">", html=False)
        self.assertContains(response, "<summary class=\"history-run__summary\">", html=False)
        self.assertContains(response, "Query details")
        self.assertContains(response, "implementation guide")
        self.assertContains(response, "Automation research report implementation guide")
        self.assertContains(response, "case study")
        self.assertContains(response, "Technical details")
        self.assertContains(response, "qdr:m")
        self.assertContains(response, "Rotate toward research reports and evidence summaries.")
        html = response.content.decode("utf-8")
        current_state_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Current research state</h2>'
        query_performance_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Query performance</h2>'
        source_quality_feedback_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Source quality feedback</h2>'
        seen_sources_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Seen sources</h2>'
        discovery_runs_header = '<h2 style="margin: 0 0 6px; font-size: 20px;">Discovery runs</h2>'
        self.assertLess(html.index(current_state_header), html.index(query_performance_header))
        self.assertLess(html.index(query_performance_header), html.index(source_quality_feedback_header))
        self.assertLess(html.index(source_quality_feedback_header), html.index(seen_sources_header))
        self.assertLess(html.index(seen_sources_header), html.index(discovery_runs_header))

    def test_current_research_state_uses_cycle_total_feedback_when_cycle_diagnostics_exist(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Cycle total feedback topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=2,
            provider_result_count=2,
            accepted_count=2,
            rejected_count=0,
            new_suggestions_count=2,
            already_known_count=0,
            diagnostics={
                "discovery_cycle": {
                    "target_visible_suggestions": 6,
                    "target_visible_new_suggestions": 6,
                    "max_immediate_rounds": 3,
                    "rounds_run": 3,
                    "round_count": 3,
                    "accumulated_visible_suggestions": 6,
                    "decision": "target_reached",
                    "round_index": 3,
                    "rounds": [],
                }
            },
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Target reached: 6 new source suggestions after 3 search rounds.")
        html = response.content.decode("utf-8")
        current_state_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Current research state</h2>'
        query_performance_header = '<h2 style="margin: 0 0 14px; font-size: 20px;">Query performance</h2>'
        current_state_section = html.split(current_state_header, 1)[1].split(query_performance_header, 1)[0]
        self.assertIn("Target reached: 6 new source suggestions after 3 search rounds.", current_state_section)
        self.assertNotIn("2 new source suggestions were added.", current_state_section)

    def test_current_research_state_falls_back_to_last_run_note_without_cycle_diagnostics(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Fallback feedback topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=1,
            provider_result_count=1,
            accepted_count=1,
            rejected_count=0,
            new_suggestions_count=1,
            already_known_count=0,
            diagnostics={},
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "1 new source suggestion was added.")

    def test_current_research_state_keeps_provider_unavailable_feedback(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Provider unavailable feedback topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_FAILED,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=1,
            provider_result_count=0,
            accepted_count=0,
            rejected_count=0,
            new_suggestions_count=0,
            already_known_count=0,
            diagnostics={
                "discovery_cycle": {
                    "target_visible_suggestions": 6,
                    "target_visible_new_suggestions": 6,
                    "max_immediate_rounds": 3,
                    "rounds_run": 1,
                    "round_count": 1,
                    "accumulated_visible_suggestions": 0,
                    "decision": "provider_unavailable",
                    "round_index": 1,
                    "rounds": [],
                }
            },
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Provider results could not be loaded.")

    def test_source_discovery_run_diagnostics_include_source_quality_feedback(self) -> None:
        source_research_result = MagicMock()
        source_research_result.diagnostics = {
            "query_performance": [
                {
                    "query": "bitcoin market structure report",
                    "provider": "serpapi",
                    "returned_count": 1,
                    "accepted_count": 1,
                    "rejected_count": 0,
                    "duplicate_count": 0,
                    "visible_new_suggestions_count": 0,
                    "status": "useful",
                },
                {
                    "query": "bitcoin price prediction",
                    "provider": "serpapi",
                    "returned_count": 1,
                    "accepted_count": 0,
                    "rejected_count": 1,
                    "duplicate_count": 0,
                    "visible_new_suggestions_count": 0,
                    "status": "weak",
                },
            ]
        }
        source_research_result.provider_result.provider_name = "serpapi"
        source_research_result.evaluated_candidates = (
            MagicMock(
                url="https://glassnode.com/reports/bitcoin-market-structure",
                title="Bitcoin market structure weekly commentary",
                snippet="ETF flows and liquidity signals",
                candidate_type="article",
                normalized_url="https://glassnode.com/reports/bitcoin-market-structure",
                status=SimpleNamespace(value="accepted"),
                diagnostics={"query": "bitcoin market structure report"},
                rejection_reasons=(),
            ),
            MagicMock(
                url="https://example.com/bitcoin-price-prediction",
                title="Bitcoin price prediction for beginners",
                snippet="Will BTC hit a new high?",
                candidate_type="article",
                normalized_url="https://example.com/bitcoin-price-prediction",
                status=SimpleNamespace(value="rejected"),
                diagnostics={
                    "query": "bitcoin price prediction",
                    "quality_rejection_reason": "not enough substantive signals",
                },
                rejection_reasons=("not enough substantive signals",),
            ),
        )

        diagnostics = _build_source_discovery_run_diagnostics(
            source_research_result=source_research_result,
            known_normalized_urls=set(),
            shown_candidates=[
                {
                    "url": "https://glassnode.com/reports/bitcoin-market-structure",
                    "query": "bitcoin market structure report",
                }
            ],
        )

        feedback = diagnostics["source_quality_feedback"]
        self.assertEqual(feedback["quality_rejected_count"], 1)
        self.assertEqual(feedback["shown_count"], 1)
        self.assertTrue(any(item["material_type"] == "price_prediction_live_price" for item in feedback["weak_material_types"]))
        self.assertTrue(any(item["material_type"] == "on_chain_analysis" for item in feedback["preferred_material_types_found"]))
        self.assertTrue(feedback["planner_quality_guidance"])

    def test_research_history_page_renders_partial_failure_warning(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Research history partial failure topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_PARTIAL_FAILED,
            search_recency_months=1,
            search_time_filter="qdr:m",
            query_count=4,
            provider_result_count=2,
            accepted_count=1,
            rejected_count=1,
            new_suggestions_count=0,
            already_known_count=1,
            diagnostics={
                "selected_query_angle_key": "research_report",
                "selected_query_angle_suffix": "research report",
                "selected_query_angle_reason": "Rotate toward research reports and evidence summaries.",
                "previous_discovery_run_count": 2,
                "duplicate_url_count": 0,
                "provider_errors": [
                    {
                        "query": "Automation research report case study",
                        "message": "SerpAPI returned an API error.",
                    }
                ],
                "per_query_result_counts": [
                    {
                        "intent": "implementation_guide",
                        "query": "Automation research report implementation guide",
                        "result_count": 2,
                    }
                ],
            },
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Discovery partially completed")
        self.assertNotContains(response, '<h2 class="history-run__title">partial_failed</h2>', html=False)
        self.assertContains(response, "Provider warning")
        self.assertContains(
            response,
            "2 URLs returned",
        )
        self.assertContains(response, "0 visible new suggestions")
        self.assertContains(response, "partial run")
        self.assertNotContains(response, "1 passed, 1 rejected")
        self.assertNotContains(response, "1 known/duplicate")
        self.assertContains(response, "Some provider queries returned results, but at least one query failed.")
        self.assertContains(response, "Some searches could not be completed. Other searches still returned results.")
        self.assertContains(response, "Technical reason")
        self.assertContains(response, "SerpAPI returned an API error.")
        self.assertContains(
            response,
            "These counts describe different pipeline checks and may overlap; they are not an additive breakdown.",
        )
        self.assertContains(response, "Passed filtering")
        self.assertContains(response, "Rejected by filters")
        self.assertContains(response, "Already known or duplicate")
        self.assertContains(response, "Query details")
        self.assertContains(response, "Technical details")
        self.assertNotContains(response, "Source discovery details")

    def test_research_history_page_renders_seen_source_history_with_user_facing_labels(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Seen source history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/kept-source",
            url="https://example.org/kept-source",
            title="Kept source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=3,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/removed-source",
            url="https://example.org/removed-source",
            title="Removed source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
            seen_count=2,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/rejected-source",
            url="https://example.org/rejected-source",
            title="Rejected source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_COMMERCIAL_REJECTED,
            seen_count=1,
            freshness_status="very_stale",
            detected_publication_year=2018,
            quality_rejection_reason="rejected because: product/demo/pricing intent",
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/shown-source",
            url="https://example.org/shown-source",
            title="Shown source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/seen-source",
            url="https://example.org/seen-source",
            title="Seen source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED,
            seen_count=4,
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Seen sources")
        self.assertContains(response, "<table", html=False)
        self.assertContains(response, "View details")
        self.assertContains(response, '<span class="muted">—</span>', html=False)
        self.assertContains(response, "Kept source")
        self.assertContains(
            response,
            '<a href="https://example.org/kept-source" target="_blank" rel="noopener noreferrer">Kept source</a>',
            html=False,
        )
        self.assertContains(response, "example.org")
        self.assertContains(response, "Seen count")
        self.assertContains(response, "Kept")
        self.assertContains(response, "Removed by user")
        self.assertContains(response, "Rejected by quality")
        self.assertContains(response, "Shown as suggestion")
        self.assertContains(response, "Seen only")
        self.assertContains(response, "Already known")
        self.assertContains(response, "Previously removed")
        self.assertContains(response, "Previously rejected")
        self.assertContains(response, "Commercial rejected")
        self.assertContains(response, "2018")
        self.assertContains(response, "rejected because: product/demo/pricing intent")
        self.assertContains(response, "Removed source")
        self.assertContains(response, "Shown source")
        self.assertContains(response, "Seen source")
        self.assertNotContains(response, ">removed_by_user<", html=False)
        self.assertNotContains(response, ">rejected_by_quality<", html=False)
        self.assertNotContains(response, ">already_known<", html=False)

    def test_research_history_page_backfills_kept_topic_source_without_history_row(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Backfilled kept source history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Backfilled kept source",
            url="https://example.org/backfilled-kept",
            normalized_url="https://example.org/backfilled-kept",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "kept"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Backfilled kept source")
        self.assertNotContains(response, "No seen sources yet.")
        history_item = SourceDiscoveryHistory.objects.get(
            topic=topic,
            normalized_url="https://example.org/backfilled-kept",
        )
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_sync_does_not_overwrite_removed_by_user_with_kept(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync preserves removed over kept topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Pinned discovered source",
            url="https://example.org/preserved-removed-kept",
            normalized_url="https://example.org/preserved-removed-kept",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
            seen_count=2,
        )

        sync_topic_discovered_sources_into_history(topic)

        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REMOVED_BY_USER)

    def test_sync_does_not_overwrite_removed_by_user_with_shown(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync preserves removed over shown topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Shown discovered source",
            url="https://example.org/preserved-removed-shown",
            normalized_url="https://example.org/preserved-removed-shown",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
            seen_count=2,
        )

        sync_topic_discovered_sources_into_history(topic)

        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REMOVED_BY_USER)

    def test_sync_does_not_overwrite_rejected_by_quality_with_shown(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync preserves rejected over shown topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Shown discovered source",
            url="https://example.org/preserved-rejected-shown",
            normalized_url="https://example.org/preserved-rejected-shown",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_QUALITY_REJECTED,
            seen_count=2,
        )

        sync_topic_discovered_sources_into_history(topic)

        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY)

    def test_sync_allows_shown_to_upgrade_to_kept(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync upgrades shown to kept topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Pinned discovered source",
            url="https://example.org/shown-to-kept",
            normalized_url="https://example.org/shown-to-kept",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )

        sync_topic_discovered_sources_into_history(topic)

        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_sync_allows_seen_to_upgrade_to_shown(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync upgrades seen to shown topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Shown discovered source",
            url="https://example.org/seen-to-shown",
            normalized_url="https://example.org/seen-to-shown",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        sync_topic_discovered_sources_into_history(topic)

        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)

    def test_sync_does_not_create_history_for_manual_topic_source(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync ignores manual source topic",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="Manual source",
            url="https://example.org/manual-only",
            normalized_url="https://example.org/manual-only",
            source_type="website",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )

        sync_topic_discovered_sources_into_history(topic)

        self.assertFalse(
            SourceDiscoveryHistory.objects.filter(
                topic=topic,
                normalized_url="https://example.org/manual-only",
            ).exists()
        )

    def test_sync_updates_existing_seen_history_row_to_single_kept_row(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync merges seen and kept history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        TopicSource.objects.create(
            topic=topic,
            name="AI Tools Every Teen Should Know",
            url="https://example.org/ai-tools-every-teen-should-know",
            normalized_url="https://example.org/ai-tools-every-teen-should-know",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/ai-tools-every-teen-should-know/",
            url="https://example.org/ai-tools-every-teen-should-know/",
            title="AI Tools Every Teen Should Know",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "kept"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI Tools Every Teen Should Know", count=1)
        self.assertNotContains(response, "No seen sources yet.")
        self.assertEqual(SourceDiscoveryHistory.objects.filter(topic=topic).count(), 1)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.normalized_url, "https://example.org/ai-tools-every-teen-should-know")
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_unpin_route_matches_remove_semantics_for_discovered_source(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Legacy unpin route topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Pinned discovered source",
            url="https://example.org/pinned-discovered",
            normalized_url="https://example.org/pinned-discovered",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=2,
        )

        response = self.client.post(reverse("unpin-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 302)
        source.refresh_from_db()
        self.assertFalse(source.is_pinned)
        self.assertFalse(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)

    def test_kept_sources_remove_button_posts_to_remove_topic_source(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Kept remove button topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Kept source",
            url="https://example.org/kept-remove-button",
            normalized_url="https://example.org/kept-remove-button",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(
            response,
            f'action="{reverse("remove-topic-source", args=[topic.id, source.id])}"',
            html=False,
        )
        self.assertNotContains(
            response,
            f'action="{reverse("unpin-topic-source", args=[topic.id, source.id])}"',
            html=False,
        )

    def test_refresh_replacement_marks_pruned_shown_history_as_seen(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Refresh replacement history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        old_source = TopicSource.objects.create(
            topic=topic,
            name="Old shown source",
            url="https://example.org/old-shown",
            normalized_url="https://example.org/old-shown",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=old_source,
            normalized_url=old_source.normalized_url,
            url=old_source.url,
            title=old_source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )

        _upsert_and_build_source_candidates(
            topic,
            [
                {
                    "url": "https://example.org/new-shown",
                    "title": "New shown source",
                    "default_selected": True,
                }
            ],
            prune_missing_discovered=True,
        )

        self.assertFalse(TopicSource.objects.filter(pk=old_source.pk).exists())
        history_item = SourceDiscoveryHistory.objects.get(topic=topic, normalized_url=old_source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SEEN)
        shown_response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "shown"})
        self.assertNotContains(shown_response, "Old shown source")

    @patch("apps.digests.views.resolve_source_candidates")
    def test_rediscovery_preserves_checked_discovered_suggestion_while_pruning_unchecked_stale_suggestion(
        self,
        mock_resolve_source_candidates,
    ) -> None:
        mock_resolve_source_candidates.side_effect = [
            [
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
            ],
            [
                {
                    "url": "https://dev.to/t/golang",
                    "title": "DEV Community / #golang",
                    "description": "Broad Go engineering stream.",
                    "source_type": "devto_tag",
                    "platform": "dev.to",
                    "recent_article_count": 5,
                    "has_recent_article_count": True,
                    "default_selected": False,
                    "candidate_origin": "discovered",
                }
            ],
        ]
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Rediscovery preserves checked suggestions",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["engineering"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        first_discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": TopicSourceMode.DISCOVERY_ONLY,
                "run_research": "1",
            },
        )

        self.assertEqual(first_discovery_response.status_code, 200)
        discovered_sources = {source.url: source for source in topic.sources.order_by("id")}
        python_source = discovered_sources["https://dev.to/t/python"]
        django_source = discovered_sources["https://dev.to/t/django"]
        self.assertTrue(python_source.is_active)
        self.assertTrue(django_source.is_active)

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
                "run_research": "1",
            },
        )

        self.assertEqual(rediscovery_response.status_code, 200)
        self.assertFalse(TopicSource.objects.filter(pk=python_source.pk).exists())
        django_source.refresh_from_db()
        self.assertTrue(django_source.is_active)
        self.assertTrue(TopicSource.objects.filter(topic=topic, url="https://dev.to/t/golang").exists())

        html = rediscovery_response.content.decode("utf-8")
        new_section = html.rsplit("New suggestions", 1)[1]
        self.assertNotIn("DEV Community / #python", new_section)
        self.assertIn("DEV Community / #django", new_section)
        self.assertIn("DEV Community / #golang", new_section)
        django_checkbox = new_section.split("DEV Community / #django", 1)[0].rsplit('value="1"', 1)[1].split(">", 1)[0]
        golang_checkbox = new_section.split("DEV Community / #golang", 1)[0].rsplit('value="1"', 1)[1].split(">", 1)[0]
        self.assertIn("checked", django_checkbox)
        self.assertIn("checked", golang_checkbox)

    def test_research_history_current_state_cards_use_current_discovered_topic_sources(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Current research state counts topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        kept_source = TopicSource.objects.create(
            topic=topic,
            name="Current kept source",
            url="https://example.org/current-kept",
            normalized_url="https://example.org/current-kept",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        shown_source = TopicSource.objects.create(
            topic=topic,
            name="Current shown source",
            url="https://example.org/current-shown",
            normalized_url="https://example.org/current-shown",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=kept_source,
            normalized_url=kept_source.normalized_url,
            url=kept_source.url,
            title=kept_source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=shown_source,
            normalized_url=shown_source.normalized_url,
            url=shown_source.url,
            title=shown_source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/stale-shown",
            url="https://example.org/stale-shown",
            title="Stale shown source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Current research state")
        cards = {item["label"]: item["value"] for item in response.context["current_research_state"]["cards"]}
        self.assertEqual(cards["Kept"], "1")
        self.assertEqual(cards["Shown now"], "1")
        self.assertNotIn("Removed", cards)

    def test_remove_from_kept_updates_current_research_state_counts(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Remove updates current research state topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Current kept source",
            url="https://example.org/current-kept-remove",
            normalized_url="https://example.org/current-kept-remove",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=source,
            normalized_url=source.normalized_url,
            url=source.url,
            title=source.name,
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))
        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        cards = {item["label"]: item["value"] for item in response.context["current_research_state"]["cards"]}
        self.assertEqual(cards["Kept"], "0")
        self.assertEqual(cards["Shown now"], "1")
        self.assertNotIn("Removed", cards)

    def test_sync_merges_trailing_slash_history_without_creating_duplicate_row(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync trailing slash merge topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Trailing slash source",
            url="https://example.org/trailing-slash-source",
            normalized_url="https://example.org/trailing-slash-source",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/trailing-slash-source/",
            url="https://example.org/trailing-slash-source/",
            title="Trailing slash source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        sync_topic_discovered_sources_into_history(topic)

        self.assertEqual(SourceDiscoveryHistory.objects.filter(topic=topic).count(), 1)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.normalized_url, source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_sync_merges_query_noise_history_without_creating_duplicate_row(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync query noise merge topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Query noise source",
            url="https://example.org/query-noise-source",
            normalized_url="https://example.org/query-noise-source",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/query-noise-source?utm_source=newsletter&ref=feed",
            url="https://example.org/query-noise-source?utm_source=newsletter&ref=feed",
            title="Query noise source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )

        sync_topic_discovered_sources_into_history(topic)

        self.assertEqual(SourceDiscoveryHistory.objects.filter(topic=topic).count(), 1)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.normalized_url, source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_sync_merges_srsltid_history_without_creating_duplicate_rows(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Sync srsltid merge topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="AI Tools Every Teen Should Know",
            url="https://example.org/ai-tools-every-teen-should-know",
            normalized_url="https://example.org/ai-tools-every-teen-should-know",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/ai-tools-every-teen-should-know?srsltid=abc123",
            url="https://example.org/ai-tools-every-teen-should-know?srsltid=abc123",
            title="AI Tools Every Teen Should Know",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/ai-tools-every-teen-should-know?srsltid=def456",
            url="https://example.org/ai-tools-every-teen-should-know?srsltid=def456",
            title="AI Tools Every Teen Should Know",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_DUPLICATE_DOMAIN,
            seen_count=1,
        )

        sync_topic_discovered_sources_into_history(topic)

        self.assertEqual(SourceDiscoveryHistory.objects.filter(topic=topic).count(), 1)
        history_item = SourceDiscoveryHistory.objects.get(topic=topic)
        self.assertEqual(history_item.normalized_url, source.normalized_url)
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)

    def test_research_history_page_filters_seen_source_history_by_status(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Filtered seen source history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/kept-only",
            url="https://example.org/kept-only",
            title="Kept only source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=2,
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/removed-only",
            url="https://example.org/removed-only",
            title="Removed only source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
            seen_count=2,
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "kept"})

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "All")
        self.assertContains(response, "Kept")
        self.assertNotContains(response, '>Removed<', html=False)
        self.assertContains(response, '#seen-sources', html=False)
        self.assertContains(response, '?status=kept#seen-sources', html=False)
        self.assertContains(response, "Kept only source")
        self.assertNotContains(response, "Removed only source")

        removed_response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "removed"})

        self.assertEqual(removed_response.status_code, 200)
        self.assertContains(removed_response, 'name="status" value="removed"', html=False)
        self.assertNotContains(removed_response, '>Removed<', html=False)
        self.assertContains(removed_response, "Removed only source")
        self.assertNotContains(removed_response, "Kept only source")

    def test_research_history_page_seen_source_details_show_secondary_fields(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Seen source history details topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/detail-source",
            url="https://example.org/detail-source",
            title="Detail source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_STALE_REJECTED,
            seen_count=5,
            freshness_status="very_stale",
            detected_publication_year=2018,
            quality_rejection_reason="rejected because: stale source outside recency window",
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/unknown-date-source",
            url="https://example.org/unknown-date-source",
            title="Unknown date source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_SEEN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
            seen_count=1,
            freshness_status="unknown",
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "View details", count=1)
        self.assertNotContains(response, "Source URL")
        self.assertNotContains(response, "Open source")
        self.assertContains(
            response,
            '<a href="https://example.org/detail-source" target="_blank" rel="noopener noreferrer">Detail source</a>',
            html=False,
        )
        self.assertNotContains(response, ">https://example.org/detail-source<", html=False)
        self.assertContains(response, '<span class="muted">—</span>', html=False)
        self.assertContains(response, "First seen")
        self.assertContains(response, "Freshness")
        self.assertContains(response, "Publication year")
        self.assertContains(response, "Quality note")
        self.assertContains(response, "rejected because: stale source outside recency window")
        self.assertNotContains(response, ">Unknown date<", html=False)

    def test_research_history_page_pagination_links_keep_seen_sources_anchor(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Seen source history pagination topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        for index in range(30):
            SourceDiscoveryHistory.objects.create(
                user=topic.user,
                topic=topic,
                normalized_url=f"https://example.org/paginated-{index}",
                url=f"https://example.org/paginated-{index}",
                title=f"Paginated source {index}",
                domain="example.org",
                status=SourceDiscoveryHistory.STATUS_SEEN,
                last_run_outcome=SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN,
                seen_count=1,
            )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Next")
        self.assertContains(response, '?page=2#seen-sources', html=False)

    def test_remove_kept_source_without_history_creates_shown_history_and_keeps_source(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Shown history fallback topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        source = TopicSource.objects.create(
            topic=topic,
            name="Kept without history",
            url="https://example.org/kept-without-history",
            normalized_url="https://example.org/kept-without-history",
            source_type="website",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )

        response = self.client.post(reverse("remove-topic-source", args=[topic.id, source.id]))

        self.assertEqual(response.status_code, 200)
        source.refresh_from_db()
        self.assertFalse(source.is_pinned)
        self.assertFalse(source.is_active)
        history_item = SourceDiscoveryHistory.objects.get(
            topic=topic,
            normalized_url="https://example.org/kept-without-history",
        )
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_SHOWN)
        shown_response = self.client.get(reverse("topic-research-history", args=[topic.id]), {"status": "shown"})
        self.assertContains(shown_response, "Kept without history")

    def test_research_history_page_shows_empty_state_when_no_runs_or_seen_sources_exist(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Empty research history topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research history")
        self.assertContains(response, "No query performance data yet.")
        self.assertContains(
            response,
            "Run source discovery to see which queries were used and what results they produced.",
        )
        self.assertContains(response, "No source quality feedback yet.")
        self.assertContains(
            response,
            "Run source discovery to see which domains, material types, and rejection patterns are dominating recent results.",
        )
        self.assertContains(response, "No research history yet.")
        self.assertContains(response, "Run Find sources to start source discovery for this topic.")
        self.assertContains(response, "No seen sources yet.")

    @patch("apps.digests.views.run_source_research")
    def test_opening_research_history_does_not_trigger_provider_search(self, mock_run_source_research) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Research history read-only topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        response = self.client.get(reverse("topic-research-history", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        mock_run_source_research.assert_not_called()

    def test_main_topic_page_does_not_render_seen_source_history(self) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Main page stays clean topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.org/clean-history",
            url="https://example.org/clean-history",
            title="Clean history source",
            domain="example.org",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
            seen_count=2,
        )

        response = self.client.get(reverse("topic-workspace", args=[topic.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Research history")
        self.assertNotContains(response, "Seen sources")
        self.assertNotContains(response, "Removed by user")
        self.assertNotContains(response, "Previously removed")
        self.assertNotContains(response, "Current research state")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_all_rejected_refresh_keeps_existing_new_suggestions(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation consulting services",
                    "link": "https://example.com/services/ai-automation-refresh",
                    "snippet": "Book a demo and contact sales to see how we help businesses automate.",
                    "source": "Example",
                }
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Rejected refresh keeps suggestions",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing suggestion",
            url="https://example.com/existing-rejected",
            normalized_url="https://example.com/existing-rejected",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )

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
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.pk).exists())
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "DigestFlow could not reach the 6-source target after 3 search rounds with the current search strategy.",
        )
        history_item = SourceDiscoveryHistory.objects.get(
            topic=topic,
            normalized_url="https://example.com/services/ai-automation-refresh",
        )
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY)
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_all_already_known_refresh_keeps_existing_new_suggestions(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Already known refresh keeps suggestions",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        existing_source = TopicSource.objects.create(
            topic=topic,
            name="Existing suggestion",
            url="https://example.com/already-known",
            normalized_url="https://example.com/already-known",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=True,
        )
        history_item = SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.com/already-known",
            url="https://example.com/already-known",
            title="Existing suggestion",
            status=SourceDiscoveryHistory.STATUS_SHOWN,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
            first_seen_at=timezone.now(),
            last_seen_at=timezone.now(),
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Existing suggestion",
                    "link": "https://example.com/already-known",
                    "snippet": "Recent case study with implementation details and tradeoffs.",
                    "source": "Example",
                }
            ],
        )

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
        self.assertTrue(TopicSource.objects.filter(pk=existing_source.pk).exists())
        self.assertEqual(TopicSource.objects.filter(topic=topic, normalized_url="https://example.com/already-known").count(), 1)
        history_item.refresh_from_db()
        self.assertEqual(history_item.seen_count, 4)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN)
        self.assertContains(response, "Source discovery partially completed")
        self.assertContains(
            response,
            "DigestFlow could not reach the 6-source target after 3 search rounds with the current search strategy.",
        )
        self.assertContains(response, "Existing suggestion")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_previously_removed_url_does_not_resurface_as_new_suggestion(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Previously removed source topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        history_item = SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.com/previously-removed",
            url="https://example.com/previously-removed",
            title="Previously removed source",
            status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
            first_seen_at=timezone.now(),
            last_seen_at=timezone.now(),
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Previously removed source",
                    "link": "https://example.com/previously-removed",
                    "snippet": "Recent case study with implementation details and tradeoffs.",
                    "source": "Example",
                }
            ],
        )

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
        self.assertEqual(topic.sources.filter(normalized_url="https://example.com/previously-removed").count(), 0)
        history_item.refresh_from_db()
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REMOVED_BY_USER)
        self.assertEqual(history_item.seen_count, 4)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED)
        self.assertContains(response, "New suggestions · 0")
        self.assertContains(response, "No new suggestions yet.")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_previously_rejected_url_does_not_resurface_as_new_suggestion(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Previously rejected source topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        history_item = SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            normalized_url="https://example.com/previously-rejected",
            url="https://example.com/previously-rejected",
            title="Previously rejected source",
            status=SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_QUALITY_REJECTED,
            seen_count=1,
            quality_rejection_reason="rejected because: generic benefits/listicle SEO pattern",
            first_seen_at=timezone.now(),
            last_seen_at=timezone.now(),
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Previously rejected source",
                    "link": "https://example.com/previously-rejected",
                    "snippet": "Recent case study with implementation details, methodology, and tradeoffs.",
                    "source": "Example",
                }
            ],
        )

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
        self.assertEqual(topic.sources.filter(normalized_url="https://example.com/previously-rejected").count(), 0)
        history_item.refresh_from_db()
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY)
        self.assertEqual(history_item.seen_count, 4)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED)
        self.assertContains(response, "New suggestions · 0")
        self.assertContains(response, "No new suggestions yet.")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_kept_url_does_not_resurface_as_new_suggestion(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Kept source topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        pinned_source = TopicSource.objects.create(
            topic=topic,
            name="Pinned source",
            url="https://example.com/kept-source",
            normalized_url="https://example.com/kept-source",
            source_type="generic_html",
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=True,
            is_active=True,
        )
        history_item = SourceDiscoveryHistory.objects.create(
            user=topic.user,
            topic=topic,
            topic_source=pinned_source,
            normalized_url="https://example.com/kept-source",
            url="https://example.com/kept-source",
            title="Pinned source",
            status=SourceDiscoveryHistory.STATUS_KEPT,
            last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
            seen_count=1,
            first_seen_at=timezone.now(),
            last_seen_at=timezone.now(),
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Pinned source",
                    "link": "https://example.com/kept-source",
                    "snippet": "Recent case study with implementation details and tradeoffs.",
                    "source": "Example",
                }
            ],
        )

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
        self.assertEqual(topic.sources.filter(normalized_url="https://example.com/kept-source").count(), 1)
        pinned_source.refresh_from_db()
        self.assertTrue(pinned_source.is_pinned)
        history_item.refresh_from_db()
        self.assertEqual(history_item.status, SourceDiscoveryHistory.STATUS_KEPT)
        self.assertEqual(history_item.seen_count, 4)
        self.assertEqual(history_item.last_run_outcome, SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN)
        html = response.content.decode("utf-8")
        new_section = html.split("New suggestions · 0", 1)[1].split("Ready to generate", 1)[0]
        self.assertNotIn("Pinned source", new_section)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_duplicate_url_inside_same_provider_result_set_creates_at_most_one_visible_suggestion(
        self,
        mock_urlopen,
    ) -> None:
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Duplicate provider url topic",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["automation"],
            focus_initialized=True,
            excluded_keywords=[],
        )
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Fresh automation source",
                    "link": "https://example.com/duplicate-provider-url",
                    "snippet": "Recent case study with implementation details and tradeoffs.",
                    "source": "Example",
                },
                {
                    "position": 2,
                    "title": "Fresh automation source duplicate",
                    "link": "https://example.com/duplicate-provider-url",
                    "snippet": "Recent case study with implementation details and tradeoffs.",
                    "source": "Example",
                },
            ],
        )

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
        self.assertEqual(topic.sources.filter(normalized_url="https://example.com/duplicate-provider-url").count(), 1)
        self.assertEqual(
            SourceDiscoveryHistory.objects.filter(
                topic=topic,
                normalized_url="https://example.com/duplicate-provider-url",
            ).count(),
            1,
        )

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_filters_low_quality_commercial_results(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation consulting services",
                    "link": "https://example.com/services/ai-automation",
                    "snippet": "Book a demo and contact sales to see how we help businesses automate.",
                    "source": "Example",
                },
                {
                    "position": 2,
                    "title": "AI automation adoption survey: implementation risks and data",
                    "link": "https://example.com/adoption-survey",
                    "snippet": "Survey data, implementation risks, report findings, and limitations for adoption.",
                    "source": "Example",
                },
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation filtering",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI automation adoption survey: implementation risks and data")
        self.assertNotContains(response, "AI automation consulting services")
        topic = Topic.objects.get(name="AI automation filtering")
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_filters_generic_benefits_pages_across_domains(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "Top 10 benefits of sleep training for your baby",
                    "link": "https://example.com/sleep-benefits",
                    "snippet": "Transform your family nights and boost your baby's sleep with simple wins.",
                    "source": "Example",
                },
                {
                    "position": 2,
                    "title": "Infant sleep intervention study: methodology and limitations",
                    "link": "https://example.com/sleep-study",
                    "snippet": "Study evidence, methodology, and limitations for infant sleep intervention.",
                    "source": "Example",
                },
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Child sleep filtering",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["infant sleep"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Infant sleep intervention study: methodology and limitations")
        self.assertNotContains(response, "Top 10 benefits of sleep training for your baby")
        topic.refresh_from_db()
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_provider_backed_find_sources_filters_stale_2018_results(
        self,
        mock_urlopen,
    ) -> None:
        self._mock_serpapi_urlopen(
            mock_urlopen,
            [
                {
                    "position": 1,
                    "title": "AI automation adoption report 2018",
                    "link": "https://example.com/2018/adoption-report",
                    "snippet": "2018 report with methodology, findings, and implementation details.",
                    "source": "Example",
                    "date": "2018-04-03",
                },
                {
                    "position": 2,
                    "title": "AI automation adoption survey: implementation risks and data",
                    "link": "https://example.com/2026/adoption-survey",
                    "snippet": "2026-05-12 survey data, implementation risks, report findings, and limitations for adoption.",
                    "source": "Example",
                    "date": "2026-05-12",
                },
            ],
        )
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="AI automation recency filtering",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["AI automation"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "AI automation adoption survey: implementation risks and data")
        self.assertNotContains(response, "AI automation adoption report 2018")
        topic.refresh_from_db()
        self.assertEqual(topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).count(), 1)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 0)

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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Plural source count",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["python automation"],
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
                "run_research": "1",
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
            keywords=["python community"],
            focus_initialized=True,
            excluded_keywords=[],
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
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "My sources")
        self.assertContains(response, "Research sources")
        self.assertContains(response, "New suggestions")
        self.assertContains(response, "Saved source")
        self.assertContains(response, "New source")
        self.assertNotContains(response, "Fresh discovery result.")
        self.assertNotContains(response, "Already saved on the topic.")
        self.assertContains(
            response,
            f'<form id="workspace-run-form" method="post" action="{reverse("run-pipeline", args=[topic.id])}"',
            html=False,
        )
        self.assertNotContains(response, 'name="selected_source_urls"', html=False)

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
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a manual source link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
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

        self.assertIn("Add a manual source link and press Enter", html)
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
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a manual source link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
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
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a manual source link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
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
            '<input\n                                    id="topic-source-url"\n                                    type="url"\n                                    name="source_url"\n                                    placeholder="Add a manual source link and press Enter"\n                                    value=""\n                                    autocomplete="off"\n                                    data-source-feedback-input',
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Preview only",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["AI agents"],
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
                "run_research": "1",
            },
        )

        self.assertEqual(response.status_code, 200)
        topic.refresh_from_db()
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
            keywords=["AI engineering"],
            focus_initialized=True,
            excluded_keywords=[],
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
        self.assertIn("My sources", workspace_html)
        self.assertIn("Research sources", workspace_html)
        self.assertIn("New suggestions", workspace_html)
        self.assertIn("DEV Community / #ai", workspace_html)
        self.assertIn("Select at least one my source before running this digest.", workspace_html)
        self.assertIn("0 my sources", workspace_html)
        self.assertIn("1 research source", workspace_html)
        saved_section = workspace_html.split("My sources", 1)[1].split("Research sources", 1)[0]
        new_section = workspace_html.rsplit("New suggestions", 1)[1]
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Python discovery",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["Python"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic.refresh_from_db()
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
        self.assertIn("Find or keep at least one research source before running this digest.", refreshed_html)
        self.assertNotIn("My sources", refreshed_html)
        self.assertIn("data-run-source-count-button", refreshed_html)
        self.assertIn("disabled", refreshed_html)
        new_section = refreshed_html.rsplit("New suggestions", 1)[1]
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Python browser flow",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["Python"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic.refresh_from_db()
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
        self.assertIn("Research sources", refreshed_html)
        self.assertIn("New suggestions", refreshed_html)
        self.assertNotIn("My sources", refreshed_html)
        self.assertIn("DEV Community / #python", refreshed_html)
        self.assertIn("Find or keep at least one research source before running this digest.", refreshed_html)
        self.assertIn("data-run-source-count-button", refreshed_html)
        self.assertIn("disabled", refreshed_html)
        new_section = refreshed_html.rsplit("New suggestions", 1)[1]
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
        topic = Topic.objects.create(
            user=self._get_ui_user(),
            name="Python rediscovery flow",
            source_mode=TopicSourceMode.DISCOVERY_ONLY,
            keywords=["Python"],
            focus_initialized=True,
            excluded_keywords=[],
        )

        discovery_response = self.client.post(
            reverse("discover-sources"),
            data={
                "topic_id": topic.id,
                "topic_name": topic.name,
                "source_url": "",
                "source_mode": topic.source_mode,
                "run_research": "1",
            },
        )

        self.assertEqual(discovery_response.status_code, 200)
        topic.refresh_from_db()
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
                "run_research": "1",
            },
        )

        self.assertEqual(rediscovery_response.status_code, 200)
        python_source.refresh_from_db()
        django_source.refresh_from_db()
        self.assertFalse(python_source.is_active)
        self.assertTrue(django_source.is_active)
        self.assertContains(rediscovery_response, "1 selected source will be used in the next digest run.")
        html = rediscovery_response.content.decode("utf-8")
        new_section = html.rsplit("New suggestions", 1)[1]
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
