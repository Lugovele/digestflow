from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource, TopicSourceOrigin
from services.sources.candidates import SourceCandidateStatus
from services.sources.content_research_planner import ContentResearchPlannerResult
from services.sources.research_orchestrator import run_source_research
from services.sources.research_review import (
    build_research_review_context,
    build_topic_source_payloads_from_review_items,
    get_persistable_research_candidates,
)
from services.sources.research_queries import build_research_query_plan
from services.sources.search_provider import FakeSearchProvider


class _TopicStub:
    def __init__(self, name: str, keywords) -> None:
        self.name = name
        self.keywords = keywords


def _forced_fallback_planner_result() -> ContentResearchPlannerResult:
    return ContentResearchPlannerResult(
        planner_status="fallback_used",
        fallback_used=True,
        final_queries=(),
        error_message="Forced deterministic planner fallback for review tests.",
    )


class SourceResearchReviewTests(SimpleTestCase):
    def _build_deterministic_plan(self, topic) -> object:
        with patch("services.sources.research_queries.create_content_research_plan") as mock_create_content_research_plan:
            mock_create_content_research_plan.return_value = _forced_fallback_planner_result()
            return build_research_query_plan(topic)

    def _run_source_research_with_plan(self, topic, provider):
        plan = self._build_deterministic_plan(topic)
        with patch("services.sources.research_orchestrator.build_research_query_plan", return_value=plan):
            result = run_source_research(topic, provider)
        return plan, result

    def test_review_context_accepts_source_research_result(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])
        _, result = self._run_source_research_with_plan(topic, FakeSearchProvider({}))

        context = build_research_review_context(result)

        self.assertIn("total_review_item_count", context.diagnostics)

    def test_review_context_preserves_candidate_review_items(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "Safe sleep guidance for infants",
                        "url": "https://example.com/safe-sleep",
                        "snippet": "Evidence-based safe sleep guide.",
                    }
                ]
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)

        context = build_research_review_context(result)

        self.assertEqual(len(context.review_items), 1)
        self.assertEqual(context.review_items[0].label, "Safe sleep guidance for infants")

    def test_accepted_candidates_are_counted_as_accepted_and_selectable(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "Safe sleep guidance for infants",
                        "url": "https://example.com/safe-sleep",
                        "snippet": "Evidence-based safe sleep guide.",
                    }
                ]
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        self.assertEqual(context.diagnostics["accepted_count"], 1)
        self.assertEqual(context.diagnostics["selectable_review_item_count"], 1)
        self.assertEqual(context.accepted_count, 1)
        self.assertEqual(context.selectable_review_item_count, 1)
        self.assertEqual(context.review_items[0].status, SourceCandidateStatus.ACCEPTED)

    def test_needs_review_candidates_are_counted_separately_and_remain_selectable(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "AI automation Zapier implementation guide",
                        "url": "https://example.com/guide",
                        "snippet": "Implementation guide for AI automation workflows with Zapier.",
                    }
                ],
                plan.query_items[1].query: [
                    {
                        "title": "AI automation Make case study",
                        "url": "https://example.com/variant",
                        "snippet": "Case study for AI automation workflows built with Make.",
                    }
                ],
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        self.assertEqual(context.diagnostics["accepted_count"], 1)
        self.assertEqual(context.diagnostics["needs_review_count"], 1)
        self.assertEqual(context.diagnostics["rejected_count"], 0)
        self.assertEqual(context.diagnostics["selectable_review_item_count"], 2)
        self.assertEqual(context.accepted_count, 1)
        self.assertEqual(context.needs_review_count, 1)
        self.assertEqual(context.rejected_count, 0)
        self.assertEqual(context.selectable_review_item_count, 2)

    def test_rejected_candidates_are_counted_as_rejected_and_non_selectable(self) -> None:
        topic = _TopicStub("Travel planning", ["family travel"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "Home",
                        "url": "https://example.com/",
                        "snippet": "Welcome.",
                    }
                ]
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        self.assertEqual(context.diagnostics["accepted_count"], 0)
        self.assertEqual(context.diagnostics["needs_review_count"], 0)
        self.assertEqual(context.diagnostics["rejected_count"], 1)
        self.assertEqual(context.diagnostics["selectable_review_item_count"], 0)
        self.assertEqual(context.accepted_count, 0)
        self.assertEqual(context.needs_review_count, 0)
        self.assertEqual(context.rejected_count, 1)
        self.assertEqual(context.selectable_review_item_count, 0)
        self.assertFalse(context.review_items[0].is_selectable)

    def test_persistable_candidates_exclude_rejected_items(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "AI automation Zapier implementation guide",
                        "url": "https://example.com/guide",
                        "snippet": "Implementation guide for AI automation workflows with Zapier.",
                    }
                ],
                plan.query_items[1].query: [
                    {
                        "title": "AI automation Make case study",
                        "url": "https://example.com/variant",
                        "snippet": "Case study for AI automation workflows built with Make.",
                    }
                ],
                plan.query_items[2].query: [
                    {
                        "title": "Home",
                        "url": "https://reject.example.com/",
                        "snippet": "Welcome.",
                    }
                ],
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)
        persistable = get_persistable_research_candidates(context.review_items)

        self.assertEqual(len(persistable), 2)
        self.assertTrue(all(item.can_be_persisted for item in persistable))
        self.assertEqual(context.diagnostics["persistable_count"], 2)
        self.assertEqual(context.persistable_count, 2)

    def test_diagnostics_and_rejection_reasons_are_preserved(self) -> None:
        topic = _TopicStub("Travel planning", ["family travel"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "Home",
                        "url": "https://example.com/",
                        "snippet": "Welcome.",
                    }
                ]
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        self.assertTrue(context.review_items[0].rejection_reasons)
        self.assertIn("provider_name", context.review_items[0].diagnostics)

    def test_empty_research_result_is_handled_safely(self) -> None:
        context = build_research_review_context(
            self._run_source_research_with_plan(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))[1]
        )

        self.assertEqual(context.review_items, ())
        self.assertEqual(context.persistable_items, ())
        self.assertEqual(context.diagnostics["total_review_item_count"], 0)
        self.assertEqual(context.total_review_item_count, 0)

    def test_direct_count_accessors_match_diagnostics(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "AI automation Zapier implementation guide",
                        "url": "https://example.com/guide",
                        "snippet": "Implementation guide for AI automation workflows with Zapier.",
                    }
                ],
                plan.query_items[1].query: [
                    {
                        "title": "AI automation Make case study",
                        "url": "https://example.com/variant",
                        "snippet": "Case study for AI automation workflows built with Make.",
                    }
                ],
                plan.query_items[2].query: [
                    {
                        "title": "Home",
                        "url": "https://reject.example.com/",
                        "snippet": "Welcome.",
                    }
                ],
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        self.assertEqual(context.total_review_item_count, context.diagnostics["total_review_item_count"])
        self.assertEqual(context.selectable_review_item_count, context.diagnostics["selectable_review_item_count"])
        self.assertEqual(context.accepted_count, context.diagnostics["accepted_count"])
        self.assertEqual(context.needs_review_count, context.diagnostics["needs_review_count"])
        self.assertEqual(context.rejected_count, context.diagnostics["rejected_count"])
        self.assertEqual(context.persistable_count, context.diagnostics["persistable_count"])

    def test_no_http_or_template_context_is_required(self) -> None:
        context = build_research_review_context(
            self._run_source_research_with_plan(_TopicStub("Education for teenagers", ["study habits"]), FakeSearchProvider({}))[1]
        )

        self.assertIsInstance(context.diagnostics, dict)

    @patch("socket.create_connection", side_effect=AssertionError("network should not be used"))
    def test_no_external_network_or_api_call_is_made(self, _mock_network) -> None:
        context = build_research_review_context(
            self._run_source_research_with_plan(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))[1]
        )

        self.assertEqual(context.review_items, ())

    def test_persistence_payloads_remain_explicit_and_pure(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = self._build_deterministic_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "AI automation Zapier implementation guide",
                        "url": "https://example.com/guide",
                        "snippet": "Implementation guide for AI automation workflows with Zapier.",
                    }
                ]
            }
        )
        _, result = self._run_source_research_with_plan(topic, provider)
        context = build_research_review_context(result)

        payloads = build_topic_source_payloads_from_review_items(context.review_items)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(payloads[0]["url"], "https://example.com/guide")
        self.assertEqual(payloads[0]["origin"], TopicSourceOrigin.DISCOVERED)


class SourceResearchReviewPersistenceTests(TestCase):
    def _run_source_research_with_plan(self, topic, provider):
        with patch("services.sources.research_queries.create_content_research_plan") as mock_create_content_research_plan:
            mock_create_content_research_plan.return_value = _forced_fallback_planner_result()
            plan = build_research_query_plan(topic)
        with patch("services.sources.research_orchestrator.build_research_query_plan", return_value=plan):
            return run_source_research(topic, provider)

    def test_no_topic_source_rows_are_created(self) -> None:
        user = get_user_model().objects.create_user(username="research-review-user", password="pw")
        topic = Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()

        context = build_research_review_context(self._run_source_research_with_plan(topic, FakeSearchProvider({})))

        self.assertEqual(context.review_items, ())
        self.assertEqual(TopicSource.objects.count(), before)
