from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource
from services.sources.candidates import SourceCandidateStatus
from services.sources.research_orchestrator import SourceResearchResult, run_source_research
from services.sources.research_queries import build_research_query_plan
from services.sources.search_provider import FakeSearchProvider, SearchProviderResult


class _TopicStub:
    def __init__(self, name: str, keywords) -> None:
        self.name = name
        self.keywords = keywords


class SourceResearchOrchestratorTests(SimpleTestCase):
    def test_orchestrator_builds_query_plan(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])

        result = run_source_research(topic, FakeSearchProvider({}))

        self.assertEqual(result.query_plan.topic_name, "Infant sleep")
        self.assertTrue(result.query_plan.query_items)

    def test_orchestrator_calls_provider_boundary(self) -> None:
        topic = _TopicStub("Travel planning", ["family travel"])

        with patch("services.sources.research_orchestrator.search_research_query_plan") as mock_search:
            mock_search.return_value = SearchProviderResult(provider_name="fake", results=(), diagnostics={})
            run_source_research(topic, FakeSearchProvider({}))

        mock_search.assert_called_once()

    def test_orchestrator_converts_raw_results_to_candidate_inputs(self) -> None:
        topic = _TopicStub("Travel planning", ["family travel"])
        provider = FakeSearchProvider(
            {
                "Travel planning family travel official guidelines": [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "Snippet"}
                ]
            }
        )

        result = run_source_research(topic, provider)

        self.assertEqual(len(result.candidate_inputs), 1)
        self.assertEqual(result.candidate_inputs[0].url, "https://example.com/guide")

    def test_orchestrator_evaluates_candidates(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])
        plan = build_research_query_plan(topic)
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

        result = run_source_research(topic, provider)

        self.assertEqual(len(result.evaluated_candidates), 1)
        self.assertEqual(result.evaluated_candidates[0].status, SourceCandidateStatus.ACCEPTED)

    def test_orchestrator_builds_review_items(self) -> None:
        topic = _TopicStub("Infant sleep", ["safe sleep", "SIDS"])
        plan = build_research_query_plan(topic)
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

        result = run_source_research(topic, provider)

        self.assertEqual(len(result.review_items), 1)
        self.assertTrue(result.review_items[0].is_selectable)

    def test_output_result_exposes_all_major_pipeline_stages(self) -> None:
        result = run_source_research(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))

        self.assertIsInstance(result, SourceResearchResult)
        self.assertIsNotNone(result.query_plan)
        self.assertIsNotNone(result.provider_result)
        self.assertIsNotNone(result.candidate_inputs)
        self.assertIsNotNone(result.evaluated_candidates)
        self.assertIsNotNone(result.review_items)
        self.assertIsNotNone(result.diagnostics)

    def test_diagnostics_include_useful_stage_counts(self) -> None:
        result = run_source_research(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))

        self.assertIn("query_count", result.diagnostics)
        self.assertIn("raw_result_count", result.diagnostics)
        self.assertIn("candidate_input_count", result.diagnostics)
        self.assertIn("evaluated_candidate_count", result.diagnostics)
        self.assertIn("review_item_count", result.diagnostics)
        self.assertIn("accepted_candidate_count", result.diagnostics)
        self.assertIn("needs_review_candidate_count", result.diagnostics)
        self.assertIn("rejected_candidate_count", result.diagnostics)
        self.assertIn("non_accepted_candidate_count", result.diagnostics)
        self.assertIn("selectable_review_item_count", result.diagnostics)

    def test_accepted_and_selectable_candidates_are_represented_correctly(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = build_research_query_plan(topic)
        provider = FakeSearchProvider(
            {
                plan.query_items[0].query: [
                    {
                        "title": "AI automation guide",
                        "url": "https://example.com/ai-guide",
                        "snippet": "Implementation guide for AI automation with Zapier.",
                    }
                ]
            }
        )

        result = run_source_research(topic, provider)

        self.assertEqual(result.diagnostics["accepted_candidate_count"], 1)
        self.assertEqual(result.diagnostics["needs_review_candidate_count"], 0)
        self.assertEqual(result.diagnostics["rejected_candidate_count"], 0)
        self.assertEqual(result.diagnostics["non_accepted_candidate_count"], 0)
        self.assertEqual(result.diagnostics["selectable_review_item_count"], 1)
        self.assertEqual(result.review_items[0].status, SourceCandidateStatus.ACCEPTED)

    def test_rejected_and_non_selectable_candidates_are_represented_correctly(self) -> None:
        topic = _TopicStub("Travel planning", ["family travel"])
        provider = FakeSearchProvider(
            {
                "Travel planning family travel official guidelines": [
                    {
                        "title": "Home",
                        "url": "https://example.com/",
                        "snippet": "Welcome.",
                    }
                ]
            }
        )

        result = run_source_research(topic, provider)

        self.assertEqual(result.diagnostics["accepted_candidate_count"], 0)
        self.assertEqual(result.diagnostics["needs_review_candidate_count"], 0)
        self.assertEqual(result.diagnostics["rejected_candidate_count"], 1)
        self.assertEqual(result.diagnostics["non_accepted_candidate_count"], 1)
        self.assertEqual(result.diagnostics["selectable_review_item_count"], 0)
        self.assertFalse(result.review_items[0].is_selectable)

    def test_needs_review_candidates_are_not_counted_as_rejected(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        plan = build_research_query_plan(topic)
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

        result = run_source_research(topic, provider)

        self.assertEqual(result.diagnostics["accepted_candidate_count"], 1)
        self.assertEqual(result.diagnostics["needs_review_candidate_count"], 1)
        self.assertEqual(result.diagnostics["rejected_candidate_count"], 0)
        self.assertEqual(result.diagnostics["non_accepted_candidate_count"], 1)
        self.assertEqual(result.diagnostics["selectable_review_item_count"], 2)

    def test_empty_provider_results_are_handled_safely(self) -> None:
        result = run_source_research(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.candidate_inputs, ())
        self.assertEqual(result.evaluated_candidates, ())
        self.assertEqual(result.review_items, ())

    def test_duplicate_provider_results_are_handled_through_existing_pipeline_behavior(self) -> None:
        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        provider = FakeSearchProvider(
            {
                "AI automation Zapier implementation guide": [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "One"}
                ],
                "AI automation Make case study": [
                    {"title": "Guide duplicate", "url": "https://example.com/guide", "snippet": "Duplicate"}
                ],
            }
        )

        result = run_source_research(topic, provider)

        self.assertEqual(result.diagnostics["raw_result_count"], 1)
        self.assertEqual(result.diagnostics["candidate_input_count"], 1)

    def test_no_http_or_template_context_is_required(self) -> None:
        result = run_source_research(_TopicStub("Education for teenagers", ["study habits"]), FakeSearchProvider({}))

        self.assertEqual(result.query_plan.topic_name, "Education for teenagers")

    @patch("socket.create_connection", side_effect=AssertionError("network should not be used"))
    def test_no_external_network_or_api_call_is_made(self, _mock_network) -> None:
        result = run_source_research(_TopicStub("Travel planning", ["family travel"]), FakeSearchProvider({}))

        self.assertEqual(result.provider_result.results, ())


class SourceResearchOrchestratorPersistenceTests(TestCase):
    def test_no_topic_source_rows_are_created(self) -> None:
        user = get_user_model().objects.create_user(username="orchestrator-user", password="pw")
        topic = Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()

        result = run_source_research(topic, FakeSearchProvider({}))

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(TopicSource.objects.count(), before)
