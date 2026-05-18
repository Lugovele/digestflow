from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource
from services.sources.candidates import SourceCandidateInput
from services.sources.search_candidates import (
    search_provider_result_to_candidate_inputs,
    search_result_to_candidate_input,
    search_results_to_candidate_inputs,
)
from services.sources.search_provider import (
    FakeSearchProvider,
    RawSearchResult,
    search_research_query_plan,
)
from services.sources.research_queries import ResearchQueryIntent, build_research_query_plan


class _TopicStub:
    def __init__(self, name: str, keywords) -> None:
        self.name = name
        self.keywords = keywords


class SourceSearchCandidateAdapterTests(SimpleTestCase):
    def test_raw_search_result_converts_to_candidate_input(self) -> None:
        raw_result = RawSearchResult(
            query="AI automation Zapier Make implementation guide",
            title="AI automation guide",
            url="https://example.com/ai-automation",
            snippet="Practical automation guide.",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.IMPLEMENTATION_GUIDE,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertIsInstance(candidate, SourceCandidateInput)

    def test_url_title_and_snippet_are_preserved(self) -> None:
        raw_result = RawSearchResult(
            query="Infant sleep official guidelines",
            title="Safe sleep guidance",
            url="https://example.com/safe-sleep",
            snippet="Evidence-based infant safe sleep guide.",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.OFFICIAL_GUIDELINES,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.url, "https://example.com/safe-sleep")
        self.assertEqual(candidate.title, "Safe sleep guidance")
        self.assertEqual(candidate.snippet, "Evidence-based infant safe sleep guide.")

    def test_provider_name_is_preserved_in_diagnostics(self) -> None:
        raw_result = RawSearchResult(
            query="Travel planning expert advice",
            title="Family travel guide",
            url="https://example.com/travel-guide",
            snippet="Family travel planning help.",
            rank=2,
            provider_name="fake",
            intent=ResearchQueryIntent.EXPERT_ADVICE,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.diagnostics["provider_name"], "fake")

    def test_originating_query_is_preserved_in_diagnostics(self) -> None:
        raw_result = RawSearchResult(
            query="newborn sleep environment SIDS prevention",
            title="Newborn sleep environment guide",
            url="https://example.com/newborn-sleep",
            snippet="Safe environment checklist.",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.EVIDENCE_BASED,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.diagnostics["query"], "newborn sleep environment SIDS prevention")

    def test_originating_intent_is_preserved_in_diagnostics(self) -> None:
        raw_result = RawSearchResult(
            query="n8n AI automation engineering blog",
            title="n8n engineering post",
            url="https://example.com/n8n-post",
            snippet="Automation engineering notes.",
            rank=3,
            provider_name="fake",
            intent=ResearchQueryIntent.ENGINEERING_BLOG,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.diagnostics["intent"], "engineering_blog")

    def test_rank_is_preserved_in_diagnostics(self) -> None:
        raw_result = RawSearchResult(
            query="AI automation best practices",
            title="Best practices",
            url="https://example.com/best-practices",
            snippet="Helpful practices.",
            rank=4,
            provider_name="fake",
            intent=ResearchQueryIntent.BEST_PRACTICES,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.diagnostics["rank"], 4)

    def test_origin_reason_is_human_readable(self) -> None:
        raw_result = RawSearchResult(
            query="AI automation Zapier Make implementation guide",
            title="Guide",
            url="https://example.com/guide",
            snippet="Snippet",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.IMPLEMENTATION_GUIDE,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertIn("found by fake search provider", candidate.origin_reason)
        self.assertIn("AI automation Zapier Make implementation guide", candidate.origin_reason)

    def test_provider_result_converts_all_raw_results_to_candidate_inputs(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        first_query = plan.query_items[0].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "Snippet"},
                    {"title": "Checklist", "url": "https://example.com/checklist", "snippet": "Checklist"},
                ]
            }
        )
        provider_result = search_research_query_plan(plan, provider)

        candidates = search_provider_result_to_candidate_inputs(provider_result)

        self.assertEqual(len(candidates), 2)

    def test_ordering_is_stable(self) -> None:
        raw_results = [
            RawSearchResult(
                query="Travel planning expert advice",
                title="First",
                url="https://example.com/1",
                snippet="One",
                rank=1,
                provider_name="fake",
                intent=ResearchQueryIntent.EXPERT_ADVICE,
            ),
            RawSearchResult(
                query="Travel planning expert advice",
                title="Second",
                url="https://example.com/2",
                snippet="Two",
                rank=2,
                provider_name="fake",
                intent=ResearchQueryIntent.EXPERT_ADVICE,
            ),
        ]

        candidates = search_results_to_candidate_inputs(raw_results)

        self.assertEqual(candidates[0].url, "https://example.com/1")
        self.assertEqual(candidates[1].url, "https://example.com/2")

    def test_empty_provider_result_returns_empty_candidate_list(self) -> None:
        candidates = search_provider_result_to_candidate_inputs(
            search_research_query_plan(
                build_research_query_plan(_TopicStub("Travel planning", ["family travel"])),
                FakeSearchProvider({}),
            )
        )

        self.assertEqual(candidates, [])

    def test_adapter_does_not_evaluate_candidates(self) -> None:
        raw_result = RawSearchResult(
            query="Travel planning expert advice",
            title="Guide",
            url="https://example.com/guide",
            snippet="Snippet",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.EXPERT_ADVICE,
        )

        with patch("services.sources.candidates.evaluate_source_candidate", side_effect=AssertionError("should not run")):
            candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.url, "https://example.com/guide")

    def test_adapter_does_not_require_http_or_template_context(self) -> None:
        raw_result = RawSearchResult(
            query="Travel planning expert advice",
            title="Guide",
            url="https://example.com/guide",
            snippet="Snippet",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.EXPERT_ADVICE,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.title, "Guide")

    @patch("socket.create_connection", side_effect=AssertionError("network should not be used"))
    def test_adapter_does_not_call_external_network(self, _mock_network) -> None:
        raw_result = RawSearchResult(
            query="Travel planning expert advice",
            title="Guide",
            url="https://example.com/guide",
            snippet="Snippet",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.EXPERT_ADVICE,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.url, "https://example.com/guide")


class SourceSearchCandidateAdapterPersistenceTests(TestCase):
    def test_adapter_does_not_create_topic_sources(self) -> None:
        user = get_user_model().objects.create_user(username="search-candidate-user", password="pw")
        Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()

        raw_result = RawSearchResult(
            query="Infant sleep official guidelines",
            title="Safe sleep guidance",
            url="https://example.com/safe-sleep",
            snippet="Evidence-based infant safe sleep guide.",
            rank=1,
            provider_name="fake",
            intent=ResearchQueryIntent.OFFICIAL_GUIDELINES,
        )

        candidate = search_result_to_candidate_input(raw_result)

        self.assertEqual(candidate.url, "https://example.com/safe-sleep")
        self.assertEqual(TopicSource.objects.count(), before)
