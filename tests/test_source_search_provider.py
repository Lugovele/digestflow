import json
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings

from apps.topics.models import Topic, TopicSource
from services.sources.research_queries import build_research_query_plan
from services.sources.serpapi_provider import SearchProviderRuntimeError, SerpApiSearchProvider
from services.sources.search_provider import (
    FakeSearchProvider,
    RawSearchResult,
    search_research_query_plan,
)


class _TopicStub:
    def __init__(self, name: str, keywords) -> None:
        self.name = name
        self.keywords = keywords


class SourceSearchProviderTests(SimpleTestCase):
    def test_search_provider_boundary_accepts_research_query_plan(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "SIDS"]))
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.diagnostics["query_count"], len(plan.query_items))

    def test_fake_provider_returns_deterministic_raw_search_results(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make", "n8n"]))
        first_query = plan.query_items[0].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {
                        "title": "AI automation guide",
                        "url": "https://example.com/ai-automation",
                        "snippet": "Practical automation guide.",
                    }
                ]
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(len(result.results), 1)
        self.assertIsInstance(result.results[0], RawSearchResult)
        self.assertEqual(result.results[0].title, "AI automation guide")

    def test_originating_query_is_preserved(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "newborn"]))
        first_query = plan.query_items[0].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {"title": "Safe sleep article", "url": "https://example.com/safe-sleep", "snippet": "Snippet"}
                ]
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results[0].query, first_query)

    def test_originating_intent_is_preserved(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "newborn"]))
        first_item = plan.query_items[0]
        provider = FakeSearchProvider(
            {
                first_item.query: [
                    {"title": "Safe sleep article", "url": "https://example.com/safe-sleep", "snippet": "Snippet"}
                ]
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results[0].intent, first_item.intent)

    def test_rank_is_assigned_predictably(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make"]))
        first_query = plan.query_items[0].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {"title": "First", "url": "https://example.com/1", "snippet": "One"},
                    {"title": "Second", "url": "https://example.com/2", "snippet": "Two"},
                ]
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results[0].rank, 1)
        self.assertEqual(result.results[1].rank, 2)

    def test_provider_name_is_exposed(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.provider_name, "fake")

    @patch("services.sources.serpapi_provider.urlopen")
    def test_serpapi_provider_maps_organic_results(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "organic_results": [
                    {
                        "position": 1,
                        "title": "AI automation guide",
                        "link": "https://example.com/ai-automation",
                        "snippet": "Practical automation guide.",
                        "source": "Example",
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        provider = SerpApiSearchProvider(api_key="test-key")

        results = provider.search("AI automation", intent=plan_query_intent())

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "AI automation guide")
        self.assertEqual(results[0]["url"], "https://example.com/ai-automation")
        self.assertEqual(results[0]["snippet"], "Practical automation guide.")
        self.assertEqual(results[0]["rank"], 1)
        self.assertEqual(results[0]["source"], "Example")
        self.assertEqual(results[0]["published_at"], "")

    @patch("services.sources.serpapi_provider.urlopen")
    def test_serpapi_provider_includes_default_recency_filter(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"organic_results": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        provider = SerpApiSearchProvider(api_key="test-key")

        provider.search("AI automation", intent=plan_query_intent())

        request = mock_urlopen.call_args.args[0]
        parsed = urlparse(request.full_url)
        params = parse_qs(parsed.query)
        self.assertEqual(params.get("tbs"), ["qdr:m"])

    @override_settings(SEARCH_RECENCY_MONTHS=3)
    @patch("services.sources.serpapi_provider.urlopen")
    def test_serpapi_provider_uses_configurable_recency_filter(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"organic_results": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        provider = SerpApiSearchProvider(api_key="test-key", recency_months=3, time_filter="qdr:m3")

        provider.search("AI automation", intent=plan_query_intent())

        request = mock_urlopen.call_args.args[0]
        parsed = urlparse(request.full_url)
        params = parse_qs(parsed.query)
        self.assertEqual(params.get("tbs"), ["qdr:m3"])

    @patch("services.sources.serpapi_provider.urlopen")
    def test_serpapi_provider_handles_empty_organic_results_safely(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps({"organic_results": []}).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        provider = SerpApiSearchProvider(api_key="test-key")

        results = provider.search("AI automation", intent=plan_query_intent())

        self.assertEqual(results, [])

    @patch("services.sources.serpapi_provider.urlopen")
    def test_serpapi_provider_raises_structured_runtime_error_on_http_failure(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = HTTPError(
            url="https://serpapi.com/search.json",
            code=503,
            msg="service unavailable",
            hdrs=None,
            fp=None,
        )
        provider = SerpApiSearchProvider(api_key="test-key")

        with self.assertRaises(SearchProviderRuntimeError) as exc_info:
            provider.search("AI automation", intent=plan_query_intent())

        self.assertEqual(exc_info.exception.diagnostics["provider_http_status"], 503)
        self.assertEqual(exc_info.exception.diagnostics["provider_error_type"], "http_error")

    def test_diagnostics_include_query_count_and_raw_result_count(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        first_query = plan.query_items[0].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "Snippet"}
                ]
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.diagnostics["query_count"], len(plan.query_items))
        self.assertEqual(result.diagnostics["raw_result_count"], 1)
        self.assertTrue(result.diagnostics["per_query_result_counts"])

    def test_empty_provider_results_are_handled_safely(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results, ())
        self.assertEqual(result.diagnostics["raw_result_count"], 0)

    def test_duplicate_raw_urls_are_handled_predictably(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make"]))
        first_query = plan.query_items[0].query
        second_query = plan.query_items[1].query
        provider = FakeSearchProvider(
            {
                first_query: [
                    {"title": "Guide", "url": "https://example.com/guide", "snippet": "Snippet"}
                ],
                second_query: [
                    {"title": "Guide copy", "url": "https://example.com/guide", "snippet": "Duplicate"}
                ],
            }
        )

        result = search_research_query_plan(plan, provider)

        self.assertEqual(len(result.results), 1)
        self.assertEqual(result.diagnostics["duplicate_url_count"], 1)

    def test_no_source_candidate_evaluation_happens_in_this_layer(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        provider = FakeSearchProvider({})

        with patch("services.sources.candidates.evaluate_source_candidates", side_effect=AssertionError("should not run")):
            result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results, ())

    def test_no_http_or_template_context_is_required(self) -> None:
        plan = build_research_query_plan(_TopicStub("Education for teenagers", ["study habits"]))
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.provider_name, "fake")

    @patch("socket.create_connection", side_effect=AssertionError("network should not be used"))
    def test_no_external_network_or_api_call_is_made(self, _mock_network) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel"]))
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results, ())

    @patch("services.sources.serpapi_provider.urlopen")
    def test_search_provider_boundary_records_serpapi_runtime_failures_in_diagnostics(self, mock_urlopen) -> None:
        mock_urlopen.side_effect = HTTPError(
            url="https://serpapi.com/search.json",
            code=503,
            msg="service unavailable",
            hdrs=None,
            fp=None,
        )
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make"]))
        provider = SerpApiSearchProvider(api_key="test-key")

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results, ())
        self.assertEqual(result.provider_name, "serpapi")
        self.assertEqual(result.diagnostics["provider_error_count"], len(plan.query_items))
        self.assertTrue(result.diagnostics["provider_errors"])
        self.assertNotIn("test-key", json.dumps(result.diagnostics))

    @override_settings(SEARCH_RECENCY_MONTHS=1)
    @patch("services.sources.serpapi_provider.urlopen")
    def test_search_provider_boundary_surfaces_recency_diagnostics(self, mock_urlopen) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "organic_results": [
                    {
                        "position": 1,
                        "title": "Recent report",
                        "link": "https://example.com/recent-report",
                        "snippet": "Snippet",
                        "source": "Example",
                        "date": "2026-05-02",
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make"]))
        provider = SerpApiSearchProvider(api_key="test-key", recency_months=1, time_filter="qdr:m")

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results[0].diagnostics["provider_published_at"], "2026-05-02")


class SourceSearchProviderPersistenceTests(TestCase):
    def test_search_provider_boundary_does_not_create_topic_sources(self) -> None:
        user = get_user_model().objects.create_user(username="search-provider-user", password="pw")
        topic = Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()
        plan = build_research_query_plan(topic)
        provider = FakeSearchProvider({})

        result = search_research_query_plan(plan, provider)

        self.assertEqual(result.results, ())
        self.assertEqual(TopicSource.objects.count(), before)


def plan_query_intent():
    plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier"]))
    return plan.query_items[0].intent
