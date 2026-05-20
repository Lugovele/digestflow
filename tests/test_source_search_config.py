from django.test import SimpleTestCase, override_settings

from apps.topics.models import TopicSourceMode
from services.sources.research_orchestrator import run_source_research
from services.sources.search_config import resolve_configured_search_provider
from services.sources.serpapi_provider import SerpApiSearchProvider


class _TopicStub:
    def __init__(self, name: str, keywords, source_mode: str) -> None:
        self.name = name
        self.keywords = keywords
        self.source_mode = source_mode

    @property
    def uses_source_discovery(self) -> bool:
        return self.source_mode in {
            TopicSourceMode.DISCOVERY_ONLY,
            TopicSourceMode.HYBRID,
        }


class SourceSearchConfigTests(SimpleTestCase):
    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_provider_disabled_returns_disabled_diagnostics(self) -> None:
        topic = _TopicStub("AI workflows", ["automation"], TopicSourceMode.DISCOVERY_ONLY)

        resolution = resolve_configured_search_provider(topic)
        result = run_source_research(topic)

        self.assertIsNone(resolution.provider)
        self.assertEqual(resolution.diagnostics["search_provider_status"], "disabled")
        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.diagnostics["search_provider_status"], "disabled")
        self.assertEqual(result.diagnostics["research_execution_status"], "blocked")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_provider_missing_required_config_returns_missing_config_diagnostics(self) -> None:
        topic = _TopicStub("AI workflows", ["automation"], TopicSourceMode.DISCOVERY_ONLY)

        result = run_source_research(topic)

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.diagnostics["search_provider_status"], "missing_config")
        self.assertEqual(
            result.diagnostics["search_provider_missing_settings"],
            ("SEARCH_PROVIDER_API_KEY",),
        )
        self.assertIn("missing required credentials", result.diagnostics["search_provider_error"])

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    def test_serpapi_provider_with_api_key_resolves_as_ready(self) -> None:
        topic = _TopicStub("AI workflows", ["automation"], TopicSourceMode.DISCOVERY_ONLY)

        resolution = resolve_configured_search_provider(topic)

        self.assertIsInstance(resolution.provider, SerpApiSearchProvider)
        self.assertEqual(resolution.diagnostics["search_provider_status"], "ready")
        self.assertEqual(resolution.diagnostics["search_provider_name"], "serpapi")
        self.assertTrue(resolution.diagnostics["search_provider_configured"])

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="tavily",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    def test_unsupported_provider_returns_not_implemented_diagnostics(self) -> None:
        topic = _TopicStub("AI workflows", ["automation"], TopicSourceMode.DISCOVERY_ONLY)

        result = run_source_research(topic)

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.diagnostics["search_provider_status"], "not_implemented")
        self.assertEqual(result.diagnostics["search_provider_name"], "tavily")
        self.assertIn("not implemented yet", result.diagnostics["search_provider_error"])

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_manual_source_only_path_is_not_broken_when_provider_is_disabled(self) -> None:
        topic = _TopicStub("Manual only", ["manual source"], TopicSourceMode.CURATED_ONLY)

        result = run_source_research(topic)

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.diagnostics["search_provider_status"], "disabled")
        self.assertFalse(result.diagnostics["research_required_for_topic"])
        self.assertEqual(result.diagnostics["research_execution_status"], "skipped_not_required")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=False,
        SEARCH_PROVIDER="",
        SEARCH_PROVIDER_API_KEY="",
    )
    def test_research_required_path_returns_clear_diagnostics_instead_of_exception(self) -> None:
        topic = _TopicStub("Research required", ["automation"], TopicSourceMode.HYBRID)

        result = run_source_research(topic)

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(result.candidate_inputs, ())
        self.assertEqual(result.review_items, ())
        self.assertEqual(result.diagnostics["search_provider_status"], "disabled")
        self.assertTrue(result.diagnostics["research_required_for_topic"])
        self.assertEqual(result.diagnostics["research_execution_status"], "blocked")
