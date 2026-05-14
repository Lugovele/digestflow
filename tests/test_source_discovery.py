from unittest.mock import patch

from django.test import SimpleTestCase

from apps.topics.models import TopicSourceMode
from services.sources.discovery import (
    CuratedSourceSeed,
    TopicSourceDiscoveryRequest,
    discover_sources,
    resolve_source_candidates,
)


class SourceDiscoveryTests(SimpleTestCase):
    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_discover_sources_returns_curated_candidates_for_ai_agents(self, mock_fetch_dev_to_article_list) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 7

        candidates = discover_sources("AI agents", limit=6)

        self.assertGreaterEqual(len(candidates), 5)
        self.assertEqual(candidates[0]["source_type"], "devto_tag")
        self.assertEqual(candidates[0]["platform"], "dev.to")
        self.assertIn("quality_estimate", candidates[0])
        self.assertIn("recent_article_count", candidates[0])
        self.assertTrue(any(candidate["url"] == "https://dev.to/t/security" for candidate in candidates))

    @patch("services.sources.discovery.get_rss_debug_snapshot")
    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_discover_sources_keeps_manual_source_in_candidates(
        self,
        mock_fetch_dev_to_article_list,
        mock_get_rss_debug_snapshot,
    ) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 5
        mock_get_rss_debug_snapshot.return_value = {"total_entries": 3}

        candidates = discover_sources("MCP", manual_source_url="https://example.com/feed.xml", limit=5)

        self.assertEqual(candidates[0]["url"], "https://example.com/feed.xml")
        self.assertEqual(candidates[0]["quality_estimate"], "manual")
        self.assertTrue(candidates[0]["default_selected"])

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_resolve_source_candidates_supports_curated_only_mode(self, mock_fetch_dev_to_article_list) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="AI agents",
                source_mode=TopicSourceMode.CURATED_ONLY,
                curated_sources=[
                    CuratedSourceSeed(
                        url="https://dev.to/t/security",
                        title="Saved security source",
                        description="Curated security coverage.",
                    )
                ],
                manual_source_url="https://example.com/feed.xml",
                limit=5,
            )
        )

        self.assertEqual([candidate["candidate_origin"] for candidate in candidates], ["curated", "curated"])
        self.assertEqual(candidates[0]["url"], "https://example.com/feed.xml")
        self.assertEqual(candidates[1]["url"], "https://dev.to/t/security")

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_resolve_source_candidates_supports_hybrid_mode(self, mock_fetch_dev_to_article_list) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="AI agents",
                source_mode=TopicSourceMode.HYBRID,
                curated_sources=[
                    CuratedSourceSeed(
                        url="https://dev.to/t/security",
                        title="Saved security source",
                        description="Curated security coverage.",
                    )
                ],
                limit=5,
            )
        )

        self.assertEqual(candidates[0]["candidate_origin"], "curated")
        self.assertTrue(any(candidate["candidate_origin"] == "discovered" for candidate in candidates[1:]))
        self.assertTrue(any(candidate["url"] == "https://dev.to/t/ai" for candidate in candidates))

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_discovery_only_mode_keeps_explicit_manual_source_but_skips_curated_pool(
        self, mock_fetch_dev_to_article_list
    ) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="AI agents",
                source_mode=TopicSourceMode.DISCOVERY_ONLY,
                manual_source_url="https://example.com/feed.xml",
                curated_sources=[
                    CuratedSourceSeed(
                        url="https://dev.to/t/security",
                        title="Saved security source",
                    )
                ],
                limit=5,
            )
        )

        self.assertEqual(candidates[0]["url"], "https://example.com/feed.xml")
        self.assertTrue(all(candidate["candidate_origin"] != "curated" for candidate in candidates[1:]))
        self.assertTrue(any(candidate["candidate_origin"] == "discovered" for candidate in candidates[1:]))

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_resolve_source_candidates_dedupes_by_normalized_url(self, mock_fetch_dev_to_article_list) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="AI agents",
                source_mode=TopicSourceMode.HYBRID,
                manual_source_url="https://dev.to/t/ai",
                curated_sources=[
                    CuratedSourceSeed(
                        url="https://dev.to/t/ai",
                        title="Saved ai source",
                    )
                ],
                limit=5,
            )
        )

        ai_candidates = [candidate for candidate in candidates if candidate["url"] == "https://dev.to/t/ai"]
        self.assertEqual(len(ai_candidates), 1)

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_focus_terms_influence_discovery_templates(self, mock_fetch_dev_to_article_list) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="Operations",
                focus_terms=("workflow automation",),
                source_mode=TopicSourceMode.DISCOVERY_ONLY,
                limit=5,
            )
        )

        candidate_urls = {candidate["url"] for candidate in candidates}
        self.assertIn("https://dev.to/t/python", candidate_urls)
        self.assertIn("https://dev.to/t/devops", candidate_urls)

    @patch("services.sources.discovery.fetch_dev_to_article_list")
    def test_non_technical_topic_does_not_fall_back_to_python_devops_sources(
        self,
        mock_fetch_dev_to_article_list,
    ) -> None:
        mock_fetch_dev_to_article_list.return_value = [{"id": 1}] * 9

        candidates = resolve_source_candidates(
            TopicSourceDiscoveryRequest(
                topic="Baby sleeping",
                focus_terms=("infant sleep", "baby bedtime", "sleep regression"),
                source_mode=TopicSourceMode.DISCOVERY_ONLY,
                limit=5,
            )
        )

        self.assertEqual(candidates, [])
