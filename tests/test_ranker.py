import json
from pathlib import Path

from django.test import SimpleTestCase

from services.processing.ranker import rank_source_items

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures" / "ranking"


def load_ranking_fixture(name: str) -> dict:
    return json.loads((FIXTURES_DIR / f"{name}.json").read_text(encoding="utf-8"))


class RankSourceItemsTests(SimpleTestCase):
    def test_keyword_match_improves_ranking_score(self):
        items = [
            {
                "title": "General workflow note",
                "url": "https://example.com/low",
                "source": "Example Blog",
                "snippet": "A short neutral update without measurable result.",
            },
            {
                "title": "AI automation rollout",
                "url": "https://example.com/high",
                "source": "Example Research",
                "snippet": (
                    "The team reduced manual reporting time by 35% and improved weekly review "
                    "quality after changing the workflow."
                ),
            },
        ]

        selected, ranking_scores = rank_source_items(
            items,
            keywords=["AI automation"],
            top_n=2,
            min_quality_score=0.0,
        )

        self.assertEqual(selected[0]["url"], "https://example.com/high")
        self.assertGreater(ranking_scores[0]["score"], ranking_scores[1]["score"])

    def test_excluded_keywords_reduce_quality_and_filter_weak_items(self):
        items = [
            {
                "title": "AI automation success",
                "url": "https://example.com/keep",
                "source": "Example Research",
                "snippet": "The rollout reduced manual work by 30% and improved reporting quality.",
            },
            {
                "title": "Crypto trend report",
                "url": "https://example.com/drop",
                "source": "Example Report",
                "snippet": "Crypto growth remained strong across several exchanges this quarter.",
            },
        ]

        selected, _ = rank_source_items(
            items,
            keywords=["AI automation"],
            excluded_keywords=["crypto"],
            top_n=2,
            min_quality_score=0.4,
        )

        self.assertEqual([item["url"] for item in selected], ["https://example.com/keep"])

    def test_equal_scores_keep_original_order(self):
        items = [
            {
                "title": "First",
                "url": "https://example.com/first",
                "source": "Example Blog",
                "snippet": "Neutral snippet with no ranking keywords at all.",
            },
            {
                "title": "Second",
                "url": "https://example.com/second",
                "source": "Example Blog",
                "snippet": "Another neutral snippet with no ranking words.",
            },
        ]

        selected, _ = rank_source_items(items, top_n=2, min_quality_score=0.0)

        self.assertEqual(
            [item["url"] for item in selected],
            ["https://example.com/first", "https://example.com/second"],
        )

    def test_top_n_limits_selected_items(self):
        items = [
            {
                "title": "One",
                "url": "https://example.com/1",
                "source": "Example Research",
                "snippet": "Reduced time by 20% with a longer operational summary for the team.",
            },
            {
                "title": "Two",
                "url": "https://example.com/2",
                "source": "Example Blog",
                "snippet": "Improved handoff quality in a measurable way for one workflow.",
            },
            {
                "title": "Three",
                "url": "https://example.com/3",
                "source": "Example Report",
                "snippet": "Growth reached 12% after the rollout and cut review cycles.",
            },
        ]

        selected, _ = rank_source_items(items, top_n=2, min_quality_score=0.0)

        self.assertEqual(len(selected), 2)

    def test_ranking_scores_contains_explainable_metadata(self):
        items = [
            {
                "title": "One",
                "url": "https://example.com/1",
                "source": "Example Blog",
                "snippet": "A neutral source snippet.",
                "metadata": {
                    "content_tier": "rich_summary",
                    "content_length": 160,
                },
            }
        ]

        _, ranking_scores = rank_source_items(items, top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["title"], "One")
        self.assertEqual(ranking_scores[0]["url"], "https://example.com/1")
        self.assertEqual(ranking_scores[0]["source_name"], "Example Blog")
        self.assertEqual(ranking_scores[0]["score"], 0.0)
        self.assertEqual(ranking_scores[0]["quality_score"], 0.0)
        self.assertEqual(ranking_scores[0]["final_quality_score"], 0.0)
        self.assertEqual(ranking_scores[0]["content_tier"], "rich_summary")
        self.assertEqual(ranking_scores[0]["content_length"], 160)
        self.assertEqual(ranking_scores[0]["primary_article_type"], "lightweight_post")
        self.assertEqual(ranking_scores[0]["secondary_article_tags"], [])
        self.assertEqual(ranking_scores[0]["article_type"], "lightweight_post")
        self.assertEqual(ranking_scores[0]["article_type_score_modifier"], -0.5)
        self.assertIn("topic_relevance_score", ranking_scores[0])
        self.assertIn("topic_relevance_reason", ranking_scores[0])
        self.assertIn("relevance_signals", ranking_scores[0])
        self.assertIn("weak_relevance_signals", ranking_scores[0])
        self.assertIn("missing_relevance_signals", ranking_scores[0])
        self.assertIn("topic_specificity_score", ranking_scores[0])
        self.assertIn("topic_specificity_reason", ranking_scores[0])
        self.assertIn("specificity_signals", ranking_scores[0])
        self.assertIn("generic_topic_signals", ranking_scores[0])
        self.assertIn("evidence_score", ranking_scores[0])
        self.assertIn("practical_value_score", ranking_scores[0])
        self.assertIn("novelty_score", ranking_scores[0])
        self.assertIn("article_type", ranking_scores[0])
        self.assertIn("primary_article_type", ranking_scores[0])
        self.assertIn("secondary_article_tags", ranking_scores[0])
        self.assertIn("weighted_secondary_tags", ranking_scores[0])
        self.assertIn("dominant_tags", ranking_scores[0])
        self.assertIn("supporting_tags", ranking_scores[0])
        self.assertIn("weak_tags", ranking_scores[0])
        self.assertIn("article_type_reason", ranking_scores[0])
        self.assertIn("article_type_score_modifier", ranking_scores[0])
        self.assertIn("classification_signal_summary", ranking_scores[0])
        self.assertIn("dominant_theme_reason", ranking_scores[0])
        self.assertIn("primary_type_override_reason", ranking_scores[0])
        self.assertIn("heading_diagnostics", ranking_scores[0])
        self.assertTrue(ranking_scores[0]["quality_reasons"])
        self.assertTrue(ranking_scores[0]["rejection_reasons"])
        self.assertEqual(
            ranking_scores[0]["diagnostic_warnings"],
            ["no topic keywords were provided; relevance is based on article signals only"],
        )

    def test_community_update_article_type_is_detected(self):
        items = [
            {
                "title": "Congrats to the challenge winners",
                "url": "https://example.com/community",
                "source": "Example Community",
                "snippet": "A short roundup of this week's challenge winners and community highlights.",
                "metadata": {"content_tier": "rich_summary", "content_length": 140},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "community_update")
        self.assertEqual(ranking_scores[0]["article_type"], "community_update")
        self.assertEqual(ranking_scores[0]["secondary_article_tags"], ["event"])
        self.assertEqual(ranking_scores[0]["dominant_tags"], ["event"])
        self.assertEqual(ranking_scores[0]["supporting_tags"], [])
        self.assertEqual(ranking_scores[0]["weak_tags"], [])
        self.assertLess(ranking_scores[0]["article_type_score_modifier"], 0)

    def test_multi_agent_article_gets_nonzero_relevance_for_ai_agents_topic(self):
        items = [
            {
                "title": "Deploying a Multi-Agent System with Terraform and Cloud Run",
                "url": "https://example.com/multi-agent",
                "source": "Example Engineering",
                "snippet": (
                    "This guide explains how to deploy a multi-agent system with shared memory, "
                    "workflow orchestration, and Cloud Run infrastructure."
                ),
                "metadata": {
                    "content_tier": "full_article",
                    "content_length": 820,
                },
            }
        ]

        _, ranking_scores = rank_source_items(
            items,
            keywords=["ai agents"],
            top_n=1,
            min_quality_score=0.0,
        )

        self.assertGreater(ranking_scores[0]["topic_relevance_score"], 0.0)
        self.assertGreaterEqual(ranking_scores[0]["topic_relevance_score"], 3.0)
        self.assertGreaterEqual(ranking_scores[0]["topic_specificity_score"], 1.5)
        self.assertGreater(ranking_scores[0]["quality_score"], 0.0)
        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertEqual(ranking_scores[0]["article_type"], "tutorial")
        self.assertEqual(
            ranking_scores[0]["secondary_article_tags"],
            ["multi_agent", "cloud", "devops", "terraform", "google_cloud", "memory"],
        )
        self.assertEqual(
            ranking_scores[0]["dominant_tags"],
            ["multi_agent", "cloud", "devops", "terraform", "google_cloud"],
        )
        self.assertIn("memory", ranking_scores[0]["supporting_tags"])
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["multi_agent"]["strength"], 2.0)
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["memory"]["strength"], 1.0)
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["terraform"]["title_matches"], ["terraform"])
        self.assertGreaterEqual(ranking_scores[0]["weighted_secondary_tags"]["terraform"]["editorial_weight"], 2.0)
        self.assertTrue(ranking_scores[0]["weighted_secondary_tags"]["terraform"]["centrality_reason"])
        self.assertIn(
            ranking_scores[0]["quality_reasons"][0],
            {"strong relevance to topic", "relevant topic signal present"},
        )

    def test_oauth_agents_article_is_classified_as_architecture_security(self):
        items = [
            {
                "title": "How to Authorize AI Agents Using Token Exchange Open Standards",
                "url": "https://example.com/how-to-agents",
                "source": "Example Engineering",
                "snippet": (
                    "This article explains authorization, token exchange, and security controls for "
                    "AI agents using OAuth-style standards and policy boundaries."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 760},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "architecture_security")
        self.assertEqual(ranking_scores[0]["article_type"], "architecture_security")
        self.assertGreaterEqual(ranking_scores[0]["topic_relevance_score"], 3.0)
        self.assertGreaterEqual(ranking_scores[0]["topic_specificity_score"], 1.5)
        self.assertEqual(
            ranking_scores[0]["secondary_article_tags"],
            ["ai_agents", "security", "auth", "oauth"],
        )
        self.assertEqual(
            ranking_scores[0]["dominant_tags"],
            ["ai_agents", "security", "auth", "oauth"],
        )
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["security"]["strength"], 2.0)
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["oauth"]["strength"], 2.0)
        self.assertGreaterEqual(ranking_scores[0]["article_type_score_modifier"], 0.0)

    def test_introducing_article_is_classified_as_announcement(self):
        items = [
            {
                "title": "Introducing the new agent console release",
                "url": "https://example.com/announcement",
                "source": "Example Product",
                "snippet": "A new release introduces updated controls and a refreshed console experience.",
                "metadata": {"content_tier": "rich_summary", "content_length": 180},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "announcement")
        self.assertEqual(ranking_scores[0]["article_type"], "announcement")
        self.assertEqual(ranking_scores[0]["secondary_article_tags"], ["product_update"])
        self.assertEqual(ranking_scores[0]["article_type_score_modifier"], 0.0)

    def test_marketing_article_is_classified_as_marketing(self):
        items = [
            {
                "title": "Customer story: scale your AI agents faster",
                "url": "https://example.com/marketing",
                "source": "Example Product",
                "snippet": "Book a demo to see how customers scale their AI agents with our new pricing plan.",
                "metadata": {"content_tier": "rich_summary", "content_length": 170},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "marketing")
        self.assertEqual(ranking_scores[0]["article_type"], "marketing")
        self.assertEqual(ranking_scores[0]["secondary_article_tags"], ["ai_agents"])
        self.assertLess(ranking_scores[0]["article_type_score_modifier"], 0)

    def test_local_testing_multi_agent_article_gets_hierarchical_tags(self):
        items = [
            {
                "title": "Local Testing of a Multi-Agent System with Memory",
                "url": "https://example.com/local-testing",
                "source": "Example Engineering",
                "snippet": (
                    "A practical guide to local testing for a multi-agent system with shared memory, "
                    "test harnesses, and workflow checks."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 700},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertEqual(ranking_scores[0]["article_type"], "tutorial")
        self.assertGreaterEqual(ranking_scores[0]["topic_specificity_score"], 1.0)
        self.assertEqual(
            ranking_scores[0]["secondary_article_tags"],
            ["multi_agent", "testing", "memory"],
        )
        self.assertEqual(ranking_scores[0]["dominant_tags"], ["multi_agent", "testing", "memory"])

    def test_heading_match_increases_editorial_weight_for_relevant_tag(self):
        items = [
            {
                "title": "Deploying agents in production",
                "url": "https://example.com/heading-match",
                "source": "Example Engineering",
                "content": (
                    "This guide explains deployment tradeoffs.\n"
                    "## Long-Term Memory\n"
                    "Long-term memory helps the agent system preserve context.\n"
                    "Long-term memory is also part of the production rollout plan."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 820},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        memory_tag = ranking_scores[0]["weighted_secondary_tags"]["memory"]
        self.assertIn("long term memory", memory_tag["heading_matches"])
        self.assertGreater(memory_tag["heading_weight_component"], 0.0)
        self.assertGreaterEqual(memory_tag["editorial_weight"], 2.0)
        self.assertIn("memory", ranking_scores[0]["supporting_tags"])
        heading_diagnostics = ranking_scores[0]["heading_diagnostics"]
        self.assertEqual(heading_diagnostics["heading_source"], "inferred")
        self.assertGreater(heading_diagnostics["heading_count"], 0)
        self.assertIn("Long-Term Memory", heading_diagnostics["detected_headings"])
        self.assertIn("long term memory", heading_diagnostics["normalized_headings"])
        self.assertIn("memory", heading_diagnostics["matched_heading_tags"])
        self.assertIn("Long-Term Memory", heading_diagnostics["matched_heading_tags"]["memory"]["matches"])
        self.assertFalse(memory_tag["heading_boost_capped"])
        self.assertIn("heading", memory_tag["dominant_signal_sources"])

    def test_plain_text_article_reports_no_headings(self):
        items = [
            {
                "title": "Agent operations note",
                "url": "https://example.com/no-headings",
                "source": "Example Engineering",
                "content": "This article is plain text without section markers or heading structure.",
                "metadata": {"content_tier": "full_article", "content_length": 420},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        heading_diagnostics = ranking_scores[0]["heading_diagnostics"]
        self.assertEqual(heading_diagnostics["heading_source"], "none")
        self.assertEqual(heading_diagnostics["heading_count"], 0)
        self.assertEqual(heading_diagnostics["detected_headings"], [])
        self.assertEqual(heading_diagnostics["normalized_headings"], [])
        self.assertEqual(heading_diagnostics["matched_heading_tags"], {})

    def test_explicit_heading_metadata_is_preserved_in_heading_diagnostics(self):
        items = [
            {
                "title": "Designing a team of agents",
                "url": "https://example.com/designing-agents",
                "source": "DEV Community",
                "content": "Flattened article content without visible heading markers.",
                "metadata": {
                    "content_tier": "full_article",
                    "content_length": 820,
                    "headings": ["Architecture", "Long-Term Memory", "Governance Layer"],
                    "raw_html_heading_count": 4,
                    "extracted_heading_count": 3,
                    "heading_extraction_strategy": "markdown_headings",
                    "sample_detected_headings": ["Architecture", "Long-Term Memory"],
                },
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        heading_diagnostics = ranking_scores[0]["heading_diagnostics"]
        self.assertEqual(heading_diagnostics["heading_source"], "explicit")
        self.assertEqual(heading_diagnostics["heading_count"], 3)
        self.assertEqual(heading_diagnostics["raw_html_heading_count"], 4)
        self.assertEqual(heading_diagnostics["extracted_heading_count"], 3)
        self.assertEqual(heading_diagnostics["heading_extraction_strategy"], "markdown_headings")
        self.assertEqual(heading_diagnostics["sample_detected_headings"], ["Architecture", "Long-Term Memory"])
        self.assertEqual(
            heading_diagnostics["detected_headings"],
            ["Architecture", "Long-Term Memory", "Governance Layer"],
        )

    def test_generic_cloud_devops_article_does_not_become_architecture_security(self):
        items = [
            {
                "title": "Deploying workflow services to Cloud Run",
                "url": "https://example.com/cloud-devops",
                "source": "Example Engineering",
                "snippet": (
                    "A deployment guide for workflow services using infrastructure automation, "
                    "Cloud Run, and rollout steps."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 680},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertNotEqual(ranking_scores[0]["article_type"], "architecture_security")
        self.assertLess(ranking_scores[0]["topic_specificity_score"], 1.0)
        self.assertNotIn("security", ranking_scores[0]["secondary_article_tags"])
        self.assertNotIn("auth", ranking_scores[0]["secondary_article_tags"])

    def test_deployment_guide_with_security_section_stays_tutorial(self):
        items = [
            {
                "title": "Deploying a Multi-Agent System with Terraform and Cloud Run",
                "url": "https://example.com/deployment-security",
                "source": "Example Engineering",
                "snippet": (
                    "A deployment guide for a multi-agent system using Terraform and Cloud Run. "
                    "The article also recommends service accounts, least privilege, permissions, and credentials hygiene."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 930},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertEqual(
            ranking_scores[0]["dominant_tags"],
            ["multi_agent", "cloud", "devops", "terraform", "google_cloud"],
        )
        self.assertIn("security", ranking_scores[0]["supporting_tags"])
        self.assertIn("auth", ranking_scores[0]["supporting_tags"])
        self.assertNotIn("security", ranking_scores[0]["dominant_tags"])
        self.assertNotIn("auth", ranking_scores[0]["dominant_tags"])
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["security"]["strength"], 1.0)
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["auth"]["strength"], 1.0)
        self.assertIn("editorial center", ranking_scores[0]["primary_type_override_reason"])

    def test_incidental_security_word_does_not_add_security_tag(self):
        items = [
            {
                "title": "Deploying a Multi-Agent System with Terraform and Cloud Run",
                "url": "https://example.com/incidental-security",
                "source": "Example Engineering",
                "snippet": (
                    "This tutorial covers deployment, Terraform, Cloud Run, and workflow checks. "
                    "A final note mentions security once without discussing authorization or tokens."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 760},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertNotIn("security", ranking_scores[0]["secondary_article_tags"])
        self.assertIn("security", ranking_scores[0]["weak_tags"])
        self.assertEqual(ranking_scores[0]["weighted_secondary_tags"]["security"]["strength"], 0.5)

    def test_body_only_repetition_saturates_without_forcing_dominance(self):
        repeated_cloud = (
            (
                "This opening section discusses general agent operations, service reliability, "
                "editorial tradeoffs, collaboration patterns, and system maintenance without naming "
                "specific platforms or deployment stacks. The introduction stays neutral and focuses "
                "on operational framing before later implementation notes mention concrete infrastructure details. "
            )
            + " ".join(["cloud run deployment infrastructure"] * 40)
        )
        items = [
            {
                "title": "Operating an agent service",
                "url": "https://example.com/cloud-saturation",
                "source": "Example Engineering",
                "content": repeated_cloud,
                "metadata": {"content_tier": "full_article", "content_length": len(repeated_cloud)},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        cloud_tag = ranking_scores[0]["weighted_secondary_tags"]["cloud"]
        devops_tag = ranking_scores[0]["weighted_secondary_tags"]["devops"]
        self.assertLessEqual(cloud_tag["body_weight_component"], 1.3)
        self.assertTrue(cloud_tag["body_saturation_applied"])
        self.assertNotIn("cloud", ranking_scores[0]["dominant_tags"])
        self.assertIn("cloud", ranking_scores[0]["supporting_tags"])
        self.assertLessEqual(devops_tag["body_weight_component"], 1.3)
        self.assertNotIn("devops", ranking_scores[0]["dominant_tags"])

    def test_fixture_multi_agent_tutorial_keeps_infra_and_mcp_in_calibrated_roles(self):
        fixture = load_ranking_fixture("multi_agent_tutorial")

        _, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.0,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertEqual(score["primary_article_type"], expected["primary_article_type"])
        for tag in expected["dominant_tags"]:
            self.assertIn(tag, score["dominant_tags"])
        for tag in expected["supporting_tags"]:
            self.assertIn(tag, score["supporting_tags"])
        for tag in expected["not_dominant_tags"]:
            self.assertNotIn(tag, score["dominant_tags"])
        self.assertGreater(score["weighted_secondary_tags"]["mcp"]["heading_weight_component"], 0.0)
        self.assertLess(score["weighted_secondary_tags"]["mcp"]["heading_weight_component"], 1.5)
        self.assertEqual(score["weighted_secondary_tags"]["mcp"]["strength"], 1.0)
        self.assertIn("heading", score["weighted_secondary_tags"]["security"]["dominant_signal_sources"])

    def test_fixture_mcp_gateway_article_keeps_mcp_dominant_and_cloud_supporting(self):
        fixture = load_ranking_fixture("mcp_gateway_article")

        _, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.0,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertEqual(score["primary_article_type"], expected["primary_article_type"])
        for tag in expected["dominant_tags"]:
            self.assertIn(tag, score["dominant_tags"])
        for tag in expected["supporting_tags"]:
            self.assertIn(tag, score["supporting_tags"])
        for tag in expected["not_dominant_tags"]:
            self.assertNotIn(tag, score["dominant_tags"])
        self.assertGreater(score["weighted_secondary_tags"]["mcp"]["heading_weight_component"], 0.0)
        self.assertFalse(score["weighted_secondary_tags"]["mcp"]["heading_boost_capped"])
        self.assertTrue(score["weighted_secondary_tags"]["cloud"]["heading_boost_capped"])
        self.assertIn("Google Cloud Docs (Knowledge Tool)", score["heading_diagnostics"]["detected_headings"])

    def test_fixture_security_article_stays_architecture_security(self):
        fixture = load_ranking_fixture("security_heavy_article")

        _, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.0,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertEqual(score["primary_article_type"], expected["primary_article_type"])
        for tag in expected["dominant_tags"]:
            self.assertIn(tag, score["dominant_tags"])
        self.assertIn("devops", score["supporting_tags"])
        for tag in expected["not_dominant_tags"]:
            self.assertNotIn(tag, score["dominant_tags"])

    def test_fixture_system_design_article_is_not_unknown(self):
        fixture = load_ranking_fixture("system_design_agents")

        _, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.0,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertEqual(score["primary_article_type"], expected["primary_article_type"])
        self.assertIn("auth", score["supporting_tags"])
        self.assertIn("memory", score["supporting_tags"])
        for tag in expected["not_dominant_tags"]:
            self.assertNotIn(tag, score["dominant_tags"])

    def test_fixture_community_update_is_penalized_and_event_dominant(self):
        fixture = load_ranking_fixture("community_update")

        selected, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.9,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertEqual(score["primary_article_type"], expected["primary_article_type"])
        for tag in expected["dominant_tags"]:
            self.assertIn(tag, score["dominant_tags"])
        self.assertEqual(selected, [])
        self.assertIn("community update", score["rejection_reasons"])

    def test_fixture_broad_ai_article_stays_weak_for_ai_agents(self):
        fixture = load_ranking_fixture("broad_ai_article")

        selected, ranking_scores = rank_source_items(
            [fixture["item"]],
            keywords=fixture["keywords"],
            top_n=1,
            min_quality_score=0.8,
        )

        score = ranking_scores[0]
        expected = fixture["expected"]
        self.assertLessEqual(score["topic_relevance_score"], expected["max_topic_relevance_score"])
        self.assertLessEqual(score["topic_specificity_score"], expected["max_topic_specificity_score"])
        self.assertEqual(selected, [])
        self.assertIn("low novelty", score["rejection_reasons"])

    def test_capability_building_article_does_not_become_architecture_security_from_credentials_mentions(self):
        items = [
            {
                "title": "Building Capabilities for a Multi-Agent System with Google ADK, MCP, and Cloud Run",
                "url": "https://example.com/capabilities",
                "source": "Example Engineering",
                "snippet": (
                    "This guide shows how to build capabilities for a multi-agent system with Google ADK, MCP, "
                    "Cloud Run, and deployment workflows, with a brief note about credentials and permissions."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 910},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertIn(ranking_scores[0]["primary_article_type"], {"tutorial", "deep_technical"})
        self.assertNotEqual(ranking_scores[0]["primary_article_type"], "architecture_security")
        self.assertIn("multi_agent", ranking_scores[0]["dominant_tags"])
        self.assertIn("mcp", ranking_scores[0]["dominant_tags"])
        self.assertIn("adk", ranking_scores[0]["dominant_tags"])
        self.assertIn("cloud", ranking_scores[0]["dominant_tags"])
        self.assertIn("google_cloud", ranking_scores[0]["dominant_tags"])
        self.assertNotIn("product_update", ranking_scores[0]["dominant_tags"])

    def test_personalized_multi_agent_memory_article_makes_memory_dominant(self):
        items = [
            {
                "title": "Architect a Personalized Multi-Agent System with Long-Term Memory",
                "url": "https://example.com/personalized-memory",
                "source": "Example Engineering",
                "snippet": (
                    "This guide explains how to architect a personalized multi-agent system with long-term memory, "
                    "Cloud Run, MCP integrations, and Google ADK building blocks."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 980},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertIn("multi_agent", ranking_scores[0]["dominant_tags"])
        self.assertIn("memory", ranking_scores[0]["dominant_tags"])
        self.assertIn("cloud", ranking_scores[0]["supporting_tags"])
        self.assertIn("google_cloud", ranking_scores[0]["supporting_tags"])
        self.assertIn("mcp", ranking_scores[0]["supporting_tags"])
        self.assertIn("adk", ranking_scores[0]["supporting_tags"])

    def test_conceptual_agent_design_article_is_not_unknown(self):
        items = [
            {
                "title": "Designing a team of agents",
                "url": "https://example.com/system-design",
                "source": "Example Research",
                "snippet": (
                    "This article explores how agents coordinate, how planner and executor roles differ, "
                    "and which architecture patterns help a team of agents stay aligned."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 780},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "system_design")
        self.assertEqual(ranking_scores[0]["article_type"], "system_design")
        self.assertNotEqual(ranking_scores[0]["primary_article_type"], "unknown")
        self.assertGreaterEqual(ranking_scores[0]["article_type_score_modifier"], 0.0)

    def test_generic_ai_recipe_helper_does_not_get_high_specificity_for_ai_agents(self):
        items = [
            {
                "title": "I Built My Mom an AI Recipe Helper for Mother's Day",
                "url": "https://example.com/recipe-helper",
                "source": "Example Personal Blog",
                "snippet": (
                    "A personal project using OpenAI and a small app interface to suggest recipes, "
                    "shopping ideas, and meal planning help."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 720},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["AI agents"], top_n=1, min_quality_score=0.0)

        self.assertGreater(ranking_scores[0]["topic_relevance_score"], 0.0)
        self.assertLess(ranking_scores[0]["topic_relevance_score"], 3.0)
        self.assertEqual(ranking_scores[0]["topic_relevance_score"], 1.0)
        self.assertEqual(ranking_scores[0]["topic_relevance_reason"], "matched only broad AI-adjacent signals")
        self.assertLess(ranking_scores[0]["topic_specificity_score"], 1.0)
        self.assertIn("helper", ranking_scores[0]["generic_topic_signals"])
        self.assertEqual(ranking_scores[0]["specificity_signals"], [])

    def test_generic_llm_cloud_article_does_not_get_high_specificity_for_ai_agents(self):
        items = [
            {
                "title": "Scaling LLM apps on Cloud Run",
                "url": "https://example.com/llm-cloud",
                "source": "Example Engineering",
                "snippet": (
                    "This article covers Cloud Run deployment, app scaling, and LLM request handling, "
                    "but it does not discuss agents, orchestration, memory, or tool use."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 840},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["AI agents"], top_n=1, min_quality_score=0.0)

        self.assertLess(ranking_scores[0]["topic_relevance_score"], 3.0)
        self.assertEqual(ranking_scores[0]["topic_relevance_score"], 1.0)
        self.assertLess(ranking_scores[0]["topic_specificity_score"], 1.0)
        self.assertIn("llm", ranking_scores[0]["generic_topic_signals"])
        self.assertIn("cloud", ranking_scores[0]["generic_topic_signals"])

    def test_claude_docker_model_runner_article_does_not_get_high_relevance_for_ai_agents(self):
        items = [
            {
                "title": "Using Claude Code with Docker Model Runner",
                "url": "https://example.com/claude-docker",
                "source": "Example Engineering",
                "snippet": (
                    "A practical guide to Claude Code, Docker Model Runner, and local LLM workflows "
                    "for faster development and testing."
                ),
                "metadata": {"content_tier": "full_article", "content_length": 760},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["AI agents"], top_n=1, min_quality_score=0.0)

        self.assertLess(ranking_scores[0]["topic_relevance_score"], 3.0)
        self.assertIn(ranking_scores[0]["topic_relevance_score"], {0.0, 1.0})
        self.assertEqual(ranking_scores[0]["topic_relevance_reason"], "matched only broad AI-adjacent signals")
        self.assertIn("claude", ranking_scores[0]["weak_relevance_signals"])
        self.assertIn("docker", ranking_scores[0]["weak_relevance_signals"])

    def test_generic_unrelated_tutorial_does_not_gain_high_relevance_from_format_alone(self):
        items = [
            {
                "title": "How to organize your design files",
                "url": "https://example.com/design-files",
                "source": "Example Blog",
                "snippet": "A step-by-step guide to organizing folders, templates, and review notes.",
                "metadata": {"content_tier": "full_article", "content_length": 640},
            }
        ]

        _, ranking_scores = rank_source_items(items, keywords=["ai agents"], top_n=1, min_quality_score=0.0)

        self.assertEqual(ranking_scores[0]["primary_article_type"], "tutorial")
        self.assertEqual(ranking_scores[0]["article_type"], "tutorial")
        self.assertEqual(ranking_scores[0]["topic_relevance_score"], 0.0)
        self.assertIn("low relevance", ranking_scores[0]["rejection_reasons"])

    def test_sparse_article_inputs_produce_diagnostics_warning_instead_of_silent_zero(self):
        items = [
            {
                "title": "AI Agents quick note",
                "url": "https://example.com/ai-agents-note",
                "source": "Example Blog",
                "snippet": "",
                "content": "",
                "metadata": {
                    "content_tier": "weak_snippet",
                    "content_length": 0,
                },
            }
        ]

        _, ranking_scores = rank_source_items(
            items,
            keywords=["ai agents"],
            top_n=1,
            min_quality_score=0.0,
        )

        self.assertGreater(ranking_scores[0]["topic_relevance_score"], 0.0)
        self.assertTrue(ranking_scores[0]["diagnostic_warnings"])
        self.assertIn("article text is sparse", ranking_scores[0]["diagnostic_warnings"][0])

    def test_items_below_threshold_are_not_selected(self):
        items = [
            {
                "title": "Weak one",
                "url": "https://example.com/weak-1",
                "source": "Example Blog",
                "snippet": "Short note.",
            },
            {
                "title": "Weak two",
                "url": "https://example.com/weak-2",
                "source": "Example Blog",
                "snippet": "Another short note.",
            },
        ]

        selected, _ = rank_source_items(items, top_n=3, min_quality_score=0.4)

        self.assertEqual(selected, [])
