import json
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings

from apps.digests.models import SourceDiscoveryRun
from apps.topics.models import Topic
from services.sources.content_research_planner import (
    MAX_FINAL_QUERY_COUNT,
    PROMPT_TEMPLATE_PATH,
    build_content_research_planner_prompt,
    create_content_research_plan,
)
from services.sources.query_history_summary import build_query_history_summary
from services.sources.source_quality_feedback import (
    build_source_quality_feedback,
    classify_source_quality_pattern,
)


class _TopicStub:
    def __init__(self, title: str, keywords, description: str = "") -> None:
        self.name = title
        self.keywords = keywords
        self.description = description


class ContentResearchPlannerTests(SimpleTestCase):
    def test_prompt_template_is_loaded_and_rendered(self) -> None:
        prompt = build_content_research_planner_prompt(
            "AI Education Teens",
            ["AI literacy", "classroom practice"],
        )

        self.assertTrue(PROMPT_TEMPLATE_PATH.exists())
        self.assertIn("Topic title: AI Education Teens", prompt)
        self.assertIn("Topic keywords: AI literacy, classroom practice", prompt)
        self.assertIn(f"Generate no more than {MAX_FINAL_QUERY_COUNT} final search queries.", prompt)

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_ai_planner_returns_valid_json_and_extracts_queries(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "AI education for teenagers in current practice.",
                    "content_research_goal": "Find fresh, practical materials for a digest and post.",
                    "source_selection_criteria": {
                        "must_be_relevant_to": ["AI use in teen education"],
                        "preferred_material_types": ["case study", "expert opinion"],
                        "freshness_signals": ["recent examples"],
                        "post_value_signals": ["trade-offs"],
                        "relevance_boundary": "Stay focused on teen learning and teaching practice.",
                    },
                    "content_tension_opportunities": [
                        {"tension": "AI help vs overreliance", "why_it_matters": "The trade-off makes the post stronger."}
                    ],
                    "search_angles": [
                        {"angle": "school implementation", "purpose": "Find real-world practice examples."}
                    ],
                    "queries": [
                        "AI education teens recent classroom examples",
                        "AI education teens expert opinion trade-offs",
                    ],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy", "classroom practice"]))

        self.assertEqual(result.planner_status, "ai_planned")
        self.assertFalse(result.fallback_used)
        self.assertEqual(
            result.final_queries,
            (
                "AI education teens recent classroom examples",
                "AI education teens expert opinion trade-offs",
            ),
        )
        self.assertEqual(result.topic_interpretation, "AI education for teenagers in current practice.")

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_invalid_json_triggers_fallback(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(text="{not valid json")

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(result.planner_status, "fallback_used")
        self.assertTrue(result.fallback_used)
        self.assertTrue(result.error_message)
        self.assertTrue(result.final_queries)

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_empty_queries_triggers_fallback(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Interpretation",
                    "content_research_goal": "Goal",
                    "source_selection_criteria": {
                        "must_be_relevant_to": [],
                        "preferred_material_types": [],
                        "freshness_signals": [],
                        "post_value_signals": [],
                        "relevance_boundary": "",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(result.planner_status, "fallback_used")
        self.assertTrue(result.final_queries)

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_duplicate_queries_are_removed(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Interpretation",
                    "content_research_goal": "Goal",
                    "source_selection_criteria": {
                        "must_be_relevant_to": [],
                        "preferred_material_types": [],
                        "freshness_signals": [],
                        "post_value_signals": [],
                        "relevance_boundary": "",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [
                        "AI education teens case study outcomes",
                        "AI education teens case study outcomes",
                        "AI education teens expert opinion classroom practice",
                    ],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(
            result.final_queries,
            (
                "AI education teens case study outcomes",
                "AI education teens expert opinion classroom practice",
            ),
        )

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_query_count_is_limited(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Interpretation",
                    "content_research_goal": "Goal",
                    "source_selection_criteria": {
                        "must_be_relevant_to": [],
                        "preferred_material_types": [],
                        "freshness_signals": [],
                        "post_value_signals": [],
                        "relevance_boundary": "",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [f"AI education teens query number {index} practical examples" for index in range(12)],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(len(result.final_queries), MAX_FINAL_QUERY_COUNT)

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_very_short_or_generic_queries_are_removed(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Interpretation",
                    "content_research_goal": "Goal",
                    "source_selection_criteria": {
                        "must_be_relevant_to": [],
                        "preferred_material_types": [],
                        "freshness_signals": [],
                        "post_value_signals": [],
                        "relevance_boundary": "",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [
                        "AI",
                        "AI Education Teens",
                        "AI education teens practical case study",
                    ],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(result.final_queries, ("AI education teens practical case study",))

    def test_planner_uses_title_and_keywords_not_description(self) -> None:
        topic = _TopicStub(
            "AI Education Teens",
            ["AI literacy", "classroom practice"],
            description="This should not appear in the prompt.",
        )

        prompt = build_content_research_planner_prompt(topic.name, topic.keywords)

        self.assertIn("AI Education Teens", prompt)
        self.assertIn("AI literacy, classroom practice", prompt)
        self.assertNotIn("This should not appear in the prompt.", prompt)

    @override_settings(OPENAI_API_KEY="")
    def test_fallback_queries_are_generated_from_title_and_keywords(self) -> None:
        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        self.assertEqual(result.planner_status, "fallback_used")
        self.assertIn("AI Education Teens AI literacy recent examples", result.final_queries)
        self.assertIn("AI Education Teens AI literacy expert opinion", result.final_queries)

    @override_settings(OPENAI_API_KEY="")
    def test_planner_result_includes_diagnostics(self) -> None:
        result = create_content_research_plan(_TopicStub("AI Education Teens", ["AI literacy"]))

        diagnostics = result.diagnostics
        self.assertIn("planner_status", diagnostics)
        self.assertIn("fallback_used", diagnostics)
        self.assertIn("final_queries", diagnostics)
        self.assertIn("source_selection_criteria", diagnostics)

    def test_prompt_requests_fresh_practical_post_worthy_materials_and_tension_opportunities(self) -> None:
        prompt = build_content_research_planner_prompt(
            "AI Education Teens",
            ["AI literacy", "classroom practice"],
        )

        self.assertIn("fresh, practical, discussion-worthy materials", prompt)
        self.assertIn("digest and a LinkedIn-style post", prompt)
        self.assertIn("conflicting opinions, opposite practices, trade-offs, and different outcomes", prompt)
        self.assertIn("Do not include stale explicit years older than", prompt)
        self.assertIn("Prefer temporal wording such as latest, current, recent, or this month.", prompt)
        self.assertIn('"content_tension_opportunities"', prompt)
        self.assertIn("Return valid JSON only.", prompt)

    def test_prompt_can_render_compact_query_history_summary(self) -> None:
        prompt = build_content_research_planner_prompt(
            "AI Education Teens",
            ["AI literacy", "classroom practice"],
            query_history_summary={
                "history_available": True,
                "recent_run_count": 2,
                "total_query_rows": 3,
                "useful_queries": [
                    {
                        "query": "AI education teens classroom examples",
                        "status": "useful",
                        "returned_count": 4,
                        "accepted_count": 2,
                        "visible_new_suggestions_count": 1,
                        "duplicate_count": 0,
                        "angle": "recent examples",
                    }
                ],
                "weak_queries": [],
                "duplicate_heavy_queries": [],
                "provider_error_queries": [],
                "quality_rejected_queries": [],
                "useful_angles": [{"angle": "recent examples", "count": 1}],
                "weak_angles": [],
                "provider_error_angles": [],
                "stale_year_patterns": [],
                "weak_material_types": [{"material_type": "social/profile/forum", "count": 2}],
                "preferred_material_types_found": [{"material_type": "institutional / analyst report", "count": 1}],
                "weak_domains": [{"domain": "quora.com", "count": 2}],
                "dominant_rejection_reasons": [{"reason": "not enough substantive signals", "count": 2}],
                "quality_guidance": ["Broad beginner or SEO-style guide phrasing is producing weak pages."],
                "planning_guidance": ["Useful directions so far: recent examples. Create fresh variants around those angles instead of reusing the same wording."],
            },
        )

        self.assertIn("Recent query history summary:", prompt)
        self.assertIn("AI education teens classroom examples", prompt)
        self.assertIn("Weak material types", prompt)
        self.assertIn("quora.com", prompt)
        self.assertIn("Broad beginner or SEO-style guide phrasing is producing weak pages.", prompt)
        self.assertIn("Useful directions so far: recent examples. Create fresh variants around those angles instead of reusing the same wording.", prompt)
        self.assertNotIn('"query_performance"', prompt)

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_stale_explicit_years_are_removed_from_fresh_ai_queries(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Bitcoin market research in current practice.",
                    "content_research_goal": "Find current, post-worthy market material.",
                    "source_selection_criteria": {
                        "must_be_relevant_to": ["Bitcoin market"],
                        "preferred_material_types": ["expert opinion", "analysis"],
                        "freshness_signals": ["recent"],
                        "post_value_signals": ["trade-offs"],
                        "relevance_boundary": "Stay close to the topic.",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [
                        "current opinions on Bitcoin price predictions 2023",
                        "institutional investment trends in Bitcoin 2023",
                    ],
                }
            )
        )

        result = create_content_research_plan(_TopicStub("Bitcoin market", ["market structure"]))

        self.assertTrue(result.final_queries)
        self.assertTrue(all("2023" not in query for query in result.final_queries))
        self.assertTrue(any("current" in query.casefold() or "latest" in query.casefold() for query in result.final_queries))

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_quality_guidance_pushes_material_oriented_terms_into_ai_queries(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Bitcoin market research in current practice.",
                    "content_research_goal": "Find current, post-worthy market material.",
                    "source_selection_criteria": {
                        "must_be_relevant_to": ["Bitcoin market"],
                        "preferred_material_types": ["analysis"],
                        "freshness_signals": ["recent"],
                        "post_value_signals": ["trade-offs"],
                        "relevance_boundary": "Stay close to the topic.",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [
                        {
                            "angle": "Recent shifts in institutional investment strategies in Bitcoin",
                            "purpose": "To explore how these strategies are shaping market trends and impacting retail investors.",
                        },
                        {
                            "angle": "Bitcoin volatility and retail trading outcomes",
                            "purpose": "To explore how volatility is affecting new investors.",
                        },
                    ],
                    "queries": [
                        "current trends in retail investor strategies for Bitcoin",
                        "recent adaptations in Bitcoin trading approaches for new investors",
                        "market responses to Bitcoin volatility and investment sentiment this month",
                    ],
                }
            )
        )

        topic = _TopicStub("Bitcoin market", ["market structure"])
        with patch(
            "services.sources.content_research_planner.build_query_history_summary",
            return_value={
                "history_available": True,
                "recent_run_count": 2,
                "total_query_rows": 8,
                "quality_guidance": [
                    "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest."
                ],
                "preferred_material_types_found": [{"material_type": "market data / flow analysis", "count": 2}],
                "planning_guidance": [],
            },
        ):
            result = create_content_research_plan(topic)

        joined = " || ".join(result.final_queries).casefold()
        material_terms = ("etf flows", "institutional flows", "funding rates", "open interest")
        self.assertGreaterEqual(sum(1 for query in result.final_queries if any(term in query.casefold() for term in material_terms)), 3)
        self.assertTrue(any(term in joined for term in material_terms))
        generic_retail_needles = ("retail investor", "new investors", "trading strategies", "trading approaches")
        self.assertLessEqual(
            sum(1 for query in result.final_queries if any(needle in query.casefold() for needle in generic_retail_needles)),
            1,
        )
        query_to_angle = {
            query.casefold(): angle
            for query, angle in zip(result.final_queries, result.search_angles, strict=False)
        }
        self.assertIn("ETF flows / fund flows", query_to_angle["bitcoin market analysis etf flows latest"]["angle"])
        self.assertIn("institutional flows", query_to_angle["bitcoin market analysis institutional flows latest"]["angle"])
        self.assertIn("derivatives / market structure", query_to_angle["bitcoin market analysis funding rates latest"]["angle"])
        self.assertNotIn("retail investors", query_to_angle["bitcoin market analysis etf flows latest"]["purpose"].casefold())

    @override_settings(OPENAI_API_KEY="sk-test", OPENAI_MODEL="gpt-test")
    @patch("services.sources.content_research_planner.OpenAIClient.generate_text")
    def test_noisy_topic_prefix_is_removed_when_canonical_bitcoin_term_is_present(self, mock_generate_text) -> None:
        mock_generate_text.return_value = SimpleNamespace(
            text=json.dumps(
                {
                    "topic_interpretation": "Bitcoin market research in current practice.",
                    "content_research_goal": "Find current, post-worthy market material.",
                    "source_selection_criteria": {
                        "must_be_relevant_to": ["Bitcoin market"],
                        "preferred_material_types": ["analysis"],
                        "freshness_signals": ["recent"],
                        "post_value_signals": ["trade-offs"],
                        "relevance_boundary": "Stay close to the topic.",
                    },
                    "content_tension_opportunities": [],
                    "search_angles": [],
                    "queries": [
                        "bitcion market Bitcoin market analysis ETF flows latest",
                        "bitcion market Bitcoin market analysis institutional flows latest",
                    ],
                }
            )
        )

        topic = _TopicStub("bitcion market", ["market analysis"])
        result = create_content_research_plan(topic)

        self.assertTrue(result.final_queries)
        self.assertTrue(all("bitcion" not in query.casefold() for query in result.final_queries))
        self.assertTrue(all("bitcion market bitcoin market analysis" not in query.casefold() for query in result.final_queries))
        self.assertTrue(any(query.startswith("Bitcoin market analysis") for query in result.final_queries))


class ContentResearchPlannerHistorySummaryTests(TestCase):
    def _create_topic(self, name: str = "AI Education Teens") -> Topic:
        normalized_name = name.casefold().replace(" ", "-")
        user = get_user_model().objects.create_user(username=f"user-{normalized_name}", password="pw")
        return Topic.objects.create(user=user, name=name, keywords=["AI literacy", "classroom practice"])

    def test_no_history_topics_return_empty_history_summary(self) -> None:
        topic = self._create_topic("No history topic")

        summary = build_query_history_summary(topic)

        self.assertFalse(summary["history_available"])
        self.assertEqual(summary["recent_run_count"], 0)
        self.assertEqual(summary["useful_queries"], [])
        self.assertEqual(summary["planning_guidance"], [])

    def test_history_summary_separates_useful_weak_and_duplicate_heavy_queries(self) -> None:
        topic = self._create_topic("History summary topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "alpha useful query",
                        "angle": "recent examples",
                        "purpose": "Find concrete examples.",
                        "returned_count": 5,
                        "accepted_count": 2,
                        "rejected_count": 1,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 1,
                        "status": "useful",
                    },
                    {
                        "query": "beta weak query",
                        "angle": "expert opinion",
                        "purpose": "Find expert opinions.",
                        "returned_count": 3,
                        "accepted_count": 0,
                        "rejected_count": 2,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 0,
                        "status": "weak",
                    },
                    {
                        "query": "gamma duplicate query",
                        "angle": "case study",
                        "purpose": "Find case studies.",
                        "returned_count": 4,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 3,
                        "visible_new_suggestions_count": 0,
                        "status": "duplicate_heavy",
                    },
                ]
            },
        )

        summary = build_query_history_summary(topic)

        self.assertTrue(summary["history_available"])
        self.assertEqual(summary["recent_run_count"], 1)
        self.assertEqual(summary["useful_queries"][0]["query"], "alpha useful query")
        self.assertEqual(summary["weak_queries"][0]["query"], "beta weak query")
        self.assertEqual(summary["duplicate_heavy_queries"][0]["query"], "gamma duplicate query")
        self.assertEqual(summary["useful_angles"][0]["angle"], "recent examples")
        self.assertTrue(summary["planning_guidance"])

    def test_source_quality_patterns_are_classified_deterministically(self) -> None:
        social = classify_source_quality_pattern(
            url="https://www.facebook.com/some-post",
            title="Bitcoin market chat",
            snippet="Community post",
        )
        self.assertEqual(social["weak_material_type"], "social_profile_forum")

        prediction = classify_source_quality_pattern(
            url="https://example.com/bitcoin-price-prediction",
            title="Bitcoin price prediction for new investors",
            snippet="Will BTC hit a new high?",
        )
        self.assertEqual(prediction["weak_material_type"], "price_prediction_live_price")

        beginner = classify_source_quality_pattern(
            url="https://example.com/ultimate-guide",
            title="Ultimate guide to crypto trading for beginners",
            snippet="A broad walkthrough.",
        )
        self.assertEqual(beginner["weak_material_type"], "beginner_seo_guide")

        preferred = classify_source_quality_pattern(
            url="https://glassnode.com/reports/bitcoin-market-structure",
            title="Bitcoin market structure and ETF flows weekly commentary",
            snippet="On-chain and liquidity signals.",
        )
        self.assertEqual(preferred["preferred_material_type"], "on_chain_analysis")

    def test_build_source_quality_feedback_summarizes_weak_and_preferred_patterns(self) -> None:
        evaluated_candidates = (
            SimpleNamespace(
                url="https://facebook.com/some-thread",
                title="Bitcoin discussion",
                snippet="Community thread",
                candidate_type="article",
                normalized_url="https://facebook.com/some-thread",
                status=SimpleNamespace(value="rejected"),
                diagnostics={"quality_rejection_reason": "not enough substantive signals"},
                rejection_reasons=("not enough substantive signals",),
            ),
            SimpleNamespace(
                url="https://example.com/bitcoin-price-prediction",
                title="Bitcoin price prediction 2026",
                snippet="Will BTC hit a new high?",
                candidate_type="article",
                normalized_url="https://example.com/bitcoin-price-prediction",
                status=SimpleNamespace(value="rejected"),
                diagnostics={"quality_rejection_reason": "not enough substantive signals"},
                rejection_reasons=("not enough substantive signals",),
            ),
            SimpleNamespace(
                url="https://glassnode.com/reports/bitcoin-market-structure",
                title="Bitcoin market structure weekly commentary",
                snippet="ETF flows and liquidity signals",
                candidate_type="article",
                normalized_url="https://glassnode.com/reports/bitcoin-market-structure",
                status=SimpleNamespace(value="accepted"),
                diagnostics={},
                rejection_reasons=(),
            ),
        )
        result = SimpleNamespace(evaluated_candidates=evaluated_candidates)

        feedback = build_source_quality_feedback(
            source_research_result=result,
            shown_candidates=[{"url": "https://glassnode.com/reports/bitcoin-market-structure"}],
            known_normalized_urls=set(),
        )

        self.assertEqual(feedback["quality_rejected_count"], 2)
        self.assertEqual(feedback["shown_count"], 1)
        self.assertEqual(feedback["weak_domains"][0]["domain"], "facebook.com")
        self.assertTrue(any(item["material_type"] == "price_prediction_live_price" for item in feedback["weak_material_types"]))
        self.assertTrue(any(item["material_type"] == "on_chain_analysis" for item in feedback["preferred_material_types_found"]))
        self.assertTrue(feedback["planner_quality_guidance"])

    def test_provider_errors_are_not_classified_as_weak_query_directions(self) -> None:
        topic = self._create_topic("Provider error summary topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_PARTIAL_FAILED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "provider failed query",
                        "angle": "expert analysis",
                        "purpose": "Find current analysis.",
                        "returned_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 0,
                        "status": "partial_error",
                        "error_message": "SerpAPI returned an API error.",
                    },
                    {
                        "query": "real weak query",
                        "angle": "retail behavior",
                        "purpose": "Find retail behavior.",
                        "returned_count": 3,
                        "accepted_count": 0,
                        "rejected_count": 2,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 0,
                        "status": "weak",
                    },
                ]
            },
        )

        summary = build_query_history_summary(topic)

        self.assertEqual([item["query"] for item in summary["provider_error_queries"]], ["provider failed query"])
        self.assertEqual([item["query"] for item in summary["weak_queries"]], ["real weak query"])
        self.assertEqual(summary["provider_error_angles"][0]["angle"], "expert analysis")
        self.assertTrue(any("Do not treat those rows as proof that the angle is weak" in item for item in summary["planning_guidance"]))

    def test_history_summary_includes_quality_feedback_guidance_from_previous_runs(self) -> None:
        topic = self._create_topic("Quality guidance topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "bitcoin market structure research report",
                        "angle": "market structure",
                        "purpose": "Find higher-substance analysis.",
                        "returned_count": 4,
                        "accepted_count": 1,
                        "rejected_count": 1,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 1,
                        "status": "useful",
                    }
                ],
                "source_quality_feedback": {
                    "quality_rejected_count": 3,
                    "known_or_duplicate_count": 1,
                    "shown_count": 1,
                    "dominant_rejection_reasons": [{"reason": "not enough substantive signals", "count": 3}],
                    "weak_domains": [{"domain": "quora.com", "count": 2, "reason": "social/profile/forum"}],
                    "weak_material_types": [{"material_type": "beginner_seo_guide", "label": "beginner / SEO guide", "count": 2}],
                    "preferred_material_types_found": [{"material_type": "market_data_flow_analysis", "label": "market data / flow analysis", "count": 1}],
                    "main_quality_issue": "beginner / SEO guide results dominate recent rejected candidates",
                    "planner_quality_guidance": [
                        "Broad beginner or SEO-style guide phrasing is producing weak pages. Avoid 'for beginners', 'ultimate guide', or generic strategy phrasing.",
                        "Prefer material types like market data / flow analysis. Use query terms such as ETF flows, institutional flows, funding rates, open interest.",
                    ],
                },
            },
        )

        summary = build_query_history_summary(topic)

        self.assertTrue(any(item["domain"] == "quora.com" for item in summary["weak_domains"]))
        self.assertTrue(any(item["reason"] == "not enough substantive signals" for item in summary["dominant_rejection_reasons"]))
        self.assertTrue(any(item["material_type"] == "beginner / SEO guide" for item in summary["weak_material_types"]))
        self.assertTrue(any("ETF flows" in item for item in summary["planning_guidance"]))
        self.assertTrue(any("quora.com" in item for item in summary["planning_guidance"]))

    def test_recent_duplicate_heavy_surface_is_classified_as_exhausted(self) -> None:
        topic = self._create_topic("Exhausted surface topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "Bitcoin ETF flows latest",
                        "angle": "ETF flows",
                        "purpose": "Track ETF flow data.",
                        "returned_count": 6,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 5,
                        "visible_new_suggestions_count": 0,
                        "status": "duplicate_heavy",
                        "surface_key": "etf_flows_report",
                    },
                    {
                        "query": "Bitcoin ETF flows weekly report",
                        "angle": "ETF flows",
                        "purpose": "Track ETF flow data.",
                        "returned_count": 5,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 4,
                        "visible_new_suggestions_count": 0,
                        "status": "duplicate_heavy",
                        "surface_key": "etf_flows_report",
                    },
                ]
            },
        )

        summary = build_query_history_summary(topic)

        self.assertEqual(summary["search_surface_memory"]["avoided_surfaces"], ["etf_flows_report"])
        surface_row = summary["search_surface_memory"]["surfaces"][0]
        self.assertEqual(surface_row["surface_key"], "etf_flows_report")
        self.assertEqual(surface_row["status"], "exhausted")

    def test_provider_error_only_surface_is_not_marked_exhausted(self) -> None:
        topic = self._create_topic("Provider uncertain surface topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_PARTIAL_FAILED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "Bitcoin analyst report latest",
                        "angle": "analyst report",
                        "purpose": "Find analyst viewpoints.",
                        "returned_count": 0,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 0,
                        "status": "partial_error",
                        "error_message": "SerpAPI error",
                        "surface_key": "analyst_report",
                    }
                ]
            },
        )

        summary = build_query_history_summary(topic)

        self.assertEqual(summary["search_surface_memory"]["avoided_surfaces"], [])
        surface_row = summary["search_surface_memory"]["surfaces"][0]
        self.assertEqual(surface_row["surface_key"], "analyst_report")
        self.assertEqual(surface_row["status"], "unknown")

    @override_settings(OPENAI_API_KEY="")
    @patch("services.sources.content_research_planner.build_query_history_summary")
    def test_first_round_query_planning_avoids_exhausted_surfaces_and_prefers_underexplored(self, mock_history_summary) -> None:
        mock_history_summary.return_value = {
            "history_available": True,
            "recent_run_count": 3,
            "malformed_run_count": 0,
            "total_query_rows": 9,
            "useful_queries": [],
            "weak_queries": [],
            "duplicate_heavy_queries": [],
            "provider_error_queries": [],
            "quality_rejected_queries": [],
            "useful_angles": [],
            "weak_angles": [],
            "provider_error_angles": [],
            "stale_year_patterns": [],
            "weak_material_types": [],
            "preferred_material_types_found": [{"material_type": "market data / flow analysis", "count": 2}],
            "weak_domains": [],
            "dominant_rejection_reasons": [],
            "quality_guidance": [
                "Use query terms such as ETF flows, institutional flows, funding rates, market structure, analyst report.",
            ],
            "recent_query_texts": [
                "Bitcoin market analysis ETF flows latest",
                "Bitcoin market analysis institutional flows latest",
            ],
            "search_surface_memory": {
                "recent_run_count": 3,
                "surfaces": [
                    {
                        "surface_key": "etf_flows_report",
                        "status": "exhausted",
                        "visible_count": 0,
                        "known_duplicate_count": 6,
                        "quality_rejected_count": 0,
                        "returned_count": 6,
                        "last_seen": "2026-06-02T00:00:00+00:00",
                        "reason": "Recent clicks mostly hit already-known or duplicate URLs.",
                    },
                    {
                        "surface_key": "market_structure_report",
                        "status": "useful",
                        "visible_count": 2,
                        "known_duplicate_count": 0,
                        "quality_rejected_count": 0,
                        "returned_count": 3,
                        "last_seen": "2026-06-02T00:00:00+00:00",
                        "reason": "Recent clicks still surfaced visible suggestions.",
                    },
                    {
                        "surface_key": "on_chain_exchange_reserves_analysis",
                        "status": "underexplored",
                        "visible_count": 0,
                        "known_duplicate_count": 0,
                        "quality_rejected_count": 0,
                        "returned_count": 0,
                        "last_seen": None,
                        "reason": "Preferred adjacent surface has little or no recent coverage.",
                    },
                ],
                "avoided_surfaces": ["etf_flows_report", "institutional_flows_report"],
                "preferred_surfaces": ["market_structure_report", "analyst_report"],
                "underexplored_surfaces": ["on_chain_exchange_reserves_analysis"],
            },
            "planning_guidance": [
                "Avoid starting with exhausted surfaces from recent clicks: ETF flows weekly report, institutional fund flows report.",
                "Try underexplored adjacent surfaces next: on-chain exchange reserves analysis.",
            ],
        }

        topic = self._create_topic("bitcion market")
        result = create_content_research_plan(topic)

        self.assertGreaterEqual(len(result.final_queries), 3)
        self.assertTrue(any("market structure report" in query.casefold() for query in result.final_queries))
        self.assertTrue(any("on-chain exchange reserves analysis" in query.casefold() for query in result.final_queries))
        self.assertFalse(any("etf flows weekly report" in query.casefold() for query in result.final_queries))
        self.assertFalse(any(query.casefold() == "bitcoin market analysis etf flows latest" for query in result.final_queries))

    def test_stale_year_and_duplicate_heavy_patterns_influence_guidance(self) -> None:
        topic = self._create_topic("Pattern guidance topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "institutional investment trends in Bitcoin 2023",
                        "angle": "institutional flows",
                        "purpose": "Track institutional positioning.",
                        "returned_count": 5,
                        "accepted_count": 0,
                        "rejected_count": 0,
                        "duplicate_count": 3,
                        "visible_new_suggestions_count": 0,
                        "status": "duplicate_heavy",
                    },
                    {
                        "query": "macro liquidity outlook for Bitcoin",
                        "angle": "macro liquidity",
                        "purpose": "Connect Bitcoin to macro liquidity.",
                        "returned_count": 4,
                        "accepted_count": 1,
                        "rejected_count": 0,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 1,
                        "status": "useful",
                    },
                ]
            },
        )

        summary = build_query_history_summary(topic)

        self.assertEqual(summary["stale_year_patterns"][0]["pattern"], "2023")
        self.assertEqual(summary["duplicate_heavy_queries"][0]["query"], "institutional investment trends in Bitcoin 2023")
        self.assertTrue(any("Avoid stale explicit years" in item for item in summary["planning_guidance"]))
        self.assertTrue(any("Duplicate-heavy directions look exhausted" in item for item in summary["planning_guidance"]))
        self.assertTrue(any("Useful directions so far: macro liquidity" in item for item in summary["planning_guidance"]))

    def test_malformed_history_diagnostics_do_not_crash_summary(self) -> None:
        topic = self._create_topic("Malformed history topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={"query_performance": ["bad-row", 7, None]},
        )

        summary = build_query_history_summary(topic)

        self.assertFalse(summary["history_available"])
        self.assertEqual(summary["total_query_rows"], 0)

    @override_settings(OPENAI_API_KEY="")
    def test_planner_result_includes_compact_history_summary_in_prompt_and_diagnostics(self) -> None:
        topic = self._create_topic("Planner history topic")
        SourceDiscoveryRun.objects.create(
            user=topic.user,
            topic=topic,
            provider_name="serpapi",
            status=SourceDiscoveryRun.STATUS_COMPLETED,
            diagnostics={
                "query_performance": [
                    {
                        "query": "history aware alpha",
                        "angle": "recent examples",
                        "purpose": "Find current examples.",
                        "returned_count": 4,
                        "accepted_count": 1,
                        "rejected_count": 1,
                        "duplicate_count": 0,
                        "visible_new_suggestions_count": 1,
                        "status": "useful",
                    }
                ]
            },
        )

        result = create_content_research_plan(topic)

        self.assertEqual(result.planner_status, "fallback_used")
        self.assertTrue(result.query_history_summary["history_available"])
        self.assertEqual(result.query_history_summary["useful_queries"][0]["query"], "history aware alpha")
        self.assertIn("search_surface_memory", result.query_history_summary)
        self.assertIn("Recent query history summary:", result.prompt)
        self.assertIn("history aware alpha", result.prompt)
        self.assertIn("Search surface memory", result.prompt)
        self.assertNotIn('"query_performance"', result.prompt)
