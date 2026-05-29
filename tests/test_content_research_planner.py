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
                "useful_angles": [{"angle": "recent examples", "count": 1}],
                "weak_angles": [],
                "planning_guidance": ["Useful directions so far: recent examples. Create fresh variants around those angles instead of reusing the same wording."],
            },
        )

        self.assertIn("Recent query history summary:", prompt)
        self.assertIn("AI education teens classroom examples", prompt)
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
        self.assertIn("Recent query history summary:", result.prompt)
        self.assertIn("history aware alpha", result.prompt)
        self.assertNotIn('"query_performance"', result.prompt)
