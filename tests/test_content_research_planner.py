import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase, override_settings

from services.sources.content_research_planner import (
    MAX_FINAL_QUERY_COUNT,
    build_content_research_planner_prompt,
    create_content_research_plan,
)


class _TopicStub:
    def __init__(self, title: str, keywords, description: str = "") -> None:
        self.name = title
        self.keywords = keywords
        self.description = description


class ContentResearchPlannerTests(SimpleTestCase):
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
        self.assertIn('"content_tension_opportunities"', prompt)
