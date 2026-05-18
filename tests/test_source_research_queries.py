from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource
from services.sources.research_queries import (
    ResearchQueryIntent,
    build_research_query_plan,
)


class _TopicStub:
    def __init__(self, name: str, keywords: list[str]) -> None:
        self.name = name
        self.keywords = keywords


class SourceResearchQueryPlanTests(SimpleTestCase):
    def test_query_planner_uses_topic_name(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "SIDS"]))

        self.assertEqual(plan.topic_name, "Infant sleep")
        self.assertTrue(any("Infant sleep" in item.query for item in plan.query_items))

    def test_query_planner_uses_topic_keywords(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "newborn"]))

        rendered_queries = " ".join(item.query for item in plan.query_items)
        self.assertIn("safe sleep", rendered_queries)
        self.assertIn("newborn", rendered_queries)

    def test_comma_separated_string_keywords_are_parsed_as_phrases(self) -> None:
        plan = build_research_query_plan(
            _TopicStub(
                "Infant sleep",
                "safe sleep, SIDS, newborn, sleep environment",
            )
        )

        self.assertEqual(
            plan.topic_keywords,
            ("safe sleep", "SIDS", "newborn", "sleep environment"),
        )

    def test_multi_word_keyword_phrases_are_preserved(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", "safe sleep, sleep environment")
        )

        self.assertIn("safe sleep", plan.topic_keywords)
        self.assertIn("sleep environment", plan.topic_keywords)

    def test_empty_comma_values_are_ignored(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", "safe sleep, , newborn,  , sleep environment")
        )

        self.assertEqual(
            plan.topic_keywords,
            ("safe sleep", "newborn", "sleep environment"),
        )

    def test_query_planner_does_not_introduce_focus_terms_concept(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "SIDS"]))

        self.assertFalse(hasattr(plan, "focus_terms"))
        self.assertFalse(any("focus_terms" in item.diagnostics for item in plan.query_items))

    def test_iterable_keywords_still_work(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", ["safe sleep", "SIDS", "newborn"])
        )

        self.assertEqual(plan.topic_keywords, ("safe sleep", "SIDS", "newborn"))

    def test_iterable_keywords_containing_comma_separated_strings_are_split_cleanly(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", ["safe sleep, SIDS", "newborn", "sleep environment"])
        )

        self.assertEqual(
            plan.topic_keywords,
            ("safe sleep", "SIDS", "newborn", "sleep environment"),
        )

    def test_no_character_splitting_regression(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", "safe sleep, SIDS, newborn, sleep environment")
        )

        self.assertNotIn("s", plan.topic_keywords)
        self.assertNotIn("a", plan.topic_keywords)
        self.assertNotIn("f", plan.topic_keywords)

    def test_query_planner_does_not_invent_unrelated_topic_keywords(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make", "n8n", "small business workflows"]))

        rendered_queries = " ".join(item.query.casefold() for item in plan.query_items)
        self.assertNotIn("langchain", rendered_queries)
        self.assertNotIn("autogpt", rendered_queries)
        self.assertNotIn("robotics", rendered_queries)
        self.assertNotIn("enterprise rpa", rendered_queries)

    def test_generated_queries_are_not_one_giant_keyword_dump(self) -> None:
        plan = build_research_query_plan(
            _TopicStub(
                "Infant sleep",
                ["safe sleep", "SIDS", "newborn", "sleep environment"],
            )
        )

        for item in plan.query_items:
            self.assertLessEqual(len(item.query.split()), 12)
        self.assertTrue(any("SIDS" not in item.query for item in plan.query_items))

    def test_generated_queries_use_keyword_phrases_not_characters(self) -> None:
        plan = build_research_query_plan(
            _TopicStub("Infant sleep", "safe sleep, SIDS, newborn, sleep environment")
        )

        queries = [item.query for item in plan.query_items]
        self.assertTrue(any("safe sleep" in query for query in queries))
        self.assertTrue(any("sleep environment" in query for query in queries))
        self.assertFalse(any(" s a " in f" {query} " for query in queries))

    def test_generated_queries_are_deduplicated(self) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel", "family travel"]))

        queries = [item.query for item in plan.query_items]
        self.assertEqual(len(queries), len(set(queries)))

    def test_general_topic_gets_general_intents(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "SIDS", "newborn"]))

        intents = {item.intent for item in plan.query_items}
        self.assertEqual(plan.topic_domain, "general")
        self.assertIn(ResearchQueryIntent.OFFICIAL_GUIDELINES, intents)
        self.assertIn(ResearchQueryIntent.EVIDENCE_BASED, intents)
        self.assertIn(ResearchQueryIntent.ORGANIZATION_RESOURCES, intents)
        self.assertNotIn(ResearchQueryIntent.ENGINEERING_BLOG, intents)

    def test_technical_topic_gets_technical_intents(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make", "n8n", "small business workflows"]))

        intents = {item.intent for item in plan.query_items}
        self.assertEqual(plan.topic_domain, "technical")
        self.assertIn(ResearchQueryIntent.IMPLEMENTATION_GUIDE, intents)
        self.assertIn(ResearchQueryIntent.CASE_STUDY, intents)
        self.assertIn(ResearchQueryIntent.ENGINEERING_BLOG, intents)
        self.assertIn(ResearchQueryIntent.BEST_PRACTICES, intents)

    def test_query_items_expose_intent_labels_and_reasons(self) -> None:
        plan = build_research_query_plan(_TopicStub("Infant sleep", ["safe sleep", "SIDS"]))

        first_item = plan.query_items[0]
        self.assertIsInstance(first_item.intent, ResearchQueryIntent)
        self.assertTrue(first_item.reason)
        self.assertTrue(first_item.source_type_hint)

    def test_query_plan_exposes_useful_diagnostics(self) -> None:
        plan = build_research_query_plan(_TopicStub("AI automation", ["Zapier", "Make", "n8n"]))

        self.assertEqual(plan.diagnostics["topic_domain"], "technical")
        self.assertIn("domain_diagnostics", plan.diagnostics)
        self.assertIn("query_count", plan.diagnostics)
        self.assertIn("used_topic_keywords", plan.diagnostics)

    def test_query_planning_does_not_require_http_or_template_context(self) -> None:
        plan = build_research_query_plan(_TopicStub("Education for teenagers", ["study habits", "online learning"]))

        self.assertEqual(plan.topic_name, "Education for teenagers")
        self.assertTrue(plan.query_items)

    @patch("socket.create_connection", side_effect=AssertionError("network should not be used"))
    def test_query_planning_does_not_call_external_network(self, _mock_network) -> None:
        plan = build_research_query_plan(_TopicStub("Travel planning", ["family travel", "budget travel"]))

        self.assertTrue(plan.query_items)


class SourceResearchQueryPlanPersistenceTests(TestCase):
    def test_query_planning_does_not_create_topic_sources(self) -> None:
        user = get_user_model().objects.create_user(username="query-plan-user", password="pw")
        topic = Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()

        plan = build_research_query_plan(topic)

        self.assertTrue(plan.query_items)
        self.assertEqual(TopicSource.objects.count(), before)
