import json
from unittest.mock import MagicMock, patch

from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase, override_settings

from apps.topics.models import Topic, TopicSource
from services.sources.candidates import SourceCandidateStatus
from services.sources.research_orchestrator import SourceResearchResult, run_source_research
from services.sources.research_queries import build_research_query_plan
from services.sources.search_provider import FakeSearchProvider, SearchProviderResult
from services.sources.serpapi_provider import SerpApiSearchProvider


class _TopicStub:
    def __init__(self, name: str, keywords) -> None:
        self.name = name
        self.keywords = keywords


class SourceResearchOrchestratorTests(SimpleTestCase):
    def _provider_for_first_query(self, topic_name: str, keywords, results: list[dict]) -> FakeSearchProvider:
        topic = _TopicStub(topic_name, keywords)
        plan = build_research_query_plan(topic)
        return FakeSearchProvider({plan.query_items[0].query: results})

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
        self.assertIn("selected_query_angle_key", result.diagnostics)
        self.assertIn("selected_query_angle_suffix", result.diagnostics)

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

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.research_orchestrator.search_research_query_plan")
    def test_orchestrator_uses_configured_serpapi_provider_when_no_provider_is_passed(
        self,
        mock_search,
    ) -> None:
        mock_search.return_value = SearchProviderResult(provider_name="serpapi", results=(), diagnostics={})

        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        result = run_source_research(topic)

        provider = mock_search.call_args.args[1]
        self.assertIsInstance(provider, SerpApiSearchProvider)
        self.assertEqual(result.diagnostics["search_provider_status"], "ready")
        self.assertEqual(result.diagnostics["search_provider_name"], "serpapi")

    @override_settings(
        SEARCH_PROVIDER_ENABLED=True,
        SEARCH_PROVIDER="serpapi",
        SEARCH_PROVIDER_API_KEY="test-key",
    )
    @patch("services.sources.serpapi_provider.urlopen")
    def test_orchestrator_can_use_configured_serpapi_provider_end_to_end_with_mocked_http(
        self,
        mock_urlopen,
    ) -> None:
        response = MagicMock()
        response.read.return_value = json.dumps(
            {
                "organic_results": [
                    {
                        "position": 1,
                        "title": "AI automation implementation guide",
                        "link": "https://example.com/ai-guide",
                        "snippet": "Practical guide for AI automation with Zapier and Make.",
                        "source": "Example",
                    }
                ]
            }
        ).encode("utf-8")
        mock_urlopen.return_value.__enter__.return_value = response

        result = run_source_research(_TopicStub("AI automation", ["Zapier", "Make"]))

        self.assertEqual(result.diagnostics["search_provider_status"], "ready")
        self.assertEqual(result.diagnostics["search_provider_name"], "serpapi")
        self.assertEqual(result.diagnostics["raw_result_count"], 1)
        self.assertEqual(len(result.candidate_inputs), 1)
        self.assertEqual(result.candidate_inputs[0].url, "https://example.com/ai-guide")

    def test_generic_benefits_listicles_are_rejected_across_domains(self) -> None:
        ai_result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "Top 10 benefits of AI automation for businesses",
                        "url": "https://example.com/benefits-ai",
                        "snippet": "Transform your business, streamline your operations, and boost productivity fast.",
                    }
                ],
            ),
        )
        sleep_result = run_source_research(
            _TopicStub("Child sleep", ["baby sleep"]),
            self._provider_for_first_query(
                "Child sleep",
                ["baby sleep"],
                [
                    {
                        "title": "Top 10 benefits of sleep training for your baby",
                        "url": "https://example.com/benefits-sleep",
                        "snippet": "Transform your family nights and boost better sleep with easy wins.",
                    }
                ],
            ),
        )

        for result in (ai_result, sleep_result):
            self.assertEqual(result.evaluated_candidates[0].status, SourceCandidateStatus.REJECTED)
            self.assertIn("generic_benefits_listicle", result.evaluated_candidates[0].diagnostics["source_content_type"])
            self.assertIn("generic benefits/listicle SEO pattern", " ".join(result.evaluated_candidates[0].rejection_reasons))

    def test_service_and_product_pages_are_rejected_across_domains(self) -> None:
        ai_result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "AI automation consulting services",
                        "url": "https://example.com/services/ai-automation",
                        "snippet": "Book a demo and contact sales to see how we help businesses automate.",
                    },
                    {
                        "title": "AI automation platform",
                        "url": "https://example.com/platform/ai",
                        "snippet": "Our platform helps teams get started today.",
                    },
                ],
            ),
        )
        sleep_result = run_source_research(
            _TopicStub("Child sleep", ["baby sleep"]),
            self._provider_for_first_query(
                "Child sleep",
                ["baby sleep"],
                [
                    {
                        "title": "Baby sleep consultant program",
                        "url": "https://example.com/services/sleep-consulting",
                        "snippet": "Schedule a call for our sleep consulting services.",
                    }
                ],
            ),
        )

        self.assertTrue(all(candidate.status == SourceCandidateStatus.REJECTED for candidate in ai_result.evaluated_candidates))
        self.assertEqual(sleep_result.evaluated_candidates[0].status, SourceCandidateStatus.REJECTED)
        self.assertTrue(
            any(
                phrase in " ".join(sleep_result.evaluated_candidates[0].rejection_reasons)
                for phrase in ("commercial service-page signals", "product/demo/pricing intent")
            )
        )

    def test_research_study_and_expert_sources_are_accepted(self) -> None:
        result = run_source_research(
            _TopicStub("Child sleep", ["infant sleep"]),
            self._provider_for_first_query(
                "Child sleep",
                ["infant sleep"],
                [
                    {
                        "title": "Infant sleep intervention study: methodology and limitations",
                        "url": "https://example.org/study",
                        "snippet": "Study with methodology, evidence, and limitations for infant sleep intervention.",
                    },
                    {
                        "title": "Expert perspective on sleep training risks and tradeoffs",
                        "url": "https://analysis.example.org/perspective",
                        "snippet": "Expert analysis of risks, tradeoffs, and debate around sleep training.",
                    },
                ],
            ),
        )

        self.assertTrue(all(candidate.status == SourceCandidateStatus.ACCEPTED for candidate in result.evaluated_candidates))
        self.assertIn(
            result.evaluated_candidates[0].diagnostics["source_content_type"],
            {"scientific_study", "research_report", "survey_or_data_report"},
        )
        self.assertEqual(
            result.evaluated_candidates[1].diagnostics["source_content_type"],
            "debate_or_perspective",
        )

    def test_news_style_reporting_is_classified_as_news_article(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "New report highlights AI automation risks for small businesses",
                        "url": "https://news.example.org/ai-risks-report",
                        "snippet": "Recent coverage says businesses are reassessing automation plans after new findings and expert warnings.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(candidate.diagnostics["source_content_type"], "news_article")

    def test_research_report_remains_classified_as_survey_or_data_report(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "AI automation adoption survey: implementation risks and data",
                        "url": "https://research.example.org/adoption-survey",
                        "snippet": "Survey data, methodology, findings, limitations, and adoption trends.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(candidate.diagnostics["source_content_type"], "survey_or_data_report")

    def test_guideline_remains_classified_as_institutional_guideline(self) -> None:
        result = run_source_research(
            _TopicStub("Child sleep", ["infant sleep"]),
            self._provider_for_first_query(
                "Child sleep",
                ["infant sleep"],
                [
                    {
                        "title": "Pediatric sleep guideline: evidence, recommendations, and risks",
                        "url": "https://pediatrics.example.org/guideline/sleep",
                        "snippet": "Clinical guideline with evidence, recommendations, risks, and limitations for infant sleep.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(candidate.diagnostics["source_content_type"], "institutional_guideline")

    def test_news_like_promotional_page_is_still_rejected(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "Latest AI automation platform update - book a demo",
                        "url": "https://example.com/platform/update",
                        "snippet": "Our platform helps you transform operations. Contact sales to get started.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.REJECTED)
        self.assertIn(
            "product/demo/pricing intent",
            " ".join(candidate.rejection_reasons),
        )

    def test_recent_research_report_is_accepted_with_freshness_signals(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "AI automation adoption survey: implementation risks and data",
                        "url": "https://research.example.org/2026/05/adoption-survey",
                        "snippet": "2026-05-12 survey data, methodology, findings, limitations, and adoption trends.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertIn(candidate.diagnostics["freshness_status"], {"fresh", "acceptable"})

    def test_stale_2018_result_is_rejected(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "AI automation adoption report 2018",
                        "url": "https://research.example.org/2018/adoption-report",
                        "snippet": "2018 report with methodology, findings, and implementation details.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.REJECTED)
        self.assertIn(candidate.diagnostics["freshness_status"], {"stale", "very_stale"})
        self.assertIn("stale", " ".join(candidate.rejection_reasons))

    def test_unknown_date_substantive_result_can_pass(self) -> None:
        result = run_source_research(
            _TopicStub("Child sleep", ["infant sleep"]),
            self._provider_for_first_query(
                "Child sleep",
                ["infant sleep"],
                [
                    {
                        "title": "Pediatric sleep guideline: evidence, recommendations, and risks",
                        "url": "https://pediatrics.example.org/guideline/sleep",
                        "snippet": "Evidence, recommendations, methodology, and risks for infant sleep.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertIn(candidate.status, {SourceCandidateStatus.ACCEPTED, SourceCandidateStatus.NEEDS_REVIEW})
        self.assertEqual(candidate.diagnostics["freshness_status"], "unknown")

    def test_unknown_date_weak_generic_result_is_rejected(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "The ultimate guide to workflow automation",
                        "url": "https://example.com/ultimate-guide",
                        "snippet": "Boost your business and streamline your operations.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.REJECTED)
        self.assertEqual(candidate.diagnostics["freshness_status"], "unknown")

    def test_fresh_service_page_is_still_rejected(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "Latest AI automation platform update - book a demo",
                        "url": "https://example.com/2026/platform/update",
                        "snippet": "2026-05-14 our platform helps you transform operations. Contact sales to get started.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.REJECTED)
        rejection_text = " ".join(candidate.rejection_reasons) + " " + candidate.diagnostics["quality_rejection_reason"]
        self.assertTrue(
            "commercial" in rejection_text or "product/demo/pricing intent" in rejection_text
        )

    def test_case_study_and_substantive_company_blog_can_pass(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "Workflow automation case study: implementation details and lessons learned",
                        "url": "https://example.com/case-study",
                        "snippet": "Concrete implementation details, examples, tradeoffs, and lessons learned.",
                    },
                    {
                        "title": "Company engineering blog: automation migration methodology and tradeoffs",
                        "url": "https://company.example.com/blog/automation-migration",
                        "snippet": "Implementation details, methodology, limitations, and real examples from the migration.",
                    },
                ],
            ),
        )

        self.assertTrue(all(candidate.status == SourceCandidateStatus.ACCEPTED for candidate in result.evaluated_candidates))
        self.assertEqual(result.evaluated_candidates[0].diagnostics["source_content_type"], "concrete_case_study")
        self.assertIn(
            result.evaluated_candidates[1].diagnostics["source_content_type"],
            {
                "research_report",
                "substantive_longform_article",
                "practical_guide_with_concrete_steps",
                "concrete_case_study",
            },
        )

    def test_generic_company_blog_without_substance_is_rejected(self) -> None:
        result = run_source_research(
            _TopicStub("AI automation", ["workflow automation"]),
            self._provider_for_first_query(
                "AI automation",
                ["workflow automation"],
                [
                    {
                        "title": "Why your business needs workflow automation",
                        "url": "https://company.example.com/blog/workflow-benefits",
                        "snippet": "Boost your business, transform your operations, and get started today with our platform.",
                    }
                ],
            ),
        )

        candidate = result.evaluated_candidates[0]
        self.assertEqual(candidate.status, SourceCandidateStatus.REJECTED)
        self.assertIn(
            candidate.diagnostics["source_content_type"],
            {
                "generic_benefits_listicle",
                "generic_company_blog_without_substance",
                "vague_promotional_article",
                "lead_generation_article",
            },
        )

    @override_settings(SEARCH_RECENCY_MONTHS=3)
    @patch("services.sources.research_orchestrator.search_research_query_plan")
    def test_orchestrator_surfaces_configured_recency_diagnostics(self, mock_search) -> None:
        mock_search.return_value = SearchProviderResult(
            provider_name="serpapi",
            results=(),
            diagnostics={},
        )

        topic = _TopicStub("AI automation", ["Zapier", "Make"])
        result = run_source_research(topic)

        self.assertEqual(result.diagnostics["search_recency_months"], 3)
        self.assertEqual(result.diagnostics["search_time_filter"], "qdr:m3")


class SourceResearchOrchestratorPersistenceTests(TestCase):
    def test_no_topic_source_rows_are_created(self) -> None:
        user = get_user_model().objects.create_user(username="orchestrator-user", password="pw")
        topic = Topic.objects.create(user=user, name="Infant sleep", keywords=["safe sleep", "SIDS"])
        before = TopicSource.objects.count()

        result = run_source_research(topic, FakeSearchProvider({}))

        self.assertEqual(result.provider_result.results, ())
        self.assertEqual(TopicSource.objects.count(), before)
