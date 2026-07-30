import json
import re
from pathlib import Path

from django.test import SimpleTestCase

from services.packaging.linkedin_post_pipeline import (
    PipelineInput,
    SelectedArticle,
    build_angle_decision_from_contextual_evidence_pack,
    build_article_evidence_pack_from_pipeline_input,
    build_contextual_evidence_pack_from_article_evidence_pack,
    build_post_brief_from_angle_decision,
    validate_angle_decision_for_contextual_evidence,
    validate_article_evidence_pack_for_pipeline_input,
    validate_contextual_evidence_pack_for_article_evidence,
    validate_linkedin_post_stage_relationships,
    validate_pipeline_input,
)


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "linkedin_post_cases"
FINANCE_FIXTURE_CASE_IDS = {"topic_200_digest_134"}
FINANCE_SPECIFIC_TERMS = [
    "investment advice",
    "financial advice",
    "time the market",
    "market timing",
    "bitcoin",
    "crypto",
]
FINANCE_SPECIFIC_WORDS = ["buy", "sell", "hold"]


def _fixture_case_ids() -> list[str]:
    return sorted(path.stem for path in FIXTURE_DIR.glob("*.json"))


def _load_fixture(case_id: str) -> dict:
    fixture_path = FIXTURE_DIR / f"{case_id}.json"
    with fixture_path.open(encoding="utf-8") as fixture_file:
        return json.load(fixture_file)


def _build_pipeline_input_from_fixture(fixture: dict) -> PipelineInput:
    articles = [
        SelectedArticle(
            source_index=article["source_index"],
            title=article["title"],
            url=article["url"],
            summary=article["summary"],
            key_points=article.get("key_points", []),
            source_name=str(article.get("source_name") or "").strip(),
            published_at=str(article.get("published_at") or "").strip(),
            content_type=str(article.get("content_type") or "").strip(),
            confidence=article.get("confidence"),
        )
        for article in fixture["articles"]
    ]
    pipeline_input = PipelineInput(
        digest_id=fixture["digest"]["id"],
        topic_name=fixture["topic"]["name"],
        digest_title=fixture["digest"]["title"],
        articles=articles,
        author_profile={},
    )
    validate_pipeline_input(pipeline_input)
    return pipeline_input


def _run_staged_chain(pipeline_input: PipelineInput) -> dict:
    article_evidence_pack = build_article_evidence_pack_from_pipeline_input(
        pipeline_input
    )
    contextual_evidence_pack = build_contextual_evidence_pack_from_article_evidence_pack(
        article_evidence_pack
    )
    angle_decision = build_angle_decision_from_contextual_evidence_pack(
        contextual_evidence_pack
    )
    post_brief = build_post_brief_from_angle_decision(
        contextual_evidence_pack,
        angle_decision,
    )
    return {
        "article_evidence_pack": article_evidence_pack,
        "contextual_evidence_pack": contextual_evidence_pack,
        "angle_decision": angle_decision,
        "post_brief": post_brief,
    }


def _summarize_post_brief_output(case_id: str, chain: dict) -> dict:
    angle_decision = chain["angle_decision"]
    post_brief = chain["post_brief"]
    return {
        "case_id": case_id,
        "selected_evidence_ids": list(angle_decision.supporting_evidence_ids),
        "controlling_angle": angle_decision.controlling_angle,
        "selected_evidence_roles": [
            item.role_in_post for item in post_brief.evidence_to_use
        ],
        "opening_direction": post_brief.opening_direction,
        "core_point": post_brief.core_point,
        "practical_point": post_brief.practical_point,
        "cta_direction": post_brief.cta_direction,
    }


class LinkedInPostFixtureEvaluationTests(SimpleTestCase):
    case_ids = _fixture_case_ids()

    def test_fixture_top_level_shape_is_valid(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)

                self.assertEqual(fixture["case_id"], case_id)
                self.assertTrue(fixture["source_type"].strip())
                self.assertIsInstance(fixture["topic"]["id"], int)
                self.assertTrue(fixture["topic"]["name"].strip())
                self.assertTrue(fixture["topic"]["source_mode"].strip())
                self.assertIsInstance(fixture["digest"]["id"], int)
                self.assertIsInstance(fixture["digest"]["run_id"], int)
                self.assertTrue(fixture["digest"]["title"].strip())
                self.assertGreaterEqual(len(fixture["articles"]), 1)
                for expected_index, article in enumerate(fixture["articles"]):
                    self.assertEqual(article["source_index"], expected_index)
                    self.assertTrue(article["url"].strip())
                    self.assertTrue(article["title"].strip())
                    self.assertTrue(article["summary"].strip())
                    self.assertIsInstance(article["key_points"], list)

    def test_fixture_expected_behavior_metadata_is_present(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                expected_behavior = fixture["expected_behavior"]

                self.assertTrue(expected_behavior["must_preserve"])
                self.assertTrue(expected_behavior["must_not_infer"])
                self.assertTrue(expected_behavior["preferred_angle"].strip())
                for field_name in [
                    "generic_output_risk",
                    "source_dominance_risk",
                    "metric_hallucination_risk",
                    "investment_advice_risk",
                ]:
                    if field_name in expected_behavior:
                        self.assertTrue(expected_behavior[field_name].strip())

    def test_live_source_fixtures_run_through_deterministic_staged_chain_to_post_brief(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)

                chain = _run_staged_chain(pipeline_input)

                validate_article_evidence_pack_for_pipeline_input(
                    pipeline_input,
                    chain["article_evidence_pack"],
                )
                validate_contextual_evidence_pack_for_article_evidence(
                    chain["article_evidence_pack"],
                    chain["contextual_evidence_pack"],
                )
                validate_angle_decision_for_contextual_evidence(
                    chain["contextual_evidence_pack"],
                    chain["angle_decision"],
                )
                validate_linkedin_post_stage_relationships(
                    pipeline_input,
                    chain["article_evidence_pack"],
                    chain["contextual_evidence_pack"],
                    chain["angle_decision"],
                    chain["post_brief"],
                )
                self.assertTrue(chain["angle_decision"].supporting_evidence_ids)

    def test_fixture_selected_evidence_is_preserved_into_post_brief(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)

                selected_ids = chain["angle_decision"].supporting_evidence_ids
                brief_evidence = chain["post_brief"].evidence_to_use
                contextual_by_id = {
                    item.evidence_id: item
                    for item in chain["contextual_evidence_pack"].items
                }

                self.assertEqual(len(selected_ids), len(set(selected_ids)))
                self.assertEqual(
                    [item.evidence_id for item in brief_evidence],
                    selected_ids,
                )
                for item in brief_evidence:
                    self.assertIn(item.evidence_id, contextual_by_id)
                    self.assertEqual(
                        item.evidence_text,
                        contextual_by_id[item.evidence_id].evidence_text,
                    )

    def test_fixture_post_brief_preserves_selected_evidence_order(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                selected_ids = chain["angle_decision"].supporting_evidence_ids

                self.assertEqual(
                    [item.evidence_id for item in chain["post_brief"].evidence_to_use],
                    selected_ids,
                )

    def test_fixture_post_brief_does_not_expose_runtime_or_repair_fields(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                post_brief = chain["post_brief"]
                forbidden_fields = {
                    "provider",
                    "response_text",
                    "fallback_reason",
                    "tokens",
                    "estimated_cost_usd",
                    "repair",
                    "quality_review",
                    "debug",
                }

                for field_name in forbidden_fields:
                    self.assertFalse(hasattr(post_brief, field_name))

    def test_fixture_angle_and_post_brief_avoid_narrow_investment_advice_phrases(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                angle_decision = chain["angle_decision"]
                post_brief = chain["post_brief"]
                evaluated_text = "\n".join(
                    [
                        angle_decision.controlling_angle,
                        angle_decision.reader_problem,
                        angle_decision.author_position,
                        angle_decision.main_tension,
                        post_brief.opening_direction,
                        post_brief.pattern_interrupt,
                        post_brief.core_point,
                        post_brief.practical_point,
                        post_brief.ending_direction,
                        post_brief.cta_direction,
                    ]
                ).lower()

                for phrase in [
                    "you should buy",
                    "buy bitcoin",
                    "sell bitcoin",
                    "hold bitcoin",
                    "time the market",
                ]:
                    self.assertNotIn(phrase, evaluated_text)

    def test_non_finance_fixtures_do_not_get_finance_specific_guardrail_wording(self) -> None:
        for case_id in self.case_ids:
            if case_id in FINANCE_FIXTURE_CASE_IDS:
                continue
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                angle_decision = chain["angle_decision"]
                post_brief = chain["post_brief"]
                evaluated_text = "\n".join(
                    [
                        angle_decision.controlling_angle,
                        angle_decision.reader_problem,
                        angle_decision.author_position,
                        angle_decision.main_tension,
                        "\n".join(angle_decision.angle_to_avoid),
                        post_brief.opening_direction,
                        post_brief.pattern_interrupt,
                        post_brief.core_point,
                        post_brief.practical_point,
                        post_brief.ending_direction,
                        post_brief.cta_direction,
                    ]
                ).lower()

                for term in FINANCE_SPECIFIC_TERMS:
                    self.assertNotIn(term, evaluated_text)
                for word in FINANCE_SPECIFIC_WORDS:
                    self.assertIsNone(
                        re.search(rf"\b{re.escape(word)}\b", evaluated_text)
                    )

    def test_primary_editorial_fields_do_not_emit_process_language(self) -> None:
        process_phrases = [
            "keep claims attributed",
            "use selected evidence",
            "avoid unsupported conclusions",
            "avoid investment advice",
            "separate signals",
            "source-grounded",
        ]
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                angle_decision = chain["angle_decision"]
                post_brief = chain["post_brief"]
                editorial_text = "\n".join(
                    [
                        angle_decision.controlling_angle,
                        angle_decision.author_position,
                        post_brief.core_point,
                    ]
                ).lower()

                for phrase in process_phrases:
                    self.assertNotIn(phrase, editorial_text)

    def test_bitcoin_market_fixture_produces_content_bearing_thesis(self) -> None:
        fixture = _load_fixture("topic_200_digest_134")
        pipeline_input = _build_pipeline_input_from_fixture(fixture)
        chain = _run_staged_chain(pipeline_input)
        angle_decision = chain["angle_decision"]
        post_brief = chain["post_brief"]

        self.assertEqual(
            angle_decision.supporting_evidence_ids,
            ["a0-summary", "a1-kp0", "a2-summary"],
        )
        selected_roles_by_id = {
            item.evidence_id: item.role_in_post
            for item in post_brief.evidence_to_use
        }
        self.assertIn("proof support", selected_roles_by_id["a1-kp0"])
        self.assertNotIn("practical_point support", selected_roles_by_id["a1-kp0"])
        editorial_text = "\n".join(
            [
                angle_decision.controlling_angle,
                angle_decision.reader_problem,
                angle_decision.author_position,
                angle_decision.main_tension,
                post_brief.core_point,
                post_brief.practical_point,
            ]
        ).lower()

        for content_term in [
            "adoption",
            "forecasts",
            "confidence",
            "risk",
            "conditional",
        ]:
            self.assertIn(content_term, editorial_text)
        for defective_phrase in [
            "keep claims attributed",
            "use selected evidence",
            "avoid unsupported conclusions",
            "avoid investment advice",
            "separate signals",
            "source-grounded",
            "16.99% cagr",
            "single clean narrative",
            "signals qualify one another",
            "operate on different horizons",
        ]:
            self.assertNotIn(defective_phrase, editorial_text)
        for unselected_signal in [
            "technology reliability",
            "technology improvement",
            "transaction efficiency",
            "regulatory clarity",
            "institutional adoption",
            "defi",
        ]:
            self.assertNotIn(unselected_signal, editorial_text)
        for causal_strengthening in [
            "led to stability",
            "caused stability",
            "creates stability",
            "prevents downturns",
            "prevents declines",
            "guarantees a bottom",
        ]:
            self.assertNotIn(causal_strengthening, editorial_text)

    def test_non_finance_fixtures_do_not_emit_market_or_regulatory_signal_labels(self) -> None:
        forbidden_by_case = {
            "topic_214_digest_128": ["regulatory clarity"],
            "topic_215_digest_129": ["price positioning"],
        }
        for case_id, forbidden_signals in forbidden_by_case.items():
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)
                angle_decision = chain["angle_decision"]
                post_brief = chain["post_brief"]
                editorial_text = "\n".join(
                    [
                        angle_decision.controlling_angle,
                        angle_decision.reader_problem,
                        angle_decision.author_position,
                        angle_decision.main_tension,
                        post_brief.core_point,
                        post_brief.practical_point,
                    ]
                ).lower()

                for forbidden_signal in forbidden_signals:
                    self.assertNotIn(forbidden_signal, editorial_text)

    def test_fixture_post_brief_output_summary_helper_is_available(self) -> None:
        for case_id in self.case_ids:
            with self.subTest(case_id=case_id):
                fixture = _load_fixture(case_id)
                pipeline_input = _build_pipeline_input_from_fixture(fixture)
                chain = _run_staged_chain(pipeline_input)

                summary = _summarize_post_brief_output(case_id, chain)

                self.assertEqual(summary["case_id"], case_id)
                self.assertEqual(
                    summary["selected_evidence_ids"],
                    chain["angle_decision"].supporting_evidence_ids,
                )
                self.assertTrue(summary["controlling_angle"].strip())
                self.assertEqual(
                    len(summary["selected_evidence_roles"]),
                    len(chain["post_brief"].evidence_to_use),
                )
