import json
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


class LinkedInPostFixtureEvaluationTests(SimpleTestCase):
    case_id = "topic_200_digest_134"

    def test_fixture_expected_behavior_metadata_is_present(self) -> None:
        fixture = _load_fixture(self.case_id)
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

    def test_live_source_fixture_runs_through_deterministic_staged_chain_to_post_brief(self) -> None:
        fixture = _load_fixture(self.case_id)
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

    def test_fixture_selected_evidence_is_preserved_into_post_brief(self) -> None:
        fixture = _load_fixture(self.case_id)
        pipeline_input = _build_pipeline_input_from_fixture(fixture)
        chain = _run_staged_chain(pipeline_input)

        selected_ids = chain["angle_decision"].supporting_evidence_ids
        brief_evidence = chain["post_brief"].evidence_to_use
        contextual_by_id = {
            item.evidence_id: item for item in chain["contextual_evidence_pack"].items
        }

        self.assertEqual(
            [item.evidence_id for item in brief_evidence],
            selected_ids,
        )
        for item in brief_evidence:
            self.assertEqual(
                item.evidence_text,
                contextual_by_id[item.evidence_id].evidence_text,
            )

    def test_fixture_post_brief_preserves_selected_evidence_order(self) -> None:
        fixture = _load_fixture(self.case_id)
        pipeline_input = _build_pipeline_input_from_fixture(fixture)
        chain = _run_staged_chain(pipeline_input)
        selected_ids = chain["angle_decision"].supporting_evidence_ids

        self.assertEqual(
            [item.evidence_id for item in chain["post_brief"].evidence_to_use],
            selected_ids,
        )

    def test_fixture_post_brief_does_not_expose_runtime_or_repair_fields(self) -> None:
        fixture = _load_fixture(self.case_id)
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
        fixture = _load_fixture(self.case_id)
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
