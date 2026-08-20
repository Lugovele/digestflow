from __future__ import annotations

import json

from django.test import SimpleTestCase

from services.packaging.linkedin_post_pipeline import (
    AngleDecision,
    AuthorialVoiceDirective,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_repair_target_enforcement import (
    RepairTargetEnforcementDiagnostics,
    evaluate_repair_target_enforcement,
)


class LinkedInPostRepairTargetEnforcementTests(SimpleTestCase):
    def test_author_pov_requires_score_five_when_explicit_author_statement_required(self) -> None:
        diagnostics = evaluate_repair_target_enforcement(
            quality_review=_quality_review(
                scores={"author_point_of_view": 4},
                failed_criteria=[],
            ),
            initiating_failed_criterion="author_point_of_view",
            angle_decision=_angle_decision(explicit=True),
        )

        self.assertEqual(diagnostics.initiating_failed_criterion, "author_point_of_view")
        self.assertEqual(diagnostics.target_quality_score, 4)
        self.assertEqual(diagnostics.target_required_minimum, 5)
        self.assertFalse(diagnostics.repair_target_fixed)
        self.assertIn("score 4 < required 5", diagnostics.repair_target_failure_reason)

    def test_author_pov_score_five_satisfies_explicit_author_statement_requirement(self) -> None:
        diagnostics = evaluate_repair_target_enforcement(
            quality_review=_quality_review(
                scores={"author_point_of_view": 5},
                failed_criteria=[],
            ),
            initiating_failed_criterion="author_point_of_view",
            angle_decision=_angle_decision(explicit=True),
        )

        self.assertTrue(diagnostics.repair_target_fixed)
        self.assertEqual(diagnostics.target_required_minimum, 5)

    def test_author_pov_score_five_requirement_reads_canonical_angle_decision_object(self) -> None:
        diagnostics = evaluate_repair_target_enforcement(
            quality_review=_quality_review(
                scores={"author_point_of_view": 4},
                failed_criteria=[],
            ),
            initiating_failed_criterion="author_point_of_view",
            angle_decision=_angle_decision_object(explicit=True),
        )

        self.assertEqual(diagnostics.target_quality_score, 4)
        self.assertEqual(diagnostics.target_required_minimum, 5)
        self.assertFalse(diagnostics.repair_target_fixed)
        self.assertIn("author_point_of_view score 4 < required 5", diagnostics.repair_target_failure_reason)

    def test_cta_uses_canonical_threshold_without_hardcoded_required_minimum(self) -> None:
        rubric = get_quality_evaluator_rubric_payload()
        self.assertNotIn("cta", rubric.required_minimums)

        diagnostics = evaluate_repair_target_enforcement(
            quality_review=_quality_review(scores={"cta": 3}, failed_criteria=[]),
            initiating_failed_criterion="cta",
            angle_decision=_angle_decision(explicit=True),
        )

        self.assertEqual(diagnostics.target_required_minimum, rubric.score_max - 1)
        self.assertFalse(diagnostics.repair_target_fixed)
        self.assertIn("cta score 3 < required 4", diagnostics.repair_target_failure_reason)

    def test_target_still_failed_is_not_fixed_even_when_score_meets_threshold(self) -> None:
        diagnostics = evaluate_repair_target_enforcement(
            quality_review=_quality_review(
                scores={"cta": 5},
                failed_criteria=["cta"],
            ),
            initiating_failed_criterion="cta",
            angle_decision=_angle_decision(explicit=True),
        )

        self.assertFalse(diagnostics.repair_target_fixed)
        self.assertIn("remains in failed_criteria", diagnostics.repair_target_failure_reason)

    def test_unavailable_quality_review_is_not_classified_as_target_failure(self) -> None:
        diagnostics = evaluate_repair_target_enforcement(
            quality_review=None,
            initiating_failed_criterion="cta",
            angle_decision=_angle_decision(explicit=True),
        )

        self.assertIsNone(diagnostics.target_quality_score)
        self.assertIsNone(diagnostics.repair_target_fixed)
        self.assertIn("quality review unavailable", diagnostics.repair_target_failure_reason)

    def test_to_dict_is_json_serializable(self) -> None:
        diagnostics = RepairTargetEnforcementDiagnostics(
            initiating_failed_criterion="cta",
            target_quality_score=4,
            target_required_minimum=4,
            repair_target_fixed=True,
            repair_target_failure_reason="cta fixed",
        )

        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)

        self.assertIn("repair_target_fixed", serialized)


def _angle_decision(*, explicit: bool) -> dict:
    return {
        "authorial_voice_directive": {
            "personal_presence_requirement": (
                "explicit_author_owned_statement_required" if explicit else "none"
            )
        }
    }


def _angle_decision_object(*, explicit: bool) -> AngleDecision:
    return AngleDecision(
        controlling_angle="Compare what the evidence supports with what market rhetoric implies.",
        reader_problem="Readers need a clearer distinction between signal and narrative.",
        author_position="The author rejects treating policy attention as proof of adoption.",
        main_tension="Policy momentum versus evidence discipline.",
        supporting_evidence_ids=["ev-1"],
        angle_to_avoid=["unsupported certainty"],
        authorial_voice_directive=AuthorialVoiceDirective(
            authorial_observation="The evidence supports a narrower reading than the market narrative.",
            rejected_reading="Reject treating attention as proof of stable adoption.",
            why_distinction_matters="The distinction matters because readers need evidence-bounded judgment.",
            personal_presence_requirement=(
                "explicit_author_owned_statement_required" if explicit else "none"
            ),
            first_person_policy="allowed_not_required",
            forbidden_author_claims=("personal experience",),
        ),
    )


def _quality_review(*, scores: dict[str, int], failed_criteria: list[str]) -> dict:
    canonical_scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 5,
        "practical_value": 4,
        "cta": 4,
    }
    canonical_scores.update(scores)
    return {
        "scores": canonical_scores,
        "total_score": sum(canonical_scores.values()),
        "pass": True,
        "failed_criteria": failed_criteria,
        "automatic_fail_reason": "",
        "notes": [],
    }
