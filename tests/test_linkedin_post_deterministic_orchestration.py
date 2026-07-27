from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_deterministic_orchestration
from services.packaging.linkedin_post_deterministic_orchestration import (
    FinalPostDecisionReadyResult,
    prepare_final_post_decision_ready_result,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NOT_READY,
    ACTION_REPAIR_MECHANICAL,
    FinalPostAttemptHistory,
)
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput


@dataclass(frozen=True)
class EvidenceUseStub:
    evidence_id: str
    evidence_text: str
    role_in_post: str


@dataclass(frozen=True)
class PostBriefStub:
    evidence_to_use: list[EvidenceUseStub]
    core_point: str = "Use only selected evidence."


class LinkedInPostDeterministicOrchestrationTests(SimpleTestCase):
    def test_accepted_candidate_returns_decision_ready_result(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(),
            quality_review=_passing_quality_review(),
        )

        self.assertIsInstance(result, FinalPostDecisionReadyResult)
        self.assertTrue(result.gate_output.validation_passed)
        self.assertTrue(result.gate_output.diagnostics.system_linkedin_ready)
        self.assertEqual(result.decision.action, ACTION_ACCEPT)

    def test_schema_invalid_candidate_routes_through_decision_controller(self) -> None:
        payload = _valid_payload()
        payload.pop("post_text")

        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload=payload),
            quality_review=_passing_quality_review(),
        )

        self.assertFalse(result.gate_output.validation_passed)
        self.assertIn("post_text", result.gate_output.validation_error)
        self.assertEqual(result.decision.action, ACTION_REPAIR_MECHANICAL)

    def test_evidence_leak_routes_as_mechanical_failure(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(
                payload=_valid_payload(
                    post_text="This human-facing post leaks a0-summary.",
                )
            ),
            quality_review=_passing_quality_review(),
        )

        self.assertEqual(result.decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(
            result.gate_output.diagnostics.evidence_id_leaks[0].phrase,
            "a0-summary",
        )

    def test_scaffold_leak_routes_as_mechanical_failure(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(
                payload=_valid_payload(
                    hook_variants=[
                        "Separate signals before overclaiming.",
                        "Second hook",
                        "Third hook",
                    ],
                )
            ),
            quality_review=_passing_quality_review(),
        )

        self.assertEqual(result.decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(
            result.gate_output.diagnostics.scaffold_phrase_leaks[0].phrase,
            "separate signals",
        )

    def test_source_summary_leak_routes_as_mechanical_failure(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(
                payload=_valid_payload(
                    cta_variants=[
                        "According to one source, this matters.",
                        "Second CTA",
                        "Third CTA",
                    ],
                )
            ),
            quality_review=_passing_quality_review(),
        )

        self.assertEqual(result.decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(
            result.gate_output.diagnostics.source_summary_phrase_leaks[0].phrase,
            "according to one source",
        )

    def test_selected_evidence_ids_are_preserved_from_post_brief(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(
                evidence_to_use=[
                    EvidenceUseStub("a2-summary", "Third evidence.", "proof"),
                    EvidenceUseStub("a0-summary", "First evidence.", "hook"),
                ],
            ),
            candidate_output=_candidate_output(),
            quality_review=_passing_quality_review(),
        )

        self.assertEqual(
            result.gate_output.selected_evidence_ids,
            ("a2-summary", "a0-summary"),
        )

    def test_accepts_dict_style_post_brief_for_selected_evidence_ids(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_id": "a1-kp0",
                        "evidence_text": "Evidence text.",
                        "role_in_post": "proof",
                    }
                ]
            },
            candidate_output=_candidate_output(),
            quality_review=_passing_quality_review(),
        )

        self.assertEqual(result.gate_output.selected_evidence_ids, ("a1-kp0",))

    def test_missing_evidence_to_use_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={"core_point": "No selected evidence boundary."},
            expected_message="PostBrief.evidence_to_use must be a non-empty list or tuple.",
        )

    def test_none_evidence_to_use_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={"evidence_to_use": None},
            expected_message="PostBrief.evidence_to_use must be a non-empty list or tuple.",
        )

    def test_empty_evidence_to_use_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={"evidence_to_use": []},
            expected_message="PostBrief.evidence_to_use must be a non-empty list or tuple.",
        )

    def test_missing_evidence_id_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_text": "Evidence text.",
                        "role_in_post": "proof",
                    }
                ]
            },
            expected_message=(
                "PostBrief.evidence_to_use[0].evidence_id must be a non-empty string."
            ),
        )

    def test_none_evidence_id_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_id": None,
                        "evidence_text": "Evidence text.",
                        "role_in_post": "proof",
                    }
                ]
            },
            expected_message=(
                "PostBrief.evidence_to_use[0].evidence_id must be a non-empty string."
            ),
        )

    def test_empty_evidence_id_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_id": "",
                        "evidence_text": "Evidence text.",
                        "role_in_post": "proof",
                    }
                ]
            },
            expected_message=(
                "PostBrief.evidence_to_use[0].evidence_id must be a non-empty string."
            ),
        )

    def test_whitespace_evidence_id_fails_before_gate_or_decision_controller(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_id": "   ",
                        "evidence_text": "Evidence text.",
                        "role_in_post": "proof",
                    }
                ]
            },
            expected_message=(
                "PostBrief.evidence_to_use[0].evidence_id must be a non-empty string."
            ),
        )

    def test_mixed_valid_and_invalid_evidence_items_fail_without_skipping_invalid_item(self) -> None:
        self._assert_malformed_evidence_boundary_fails(
            post_brief={
                "evidence_to_use": [
                    {
                        "evidence_id": "a0-summary",
                        "evidence_text": "Valid evidence.",
                        "role_in_post": "proof",
                    },
                    {
                        "evidence_id": "",
                        "evidence_text": "Invalid evidence.",
                        "role_in_post": "practical_point",
                    },
                ]
            },
            expected_message=(
                "PostBrief.evidence_to_use[1].evidence_id must be a non-empty string."
            ),
        )

    def test_missing_quality_review_explicitly_returns_not_ready_without_repair_execution(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(),
            quality_review=None,
        )

        self.assertTrue(result.gate_output.validation_passed)
        self.assertEqual(result.decision.action, ACTION_NOT_READY)
        self.assertEqual(result.decision.reason, "missing quality review")
        self.assertIsNone(result.decision.repair_type)
        self.assertFalse(result.decision.needs_human_review)

    def test_inputs_are_not_mutated(self) -> None:
        post_brief = _post_brief()
        payload = _valid_payload()
        candidate_output = _candidate_output(payload=payload)
        quality_review = _passing_quality_review()
        post_brief_before = copy.deepcopy(post_brief)
        payload_before = copy.deepcopy(payload)
        quality_review_before = copy.deepcopy(quality_review)

        result = prepare_final_post_decision_ready_result(
            post_brief=post_brief,
            candidate_output=candidate_output,
            quality_review=quality_review,
        )

        self.assertEqual(post_brief, post_brief_before)
        self.assertEqual(payload, payload_before)
        self.assertEqual(quality_review, quality_review_before)
        self.assertIs(result.post_brief, post_brief)
        self.assertIs(result.candidate_output, candidate_output)
        self.assertIs(result.gate_output.payload, payload)

    def test_attempt_history_is_preserved_when_provided(self) -> None:
        history = FinalPostAttemptHistory(attempts=[])

        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(),
            quality_review=_passing_quality_review(),
            attempt_history=history,
        )

        self.assertIs(result.attempt_history, history)

    def test_result_to_dict_is_json_serializable(self) -> None:
        result = prepare_final_post_decision_ready_result(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(),
            quality_review=_passing_quality_review(),
        )

        serialized = json.dumps(result.to_dict(), sort_keys=True)

        self.assertIn("selected_evidence_ids", serialized)
        self.assertIn("deterministic_checks_passed", serialized)

    def test_orchestration_does_not_import_generator_or_django_models(self) -> None:
        source = inspect.getsource(linkedin_post_deterministic_orchestration)

        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("apps.packaging.models", source)

    def test_orchestration_does_not_execute_prompt_api_provider_model_or_repair(self) -> None:
        source = inspect.getsource(linkedin_post_deterministic_orchestration)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("prompt_registry", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)
        self.assertNotIn("model=", source)

    def test_orchestration_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_deterministic_orchestration).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)

    def _assert_malformed_evidence_boundary_fails(
        self,
        *,
        post_brief,
        expected_message: str,
    ) -> None:
        post_brief_before = copy.deepcopy(post_brief)
        candidate_output = _candidate_output()
        candidate_payload_before = copy.deepcopy(candidate_output.payload)

        with (
            patch.object(
                linkedin_post_deterministic_orchestration,
                "run_final_post_deterministic_gate",
            ) as gate,
            patch.object(
                linkedin_post_deterministic_orchestration,
                "FinalPostDecisionController",
            ) as controller,
        ):
            with self.assertRaises(ValueError) as captured:
                prepare_final_post_decision_ready_result(
                    post_brief=post_brief,
                    candidate_output=candidate_output,
                    quality_review=_passing_quality_review(),
                )

        self.assertEqual(str(captured.exception), expected_message)
        gate.assert_not_called()
        controller.assert_not_called()
        self.assertEqual(post_brief, post_brief_before)
        self.assertEqual(candidate_output.payload, candidate_payload_before)


def _post_brief(
    *,
    evidence_to_use: list[EvidenceUseStub] | None = None,
) -> PostBriefStub:
    return PostBriefStub(
        evidence_to_use=evidence_to_use
        or [
            EvidenceUseStub(
                evidence_id="a0-summary",
                evidence_text="Remote work policy needs clearer expectations.",
                role_in_post="proof",
            ),
            EvidenceUseStub(
                evidence_id="a1-kp0",
                evidence_text="Isolation risk needs explicit support.",
                role_in_post="practical_point",
            ),
        ]
    )


def _candidate_output(*, payload: dict | None = None) -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=payload or _valid_payload(),
        raw_output=None,
        provider=None,
        model=None,
        prompt_name=None,
        prompt_version=None,
        token_usage=None,
        cost_metadata=None,
    )


def _valid_payload(**overrides) -> dict:
    payload = {
        "post_text": "Clear final post text for LinkedIn.",
        "hook_variants": ["First hook", "Second hook", "Third hook"],
        "cta_variants": ["First CTA", "Second CTA", "Third CTA"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }
    payload.update(overrides)
    return payload


def _passing_quality_review(**overrides) -> dict:
    quality_review = {
        "scores": {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 3,
            "author_point_of_view": 4,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        },
        "total_score": 37,
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
    }
    quality_review.update(overrides)
    return quality_review
