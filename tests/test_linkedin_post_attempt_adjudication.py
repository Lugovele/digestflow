from __future__ import annotations

import copy
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_attempt_adjudication
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_NOT_RUN,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_and_quality,
    build_final_post_attempt_outcome_from_gate_grounding_and_quality,
)
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NEEDS_HUMAN_REVIEW,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_REPAIR_MECHANICAL,
    FinalPostAttemptHistory,
)
from services.packaging.linkedin_post_flow_decision import FinalPostDecisionController
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_PASS,
    FinalPostSemanticGroundingState,
    normalize_semantic_grounding_review_result,
)


class LinkedInPostAttemptAdjudicationTests(SimpleTestCase):
    def test_passing_gate_and_passing_quality_review_returns_accepted_outcome(self) -> None:
        payload = _valid_payload()
        candidate_output = _candidate_output(payload)
        gate_output = _passing_gate_output(payload)

        quality_review = _quality_review(passed=True)
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=candidate_output,
            gate_output=gate_output,
            quality_evaluation=_quality_state(quality_review),
            attempt_index=0,
            created_at="2026-07-29T10:00:00Z",
        )

        self.assertEqual(outcome.outcome, OUTCOME_ACCEPTED)
        self.assertIs(outcome.accepted_result.accepted_payload, payload)
        self.assertEqual(outcome.attempt.created_at, "2026-07-29T10:00:00Z")
        self.assertEqual(outcome.attempt.quality_review["scores"], quality_review["scores"])

    def test_valid_quality_fail_returns_editorial_repair_outcome(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=_quality_state(
                _quality_review(
                    passed=False,
                    total_score=34,
                    failed_criteria=["human_voice"],
                )
            ),
            attempt_index=0,
            repair_plan={"repair_type": "editorial"},
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertTrue(outcome.repair_required)
        self.assertEqual(outcome.decision.repair_type, "editorial")
        self.assertIsNone(outcome.accepted_result)
        self.assertIsNone(outcome.terminal_result)
        self.assertEqual(outcome.repair_plan, {"repair_type": "editorial"})

    def test_grounding_failure_blocks_acceptance_even_when_quality_would_pass(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            semantic_grounding=_semantic_state(passed=False),
            quality_evaluation=_quality_state(_quality_review(passed=True)),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertEqual(outcome.decision.action, "repair_editorial")
        self.assertEqual(outcome.decision.repair_type, "semantic_grounding")
        self.assertIsNone(outcome.accepted_result)
        self.assertIsNone(outcome.attempt.quality_review)
        self.assertEqual(outcome.repair_plan["failed_claim_ids"], ["c1"])

    def test_grounding_technical_failure_returns_not_ready_without_quality_review(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            semantic_grounding=FinalPostSemanticGroundingState(
                status=GROUNDING_STATUS_NOT_READY,
                grounding_review=None,
                error_code="parse_failed",
                error_message="malformed grounding JSON",
            ),
            quality_evaluation=_quality_state(_quality_review(passed=True)),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertFalse(outcome.repair_required)
        self.assertIsNone(outcome.attempt.quality_review)
        self.assertIn("parse_failed", outcome.reason)

    def test_grounding_pass_delegates_to_editorial_quality_outcome(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            semantic_grounding=_semantic_state(passed=True),
            quality_evaluation=_quality_state(
                _quality_review(passed=False, failed_criteria=["cta"])
            ),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertEqual(outcome.decision.repair_type, "editorial")
        self.assertEqual(outcome.attempt.quality_review["failed_criteria"], ["cta"])

    def test_human_review_flags_return_human_review_outcome(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=_quality_state(
                _quality_review(
                    requires_human_review=True,
                    blocking_factuality_ambiguity=True,
                )
            ),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NEEDS_HUMAN_REVIEW)
        self.assertTrue(outcome.decision.needs_human_review)
        self.assertIsNone(outcome.accepted_result)

    def test_deterministic_failure_ignores_quality_review_and_routes_mechanical(self) -> None:
        payload = _valid_payload(post_text="This leaks a0-summary into human text.")
        candidate_output = _candidate_output(payload)
        gate_output = _gate_output(payload, schema_validation_passed=True)

        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=candidate_output,
            gate_output=gate_output,
            quality_evaluation=_quality_state(_quality_review(passed=True)),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertEqual(outcome.decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(outcome.decision.repair_type, "mechanical")
        self.assertIsNone(outcome.attempt.quality_review)

    def test_execution_failure_state_returns_distinct_not_ready_outcome(self) -> None:
        outcome = _outcome_for_quality_failure(
            status=QUALITY_EVALUATION_EXECUTION_FAILED,
            error_code="execution_failed",
            error_message="provider unavailable",
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertIn("quality evaluation execution_failed", outcome.reason)
        self.assertIn("execution_failed", outcome.reason)
        self.assertIn("provider unavailable", outcome.reason)
        self.assertFalse(outcome.repair_required)

    def test_parse_failure_state_returns_distinct_not_ready_outcome(self) -> None:
        outcome = _outcome_for_quality_failure(
            status=QUALITY_EVALUATION_PARSE_FAILED,
            error_code="malformed_json",
            error_message="invalid evaluator output",
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertIn("quality evaluation parse_failed", outcome.reason)
        self.assertIn("malformed_json", outcome.reason)

    def test_normalization_failure_state_returns_distinct_not_ready_outcome(self) -> None:
        outcome = _outcome_for_quality_failure(
            status=QUALITY_EVALUATION_NORMALIZATION_FAILED,
            error_code="normalization_failed",
            error_message="missing human_voice",
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertIn("quality evaluation normalization_failed", outcome.reason)
        self.assertIn("missing human_voice", outcome.reason)

    def test_quality_not_run_after_passing_gate_returns_not_ready(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=FinalPostQualityEvaluationState(
                status=QUALITY_EVALUATION_NOT_RUN,
                quality_review=None,
            ),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertEqual(outcome.reason, "quality evaluation not run")

    def test_ready_state_missing_review_returns_not_ready_without_editorial_repair(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=FinalPostQualityEvaluationState(
                status=QUALITY_EVALUATION_READY,
                quality_review=None,
            ),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertEqual(
            outcome.reason,
            "quality evaluation ready state missing normalized review",
        )
        self.assertFalse(outcome.repair_required)

    def test_existing_gate_output_is_consumed_and_not_rerun(self) -> None:
        payload = _valid_payload()

        with patch(
            "services.packaging.linkedin_post_attempt_adjudication."
            "build_final_post_attempt_outcome"
        ) as outcome_builder:
            expected_outcome = object()
            outcome_builder.return_value = expected_outcome
            result = build_final_post_attempt_outcome_from_gate_and_quality(
                post_brief=_post_brief(),
                candidate_output=_candidate_output(payload),
                gate_output=_passing_gate_output(payload),
                quality_evaluation=_quality_state(_quality_review()),
                attempt_index=5,
            )

        self.assertIs(result, expected_outcome)
        self.assertNotIn(
            "run_final_post_deterministic_gate",
            inspect.getsource(linkedin_post_attempt_adjudication),
        )

    def test_candidate_gate_payload_mismatch_fails_before_decision(self) -> None:
        candidate_payload = _valid_payload(post_text="Candidate payload.")
        gate_payload = _valid_payload(post_text="Different gate payload.")

        with patch.object(FinalPostDecisionController, "decide") as decide:
            with self.assertRaisesRegex(
                ValueError,
                "CandidateWriterOutput.payload must match DeterministicGateOutput.payload.",
            ):
                build_final_post_attempt_outcome_from_gate_and_quality(
                    post_brief=_post_brief(),
                    candidate_output=_candidate_output(candidate_payload),
                    gate_output=_passing_gate_output(gate_payload),
                    quality_evaluation=_quality_state(_quality_review()),
                    attempt_index=0,
                )

        decide.assert_not_called()

    def test_unsupported_quality_state_fails_before_decision(self) -> None:
        payload = _valid_payload()

        with patch.object(FinalPostDecisionController, "decide") as decide:
            with self.assertRaisesRegex(
                ValueError,
                "Unsupported FinalPostQualityEvaluationState.status: invented",
            ):
                build_final_post_attempt_outcome_from_gate_and_quality(
                    post_brief=_post_brief(),
                    candidate_output=_candidate_output(payload),
                    gate_output=_passing_gate_output(payload),
                    quality_evaluation=FinalPostQualityEvaluationState(
                        status="invented",
                        quality_review=None,
                    ),
                    attempt_index=0,
                )

        decide.assert_not_called()

    def test_attempt_history_is_appended_immutably(self) -> None:
        payload = _valid_payload()
        prior_outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=_quality_state(_quality_review()),
            attempt_index=0,
        )
        history = prior_outcome.attempt_history

        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=_quality_state(_quality_review()),
            attempt_history=history,
            attempt_index=1,
        )

        self.assertEqual(len(history.attempts), 1)
        self.assertEqual(len(outcome.attempt_history.attempts), 2)
        self.assertEqual(outcome.attempt_history.attempts[0], history.attempts[0])
        self.assertEqual(outcome.attempt_history.attempts[1].attempt_index, 1)

    def test_inputs_are_not_mutated(self) -> None:
        payload = _valid_payload()
        post_brief = _post_brief()
        candidate_output = _candidate_output(payload)
        gate_output = _passing_gate_output(payload)
        quality_evaluation = _quality_state(_quality_review())
        history = FinalPostAttemptHistory(attempts=[])
        before = copy.deepcopy(
            {
                "post_brief": post_brief,
                "candidate_output": candidate_output,
                "gate_output": gate_output,
                "quality_evaluation": quality_evaluation,
                "history": history,
            }
        )

        build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=post_brief,
            candidate_output=candidate_output,
            gate_output=gate_output,
            quality_evaluation=quality_evaluation,
            attempt_history=history,
            attempt_index=0,
        )

        self.assertEqual(post_brief, before["post_brief"])
        self.assertEqual(candidate_output, before["candidate_output"])
        self.assertEqual(gate_output, before["gate_output"])
        self.assertEqual(quality_evaluation, before["quality_evaluation"])
        self.assertEqual(history, before["history"])

    def test_quality_evaluation_state_to_dict_defensively_copies_metadata(self) -> None:
        state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_READY,
            quality_review={"scores": {"hook": 4}},
            metadata={"raw": {"id": "resp"}},
        )

        serialized = state.to_dict()
        serialized["quality_review"]["scores"]["hook"] = 1
        serialized["metadata"]["raw"]["id"] = "changed"

        self.assertEqual(state.quality_review, {"scores": {"hook": 4}})
        self.assertEqual(state.metadata, {"raw": {"id": "resp"}})

    def test_result_to_dict_is_json_serializable(self) -> None:
        payload = _valid_payload()
        outcome = build_final_post_attempt_outcome_from_gate_and_quality(
            post_brief=_post_brief(),
            candidate_output=_candidate_output(payload),
            gate_output=_passing_gate_output(payload),
            quality_evaluation=_quality_state(_quality_review()),
            attempt_index=0,
        )

        serialized = json.dumps(outcome.to_dict(), sort_keys=True)

        self.assertIn("attempt_history", serialized)
        self.assertIn("Final post text.", serialized)

    def test_adjudication_module_has_no_provider_prompt_runtime_or_repair_dependencies(self) -> None:
        source = inspect.getsource(linkedin_post_attempt_adjudication)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("apps.ai.client", source)
        self.assertNotIn("django.conf", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("linkedin_post_prompt_registry", source)
        self.assertNotIn("linkedin_post_prompt_renderers", source)
        self.assertNotIn("linkedin_post_quality_evaluator_execution", source)
        self.assertNotIn("linkedin_post_quality_evaluator_parser", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)
        self.assertNotIn("raw_article", source.lower())
        self.assertNotIn("articles", source.lower())


def _outcome_for_quality_failure(
    *,
    status: str,
    error_code: str,
    error_message: str,
):
    payload = _valid_payload()
    return build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=_post_brief(),
        candidate_output=_candidate_output(payload),
        gate_output=_passing_gate_output(payload),
        quality_evaluation=FinalPostQualityEvaluationState(
            status=status,
            quality_review=None,
            error_code=error_code,
            error_message=error_message,
        ),
        attempt_index=0,
    )


def _quality_state(quality_review: dict) -> FinalPostQualityEvaluationState:
    return FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_READY,
        quality_review=quality_review,
    )


def _semantic_state(*, passed: bool) -> FinalPostSemanticGroundingState:
    review = normalize_semantic_grounding_review_result(
        {
            "pass": passed,
            "claims": [
                {
                    "claim_id": "c1",
                    "field_name": "post_text",
                    "value_index": None,
                    "claim_text": "Final post text.",
                    "claim_type": "author_interpretation",
                    "support_status": "supported" if passed else "unsupported",
                    "severity": "info" if passed else "major",
                    "supported_evidence_ids": ["a0-summary"],
                    "required_qualifications": [],
                    "missing_qualifications": [],
                    "rationale": "Grounding rationale.",
                    "repair_hint": "" if passed else "Remove unsupported claim.",
                }
            ],
            "failed_claim_ids": [] if passed else ["c1"],
            "automatic_fail_reason": "" if passed else "unsupported claim",
            "requires_human_review": False,
            "human_review_reason": "",
            "repairable": True,
            "repair_instructions": [] if passed else ["Remove unsupported claim."],
        },
        selected_evidence_ids=["a0-summary"],
    )
    return FinalPostSemanticGroundingState(
        status=GROUNDING_STATUS_PASS if passed else GROUNDING_STATUS_FAIL,
        grounding_review=review,
    )


def _candidate_output(payload: dict) -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=payload,
        raw_output=None,
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        token_usage={"total_tokens": 12},
        cost_metadata={"estimated_usd": "0.01"},
    )


def _passing_gate_output(payload: dict) -> DeterministicGateOutput:
    return _gate_output(payload, schema_validation_passed=True)


def _gate_output(
    payload: dict,
    *,
    schema_validation_passed: bool,
    validation_error: str = "",
) -> DeterministicGateOutput:
    return DeterministicGateOutput(
        payload=payload,
        validation_passed=schema_validation_passed,
        validation_error=validation_error,
        diagnostics=diagnose_final_post_payload(
            payload,
            selected_evidence_ids=["a0-summary"],
            schema_validation_passed=schema_validation_passed,
            schema_validation_error=validation_error,
        ),
        selected_evidence_ids=("a0-summary",),
    )


def _post_brief() -> dict:
    return {
        "evidence_to_use": [
            {
                "evidence_id": "a0-summary",
                "evidence_text": "Remote work policy needs clearer expectations.",
                "role_in_post": "proof",
            }
        ]
    }


def _valid_payload(**overrides) -> dict:
    payload = {
        "post_text": "Final post text.",
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


def _quality_review(
    *,
    passed: bool = True,
    total_score: int = 37,
    failed_criteria: list[str] | None = None,
    requires_human_review: bool | None = None,
    blocking_factuality_ambiguity: bool | None = None,
) -> dict:
    review = {
        "scores": {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 5,
            "practical_value": 4,
            "cta": 4,
        },
        "total_score": total_score,
        "pass": passed,
        "failed_criteria": failed_criteria or [],
        "automatic_fail_reason": "",
        "notes": ["Ready."],
    }
    if requires_human_review is not None:
        review["requires_human_review"] = requires_human_review
    if blocking_factuality_ambiguity is not None:
        review["blocking_factuality_ambiguity"] = blocking_factuality_ambiguity
    return review
