from __future__ import annotations

import copy
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_attempt_outcome
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NEEDS_HUMAN_REVIEW,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
    OUTCOME_TRY_ALTERNATIVE_MODEL,
    FinalPostAttemptOutcome,
    build_final_post_attempt_outcome,
)
from services.packaging.linkedin_post_deterministic_orchestration import (
    FinalPostDecisionReadyResult,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    ACTION_REPAIR_MECHANICAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    STATUS_ACCEPTED,
    STATUS_NEEDS_HUMAN_REVIEW,
    STATUS_NOT_READY,
    STATUS_TRY_ALTERNATIVE_MODEL,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)


class LinkedInPostAttemptOutcomeTests(SimpleTestCase):
    def test_accept_decision_records_attempt_and_returns_accepted_outcome(self) -> None:
        decision_ready = _decision_ready_result(decision=_decision(ACTION_ACCEPT))

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=decision_ready,
            attempt_index=0,
            created_at="2026-07-28T10:00:00Z",
        )

        self.assertIsInstance(outcome, FinalPostAttemptOutcome)
        self.assertEqual(outcome.outcome, OUTCOME_ACCEPTED)
        self.assertEqual(outcome.accepted_result.status, STATUS_ACCEPTED)
        self.assertEqual(len(outcome.attempt_history.attempts), 1)

    def test_accept_uses_original_candidate_payload_as_accepted_payload(self) -> None:
        payload = _payload(post_text="Accepted final post.")
        decision_ready = _decision_ready_result(
            candidate_output=_candidate_output(payload=payload),
            gate_output=_gate_output(payload=payload),
            decision=_decision(ACTION_ACCEPT),
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=decision_ready,
            attempt_index=0,
        )

        self.assertIs(outcome.accepted_result.accepted_payload, payload)
        self.assertEqual(outcome.accepted_result.accepted_payload["post_text"], "Accepted final post.")

    def test_mechanical_repair_records_attempt_and_returns_repair_required_outcome(self) -> None:
        repair_plan = _repair_plan(repair_type="mechanical")
        decision = _decision(ACTION_REPAIR_MECHANICAL, repair_type="mechanical")

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision),
            attempt_index=0,
            repair_plan=repair_plan,
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertTrue(outcome.repair_required)
        self.assertIsNone(outcome.accepted_result)
        self.assertIsNone(outcome.terminal_result)
        self.assertEqual(outcome.repair_plan, repair_plan)
        self.assertEqual(len(outcome.attempt_history.attempts), 1)

    def test_editorial_repair_records_attempt_and_returns_repair_required_outcome(self) -> None:
        repair_plan = _repair_plan(repair_type="editorial")
        decision = _decision(ACTION_REPAIR_EDITORIAL, repair_type="editorial")

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision),
            attempt_index=0,
            repair_plan=repair_plan,
        )

        self.assertEqual(outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertTrue(outcome.repair_required)
        self.assertIsNone(outcome.accepted_result)
        self.assertIsNone(outcome.terminal_result)
        self.assertEqual(outcome.decision.repair_type, "editorial")
        self.assertEqual(outcome.attempt.repair_plan, repair_plan)

    def test_repair_reasons_are_preserved_from_decision(self) -> None:
        decision = _decision(
            ACTION_REPAIR_MECHANICAL,
            reason="schema_validation_failed, evidence_ids_in_human_text",
            repair_type="mechanical",
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision),
            attempt_index=0,
            repair_plan=_repair_plan(repair_type="mechanical"),
        )

        self.assertEqual(outcome.reason, "schema_validation_failed, evidence_ids_in_human_text")
        self.assertEqual(outcome.attempt.decision.reason, outcome.reason)

    def test_not_ready_records_attempt_and_returns_terminal_non_accepted_outcome(self) -> None:
        decision = _decision(ACTION_NOT_READY, reason="missing quality review")

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision, quality_review=None),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NOT_READY)
        self.assertEqual(outcome.terminal_result.status, STATUS_NOT_READY)
        self.assertIsNone(outcome.terminal_result.accepted_payload)
        self.assertIsNone(outcome.accepted_result)
        self.assertFalse(outcome.repair_required)

    def test_human_review_decision_records_attempt_and_preserves_outcome(self) -> None:
        decision = _decision(
            ACTION_NEEDS_HUMAN_REVIEW,
            reason="quality review flagged a human-review risk",
            needs_human_review=True,
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_NEEDS_HUMAN_REVIEW)
        self.assertEqual(outcome.terminal_result.status, STATUS_NEEDS_HUMAN_REVIEW)
        self.assertTrue(outcome.decision.needs_human_review)
        self.assertIsNone(outcome.accepted_result)

    def test_alternative_model_decision_records_attempt_and_preserves_target_metadata(self) -> None:
        decision = _decision(
            ACTION_TRY_ALTERNATIVE_MODEL,
            reason="editorial repair attempts exhausted and alternative model is available",
            target_model_provider="gemini",
            target_model_name="gemini-review-model",
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=decision),
            attempt_index=0,
        )

        self.assertEqual(outcome.outcome, OUTCOME_TRY_ALTERNATIVE_MODEL)
        self.assertEqual(outcome.terminal_result.status, STATUS_TRY_ALTERNATIVE_MODEL)
        self.assertEqual(outcome.decision.target_model_provider, "gemini")
        self.assertEqual(outcome.decision.target_model_name, "gemini-review-model")

    def test_existing_history_is_appended_immutably(self) -> None:
        prior_attempt = _attempt(attempt_index=0)
        history = FinalPostAttemptHistory(attempts=[prior_attempt])
        decision_ready = _decision_ready_result(
            attempt_history=history,
            decision=_decision(ACTION_ACCEPT),
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=decision_ready,
            attempt_index=1,
        )

        self.assertEqual(len(history.attempts), 1)
        self.assertEqual(history.attempts[0], prior_attempt)
        self.assertEqual(len(outcome.attempt_history.attempts), 2)
        self.assertEqual(outcome.attempt_history.attempts[0], prior_attempt)
        self.assertEqual(outcome.attempt_history.attempts[1].attempt_index, 1)

    def test_empty_history_produces_one_attempt_history(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=_decision(ACTION_ACCEPT)),
            attempt_index=0,
        )

        self.assertEqual(len(outcome.attempt_history.attempts), 1)
        self.assertEqual(outcome.attempt_history.latest_attempt(), outcome.attempt)

    def test_decision_ready_result_is_not_mutated(self) -> None:
        decision_ready = _decision_ready_result(decision=_decision(ACTION_ACCEPT))
        before = copy.deepcopy(decision_ready)

        build_final_post_attempt_outcome(
            decision_ready_result=decision_ready,
            attempt_index=0,
        )

        self.assertEqual(decision_ready, before)

    def test_candidate_metadata_is_preserved_in_attempt(self) -> None:
        candidate_output = _candidate_output(
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            token_usage={"input_tokens": 100, "output_tokens": 50},
            cost_metadata={"estimated_usd": "0.01"},
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(candidate_output=candidate_output),
            attempt_index=0,
        )

        self.assertEqual(outcome.attempt.provider, "openai")
        self.assertEqual(outcome.attempt.model, "gpt-4.1-2025-04-14")
        self.assertEqual(outcome.attempt.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(outcome.attempt.prompt_version, "1.0")
        self.assertEqual(outcome.attempt.token_usage, {"input_tokens": 100, "output_tokens": 50})
        self.assertEqual(outcome.attempt.cost_metadata, {"estimated_usd": "0.01"})

    def test_none_candidate_metadata_is_preserved_without_defaults(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(candidate_output=_candidate_output()),
            attempt_index=0,
        )

        self.assertIsNone(outcome.attempt.provider)
        self.assertIsNone(outcome.attempt.model)
        self.assertIsNone(outcome.attempt.prompt_name)
        self.assertIsNone(outcome.attempt.prompt_version)
        self.assertIsNone(outcome.attempt.token_usage)
        self.assertIsNone(outcome.attempt.cost_metadata)

    def test_quality_review_is_preserved(self) -> None:
        quality_review = _quality_review(total_score=35, passed=False)

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(quality_review=quality_review),
            attempt_index=0,
        )

        self.assertIs(outcome.attempt.quality_review, quality_review)

    def test_gate_output_facts_are_preserved_in_attempt(self) -> None:
        gate_output = _gate_output(
            payload=_payload(post_text="Failed validation payload."),
            validation_passed=False,
            validation_error="missing post_text",
        )

        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(
                gate_output=gate_output,
                decision=_decision(ACTION_REPAIR_MECHANICAL, repair_type="mechanical"),
            ),
            attempt_index=0,
        )

        self.assertIs(outcome.attempt.payload, gate_output.payload)
        self.assertFalse(outcome.attempt.validation_passed)
        self.assertEqual(outcome.attempt.validation_error, "missing post_text")
        self.assertIs(outcome.attempt.diagnostics, gate_output.diagnostics)

    def test_explicit_created_at_and_attempt_index_are_preserved(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(),
            attempt_index=7,
            created_at="2026-07-28T11:00:00Z",
        )

        self.assertEqual(outcome.attempt.attempt_index, 7)
        self.assertEqual(outcome.attempt.created_at, "2026-07-28T11:00:00Z")

    def test_unknown_action_raises_before_history_recording(self) -> None:
        history = FinalPostAttemptHistory(attempts=[])
        decision_ready = _decision_ready_result(
            attempt_history=history,
            decision=_decision("invented_action"),
        )

        with self.assertRaisesRegex(
            ValueError,
            "Unsupported FinalPostDecision.action: invented_action",
        ):
            build_final_post_attempt_outcome(
                decision_ready_result=decision_ready,
                attempt_index=0,
            )

        self.assertEqual(history.attempts, [])

    def test_result_is_json_serializable(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=_decision(ACTION_ACCEPT)),
            attempt_index=0,
        )

        serialized = json.dumps(outcome.to_dict(), sort_keys=True)

        self.assertIn("accepted_result", serialized)
        self.assertIn("attempt_history", serialized)
        self.assertIn("Final post text.", serialized)

    def test_attempt_record_payload_shape_matches_final_post_attempt_contract(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(),
            attempt_index=0,
        )

        self.assertEqual(
            set(outcome.attempt.to_dict()),
            {
                "attempt_index",
                "payload",
                "validation_passed",
                "validation_error",
                "diagnostics",
                "quality_review",
                "repair_plan",
                "decision",
                "provider",
                "model",
                "prompt_name",
                "prompt_version",
                "token_usage",
                "cost_metadata",
                "created_at",
                "parent_attempt_index",
            },
        )

    def test_accepted_outcome_preserves_post_text_for_later_content_package_compatibility(self) -> None:
        outcome = build_final_post_attempt_outcome(
            decision_ready_result=_decision_ready_result(decision=_decision(ACTION_ACCEPT)),
            attempt_index=0,
        )

        self.assertEqual(
            outcome.accepted_result.accepted_payload["post_text"],
            "Final post text.",
        )

    def test_outcome_module_does_not_call_gate_decision_controller_quality_or_repair(self) -> None:
        source = inspect.getsource(linkedin_post_attempt_outcome)

        self.assertNotIn("run_final_post_deterministic_gate", source)
        self.assertNotIn("FinalPostDecisionController", source)
        self.assertNotIn("QualityEvaluator", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)

    def test_outcome_module_does_not_import_runtime_persistence_prompt_or_provider_code(self) -> None:
        source = inspect.getsource(linkedin_post_attempt_outcome)
        lowered = source.lower()

        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("ContentPackage", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("apps.packaging.models", source)
        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("prompt_registry", source)
        self.assertNotIn("raw_article", lowered)
        self.assertNotIn("articles", lowered)


def _decision_ready_result(
    *,
    candidate_output: CandidateWriterOutput | None = None,
    gate_output: DeterministicGateOutput | None = None,
    decision: FinalPostDecision | None = None,
    quality_review: dict | None = None,
    attempt_history: FinalPostAttemptHistory | None = None,
) -> FinalPostDecisionReadyResult:
    candidate = candidate_output or _candidate_output(payload=_payload())
    gate = gate_output or _gate_output(payload=candidate.payload)
    return FinalPostDecisionReadyResult(
        post_brief={"evidence_to_use": [{"evidence_id": "a0-summary"}]},
        candidate_output=candidate,
        gate_output=gate,
        decision=decision or _decision(ACTION_ACCEPT),
        quality_review=quality_review if quality_review is not None else _quality_review(),
        attempt_history=attempt_history or FinalPostAttemptHistory(attempts=[]),
    )


def _candidate_output(
    *,
    payload: dict | None = None,
    provider: str | None = None,
    model: str | None = None,
    prompt_name: str | None = None,
    prompt_version: str | None = None,
    token_usage: dict | None = None,
    cost_metadata: dict | None = None,
) -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=payload or _payload(),
        raw_output=None,
        provider=provider,
        model=model,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        token_usage=token_usage,
        cost_metadata=cost_metadata,
    )


def _gate_output(
    *,
    payload: dict | None = None,
    validation_passed: bool = True,
    validation_error: str = "",
) -> DeterministicGateOutput:
    payload_value = payload or _payload()
    return DeterministicGateOutput(
        payload=payload_value,
        validation_passed=validation_passed,
        validation_error=validation_error,
        diagnostics=diagnose_final_post_payload(
            payload_value,
            selected_evidence_ids=["a0-summary"],
            schema_validation_passed=validation_passed,
            schema_validation_error=validation_error,
        ),
        selected_evidence_ids=("a0-summary",),
    )


def _attempt(*, attempt_index: int = 0) -> FinalPostAttempt:
    return FinalPostAttempt(
        attempt_index=attempt_index,
        payload=_payload(),
        validation_passed=True,
        validation_error="",
        diagnostics=_gate_output().diagnostics,
        quality_review=_quality_review(),
        repair_plan=None,
        decision=_decision(ACTION_ACCEPT),
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        token_usage=None,
        cost_metadata=None,
        created_at="2026-07-28T10:00:00Z",
        parent_attempt_index=None,
    )


def _decision(
    action: str,
    *,
    reason: str = "decision reason",
    repair_type: str | None = None,
    target_model_provider: str | None = None,
    target_model_name: str | None = None,
    needs_human_review: bool = False,
) -> FinalPostDecision:
    return FinalPostDecision(
        action=action,
        reason=reason,
        repair_type=repair_type,
        target_model_provider=target_model_provider,
        target_model_name=target_model_name,
        needs_human_review=needs_human_review,
    )


def _quality_review(*, total_score: int = 37, passed: bool = True) -> dict:
    return {
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
        "total_score": total_score,
        "pass": passed,
        "failed_criteria": [],
        "automatic_fail_reason": "",
    }


def _repair_plan(*, repair_type: str) -> dict:
    return {
        "failed_criterion": "mechanical" if repair_type == "mechanical" else "human_voice",
        "repair_scope": "human-facing text",
        "repair_instruction": "Repair the specific failed criterion.",
        "preserve": ["selected evidence", "controlling angle"],
        "avoid": ["new facts", "unsupported claims"],
        "repair_type": repair_type,
    }


def _payload(**overrides) -> dict:
    payload = {
        "post_text": "Final post text.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
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
