from __future__ import annotations

import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_contracts
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NOT_READY,
    FINAL_POST_DECISION_ACTIONS,
    FINAL_POST_FLOW_STATUSES,
    STATUS_ACCEPTED,
    STATUS_NOT_READY,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
    FinalPostFlowResult,
)


class LinkedInPostFlowContractsTests(SimpleTestCase):
    def test_final_post_decision_to_dict(self) -> None:
        decision = FinalPostDecision(
            action=ACTION_ACCEPT,
            reason="quality threshold passed",
            repair_type=None,
            target_model_provider=None,
            target_model_name=None,
            needs_human_review=False,
        )

        self.assertEqual(
            decision.to_dict(),
            {
                "action": "accept",
                "reason": "quality threshold passed",
                "repair_type": None,
                "target_model_provider": None,
                "target_model_name": None,
                "needs_human_review": False,
            },
        )
        self.assertIn(ACTION_ACCEPT, FINAL_POST_DECISION_ACTIONS)

    def test_final_post_attempt_to_dict_with_diagnostics(self) -> None:
        diagnostics = _diagnostics()
        decision = _decision()
        attempt = _attempt(diagnostics=diagnostics, decision=decision)

        attempt_dict = attempt.to_dict()

        self.assertEqual(attempt_dict["attempt_index"], 0)
        self.assertEqual(attempt_dict["payload"]["post_text"], "Final post text.")
        self.assertTrue(attempt_dict["validation_passed"])
        self.assertEqual(attempt_dict["diagnostics"], diagnostics.to_dict())
        self.assertEqual(attempt_dict["decision"], decision.to_dict())
        self.assertEqual(attempt_dict["provider"], "openai")
        self.assertEqual(attempt_dict["model"], "gpt-4.1-2025-04-14")

    def test_final_post_attempt_history_latest_attempt(self) -> None:
        first_attempt = _attempt(attempt_index=0)
        second_attempt = _attempt(attempt_index=1, parent_attempt_index=0)
        history = FinalPostAttemptHistory(attempts=[first_attempt]).add_attempt(second_attempt)

        self.assertEqual(history.latest_attempt(), second_attempt)
        self.assertEqual(len(history.attempts), 2)

    def test_empty_attempt_history_latest_attempt_is_none(self) -> None:
        history = FinalPostAttemptHistory(attempts=[])

        self.assertIsNone(history.latest_attempt())

    def test_final_post_attempt_history_to_dict(self) -> None:
        history = FinalPostAttemptHistory(attempts=[_attempt()])

        history_dict = history.to_dict()

        self.assertEqual(len(history_dict["attempts"]), 1)
        self.assertEqual(history_dict["attempts"][0]["attempt_index"], 0)

    def test_final_post_flow_result_to_dict(self) -> None:
        result = _flow_result(status=STATUS_ACCEPTED, accepted_payload=_payload())

        result_dict = result.to_dict()

        self.assertEqual(result_dict["status"], "accepted")
        self.assertEqual(result_dict["accepted_payload"]["post_text"], "Final post text.")
        self.assertEqual(result_dict["final_decision"]["action"], "accept")
        self.assertIn(STATUS_ACCEPTED, FINAL_POST_FLOW_STATUSES)

    def test_accepted_flow_result_can_include_accepted_payload(self) -> None:
        result = _flow_result(status=STATUS_ACCEPTED, accepted_payload=_payload())

        self.assertEqual(result.accepted_payload, _payload())
        self.assertEqual(result.final_decision.action, ACTION_ACCEPT)

    def test_not_ready_flow_result_preserves_decision_and_history(self) -> None:
        decision = FinalPostDecision(
            action=ACTION_NOT_READY,
            reason="max attempts reached",
            repair_type=None,
            target_model_provider=None,
            target_model_name=None,
            needs_human_review=False,
        )
        history = FinalPostAttemptHistory(attempts=[_attempt(decision=decision)])
        result = FinalPostFlowResult(
            status=STATUS_NOT_READY,
            accepted_payload=None,
            attempt_history=history,
            final_decision=decision,
            reason="max attempts reached",
        )

        result_dict = result.to_dict()

        self.assertIsNone(result_dict["accepted_payload"])
        self.assertEqual(result_dict["final_decision"]["action"], "not_ready")
        self.assertEqual(result_dict["attempt_history"]["attempts"][0]["decision"]["action"], "not_ready")

    def test_flow_contracts_are_json_serializable(self) -> None:
        result_dict = _flow_result(status=STATUS_ACCEPTED, accepted_payload=_payload()).to_dict()

        serialized = json.dumps(result_dict, sort_keys=True)

        self.assertIn("Final post text.", serialized)

    def test_flow_contract_module_does_not_call_api_or_execute_prompts(self) -> None:
        source = inspect.getsource(linkedin_post_flow_contracts)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)


def _flow_result(*, status: str, accepted_payload: dict | None) -> FinalPostFlowResult:
    decision = _decision()
    return FinalPostFlowResult(
        status=status,
        accepted_payload=accepted_payload,
        attempt_history=FinalPostAttemptHistory(attempts=[_attempt(decision=decision)]),
        final_decision=decision,
        reason="accepted",
    )


def _attempt(
    *,
    attempt_index: int = 0,
    diagnostics=None,
    decision=None,
    parent_attempt_index: int | None = None,
) -> FinalPostAttempt:
    return FinalPostAttempt(
        attempt_index=attempt_index,
        payload=_payload(),
        validation_passed=True,
        validation_error="",
        diagnostics=diagnostics or _diagnostics(),
        quality_review={
            "total_score": 37,
            "pass": True,
            "failed_criteria": [],
        },
        repair_plan=None,
        decision=decision or _decision(),
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_name="final_post_from_brief",
        prompt_version="1.0",
        token_usage={"input_tokens": 100, "output_tokens": 50},
        cost_metadata={"estimated_usd": "0.01"},
        created_at="2026-07-05T10:00:00Z",
        parent_attempt_index=parent_attempt_index,
    )


def _decision() -> FinalPostDecision:
    return FinalPostDecision(
        action=ACTION_ACCEPT,
        reason="quality threshold passed",
        repair_type=None,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _diagnostics():
    return diagnose_final_post_payload(
        _payload(),
        selected_evidence_ids=["a0-summary"],
        schema_validation_passed=True,
    )


def _payload() -> dict:
    return {
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
