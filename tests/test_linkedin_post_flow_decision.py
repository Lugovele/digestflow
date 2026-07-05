from __future__ import annotations

import inspect

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_decision
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NEEDS_HUMAN_REVIEW,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    ACTION_REPAIR_MECHANICAL,
    ACTION_TRY_ALTERNATIVE_MODEL,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
)
from services.packaging.linkedin_post_flow_decision import (
    FinalPostDecisionController,
    FinalPostDecisionPolicy,
)


class LinkedInPostFlowDecisionTests(SimpleTestCase):
    def test_accept_when_deterministic_and_quality_gates_pass(self) -> None:
        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=_passing_quality_review(),
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_ACCEPT)
        self.assertFalse(decision.needs_human_review)

    def test_repair_mechanical_when_validation_fails_and_attempts_remain(self) -> None:
        decision = _controller().decide(
            validation_passed=False,
            validation_error="missing post_text",
            diagnostics=_passing_diagnostics(),
            quality_review=None,
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(decision.repair_type, "mechanical")

    def test_repair_mechanical_when_diagnostics_fail_and_attempts_remain(self) -> None:
        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_failing_diagnostics(),
            quality_review=None,
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertEqual(decision.repair_type, "mechanical")

    def test_missing_quality_review_returns_not_ready_after_deterministic_pass(self) -> None:
        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=None,
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_NOT_READY)
        self.assertEqual(decision.reason, "missing quality review")

    def test_mechanical_failure_takes_precedence_over_human_review_flags(self) -> None:
        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_failing_diagnostics(),
            quality_review={
                **_passing_quality_review(),
                "blocking_factuality_ambiguity": True,
            },
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_REPAIR_MECHANICAL)
        self.assertFalse(decision.needs_human_review)

    def test_not_ready_when_mechanical_failures_persist_after_max_repairs(self) -> None:
        history = _history(
            _attempt(
                decision=FinalPostDecision(
                    action=ACTION_REPAIR_MECHANICAL,
                    reason="previous mechanical repair",
                    repair_type="mechanical",
                    target_model_provider=None,
                    target_model_name=None,
                    needs_human_review=False,
                )
            )
        )

        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_failing_diagnostics(),
            quality_review=None,
            attempt_history=history,
        )

        self.assertEqual(decision.action, ACTION_NOT_READY)

    def test_repair_editorial_when_quality_score_below_threshold(self) -> None:
        quality_review = _passing_quality_review(total_score=35, passed=False)

        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=quality_review,
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_REPAIR_EDITORIAL)
        self.assertEqual(decision.repair_type, "editorial")

    def test_repair_editorial_when_human_voice_below_minimum(self) -> None:
        quality_review = _passing_quality_review(
            passed=False,
            scores={**_passing_scores(), "human_voice": 3},
            failed_criteria=["human_voice"],
        )

        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=quality_review,
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_REPAIR_EDITORIAL)
        self.assertIn("human_voice", decision.reason)

    def test_try_alternative_model_when_editorial_repair_exhausted_and_allowed(self) -> None:
        history = _history(
            _attempt(
                decision=FinalPostDecision(
                    action=ACTION_REPAIR_EDITORIAL,
                    reason="previous editorial repair",
                    repair_type="editorial",
                    target_model_provider=None,
                    target_model_name=None,
                    needs_human_review=False,
                )
            )
        )

        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=_passing_quality_review(total_score=35, passed=False),
            attempt_history=history,
            policy=FinalPostDecisionPolicy(allow_alternative_model=True),
            alternative_model_available=True,
            target_model_provider="gemini",
            target_model_name="gemini-review-model",
        )

        self.assertEqual(decision.action, ACTION_TRY_ALTERNATIVE_MODEL)
        self.assertEqual(decision.target_model_provider, "gemini")
        self.assertEqual(decision.target_model_name, "gemini-review-model")

    def test_not_ready_when_editorial_repair_exhausted_and_no_alternative_model(self) -> None:
        history = _history(
            _attempt(
                decision=FinalPostDecision(
                    action=ACTION_REPAIR_EDITORIAL,
                    reason="previous editorial repair",
                    repair_type="editorial",
                    target_model_provider=None,
                    target_model_name=None,
                    needs_human_review=False,
                )
            )
        )

        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review=_passing_quality_review(total_score=35, passed=False),
            attempt_history=history,
        )

        self.assertEqual(decision.action, ACTION_NOT_READY)

    def test_needs_human_review_when_blocking_factuality_or_sensitive_risk_present(self) -> None:
        decision = _controller().decide(
            validation_passed=True,
            diagnostics=_passing_diagnostics(),
            quality_review={
                **_passing_quality_review(),
                "blocking_factuality_ambiguity": True,
            },
            attempt_history=_history(),
        )

        self.assertEqual(decision.action, ACTION_NEEDS_HUMAN_REVIEW)
        self.assertTrue(decision.needs_human_review)

    def test_controller_does_not_call_api_or_execute_prompts(self) -> None:
        source = inspect.getsource(linkedin_post_flow_decision)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)


def _controller() -> FinalPostDecisionController:
    return FinalPostDecisionController()


def _history(*attempts: FinalPostAttempt) -> FinalPostAttemptHistory:
    return FinalPostAttemptHistory(attempts=list(attempts))


def _attempt(*, decision: FinalPostDecision | None = None) -> FinalPostAttempt:
    return FinalPostAttempt(
        attempt_index=0,
        payload=_payload(),
        validation_passed=True,
        validation_error="",
        diagnostics=_passing_diagnostics(),
        quality_review=_passing_quality_review(),
        repair_plan=None,
        decision=decision,
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_name="final_post_from_brief",
        prompt_version="1.0",
        token_usage=None,
        cost_metadata=None,
        created_at="2026-07-05T10:00:00Z",
        parent_attempt_index=None,
    )


def _passing_diagnostics():
    return diagnose_final_post_payload(
        _payload(),
        selected_evidence_ids=["a0-summary"],
        schema_validation_passed=True,
    )


def _failing_diagnostics():
    return diagnose_final_post_payload(
        _payload(post_text="This mentions selected evidence directly."),
        selected_evidence_ids=[],
        schema_validation_passed=True,
    )


def _passing_quality_review(**overrides) -> dict:
    quality_review = {
        "scores": _passing_scores(),
        "total_score": 37,
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
    }
    quality_review.update(overrides)
    return quality_review


def _passing_scores() -> dict:
    return {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 3,
        "author_point_of_view": 4,
        "human_voice": 4,
        "practical_value": 4,
        "cta": 4,
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
