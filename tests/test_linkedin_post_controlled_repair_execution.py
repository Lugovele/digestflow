from __future__ import annotations

import ast
import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from apps.ai.client import AI_REASONING_EFFORT_MINIMAL
from services.packaging import linkedin_post_controlled_repair_execution
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NEEDS_HUMAN_REVIEW,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
)
from services.packaging.linkedin_post_controlled_repair_contract import (
    FAILURE_REPAIR_INELIGIBLE,
    FAILURE_REPAIR_WRITER_ADAPTATION,
    FAILURE_REPAIR_WRITER_EMPTY_RESPONSE,
    FAILURE_REPAIR_WRITER_PARSE,
    FAILURE_REPAIR_WRITER_PROVIDER,
    FAILURE_REPAIR_WRITER_REQUEST,
    FAILURE_REPAIRED_DETERMINISTIC_GATE,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_NORMALIZATION,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_EMPTY_RESPONSE,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_PARSE,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_PROVIDER,
    FAILURE_REPAIRED_QUALITY_REVIEW_NORMALIZATION,
    REPAIR_ELIGIBLE,
    REPAIR_INELIGIBLE,
    FinalPostControlledRepairRequest,
)
from services.packaging.linkedin_post_controlled_repair_execution import (
    execute_final_post_controlled_repair_attempt,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FinalPostAttemptRequest,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_flow_decision import FinalPostDecisionPolicy
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)
from services.packaging.linkedin_post_model_role_policy import OPENAI_FINAL_POST_MODEL
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
    POST_TEXT_TOO_LONG,
)


class FinalPostControlledRepairExecutionTests(SimpleTestCase):
    def test_accepted_initial_attempt_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=True)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(result.repair_eligibility.status, REPAIR_INELIGIBLE)
        self.assertFalse(result.repair_executed)
        self.assertEqual(repair_client.call_count, 0)
        self.assertEqual(result.terminal_outcome, OUTCOME_ACCEPTED)
        self.assertEqual(
            result.accepted_payload["post_text"],
            "Initial candidate text from fake provider.",
        )
        self.assertIsNone(result.failure_code)

    def test_initial_technical_failure_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=FailingFakeClient(RuntimeError("secret")),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertFalse(result.repair_executed)
        self.assertEqual(repair_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_REPAIR_INELIGIBLE)
        self.assertIn("initial attempt failed", result.failure_message)

    def test_repair_role_config_failure_preflights_before_initial_candidate(
        self,
    ) -> None:
        candidate_client = QueuedFakeClient(_provider_response(_candidate_json()))
        semantic_client = _passing_semantic_client()
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=False)))
        )
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(repair_provider="gemini", repair_model="unsupported-model"),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 0)
        self.assertEqual(semantic_client.call_count, 0)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(repair_client.call_count, 0)
        self.assertFalse(result.repair_executed)
        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_REQUEST)
        self.assertEqual(result.failure_stage, "repair_writer_request")
        self.assertEqual(result.initial_attempt_result.candidate_writer_invocation_count, 0)
        self.assertEqual(result.repair_invocation_count, 0)

    def test_controlled_repair_preflight_uses_locked_semantic_grounding_defaults(
        self,
    ) -> None:
        selections = linkedin_post_controlled_repair_execution._controlled_repair_role_selections(
            _controlled_request(),
            candidate_writer_client=None,
            semantic_grounding_client=None,
            quality_evaluator_client=None,
            repair_writer_client=None,
        )
        semantic_selection = selections[1]

        self.assertEqual(semantic_selection.role, "semantic_grounding")
        self.assertEqual(semantic_selection.provider, "gemini")
        self.assertEqual(semantic_selection.model, "gemini-3.6-flash")

    def test_initial_deterministic_gate_failure_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(
                    json.dumps(_candidate_payload(post_text="This leaks ev-1."))
                )
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertFalse(result.repair_executed)
        self.assertEqual(repair_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_REPAIR_INELIGIBLE)
        self.assertIn("initial attempt failed", result.failure_message)

    def test_human_review_quality_outcome_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(
                    json.dumps(
                        _quality_review_payload(
                            passed=True,
                            blocking_factuality_ambiguity=True,
                        )
                    )
                )
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertFalse(result.repair_executed)
        self.assertEqual(repair_client.call_count, 0)
        self.assertEqual(result.terminal_outcome, OUTCOME_NEEDS_HUMAN_REVIEW)
        self.assertEqual(result.failure_code, FAILURE_REPAIR_INELIGIBLE)

    def test_repair_disabled_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(repair_enabled=False),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertFalse(result.repair_executed)
        self.assertEqual(result.repair_eligibility.reason, "repair disabled")
        self.assertEqual(repair_client.call_count, 0)

    def test_exhausted_attempt_budget_skips_repair(self) -> None:
        repair_client = QueuedFakeClient(_provider_response(_candidate_json()))

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(
                initial_attempt_request=_attempt_request(max_attempts=1)
            ),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertFalse(result.repair_executed)
        self.assertEqual(
            result.repair_eligibility.reason,
            "initial request does not allow repair attempts",
        )
        self.assertEqual(repair_client.call_count, 0)

    def test_eligible_repair_executes_once_and_accepts_repaired_payload(self) -> None:
        candidate_client = QueuedFakeClient(_provider_response(_candidate_json()))
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=False))),
            _provider_response(json.dumps(_quality_review_payload(passed=True))),
        )
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Repaired human post."))
        )
        semantic_client = _passing_semantic_client()
        post_brief = _post_brief()
        angle_decision = _angle_decision()
        request = _controlled_request(execution_metadata={"audit": "debug-only"})
        request_before = copy.deepcopy(request)
        post_brief_before = copy.deepcopy(post_brief)
        angle_decision_before = copy.deepcopy(angle_decision)

        result = execute_final_post_controlled_repair_attempt(
            request,
            post_brief=post_brief,
            angle_decision=angle_decision,
            selected_evidence_ids=("ev-1", "ev-2"),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
            repair_writer_client=repair_client,
        )
        serialized = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

        self.assertEqual(result.repair_eligibility.status, REPAIR_ELIGIBLE)
        self.assertTrue(result.repair_executed)
        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(repair_client.call_count, 1)
        self.assertEqual(semantic_client.call_count, 2)
        self.assertEqual(semantic_client.max_output_tokens, 4800)
        self.assertTrue(semantic_client.json_mode)
        self.assertEqual(
            semantic_client.extra_kwargs.get("reasoning_effort"),
            AI_REASONING_EFFORT_MINIMAL,
        )
        self.assertEqual(evaluator_client.call_count, 2)
        self.assertEqual(evaluator_client.max_output_tokens, 2400)
        self.assertTrue(evaluator_client.json_mode)
        self.assertEqual(result.candidate_writer_invocation_count, 1)
        self.assertEqual(result.repair_invocation_count, 1)
        self.assertEqual(result.quality_evaluator_invocation_count, 2)
        self.assertEqual(result.terminal_outcome, OUTCOME_ACCEPTED)
        self.assertEqual(result.accepted_payload["post_text"], "Repaired human post.")
        self.assertEqual(
            result.initial_attempt_result.candidate_writer_output.payload["post_text"],
            "Initial candidate text from fake provider.",
        )
        self.assertEqual(
            set(result.initial_attempt_result.candidate_writer_output.payload),
            {"post_text"},
        )
        self.assertEqual(
            set(result.repaired_candidate_output.payload),
            set(_candidate_payload(post_text="Repaired human post.")),
        )
        self.assertIn("Repaired human post.", serialized)
        self.assertEqual(request, request_before)
        self.assertEqual(post_brief, post_brief_before)
        self.assertEqual(angle_decision, angle_decision_before)

    def test_anthropic_repair_writer_executes_when_allowed_by_role_policy(self) -> None:
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Claude repaired post."))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(
                repair_provider="anthropic",
                repair_model="claude-sonnet-5",
            ),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response(json.dumps(_quality_review_payload(passed=True))),
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertTrue(result.repair_executed)
        self.assertEqual(repair_client.call_count, 1)
        self.assertEqual(result.repaired_candidate_output.provider, "anthropic")
        self.assertEqual(result.repaired_candidate_output.model, "claude-sonnet-5")
        self.assertEqual(result.accepted_payload["post_text"], "Claude repaired post.")

    def test_repaired_post_uses_candidate_post_gate_not_full_payload_gate(self) -> None:
        self.assertFalse(
            hasattr(
                linkedin_post_controlled_repair_execution,
                "run_final_post_deterministic_gate",
            )
        )

        with patch.object(
            linkedin_post_controlled_repair_execution,
            "run_candidate_post_deterministic_gate",
            wraps=linkedin_post_controlled_repair_execution.run_candidate_post_deterministic_gate,
        ) as candidate_gate:
            result = execute_final_post_controlled_repair_attempt(
                _controlled_request(),
                candidate_writer_client=QueuedFakeClient(
                    _provider_response(_candidate_json())
                ),
                semantic_grounding_client=_passing_semantic_client(),
                quality_evaluator_client=QueuedFakeClient(
                    _provider_response(json.dumps(_quality_review_payload(passed=False))),
                    _provider_response(json.dumps(_quality_review_payload(passed=True))),
                ),
                repair_writer_client=QueuedFakeClient(
                    _provider_response(_candidate_json(post_text="Repaired human post."))
                ),
                **_flow_kwargs(),
            )

        self.assertEqual(result.terminal_outcome, OUTCOME_ACCEPTED)
        self.assertGreaterEqual(candidate_gate.call_count, 1)
        repaired_gate_call = candidate_gate.call_args_list[-1]
        self.assertEqual(
            repaired_gate_call.args[0].payload,
            {"post_text": "Repaired human post."},
        )
        self.assertEqual(
            repaired_gate_call.kwargs["selected_evidence_ids"],
            ("ev-1", "ev-2"),
        )

    def test_repair_prompt_uses_allowlisted_findings_without_raw_metadata(self) -> None:
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Repaired text."))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(execution_metadata={"runtime": "metadata-sentinel"}),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response(json.dumps(_quality_review_payload(passed=True))),
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )
        prompt_text = repair_client.prompts[0]

        self.assertNotIn("provider-metadata", prompt_text)
        self.assertNotIn("metadata-sentinel", prompt_text)
        self.assertNotIn("raw_provider_response", prompt_text)
        self.assertIn("human_voice", prompt_text)
        self.assertIn("selected evidence only", prompt_text)
        for forbidden in (
            "FinalPostPayload",
            "hook_variants",
            "cta_variants",
            "hashtags",
            "quality_checks",
            "carousel_outline",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, prompt_text)
        self.assertEqual(
            json.loads(
                result.repair_prompt_render.variables["selected_evidence_json"]
            )[0]["evidence_id"],
            "ev-1",
        )

    def test_semantic_grounding_failure_is_eligible_for_one_controlled_repair(
        self,
    ) -> None:
        semantic_client = QueuedFakeClient(
            _provider_response(json.dumps(_semantic_review_payload(passed=False))),
            _provider_response(json.dumps(_semantic_review_payload(passed=True))),
        )
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Grounded repaired text."))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(result.repair_eligibility.status, REPAIR_ELIGIBLE)
        self.assertIn("semantic_grounding repair", result.repair_eligibility.reason)
        self.assertTrue(result.repair_executed)
        self.assertEqual(semantic_client.call_count, 2)
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertEqual(repair_client.call_count, 1)
        self.assertEqual(result.semantic_grounding_invocation_count, 2)
        self.assertEqual(result.quality_evaluator_invocation_count, 1)
        self.assertEqual(result.terminal_outcome, OUTCOME_ACCEPTED)
        self.assertEqual(
            result.repair_prompt_render.variables["repair_instruction_json"],
            json.dumps(
                {
                    "repair_type": "semantic_grounding",
                    "failed_claim_ids": ["c1"],
                    "repair_scope": "CandidatePost.post_text",
                    "repair_instruction": "Remove unsupported wording.",
                    "decision_reason": "semantic grounding failed: unsupported claim",
                    "preserve": [
                        "selected evidence only",
                        "AngleDecision.controlling_angle",
                        "source qualifications such as likely, may, projected, and risk remains",
                        "distinctive original phrasing unless it is the grounding defect",
                        "valid CandidatePost JSON",
                    ],
                    "avoid": [
                        "full-post rewrite when a local repair is enough",
                        "generic author markers or template transitions",
                        "new facts",
                        "stronger certainty than selected evidence",
                        "unsupported causal language",
                        "recovery/stability/optimism drift",
                    ],
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            ),
        )

    def test_repaired_semantic_grounding_failure_skips_second_quality_review(
        self,
    ) -> None:
        semantic_client = QueuedFakeClient(
            _provider_response(json.dumps(_semantic_review_payload(passed=False))),
            _provider_response(json.dumps(_semantic_review_payload(passed=False))),
        )
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Still ungrounded."))
            ),
            **_flow_kwargs(),
        )

        self.assertTrue(result.repair_executed)
        self.assertEqual(result.failure_code, FAILURE_REPAIRED_SEMANTIC_GROUNDING)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.semantic_grounding_invocation_count, 2)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIsNotNone(result.repaired_semantic_grounding_state)
        self.assertIsNone(result.repaired_quality_evaluation_state)
        self.assertEqual(result.terminal_outcome, OUTCOME_NOT_READY)

    def test_repaired_grounding_pass_with_human_review_flag_skips_second_quality_review(
        self,
    ) -> None:
        invalid_human_review_pass = _semantic_review_payload(passed=True)
        invalid_human_review_pass["requires_human_review"] = True
        invalid_human_review_pass["human_review_reason"] = (
            "Needs human factuality review."
        )
        semantic_client = QueuedFakeClient(
            _provider_response(json.dumps(_semantic_review_payload(passed=False))),
            _provider_response(json.dumps(invalid_human_review_pass)),
        )
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Repaired text."))
            ),
            **_flow_kwargs(),
        )

        self.assertTrue(result.repair_executed)
        self.assertEqual(
            result.failure_code,
            FAILURE_REPAIRED_SEMANTIC_GROUNDING_NORMALIZATION,
        )
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.semantic_grounding_invocation_count, 2)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIsNotNone(result.repaired_semantic_grounding_state)
        self.assertIsNone(result.repaired_quality_evaluation_state)
        self.assertEqual(result.terminal_outcome, OUTCOME_NOT_READY)

    def test_repaired_deterministic_gate_failure_skips_second_quality_review(self) -> None:
        evaluator_client = QueuedFakeClient(
            _provider_response(json.dumps(_quality_review_payload(passed=False)))
        )
        repair_client = QueuedFakeClient(
            _provider_response(
                _candidate_json(post_text="Repaired text still leaks ev-1.")
            )
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIRED_DETERMINISTIC_GATE)
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertEqual(result.repair_invocation_count, 1)
        self.assertIsNotNone(result.repaired_deterministic_gate_output)
        self.assertIsNone(result.repaired_quality_evaluation_state)

    def test_repaired_quality_fail_does_not_execute_second_repair(self) -> None:
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Still too generic."))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(repair_client.call_count, 1)
        self.assertEqual(result.repair_invocation_count, 1)
        self.assertEqual(result.terminal_outcome, OUTCOME_NOT_READY)
        self.assertIsNone(result.accepted_payload)
        self.assertEqual(
            result.repaired_attempt_outcome.outcome,
            OUTCOME_NOT_READY,
        )

    def test_repair_provider_failure_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=FailingFakeClient(RuntimeError("secret")),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_PROVIDER)
        self.assertEqual(result.failure_message, "provider invocation failed")
        self.assertEqual(result.repair_invocation_count, 1)

    def test_repair_request_failure_does_not_mark_repair_executed(self) -> None:
        repair_client = QueuedFakeClient(
            _provider_response(_candidate_json(post_text="Should not be called."))
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(repair_provider="unsupported"),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=repair_client,
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_REQUEST)
        self.assertEqual(result.failure_stage, "repair_writer_request")
        self.assertFalse(result.repair_executed)
        self.assertEqual(result.repair_invocation_count, 0)
        self.assertEqual(repair_client.call_count, 0)

    def test_repair_empty_response_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=QueuedFakeClient(_provider_response("   ")),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_EMPTY_RESPONSE)
        self.assertEqual(result.repair_invocation_count, 1)

    def test_repair_parse_failure_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=QueuedFakeClient(_provider_response("{not-json")),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_PARSE)
        self.assertEqual(result.repair_invocation_count, 1)

    def test_repair_adaptation_failure_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=QueuedFakeClient(
                _provider_response(
                    json.dumps({"post_text": "Unexpected packaging.", "hook_variants": []})
                )
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_ADAPTATION)
        self.assertEqual(result.repair_invocation_count, 1)
        self.assertEqual(
            result.repair_writer_structural_diagnostics.structural_failure_category,
            CANDIDATE_POST_OTHER_VALIDATION_FAILURE,
        )
        self.assertEqual(
            result.repair_writer_structural_diagnostics.parsed_top_level_keys,
            ("hook_variants", "post_text"),
        )

    def test_repair_adaptation_failure_records_post_text_length_diagnostics(
        self,
    ) -> None:
        overlength = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False)))
            ),
            repair_writer_client=QueuedFakeClient(
                _provider_response(json.dumps({"post_text": overlength}))
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIR_WRITER_ADAPTATION)
        diagnostics = result.repair_writer_structural_diagnostics
        serialized_diagnostics = json.dumps(diagnostics.to_dict(), sort_keys=True)
        self.assertEqual(diagnostics.structural_failure_category, POST_TEXT_TOO_LONG)
        self.assertEqual(
            diagnostics.post_text_character_count,
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertFalse(diagnostics.post_text_within_candidate_max_length)
        self.assertNotIn(overlength, serialized_diagnostics)

    def test_repaired_evaluator_provider_failure_is_distinct(self) -> None:
        evaluator_client = QueuedThenFailingClient(
            _provider_response(json.dumps(_quality_review_payload(passed=False))),
            RuntimeError("secret"),
        )

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Repaired text."))
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIRED_QUALITY_EVALUATOR_PROVIDER)
        self.assertEqual(result.quality_evaluator_invocation_count, 2)
        self.assertNotIn("secret", result.failure_message)

    def test_repaired_evaluator_empty_response_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response(""),
            ),
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Repaired text."))
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(
            result.failure_code,
            FAILURE_REPAIRED_QUALITY_EVALUATOR_EMPTY_RESPONSE,
        )
        self.assertEqual(result.quality_evaluator_invocation_count, 2)

    def test_repaired_evaluator_parse_failure_is_distinct(self) -> None:
        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response("{not-json"),
            ),
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Repaired text."))
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_REPAIRED_QUALITY_EVALUATOR_PARSE)
        self.assertEqual(result.quality_evaluator_invocation_count, 2)

    def test_repaired_quality_normalization_failure_is_distinct(self) -> None:
        invalid_review = _quality_review_payload(passed=True)
        invalid_review["scores"].pop("human_voice")

        result = execute_final_post_controlled_repair_attempt(
            _controlled_request(),
            candidate_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=QueuedFakeClient(
                _provider_response(json.dumps(_quality_review_payload(passed=False))),
                _provider_response(json.dumps(invalid_review)),
            ),
            repair_writer_client=QueuedFakeClient(
                _provider_response(_candidate_json(post_text="Repaired text."))
            ),
            **_flow_kwargs(),
        )

        self.assertEqual(
            result.failure_code,
            FAILURE_REPAIRED_QUALITY_REVIEW_NORMALIZATION,
        )
        self.assertEqual(result.quality_evaluator_invocation_count, 2)

    def test_execution_module_does_not_import_forbidden_runtime_layers(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_controlled_repair_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
            "services.sources",
        }
        forbidden_symbols = {
            "ContentPackage",
            "OpenAIClient",
            "generate_content_package_for_digest",
            "GeminiClient",
            "raw_articles",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


class QueuedFakeClient:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self.responses = list(responses)
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        json_mode: bool,
        allow_json_mode_fallback: bool = True,
        **kwargs,
    ) -> SimpleNamespace:
        self.call_count += 1
        self.prompts.append(prompt)
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.allow_json_mode_fallback = allow_json_mode_fallback
        self.extra_kwargs = copy.deepcopy(kwargs)
        if not self.responses:
            raise AssertionError("No queued fake response available.")
        return self.responses.pop(0)


class FailingFakeClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.call_count = 0

    def generate_text(self, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        raise self.exc


class QueuedThenFailingClient:
    def __init__(self, first_response: SimpleNamespace, exc: Exception) -> None:
        self.first_response = first_response
        self.exc = exc
        self.call_count = 0

    def generate_text(self, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        if self.call_count == 1:
            return self.first_response
        raise self.exc


def _controlled_request(
    *,
    initial_attempt_request: FinalPostAttemptRequest | None = None,
    repair_enabled: bool = True,
    repair_provider: str | None = "openai",
    repair_model: str = OPENAI_FINAL_POST_MODEL,
    max_controlled_attempts: int = 2,
    execution_metadata: dict | None = None,
) -> FinalPostControlledRepairRequest:
    return FinalPostControlledRepairRequest(
        initial_attempt_request=(
            _attempt_request() if initial_attempt_request is None else initial_attempt_request
        ),
        repair_prompt_text="Repair Writer prompt text.",
        repair_provider=repair_provider,
        repair_model=repair_model,
        repair_max_output_tokens=1200,
        repair_enabled=repair_enabled,
        max_controlled_attempts=max_controlled_attempts,
        execution_metadata=copy.deepcopy(execution_metadata),
    )


def _attempt_request(
    *,
    candidate_writer_provider: str | None = "openai",
    max_attempts: int = 2,
    policy: object | None = None,
) -> FinalPostAttemptRequest:
    return FinalPostAttemptRequest(
        candidate_writer_render=CandidateWriterPromptRender(
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
            variables={
                "post_brief_json": "{}",
                "angle_decision_json": "{}",
                "selected_evidence_json": "[]",
                "candidate_writer_input_json": "{}",
            },
            input_text="## CANDIDATE_WRITER_INPUT_JSON\n{}",
        ),
        candidate_writer_prompt_text="Candidate Writer prompt text.",
        quality_rubric=get_quality_evaluator_rubric_payload().to_dict(),
        quality_evaluator_prompt_text="Quality Evaluator prompt text.",
        attempt_index=0,
        max_attempts=max_attempts,
        candidate_writer_provider=candidate_writer_provider,
        candidate_writer_model=OPENAI_FINAL_POST_MODEL,
        candidate_writer_max_output_tokens=1200,
        semantic_grounding_prompt_text="Semantic grounding prompt text.",
        semantic_grounding_provider=None,
        semantic_grounding_model=None,
        semantic_grounding_max_output_tokens=None,
        quality_evaluator_provider="openai",
        quality_evaluator_model=OPENAI_FINAL_POST_MODEL,
        quality_evaluator_max_output_tokens=None,
        policy=policy or FinalPostDecisionPolicy(max_total_attempts=2),
    )


def _flow_kwargs() -> dict:
    return {
        "post_brief": _post_brief(),
        "angle_decision": _angle_decision(),
        "selected_evidence_ids": ("ev-1", "ev-2"),
    }


def _post_brief() -> dict:
    return {
        "core_point": "Clear remote work policies need human operating habits.",
        "evidence_to_use": [
            {
                "evidence_id": "ev-1",
                "evidence_text": "Remote work policies need clear expectations.",
                "role_in_post": "proof",
            },
            {
                "evidence_id": "ev-2",
                "evidence_text": "Inclusive work cultures should account for isolation.",
                "role_in_post": "practical_point",
            },
        ],
    }


def _angle_decision() -> dict:
    return {
        "controlling_angle": "Remote policies need clarity and inclusion.",
        "author_position": "Leaders should connect policy clarity with team trust.",
        "authorial_voice_directive": _authorial_voice_directive(),
    }


def _authorial_voice_directive() -> dict:
    return {
        "authorial_observation": (
            "The author notices that remote policy clarity and isolation support "
            "must be evaluated together."
        ),
        "rejected_reading": (
            "Reject treating documented remote policies as proof that inclusion "
            "and isolation risks are solved."
        ),
        "why_distinction_matters": (
            "The distinction matters because sustainable remote work needs both "
            "operating clarity and human support."
        ),
        "personal_presence_requirement": "explicit_author_owned_statement_required",
        "first_person_policy": "allowed_not_required",
        "forbidden_author_claims": [
            "personal experience",
            "professional authority",
            "direct market exposure",
            "client or customer stories",
            "invented emotional reaction",
            "biographical claims",
        ],
    }


def _provider_response(raw_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=raw_text,
        raw={"id": "resp-1", "metadata_sentinel": "provider-metadata"},
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def _passing_semantic_client() -> QueuedFakeClient:
    return QueuedFakeClient(
        _provider_response(json.dumps(_semantic_review_payload(passed=True))),
        _provider_response(json.dumps(_semantic_review_payload(passed=True))),
    )


def _semantic_review_payload(*, passed: bool = True) -> dict:
    return {
        "pass": passed,
        "claims": [
            {
                "claim_id": "c1",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "Initial candidate text from fake provider.",
                "claim_type": "author_interpretation",
                "support_status": "supported" if passed else "unsupported",
                "severity": "info" if passed else "major",
                "supported_evidence_ids": ["ev-1"],
                "required_qualifications": [],
                "missing_qualifications": [],
                "rationale": "Grounded in selected evidence.",
                "repair_hint": "" if passed else "Remove unsupported wording.",
            }
        ],
        "failed_claim_ids": [] if passed else ["c1"],
        "automatic_fail_reason": "" if passed else "unsupported claim",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": not passed,
        "repair_instructions": (
            []
            if passed
            else [
                {
                    "claim_id": "c1",
                    "instruction": "Remove unsupported wording.",
                }
            ]
        ),
    }


def _candidate_json(
    *,
    post_text: str = "Initial candidate text from fake provider.",
) -> str:
    return json.dumps(_candidate_payload(post_text=post_text))


def _candidate_payload(
    *,
    post_text: str = "Initial candidate text from fake provider.",
) -> dict:
    return {"post_text": post_text}


def _quality_review_payload(
    *,
    passed: bool = True,
    total_score: int | None = None,
    failed_criteria: list[str] | None = None,
    blocking_factuality_ambiguity: bool = False,
) -> dict:
    scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 5 if passed else 3,
        "practical_value": 4,
        "cta": 4,
    }
    return {
        "scores": scores,
        "total_score": sum(scores.values()) if total_score is None else total_score,
        "pass": passed,
        "failed_criteria": failed_criteria or ([] if passed else ["human_voice"]),
        "automatic_fail_reason": "",
        "notes": ["Evaluator note."],
        "criterion_rationales": _criterion_rationales(scores),
        "blocking_factuality_ambiguity": blocking_factuality_ambiguity,
    }


def _criterion_rationales(scores: dict[str, int]) -> dict[str, dict[str, object]]:
    return {
        criterion: {
            "score": score,
            "max_score": 5,
            "rationale": f"{criterion} rationale tied to the candidate text.",
            "post_text_evidence": f"{criterion} evidence from post_text.",
            "failure_reason": "",
        }
        for criterion, score in scores.items()
    }


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.update(
                alias.asname or alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names)
    return symbols
