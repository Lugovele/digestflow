from __future__ import annotations

import ast
import copy
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_final_post_attempt_execution
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
)
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
    OUTCOME_TRY_ALTERNATIVE_MODEL,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_ADAPTATION,
    FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE,
    FAILURE_CANDIDATE_WRITER_PARSE,
    FAILURE_CANDIDATE_WRITER_PROVIDER,
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_DETERMINISTIC_GATE,
    FAILURE_QUALITY_EVALUATOR_EMPTY_RESPONSE,
    FAILURE_QUALITY_EVALUATOR_PARSE,
    FAILURE_QUALITY_EVALUATOR_PROVIDER,
    FAILURE_QUALITY_EVALUATOR_REQUEST,
    FAILURE_QUALITY_REVIEW_NORMALIZATION,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_CANDIDATE_WRITER_EXECUTION,
    STAGE_CANDIDATE_WRITER_PARSE,
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_DETERMINISTIC_GATE,
    STAGE_QUALITY_EVALUATOR_EXECUTION,
    STAGE_QUALITY_EVALUATOR_PARSE,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_QUALITY_REVIEW_NORMALIZATION,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCEEDED,
    FinalPostAttemptRequest,
)
from services.packaging.linkedin_post_final_post_attempt_execution import (
    execute_final_post_standalone_attempt,
    execute_final_post_standalone_candidate_attempt,
)
from services.packaging.linkedin_post_flow_decision import (
    FinalPostDecisionPolicy,
)
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)


class FinalPostStandaloneAttemptExecutionTests(SimpleTestCase):
    def test_successful_fake_provider_response_reaches_candidate_text_and_gate(
        self,
    ) -> None:
        request = _request(execution_metadata={"trace": "metadata-sentinel"})
        request_before = copy.deepcopy(request)
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        result = execute_final_post_standalone_candidate_attempt(
            request,
            selected_evidence_ids=("ev-1", "ev-2"),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(fake_client.call_count, 1)
        self.assertEqual(result.completed_stage, STAGE_DETERMINISTIC_GATE)
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.candidate_writer_invocation_count, 1)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertEqual(
            result.candidate_writer_output.payload["post_text"],
            "Real candidate text from fake provider.",
        )
        self.assertEqual(
            result.to_dict()["candidate_writer_output"]["payload"]["post_text"],
            "Real candidate text from fake provider.",
        )
        self.assertEqual(
            result.deterministic_gate_output.payload["post_text"],
            "Real candidate text from fake provider.",
        )
        self.assertTrue(result.deterministic_gate_output.validation_passed)
        self.assertTrue(
            result.deterministic_gate_output.diagnostics.system_linkedin_ready
        )
        self.assertEqual(request, request_before)
        self.assertNotIn(
            "metadata-sentinel",
            json.dumps(result.candidate_writer_output.payload, sort_keys=True),
        )

    def test_request_configuration_failure_invokes_client_zero_times(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        result = execute_final_post_standalone_candidate_attempt(
            _request(candidate_writer_provider=""),
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(fake_client.call_count, 0)
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_REQUEST)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_REQUEST)
        self.assertIsNone(result.completed_stage)
        self.assertIsNone(result.candidate_writer_raw_response)
        self.assertIsNone(result.candidate_writer_output)
        self.assertEqual(result.candidate_writer_invocation_count, 0)

    def test_malformed_render_metadata_fails_request_before_client_call(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))
        malformed_render = SimpleNamespace(
            input_text="## CANDIDATE_WRITER_INPUT_JSON\n{}"
        )
        request = _request(candidate_writer_render=malformed_render)
        request_before = copy.deepcopy(request)

        result = execute_final_post_standalone_candidate_attempt(
            request,
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )
        result_dict = result.to_dict()
        serialized = json.dumps(result_dict, allow_nan=False, sort_keys=True)

        self.assertEqual(fake_client.call_count, 0)
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_REQUEST)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_REQUEST)
        self.assertEqual(
            result.failure_message,
            "missing candidate writer render metadata: prompt_name",
        )
        self.assertIsNone(result.completed_stage)
        self.assertIsNone(result.candidate_writer_raw_response)
        self.assertIsNone(result.parsed_candidate)
        self.assertIsNone(result.candidate_writer_output)
        self.assertEqual(result.candidate_writer_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIn("unsupported_candidate_writer_render", serialized)
        self.assertEqual(
            result_dict["request"]["candidate_writer_render"]["input_text"],
            "## CANDIDATE_WRITER_INPUT_JSON\n{}",
        )
        self.assertEqual(request, request_before)
        self.assertIs(request.candidate_writer_render, malformed_render)
        self.assertFalse(hasattr(malformed_render, "prompt_name"))
        self.assertIn(
            STAGE_QUALITY_EVALUATOR_REQUEST,
            [
                status.stage
                for status in result.stage_statuses
                if status.status == STATUS_SKIPPED
            ],
        )

    def test_provider_failure_does_not_parse_or_adapt(self) -> None:
        fake_client = FailingCandidateWriterClient(RuntimeError("secret raw output"))

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "parse_candidate_writer_raw_response",
        ) as parse_raw:
            with patch.object(
                linkedin_post_final_post_attempt_execution,
                "build_candidate_writer_output_from_parsed_response",
            ) as adapt_payload:
                result = execute_final_post_standalone_candidate_attempt(
                    _request(),
                    selected_evidence_ids=("ev-1",),
                    candidate_writer_client=fake_client,
                )

        self.assertEqual(fake_client.call_count, 1)
        parse_raw.assert_not_called()
        adapt_payload.assert_not_called()
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_EXECUTION)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_PROVIDER)
        self.assertEqual(result.failure_message, "provider invocation failed")
        self.assertNotIn("secret raw output", result.failure_message)
        self.assertIsNotNone(result.candidate_writer_raw_response)
        self.assertEqual(result.candidate_writer_invocation_count, 1)

    def test_empty_response_does_not_produce_candidate_output(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response("   "))

        result = execute_final_post_standalone_candidate_attempt(
            _request(),
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(fake_client.call_count, 1)
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_EXECUTION)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE)
        self.assertIsNone(result.parsed_candidate)
        self.assertIsNone(result.candidate_writer_output)
        self.assertEqual(result.candidate_writer_invocation_count, 1)

    def test_parse_failure_does_not_invoke_adaptation(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response("{not-json"))

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "build_candidate_writer_output_from_parsed_response",
        ) as adapt_payload:
            result = execute_final_post_standalone_candidate_attempt(
                _request(),
                selected_evidence_ids=("ev-1",),
                candidate_writer_client=fake_client,
            )

        adapt_payload.assert_not_called()
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_PARSE)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_PARSE)
        self.assertEqual(result.completed_stage, STAGE_CANDIDATE_WRITER_EXECUTION)
        self.assertIsNone(result.parsed_candidate)
        self.assertIsNone(result.candidate_writer_output)

    def test_adaptation_failure_does_not_invoke_deterministic_gate(self) -> None:
        fake_client = FakeCandidateWriterClient(
            _provider_response(json.dumps({"post_text": "Missing fields"}))
        )

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "run_final_post_deterministic_gate",
        ) as gate:
            result = execute_final_post_standalone_candidate_attempt(
                _request(),
                selected_evidence_ids=("ev-1",),
                candidate_writer_client=fake_client,
            )

        gate.assert_not_called()
        self.assertEqual(result.failure_stage, STAGE_CANDIDATE_WRITER_ADAPTATION)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_ADAPTATION)
        self.assertEqual(result.completed_stage, STAGE_CANDIDATE_WRITER_PARSE)
        self.assertEqual(result.parsed_candidate, {"post_text": "Missing fields"})
        self.assertIsNone(result.candidate_writer_output)

    def test_deterministic_gate_failure_preserves_candidate_text_and_skips_quality(
        self,
    ) -> None:
        leaking_payload = _candidate_payload(
            post_text="The evidence ev-1 says this candidate leaks an ID."
        )
        fake_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(leaking_payload))
        )

        result = execute_final_post_standalone_candidate_attempt(
            _request(),
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(result.failure_stage, STAGE_DETERMINISTIC_GATE)
        self.assertEqual(result.failure_code, FAILURE_DETERMINISTIC_GATE)
        self.assertEqual(result.completed_stage, STAGE_CANDIDATE_WRITER_ADAPTATION)
        self.assertEqual(
            result.candidate_writer_output.payload["post_text"],
            leaking_payload["post_text"],
        )
        self.assertFalse(result.deterministic_gate_output.diagnostics.system_linkedin_ready)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIn(
            STAGE_QUALITY_EVALUATOR_REQUEST,
            [status.stage for status in result.stage_statuses if status.status == STATUS_SKIPPED],
        )

    def test_gate_runs_once_after_successful_adaptation(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "run_final_post_deterministic_gate",
            wraps=linkedin_post_final_post_attempt_execution.run_final_post_deterministic_gate,
        ) as gate:
            result = execute_final_post_standalone_candidate_attempt(
                _request(),
                selected_evidence_ids=("ev-2", "ev-1"),
                candidate_writer_client=fake_client,
            )

        gate.assert_called_once()
        self.assertEqual(gate.call_args.kwargs["selected_evidence_ids"], ("ev-2", "ev-1"))
        self.assertEqual(result.deterministic_gate_output.selected_evidence_ids, ("ev-2", "ev-1"))

    def test_result_serialization_is_strict_json_safe_and_defensive(self) -> None:
        request = _request(execution_metadata={"trace": {"id": "trace-1"}})
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        result = execute_final_post_standalone_candidate_attempt(
            request,
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )
        result_dict = result.to_dict()
        serialized = json.dumps(result_dict, allow_nan=False, sort_keys=True)
        result_dict["request"]["execution_metadata"]["trace"]["id"] = "mutated"
        result_dict["candidate_writer_output"]["payload"]["post_text"] = "mutated"

        self.assertIn("Real candidate text from fake provider.", serialized)
        self.assertEqual(request.execution_metadata["trace"]["id"], "trace-1")
        self.assertEqual(
            result.candidate_writer_output.payload["post_text"],
            "Real candidate text from fake provider.",
        )

    def test_stage_statuses_preserve_order_and_failure_mapping(self) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        result = execute_final_post_standalone_candidate_attempt(
            _request(),
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                ("quality_evaluator_request", STATUS_SKIPPED),
            ],
        )

    def test_full_attempt_gate_pass_runs_evaluator_and_returns_accepted_outcome(
        self,
    ) -> None:
        candidate_client = FakeCandidateWriterClient(
            _provider_response(_candidate_json())
        )
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(execution_metadata={"runtime_sentinel": "audit-only"}),
            candidate_writer_client=candidate_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertFalse(evaluator_client.json_mode)
        self.assertEqual(result.completed_stage, "attempt_outcome")
        self.assertIsNone(result.failure_code)
        self.assertEqual(result.quality_evaluator_invocation_count, 1)
        self.assertEqual(
            result.candidate_writer_output.payload["post_text"],
            "Real candidate text from fake provider.",
        )
        self.assertEqual(
            result.quality_evaluation_state.status,
            QUALITY_EVALUATION_READY,
        )
        self.assertTrue(result.quality_evaluation_state.quality_review["pass"])
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_ACCEPTED)
        self.assertEqual(
            result.final_attempt_outcome.accepted_result.accepted_payload["post_text"],
            "Real candidate text from fake provider.",
        )
        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_REQUEST, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_PARSE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_REVIEW_NORMALIZATION, STATUS_SUCCEEDED),
                ("attempt_adjudication", STATUS_SUCCEEDED),
                ("attempt_outcome", STATUS_SUCCEEDED),
            ],
        )
        self.assertNotIn(
            "runtime_sentinel",
            result.quality_evaluator_prompt_render.variables["candidate_payload_json"],
        )
        self.assertNotIn(
            "provider-metadata",
            result.quality_evaluator_prompt_render.variables["candidate_payload_json"],
        )

    def test_full_attempt_gate_failure_skips_evaluator(self) -> None:
        leaking_payload = _candidate_payload(
            post_text="This leaks ev-1 into human text."
        )
        candidate_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(leaking_payload))
        )
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=candidate_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_DETERMINISTIC_GATE)
        self.assertIsNone(result.quality_evaluator_raw_response)
        self.assertIsNone(result.quality_evaluation_state)
        self.assertIsNone(result.final_attempt_outcome)

    def test_full_attempt_valid_quality_fail_returns_repair_required_outcome(
        self,
    ) -> None:
        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(
                    json.dumps(
                        _quality_review_payload(
                            passed=False,
                            total_score=34,
                            failed_criteria=["human_voice"],
                        )
                    )
                )
            ),
            **_full_attempt_kwargs(),
        )

        self.assertIsNone(result.failure_code)
        self.assertFalse(result.quality_evaluation_state.quality_review["pass"])
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertTrue(result.final_attempt_outcome.repair_required)

    def test_full_attempt_forwards_policy_for_alternative_model_routing(
        self,
    ) -> None:
        result = execute_final_post_standalone_attempt(
            _request(
                policy=FinalPostDecisionPolicy(
                    max_total_attempts=3,
                    max_editorial_repairs=0,
                    allow_alternative_model=True,
                ),
                alternative_model_available=True,
                target_model_provider="openai",
                target_model_name="fallback-model",
            ),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(
                    json.dumps(
                        _quality_review_payload(
                            passed=False,
                            total_score=34,
                            failed_criteria=["human_voice"],
                        )
                    )
                )
            ),
            **_full_attempt_kwargs(),
        )

        self.assertEqual(
            result.final_attempt_outcome.outcome,
            OUTCOME_TRY_ALTERNATIVE_MODEL,
        )
        self.assertFalse(result.final_attempt_outcome.repair_required)
        self.assertEqual(
            result.final_attempt_outcome.decision.target_model_provider,
            "openai",
        )
        self.assertEqual(
            result.final_attempt_outcome.decision.target_model_name,
            "fallback-model",
        )

    def test_full_attempt_evaluator_request_failure_invokes_evaluator_zero_times(
        self,
    ) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(quality_evaluator_provider=""),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertEqual(result.quality_evaluation_state.status, "not_run")
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_NOT_READY)

    def test_full_attempt_invalid_rubric_preserves_built_post_editorial_input(
        self,
    ) -> None:
        partial_rubric = get_quality_evaluator_rubric_payload().to_dict()
        partial_rubric["criteria"] = {
            "hook": partial_rubric["criteria"]["hook"],
        }

        result = execute_final_post_standalone_attempt(
            _request(quality_rubric=partial_rubric),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(_quality_review_payload(passed=True)))
            ),
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertIsNotNone(result.post_editorial_input)
        self.assertEqual(result.post_editorial_input.post_brief, _post_brief())
        self.assertIsNone(result.quality_evaluator_prompt_render)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)

    def test_full_attempt_evaluator_provider_failure_preserves_raw_failure_state(
        self,
    ) -> None:
        evaluator_client = FailingCandidateWriterClient(RuntimeError("secret evidence"))

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "parse_and_normalize_quality_evaluator_response",
        ) as parse_quality:
            result = execute_final_post_standalone_attempt(
                _request(),
                candidate_writer_client=FakeCandidateWriterClient(
                    _provider_response(_candidate_json())
                ),
                quality_evaluator_client=evaluator_client,
                **_full_attempt_kwargs(),
            )

        parse_quality.assert_not_called()
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_PROVIDER)
        self.assertEqual(result.quality_evaluation_state.status, QUALITY_EVALUATION_EXECUTION_FAILED)
        self.assertEqual(result.quality_evaluator_invocation_count, 1)
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_NOT_READY)
        self.assertNotIn("secret evidence", result.failure_message)
        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_REQUEST, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_EXECUTION, STATUS_FAILED),
            ],
        )

    def test_full_attempt_empty_evaluator_response_is_distinct(self) -> None:
        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(_provider_response("")),
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_EMPTY_RESPONSE)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_EXECUTION)
        self.assertEqual(result.quality_evaluation_state.status, QUALITY_EVALUATION_EXECUTION_FAILED)
        self.assertIsNone(result.quality_evaluation_state.quality_review)

    def test_full_attempt_evaluator_parse_failure_is_distinct(self) -> None:
        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response("{not-json")
            ),
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_PARSE)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_PARSE)
        self.assertEqual(result.quality_evaluation_state.status, QUALITY_EVALUATION_PARSE_FAILED)
        self.assertIsNone(result.quality_evaluation_state.quality_review)
        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_REQUEST, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_PARSE, STATUS_FAILED),
            ],
        )

    def test_full_attempt_quality_normalization_failure_is_distinct(self) -> None:
        invalid_review = _quality_review_payload(passed=True)
        invalid_review["scores"].pop("human_voice")

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(invalid_review))
            ),
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_QUALITY_REVIEW_NORMALIZATION)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_REVIEW_NORMALIZATION)
        self.assertEqual(
            result.quality_evaluation_state.status,
            QUALITY_EVALUATION_NORMALIZATION_FAILED,
        )
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_NOT_READY)
        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_REQUEST, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_QUALITY_EVALUATOR_PARSE, STATUS_SUCCEEDED),
                (STAGE_QUALITY_REVIEW_NORMALIZATION, STATUS_FAILED),
            ],
        )

    def test_full_attempt_success_carries_candidate_invocation_count(self) -> None:
        execute_candidate_attempt = (
            linkedin_post_final_post_attempt_execution
            .execute_final_post_standalone_candidate_attempt
        )

        def execute_candidate_with_retries(*args, **kwargs):
            candidate_result = execute_candidate_attempt(*args, **kwargs)
            return replace(candidate_result, candidate_writer_invocation_count=2)

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "execute_final_post_standalone_candidate_attempt",
            side_effect=execute_candidate_with_retries,
        ):
            result = execute_final_post_standalone_attempt(
                _request(),
                candidate_writer_client=FakeCandidateWriterClient(
                    _provider_response(_candidate_json())
                ),
                quality_evaluator_client=FakeCandidateWriterClient(
                    _provider_response(json.dumps(_quality_review_payload(passed=True)))
                ),
                **_full_attempt_kwargs(),
            )

        self.assertEqual(result.candidate_writer_invocation_count, 2)

    def test_full_attempt_inputs_are_not_mutated_and_result_is_json_safe(self) -> None:
        request = _request()
        post_brief = _post_brief()
        angle_decision = _angle_decision()
        request_before = copy.deepcopy(request)
        post_brief_before = copy.deepcopy(post_brief)
        angle_decision_before = copy.deepcopy(angle_decision)

        result = execute_final_post_standalone_attempt(
            request,
            post_brief=post_brief,
            angle_decision=angle_decision,
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(_quality_review_payload(passed=True)))
            ),
        )
        serialized = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn("Real candidate text from fake provider.", serialized)
        self.assertEqual(request, request_before)
        self.assertEqual(post_brief, post_brief_before)
        self.assertEqual(angle_decision, angle_decision_before)

    def test_execution_module_does_not_import_forbidden_runtime_layers(self) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_final_post_attempt_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
        }
        forbidden_symbols = {
            "ContentPackage",
            "OpenAIClient",
            "RepairAgent",
            "TargetedRepairPlan",
            "generate_content_package_for_digest",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


class FakeCandidateWriterClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_text(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        json_mode: bool,
    ) -> SimpleNamespace:
        self.call_count += 1
        self.prompts.append(prompt)
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        return self.response


class FailingCandidateWriterClient:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.call_count = 0

    def generate_text(self, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        raise self.exc


def _request(
    *,
    candidate_writer_render: object | None = None,
    candidate_writer_provider: str | None = "openai",
    candidate_writer_model: str | None = "candidate-model",
    candidate_writer_max_output_tokens: object = 1200,
    quality_evaluator_provider: str | None = "openai",
    quality_evaluator_model: str | None = "quality-model",
    quality_evaluator_max_output_tokens: object = 900,
    quality_rubric: object | dict | None = None,
    policy: object | dict | None = None,
    alternative_model_available: bool = False,
    target_model_provider: str | None = None,
    target_model_name: str | None = None,
    execution_metadata: dict | None = None,
) -> FinalPostAttemptRequest:
    return FinalPostAttemptRequest(
        candidate_writer_render=(
            _render() if candidate_writer_render is None else candidate_writer_render
        ),
        candidate_writer_prompt_text="Candidate Writer prompt text.",
        quality_rubric=(
            get_quality_evaluator_rubric_payload().to_dict()
            if quality_rubric is None
            else quality_rubric
        ),
        quality_evaluator_prompt_text="Quality prompt not used.",
        attempt_index=0,
        max_attempts=1,
        candidate_writer_provider=candidate_writer_provider,
        candidate_writer_model=candidate_writer_model,
        candidate_writer_max_output_tokens=candidate_writer_max_output_tokens,
        quality_evaluator_provider=quality_evaluator_provider,
        quality_evaluator_model=quality_evaluator_model,
        quality_evaluator_max_output_tokens=quality_evaluator_max_output_tokens,
        policy=policy,
        alternative_model_available=alternative_model_available,
        target_model_provider=target_model_provider,
        target_model_name=target_model_name,
        execution_metadata=copy.deepcopy(execution_metadata),
    )


def _render() -> CandidateWriterPromptRender:
    return CandidateWriterPromptRender(
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
    )


def _provider_response(raw_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=raw_text,
        raw={"id": "resp-1", "metadata_sentinel": "provider-metadata"},
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )


def _candidate_json() -> str:
    return json.dumps(_candidate_payload())


def _candidate_payload(*, post_text: str = "Real candidate text from fake provider.") -> dict:
    return {
        "post_text": post_text,
        "hook_variants": [
            "A practical remote work policy starts here.",
            "Remote policy is not just a document.",
            "Hybrid work needs clearer operating habits.",
        ],
        "cta_variants": [
            "What would you clarify first in a remote policy?",
            "Where does your team still need shared expectations?",
            "What makes hybrid work sustainable in your organization?",
        ],
        "hashtags": ["#remotework", "#futureofwork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }


def _full_attempt_kwargs() -> dict:
    return {
        "post_brief": _post_brief(),
        "angle_decision": _angle_decision(),
        "selected_evidence_ids": ("ev-1",),
    }


def _post_brief() -> dict:
    return {
        "core_point": "Clear remote work policies need human operating habits.",
        "evidence_to_use": [
            {
                "evidence_id": "ev-1",
                "evidence_text": "Remote work policies need clear expectations.",
                "role_in_post": "proof",
            }
        ],
    }


def _angle_decision() -> dict:
    return {
        "controlling_angle": "Remote policies need clarity and inclusion.",
        "author_position": "Leaders should connect policy clarity with team trust.",
    }


def _quality_review_payload(
    *,
    passed: bool = True,
    total_score: int = 37,
    failed_criteria: list[str] | None = None,
) -> dict:
    return {
        "scores": {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 5 if passed else 3,
            "practical_value": 4,
            "cta": 4,
        },
        "total_score": total_score,
        "pass": passed,
        "failed_criteria": failed_criteria or [],
        "automatic_fail_reason": "",
        "notes": ["Evaluator note."],
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
