from __future__ import annotations

import ast
import copy
from dataclasses import replace
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from apps.ai.client import AI_REASONING_EFFORT_MINIMAL
from apps.ai.client import AI_THINKING_MODE_DISABLED
from apps.ai.client import AI_THINKING_MODE_PROVIDER_DEFAULT
from services.packaging import linkedin_post_final_post_attempt_execution
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
)
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NEEDS_HUMAN_REVIEW,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
    OUTCOME_TRY_ALTERNATIVE_MODEL,
)
from services.packaging.linkedin_post_candidate_writer_execution import (
    EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
    PARSER_ERROR_MALFORMED_JSON,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_ADAPTATION,
    FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE,
    FAILURE_CANDIDATE_WRITER_PARSE,
    FAILURE_CANDIDATE_WRITER_PROVIDER,
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_DETERMINISTIC_GATE,
    FAILURE_SEMANTIC_GROUNDING_EMPTY_RESPONSE,
    FAILURE_SEMANTIC_GROUNDING_NORMALIZATION,
    FAILURE_SEMANTIC_GROUNDING_PARSE,
    FAILURE_SEMANTIC_GROUNDING_PROVIDER,
    FAILURE_SEMANTIC_GROUNDING_REQUEST,
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
    STAGE_SEMANTIC_GROUNDING_EXECUTION,
    STAGE_SEMANTIC_GROUNDING_NORMALIZATION,
    STAGE_SEMANTIC_GROUNDING_PARSE,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
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
        self.assertEqual(fake_client.thinking_mode, "provider_default")
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

    def test_anthropic_candidate_writer_reaches_execution_with_disabled_thinking(
        self,
    ) -> None:
        fake_client = FakeCandidateWriterClient(_provider_response(_candidate_json()))

        result = execute_final_post_standalone_candidate_attempt(
            _request(
                candidate_writer_provider="anthropic",
                candidate_writer_model="claude-sonnet-5",
                candidate_writer_max_output_tokens=4000,
            ),
            selected_evidence_ids=("ev-1", "ev-2"),
            candidate_writer_client=fake_client,
        )

        self.assertEqual(fake_client.call_count, 1)
        self.assertEqual(fake_client.max_output_tokens, 4000)
        self.assertEqual(fake_client.thinking_mode, AI_THINKING_MODE_DISABLED)
        self.assertEqual(result.completed_stage, STAGE_DETERMINISTIC_GATE)
        self.assertIsNone(result.failure_code)

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

    def test_empty_response_propagates_safe_provider_diagnostics_to_failure_metadata(
        self,
    ) -> None:
        provider_metadata = {
            "provider": "anthropic",
            "model": "claude-sonnet-5",
            "stop_reason": "max_tokens",
            "content_block_types": ["thinking"],
            "input_tokens": 6616,
            "output_tokens": 4000,
            "thinking_tokens": 4000,
        }
        fake_client = FakeCandidateWriterClient(
            _provider_response("", provider_response_metadata=provider_metadata)
        )

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "parse_candidate_writer_raw_response",
        ) as parse_raw:
            with patch.object(
                linkedin_post_final_post_attempt_execution,
                "build_candidate_writer_output_from_parsed_response",
            ) as adapt_payload:
                result = execute_final_post_standalone_candidate_attempt(
                    _request(
                        candidate_writer_provider="anthropic",
                        candidate_writer_model="claude-sonnet-5",
                    ),
                    selected_evidence_ids=("ev-1",),
                    candidate_writer_client=fake_client,
                )

        parse_raw.assert_not_called()
        adapt_payload.assert_not_called()
        self.assertEqual(fake_client.call_count, 1)
        self.assertEqual(result.failure_code, FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE)
        self.assertEqual(result.candidate_writer_invocation_count, 1)
        self.assertEqual(result.semantic_grounding_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        failed_status = next(
            status
            for status in result.stage_statuses
            if status.stage == STAGE_CANDIDATE_WRITER_EXECUTION
        )
        self.assertEqual(
            failed_status.metadata,
            {
                "provider_response_metadata": provider_metadata,
                "empty_text_classification": (
                    EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT
                ),
            },
        )
        serialized = json.dumps(result.to_dict(), sort_keys=True)
        self.assertIn(EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT, serialized)
        self.assertNotIn("raw request body", serialized)
        self.assertNotIn("secret provider thinking text", serialized)

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
        failed_status = next(
            status
            for status in result.stage_statuses
            if status.stage == STAGE_CANDIDATE_WRITER_PARSE
        )
        diagnostics = failed_status.metadata[
            METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS
        ]
        self.assertEqual(diagnostics["failure_stage"], STAGE_CANDIDATE_WRITER_PARSE)
        self.assertEqual(diagnostics["parser_error_code"], PARSER_ERROR_MALFORMED_JSON)
        self.assertEqual(diagnostics["candidate_text_length"], len("{not-json"))
        serialized = json.dumps(diagnostics, sort_keys=True)
        self.assertNotIn("{not-json", serialized)

    def test_adaptation_failure_does_not_invoke_deterministic_gate(self) -> None:
        fake_client = FakeCandidateWriterClient(
            _provider_response(json.dumps({"post_text": "Unexpected packaging", "hook_variants": []}))
        )

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "run_candidate_post_deterministic_gate",
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
        self.assertEqual(result.parsed_candidate, {"post_text": "Unexpected packaging", "hook_variants": []})
        self.assertIsNone(result.candidate_writer_output)
        failed_status = next(
            status
            for status in result.stage_statuses
            if status.stage == STAGE_CANDIDATE_WRITER_ADAPTATION
        )
        self.assertEqual(
            failed_status.metadata["adaptation_error_code"],
            "invalid_candidate_post",
        )
        self.assertEqual(
            failed_status.metadata["safe_details"]["unexpected_fields"],
            ["hook_variants"],
        )
        diagnostics = failed_status.metadata[
            METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS
        ]
        self.assertEqual(diagnostics["failure_stage"], STAGE_CANDIDATE_WRITER_ADAPTATION)
        self.assertEqual(diagnostics["adapter_error_code"], "payload_contract_violation")
        self.assertEqual(diagnostics["unexpected_fields"], ["hook_variants"])
        self.assertEqual(result.semantic_grounding_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)

    def test_overlength_adaptation_failure_preserves_parsed_post_text(
        self,
    ) -> None:
        overlength_text = "x" * (FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1)
        fake_client = FakeCandidateWriterClient(
            _provider_response(json.dumps({"post_text": overlength_text}))
        )

        with patch.object(
            linkedin_post_final_post_attempt_execution,
            "run_candidate_post_deterministic_gate",
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
        self.assertEqual(result.parsed_candidate, {"post_text": overlength_text})
        self.assertIsNone(result.candidate_writer_output)
        failed_status = next(
            status
            for status in result.stage_statuses
            if status.stage == STAGE_CANDIDATE_WRITER_ADAPTATION
        )
        diagnostics = failed_status.metadata[
            METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS
        ]
        self.assertEqual(diagnostics["failure_stage"], STAGE_CANDIDATE_WRITER_ADAPTATION)
        self.assertEqual(diagnostics["adapter_error_code"], "invalid_field_values")
        self.assertEqual(diagnostics["field_violations"][0]["field_name"], "post_text")
        self.assertEqual(
            diagnostics["field_violations"][0]["reason_code"],
            "above_max_length",
        )
        self.assertEqual(
            diagnostics["field_violations"][0]["actual_length"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS + 1,
        )
        self.assertEqual(result.semantic_grounding_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)

    def test_structural_failure_metadata_resanitizes_diagnostics_before_transport(
        self,
    ) -> None:
        class HostileDiagnostics:
            def to_dict(self) -> dict:
                return {
                    "schema_version": "1.0",
                    "failure_stage": STAGE_CANDIDATE_WRITER_ADAPTATION,
                    "parser_error_code": None,
                    "adapter_error_code": ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
                    "top_level_json_type": "prompt: secret raw response text",
                    "received_top_level_keys": ["post_text", "api_key"],
                    "payload_contract_violation": ["hook_variants"],
                    "unexpected_fields": ["provider_payload", "debug"],
                    "invalid_field_names": ["post_text"],
                    "candidate_text_length": None,
                    "diagnostics_truncated": False,
                    "redacted_key_count": 0,
                }

        metadata = (
            linkedin_post_final_post_attempt_execution
            ._candidate_writer_structural_failure_metadata(
                SimpleNamespace(diagnostics=HostileDiagnostics())
            )
        )

        diagnostics = metadata[METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS]
        self.assertIsNone(diagnostics["top_level_json_type"])
        self.assertEqual(diagnostics["received_top_level_keys"], ["post_text"])
        self.assertEqual(diagnostics["unexpected_fields"], ["debug"])
        serialized = json.dumps(metadata, sort_keys=True)
        self.assertNotIn("secret raw response text", serialized)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("provider_payload", serialized)

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
            "run_candidate_post_deterministic_gate",
            wraps=linkedin_post_final_post_attempt_execution.run_candidate_post_deterministic_gate,
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
        grounding_client = _passing_semantic_client()

        result = execute_final_post_standalone_attempt(
            _request(execution_metadata={"runtime_sentinel": "audit-only"}),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=grounding_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(grounding_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertTrue(grounding_client.json_mode)
        self.assertEqual(grounding_client.max_output_tokens, 4800)
        self.assertEqual(grounding_client.reasoning_effort, AI_REASONING_EFFORT_MINIMAL)
        self.assertIsNone(candidate_client.reasoning_effort)
        self.assertIsNone(evaluator_client.reasoning_effort)
        self.assertTrue(evaluator_client.json_mode)
        self.assertEqual(evaluator_client.max_output_tokens, 2400)
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
                (STAGE_SEMANTIC_GROUNDING_REQUEST, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_PARSE, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_NORMALIZATION, STATUS_SUCCEEDED),
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

    def test_full_attempt_preserves_explicit_quality_evaluator_token_override(
        self,
    ) -> None:
        candidate_client = FakeCandidateWriterClient(
            _provider_response(_candidate_json())
        )
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        execute_final_post_standalone_attempt(
            _request(quality_evaluator_max_output_tokens=2200),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.max_output_tokens, 2200)
        self.assertTrue(evaluator_client.json_mode)

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
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_DETERMINISTIC_GATE)
        self.assertIsNone(result.quality_evaluator_raw_response)
        self.assertIsNone(result.quality_evaluation_state)
        self.assertIsNone(result.final_attempt_outcome)

    def test_full_attempt_grounding_request_failure_invokes_no_grounding_or_quality(
        self,
    ) -> None:
        candidate_client = FakeCandidateWriterClient(
            _provider_response(_candidate_json())
        )
        grounding_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_semantic_review_payload(passed=True)))
        )
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(semantic_grounding_provider=""),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=grounding_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 0)
        self.assertEqual(grounding_client.call_count, 0)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_REQUEST)
        self.assertEqual(result.semantic_grounding_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIsNone(result.semantic_grounding_state)
        self.assertIsNone(result.final_attempt_outcome)

    def test_full_attempt_preflights_quality_config_before_candidate_writer(
        self,
    ) -> None:
        candidate_client = FakeCandidateWriterClient(
            _provider_response(_candidate_json())
        )
        grounding_client = _passing_semantic_client()
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(quality_evaluator_provider="gemini"),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=grounding_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 0)
        self.assertEqual(grounding_client.call_count, 0)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.candidate_writer_invocation_count, 0)
        self.assertEqual(result.semantic_grounding_invocation_count, 0)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIsNone(result.final_attempt_outcome)

    def test_full_attempt_preflight_uses_locked_semantic_grounding_defaults(
        self,
    ) -> None:
        selections = linkedin_post_final_post_attempt_execution._standalone_role_selections(
            _request(),
            candidate_writer_client=None,
            semantic_grounding_client=None,
            quality_evaluator_client=None,
        )
        semantic_selection = selections[1]

        self.assertEqual(semantic_selection.role, "semantic_grounding")
        self.assertEqual(semantic_selection.provider, "gemini")
        self.assertEqual(semantic_selection.model, "gemini-3.6-flash")

    def test_full_attempt_allows_role_policy_mixed_providers(self) -> None:
        candidate_client = FakeCandidateWriterClient(
            _provider_response(_candidate_json())
        )
        grounding_client = _passing_semantic_client()
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(
                candidate_writer_provider="gemini",
                candidate_writer_model="gemini-3.6-flash",
                semantic_grounding_provider="anthropic",
                semantic_grounding_model="claude-sonnet-5",
                quality_evaluator_provider="openai",
            ),
            candidate_writer_client=candidate_client,
            semantic_grounding_client=grounding_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(candidate_client.call_count, 1)
        self.assertEqual(grounding_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 1)
        self.assertEqual(candidate_client.thinking_mode, AI_THINKING_MODE_PROVIDER_DEFAULT)
        self.assertEqual(grounding_client.thinking_mode, AI_THINKING_MODE_PROVIDER_DEFAULT)
        self.assertIsNone(grounding_client.reasoning_effort)
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_ACCEPTED)

    def test_attempt_request_has_no_candidate_writer_thinking_mode_field(self) -> None:
        self.assertFalse(hasattr(_request(), "candidate_writer_thinking_mode"))

    def test_full_attempt_grounding_provider_failure_skips_quality(self) -> None:
        grounding_client = FailingCandidateWriterClient(RuntimeError("secret"))
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=grounding_client,
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(grounding_client.call_count, 1)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_PROVIDER)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_EXECUTION)
        self.assertEqual(result.semantic_grounding_invocation_count, 1)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertNotIn("secret", result.failure_message)

    def test_full_attempt_empty_grounding_response_is_distinct(self) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response("   ")
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_EMPTY_RESPONSE)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_EXECUTION)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.semantic_grounding_state.status, "not_ready")

    def test_full_attempt_grounding_parse_failure_is_distinct(self) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response("{not-json")
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_PARSE)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_PARSE)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_NOT_READY)

    def test_full_attempt_grounding_normalization_failure_is_distinct(self) -> None:
        invalid_review = _semantic_review_payload(passed=True)
        invalid_review["claims"][0]["supported_evidence_ids"] = ["not-selected"]
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(invalid_review))
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_NORMALIZATION)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_NORMALIZATION)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIn(
            "semantic grounding response failed review normalization",
            result.failure_message,
        )

    def test_full_attempt_grounding_pass_with_human_review_flag_is_normalization_failure(
        self,
    ) -> None:
        invalid_review = _semantic_review_payload(passed=True)
        invalid_review["requires_human_review"] = True
        invalid_review["human_review_reason"] = "Needs human factuality review."
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(invalid_review))
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(result.failure_code, FAILURE_SEMANTIC_GROUNDING_NORMALIZATION)
        self.assertEqual(result.failure_stage, STAGE_SEMANTIC_GROUNDING_NORMALIZATION)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIsNone(result.quality_evaluator_raw_response)
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_NOT_READY)

    def test_full_attempt_grounding_failure_returns_claim_specific_repair(self) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(_semantic_review_payload(passed=False)))
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIsNone(result.failure_code)
        self.assertIsNone(result.failure_stage)
        self.assertEqual(result.completed_stage, "attempt_outcome")
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertEqual(
            result.final_attempt_outcome.decision.repair_type,
            "semantic_grounding",
        )
        self.assertEqual(
            result.final_attempt_outcome.repair_plan["failed_claim_ids"],
            ["c1"],
        )
        self.assertEqual(
            result.semantic_grounding_state.grounding_review.blocking_claim_ids,
            ("c1",),
        )

    def test_full_attempt_grounding_human_review_skips_quality(self) -> None:
        human_review_payload = _semantic_review_payload(passed=False)
        human_review_payload["requires_human_review"] = True
        human_review_payload["human_review_reason"] = "candidate contradicts evidence"
        human_review_payload["repairable"] = False
        human_review_payload["repair_instructions"] = []
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(human_review_payload))
            ),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIsNone(result.failure_code)
        self.assertIsNone(result.failure_stage)
        self.assertEqual(
            result.final_attempt_outcome.outcome,
            OUTCOME_NEEDS_HUMAN_REVIEW,
        )
        self.assertTrue(result.final_attempt_outcome.decision.needs_human_review)

    def test_bitcoin_causal_drift_grounding_failure_blocks_quality_evaluator(
        self,
    ) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            post_brief=_bitcoin_post_brief(),
            angle_decision=_bitcoin_angle_decision(),
            selected_evidence_ids=("a0-summary", "a1-kp0", "a2-summary"),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(_bitcoin_candidate_payload()))
            ),
            semantic_grounding_client=FakeCandidateWriterClient(
                _provider_response(json.dumps(_bitcoin_drift_grounding_payload()))
            ),
            quality_evaluator_client=evaluator_client,
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIsNone(result.failure_code)
        self.assertIsNone(result.failure_stage)
        self.assertEqual(result.completed_stage, "attempt_outcome")
        self.assertEqual(result.final_attempt_outcome.outcome, OUTCOME_REPAIR_REQUIRED)
        self.assertEqual(
            result.semantic_grounding_state.grounding_review.blocking_claim_ids,
            ("c-optimism", "c-stability", "c-recovery", "c-growth"),
        )
        self.assertIn(
            "caution influences recovery",
            result.semantic_grounding_state.grounding_review.claim_reviews[2].claim_text,
        )

    def test_full_attempt_valid_quality_fail_returns_repair_required_outcome(
        self,
    ) -> None:
        result = execute_final_post_standalone_attempt(
            _request(),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(
                    json.dumps(
                        _quality_review_payload(
                            passed=False,
                            total_score=35,
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
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=FakeCandidateWriterClient(
                _provider_response(
                    json.dumps(
                        _quality_review_payload(
                            passed=False,
                            total_score=35,
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
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)
        self.assertIsNone(result.quality_evaluation_state)
        self.assertIsNone(result.final_attempt_outcome)

    def test_full_attempt_too_small_evaluator_budget_fails_before_provider(
        self,
    ) -> None:
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(quality_evaluator_max_output_tokens=1999),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=_passing_semantic_client(),
            quality_evaluator_client=evaluator_client,
            **_full_attempt_kwargs(),
        )

        self.assertEqual(evaluator_client.call_count, 0)
        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertIn("must be at least 2000", result.failure_message)
        self.assertEqual(result.quality_evaluator_invocation_count, 0)

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
            semantic_grounding_client=_passing_semantic_client(),
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

    def test_full_attempt_post_editorial_input_failure_skips_semantic_grounding(
        self,
    ) -> None:
        semantic_client = _passing_semantic_client()
        evaluator_client = FakeCandidateWriterClient(
            _provider_response(json.dumps(_quality_review_payload(passed=True)))
        )

        result = execute_final_post_standalone_attempt(
            _request(),
            post_brief={
                "core_point": "Clear remote work policies need human operating habits.",
                "evidence_to_use": [
                    {
                        "evidence_id": "ev-mismatch",
                        "evidence_text": "Unselected evidence should not fit.",
                        "role_in_post": "proof",
                    }
                ],
            },
            angle_decision=_angle_decision(),
            selected_evidence_ids=("ev-1",),
            candidate_writer_client=FakeCandidateWriterClient(
                _provider_response(_candidate_json())
            ),
            semantic_grounding_client=semantic_client,
            quality_evaluator_client=evaluator_client,
        )

        self.assertEqual(result.failure_code, FAILURE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(result.failure_stage, STAGE_QUALITY_EVALUATOR_REQUEST)
        self.assertEqual(semantic_client.call_count, 0)
        self.assertEqual(evaluator_client.call_count, 0)
        self.assertIsNone(result.semantic_grounding_raw_response)
        self.assertIsNone(result.semantic_grounding_state)
        self.assertEqual(
            [(status.stage, status.status) for status in result.stage_statuses],
            [
                (STAGE_CANDIDATE_WRITER_REQUEST, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_PARSE, STATUS_SUCCEEDED),
                (STAGE_CANDIDATE_WRITER_ADAPTATION, STATUS_SUCCEEDED),
                (STAGE_DETERMINISTIC_GATE, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_REQUEST, STATUS_SKIPPED),
                (STAGE_QUALITY_EVALUATOR_REQUEST, STATUS_FAILED),
            ],
        )

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
                semantic_grounding_client=_passing_semantic_client(),
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
                (STAGE_SEMANTIC_GROUNDING_REQUEST, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_PARSE, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_NORMALIZATION, STATUS_SUCCEEDED),
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
            semantic_grounding_client=_passing_semantic_client(),
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
            semantic_grounding_client=_passing_semantic_client(),
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
                (STAGE_SEMANTIC_GROUNDING_REQUEST, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_PARSE, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_NORMALIZATION, STATUS_SUCCEEDED),
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
            semantic_grounding_client=_passing_semantic_client(),
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
                (STAGE_SEMANTIC_GROUNDING_REQUEST, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_EXECUTION, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_PARSE, STATUS_SUCCEEDED),
                (STAGE_SEMANTIC_GROUNDING_NORMALIZATION, STATUS_SUCCEEDED),
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
                semantic_grounding_client=_passing_semantic_client(),
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
            semantic_grounding_client=_passing_semantic_client(),
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
        allow_json_mode_fallback: bool = True,
        thinking_mode: str = "provider_default",
        reasoning_effort: str | None = None,
    ) -> SimpleNamespace:
        self.call_count += 1
        self.prompts.append(prompt)
        self.max_output_tokens = max_output_tokens
        self.json_mode = json_mode
        self.allow_json_mode_fallback = allow_json_mode_fallback
        self.thinking_mode = thinking_mode
        self.reasoning_effort = reasoning_effort
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
    candidate_writer_model: str | None = "gpt-4.1-2025-04-14",
    candidate_writer_max_output_tokens: object = 1200,
    semantic_grounding_provider: str | None = None,
    semantic_grounding_model: str | None = None,
    semantic_grounding_max_output_tokens: object = None,
    quality_evaluator_provider: str | None = "openai",
    quality_evaluator_model: str | None = "gpt-4.1-2025-04-14",
    quality_evaluator_max_output_tokens: object = None,
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
        semantic_grounding_prompt_text="Semantic grounding prompt text.",
        semantic_grounding_provider=semantic_grounding_provider,
        semantic_grounding_model=semantic_grounding_model,
        semantic_grounding_max_output_tokens=semantic_grounding_max_output_tokens,
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


def _provider_response(
    raw_text: str,
    *,
    provider_response_metadata: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=raw_text,
        raw={"id": "resp-1", "metadata_sentinel": "provider-metadata"},
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        provider_response_metadata=copy.deepcopy(provider_response_metadata),
    )


def _passing_semantic_client() -> FakeCandidateWriterClient:
    return FakeCandidateWriterClient(
        _provider_response(json.dumps(_semantic_review_payload(passed=True)))
    )


def _semantic_review_payload(*, passed: bool = True) -> dict:
    return {
        "pass": passed,
        "claims": [
            {
                "claim_id": "c1",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "Real candidate text from fake provider.",
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


def _candidate_json() -> str:
    return json.dumps(_candidate_payload())


def _candidate_payload(*, post_text: str = "Real candidate text from fake provider.") -> dict:
    return {"post_text": post_text}


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
        "authorial_voice_directive": _authorial_voice_directive(),
    }


def _bitcoin_post_brief() -> dict:
    return {
        "core_point": "Bitcoin signals require careful qualification.",
        "evidence_to_use": [
            {
                "evidence_id": "a0-summary",
                "evidence_text": (
                    "30% U.S. crypto ownership; 61% intend to invest more; "
                    "security concerns and volatility limit broader adoption."
                ),
                "role_in_post": "market_context",
            },
            {
                "evidence_id": "a1-kp0",
                "evidence_text": "16.99% CAGR projected from 2025 to 2035.",
                "role_in_post": "growth_projection",
            },
            {
                "evidence_id": "a2-summary",
                "evidence_text": (
                    "K33 says Bitcoin likely bottomed at $60K; pessimistic "
                    "positioning may limit deeper downside; risk remains."
                ),
                "role_in_post": "qualification",
            },
        ],
    }


def _bitcoin_angle_decision() -> dict:
    return {
        "controlling_angle": "Bitcoin market signals need qualified interpretation.",
        "author_position": "Do not turn likelihood and risk into certainty.",
        "authorial_voice_directive": _bitcoin_authorial_voice_directive(),
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


def _bitcoin_authorial_voice_directive() -> dict:
    return {
        "authorial_observation": (
            "The author notices that Bitcoin growth signals still need risk "
            "qualification."
        ),
        "rejected_reading": (
            "Reject turning adoption data, projections, or bottoming signals "
            "into certainty."
        ),
        "why_distinction_matters": (
            "The distinction matters because market interest does not remove "
            "security, volatility, or downside risk."
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


def _bitcoin_candidate_payload() -> dict:
    return _candidate_payload(
        post_text=(
            "Even as optimism builds and some stability emerges, many traders "
            "remain watchful, with caution influencing the pace of recovery "
            "and the path of future growth."
        )
    )


def _bitcoin_drift_grounding_payload() -> dict:
    return {
        "pass": False,
        "claims": [
            {
                "claim_id": "c-optimism",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "optimism builds",
                "claim_type": "market_condition",
                "support_status": "unsupported",
                "severity": "major",
                "supported_evidence_ids": [],
                "required_qualifications": [],
                "missing_qualifications": [],
                "rationale": "The selected evidence does not say optimism builds.",
                "repair_hint": "Remove optimism unless directly grounded.",
            },
            {
                "claim_id": "c-stability",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "some stability emerges",
                "claim_type": "forecast_or_projection",
                "support_status": "missing_required_qualification",
                "severity": "major",
                "supported_evidence_ids": ["a2-summary"],
                "required_qualifications": ["likely", "may", "risk remains"],
                "missing_qualifications": ["likely", "may", "risk remains"],
                "rationale": "Likely bottoming and possible downside limits became stability.",
                "repair_hint": "Preserve likely/may/risk framing.",
            },
            {
                "claim_id": "c-recovery",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "caution influences recovery pace",
                "claim_type": "causal_claim",
                "support_status": "causal_overreach",
                "severity": "blocking",
                "supported_evidence_ids": ["a2-summary"],
                "required_qualifications": ["may"],
                "missing_qualifications": ["may"],
                "rationale": "The evidence does not support a causal recovery claim.",
                "repair_hint": "Remove recovery-causality wording.",
            },
            {
                "claim_id": "c-growth",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "caution influences future growth",
                "claim_type": "causal_claim",
                "support_status": "causal_overreach",
                "severity": "blocking",
                "supported_evidence_ids": ["a1-kp0", "a2-summary"],
                "required_qualifications": ["projected", "may"],
                "missing_qualifications": ["projected", "may"],
                "rationale": "Projected CAGR and trader positioning do not establish this cause.",
                "repair_hint": "Separate projected growth from trader-positioning risk.",
            },
        ],
        "failed_claim_ids": [
            "c-optimism",
            "c-stability",
            "c-recovery",
            "c-growth",
        ],
        "automatic_fail_reason": "unsupported and strengthened Bitcoin claims",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": True,
        "repair_instructions": [
            {
                "claim_id": "c-optimism",
                "instruction": "Remove optimism causal drift.",
            },
            {
                "claim_id": "c-stability",
                "instruction": "Remove stability causal drift.",
            },
            {
                "claim_id": "c-recovery",
                "instruction": "Remove recovery causal drift.",
            },
            {
                "claim_id": "c-growth",
                "instruction": "Keep projected and may qualifications.",
            },
        ],
    }


def _quality_review_payload(
    *,
    passed: bool = True,
    total_score: int | None = None,
    failed_criteria: list[str] | None = None,
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
        "failed_criteria": failed_criteria or [],
        "automatic_fail_reason": "",
        "notes": ["Evaluator note."],
        "criterion_rationales": _criterion_rationales(scores),
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
