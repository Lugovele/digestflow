from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_final_post_attempt_contract
from services.packaging.linkedin_post_attempt_adjudication import (
    FinalPostQualityEvaluationState,
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_NOT_RUN,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
)
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
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
    FINAL_POST_ATTEMPT_STAGE_ORDER,
    STAGE_ATTEMPT_ADJUDICATION,
    STAGE_ATTEMPT_OUTCOME,
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
    FinalPostAttemptStageStatus,
    FinalPostStandaloneAttemptResult,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)


@dataclass(frozen=True)
class SerializableStub:
    name: str
    payload: dict

    def to_dict(self) -> dict:
        return {"name": self.name, "payload": dict(self.payload)}


class UnsupportedLeaf:
    pass


class FinalPostAttemptContractTests(SimpleTestCase):
    def test_request_to_dict_serializes_attempt_context_and_inputs(self) -> None:
        request = _request()

        request_dict = request.to_dict()

        self.assertEqual(request_dict["candidate_writer_prompt_text"], "Writer prompt.")
        self.assertEqual(request_dict["quality_evaluator_prompt_text"], "Quality prompt.")
        self.assertEqual(request_dict["candidate_writer_provider"], "openai")
        self.assertEqual(request_dict["candidate_writer_model"], "gpt-4.1-2025-04-14")
        self.assertEqual(request_dict["quality_evaluator_provider"], "openai")
        self.assertEqual(request_dict["quality_evaluator_model"], "gpt-4.1-2025-04-14")
        self.assertEqual(request_dict["attempt_index"], 0)
        self.assertEqual(request_dict["max_attempts"], 3)
        self.assertEqual(request_dict["parent_attempt_index"], None)
        self.assertEqual(request_dict["execution_metadata"], {"trace_id": "trace-1"})

    def test_stage_order_records_expected_single_attempt_sequence(self) -> None:
        self.assertEqual(
            FINAL_POST_ATTEMPT_STAGE_ORDER,
            (
                STAGE_CANDIDATE_WRITER_REQUEST,
                STAGE_CANDIDATE_WRITER_EXECUTION,
                STAGE_CANDIDATE_WRITER_PARSE,
                STAGE_CANDIDATE_WRITER_ADAPTATION,
                STAGE_DETERMINISTIC_GATE,
                STAGE_QUALITY_EVALUATOR_REQUEST,
                STAGE_QUALITY_EVALUATOR_EXECUTION,
                STAGE_QUALITY_EVALUATOR_PARSE,
                STAGE_QUALITY_REVIEW_NORMALIZATION,
                STAGE_ATTEMPT_ADJUDICATION,
                STAGE_ATTEMPT_OUTCOME,
            ),
        )

    def test_stage_status_to_dict_preserves_distinct_error_code(self) -> None:
        status = FinalPostAttemptStageStatus(
            stage=STAGE_CANDIDATE_WRITER_PARSE,
            status=STATUS_FAILED,
            error_code=FAILURE_CANDIDATE_WRITER_PARSE,
            error_message="candidate writer parse failed",
            metadata={"safe": True},
        )

        status_dict = status.to_dict()

        self.assertEqual(status_dict["stage"], STAGE_CANDIDATE_WRITER_PARSE)
        self.assertEqual(status_dict["status"], STATUS_FAILED)
        self.assertEqual(status_dict["error_code"], FAILURE_CANDIDATE_WRITER_PARSE)

    def test_result_to_dict_preserves_raw_responses_and_stage_artifacts(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_EXECUTION,
                    status=STATUS_SUCCEEDED,
                ),
                FinalPostAttemptStageStatus(
                    stage=STAGE_DETERMINISTIC_GATE,
                    status=STATUS_SUCCEEDED,
                ),
            ),
            completed_stage=STAGE_DETERMINISTIC_GATE,
            candidate_writer_raw_response=_candidate_raw_response(),
            parsed_candidate={"post_text": "Candidate text."},
            candidate_writer_output=SerializableStub(
                "candidate_output",
                {"payload": {"post_text": "Candidate text."}},
            ),
            deterministic_gate_output=SerializableStub(
                "gate_output",
                {"validation_passed": True},
            ),
            quality_evaluator_raw_response=_quality_raw_response(),
            quality_evaluation_state=FinalPostQualityEvaluationState(
                status=QUALITY_EVALUATION_READY,
                quality_review={"pass": True},
            ),
            final_attempt_outcome=SerializableStub(
                "outcome",
                {"outcome": "accepted"},
            ),
            candidate_writer_invocation_count=1,
            quality_evaluator_invocation_count=1,
            audit_metadata={"run_id": "run-1"},
        )

        result_dict = result.to_dict()

        self.assertEqual(
            result_dict["candidate_writer_raw_response"]["raw_text"],
            '{"post_text": "Candidate text."}',
        )
        self.assertEqual(
            result_dict["quality_evaluator_raw_response"]["raw_text"],
            '{"pass": true}',
        )
        self.assertEqual(result_dict["candidate_writer_invocation_count"], 1)
        self.assertEqual(result_dict["quality_evaluator_invocation_count"], 1)
        self.assertEqual(result_dict["repair_invocation_count"], 0)

    def test_candidate_failure_result_can_stop_before_candidate_output(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_EXECUTION,
                    status=STATUS_FAILED,
                    error_code=FAILURE_CANDIDATE_WRITER_PROVIDER,
                    error_message="provider invocation failed",
                ),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_stage=STAGE_CANDIDATE_WRITER_EXECUTION,
            failure_code=FAILURE_CANDIDATE_WRITER_PROVIDER,
            failure_message="provider invocation failed",
            candidate_writer_raw_response=_candidate_raw_response(
                raw_text="",
                execution_error="provider invocation failed",
            ),
            candidate_writer_invocation_count=1,
            quality_evaluator_invocation_count=0,
        )

        result_dict = result.to_dict()

        self.assertIsNone(result_dict["candidate_writer_output"])
        self.assertIsNone(result_dict["deterministic_gate_output"])
        self.assertEqual(result_dict["failure_code"], FAILURE_CANDIDATE_WRITER_PROVIDER)

    def test_parse_and_adaptation_failures_are_represented_distinctly(self) -> None:
        parse_failure = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_PARSE,
                    status=STATUS_FAILED,
                    error_code=FAILURE_CANDIDATE_WRITER_PARSE,
                ),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_EXECUTION,
            failure_stage=STAGE_CANDIDATE_WRITER_PARSE,
            failure_code=FAILURE_CANDIDATE_WRITER_PARSE,
            candidate_writer_invocation_count=1,
        )
        adaptation_failure = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
                    status=STATUS_FAILED,
                    error_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
                ),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_PARSE,
            failure_stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
            failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
            candidate_writer_invocation_count=1,
        )

        self.assertNotEqual(parse_failure.failure_code, adaptation_failure.failure_code)
        self.assertEqual(
            parse_failure.completed_stage,
            STAGE_CANDIDATE_WRITER_EXECUTION,
        )
        self.assertEqual(
            adaptation_failure.completed_stage,
            STAGE_CANDIDATE_WRITER_PARSE,
        )

    def test_gate_failure_result_can_skip_quality_evaluator(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_DETERMINISTIC_GATE,
                    status=STATUS_FAILED,
                    error_code=FAILURE_DETERMINISTIC_GATE,
                ),
                FinalPostAttemptStageStatus(
                    stage=STAGE_QUALITY_EVALUATOR_REQUEST,
                    status=STATUS_SKIPPED,
                ),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
            failure_stage=STAGE_DETERMINISTIC_GATE,
            failure_code=FAILURE_DETERMINISTIC_GATE,
            candidate_writer_invocation_count=1,
            quality_evaluator_invocation_count=0,
        )

        result_dict = result.to_dict()

        self.assertEqual(result_dict["failure_code"], FAILURE_DETERMINISTIC_GATE)
        self.assertEqual(result_dict["quality_evaluator_invocation_count"], 0)
        self.assertIsNone(result_dict["quality_evaluator_raw_response"])

    def test_quality_evaluator_failures_are_represented_distinctly(self) -> None:
        expected_failures = (
            (
                FAILURE_QUALITY_EVALUATOR_REQUEST,
                STAGE_QUALITY_EVALUATOR_REQUEST,
                QUALITY_EVALUATION_NOT_RUN,
                0,
            ),
            (
                FAILURE_QUALITY_EVALUATOR_PROVIDER,
                STAGE_QUALITY_EVALUATOR_EXECUTION,
                QUALITY_EVALUATION_EXECUTION_FAILED,
                1,
            ),
            (
                FAILURE_QUALITY_EVALUATOR_EMPTY_RESPONSE,
                STAGE_QUALITY_EVALUATOR_EXECUTION,
                QUALITY_EVALUATION_EXECUTION_FAILED,
                1,
            ),
            (
                FAILURE_QUALITY_EVALUATOR_PARSE,
                STAGE_QUALITY_EVALUATOR_PARSE,
                QUALITY_EVALUATION_PARSE_FAILED,
                1,
            ),
            (
                FAILURE_QUALITY_REVIEW_NORMALIZATION,
                STAGE_QUALITY_REVIEW_NORMALIZATION,
                QUALITY_EVALUATION_NORMALIZATION_FAILED,
                1,
            ),
        )

        for (
            error_code,
            failure_stage,
            quality_status,
            quality_invocation_count,
        ) in expected_failures:
            with self.subTest(error_code=error_code):
                result = FinalPostStandaloneAttemptResult(
                    request=_request(),
                    stage_statuses=(
                        FinalPostAttemptStageStatus(
                            stage=failure_stage,
                            status=STATUS_FAILED,
                            error_code=error_code,
                        ),
                    ),
                    completed_stage=_stage_before(failure_stage),
                    failure_stage=failure_stage,
                    failure_code=error_code,
                    quality_evaluation_state=FinalPostQualityEvaluationState(
                        status=quality_status,
                        quality_review=None,
                        error_code=error_code,
                    ),
                    candidate_writer_invocation_count=1,
                    quality_evaluator_invocation_count=quality_invocation_count,
                )

                result_dict = result.to_dict()

                self.assertEqual(result_dict["failure_stage"], failure_stage)
                self.assertEqual(result_dict["failure_code"], error_code)
                self.assertEqual(
                    result_dict["stage_statuses"][0]["stage"],
                    failure_stage,
                )
                self.assertEqual(
                    result_dict["quality_evaluation_state"]["status"],
                    quality_status,
                )
                self.assertEqual(
                    result_dict["quality_evaluator_invocation_count"],
                    quality_invocation_count,
                )

    def test_failure_examples_preserve_furthest_completed_stage(self) -> None:
        examples = (
            (
                STAGE_CANDIDATE_WRITER_REQUEST,
                FAILURE_CANDIDATE_WRITER_REQUEST,
                None,
            ),
            (
                STAGE_CANDIDATE_WRITER_EXECUTION,
                FAILURE_CANDIDATE_WRITER_PROVIDER,
                STAGE_CANDIDATE_WRITER_REQUEST,
            ),
            (
                STAGE_CANDIDATE_WRITER_PARSE,
                FAILURE_CANDIDATE_WRITER_PARSE,
                STAGE_CANDIDATE_WRITER_EXECUTION,
            ),
            (
                STAGE_CANDIDATE_WRITER_ADAPTATION,
                FAILURE_CANDIDATE_WRITER_ADAPTATION,
                STAGE_CANDIDATE_WRITER_PARSE,
            ),
            (
                STAGE_DETERMINISTIC_GATE,
                FAILURE_DETERMINISTIC_GATE,
                STAGE_CANDIDATE_WRITER_ADAPTATION,
            ),
            (
                STAGE_QUALITY_EVALUATOR_REQUEST,
                FAILURE_QUALITY_EVALUATOR_REQUEST,
                STAGE_DETERMINISTIC_GATE,
            ),
            (
                STAGE_QUALITY_EVALUATOR_EXECUTION,
                FAILURE_QUALITY_EVALUATOR_PROVIDER,
                STAGE_QUALITY_EVALUATOR_REQUEST,
            ),
            (
                STAGE_QUALITY_EVALUATOR_PARSE,
                FAILURE_QUALITY_EVALUATOR_PARSE,
                STAGE_QUALITY_EVALUATOR_EXECUTION,
            ),
            (
                STAGE_QUALITY_REVIEW_NORMALIZATION,
                FAILURE_QUALITY_REVIEW_NORMALIZATION,
                STAGE_QUALITY_EVALUATOR_PARSE,
            ),
        )

        for failure_stage, failure_code, completed_stage in examples:
            with self.subTest(failure_stage=failure_stage):
                result = FinalPostStandaloneAttemptResult(
                    request=_request(),
                    stage_statuses=(
                        FinalPostAttemptStageStatus(
                            stage=failure_stage,
                            status=STATUS_FAILED,
                            error_code=failure_code,
                        ),
                    ),
                    completed_stage=completed_stage,
                    failure_stage=failure_stage,
                    failure_code=failure_code,
                )

                result_dict = result.to_dict()

                self.assertEqual(result_dict["completed_stage"], completed_stage)
                self.assertEqual(result_dict["failure_stage"], failure_stage)
                self.assertNotEqual(
                    result_dict["completed_stage"],
                    result_dict["failure_stage"],
                )

    def test_repair_invocation_count_must_remain_zero(self) -> None:
        with self.assertRaises(ValueError):
            FinalPostStandaloneAttemptResult(
                request=_request(),
                stage_statuses=(
                    FinalPostAttemptStageStatus(
                        stage=STAGE_ATTEMPT_OUTCOME,
                        status=STATUS_SUCCEEDED,
                    ),
                ),
                repair_invocation_count=1,
            )

    def test_contract_defines_candidate_failure_codes_from_architecture(self) -> None:
        self.assertEqual(
            {
                FAILURE_CANDIDATE_WRITER_REQUEST,
                FAILURE_CANDIDATE_WRITER_PROVIDER,
                FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE,
                FAILURE_CANDIDATE_WRITER_PARSE,
                FAILURE_CANDIDATE_WRITER_ADAPTATION,
            },
            {
                "candidate_writer_request_failure",
                "candidate_writer_provider_failure",
                "candidate_writer_empty_response",
                "candidate_writer_parse_failure",
                "candidate_writer_adaptation_failure",
            },
        )

    def test_result_to_dict_is_json_serializable(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_EXECUTION,
                    status=STATUS_SUCCEEDED,
                ),
            ),
            candidate_writer_raw_response=_candidate_raw_response(),
            audit_metadata={"trace": {"id": "trace-1"}},
        )

        serialized = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn("trace-1", serialized)

    def test_to_dict_rejects_non_finite_float_values(self) -> None:
        for non_finite_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(non_finite_value=non_finite_value):
                result = FinalPostStandaloneAttemptResult(
                    request=FinalPostAttemptRequest(
                        candidate_writer_render={"score": non_finite_value},
                        candidate_writer_prompt_text="Writer prompt.",
                        quality_rubric={"total_score": 45},
                        quality_evaluator_prompt_text="Quality prompt.",
                        attempt_index=0,
                        max_attempts=3,
                    ),
                    stage_statuses=(),
                )

                with self.assertRaises(TypeError):
                    result.to_dict()

    def test_to_dict_rejects_non_finite_direct_scalar_values(self) -> None:
        for non_finite_value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(field="request_attempt_index", value=non_finite_value):
                result = FinalPostStandaloneAttemptResult(
                    request=FinalPostAttemptRequest(
                        candidate_writer_render={"score": 42},
                        candidate_writer_prompt_text="Writer prompt.",
                        quality_rubric={"total_score": 45},
                        quality_evaluator_prompt_text="Quality prompt.",
                        attempt_index=non_finite_value,
                        max_attempts=3,
                    ),
                    stage_statuses=(),
                )

                with self.assertRaises(TypeError):
                    result.to_dict()

            with self.subTest(
                field="result_invocation_count",
                value=non_finite_value,
            ):
                result = FinalPostStandaloneAttemptResult(
                    request=_request(),
                    stage_statuses=(),
                    candidate_writer_invocation_count=non_finite_value,
                )

                with self.assertRaises(TypeError):
                    result.to_dict()

    def test_to_dict_preserves_strict_json_safe_float_values(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=FinalPostAttemptRequest(
                candidate_writer_render={"score": 42.5},
                candidate_writer_prompt_text="Writer prompt.",
                quality_rubric={"total_score": 45},
                quality_evaluator_prompt_text="Quality prompt.",
                attempt_index=0,
                max_attempts=3,
            ),
            stage_statuses=(),
        )

        serialized = json.dumps(result.to_dict(), allow_nan=False, sort_keys=True)

        self.assertIn("42.5", serialized)

    def test_nested_values_are_defensively_copied_during_serialization(self) -> None:
        nested_payload = {"payload": {"marker": "original"}}
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(
                FinalPostAttemptStageStatus(
                    stage=STAGE_CANDIDATE_WRITER_PARSE,
                    status=STATUS_SUCCEEDED,
                ),
            ),
            parsed_candidate={"quality_checks": {"linkedin_ready": True}},
            candidate_writer_output=SerializableStub(
                "candidate_output",
                nested_payload,
            ),
            audit_metadata={"token_usage": {"total": 10}},
        )

        result_dict = result.to_dict()
        result_dict["parsed_candidate"]["quality_checks"]["linkedin_ready"] = False
        result_dict["candidate_writer_output"]["payload"]["payload"][
            "marker"
        ] = "mutated"
        result_dict["audit_metadata"]["token_usage"]["total"] = 99

        self.assertTrue(result.parsed_candidate["quality_checks"]["linkedin_ready"])
        self.assertEqual(nested_payload["payload"]["marker"], "original")
        self.assertEqual(result.audit_metadata["token_usage"]["total"], 10)

    def test_to_dict_rejects_unsupported_nested_leaf_values(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=FinalPostAttemptRequest(
                candidate_writer_render={"unsupported": UnsupportedLeaf()},
                candidate_writer_prompt_text="Writer prompt.",
                quality_rubric={"total_score": 45},
                quality_evaluator_prompt_text="Quality prompt.",
                attempt_index=0,
                max_attempts=3,
            ),
            stage_statuses=(),
        )

        with self.assertRaises(TypeError):
            result.to_dict()

    def test_to_dict_rejects_unsupported_direct_scalar_values(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=FinalPostAttemptRequest(
                candidate_writer_render={"supported": "value"},
                candidate_writer_prompt_text=UnsupportedLeaf(),
                quality_rubric={"total_score": 45},
                quality_evaluator_prompt_text="Quality prompt.",
                attempt_index=0,
                max_attempts=3,
            ),
            stage_statuses=(),
        )

        with self.assertRaises(TypeError):
            result.to_dict()

    def test_stage_status_to_dict_rejects_unsupported_direct_scalar_values(
        self,
    ) -> None:
        status = FinalPostAttemptStageStatus(
            stage=UnsupportedLeaf(),
            status=STATUS_FAILED,
        )

        with self.assertRaises(TypeError):
            status.to_dict()

    def test_result_to_dict_rejects_unsupported_direct_scalar_values(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=_request(),
            stage_statuses=(),
            completed_stage=UnsupportedLeaf(),
        )

        with self.assertRaises(TypeError):
            result.to_dict()

    def test_to_dict_rejects_unsupported_nested_mapping_keys(self) -> None:
        result = FinalPostStandaloneAttemptResult(
            request=FinalPostAttemptRequest(
                candidate_writer_render={"supported": "value"},
                candidate_writer_prompt_text="Writer prompt.",
                quality_rubric={"total_score": 45},
                quality_evaluator_prompt_text="Quality prompt.",
                attempt_index=0,
                max_attempts=3,
                execution_metadata={"nested": {("unsupported", "key"): "value"}},
            ),
            stage_statuses=(),
        )

        with self.assertRaises(TypeError):
            result.to_dict()

    def test_contract_module_does_not_import_runtime_or_stage_execution_layers(self) -> None:
        source = inspect.getsource(linkedin_post_final_post_attempt_contract)

        forbidden_terms = (
            "OpenAIClient",
            "generate_text",
            "execute_candidate_writer_prompt",
            "parse_candidate_writer_raw_response",
            "build_candidate_writer_output_from_parsed_response",
            "run_final_post_deterministic_gate",
            "build_post_editorial_input",
            "render_quality_evaluator_prompt_input",
            "execute_quality_evaluator_prompt",
            "parse_and_normalize_quality_evaluator_response",
            "build_final_post_attempt_outcome",
            "services.packaging.generator",
            "ContentPackage",
            "django.db",
            "apps.packaging.models",
        )
        for term in forbidden_terms:
            with self.subTest(term=term):
                self.assertNotIn(term, source)

    def test_contract_module_does_not_reference_raw_articles_or_repair_execution(
        self,
    ) -> None:
        source = inspect.getsource(linkedin_post_final_post_attempt_contract).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)
        self.assertNotIn("repair_agent", source)
        self.assertNotIn("execute_repair", source)


def _stage_before(stage: str) -> str | None:
    stage_index = FINAL_POST_ATTEMPT_STAGE_ORDER.index(stage)
    if stage_index == 0:
        return None
    return FINAL_POST_ATTEMPT_STAGE_ORDER[stage_index - 1]


def _request() -> FinalPostAttemptRequest:
    return FinalPostAttemptRequest(
        candidate_writer_render=SerializableStub(
            "candidate_render",
            {"input_text": "candidate input"},
        ),
        candidate_writer_prompt_text="Writer prompt.",
        quality_rubric={"total_score": 45},
        quality_evaluator_prompt_text="Quality prompt.",
        attempt_index=0,
        max_attempts=3,
        attempt_history=SerializableStub("history", {"attempts": []}),
        candidate_writer_provider="openai",
        candidate_writer_model="gpt-4.1-2025-04-14",
        candidate_writer_max_output_tokens=1200,
        quality_evaluator_provider="openai",
        quality_evaluator_model="gpt-4.1-2025-04-14",
        quality_evaluator_max_output_tokens=900,
        created_at="2026-07-30T00:00:00Z",
        parent_attempt_index=None,
        policy={"max_attempts": 3},
        alternative_model_available=False,
        target_model_provider=None,
        target_model_name=None,
        execution_metadata={"trace_id": "trace-1"},
    )


def _candidate_raw_response(
    *,
    raw_text: str = '{"post_text": "Candidate text."}',
    execution_error: str | None = None,
) -> CandidateWriterRawResponse:
    return CandidateWriterRawResponse(
        raw_text=raw_text,
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_metadata=PromptMetadata(
            prompt_name="candidate",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
        usage={"total_tokens": 20},
        raw_provider_response={"id": "candidate-response"},
        execution_error=execution_error,
        execution_metadata={"trace_id": "candidate-trace"},
    )


def _quality_raw_response() -> QualityEvaluatorRawResponse:
    return QualityEvaluatorRawResponse(
        raw_text='{"pass": true}',
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_metadata=PromptMetadata(
            prompt_name="quality",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/quality_evaluator.txt",
        ),
        usage={"total_tokens": 10},
        raw_provider_response={"id": "quality-response"},
    )
