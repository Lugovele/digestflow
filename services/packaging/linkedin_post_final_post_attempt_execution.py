"""Executable standalone Candidate Writer attempt boundary for PostFlow.

This module composes existing Candidate Writer execution, parsing, adaptation,
and deterministic gate stages. It does not run Quality Evaluator prompts,
adjudicate final outcomes, repair text, persist data, or connect to production
packaging runtime.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterExecutionRequest,
    CandidateWriterRawResponse,
    build_candidate_writer_execution_request,
    execute_candidate_writer_prompt,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CandidateWriterOutputAdaptationError,
    build_candidate_writer_output_from_parsed_response,
)
from services.packaging.linkedin_post_candidate_writer_parser import (
    CandidateWriterResponseParseError,
    parse_candidate_writer_raw_response,
)
from services.packaging.linkedin_post_deterministic_gate import (
    run_final_post_deterministic_gate,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_ADAPTATION,
    FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE,
    FAILURE_CANDIDATE_WRITER_PARSE,
    FAILURE_CANDIDATE_WRITER_PROVIDER,
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_DETERMINISTIC_GATE,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_CANDIDATE_WRITER_EXECUTION,
    STAGE_CANDIDATE_WRITER_PARSE,
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_DETERMINISTIC_GATE,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STATUS_FAILED,
    STATUS_SKIPPED,
    STATUS_SUCCEEDED,
    FinalPostAttemptRequest,
    FinalPostAttemptStageStatus,
    FinalPostStandaloneAttemptResult,
)


def execute_final_post_standalone_candidate_attempt(
    request: FinalPostAttemptRequest,
    *,
    selected_evidence_ids: Sequence[str],
    candidate_writer_client: Any | None = None,
) -> FinalPostStandaloneAttemptResult:
    """Run Candidate Writer through deterministic gate for one attempt."""

    candidate_writer_request = _build_candidate_writer_request(request)
    request_error = _candidate_writer_request_error(candidate_writer_request)
    if request_error is not None:
        return _failure_result(
            request=_request_with_failure_safe_candidate_writer_render(request),
            stage_statuses=(
                _failed_status(
                    STAGE_CANDIDATE_WRITER_REQUEST,
                    FAILURE_CANDIDATE_WRITER_REQUEST,
                    request_error,
                ),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            failure_stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_code=FAILURE_CANDIDATE_WRITER_REQUEST,
            failure_message=request_error,
            candidate_writer_invocation_count=0,
        )

    raw_response = execute_candidate_writer_prompt(
        candidate_writer_request,
        client=candidate_writer_client,
    )
    if raw_response.execution_error:
        failure_code = _candidate_writer_execution_failure_code(raw_response)
        return _failure_result(
            request=request,
            stage_statuses=(
                _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
                _failed_status(
                    STAGE_CANDIDATE_WRITER_EXECUTION,
                    failure_code,
                    raw_response.execution_error,
                ),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_stage=STAGE_CANDIDATE_WRITER_EXECUTION,
            failure_code=failure_code,
            failure_message=raw_response.execution_error,
            candidate_writer_raw_response=raw_response,
            candidate_writer_invocation_count=1,
        )

    try:
        parsed_candidate = parse_candidate_writer_raw_response(raw_response)
    except CandidateWriterResponseParseError as exc:
        return _failure_result(
            request=request,
            stage_statuses=(
                _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
                _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
                _failed_status(
                    STAGE_CANDIDATE_WRITER_PARSE,
                    FAILURE_CANDIDATE_WRITER_PARSE,
                    str(exc),
                ),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_EXECUTION,
            failure_stage=STAGE_CANDIDATE_WRITER_PARSE,
            failure_code=FAILURE_CANDIDATE_WRITER_PARSE,
            failure_message=str(exc),
            candidate_writer_raw_response=raw_response,
            candidate_writer_invocation_count=1,
        )

    try:
        candidate_writer_output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=parsed_candidate,
            raw_response=raw_response,
        )
    except CandidateWriterOutputAdaptationError as exc:
        return _failure_result(
            request=request,
            stage_statuses=(
                _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
                _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
                _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
                _failed_status(
                    STAGE_CANDIDATE_WRITER_ADAPTATION,
                    FAILURE_CANDIDATE_WRITER_ADAPTATION,
                    str(exc),
                ),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_PARSE,
            failure_stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
            failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
            failure_message=str(exc),
            candidate_writer_raw_response=raw_response,
            parsed_candidate=parsed_candidate,
            candidate_writer_invocation_count=1,
        )

    deterministic_gate_output = run_final_post_deterministic_gate(
        candidate_writer_output,
        selected_evidence_ids=tuple(selected_evidence_ids),
    )
    if not _gate_passed(deterministic_gate_output):
        return _failure_result(
            request=request,
            stage_statuses=(
                _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
                _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
                _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
                _succeeded_status(STAGE_CANDIDATE_WRITER_ADAPTATION),
                _failed_status(
                    STAGE_DETERMINISTIC_GATE,
                    FAILURE_DETERMINISTIC_GATE,
                    _gate_failure_message(deterministic_gate_output),
                ),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
            failure_stage=STAGE_DETERMINISTIC_GATE,
            failure_code=FAILURE_DETERMINISTIC_GATE,
            failure_message=_gate_failure_message(deterministic_gate_output),
            candidate_writer_raw_response=raw_response,
            parsed_candidate=parsed_candidate,
            candidate_writer_output=candidate_writer_output,
            deterministic_gate_output=deterministic_gate_output,
            candidate_writer_invocation_count=1,
        )

    return FinalPostStandaloneAttemptResult(
        request=request,
        stage_statuses=(
            _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
            _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
            _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
            _succeeded_status(STAGE_CANDIDATE_WRITER_ADAPTATION),
            _succeeded_status(STAGE_DETERMINISTIC_GATE),
            _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
        ),
        completed_stage=STAGE_DETERMINISTIC_GATE,
        candidate_writer_raw_response=raw_response,
        parsed_candidate=parsed_candidate,
        candidate_writer_output=candidate_writer_output,
        deterministic_gate_output=deterministic_gate_output,
        candidate_writer_invocation_count=1,
        quality_evaluator_invocation_count=0,
    )


def _build_candidate_writer_request(
    request: FinalPostAttemptRequest,
) -> CandidateWriterExecutionRequest:
    return build_candidate_writer_execution_request(
        request.candidate_writer_render,
        prompt_text=request.candidate_writer_prompt_text,
        provider=request.candidate_writer_provider,
        model=request.candidate_writer_model,
        max_output_tokens=(
            1200
            if request.candidate_writer_max_output_tokens is None
            else request.candidate_writer_max_output_tokens
        ),
        execution_metadata=request.execution_metadata,
    )


def _candidate_writer_request_error(
    request: CandidateWriterExecutionRequest,
) -> str | None:
    if not request.provider:
        return "missing candidate writer provider"
    if request.provider != "openai":
        return f"unsupported candidate writer provider: {request.provider}"
    if not request.model:
        return "missing candidate writer model"
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid candidate writer max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid candidate writer max_output_tokens: must be a positive integer"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing candidate writer prompt text"
    rendered_input_text = getattr(request.rendered_prompt_input, "input_text", None)
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing candidate writer rendered input text"
    for metadata_field in ("prompt_name", "prompt_version", "prompt_path"):
        if not hasattr(request.rendered_prompt_input, metadata_field):
            return f"missing candidate writer render metadata: {metadata_field}"
    return None


def _request_with_failure_safe_candidate_writer_render(
    request: FinalPostAttemptRequest,
) -> FinalPostAttemptRequest:
    render = request.candidate_writer_render
    if _is_json_safe_attempt_contract_value(render):
        return request
    return replace(
        request,
        candidate_writer_render=_candidate_writer_render_failure_snapshot(render),
    )


def _candidate_writer_render_failure_snapshot(render: object) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "serialization_status": "unsupported_candidate_writer_render",
        "render_type": type(render).__name__,
    }
    for field_name in (
        "prompt_name",
        "prompt_version",
        "prompt_path",
        "variables",
        "input_text",
    ):
        if hasattr(render, field_name):
            field_value = getattr(render, field_name)
            snapshot[field_name] = (
                field_value
                if _is_json_safe_attempt_contract_value(field_value)
                else {
                    "serialization_status": "unsupported_render_field",
                    "field_type": type(field_value).__name__,
                }
            )
    return snapshot


def _is_json_safe_attempt_contract_value(value: Any) -> bool:
    try:
        FinalPostAttemptRequest(
            candidate_writer_render=value,
            candidate_writer_prompt_text="probe",
            quality_rubric={},
            quality_evaluator_prompt_text="probe",
            attempt_index=0,
            max_attempts=1,
        ).to_dict()
    except TypeError:
        return False
    return True


def _candidate_writer_execution_failure_code(
    raw_response: CandidateWriterRawResponse,
) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE
    return FAILURE_CANDIDATE_WRITER_PROVIDER


def _gate_passed(deterministic_gate_output: Any) -> bool:
    diagnostics = deterministic_gate_output.diagnostics
    return (
        deterministic_gate_output.validation_passed
        and diagnostics.deterministic_checks_passed
        and diagnostics.system_linkedin_ready
    )


def _gate_failure_message(deterministic_gate_output: Any) -> str:
    if deterministic_gate_output.validation_error:
        return deterministic_gate_output.validation_error
    repair_reasons = deterministic_gate_output.diagnostics.repair_reasons
    if repair_reasons:
        return ", ".join(repair_reasons)
    return "deterministic gate failed"


def _failure_result(
    *,
    request: FinalPostAttemptRequest,
    stage_statuses: tuple[FinalPostAttemptStageStatus, ...],
    failure_stage: str,
    failure_code: str,
    failure_message: str,
    completed_stage: str | None = None,
    candidate_writer_raw_response: CandidateWriterRawResponse | None = None,
    parsed_candidate: dict[str, Any] | None = None,
    candidate_writer_output: Any | None = None,
    deterministic_gate_output: Any | None = None,
    candidate_writer_invocation_count: int = 0,
) -> FinalPostStandaloneAttemptResult:
    return FinalPostStandaloneAttemptResult(
        request=request,
        stage_statuses=stage_statuses,
        completed_stage=completed_stage,
        failure_stage=failure_stage,
        failure_code=failure_code,
        failure_message=failure_message,
        candidate_writer_raw_response=candidate_writer_raw_response,
        parsed_candidate=parsed_candidate,
        candidate_writer_output=candidate_writer_output,
        deterministic_gate_output=deterministic_gate_output,
        candidate_writer_invocation_count=candidate_writer_invocation_count,
        quality_evaluator_invocation_count=0,
    )


def _succeeded_status(stage: str) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(stage=stage, status=STATUS_SUCCEEDED)


def _failed_status(
    stage: str,
    error_code: str,
    error_message: str,
) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(
        stage=stage,
        status=STATUS_FAILED,
        error_code=error_code,
        error_message=error_message,
    )


def _skipped_status(stage: str) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(stage=stage, status=STATUS_SKIPPED)
