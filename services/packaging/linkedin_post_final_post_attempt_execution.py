"""Executable standalone final-post attempt boundary for PostFlow.

This module composes existing Candidate Writer execution, parsing, adaptation,
deterministic gate, Quality Evaluator execution, and attempt adjudication
stages. It does not repair text, persist data, or connect to production
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
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_NOT_RUN,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_and_quality,
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
from services.packaging.linkedin_post_flow_input_builders import (
    build_post_editorial_input,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_quality_evaluator_prompt_input,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorExecutionRequest,
    QualityEvaluatorRawResponse,
    build_quality_evaluator_execution_request,
    execute_quality_evaluator_prompt,
)
from services.packaging.linkedin_post_quality_evaluator_parser import (
    ERROR_NORMALIZATION_FAILED,
    QualityEvaluatorResponseParseError,
    parse_and_normalize_quality_evaluator_response,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    normalize_quality_evaluator_rubric_payload,
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


def execute_final_post_standalone_attempt(
    request: FinalPostAttemptRequest,
    *,
    post_brief: object | dict,
    angle_decision: object | dict,
    selected_evidence_ids: Sequence[str],
    candidate_writer_client: Any | None = None,
    quality_evaluator_client: Any | None = None,
) -> FinalPostStandaloneAttemptResult:
    """Run one standalone attempt through quality evaluation and adjudication."""

    candidate_result = execute_final_post_standalone_candidate_attempt(
        request,
        selected_evidence_ids=selected_evidence_ids,
        candidate_writer_client=candidate_writer_client,
    )
    if candidate_result.failure_code is not None:
        return candidate_result

    try:
        post_editorial_input = build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=angle_decision,
            candidate_output=candidate_result.candidate_writer_output,
            gate_output=candidate_result.deterministic_gate_output,
        )
    except ValueError as exc:
        quality_state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_NOT_RUN,
            quality_review=None,
            error_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            error_message=str(exc),
        )
        return _quality_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=None,
            quality_evaluator_prompt_render=None,
            quality_evaluator_raw_response=None,
            quality_evaluation_state=quality_state,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            failure_message=str(exc),
            quality_evaluator_invocation_count=0,
        )

    try:
        quality_rubric = normalize_quality_evaluator_rubric_payload(
            request.quality_rubric
        )
        quality_prompt_render = render_quality_evaluator_prompt_input(
            post_editorial_input,
            quality_rubric,
        )
        quality_request = _build_quality_evaluator_request(
            request,
            quality_prompt_render,
        )
        quality_request_error = _quality_evaluator_request_error(quality_request)
    except ValueError as exc:
        quality_state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_NOT_RUN,
            quality_review=None,
            error_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            error_message=str(exc),
        )
        return _quality_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            quality_evaluator_prompt_render=None,
            quality_evaluator_raw_response=None,
            quality_evaluation_state=quality_state,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            failure_message=str(exc),
            quality_evaluator_invocation_count=0,
        )

    if quality_request_error is not None:
        quality_state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_NOT_RUN,
            quality_review=None,
            error_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            error_message=quality_request_error,
        )
        return _quality_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            quality_evaluator_prompt_render=quality_prompt_render,
            quality_evaluator_raw_response=None,
            quality_evaluation_state=quality_state,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            failure_message=quality_request_error,
            quality_evaluator_invocation_count=0,
        )

    quality_raw_response = execute_quality_evaluator_prompt(
        quality_request,
        client=quality_evaluator_client,
    )
    if quality_raw_response.execution_error:
        failure_code = _quality_evaluator_execution_failure_code(quality_raw_response)
        quality_state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_EXECUTION_FAILED,
            quality_review=None,
            error_code=failure_code,
            error_message=quality_raw_response.execution_error,
        )
        return _quality_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            quality_evaluator_prompt_render=quality_prompt_render,
            quality_evaluator_raw_response=quality_raw_response,
            quality_evaluation_state=quality_state,
            stage=STAGE_QUALITY_EVALUATOR_EXECUTION,
            failure_code=failure_code,
            failure_message=quality_raw_response.execution_error,
            quality_evaluator_invocation_count=1,
        )

    try:
        normalized_quality_review = parse_and_normalize_quality_evaluator_response(
            quality_raw_response
        )
    except QualityEvaluatorResponseParseError as exc:
        failure_stage = (
            STAGE_QUALITY_REVIEW_NORMALIZATION
            if exc.code == ERROR_NORMALIZATION_FAILED
            else STAGE_QUALITY_EVALUATOR_PARSE
        )
        failure_code = (
            FAILURE_QUALITY_REVIEW_NORMALIZATION
            if exc.code == ERROR_NORMALIZATION_FAILED
            else FAILURE_QUALITY_EVALUATOR_PARSE
        )
        quality_state = FinalPostQualityEvaluationState(
            status=(
                QUALITY_EVALUATION_NORMALIZATION_FAILED
                if exc.code == ERROR_NORMALIZATION_FAILED
                else QUALITY_EVALUATION_PARSE_FAILED
            ),
            quality_review=None,
            error_code=failure_code,
            error_message=str(exc),
        )
        return _quality_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            quality_evaluator_prompt_render=quality_prompt_render,
            quality_evaluator_raw_response=quality_raw_response,
            quality_evaluation_state=quality_state,
            stage=failure_stage,
            failure_code=failure_code,
            failure_message=str(exc),
            quality_evaluator_invocation_count=1,
        )

    quality_state = FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_READY,
        quality_review=normalized_quality_review,
    )
    final_attempt_outcome = build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=post_brief,
        candidate_output=candidate_result.candidate_writer_output,
        gate_output=candidate_result.deterministic_gate_output,
        quality_evaluation=quality_state,
        attempt_index=request.attempt_index,
        attempt_history=request.attempt_history,
        policy=request.policy,
        alternative_model_available=request.alternative_model_available,
        target_model_provider=request.target_model_provider,
        target_model_name=request.target_model_name,
        created_at=request.created_at,
        parent_attempt_index=request.parent_attempt_index,
    )

    return FinalPostStandaloneAttemptResult(
        request=request,
        stage_statuses=(
            *_successful_candidate_statuses(),
            _succeeded_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            _succeeded_status(STAGE_QUALITY_EVALUATOR_EXECUTION),
            _succeeded_status(STAGE_QUALITY_EVALUATOR_PARSE),
            _succeeded_status(STAGE_QUALITY_REVIEW_NORMALIZATION),
            _succeeded_status(STAGE_ATTEMPT_ADJUDICATION),
            _succeeded_status(STAGE_ATTEMPT_OUTCOME),
        ),
        completed_stage=STAGE_ATTEMPT_OUTCOME,
        candidate_writer_raw_response=candidate_result.candidate_writer_raw_response,
        parsed_candidate=candidate_result.parsed_candidate,
        candidate_writer_output=candidate_result.candidate_writer_output,
        deterministic_gate_output=candidate_result.deterministic_gate_output,
        post_editorial_input=post_editorial_input,
        quality_evaluator_prompt_render=quality_prompt_render,
        quality_evaluator_raw_response=quality_raw_response,
        quality_evaluation_state=quality_state,
        final_attempt_outcome=final_attempt_outcome,
        candidate_writer_invocation_count=candidate_result.candidate_writer_invocation_count,
        quality_evaluator_invocation_count=1,
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


def _build_quality_evaluator_request(
    request: FinalPostAttemptRequest,
    quality_prompt_render: Any,
) -> QualityEvaluatorExecutionRequest:
    return build_quality_evaluator_execution_request(
        quality_prompt_render,
        prompt_text=request.quality_evaluator_prompt_text,
        provider=request.quality_evaluator_provider,
        model=request.quality_evaluator_model,
        max_output_tokens=(
            900
            if request.quality_evaluator_max_output_tokens is None
            else request.quality_evaluator_max_output_tokens
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


def _quality_evaluator_request_error(
    request: QualityEvaluatorExecutionRequest,
) -> str | None:
    if not request.provider:
        return "missing quality evaluator provider"
    if request.provider != "openai":
        return f"unsupported quality evaluator provider: {request.provider}"
    if not request.model:
        return "missing quality evaluator model"
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid quality evaluator max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid quality evaluator max_output_tokens: must be a positive integer"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing quality evaluator prompt text"
    rendered_input_text = getattr(request.rendered_prompt_input, "input_text", None)
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing quality evaluator rendered input text"
    for metadata_field in ("prompt_name", "prompt_version", "prompt_path"):
        if not hasattr(request.rendered_prompt_input, metadata_field):
            return f"missing quality evaluator render metadata: {metadata_field}"
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


def _quality_evaluator_execution_failure_code(
    raw_response: QualityEvaluatorRawResponse,
) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_QUALITY_EVALUATOR_EMPTY_RESPONSE
    return FAILURE_QUALITY_EVALUATOR_PROVIDER


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


def _quality_failure_result(
    *,
    candidate_result: FinalPostStandaloneAttemptResult,
    post_editorial_input: Any | None,
    quality_evaluator_prompt_render: Any | None,
    quality_evaluator_raw_response: QualityEvaluatorRawResponse | None,
    quality_evaluation_state: FinalPostQualityEvaluationState,
    stage: str,
    failure_code: str,
    failure_message: str,
    quality_evaluator_invocation_count: int,
) -> FinalPostStandaloneAttemptResult:
    final_attempt_outcome = build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=(
            post_editorial_input.post_brief
            if post_editorial_input is not None
            else {}
        ),
        candidate_output=candidate_result.candidate_writer_output,
        gate_output=candidate_result.deterministic_gate_output,
        quality_evaluation=quality_evaluation_state,
        attempt_index=candidate_result.request.attempt_index,
        attempt_history=candidate_result.request.attempt_history,
        policy=candidate_result.request.policy,
        alternative_model_available=candidate_result.request.alternative_model_available,
        target_model_provider=candidate_result.request.target_model_provider,
        target_model_name=candidate_result.request.target_model_name,
        created_at=candidate_result.request.created_at,
        parent_attempt_index=candidate_result.request.parent_attempt_index,
    )
    return FinalPostStandaloneAttemptResult(
        request=candidate_result.request,
        stage_statuses=_quality_failure_stage_statuses(
            stage,
            failure_code,
            failure_message,
        ),
        completed_stage=_completed_stage_before_quality_failure(stage),
        failure_stage=stage,
        failure_code=failure_code,
        failure_message=failure_message,
        candidate_writer_raw_response=candidate_result.candidate_writer_raw_response,
        parsed_candidate=candidate_result.parsed_candidate,
        candidate_writer_output=candidate_result.candidate_writer_output,
        deterministic_gate_output=candidate_result.deterministic_gate_output,
        post_editorial_input=post_editorial_input,
        quality_evaluator_prompt_render=quality_evaluator_prompt_render,
        quality_evaluator_raw_response=quality_evaluator_raw_response,
        quality_evaluation_state=quality_evaluation_state,
        final_attempt_outcome=final_attempt_outcome,
        candidate_writer_invocation_count=candidate_result.candidate_writer_invocation_count,
        quality_evaluator_invocation_count=quality_evaluator_invocation_count,
    )


def _successful_candidate_statuses() -> tuple[FinalPostAttemptStageStatus, ...]:
    return (
        _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
        _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
        _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
        _succeeded_status(STAGE_CANDIDATE_WRITER_ADAPTATION),
        _succeeded_status(STAGE_DETERMINISTIC_GATE),
    )


def _quality_failure_stage_statuses(
    stage: str,
    failure_code: str,
    failure_message: str,
) -> tuple[FinalPostAttemptStageStatus, ...]:
    quality_successes_by_failure_stage = {
        STAGE_QUALITY_EVALUATOR_REQUEST: (),
        STAGE_QUALITY_EVALUATOR_EXECUTION: (
            STAGE_QUALITY_EVALUATOR_REQUEST,
        ),
        STAGE_QUALITY_EVALUATOR_PARSE: (
            STAGE_QUALITY_EVALUATOR_REQUEST,
            STAGE_QUALITY_EVALUATOR_EXECUTION,
        ),
        STAGE_QUALITY_REVIEW_NORMALIZATION: (
            STAGE_QUALITY_EVALUATOR_REQUEST,
            STAGE_QUALITY_EVALUATOR_EXECUTION,
            STAGE_QUALITY_EVALUATOR_PARSE,
        ),
    }
    return (
        *_successful_candidate_statuses(),
        *(
            _succeeded_status(success_stage)
            for success_stage in quality_successes_by_failure_stage.get(stage, ())
        ),
        _failed_status(stage, failure_code, failure_message),
    )


def _completed_stage_before_quality_failure(stage: str) -> str:
    if stage == STAGE_QUALITY_EVALUATOR_REQUEST:
        return STAGE_DETERMINISTIC_GATE
    if stage == STAGE_QUALITY_EVALUATOR_EXECUTION:
        return STAGE_QUALITY_EVALUATOR_REQUEST
    if stage == STAGE_QUALITY_EVALUATOR_PARSE:
        return STAGE_QUALITY_EVALUATOR_EXECUTION
    if stage == STAGE_QUALITY_REVIEW_NORMALIZATION:
        return STAGE_QUALITY_EVALUATOR_PARSE
    return STAGE_DETERMINISTIC_GATE


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
