"""Executable standalone final-post attempt boundary for PostFlow.

This module composes existing Candidate Writer execution, parsing, adaptation,
deterministic gate, Quality Evaluator execution, and attempt adjudication
stages. It only allows the narrow Candidate Writer length-repair boundary; it
does not perform broad/editorial repair, persist data, or connect to production
packaging runtime.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from django.conf import settings

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
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
    structural_diagnostics_from_dict,
)
from services.packaging.linkedin_post_candidate_writer_length_repair import (
    PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_PATH,
    execute_candidate_writer_length_repair,
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
    build_final_post_attempt_outcome_from_gate_grounding_and_quality,
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
    FAILURE_SEMANTIC_GROUNDING,
    FAILURE_SEMANTIC_GROUNDING_EMPTY_RESPONSE,
    FAILURE_SEMANTIC_GROUNDING_NORMALIZATION,
    FAILURE_SEMANTIC_GROUNDING_PARSE,
    FAILURE_SEMANTIC_GROUNDING_PROVIDER,
    FAILURE_SEMANTIC_GROUNDING_REQUEST,
    STAGE_ATTEMPT_ADJUDICATION,
    STAGE_ATTEMPT_OUTCOME,
    STAGE_CANDIDATE_WRITER_ADAPTATION,
    STAGE_CANDIDATE_WRITER_EXECUTION,
    STAGE_CANDIDATE_WRITER_LENGTH_REPAIR,
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
    FinalPostAttemptStageStatus,
    FinalPostStandaloneAttemptResult,
)
from services.packaging.linkedin_post_final_post_execution_plan import (
    FinalPostExecutionRoleSelection,
    preflight_final_post_execution_plan,
)
from services.packaging.linkedin_post_flow_input_builders import (
    build_post_editorial_input,
)
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_semantic_grounding_prompt_input,
    render_quality_evaluator_prompt_input,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    DEFAULT_MAX_OUTPUT_TOKENS as DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
    QualityEvaluatorExecutionRequest,
    QualityEvaluatorRawResponse,
    build_quality_evaluator_execution_request,
    execute_quality_evaluator_prompt,
    get_quality_evaluator_execution_request_error,
)
from services.packaging.linkedin_post_quality_evaluator_parser import (
    ERROR_NORMALIZATION_FAILED,
    QualityEvaluatorResponseParseError,
    parse_and_normalize_quality_evaluator_response,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    normalize_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_PASS,
    FinalPostSemanticGroundingState,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    DEFAULT_MAX_OUTPUT_TOKENS as DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
    SemanticGroundingExecutionRequest,
    SemanticGroundingRawResponse,
    build_semantic_grounding_execution_request,
    execute_semantic_grounding_prompt,
    get_semantic_grounding_execution_request_error,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    ERROR_NORMALIZATION_FAILED as SEMANTIC_GROUNDING_ERROR_NORMALIZATION_FAILED,
    SemanticGroundingResponseParseError,
    parse_and_normalize_semantic_grounding_response,
)


def execute_final_post_standalone_candidate_attempt(
    request: FinalPostAttemptRequest,
    *,
    selected_evidence_ids: Sequence[str],
    post_brief: object | dict | None = None,
    angle_decision: object | dict | None = None,
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
                _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
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
                    metadata=_candidate_writer_failure_metadata(raw_response),
                ),
                _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
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
                    metadata=_candidate_writer_structural_failure_metadata(exc),
                ),
                _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
                _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
            ),
            completed_stage=STAGE_CANDIDATE_WRITER_EXECUTION,
            failure_stage=STAGE_CANDIDATE_WRITER_PARSE,
            failure_code=FAILURE_CANDIDATE_WRITER_PARSE,
            failure_message=str(exc),
            candidate_writer_raw_response=raw_response,
            candidate_writer_invocation_count=1,
        )

    length_repair_invocation_count = 0
    try:
        candidate_writer_output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=parsed_candidate,
            raw_response=raw_response,
        )
    except CandidateWriterOutputAdaptationError as exc:
        if post_brief is not None and angle_decision is not None:
            repair_result = execute_candidate_writer_length_repair(
                parsed_candidate=parsed_candidate,
                adaptation_error=exc,
                post_brief=post_brief,
                angle_decision=angle_decision,
                provider=candidate_writer_request.provider,
                model=candidate_writer_request.model,
                prompt_text=_candidate_writer_length_repair_prompt_text(),
                client=candidate_writer_client,
                thinking_mode=candidate_writer_request.thinking_mode,
            )
            if repair_result.failure_code is not None:
                length_repair_invocation_count = (
                    1 if repair_result.repair_executed else 0
                )
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
                            metadata=_candidate_writer_adaptation_failure_metadata(
                                exc
                            ),
                        ),
                        _failed_status(
                            STAGE_CANDIDATE_WRITER_LENGTH_REPAIR,
                            repair_result.failure_code,
                            repair_result.failure_message,
                            metadata=_candidate_writer_length_repair_metadata(
                                repair_result,
                                provider=candidate_writer_request.provider,
                                model=candidate_writer_request.model,
                            ),
                        ),
                        _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
                        _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
                    ),
                    completed_stage=STAGE_CANDIDATE_WRITER_PARSE,
                    failure_stage=STAGE_CANDIDATE_WRITER_LENGTH_REPAIR,
                    failure_code=repair_result.failure_code,
                    failure_message=repair_result.failure_message,
                    candidate_writer_raw_response=raw_response,
                    parsed_candidate=parsed_candidate,
                    candidate_writer_invocation_count=1,
                    candidate_writer_length_repair_invocation_count=(
                        length_repair_invocation_count
                    ),
                )
            if repair_result.repair_executed:
                length_repair_invocation_count = 1
                parsed_candidate = repair_result.repaired_candidate or {}
                candidate_writer_output = (
                    build_candidate_writer_output_from_parsed_response(
                        parsed_candidate=parsed_candidate,
                        raw_response=raw_response,
                    )
                )
            else:
                return _adaptation_failure_result(
                    request=request,
                    raw_response=raw_response,
                    parsed_candidate=parsed_candidate,
                    adaptation_error=exc,
                )
        else:
            return _adaptation_failure_result(
                request=request,
                raw_response=raw_response,
                parsed_candidate=parsed_candidate,
                adaptation_error=exc,
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
                _candidate_writer_length_repair_stage_status(
                    length_repair_invocation_count
                ),
                _failed_status(
                    STAGE_DETERMINISTIC_GATE,
                    FAILURE_DETERMINISTIC_GATE,
                    _gate_failure_message(deterministic_gate_output),
                ),
                _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
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
            candidate_writer_length_repair_invocation_count=(
                length_repair_invocation_count
            ),
        )

    return FinalPostStandaloneAttemptResult(
        request=request,
        stage_statuses=(
            _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
            _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
            _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
            _succeeded_status(STAGE_CANDIDATE_WRITER_ADAPTATION),
            _candidate_writer_length_repair_stage_status(length_repair_invocation_count),
            _succeeded_status(STAGE_DETERMINISTIC_GATE),
            _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
        ),
        completed_stage=STAGE_DETERMINISTIC_GATE,
        candidate_writer_raw_response=raw_response,
        parsed_candidate=parsed_candidate,
        candidate_writer_output=candidate_writer_output,
        deterministic_gate_output=deterministic_gate_output,
        candidate_writer_invocation_count=1,
        candidate_writer_length_repair_invocation_count=length_repair_invocation_count,
        quality_evaluator_invocation_count=0,
    )


def execute_final_post_standalone_attempt(
    request: FinalPostAttemptRequest,
    *,
    post_brief: object | dict,
    angle_decision: object | dict,
    selected_evidence_ids: Sequence[str],
    candidate_writer_client: Any | None = None,
    semantic_grounding_client: Any | None = None,
    quality_evaluator_client: Any | None = None,
) -> FinalPostStandaloneAttemptResult:
    """Run one standalone attempt through quality evaluation and adjudication."""

    preflight_result = preflight_final_post_execution_plan(
        _standalone_role_selections(
            request,
            candidate_writer_client=candidate_writer_client,
            semantic_grounding_client=semantic_grounding_client,
            quality_evaluator_client=quality_evaluator_client,
        )
    )
    preflight_failure = preflight_result.first_failure()
    if preflight_failure is not None:
        return _failure_result(
            request=_request_with_failure_safe_candidate_writer_render(request),
            stage_statuses=_preflight_failure_statuses(preflight_failure),
            failure_stage=preflight_failure.stage,
            failure_code=preflight_failure.failure_code,
            failure_message=preflight_failure.message,
            candidate_writer_invocation_count=0,
        )

    candidate_result = execute_final_post_standalone_candidate_attempt(
        request,
        selected_evidence_ids=selected_evidence_ids,
        post_brief=post_brief,
        angle_decision=angle_decision,
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
    except (TypeError, ValueError) as exc:
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
            semantic_grounding_completed=False,
        )

    try:
        semantic_prompt_render = render_semantic_grounding_prompt_input(
            post_editorial_input,
        )
        semantic_request = _build_semantic_grounding_request(
            request,
            semantic_prompt_render,
        )
        semantic_request_error = _semantic_grounding_request_error(semantic_request)
    except ValueError as exc:
        semantic_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            error_message=str(exc),
        )
        return _semantic_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            semantic_grounding_prompt_render=None,
            semantic_grounding_raw_response=None,
            semantic_grounding_state=semantic_state,
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            failure_message=str(exc),
            semantic_grounding_invocation_count=0,
        )

    if semantic_request_error is not None:
        semantic_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            error_message=semantic_request_error,
        )
        return _semantic_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=None,
            semantic_grounding_state=semantic_state,
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            failure_message=semantic_request_error,
            semantic_grounding_invocation_count=0,
        )

    semantic_raw_response = execute_semantic_grounding_prompt(
        semantic_request,
        client=semantic_grounding_client,
    )
    if semantic_raw_response.execution_error:
        failure_code = _semantic_grounding_execution_failure_code(
            semantic_raw_response
        )
        semantic_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=failure_code,
            error_message=semantic_raw_response.execution_error,
        )
        return _semantic_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            stage=STAGE_SEMANTIC_GROUNDING_EXECUTION,
            failure_code=failure_code,
            failure_message=semantic_raw_response.execution_error,
            semantic_grounding_invocation_count=1,
        )

    try:
        semantic_review = parse_and_normalize_semantic_grounding_response(
            semantic_raw_response,
            selected_evidence_ids=tuple(selected_evidence_ids),
        )
    except SemanticGroundingResponseParseError as exc:
        failure_stage = (
            STAGE_SEMANTIC_GROUNDING_NORMALIZATION
            if exc.code == SEMANTIC_GROUNDING_ERROR_NORMALIZATION_FAILED
            else STAGE_SEMANTIC_GROUNDING_PARSE
        )
        failure_code = (
            FAILURE_SEMANTIC_GROUNDING_NORMALIZATION
            if exc.code == SEMANTIC_GROUNDING_ERROR_NORMALIZATION_FAILED
            else FAILURE_SEMANTIC_GROUNDING_PARSE
        )
        semantic_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=failure_code,
            error_message=str(exc),
        )
        return _semantic_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            stage=failure_stage,
            failure_code=failure_code,
            failure_message=str(exc),
            semantic_grounding_invocation_count=1,
        )

    semantic_state = FinalPostSemanticGroundingState(
        status=(
            GROUNDING_STATUS_PASS
            if semantic_review.passed
            else (
                GROUNDING_STATUS_NEEDS_HUMAN_REVIEW
                if semantic_review.requires_human_review
                else GROUNDING_STATUS_FAIL
            )
        ),
        grounding_review=semantic_review,
    )
    if semantic_state.status != GROUNDING_STATUS_PASS:
        return _semantic_failure_result(
            candidate_result=candidate_result,
            post_editorial_input=post_editorial_input,
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            stage=STAGE_SEMANTIC_GROUNDING_NORMALIZATION,
            failure_code=FAILURE_SEMANTIC_GROUNDING,
            failure_message=_semantic_grounding_failure_message(semantic_state),
            semantic_grounding_invocation_count=1,
            domain_review_failure=True,
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
            semantic_grounding_prompt_render=locals().get("semantic_prompt_render"),
            semantic_grounding_raw_response=locals().get("semantic_raw_response"),
            semantic_grounding_state=locals().get("semantic_state"),
            semantic_grounding_invocation_count=1,
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
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            semantic_grounding_invocation_count=1,
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
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            semantic_grounding_invocation_count=1,
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
            semantic_grounding_prompt_render=semantic_prompt_render,
            semantic_grounding_raw_response=semantic_raw_response,
            semantic_grounding_state=semantic_state,
            semantic_grounding_invocation_count=1,
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
            *_successful_grounding_statuses(),
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
        candidate_writer_length_repair_invocation_count=(
            candidate_result.candidate_writer_length_repair_invocation_count
        ),
        semantic_grounding_prompt_render=semantic_prompt_render,
        semantic_grounding_raw_response=semantic_raw_response,
        semantic_grounding_state=semantic_state,
        semantic_grounding_invocation_count=1,
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
            DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS
            if request.quality_evaluator_max_output_tokens is None
            else request.quality_evaluator_max_output_tokens
        ),
        execution_metadata=request.execution_metadata,
    )


def _build_semantic_grounding_request(
    request: FinalPostAttemptRequest,
    semantic_prompt_render: Any,
) -> SemanticGroundingExecutionRequest:
    return build_semantic_grounding_execution_request(
        semantic_prompt_render,
        prompt_text=request.semantic_grounding_prompt_text,
        provider=request.semantic_grounding_provider,
        model=request.semantic_grounding_model,
        max_output_tokens=(
            DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS
            if request.semantic_grounding_max_output_tokens is None
            else request.semantic_grounding_max_output_tokens
        ),
        execution_metadata=request.execution_metadata,
    )


def _candidate_writer_request_error(
    request: CandidateWriterExecutionRequest,
) -> str | None:
    if not request.provider:
        return "missing candidate writer provider"
    if not request.model:
        return "missing candidate writer model"
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_CANDIDATE_WRITER,
        provider=request.provider,
        model=request.model,
    )
    if policy_failure is not None:
        return str(policy_failure)
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
    return get_quality_evaluator_execution_request_error(request)


def _semantic_grounding_request_error(
    request: SemanticGroundingExecutionRequest,
) -> str | None:
    return get_semantic_grounding_execution_request_error(request)


def _standalone_role_selections(
    request: FinalPostAttemptRequest,
    *,
    candidate_writer_client: Any | None,
    semantic_grounding_client: Any | None,
    quality_evaluator_client: Any | None,
) -> tuple[FinalPostExecutionRoleSelection, ...]:
    return (
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_CANDIDATE_WRITER,
            provider=request.candidate_writer_provider,
            model=request.candidate_writer_model,
            stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_code=FAILURE_CANDIDATE_WRITER_REQUEST,
            stage_label="candidate writer",
            validate_key=candidate_writer_client is None,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
            provider=request.semantic_grounding_provider,
            model=request.semantic_grounding_model,
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            stage_label="semantic grounding",
            validate_key=semantic_grounding_client is None,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
            provider=request.quality_evaluator_provider,
            model=request.quality_evaluator_model,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            stage_label="quality evaluator",
            validate_key=quality_evaluator_client is None,
        ),
    )


def _preflight_failure_statuses(
    failure: Any,
) -> tuple[FinalPostAttemptStageStatus, ...]:
    statuses: list[FinalPostAttemptStageStatus] = []
    for stage in (
        STAGE_CANDIDATE_WRITER_REQUEST,
        STAGE_SEMANTIC_GROUNDING_REQUEST,
        STAGE_QUALITY_EVALUATOR_REQUEST,
    ):
        if stage == failure.stage:
            statuses.append(
                _failed_status(stage, failure.failure_code, failure.message)
            )
        else:
            statuses.append(_skipped_status(stage))
    return tuple(statuses)


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


def _candidate_writer_failure_metadata(
    raw_response: CandidateWriterRawResponse,
) -> dict[str, Any] | None:
    if raw_response.execution_error != "empty provider response":
        return None
    metadata: dict[str, Any] = {}
    provider_metadata = raw_response.provider_response_metadata
    if isinstance(provider_metadata, dict):
        metadata["provider_response_metadata"] = copy.deepcopy(provider_metadata)
    if raw_response.empty_text_classification is not None:
        metadata["empty_text_classification"] = raw_response.empty_text_classification
    return metadata or None


def _candidate_writer_adaptation_failure_metadata(
    exc: CandidateWriterOutputAdaptationError,
) -> dict[str, Any] | None:
    metadata = _candidate_writer_structural_failure_metadata(exc) or {}
    if exc.safe_details:
        metadata["adaptation_error_code"] = exc.code
        metadata["safe_details"] = copy.deepcopy(exc.safe_details)
    return metadata or None


def _candidate_writer_length_repair_metadata(
    repair_result: Any,
    *,
    provider: str,
    model: str,
) -> dict[str, Any]:
    eligibility = repair_result.eligibility
    metadata = {
        "repair_eligible": bool(eligibility.eligible),
        "repair_executed": bool(repair_result.repair_executed),
        "repair_provider": provider,
        "repair_model": model,
        "original_post_text_length": eligibility.original_post_text_length,
        "repaired_post_text_length": repair_result.repaired_post_text_length,
        "maximum_allowed": eligibility.maximum_allowed,
        "repair_invocation_count": 1 if repair_result.repair_executed else 0,
        "repair_failure_code": repair_result.failure_code,
        "eligibility_reason": eligibility.reason,
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _candidate_writer_length_repair_stage_status(
    invocation_count: int,
) -> FinalPostAttemptStageStatus:
    if invocation_count:
        return FinalPostAttemptStageStatus(
            stage=STAGE_CANDIDATE_WRITER_LENGTH_REPAIR,
            status=STATUS_SUCCEEDED,
            metadata={"repair_invocation_count": invocation_count},
        )
    return _skipped_status(STAGE_CANDIDATE_WRITER_LENGTH_REPAIR)


def _adaptation_failure_result(
    *,
    request: FinalPostAttemptRequest,
    raw_response: CandidateWriterRawResponse,
    parsed_candidate: dict[str, Any],
    adaptation_error: CandidateWriterOutputAdaptationError,
) -> FinalPostStandaloneAttemptResult:
    return _failure_result(
        request=request,
        stage_statuses=(
            _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
            _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
            _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
            _failed_status(
                STAGE_CANDIDATE_WRITER_ADAPTATION,
                FAILURE_CANDIDATE_WRITER_ADAPTATION,
                str(adaptation_error),
                metadata=_candidate_writer_adaptation_failure_metadata(
                    adaptation_error
                ),
            ),
            _skipped_status(STAGE_CANDIDATE_WRITER_LENGTH_REPAIR),
            _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
            _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
        ),
        completed_stage=STAGE_CANDIDATE_WRITER_PARSE,
        failure_stage=STAGE_CANDIDATE_WRITER_ADAPTATION,
        failure_code=FAILURE_CANDIDATE_WRITER_ADAPTATION,
        failure_message=str(adaptation_error),
        candidate_writer_raw_response=raw_response,
        parsed_candidate=parsed_candidate,
        candidate_writer_invocation_count=1,
    )


def _candidate_writer_length_repair_prompt_text() -> str:
    prompt_path = Path(settings.BASE_DIR) / PROMPT_CANDIDATE_WRITER_LENGTH_REPAIR_PATH
    return prompt_path.read_text(encoding="utf-8")


def _candidate_writer_structural_failure_metadata(exc: Any) -> dict[str, Any] | None:
    diagnostics = getattr(exc, "diagnostics", None)
    if diagnostics is None or not hasattr(diagnostics, "to_dict"):
        return None
    diagnostics = structural_diagnostics_from_dict(diagnostics.to_dict())
    if diagnostics is None:
        return None
    return {
        METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS: diagnostics.to_dict()
    }


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
    candidate_writer_length_repair_invocation_count: int = 0,
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
        candidate_writer_length_repair_invocation_count=(
            candidate_writer_length_repair_invocation_count
        ),
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
    semantic_grounding_prompt_render: Any | None = None,
    semantic_grounding_raw_response: Any | None = None,
    semantic_grounding_state: FinalPostSemanticGroundingState | None = None,
    semantic_grounding_invocation_count: int = 0,
    semantic_grounding_completed: bool = True,
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
            semantic_grounding_completed=semantic_grounding_completed,
        ),
        completed_stage=_completed_stage_before_quality_failure(stage),
        failure_stage=stage,
        failure_code=failure_code,
        failure_message=failure_message,
        candidate_writer_raw_response=candidate_result.candidate_writer_raw_response,
        parsed_candidate=candidate_result.parsed_candidate,
        candidate_writer_output=candidate_result.candidate_writer_output,
        deterministic_gate_output=candidate_result.deterministic_gate_output,
        semantic_grounding_prompt_render=semantic_grounding_prompt_render,
        semantic_grounding_raw_response=semantic_grounding_raw_response,
        semantic_grounding_state=semantic_grounding_state,
        post_editorial_input=post_editorial_input,
        quality_evaluator_prompt_render=quality_evaluator_prompt_render,
        quality_evaluator_raw_response=quality_evaluator_raw_response,
        quality_evaluation_state=quality_evaluation_state,
        final_attempt_outcome=final_attempt_outcome,
        candidate_writer_invocation_count=candidate_result.candidate_writer_invocation_count,
        candidate_writer_length_repair_invocation_count=(
            candidate_result.candidate_writer_length_repair_invocation_count
        ),
        semantic_grounding_invocation_count=semantic_grounding_invocation_count,
        quality_evaluator_invocation_count=quality_evaluator_invocation_count,
    )


def _semantic_failure_result(
    *,
    candidate_result: FinalPostStandaloneAttemptResult,
    post_editorial_input: Any | None,
    semantic_grounding_prompt_render: Any | None,
    semantic_grounding_raw_response: SemanticGroundingRawResponse | None,
    semantic_grounding_state: FinalPostSemanticGroundingState,
    stage: str,
    failure_code: str,
    failure_message: str,
    semantic_grounding_invocation_count: int,
    domain_review_failure: bool = False,
) -> FinalPostStandaloneAttemptResult:
    quality_state = FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_NOT_RUN,
        quality_review=None,
        error_code=failure_code,
        error_message=failure_message,
    )
    final_attempt_outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
        post_brief=(
            post_editorial_input.post_brief
            if post_editorial_input is not None
            else {}
        ),
        candidate_output=candidate_result.candidate_writer_output,
        gate_output=candidate_result.deterministic_gate_output,
        semantic_grounding=semantic_grounding_state,
        quality_evaluation=quality_state,
        attempt_index=candidate_result.request.attempt_index,
        attempt_history=candidate_result.request.attempt_history,
        policy=candidate_result.request.policy,
        alternative_model_available=candidate_result.request.alternative_model_available,
        target_model_provider=candidate_result.request.target_model_provider,
        target_model_name=candidate_result.request.target_model_name,
        created_at=candidate_result.request.created_at,
        parent_attempt_index=candidate_result.request.parent_attempt_index,
    )
    if domain_review_failure:
        return FinalPostStandaloneAttemptResult(
            request=candidate_result.request,
            stage_statuses=_semantic_review_outcome_stage_statuses(),
            completed_stage=STAGE_ATTEMPT_OUTCOME,
            failure_stage=None,
            failure_code=None,
            failure_message="",
            candidate_writer_raw_response=candidate_result.candidate_writer_raw_response,
            parsed_candidate=candidate_result.parsed_candidate,
            candidate_writer_output=candidate_result.candidate_writer_output,
            deterministic_gate_output=candidate_result.deterministic_gate_output,
            semantic_grounding_prompt_render=semantic_grounding_prompt_render,
            semantic_grounding_raw_response=semantic_grounding_raw_response,
            semantic_grounding_state=semantic_grounding_state,
            post_editorial_input=post_editorial_input,
            quality_evaluation_state=quality_state,
            final_attempt_outcome=final_attempt_outcome,
            candidate_writer_invocation_count=(
                candidate_result.candidate_writer_invocation_count
            ),
            candidate_writer_length_repair_invocation_count=(
                candidate_result.candidate_writer_length_repair_invocation_count
            ),
            semantic_grounding_invocation_count=semantic_grounding_invocation_count,
            quality_evaluator_invocation_count=0,
        )
    return FinalPostStandaloneAttemptResult(
        request=candidate_result.request,
        stage_statuses=_semantic_failure_stage_statuses(
            stage,
            failure_code,
            failure_message,
        ),
        completed_stage=_completed_stage_before_semantic_failure(stage),
        failure_stage=stage,
        failure_code=failure_code,
        failure_message=failure_message,
        candidate_writer_raw_response=candidate_result.candidate_writer_raw_response,
        parsed_candidate=candidate_result.parsed_candidate,
        candidate_writer_output=candidate_result.candidate_writer_output,
        deterministic_gate_output=candidate_result.deterministic_gate_output,
        semantic_grounding_prompt_render=semantic_grounding_prompt_render,
        semantic_grounding_raw_response=semantic_grounding_raw_response,
        semantic_grounding_state=semantic_grounding_state,
        post_editorial_input=post_editorial_input,
        quality_evaluation_state=quality_state,
        final_attempt_outcome=final_attempt_outcome,
        candidate_writer_invocation_count=candidate_result.candidate_writer_invocation_count,
        candidate_writer_length_repair_invocation_count=(
            candidate_result.candidate_writer_length_repair_invocation_count
        ),
        semantic_grounding_invocation_count=semantic_grounding_invocation_count,
        quality_evaluator_invocation_count=0,
    )


def _successful_candidate_statuses() -> tuple[FinalPostAttemptStageStatus, ...]:
    return (
        _succeeded_status(STAGE_CANDIDATE_WRITER_REQUEST),
        _succeeded_status(STAGE_CANDIDATE_WRITER_EXECUTION),
        _succeeded_status(STAGE_CANDIDATE_WRITER_PARSE),
        _succeeded_status(STAGE_CANDIDATE_WRITER_ADAPTATION),
        _succeeded_status(STAGE_DETERMINISTIC_GATE),
    )


def _successful_grounding_statuses() -> tuple[FinalPostAttemptStageStatus, ...]:
    return (
        *_successful_candidate_statuses(),
        _succeeded_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
        _succeeded_status(STAGE_SEMANTIC_GROUNDING_EXECUTION),
        _succeeded_status(STAGE_SEMANTIC_GROUNDING_PARSE),
        _succeeded_status(STAGE_SEMANTIC_GROUNDING_NORMALIZATION),
    )


def _quality_failure_stage_statuses(
    stage: str,
    failure_code: str,
    failure_message: str,
    *,
    semantic_grounding_completed: bool = True,
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
    grounding_statuses = (
        _successful_grounding_statuses()
        if semantic_grounding_completed
        else (
            *_successful_candidate_statuses(),
            _skipped_status(STAGE_SEMANTIC_GROUNDING_REQUEST),
        )
    )
    return (
        *grounding_statuses,
        *(
            _succeeded_status(success_stage)
            for success_stage in quality_successes_by_failure_stage.get(stage, ())
        ),
        _failed_status(stage, failure_code, failure_message),
    )


def _semantic_failure_stage_statuses(
    stage: str,
    failure_code: str,
    failure_message: str,
) -> tuple[FinalPostAttemptStageStatus, ...]:
    semantic_successes_by_failure_stage = {
        STAGE_SEMANTIC_GROUNDING_REQUEST: (),
        STAGE_SEMANTIC_GROUNDING_EXECUTION: (
            STAGE_SEMANTIC_GROUNDING_REQUEST,
        ),
        STAGE_SEMANTIC_GROUNDING_PARSE: (
            STAGE_SEMANTIC_GROUNDING_REQUEST,
            STAGE_SEMANTIC_GROUNDING_EXECUTION,
        ),
        STAGE_SEMANTIC_GROUNDING_NORMALIZATION: (
            STAGE_SEMANTIC_GROUNDING_REQUEST,
            STAGE_SEMANTIC_GROUNDING_EXECUTION,
            STAGE_SEMANTIC_GROUNDING_PARSE,
        ),
    }
    return (
        *_successful_candidate_statuses(),
        *(
            _succeeded_status(success_stage)
            for success_stage in semantic_successes_by_failure_stage.get(stage, ())
        ),
        _failed_status(stage, failure_code, failure_message),
        _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
    )


def _semantic_review_outcome_stage_statuses() -> tuple[FinalPostAttemptStageStatus, ...]:
    return (
        *_successful_grounding_statuses(),
        _skipped_status(STAGE_QUALITY_EVALUATOR_REQUEST),
        _succeeded_status(STAGE_ATTEMPT_ADJUDICATION),
        _succeeded_status(STAGE_ATTEMPT_OUTCOME),
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


def _completed_stage_before_semantic_failure(stage: str) -> str:
    if stage == STAGE_SEMANTIC_GROUNDING_REQUEST:
        return STAGE_DETERMINISTIC_GATE
    if stage == STAGE_SEMANTIC_GROUNDING_EXECUTION:
        return STAGE_SEMANTIC_GROUNDING_REQUEST
    if stage == STAGE_SEMANTIC_GROUNDING_PARSE:
        return STAGE_SEMANTIC_GROUNDING_EXECUTION
    if stage == STAGE_SEMANTIC_GROUNDING_NORMALIZATION:
        return STAGE_SEMANTIC_GROUNDING_PARSE
    return STAGE_DETERMINISTIC_GATE


def _semantic_grounding_execution_failure_code(
    raw_response: SemanticGroundingRawResponse,
) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_SEMANTIC_GROUNDING_EMPTY_RESPONSE
    return FAILURE_SEMANTIC_GROUNDING_PROVIDER


def _semantic_grounding_failure_message(
    semantic_state: FinalPostSemanticGroundingState,
) -> str:
    review = semantic_state.grounding_review
    if review is None:
        return "semantic grounding failed"
    if review.automatic_fail_reason:
        return review.automatic_fail_reason
    if review.blocking_claim_ids:
        return "failed claims: " + ", ".join(review.blocking_claim_ids)
    return "semantic grounding failed"


def _succeeded_status(stage: str) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(stage=stage, status=STATUS_SUCCEEDED)


def _failed_status(
    stage: str,
    error_code: str,
    error_message: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(
        stage=stage,
        status=STATUS_FAILED,
        error_code=error_code,
        error_message=error_message,
        metadata=copy.deepcopy(metadata),
    )


def _skipped_status(stage: str) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(stage=stage, status=STATUS_SKIPPED)
