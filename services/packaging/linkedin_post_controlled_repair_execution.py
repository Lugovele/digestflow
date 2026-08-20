"""Controlled one-repair execution slice for final LinkedIn post attempts.

This module composes the existing standalone attempt, a dedicated Repair Writer
execution boundary, deterministic gate, Quality Evaluator, and attempt
adjudication for at most one repaired candidate. It does not persist data,
write ContentPackage records, call production packaging runtime, or mutate
source inputs.
"""
from __future__ import annotations

import copy
from typing import Any

from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_and_quality,
)
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_REPAIR_REQUIRED,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CandidateWriterOutputAdaptationError,
    build_candidate_writer_output_from_parsed_response,
)
from services.packaging.linkedin_post_candidate_writer_parser import (
    CandidateWriterResponseParseError,
    parse_candidate_writer_raw_response,
)
from services.packaging.linkedin_post_controlled_repair_contract import (
    FAILURE_REPAIR_INELIGIBLE,
    FAILURE_REPAIR_WRITER_ADAPTATION,
    FAILURE_REPAIR_WRITER_EMPTY_RESPONSE,
    FAILURE_REPAIR_WRITER_PARSE,
    FAILURE_REPAIR_WRITER_PROVIDER,
    FAILURE_REPAIR_TARGET_NOT_FIXED,
    FAILURE_REPAIR_WRITER_REQUEST,
    FAILURE_REPAIRED_DETERMINISTIC_GATE,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_EMPTY_RESPONSE,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_NORMALIZATION,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_PARSE,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_PROVIDER,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_EMPTY_RESPONSE,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_PARSE,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_PROVIDER,
    FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST,
    FAILURE_REPAIRED_QUALITY_REVIEW_NORMALIZATION,
    REPAIR_ELIGIBLE,
    REPAIR_INELIGIBLE,
    FinalPostControlledRepairRequest,
    FinalPostControlledRepairResult,
    FinalPostRepairEligibility,
)
from services.packaging.linkedin_post_deterministic_gate import (
    run_candidate_post_deterministic_gate,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_final_post_attempt_execution import (
    execute_final_post_standalone_attempt,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_QUALITY_EVALUATOR_REQUEST,
    FAILURE_SEMANTIC_GROUNDING_REQUEST,
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
    STATUS_SKIPPED,
    FinalPostAttemptStageStatus,
    FinalPostStandaloneAttemptResult,
)
from services.packaging.linkedin_post_final_post_execution_plan import (
    FinalPostExecutionRoleSelection,
    preflight_final_post_execution_plan,
)
from services.packaging.linkedin_post_flow_decision import (
    FinalPostDecisionPolicy,
    HUMAN_REVIEW_FLAGS,
    REQUIRED_QUALITY_MINIMUMS,
)
from services.packaging.linkedin_post_flow_input_builders import (
    build_post_editorial_input,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_quality_evaluator_prompt_input,
    render_repair_writer_prompt_input,
    render_semantic_grounding_prompt_input,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    DEFAULT_MAX_OUTPUT_TOKENS as DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
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
from services.packaging.linkedin_post_repair_target_enforcement import (
    evaluate_repair_target_enforcement,
)
from services.packaging.linkedin_post_repair_writer_execution import (
    build_repair_writer_execution_request,
    execute_repair_writer_prompt,
)
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    build_repair_writer_structural_diagnostics,
)
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_PASS,
    FinalPostSemanticGroundingState,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    PRODUCTION_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
    build_semantic_grounding_execution_request,
    execute_semantic_grounding_prompt,
    get_semantic_grounding_execution_request_error,
    production_semantic_grounding_execution_metadata,
    production_semantic_grounding_reasoning_effort,
    resolve_semantic_grounding_execution_model,
    resolve_semantic_grounding_execution_provider,
    semantic_grounding_uses_production_execution_profile,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    ERROR_NORMALIZATION_FAILED as SEMANTIC_GROUNDING_ERROR_NORMALIZATION_FAILED,
    SemanticGroundingResponseParseError,
    parse_and_normalize_semantic_grounding_response,
)


REPAIR_ATTEMPT_INDEX = 1


def execute_final_post_controlled_repair_attempt(
    request: FinalPostControlledRepairRequest,
    *,
    post_brief: object | dict,
    angle_decision: object | dict,
    selected_evidence_ids: tuple[str, ...] | list[str],
    candidate_writer_client: Any | None = None,
    semantic_grounding_client: Any | None = None,
    quality_evaluator_client: Any | None = None,
    repair_writer_client: Any | None = None,
) -> FinalPostControlledRepairResult:
    """Run one initial attempt and, when eligible, exactly one repair attempt."""

    evidence_ids = tuple(selected_evidence_ids)
    if request.repair_enabled:
        preflight_result = preflight_final_post_execution_plan(
            _controlled_repair_role_selections(
                request,
                candidate_writer_client=candidate_writer_client,
                semantic_grounding_client=semantic_grounding_client,
                quality_evaluator_client=quality_evaluator_client,
                repair_writer_client=repair_writer_client,
            )
        )
        preflight_failure = preflight_result.first_failure()
        if preflight_failure is not None:
            return _controlled_preflight_failure_result(request, preflight_failure)

    initial_result = execute_final_post_standalone_attempt(
        request.initial_attempt_request,
        post_brief=post_brief,
        angle_decision=angle_decision,
        selected_evidence_ids=evidence_ids,
        candidate_writer_client=candidate_writer_client,
        semantic_grounding_client=semantic_grounding_client,
        quality_evaluator_client=quality_evaluator_client,
    )
    eligibility = _repair_eligibility(request, initial_result)
    if not eligibility.eligible:
        return _ineligible_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
        )

    repair_instruction = _repair_instruction(initial_result)
    try:
        repair_prompt_render = render_repair_writer_prompt_input(
            original_candidate_payload=initial_result.candidate_writer_output.payload,
            post_brief=post_brief,
            angle_decision=angle_decision,
            selected_evidence=_selected_evidence_for_repair(post_brief, evidence_ids),
            deterministic_findings=_deterministic_findings(initial_result),
            quality_findings=_quality_findings(initial_result),
            repair_instruction=repair_instruction,
            attempt_index=REPAIR_ATTEMPT_INDEX,
            max_attempts=request.max_controlled_attempts,
            prompt_metadata=PromptMetadata(
                prompt_name="final_post_repair_writer",
                prompt_version="1.0",
                prompt_path=None,
            ),
        )
        repair_request = build_repair_writer_execution_request(
            repair_prompt_render,
            prompt_text=request.repair_prompt_text,
            provider=request.repair_provider,
            model=request.repair_model,
            max_output_tokens=(
                1200
                if request.repair_max_output_tokens is None
                else request.repair_max_output_tokens
            ),
            execution_metadata=request.execution_metadata,
        )
    except (TypeError, ValueError) as exc:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=FAILURE_REPAIR_WRITER_REQUEST,
            failure_stage="repair_writer_request",
            failure_message=str(exc),
            repair_prompt_render=None,
            repair_invocation_count=0,
        )

    repair_request_error = _repair_writer_request_error(repair_request)
    if repair_request_error is not None:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=FAILURE_REPAIR_WRITER_REQUEST,
            failure_stage="repair_writer_request",
            failure_message=repair_request_error,
            repair_prompt_render=repair_prompt_render,
            repair_invocation_count=0,
        )

    repair_raw_response = execute_repair_writer_prompt(
        repair_request,
        client=repair_writer_client,
    )
    if repair_raw_response.execution_error:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=_repair_writer_execution_failure_code(repair_raw_response),
            failure_stage="repair_writer_execution",
            failure_message=repair_raw_response.execution_error,
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            repair_invocation_count=1,
        )

    try:
        parsed_repair_candidate = parse_candidate_writer_raw_response(
            repair_raw_response
        )
    except CandidateWriterResponseParseError as exc:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=FAILURE_REPAIR_WRITER_PARSE,
            failure_stage="repair_writer_parse",
            failure_message=str(exc),
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            repair_invocation_count=1,
        )

    try:
        repaired_candidate_output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=parsed_repair_candidate,
            raw_response=repair_raw_response,
        )
    except CandidateWriterOutputAdaptationError as exc:
        repair_writer_structural_diagnostics = build_repair_writer_structural_diagnostics(
            parsed_repair_candidate=parsed_repair_candidate,
            adaptation_error=exc,
            repair_raw_response=repair_raw_response,
        )
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=FAILURE_REPAIR_WRITER_ADAPTATION,
            failure_stage="repair_writer_adaptation",
            failure_message=str(exc),
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            parsed_repair_candidate=parsed_repair_candidate,
            repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
            repair_invocation_count=1,
        )
    repair_writer_structural_diagnostics = build_repair_writer_structural_diagnostics(
        parsed_repair_candidate=parsed_repair_candidate,
        repair_raw_response=repair_raw_response,
    )

    repaired_gate = run_candidate_post_deterministic_gate(
        repaired_candidate_output,
        selected_evidence_ids=evidence_ids,
    )
    if not _gate_passed(repaired_gate):
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=FAILURE_REPAIRED_DETERMINISTIC_GATE,
            failure_stage="repaired_deterministic_gate",
            failure_message=_gate_failure_message(repaired_gate),
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            parsed_repair_candidate=parsed_repair_candidate,
            repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
            repaired_candidate_output=repaired_candidate_output,
            repaired_deterministic_gate_output=repaired_gate,
            repair_invocation_count=1,
        )

    grounding_result = _run_repaired_semantic_grounding(
        request=request,
        post_brief=post_brief,
        angle_decision=angle_decision,
        repaired_candidate_output=repaired_candidate_output,
        repaired_gate=repaired_gate,
        selected_evidence_ids=evidence_ids,
        semantic_grounding_client=semantic_grounding_client,
    )
    if grounding_result["failure_code"] is not None:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=grounding_result["failure_code"],
            failure_stage=grounding_result["failure_stage"],
            failure_message=grounding_result["failure_message"],
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            parsed_repair_candidate=parsed_repair_candidate,
            repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
            repaired_candidate_output=repaired_candidate_output,
            repaired_deterministic_gate_output=repaired_gate,
            repaired_semantic_grounding_prompt_render=grounding_result["prompt_render"],
            repaired_semantic_grounding_raw_response=grounding_result["raw_response"],
            repaired_semantic_grounding_state=grounding_result["grounding_state"],
            repaired_post_editorial_input=grounding_result["post_editorial_input"],
            repair_invocation_count=1,
            semantic_grounding_invocation_count=(
                initial_result.semantic_grounding_invocation_count
                + grounding_result["semantic_grounding_invocation_count"]
            ),
            quality_evaluator_invocation_count=(
                initial_result.quality_evaluator_invocation_count
            ),
        )

    quality_result = _run_repaired_quality_evaluation(
        request=request,
        post_brief=post_brief,
        angle_decision=angle_decision,
        repaired_candidate_output=repaired_candidate_output,
        repaired_gate=repaired_gate,
        quality_evaluator_client=quality_evaluator_client,
    )
    if quality_result["failure_code"] is not None:
        return _repair_failure_result(
            request=request,
            initial_result=initial_result,
            eligibility=eligibility,
            failure_code=quality_result["failure_code"],
            failure_stage=quality_result["failure_stage"],
            failure_message=quality_result["failure_message"],
            repair_prompt_render=repair_prompt_render,
            repair_writer_raw_response=repair_raw_response,
            parsed_repair_candidate=parsed_repair_candidate,
            repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
            repaired_candidate_output=repaired_candidate_output,
            repaired_deterministic_gate_output=repaired_gate,
            repaired_post_editorial_input=quality_result["post_editorial_input"],
            repaired_quality_evaluator_prompt_render=quality_result["prompt_render"],
            repaired_quality_evaluator_raw_response=quality_result["raw_response"],
            repaired_quality_evaluation_state=quality_result["quality_state"],
            repair_invocation_count=1,
            semantic_grounding_invocation_count=(
                initial_result.semantic_grounding_invocation_count
                + grounding_result["semantic_grounding_invocation_count"]
            ),
            quality_evaluator_invocation_count=(
                initial_result.quality_evaluator_invocation_count
                + quality_result["quality_invocation_count"]
            ),
        )

    repaired_outcome = build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=post_brief,
        candidate_output=repaired_candidate_output,
        gate_output=repaired_gate,
        quality_evaluation=quality_result["quality_state"],
        attempt_index=REPAIR_ATTEMPT_INDEX,
        attempt_history=initial_result.final_attempt_outcome.attempt_history,
        policy=_terminal_repair_policy(request.initial_attempt_request.policy),
        alternative_model_available=(
            request.initial_attempt_request.alternative_model_available
        ),
        target_model_provider=request.initial_attempt_request.target_model_provider,
        target_model_name=request.initial_attempt_request.target_model_name,
        repair_plan=repair_instruction,
        parent_attempt_index=request.initial_attempt_request.attempt_index,
        initiating_failed_criterion=repair_instruction.get("failed_criterion"),
        angle_decision=angle_decision,
    )
    repair_target_enforcement_diagnostics = evaluate_repair_target_enforcement(
        quality_review=quality_result["quality_state"].quality_review,
        initiating_failed_criterion=repair_instruction.get("failed_criterion"),
        angle_decision=angle_decision,
    )

    target_not_fixed = repair_target_enforcement_diagnostics.repair_target_fixed is False

    accepted_payload = (
        repaired_outcome.accepted_result.accepted_payload
        if repaired_outcome.accepted_result is not None
        else None
    )
    return FinalPostControlledRepairResult(
        request=request,
        initial_attempt_result=initial_result,
        repair_eligibility=eligibility,
        repair_executed=True,
        repair_prompt_render=repair_prompt_render,
        repair_writer_raw_response=repair_raw_response,
        parsed_repair_candidate=copy.deepcopy(parsed_repair_candidate),
        repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
        repaired_candidate_output=repaired_candidate_output,
        repaired_deterministic_gate_output=repaired_gate,
        repaired_semantic_grounding_prompt_render=grounding_result["prompt_render"],
        repaired_semantic_grounding_raw_response=grounding_result["raw_response"],
        repaired_semantic_grounding_state=grounding_result["grounding_state"],
        repaired_post_editorial_input=quality_result["post_editorial_input"],
        repaired_quality_evaluator_prompt_render=quality_result["prompt_render"],
        repaired_quality_evaluator_raw_response=quality_result["raw_response"],
        repaired_quality_evaluation_state=quality_result["quality_state"],
        repaired_attempt_outcome=repaired_outcome,
        repair_target_enforcement_diagnostics=repair_target_enforcement_diagnostics,
        accepted_payload=accepted_payload,
        terminal_outcome=repaired_outcome.outcome,
        terminal_reason=repaired_outcome.reason,
        failure_stage="repair_target_enforcement" if target_not_fixed else None,
        failure_code=FAILURE_REPAIR_TARGET_NOT_FIXED if target_not_fixed else None,
        failure_message=(
            repair_target_enforcement_diagnostics.repair_target_failure_reason
            if target_not_fixed
            else ""
        ),
        candidate_writer_invocation_count=initial_result.candidate_writer_invocation_count,
        semantic_grounding_invocation_count=(
            initial_result.semantic_grounding_invocation_count
            + grounding_result["semantic_grounding_invocation_count"]
        ),
        repair_invocation_count=1,
        quality_evaluator_invocation_count=(
            initial_result.quality_evaluator_invocation_count
            + quality_result["quality_invocation_count"]
        ),
    )


def _repair_eligibility(
    request: FinalPostControlledRepairRequest,
    initial_result: Any,
) -> FinalPostRepairEligibility:
    if not request.repair_enabled:
        return _ineligible("repair disabled")
    if request.max_controlled_attempts < 2:
        return _ineligible("controlled repair budget exhausted")
    if request.initial_attempt_request.max_attempts < 2:
        return _ineligible("initial request does not allow repair attempts")
    if (
        request.initial_attempt_request.attempt_index
        >= request.initial_attempt_request.max_attempts - 1
    ):
        return _ineligible("attempt budget exhausted")
    if initial_result.failure_code is not None:
        return _ineligible(f"initial attempt failed: {initial_result.failure_code}")
    if initial_result.deterministic_gate_output is None:
        return _ineligible("initial deterministic gate missing")
    if not _gate_passed(initial_result.deterministic_gate_output):
        return _ineligible("initial deterministic gate failed")
    if initial_result.final_attempt_outcome is None:
        return _ineligible("initial attempt outcome missing")
    if initial_result.final_attempt_outcome.outcome != OUTCOME_REPAIR_REQUIRED:
        return _ineligible(
            f"initial outcome is {initial_result.final_attempt_outcome.outcome}"
        )
    decision = initial_result.final_attempt_outcome.decision
    if decision.repair_type == "editorial":
        if initial_result.quality_evaluation_state is None:
            return _ineligible("initial quality evaluation missing")
        if initial_result.quality_evaluation_state.status != QUALITY_EVALUATION_READY:
            return _ineligible(
                f"initial quality evaluation {initial_result.quality_evaluation_state.status}"
            )
        quality_review = initial_result.quality_evaluation_state.quality_review
        if not isinstance(quality_review, dict):
            return _ineligible("initial quality review missing")
        if _quality_review_needs_human_review(quality_review):
            return _ineligible("initial quality review needs human review")
    elif decision.repair_type != "semantic_grounding":
        return _ineligible("initial decision is not repairable by controlled repair")
    if not isinstance(request.repair_prompt_text, str) or not request.repair_prompt_text.strip():
        return _ineligible("missing repair writer prompt text")
    if not request.repair_provider:
        return _ineligible("missing repair writer provider")
    if not request.repair_model:
        return _ineligible("missing repair writer model")
    return FinalPostRepairEligibility(
        status=REPAIR_ELIGIBLE,
        eligible=True,
        reason=f"initial attempt eligible for one {decision.repair_type} repair",
    )


def _ineligible(reason: str) -> FinalPostRepairEligibility:
    return FinalPostRepairEligibility(
        status=REPAIR_INELIGIBLE,
        eligible=False,
        reason=reason,
    )


def _ineligible_result(
    *,
    request: FinalPostControlledRepairRequest,
    initial_result: Any,
    eligibility: FinalPostRepairEligibility,
) -> FinalPostControlledRepairResult:
    initial_outcome = initial_result.final_attempt_outcome
    accepted_payload = (
        initial_outcome.accepted_result.accepted_payload
        if initial_outcome is not None and initial_outcome.accepted_result is not None
        else None
    )
    failure_code = None if accepted_payload is not None else FAILURE_REPAIR_INELIGIBLE
    return FinalPostControlledRepairResult(
        request=request,
        initial_attempt_result=initial_result,
        repair_eligibility=eligibility,
        repair_executed=False,
        accepted_payload=accepted_payload,
        terminal_outcome=(
            initial_outcome.outcome if initial_outcome is not None else "not_ready"
        ),
        terminal_reason=(
            initial_outcome.reason
            if accepted_payload is not None and initial_outcome is not None
            else eligibility.reason
        ),
        failure_stage=None if accepted_payload is not None else "repair_eligibility",
        failure_code=failure_code,
        failure_message="" if accepted_payload is not None else eligibility.reason,
        candidate_writer_invocation_count=initial_result.candidate_writer_invocation_count,
        semantic_grounding_invocation_count=initial_result.semantic_grounding_invocation_count,
        repair_invocation_count=0,
        quality_evaluator_invocation_count=initial_result.quality_evaluator_invocation_count,
    )


def _repair_failure_result(
    *,
    request: FinalPostControlledRepairRequest,
    initial_result: Any,
    eligibility: FinalPostRepairEligibility,
    failure_code: str,
    failure_stage: str,
    failure_message: str,
    repair_prompt_render: Any | None,
    repair_writer_raw_response: Any | None = None,
    parsed_repair_candidate: dict[str, Any] | None = None,
    repaired_candidate_output: Any | None = None,
    repair_writer_structural_diagnostics: Any | None = None,
    repaired_deterministic_gate_output: Any | None = None,
    repaired_semantic_grounding_prompt_render: Any | None = None,
    repaired_semantic_grounding_raw_response: Any | None = None,
    repaired_semantic_grounding_state: Any | None = None,
    repaired_post_editorial_input: Any | None = None,
    repaired_quality_evaluator_prompt_render: Any | None = None,
    repaired_quality_evaluator_raw_response: Any | None = None,
    repaired_quality_evaluation_state: Any | None = None,
    repair_invocation_count: int = 0,
    semantic_grounding_invocation_count: int | None = None,
    quality_evaluator_invocation_count: int | None = None,
) -> FinalPostControlledRepairResult:
    return FinalPostControlledRepairResult(
        request=request,
        initial_attempt_result=initial_result,
        repair_eligibility=eligibility,
        repair_executed=repair_invocation_count > 0,
        repair_prompt_render=repair_prompt_render,
        repair_writer_raw_response=repair_writer_raw_response,
        parsed_repair_candidate=copy.deepcopy(parsed_repair_candidate),
        repair_writer_structural_diagnostics=repair_writer_structural_diagnostics,
        repaired_candidate_output=repaired_candidate_output,
        repaired_deterministic_gate_output=repaired_deterministic_gate_output,
        repaired_semantic_grounding_prompt_render=repaired_semantic_grounding_prompt_render,
        repaired_semantic_grounding_raw_response=repaired_semantic_grounding_raw_response,
        repaired_semantic_grounding_state=repaired_semantic_grounding_state,
        repaired_post_editorial_input=repaired_post_editorial_input,
        repaired_quality_evaluator_prompt_render=repaired_quality_evaluator_prompt_render,
        repaired_quality_evaluator_raw_response=repaired_quality_evaluator_raw_response,
        repaired_quality_evaluation_state=repaired_quality_evaluation_state,
        terminal_outcome="not_ready",
        terminal_reason=failure_message,
        failure_stage=failure_stage,
        failure_code=failure_code,
        failure_message=failure_message,
        candidate_writer_invocation_count=initial_result.candidate_writer_invocation_count,
        semantic_grounding_invocation_count=(
            semantic_grounding_invocation_count
            if semantic_grounding_invocation_count is not None
            else initial_result.semantic_grounding_invocation_count
        ),
        repair_invocation_count=repair_invocation_count,
        quality_evaluator_invocation_count=(
            quality_evaluator_invocation_count
            if quality_evaluator_invocation_count is not None
            else initial_result.quality_evaluator_invocation_count
        ),
    )


def _run_repaired_quality_evaluation(
    *,
    request: FinalPostControlledRepairRequest,
    post_brief: object | dict,
    angle_decision: object | dict,
    repaired_candidate_output: Any,
    repaired_gate: Any,
    quality_evaluator_client: Any | None,
) -> dict[str, Any]:
    try:
        post_editorial_input = build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=angle_decision,
            candidate_output=repaired_candidate_output,
            gate_output=repaired_gate,
        )
        quality_rubric = normalize_quality_evaluator_rubric_payload(
            request.initial_attempt_request.quality_rubric
        )
        prompt_render = render_quality_evaluator_prompt_input(
            post_editorial_input,
            quality_rubric,
        )
        quality_request = build_quality_evaluator_execution_request(
            prompt_render,
            prompt_text=request.initial_attempt_request.quality_evaluator_prompt_text,
            provider=request.initial_attempt_request.quality_evaluator_provider,
            model=request.initial_attempt_request.quality_evaluator_model,
            max_output_tokens=(
                DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS
                if request.initial_attempt_request.quality_evaluator_max_output_tokens
                is None
                else request.initial_attempt_request.quality_evaluator_max_output_tokens
            ),
            execution_metadata=request.initial_attempt_request.execution_metadata,
        )
    except (TypeError, ValueError) as exc:
        return _quality_error(
            failure_code=FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST,
            failure_stage="repaired_quality_evaluator_request",
            failure_message=str(exc),
            quality_state=FinalPostQualityEvaluationState(
                status="not_run",
                quality_review=None,
                error_code=FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST,
                error_message=str(exc),
            ),
            post_editorial_input=locals().get("post_editorial_input"),
            prompt_render=locals().get("prompt_render"),
            raw_response=None,
            invocation_count=0,
        )

    quality_request_error = _quality_evaluator_request_error(quality_request)
    if quality_request_error is not None:
        return _quality_error(
            failure_code=FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST,
            failure_stage="repaired_quality_evaluator_request",
            failure_message=quality_request_error,
            quality_state=FinalPostQualityEvaluationState(
                status="not_run",
                quality_review=None,
                error_code=FAILURE_REPAIRED_QUALITY_EVALUATOR_REQUEST,
                error_message=quality_request_error,
            ),
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=None,
            invocation_count=0,
        )

    raw_response = execute_quality_evaluator_prompt(
        quality_request,
        client=quality_evaluator_client,
    )
    if raw_response.execution_error:
        failure_code = _repaired_quality_execution_failure_code(raw_response)
        quality_state = FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_EXECUTION_FAILED,
            quality_review=None,
            error_code=failure_code,
            error_message=raw_response.execution_error,
        )
        return _quality_error(
            failure_code=failure_code,
            failure_stage="repaired_quality_evaluator_execution",
            failure_message=raw_response.execution_error,
            quality_state=quality_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=raw_response,
            invocation_count=1,
        )

    try:
        quality_review = parse_and_normalize_quality_evaluator_response(raw_response)
    except QualityEvaluatorResponseParseError as exc:
        normalization_failed = exc.code == ERROR_NORMALIZATION_FAILED
        failure_code = (
            FAILURE_REPAIRED_QUALITY_REVIEW_NORMALIZATION
            if normalization_failed
            else FAILURE_REPAIRED_QUALITY_EVALUATOR_PARSE
        )
        quality_state = FinalPostQualityEvaluationState(
            status=(
                QUALITY_EVALUATION_NORMALIZATION_FAILED
                if normalization_failed
                else QUALITY_EVALUATION_PARSE_FAILED
            ),
            quality_review=None,
            error_code=failure_code,
            error_message=str(exc),
        )
        return _quality_error(
            failure_code=failure_code,
            failure_stage=(
                "repaired_quality_review_normalization"
                if normalization_failed
                else "repaired_quality_evaluator_parse"
            ),
            failure_message=str(exc),
            quality_state=quality_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=raw_response,
            invocation_count=1,
        )

    quality_state = FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_READY,
        quality_review=quality_review,
    )
    return {
        "failure_code": None,
        "failure_stage": None,
        "failure_message": "",
        "post_editorial_input": post_editorial_input,
        "prompt_render": prompt_render,
        "raw_response": raw_response,
        "quality_state": quality_state,
        "quality_invocation_count": 1,
    }


def _run_repaired_semantic_grounding(
    *,
    request: FinalPostControlledRepairRequest,
    post_brief: object | dict,
    angle_decision: object | dict,
    repaired_candidate_output: Any,
    repaired_gate: Any,
    selected_evidence_ids: tuple[str, ...],
    semantic_grounding_client: Any | None,
) -> dict[str, Any]:
    try:
        post_editorial_input = build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=angle_decision,
            candidate_output=repaired_candidate_output,
            gate_output=repaired_gate,
        )
        prompt_render = render_semantic_grounding_prompt_input(post_editorial_input)
        semantic_request = build_semantic_grounding_execution_request(
            prompt_render,
            prompt_text=request.initial_attempt_request.semantic_grounding_prompt_text,
            provider=resolve_semantic_grounding_execution_provider(
                request.initial_attempt_request.semantic_grounding_provider
            ),
            model=resolve_semantic_grounding_execution_model(
                request.initial_attempt_request.semantic_grounding_provider,
                request.initial_attempt_request.semantic_grounding_model,
            ),
            max_output_tokens=(
                PRODUCTION_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS
                if request.initial_attempt_request.semantic_grounding_max_output_tokens
                is None
                else request.initial_attempt_request.semantic_grounding_max_output_tokens
            ),
            reasoning_effort=(
                production_semantic_grounding_reasoning_effort()
                if semantic_grounding_uses_production_execution_profile(
                    request.initial_attempt_request.semantic_grounding_provider
                )
                else None
            ),
            execution_metadata={
                **copy.deepcopy(request.initial_attempt_request.execution_metadata or {}),
                **(
                    production_semantic_grounding_execution_metadata()
                    if semantic_grounding_uses_production_execution_profile(
                        request.initial_attempt_request.semantic_grounding_provider
                    )
                    else {}
                ),
            },
        )
    except (TypeError, ValueError) as exc:
        grounding_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST,
            error_message=str(exc),
        )
        return _semantic_error(
            failure_code=FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST,
            failure_stage="repaired_semantic_grounding_request",
            failure_message=str(exc),
            grounding_state=grounding_state,
            post_editorial_input=locals().get("post_editorial_input"),
            prompt_render=locals().get("prompt_render"),
            raw_response=None,
            invocation_count=0,
        )

    request_error = get_semantic_grounding_execution_request_error(semantic_request)
    if request_error is not None:
        grounding_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST,
            error_message=request_error,
        )
        return _semantic_error(
            failure_code=FAILURE_REPAIRED_SEMANTIC_GROUNDING_REQUEST,
            failure_stage="repaired_semantic_grounding_request",
            failure_message=request_error,
            grounding_state=grounding_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=None,
            invocation_count=0,
        )

    raw_response = execute_semantic_grounding_prompt(
        semantic_request,
        client=semantic_grounding_client,
    )
    if raw_response.execution_error:
        failure_code = _repaired_semantic_execution_failure_code(raw_response)
        grounding_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=failure_code,
            error_message=raw_response.execution_error,
        )
        return _semantic_error(
            failure_code=failure_code,
            failure_stage="repaired_semantic_grounding_execution",
            failure_message=raw_response.execution_error,
            grounding_state=grounding_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=raw_response,
            invocation_count=1,
        )

    try:
        grounding_review = parse_and_normalize_semantic_grounding_response(
            raw_response,
            selected_evidence_ids=selected_evidence_ids,
        )
    except SemanticGroundingResponseParseError as exc:
        normalization_failed = exc.code == SEMANTIC_GROUNDING_ERROR_NORMALIZATION_FAILED
        failure_code = (
            FAILURE_REPAIRED_SEMANTIC_GROUNDING_NORMALIZATION
            if normalization_failed
            else FAILURE_REPAIRED_SEMANTIC_GROUNDING_PARSE
        )
        grounding_state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_NOT_READY,
            grounding_review=None,
            error_code=failure_code,
            error_message=str(exc),
        )
        return _semantic_error(
            failure_code=failure_code,
            failure_stage=(
                "repaired_semantic_grounding_normalization"
                if normalization_failed
                else "repaired_semantic_grounding_parse"
            ),
            failure_message=str(exc),
            grounding_state=grounding_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=raw_response,
            invocation_count=1,
        )

    grounding_state = FinalPostSemanticGroundingState(
        status=(
            GROUNDING_STATUS_PASS
            if grounding_review.passed
            else (
                GROUNDING_STATUS_NEEDS_HUMAN_REVIEW
                if grounding_review.requires_human_review
                else GROUNDING_STATUS_FAIL
            )
        ),
        grounding_review=grounding_review,
    )
    if grounding_state.status != GROUNDING_STATUS_PASS:
        return _semantic_error(
            failure_code=FAILURE_REPAIRED_SEMANTIC_GROUNDING,
            failure_stage="repaired_semantic_grounding_normalization",
            failure_message=_semantic_grounding_failure_message(grounding_state),
            grounding_state=grounding_state,
            post_editorial_input=post_editorial_input,
            prompt_render=prompt_render,
            raw_response=raw_response,
            invocation_count=1,
        )

    return {
        "failure_code": None,
        "failure_stage": None,
        "failure_message": "",
        "post_editorial_input": post_editorial_input,
        "prompt_render": prompt_render,
        "raw_response": raw_response,
        "grounding_state": grounding_state,
        "semantic_grounding_invocation_count": 1,
    }


def _quality_error(
    *,
    failure_code: str,
    failure_stage: str,
    failure_message: str,
    quality_state: FinalPostQualityEvaluationState,
    post_editorial_input: Any | None,
    prompt_render: Any | None,
    raw_response: Any | None,
    invocation_count: int,
) -> dict[str, Any]:
    return {
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "failure_message": failure_message,
        "post_editorial_input": post_editorial_input,
        "prompt_render": prompt_render,
        "raw_response": raw_response,
        "quality_state": quality_state,
        "quality_invocation_count": invocation_count,
    }


def _semantic_error(
    *,
    failure_code: str,
    failure_stage: str,
    failure_message: str,
    grounding_state: FinalPostSemanticGroundingState,
    post_editorial_input: Any | None,
    prompt_render: Any | None,
    raw_response: Any | None,
    invocation_count: int,
) -> dict[str, Any]:
    return {
        "failure_code": failure_code,
        "failure_stage": failure_stage,
        "failure_message": failure_message,
        "grounding_state": grounding_state,
        "post_editorial_input": post_editorial_input,
        "prompt_render": prompt_render,
        "raw_response": raw_response,
        "semantic_grounding_invocation_count": invocation_count,
    }


def _selected_evidence_for_repair(
    post_brief: object | dict,
    selected_evidence_ids: tuple[str, ...],
) -> tuple[dict[str, str], ...]:
    evidence_to_use = _get_field(post_brief, "evidence_to_use")
    if not isinstance(evidence_to_use, (list, tuple)):
        raise ValueError("PostBrief.evidence_to_use must be a list or tuple.")
    evidence_by_id = {
        _get_field(item, "evidence_id"): item
        for item in evidence_to_use
        if _get_field(item, "evidence_id") is not None
    }
    selected = []
    for evidence_id in selected_evidence_ids:
        if evidence_id not in evidence_by_id:
            raise ValueError(
                f"PostBrief.evidence_to_use missing selected evidence ID: {evidence_id}"
            )
        item = evidence_by_id[evidence_id]
        selected.append(
            {
                "evidence_id": _require_string(item, "evidence_id"),
                "evidence_text": _require_string(item, "evidence_text"),
                "role_in_post": _require_string(item, "role_in_post"),
            }
        )
    return tuple(selected)


def _deterministic_findings(initial_result: Any) -> dict[str, Any]:
    gate = initial_result.deterministic_gate_output
    diagnostics = gate.diagnostics
    return {
        "validation_passed": gate.validation_passed,
        "validation_error": gate.validation_error,
        "system_linkedin_ready": diagnostics.system_linkedin_ready,
        "deterministic_checks_passed": diagnostics.deterministic_checks_passed,
        "repair_reasons": list(diagnostics.repair_reasons),
    }


def _quality_findings(initial_result: Any) -> dict[str, Any]:
    quality_state = getattr(initial_result, "quality_evaluation_state", None)
    quality_review = getattr(quality_state, "quality_review", None)
    if not isinstance(quality_review, dict):
        return {
            "status": getattr(quality_state, "status", "not_run"),
            "pass": None,
            "failed_criteria": [],
            "notes": [],
        }
    scores = quality_review.get("scores") or {}
    failed_minimums = {
        criterion: scores.get(criterion)
        for criterion, minimum in REQUIRED_QUALITY_MINIMUMS.items()
        if int(scores.get(criterion) or 0) < minimum
    }
    return {
        "scores": copy.deepcopy(scores),
        "total_score": quality_review.get("total_score"),
        "pass": quality_review.get("pass"),
        "failed_criteria": copy.deepcopy(quality_review.get("failed_criteria") or []),
        "failed_minimums": failed_minimums,
        "automatic_fail_reason": quality_review.get("automatic_fail_reason", ""),
        "notes": copy.deepcopy(quality_review.get("notes") or []),
    }


def _repair_instruction(initial_result: Any) -> dict[str, Any]:
    decision = initial_result.final_attempt_outcome.decision
    if decision.repair_type == "semantic_grounding":
        grounding_review = getattr(
            getattr(initial_result, "semantic_grounding_state", None),
            "grounding_review",
            None,
        )
        return {
            "repair_type": "semantic_grounding",
            "failed_claim_ids": (
                list(getattr(grounding_review, "blocking_claim_ids", ()))
                if grounding_review is not None
                else []
            ),
            "repair_scope": "CandidatePost.post_text",
            "repair_instruction": _semantic_grounding_repair_instruction(
                grounding_review,
                decision.reason,
            ),
            "decision_reason": decision.reason,
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
        }
    failed_criterion = _primary_failed_criterion(initial_result)
    return {
        "repair_type": "editorial",
        "failed_criterion": failed_criterion,
        "repair_scope": "CandidatePost.post_text",
        **_repair_target_metadata(failed_criterion),
        "repair_instruction": initial_result.final_attempt_outcome.decision.reason,
        "preserve": [
            "selected evidence only",
            "AngleDecision.controlling_angle",
            "distinctive original phrasing unless it is the failed criterion",
            "unrelated successful sentences as closely as possible",
            "valid CandidatePost JSON",
        ],
        "avoid": [
            "full-post rewrite when a local repair is enough",
            "generic author markers or template transitions",
            "appending when replacement can fix the target",
            "new facts",
            "new metrics",
            "evidence IDs in human-facing text",
            "scaffold/source-summary phrasing",
        ],
    }


def _repair_target_metadata(failed_criterion: str) -> dict[str, Any]:
    if failed_criterion == "cta":
        return {
            "target_locality": "ending_local",
            "allowed_edit_region": "final reader-facing turn",
            "replacement_preference": "replace_or_sharpen_existing_ending_do_not_append",
            "target_success_contract": "one clear reader-facing action or open reflective question",
        }
    if failed_criterion == "author_point_of_view":
        return {
            "target_locality": "sentence_local",
            "allowed_edit_region": "one local interpretive sentence or clause",
            "replacement_preference": "replace_or_tighten_one_local_sentence",
            "target_success_contract": "exactly one evidence-bounded interpretive judgment",
        }
    return {
        "target_locality": "local_to_failed_criterion",
        "allowed_edit_region": "minimal text needed for the failed criterion",
        "replacement_preference": "replace_before_appending",
        "target_success_contract": "fix the named failed criterion only",
    }


def _semantic_grounding_repair_instruction(
    grounding_review: Any | None,
    fallback_reason: str,
) -> str:
    instructions = (
        list(getattr(grounding_review, "repair_instructions", ()) or ())
        if grounding_review is not None
        else []
    )
    if instructions:
        return "; ".join(instruction.instruction for instruction in instructions)
    return fallback_reason


def _primary_failed_criterion(initial_result: Any) -> str:
    failed = (
        initial_result.quality_evaluation_state.quality_review.get("failed_criteria")
        or []
    )
    if failed:
        return str(failed[0])
    return "quality"


def _gate_passed(gate_output: Any) -> bool:
    diagnostics = gate_output.diagnostics
    return (
        gate_output.validation_passed
        and diagnostics.deterministic_checks_passed
        and diagnostics.system_linkedin_ready
    )


def _gate_failure_message(gate_output: Any) -> str:
    if gate_output.validation_error:
        return gate_output.validation_error
    if gate_output.diagnostics.repair_reasons:
        return ", ".join(gate_output.diagnostics.repair_reasons)
    return "deterministic gate failed"


def _repair_writer_request_error(request: Any) -> str | None:
    if not request.provider:
        return "missing repair writer provider"
    if not request.model:
        return "missing repair writer model"
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_REPAIR_WRITER,
        provider=request.provider,
        model=request.model,
    )
    if policy_failure is not None:
        return str(policy_failure)
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid repair writer max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid repair writer max_output_tokens: must be a positive integer"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing repair writer prompt text"
    rendered_input_text = request.rendered_prompt_input.input_text
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing repair writer rendered input text"
    return None


def _controlled_repair_role_selections(
    request: FinalPostControlledRepairRequest,
    *,
    candidate_writer_client: Any | None,
    semantic_grounding_client: Any | None,
    quality_evaluator_client: Any | None,
    repair_writer_client: Any | None,
) -> tuple[FinalPostExecutionRoleSelection, ...]:
    initial = request.initial_attempt_request
    return (
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_CANDIDATE_WRITER,
            provider=initial.candidate_writer_provider,
            model=initial.candidate_writer_model,
            stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_code=FAILURE_CANDIDATE_WRITER_REQUEST,
            stage_label="candidate writer",
            validate_key=candidate_writer_client is None,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
            provider=resolve_semantic_grounding_execution_provider(
                initial.semantic_grounding_provider
            ),
            model=resolve_semantic_grounding_execution_model(
                initial.semantic_grounding_provider,
                initial.semantic_grounding_model,
            ),
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            stage_label="semantic grounding",
            validate_key=semantic_grounding_client is None,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
            provider=initial.quality_evaluator_provider,
            model=initial.quality_evaluator_model,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            stage_label="quality evaluator",
            validate_key=quality_evaluator_client is None,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_REPAIR_WRITER,
            provider=request.repair_provider,
            model=request.repair_model,
            stage="repair_writer_request",
            failure_code=FAILURE_REPAIR_WRITER_REQUEST,
            stage_label="repair writer",
            validate_key=repair_writer_client is None,
        ),
    )


def _controlled_preflight_failure_result(
    request: FinalPostControlledRepairRequest,
    failure: Any,
) -> FinalPostControlledRepairResult:
    initial_result = FinalPostStandaloneAttemptResult(
        request=request.initial_attempt_request,
        stage_statuses=(
            FinalPostAttemptStageStatus(
                stage=STAGE_CANDIDATE_WRITER_REQUEST,
                status=STATUS_SKIPPED,
            ),
            FinalPostAttemptStageStatus(
                stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
                status=STATUS_SKIPPED,
            ),
            FinalPostAttemptStageStatus(
                stage=STAGE_QUALITY_EVALUATOR_REQUEST,
                status=STATUS_SKIPPED,
            ),
        ),
        failure_stage=failure.stage,
        failure_code=failure.failure_code,
        failure_message=failure.message,
    )
    return FinalPostControlledRepairResult(
        request=request,
        initial_attempt_result=initial_result,
        repair_eligibility=_ineligible("execution plan preflight failed"),
        repair_executed=False,
        terminal_reason="execution plan preflight failed",
        failure_stage=failure.stage,
        failure_code=failure.failure_code,
        failure_message=failure.message,
    )


def _quality_evaluator_request_error(request: Any) -> str | None:
    return get_quality_evaluator_execution_request_error(request)


def _repair_writer_execution_failure_code(raw_response: Any) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_REPAIR_WRITER_EMPTY_RESPONSE
    return FAILURE_REPAIR_WRITER_PROVIDER


def _repaired_semantic_execution_failure_code(raw_response: Any) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_REPAIRED_SEMANTIC_GROUNDING_EMPTY_RESPONSE
    return FAILURE_REPAIRED_SEMANTIC_GROUNDING_PROVIDER


def _repaired_quality_execution_failure_code(raw_response: Any) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_REPAIRED_QUALITY_EVALUATOR_EMPTY_RESPONSE
    return FAILURE_REPAIRED_QUALITY_EVALUATOR_PROVIDER


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


def _quality_review_needs_human_review(quality_review: dict[str, Any]) -> bool:
    if quality_review.get("requires_human_review") is True:
        return True
    return any(quality_review.get(flag) is True for flag in HUMAN_REVIEW_FLAGS)


def _terminal_repair_policy(policy: Any) -> FinalPostDecisionPolicy:
    allow_alternative_model = (
        policy.allow_alternative_model
        if isinstance(policy, FinalPostDecisionPolicy)
        else bool(
            policy.get("allow_alternative_model")
            if isinstance(policy, dict)
            else False
        )
    )
    return FinalPostDecisionPolicy(
        max_total_attempts=2,
        max_mechanical_repairs=0,
        max_editorial_repairs=0,
        allow_alternative_model=allow_alternative_model,
    )


def _get_field(value: object | dict, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)


def _require_string(value: object | dict, field_name: str) -> str:
    field_value = _get_field(value, field_name)
    if not isinstance(field_value, str) or not field_value.strip():
        raise ValueError(f"PostBrief.evidence_to_use.{field_name} must be non-empty.")
    return field_value
