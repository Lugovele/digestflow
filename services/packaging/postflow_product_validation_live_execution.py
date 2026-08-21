"""Benchmark-only live execution helpers for PostFlow product validation.

This module is not production packaging runtime. It records the outcome of the
current frozen stage contracts for benchmark cases and keeps historical labels
out of runtime inputs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from django.conf import settings

from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_grounding_and_quality,
)
from services.packaging.linkedin_post_controlled_repair_contract import (
    FAILURE_REPAIRED_DETERMINISTIC_GATE,
    FAILURE_REPAIRED_SEMANTIC_GROUNDING,
    FAILURE_REPAIR_TARGET_NOT_FIXED,
    FinalPostControlledRepairRequest,
)
from services.packaging.linkedin_post_controlled_repair_execution import (
    continue_final_post_controlled_repair_attempt,
)
from services.packaging.linkedin_post_deterministic_gate import (
    run_candidate_post_deterministic_gate,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    STAGE_ATTEMPT_ADJUDICATION,
    STAGE_ATTEMPT_OUTCOME,
    STAGE_DETERMINISTIC_GATE,
    STAGE_QUALITY_EVALUATOR_EXECUTION,
    STAGE_QUALITY_EVALUATOR_PARSE,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_QUALITY_REVIEW_NORMALIZATION,
    STAGE_SEMANTIC_GROUNDING_EXECUTION,
    STAGE_SEMANTIC_GROUNDING_NORMALIZATION,
    STAGE_SEMANTIC_GROUNDING_PARSE,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
    STATUS_SUCCEEDED,
    FinalPostAttemptRequest,
    FinalPostAttemptStageStatus,
    FinalPostStandaloneAttemptResult,
)
from services.packaging.linkedin_post_flow_contracts import FinalPostAttemptHistory
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput
from services.packaging.linkedin_post_flow_input_builders import build_post_editorial_input
from services.packaging.linkedin_post_prompt_renderers import (
    render_quality_evaluator_prompt_input,
    render_semantic_grounding_prompt_input,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    build_quality_evaluator_execution_request,
    execute_quality_evaluator_prompt,
)
from services.packaging.linkedin_post_quality_evaluator_parser import (
    QualityEvaluatorResponseParseError,
    parse_and_normalize_quality_evaluator_response,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_repair_writer_benchmark import (
    REPAIR_PROMPT_TEXT,
    REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    _payload_preservation as repair_writer_payload_preservation,
)
from services.packaging.linkedin_post_semantic_grounding_boundary_benchmark import (
    RUN_STATUS_BLOCK,
    RUN_STATUS_PASS,
    SemanticGroundingBoundaryRequest,
    _case_from_payload as boundary_case_from_payload,
    _live_run_record as live_boundary_run_record,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
    GROUNDING_STATUS_PASS,
    FinalPostSemanticGroundingState,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    build_semantic_grounding_execution_request,
    execute_semantic_grounding_prompt,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    SemanticGroundingResponseParseError,
    parse_and_normalize_semantic_grounding_response,
)
from services.packaging.postflow_product_validation_corpus import (
    CASE_FAMILY_CANDIDATE_QE,
    CASE_FAMILY_SEMANTIC_BOUNDARY,
    CASE_FAMILY_SOURCE_ARTICLE,
    FAILURE_CATEGORY_INFRASTRUCTURE,
    FAILURE_CATEGORY_PRODUCT,
    FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
)


LIVE_ACCEPTED_FIRST_ATTEMPT = "LIVE_ACCEPTED_FIRST_ATTEMPT"
LIVE_REPAIR_ACCEPTED = "LIVE_REPAIR_ACCEPTED"
LIVE_REPAIR_REQUIRED_NOT_EXECUTED = "LIVE_REPAIR_REQUIRED_NOT_EXECUTED"
LIVE_REPAIR_TARGET_NOT_FIXED = "LIVE_REPAIR_TARGET_NOT_FIXED"
LIVE_REPAIR_REGRESSION = "LIVE_REPAIR_REGRESSION"
LIVE_REPAIR_EXCESSIVE_REWRITE = "LIVE_REPAIR_EXCESSIVE_REWRITE"
LIVE_DETERMINISTIC_BLOCK = "LIVE_DETERMINISTIC_BLOCK"
LIVE_GROUNDING_BLOCK = "LIVE_GROUNDING_BLOCK"
LIVE_GROUNDING_INFRA_FAILURE = "LIVE_GROUNDING_INFRA_FAILURE"
LIVE_QE_REJECTED = "LIVE_QE_REJECTED"
LIVE_QE_INFRA_FAILURE = "LIVE_QE_INFRA_FAILURE"
LIVE_OVER_LENGTH = "LIVE_OVER_LENGTH"
LIVE_PARSER_OR_ADAPTER_FAILURE = "LIVE_PARSER_OR_ADAPTER_FAILURE"
LIVE_HUMAN_REVIEW = "LIVE_HUMAN_REVIEW"
LIVE_SOURCE_READY = "LIVE_SOURCE_READY"
LIVE_BOUNDARY_VALID_PRESERVED = "LIVE_BOUNDARY_VALID_PRESERVED"
LIVE_BOUNDARY_INVALID_BLOCKED = "LIVE_BOUNDARY_INVALID_BLOCKED"
LIVE_BOUNDARY_FALSE_POSITIVE = "LIVE_BOUNDARY_FALSE_POSITIVE"
LIVE_BOUNDARY_FALSE_NEGATIVE = "LIVE_BOUNDARY_FALSE_NEGATIVE"
LIVE_OTHER_INFRA_FAILURE = "LIVE_OTHER_INFRA_FAILURE"

LIVE_EXECUTION_COMPLETED = "live_completed"
LIVE_EXECUTION_FAILED = "live_failed"

PRODUCT_CORPUS_MINIMUM_CASES = 12
PRODUCT_CORPUS_PREFERRED_CASES = 20
SYSTEMIC_FAILURE_MIN_CASES = 2
SEMANTIC_GROUNDING_PROMPT_PATH = "prompts/linkedin/final_post_semantic_grounding_evaluator.txt"
QUALITY_EVALUATOR_PROMPT_PATH = "prompts/linkedin/final_post_quality_evaluator.txt"
FIXED_REPAIR_WRITER_PROVIDER = "gemini"
FIXED_REPAIR_WRITER_MODEL = "gemini-3.6-flash"

RoleExecutor = Callable[[Any], Any]


@dataclass(frozen=True)
class ProductValidationLiveExecutors:
    semantic_grounding_executor: RoleExecutor | None = None
    quality_evaluator_executor: RoleExecutor | None = None
    repair_writer_executor: RoleExecutor | None = None


def execute_product_validation_live_case(
    *,
    case: Any,
    fixture_payload: dict[str, Any],
    experiment_id: str,
    started_at: str,
    completed_at: str,
    executors: ProductValidationLiveExecutors | None = None,
) -> dict[str, Any]:
    if case.family == CASE_FAMILY_SOURCE_ARTICLE:
        return _source_record()
    if case.family == CASE_FAMILY_SEMANTIC_BOUNDARY:
        return _boundary_record(
            case=case,
            fixture_payload=fixture_payload,
            experiment_id=experiment_id,
            started_at=started_at,
            completed_at=completed_at,
            executors=executors or ProductValidationLiveExecutors(),
        )
    if case.family == CASE_FAMILY_CANDIDATE_QE:
        return _product_record(
            fixture_payload=fixture_payload,
            executors=executors or ProductValidationLiveExecutors(),
        )
    return _infra_failure("unsupported_case_family", "case family is not executable")


def planned_provider_calls_for_case_family(family: str) -> dict[str, int]:
    if family == CASE_FAMILY_SOURCE_ARTICLE:
        return _provider_counts()
    if family == CASE_FAMILY_SEMANTIC_BOUNDARY:
        return _provider_counts(semantic_grounding=1)
    if family == CASE_FAMILY_CANDIDATE_QE:
        return _provider_counts(
            semantic_grounding=2,
            quality_evaluator=2,
            repair_writer=1,
        )
    return _provider_counts()


def runtime_input_for_case(case: Any, fixture_payload: dict[str, Any]) -> dict[str, Any]:
    """Return canonical runtime inputs, intentionally excluding historical labels."""

    if case.family == CASE_FAMILY_CANDIDATE_QE:
        return {
            "candidate_payload": copy.deepcopy(fixture_payload.get("candidate_payload")),
            "post_brief": copy.deepcopy(fixture_payload.get("post_brief")),
            "angle_decision": copy.deepcopy(fixture_payload.get("angle_decision")),
            "selected_evidence": copy.deepcopy(fixture_payload.get("selected_evidence")),
        }
    if case.family == CASE_FAMILY_SEMANTIC_BOUNDARY:
        return {
            "candidate_post": fixture_payload.get("candidate_post"),
            "post_brief": copy.deepcopy(fixture_payload.get("post_brief")),
            "angle_decision": copy.deepcopy(fixture_payload.get("angle_decision")),
            "evidence": copy.deepcopy(fixture_payload.get("evidence")),
        }
    return {"fixture_path": str(case.fixture_path)}


def _source_record() -> dict[str, Any]:
    return {
        "live_execution_status": LIVE_EXECUTION_COMPLETED,
        "live_outcome": LIVE_SOURCE_READY,
        "live_failure_category": FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
        "live_failure_code": None,
        "live_failure_stage": None,
        "live_stage_outcomes": {"source": "ready"},
        "provider_invocation_counts": _provider_counts(),
        "total_provider_calls": 0,
        "publication_packaging_invocations": 0,
    }


def _boundary_record(
    *,
    case: Any,
    fixture_payload: dict[str, Any],
    experiment_id: str,
    started_at: str,
    completed_at: str,
    executors: ProductValidationLiveExecutors,
) -> dict[str, Any]:
    try:
        boundary_case = boundary_case_from_payload(
            fixture_payload,
            fixture_path=Path(case.fixture_path),
        )
        plan = _semantic_grounding_plan()
        request = SemanticGroundingBoundaryRequest(
            experiment_id=experiment_id,
            cases=(boundary_case,),
            plans=(plan,),
            allow_api=True,
        )
        run = live_boundary_run_record(
            request,
            boundary_case,
            plan,
            1,
            started_at,
            completed_at,
            render=_build_boundary_render(boundary_case),
            semantic_grounding_executor=executors.semantic_grounding_executor,
        )
    except Exception as exc:
        return _infra_failure("boundary_execution_failed", str(exc))

    status = run.get("execution_status")
    expected_blocking = bool(run.get("expected_blocking"))
    grounding_pass = bool(run.get("grounding_pass"))
    if status == RUN_STATUS_PASS and not expected_blocking:
        outcome = LIVE_BOUNDARY_VALID_PRESERVED
        category = FAILURE_CATEGORY_PRODUCT_BEHAVIOR
    elif status == RUN_STATUS_BLOCK and expected_blocking:
        outcome = LIVE_BOUNDARY_INVALID_BLOCKED
        category = FAILURE_CATEGORY_PRODUCT_BEHAVIOR
    elif status == RUN_STATUS_PASS and expected_blocking:
        outcome = LIVE_BOUNDARY_FALSE_NEGATIVE
        category = FAILURE_CATEGORY_PRODUCT
    elif status == RUN_STATUS_BLOCK and not expected_blocking:
        outcome = LIVE_BOUNDARY_FALSE_POSITIVE
        category = FAILURE_CATEGORY_PRODUCT
    else:
        outcome = LIVE_GROUNDING_INFRA_FAILURE
        category = FAILURE_CATEGORY_INFRASTRUCTURE

    counts = _provider_counts(
        semantic_grounding=int(
            run.get("provider_invocation_counts", {}).get("semantic_grounding", 0)
        )
    )
    return {
        "live_execution_status": LIVE_EXECUTION_COMPLETED,
        "live_outcome": outcome,
        "live_failure_category": category,
        "live_failure_code": run.get("failure_code"),
        "live_failure_stage": run.get("failure_stage"),
        "live_stage_outcomes": {
            "semantic_grounding": status,
            "grounding_pass": grounding_pass,
        },
        "boundary_run_summary": _safe_boundary_summary(run),
        "provider_invocation_counts": counts,
        "total_provider_calls": sum(counts.values()),
        "publication_packaging_invocations": 0,
    }


def _product_record(
    *,
    fixture_payload: dict[str, Any],
    executors: ProductValidationLiveExecutors,
) -> dict[str, Any]:
    try:
        candidate_payload = copy.deepcopy(fixture_payload["candidate_payload"])
        post_brief = copy.deepcopy(fixture_payload["post_brief"])
        angle_decision = copy.deepcopy(fixture_payload["angle_decision"])
        selected_evidence = tuple(copy.deepcopy(fixture_payload["selected_evidence"]))
        selected_evidence_ids = tuple(item["evidence_id"] for item in selected_evidence)
        candidate_output = CandidateWriterOutput(
            payload=candidate_payload,
            raw_output=None,
            provider=fixture_payload.get("writer_provider"),
            model=fixture_payload.get("writer_model"),
            prompt_name="historical_candidate_payload",
            prompt_version=fixture_payload.get("source_experiment_id"),
            token_usage=None,
            cost_metadata=None,
        )
        gate_output = run_candidate_post_deterministic_gate(
            candidate_output,
            selected_evidence_ids=selected_evidence_ids,
        )
    except Exception as exc:
        return _infra_failure("product_input_invalid", str(exc), stage="product_input")

    if not gate_output.validation_passed or not gate_output.diagnostics.deterministic_checks_passed:
        return {
            "live_execution_status": LIVE_EXECUTION_COMPLETED,
            "live_outcome": (
                LIVE_OVER_LENGTH
                if "post_text" in str(gate_output.validation_error)
                and "length" in str(gate_output.validation_error).lower()
                else LIVE_DETERMINISTIC_BLOCK
            ),
            "live_failure_category": FAILURE_CATEGORY_PRODUCT,
            "live_failure_code": "deterministic_gate_failed",
            "live_failure_stage": "deterministic_gate",
            "live_stage_outcomes": {"deterministic_gate": gate_output.to_dict()},
            "provider_invocation_counts": _provider_counts(),
            "total_provider_calls": 0,
            "publication_packaging_invocations": 0,
            "final_candidate": copy.deepcopy(candidate_payload),
        }

    try:
        post_editorial_input = build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=angle_decision,
            candidate_output=candidate_output,
            gate_output=gate_output,
        )
    except Exception as exc:
        return _infra_failure(
            "post_editorial_input_failed",
            str(exc),
            stage="post_editorial_input",
        )

    grounding_result = _run_product_grounding(
        post_editorial_input=post_editorial_input,
        selected_evidence_ids=selected_evidence_ids,
        executor=executors.semantic_grounding_executor,
    )
    if grounding_result["failure_code"] is not None:
        return _with_counts(
            _infra_failure(
                grounding_result["failure_code"],
                grounding_result["failure_message"],
                stage=grounding_result["failure_stage"],
                outcome=LIVE_GROUNDING_INFRA_FAILURE,
            ),
            semantic_grounding=grounding_result["calls"],
        )
    semantic_state = grounding_result["semantic_state"]
    if semantic_state.status != GROUNDING_STATUS_PASS:
        return _with_counts(
            {
                "live_execution_status": LIVE_EXECUTION_COMPLETED,
                "live_outcome": (
                    LIVE_HUMAN_REVIEW
                    if semantic_state.status == GROUNDING_STATUS_NEEDS_HUMAN_REVIEW
                    else LIVE_GROUNDING_BLOCK
                ),
                "live_failure_category": FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
                "live_failure_code": "semantic_grounding_block",
                "live_failure_stage": "semantic_grounding",
                "live_stage_outcomes": {
                    "deterministic_gate": "pass",
                    "semantic_grounding": semantic_state.to_dict(),
                },
                "publication_packaging_invocations": 0,
                "final_candidate": copy.deepcopy(candidate_payload),
            },
            semantic_grounding=grounding_result["calls"],
        )

    quality_result = _run_product_quality(
        post_editorial_input=post_editorial_input,
        executor=executors.quality_evaluator_executor,
    )
    if quality_result["failure_code"] is not None:
        return _with_counts(
            _infra_failure(
                quality_result["failure_code"],
                quality_result["failure_message"],
                stage=quality_result["failure_stage"],
                outcome=LIVE_QE_INFRA_FAILURE,
            ),
            semantic_grounding=grounding_result["calls"],
            quality_evaluator=quality_result["calls"],
        )

    quality_state = FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_READY,
        quality_review=quality_result["quality_review"],
    )
    attempt_request = _product_attempt_request()
    attempt_outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
        post_brief=post_brief,
        candidate_output=candidate_output,
        gate_output=gate_output,
        semantic_grounding=semantic_state,
        quality_evaluation=quality_state,
        attempt_index=attempt_request.attempt_index,
        attempt_history=attempt_request.attempt_history,
        policy=attempt_request.policy,
        alternative_model_available=attempt_request.alternative_model_available,
        target_model_provider=attempt_request.target_model_provider,
        target_model_name=attempt_request.target_model_name,
        angle_decision=angle_decision,
    )
    accepted = attempt_outcome.accepted_result is not None
    if attempt_outcome.repair_required:
        initial_result = _product_initial_attempt_result(
            request=attempt_request,
            candidate_output=candidate_output,
            gate_output=gate_output,
            post_editorial_input=post_editorial_input,
            semantic_state=semantic_state,
            quality_state=quality_state,
            attempt_outcome=attempt_outcome,
        )
        repair_result = continue_final_post_controlled_repair_attempt(
            FinalPostControlledRepairRequest(
                initial_attempt_request=attempt_request,
                repair_prompt_text=REPAIR_PROMPT_TEXT,
                repair_provider=FIXED_REPAIR_WRITER_PROVIDER,
                repair_model=FIXED_REPAIR_WRITER_MODEL,
                repair_max_output_tokens=REPAIR_WRITER_MAX_OUTPUT_TOKENS,
                execution_metadata={
                    "product_validation_role": "repair_writer",
                    "repair_writer_execution_profile": "gemini_repair_minimal_reasoning",
                    "repair_writer_reasoning_effort": "minimal",
                },
            ),
            initial_result=initial_result,
            post_brief=post_brief,
            angle_decision=angle_decision,
            selected_evidence_ids=selected_evidence_ids,
            semantic_grounding_executor=executors.semantic_grounding_executor,
            quality_evaluator_executor=executors.quality_evaluator_executor,
            repair_writer_executor=executors.repair_writer_executor,
        )
        return _product_repair_record(
            candidate_payload=candidate_payload,
            initial_semantic_state=semantic_state,
            initial_quality_state=quality_state,
            initial_attempt_outcome=attempt_outcome,
            repair_result=repair_result,
        )

    return _with_counts(
        {
            "live_execution_status": LIVE_EXECUTION_COMPLETED,
            "live_outcome": LIVE_ACCEPTED_FIRST_ATTEMPT if accepted else LIVE_QE_REJECTED,
            "live_failure_category": (
                FAILURE_CATEGORY_PRODUCT_BEHAVIOR if accepted else FAILURE_CATEGORY_PRODUCT
            ),
            "live_failure_code": None if accepted else "quality_not_accepted",
            "live_failure_stage": None if accepted else "quality_adjudication",
            "live_stage_outcomes": {
                "deterministic_gate": "pass",
                "semantic_grounding": semantic_state.to_dict(),
                "quality_evaluation": quality_state.to_dict(),
                "adjudication": attempt_outcome.to_dict(),
            },
            "quality_review_summary": _quality_summary(quality_result["quality_review"]),
            "accepted_payload": copy.deepcopy(
                attempt_outcome.accepted_result.accepted_payload
                if attempt_outcome.accepted_result
                else None
            ),
            "final_candidate": copy.deepcopy(candidate_payload),
            "publication_packaging_invocations": 0,
        },
        semantic_grounding=grounding_result["calls"],
        quality_evaluator=quality_result["calls"],
    )


def _product_attempt_request() -> FinalPostAttemptRequest:
    return FinalPostAttemptRequest(
        candidate_writer_render={},
        candidate_writer_prompt_text="benchmark fixed candidate payload",
        quality_rubric=get_quality_evaluator_rubric_payload(),
        quality_evaluator_prompt_text=_prompt_text(QUALITY_EVALUATOR_PROMPT_PATH),
        attempt_index=0,
        max_attempts=2,
        attempt_history=FinalPostAttemptHistory(attempts=[]),
        candidate_writer_provider="anthropic",
        candidate_writer_model="claude-sonnet-5",
        semantic_grounding_prompt_text=_prompt_text(SEMANTIC_GROUNDING_PROMPT_PATH),
        semantic_grounding_provider="gemini",
        semantic_grounding_model="gemini-3.6-flash",
        semantic_grounding_max_output_tokens=4800,
        quality_evaluator_provider="openai",
        quality_evaluator_model="gpt-4.1-2025-04-14",
        quality_evaluator_max_output_tokens=2400,
        execution_metadata={"product_validation_role": "candidate_quality_case"},
    )


def _product_initial_attempt_result(
    *,
    request: FinalPostAttemptRequest,
    candidate_output: CandidateWriterOutput,
    gate_output: Any,
    post_editorial_input: Any,
    semantic_state: FinalPostSemanticGroundingState,
    quality_state: FinalPostQualityEvaluationState,
    attempt_outcome: Any,
) -> FinalPostStandaloneAttemptResult:
    return FinalPostStandaloneAttemptResult(
        request=request,
        stage_statuses=(
            _succeeded_stage(STAGE_DETERMINISTIC_GATE),
            _succeeded_stage(STAGE_SEMANTIC_GROUNDING_REQUEST),
            _succeeded_stage(STAGE_SEMANTIC_GROUNDING_EXECUTION),
            _succeeded_stage(STAGE_SEMANTIC_GROUNDING_PARSE),
            _succeeded_stage(STAGE_SEMANTIC_GROUNDING_NORMALIZATION),
            _succeeded_stage(STAGE_QUALITY_EVALUATOR_REQUEST),
            _succeeded_stage(STAGE_QUALITY_EVALUATOR_EXECUTION),
            _succeeded_stage(STAGE_QUALITY_EVALUATOR_PARSE),
            _succeeded_stage(STAGE_QUALITY_REVIEW_NORMALIZATION),
            _succeeded_stage(STAGE_ATTEMPT_ADJUDICATION),
            _succeeded_stage(STAGE_ATTEMPT_OUTCOME),
        ),
        completed_stage=STAGE_ATTEMPT_OUTCOME,
        candidate_writer_output=candidate_output,
        deterministic_gate_output=gate_output,
        post_editorial_input=post_editorial_input,
        semantic_grounding_state=semantic_state,
        quality_evaluation_state=quality_state,
        final_attempt_outcome=attempt_outcome,
        candidate_writer_invocation_count=0,
        semantic_grounding_invocation_count=1,
        quality_evaluator_invocation_count=1,
        audit_metadata={"product_validation_fixed_candidate": True},
    )


def _succeeded_stage(stage: str) -> FinalPostAttemptStageStatus:
    return FinalPostAttemptStageStatus(stage=stage, status=STATUS_SUCCEEDED)


def _repair_live_outcome_and_category(repair_result: Any) -> tuple[str, str]:
    if repair_result.accepted_payload:
        return LIVE_REPAIR_ACCEPTED, FAILURE_CATEGORY_PRODUCT_BEHAVIOR
    if repair_result.failure_code == FAILURE_REPAIR_TARGET_NOT_FIXED:
        return LIVE_REPAIR_TARGET_NOT_FIXED, FAILURE_CATEGORY_PRODUCT
    if repair_result.failure_code == FAILURE_REPAIRED_DETERMINISTIC_GATE:
        return LIVE_DETERMINISTIC_BLOCK, FAILURE_CATEGORY_PRODUCT
    if repair_result.failure_code == FAILURE_REPAIRED_SEMANTIC_GROUNDING:
        return LIVE_GROUNDING_BLOCK, FAILURE_CATEGORY_PRODUCT_BEHAVIOR
    if repair_result.failure_code:
        return LIVE_OTHER_INFRA_FAILURE, FAILURE_CATEGORY_INFRASTRUCTURE
    return LIVE_QE_REJECTED, FAILURE_CATEGORY_PRODUCT

def _product_repair_record(
    *,
    candidate_payload: dict[str, Any],
    initial_semantic_state: FinalPostSemanticGroundingState,
    initial_quality_state: FinalPostQualityEvaluationState,
    initial_attempt_outcome: Any,
    repair_result: Any,
) -> dict[str, Any]:
    repaired_payload = _repaired_payload(repair_result)
    final_candidate = repaired_payload or candidate_payload
    target_diagnostics = _repair_target_diagnostics(repair_result)
    preservation = _repair_preservation(candidate_payload, repaired_payload)
    accepted_payload = copy.deepcopy(repair_result.accepted_payload)
    accepted = accepted_payload is not None
    live_outcome, live_failure_category = _repair_live_outcome_and_category(repair_result)

    record = {
        "live_execution_status": LIVE_EXECUTION_COMPLETED,
        "live_outcome": live_outcome,
        "live_failure_category": live_failure_category,
        "live_failure_code": repair_result.failure_code,
        "live_failure_stage": repair_result.failure_stage,
        "live_failure_message": repair_result.failure_message,
        "live_stage_outcomes": {
            "deterministic_gate": "pass",
            "semantic_grounding": initial_semantic_state.to_dict(),
            "quality_evaluation": initial_quality_state.to_dict(),
            "adjudication": initial_attempt_outcome.to_dict(),
            "repair": repair_result.to_dict(),
        },
        "quality_review_summary": _quality_summary(_final_quality_review(repair_result, initial_quality_state)),
        "initial_quality_review_summary": _quality_summary(initial_quality_state.quality_review),
        "accepted_payload": accepted_payload,
        "final_candidate": copy.deepcopy(final_candidate),
        "repaired_candidate": copy.deepcopy(repaired_payload),
        "repair_target_enforcement": copy.deepcopy(target_diagnostics),
        "target_repair_success": target_diagnostics.get("repair_target_fixed"),
        "target_repair_failure_reason": target_diagnostics.get("repair_target_failure_reason", ""),
        "publication_packaging_invocations": 0,
    }
    if preservation:
        record.update(
            {
                "preservation_rate": preservation.get("distinctive_preservation_rate"),
                "changed_sentence_count": preservation.get("changed_sentence_count"),
                "generic_marker_count": preservation.get("added_generic_marker_count"),
                "generic_marker_introduced": bool(preservation.get("added_generic_marker_count")),
                "material_genericization": preservation.get("added_generic_marker_count", 0) > 0,
                "payload_preservation": preservation,
            }
        )
    return _with_counts(
        record,
        candidate_writer=repair_result.candidate_writer_invocation_count,
        semantic_grounding=repair_result.semantic_grounding_invocation_count,
        quality_evaluator=repair_result.quality_evaluator_invocation_count,
        repair_writer=repair_result.repair_invocation_count,
    )


def _repaired_payload(repair_result: Any) -> dict[str, Any] | None:
    output = getattr(repair_result, "repaired_candidate_output", None)
    payload = getattr(output, "payload", None)
    return copy.deepcopy(payload) if isinstance(payload, dict) else None


def _repair_target_diagnostics(repair_result: Any) -> dict[str, Any]:
    diagnostics = getattr(repair_result, "repair_target_enforcement_diagnostics", None)
    if diagnostics is not None and hasattr(diagnostics, "to_dict"):
        return diagnostics.to_dict()
    return {}


def _repair_preservation(
    original_payload: dict[str, Any],
    repaired_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(repaired_payload, dict):
        return None
    return repair_writer_payload_preservation(original_payload, repaired_payload)


def _final_quality_review(
    repair_result: Any,
    fallback_quality_state: FinalPostQualityEvaluationState,
) -> dict[str, Any]:
    repaired_quality_state = getattr(repair_result, "repaired_quality_evaluation_state", None)
    review = getattr(repaired_quality_state, "quality_review", None)
    if isinstance(review, dict):
        return review
    return fallback_quality_state.quality_review


def _run_product_grounding(
    *,
    post_editorial_input: Any,
    selected_evidence_ids: tuple[str, ...],
    executor: RoleExecutor | None,
) -> dict[str, Any]:
    render = render_semantic_grounding_prompt_input(post_editorial_input)
    request = build_semantic_grounding_execution_request(
        render,
        prompt_text=_prompt_text(render.prompt_path or SEMANTIC_GROUNDING_PROMPT_PATH),
        provider="gemini",
        model="gemini-3.6-flash",
        max_output_tokens=4800,
        json_mode=True,
        reasoning_effort="minimal",
        execution_metadata={"product_validation_role": "semantic_grounding"},
    )
    raw_response = (executor or execute_semantic_grounding_prompt)(request)
    if getattr(raw_response, "execution_error", None):
        return {
            "failure_code": "semantic_grounding_execution_failure",
            "failure_stage": "semantic_grounding_execution",
            "failure_message": str(raw_response.execution_error),
            "calls": 1,
        }
    try:
        review = parse_and_normalize_semantic_grounding_response(
            raw_response,
            selected_evidence_ids=selected_evidence_ids,
        )
    except SemanticGroundingResponseParseError as exc:
        return {
            "failure_code": "semantic_grounding_parse_or_normalization_failure",
            "failure_stage": "semantic_grounding_parse_or_normalization",
            "failure_message": str(exc),
            "calls": 1,
        }
    return {
        "failure_code": None,
        "failure_stage": None,
        "failure_message": "",
        "calls": 1,
        "semantic_state": FinalPostSemanticGroundingState(
            status=(
                GROUNDING_STATUS_PASS
                if review.passed
                else (
                    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW
                    if review.requires_human_review
                    else GROUNDING_STATUS_FAIL
                )
            ),
            grounding_review=review,
        ),
    }


def _run_product_quality(
    *,
    post_editorial_input: Any,
    executor: RoleExecutor | None,
) -> dict[str, Any]:
    render = render_quality_evaluator_prompt_input(
        post_editorial_input,
        get_quality_evaluator_rubric_payload(),
    )
    request = build_quality_evaluator_execution_request(
        render,
        prompt_text=_prompt_text(render.prompt_path or QUALITY_EVALUATOR_PROMPT_PATH),
        provider="openai",
        model="gpt-4.1-2025-04-14",
        execution_metadata={"product_validation_role": "quality_evaluator"},
    )
    raw_response = (executor or execute_quality_evaluator_prompt)(request)
    if getattr(raw_response, "execution_error", None):
        return {
            "failure_code": "quality_evaluator_execution_failure",
            "failure_stage": "quality_evaluator_execution",
            "failure_message": str(raw_response.execution_error),
            "calls": 1,
        }
    try:
        review = parse_and_normalize_quality_evaluator_response(raw_response)
    except QualityEvaluatorResponseParseError as exc:
        return {
            "failure_code": "quality_evaluator_parse_or_normalization_failure",
            "failure_stage": "quality_evaluator_parse_or_normalization",
            "failure_message": str(exc),
            "calls": 1,
        }
    return {
        "failure_code": None,
        "failure_stage": None,
        "failure_message": "",
        "calls": 1,
        "quality_review": review,
    }


def _semantic_grounding_plan():
    from services.packaging.linkedin_post_semantic_grounding_boundary_benchmark import (
        SemanticGroundingBoundaryPlan,
    )

    return SemanticGroundingBoundaryPlan(
        plan_id="frozen_product_validation_semantic_grounding",
        provider="gemini",
        model="gemini-3.6-flash",
        max_output_tokens=4800,
        json_mode=True,
        execution_profile="grounding_minimal_reasoning",
    )


def _build_boundary_render(boundary_case: Any) -> Any:
    from services.packaging.linkedin_post_semantic_grounding_boundary_benchmark import (
        build_boundary_semantic_grounding_prompt_render,
    )

    return build_boundary_semantic_grounding_prompt_render(boundary_case)


def _prompt_text(prompt_path: str | None) -> str:
    if not prompt_path:
        raise ValueError("prompt path is required for live product validation")
    base_dir = Path(settings.BASE_DIR).resolve()
    path = (base_dir / prompt_path).resolve()
    if not _is_relative_to(path, base_dir):
        raise ValueError("prompt path must stay inside repository")
    return path.read_text(encoding="utf-8")


def _safe_boundary_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_status": run.get("execution_status"),
        "grounding_pass": run.get("grounding_pass"),
        "expected_blocking": run.get("expected_blocking"),
        "failure_stage": run.get("failure_stage"),
        "failure_code": run.get("failure_code"),
        "blocking_claim_count": run.get("blocking_claim_count"),
    }


def _quality_summary(review: dict[str, Any]) -> dict[str, Any]:
    return {
        "pass": review.get("pass"),
        "total_score": review.get("total_score"),
        "failed_criteria": copy.deepcopy(review.get("failed_criteria", [])),
        "automatic_fail_reason": review.get("automatic_fail_reason", ""),
        "scores": copy.deepcopy(review.get("scores", {})),
    }


def _infra_failure(
    code: str,
    message: str,
    *,
    stage: str = "live_execution",
    outcome: str = LIVE_OTHER_INFRA_FAILURE,
) -> dict[str, Any]:
    return {
        "live_execution_status": LIVE_EXECUTION_FAILED,
        "live_outcome": outcome,
        "live_failure_category": FAILURE_CATEGORY_INFRASTRUCTURE,
        "live_failure_code": code,
        "live_failure_stage": stage,
        "live_failure_message": str(message),
        "live_stage_outcomes": {},
        "provider_invocation_counts": _provider_counts(),
        "total_provider_calls": 0,
        "publication_packaging_invocations": 0,
    }


def _with_counts(
    record: dict[str, Any],
    *,
    candidate_writer: int = 0,
    semantic_grounding: int = 0,
    quality_evaluator: int = 0,
    repair_writer: int = 0,
) -> dict[str, Any]:
    counts = _provider_counts(
        candidate_writer=candidate_writer,
        semantic_grounding=semantic_grounding,
        quality_evaluator=quality_evaluator,
        repair_writer=repair_writer,
    )
    result = copy.deepcopy(record)
    result["provider_invocation_counts"] = counts
    result["total_provider_calls"] = sum(counts.values())
    result["publication_packaging_invocations"] = 0
    return result


def _provider_counts(
    *,
    candidate_writer: int = 0,
    semantic_grounding: int = 0,
    quality_evaluator: int = 0,
    repair_writer: int = 0,
) -> dict[str, int]:
    return {
        "candidate_writer_provider_api_calls": candidate_writer,
        "semantic_grounding_provider_api_calls": semantic_grounding,
        "quality_evaluator_provider_api_calls": quality_evaluator,
        "repair_writer_provider_api_calls": repair_writer,
        "publication_packaging_invocations": 0,
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False
