"""Manual smoke runner for final LinkedIn post provider flows.

This module builds explicit smoke-test inputs and delegates to the existing
standalone / controlled-repair orchestration APIs. It does not persist data,
touch ContentPackage records, import production packaging runtime, or duplicate
candidate parsing, gating, quality evaluation, repair, or adjudication logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import is_dataclass
from dataclasses import asdict
import copy
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any

from django.conf import settings

from services.packaging.linkedin_post_controlled_repair_contract import (
    FAILURE_REPAIR_WRITER_REQUEST,
    FinalPostControlledRepairRequest,
)
from services.packaging.linkedin_post_controlled_repair_execution import (
    execute_final_post_controlled_repair_attempt,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
    structural_diagnostics_from_dict,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE,
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_QUALITY_EVALUATOR_REQUEST,
    FAILURE_SEMANTIC_GROUNDING_REQUEST,
    FinalPostAttemptRequest,
    STAGE_CANDIDATE_WRITER_EXECUTION,
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
)
from services.packaging.linkedin_post_final_post_attempt_execution import (
    execute_final_post_standalone_attempt,
)
from services.packaging.linkedin_post_final_post_execution_plan import (
    FinalPostExecutionPlanPreflightResult,
    FinalPostExecutionRoleSelection,
    preflight_final_post_execution_plan,
)
from services.packaging.linkedin_post_flow_decision import FinalPostDecisionPolicy
from services.packaging.linkedin_post_flow_input_builders import (
    build_candidate_writer_input,
)
from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    get_prompt_contract,
    prompt_contract_to_prompt_metadata,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_candidate_writer_prompt_input,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    DEFAULT_MAX_OUTPUT_TOKENS as DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
)
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
)


SMOKE_MODE_STANDALONE = "standalone"
SMOKE_MODE_CONTROLLED_REPAIR = "controlled-repair"
SMOKE_MODES = (SMOKE_MODE_STANDALONE, SMOKE_MODE_CONTROLLED_REPAIR)

SMOKE_STATUS_DRY_RUN = "dry_run"
SMOKE_STATUS_COMPLETED = "completed"
SMOKE_STATUS_CONFIG_ERROR = "config_error"
SMOKE_STATUS_EXECUTION_FAILED = "execution_failed"
SMOKE_STATUS_SCENARIO_MISMATCH = "scenario_mismatch"

EXIT_OK = 0
EXIT_CONFIG_ERROR = 1
EXIT_EXECUTION_FAILURE = 2
EXIT_SCENARIO_MISMATCH = 3

MAX_CANDIDATE_WRITER_CALLS = 1
MAX_QUALITY_EVALUATOR_CALLS_STANDALONE = 1
MAX_QUALITY_EVALUATOR_CALLS_CONTROLLED_REPAIR = 2
MAX_REPAIR_WRITER_CALLS = 1

DEFAULT_CANDIDATE_MAX_OUTPUT_TOKENS = 1200
DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS = 2400
DEFAULT_QUALITY_MAX_OUTPUT_TOKENS = DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS
DEFAULT_REPAIR_MAX_OUTPUT_TOKENS = 1200

SECRET_FIELD_FRAGMENTS = (
    "api_key",
    "apikey",
    "authorization",
    "bearer",
    "token",
    "secret",
    "password",
    "headers",
)
PROVIDER_DIAGNOSTIC_FIELDS = (
    "provider",
    "model",
    "stop_reason",
    "content_block_types",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
    "empty_text_classification",
)


@dataclass(frozen=True)
class FinalPostSmokeRunRequest:
    input_path: Path
    mode: str = SMOKE_MODE_STANDALONE
    allow_api: bool = False
    save_output: bool = False
    include_raw_responses: bool = False
    expect_repair: bool = False
    output_dir: Path | None = None
    candidate_provider: str | None = None
    candidate_model: str | None = None
    semantic_grounding_provider: str | None = None
    semantic_grounding_model: str | None = None
    quality_evaluator_provider: str | None = None
    quality_evaluator_model: str | None = None
    repair_provider: str | None = None
    repair_model: str | None = None
    candidate_max_output_tokens: int = DEFAULT_CANDIDATE_MAX_OUTPUT_TOKENS
    semantic_grounding_max_output_tokens: int = DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS
    quality_evaluator_max_output_tokens: int = DEFAULT_QUALITY_MAX_OUTPUT_TOKENS
    repair_max_output_tokens: int = DEFAULT_REPAIR_MAX_OUTPUT_TOKENS


@dataclass(frozen=True)
class FinalPostSmokeRunResult:
    status: str
    exit_code: int
    mode: str
    input_path: str
    provider_models: dict[str, dict[str, str]]
    invocation_budget: dict[str, int]
    invocation_counts: dict[str, int]
    dry_run: bool
    repair_enabled: bool
    repair_executed: bool
    initial_outcome: str | None
    final_outcome: str | None
    accepted: bool
    final_post_text: str
    safe_failure_code: str | None = None
    safe_failure_message: str = ""
    deterministic_gate_passed: bool | None = None
    quality_passed: bool | None = None
    saved_output_path: str | None = None
    sanitized_result: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "mode": self.mode,
            "input_path": self.input_path,
            "provider_models": copy.deepcopy(self.provider_models),
            "invocation_budget": copy.deepcopy(self.invocation_budget),
            "invocation_counts": copy.deepcopy(self.invocation_counts),
            "dry_run": self.dry_run,
            "repair_enabled": self.repair_enabled,
            "repair_executed": self.repair_executed,
            "initial_outcome": self.initial_outcome,
            "final_outcome": self.final_outcome,
            "accepted": self.accepted,
            "final_post_text": self.final_post_text,
            "safe_failure_code": self.safe_failure_code,
            "safe_failure_message": self.safe_failure_message,
            "deterministic_gate_passed": self.deterministic_gate_passed,
            "quality_passed": self.quality_passed,
            "saved_output_path": self.saved_output_path,
            "sanitized_result": copy.deepcopy(self.sanitized_result),
        }


class FinalPostSmokeInputError(ValueError):
    """Raised when a manual smoke fixture or configuration is unsafe."""


def run_final_post_smoke(
    request: FinalPostSmokeRunRequest,
    *,
    candidate_writer_client: Any | None = None,
    semantic_grounding_client: Any | None = None,
    quality_evaluator_client: Any | None = None,
    repair_writer_client: Any | None = None,
) -> FinalPostSmokeRunResult:
    """Run or preview a manual smoke flow using existing orchestration APIs."""

    try:
        prepared = _prepare_smoke_run(request)
        if not request.allow_api:
            result = _dry_run_result(request, prepared)
        else:
            try:
                if request.mode == SMOKE_MODE_STANDALONE:
                    result = _run_standalone_smoke(
                        request,
                        prepared,
                        candidate_writer_client=candidate_writer_client,
                        semantic_grounding_client=semantic_grounding_client,
                        quality_evaluator_client=quality_evaluator_client,
                    )
                else:
                    result = _run_controlled_repair_smoke(
                        request,
                        prepared,
                        candidate_writer_client=candidate_writer_client,
                        semantic_grounding_client=semantic_grounding_client,
                        quality_evaluator_client=quality_evaluator_client,
                        repair_writer_client=repair_writer_client,
                    )
            except Exception:
                result = _execution_error_result(request, prepared)
    except FinalPostSmokeInputError as exc:
        return _config_error_result(request, str(exc))

    if request.save_output:
        saved_path = _save_smoke_output(result, request.output_dir)
        return _replace_saved_output_path(result, saved_path)
    return result


def _prepare_smoke_run(request: FinalPostSmokeRunRequest) -> dict[str, Any]:
    input_path = Path(request.input_path).resolve()
    payload = _load_smoke_input(input_path)
    _validate_mode(request.mode)
    _reject_secret_bearing_input(payload)
    _reject_runtime_fixture_fields(payload)

    post_brief = _require_mapping(payload, "post_brief")
    angle_decision = _require_mapping(payload, "angle_decision")
    selected_evidence_ids = _selected_evidence_ids_from_post_brief(post_brief)

    candidate_contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)
    quality_contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    candidate_prompt_text = _read_prompt_text(candidate_contract.prompt_path)
    quality_prompt_text = _read_prompt_text(quality_contract.prompt_path)
    semantic_grounding_prompt_path = (
        "prompts/linkedin/final_post_semantic_grounding_evaluator.txt"
    )
    semantic_grounding_prompt_text = _read_prompt_text(semantic_grounding_prompt_path)

    try:
        candidate_input = build_candidate_writer_input(
            post_brief,
            angle_decision,
            prompt_metadata=prompt_contract_to_prompt_metadata(candidate_contract),
        )
        candidate_render = render_candidate_writer_prompt_input(candidate_input)
    except (TypeError, ValueError) as exc:
        raise FinalPostSmokeInputError(str(exc)) from exc

    provider_models = {
        "candidate_writer": {
            "provider": _resolve_provider(request.candidate_provider),
            "model": _resolve_model(request.candidate_model),
        },
        "semantic_grounding": {
            "provider": _resolve_provider(request.semantic_grounding_provider),
            "model": _resolve_model(request.semantic_grounding_model),
        },
        "quality_evaluator": {
            "provider": _resolve_provider(request.quality_evaluator_provider),
            "model": _resolve_model(request.quality_evaluator_model),
        },
    }
    if request.mode == SMOKE_MODE_CONTROLLED_REPAIR:
        provider_models["repair_writer"] = {
            "provider": _resolve_provider(request.repair_provider),
            "model": _resolve_model(request.repair_model),
        }

    execution_plan = _preflight_provider_models(
        provider_models,
        validate_keys=request.allow_api,
    )

    policy = FinalPostDecisionPolicy(max_total_attempts=2)
    initial_request = FinalPostAttemptRequest(
        candidate_writer_render=candidate_render,
        candidate_writer_prompt_text=candidate_prompt_text,
        quality_rubric=get_quality_evaluator_rubric_payload(),
        quality_evaluator_prompt_text=quality_prompt_text,
        attempt_index=0,
        max_attempts=2 if request.mode == SMOKE_MODE_CONTROLLED_REPAIR else 1,
        candidate_writer_provider=provider_models["candidate_writer"]["provider"],
        candidate_writer_model=provider_models["candidate_writer"]["model"],
        candidate_writer_max_output_tokens=request.candidate_max_output_tokens,
        semantic_grounding_prompt_text=semantic_grounding_prompt_text,
        semantic_grounding_provider=provider_models["semantic_grounding"]["provider"],
        semantic_grounding_model=provider_models["semantic_grounding"]["model"],
        semantic_grounding_max_output_tokens=request.semantic_grounding_max_output_tokens,
        quality_evaluator_provider=provider_models["quality_evaluator"]["provider"],
        quality_evaluator_model=provider_models["quality_evaluator"]["model"],
        quality_evaluator_max_output_tokens=request.quality_evaluator_max_output_tokens,
        policy=policy,
        execution_metadata={
            "manual_smoke_only": True,
            "input_path": str(input_path),
            "raw_responses_default": "hidden",
        },
    )

    repair_prompt_text = ""
    if request.mode == SMOKE_MODE_CONTROLLED_REPAIR:
        repair_prompt_text = _require_string(payload, "repair_prompt_text")

    return {
        "input_path": input_path,
        "payload": copy.deepcopy(payload),
        "post_brief": copy.deepcopy(post_brief),
        "angle_decision": copy.deepcopy(angle_decision),
        "selected_evidence_ids": selected_evidence_ids,
        "initial_request": initial_request,
        "repair_request": (
            FinalPostControlledRepairRequest(
                initial_attempt_request=initial_request,
                repair_prompt_text=repair_prompt_text,
                repair_provider=provider_models.get("repair_writer", {}).get("provider"),
                repair_model=provider_models.get("repair_writer", {}).get("model"),
                repair_max_output_tokens=request.repair_max_output_tokens,
                repair_enabled=True,
                max_controlled_attempts=2,
                execution_metadata={
                    "manual_smoke_only": True,
                    "input_path": str(input_path),
                    "raw_responses_default": "hidden",
                },
            )
            if request.mode == SMOKE_MODE_CONTROLLED_REPAIR
            else None
        ),
        "provider_models": provider_models,
        "execution_plan": execution_plan,
        "prompt_paths": {
            "candidate_writer": candidate_contract.prompt_path,
            "semantic_grounding": semantic_grounding_prompt_path,
            "quality_evaluator": quality_contract.prompt_path,
            "repair_writer": "fixture:repair_prompt_text"
            if request.mode == SMOKE_MODE_CONTROLLED_REPAIR
            else None,
        },
    }


def _run_standalone_smoke(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
    *,
    candidate_writer_client: Any | None,
    semantic_grounding_client: Any | None,
    quality_evaluator_client: Any | None,
) -> FinalPostSmokeRunResult:
    result = execute_final_post_standalone_attempt(
        prepared["initial_request"],
        post_brief=prepared["post_brief"],
        angle_decision=prepared["angle_decision"],
        selected_evidence_ids=prepared["selected_evidence_ids"],
        candidate_writer_client=candidate_writer_client,
        semantic_grounding_client=semantic_grounding_client,
        quality_evaluator_client=quality_evaluator_client,
    )
    return _result_from_standalone(request, prepared, result)


def _run_controlled_repair_smoke(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
    *,
    candidate_writer_client: Any | None,
    semantic_grounding_client: Any | None,
    quality_evaluator_client: Any | None,
    repair_writer_client: Any | None,
) -> FinalPostSmokeRunResult:
    result = execute_final_post_controlled_repair_attempt(
        prepared["repair_request"],
        post_brief=prepared["post_brief"],
        angle_decision=prepared["angle_decision"],
        selected_evidence_ids=prepared["selected_evidence_ids"],
        candidate_writer_client=candidate_writer_client,
        semantic_grounding_client=semantic_grounding_client,
        quality_evaluator_client=quality_evaluator_client,
        repair_writer_client=repair_writer_client,
    )
    return _result_from_controlled_repair(request, prepared, result)


def _dry_run_result(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
) -> FinalPostSmokeRunResult:
    return FinalPostSmokeRunResult(
        status=SMOKE_STATUS_DRY_RUN,
        exit_code=EXIT_OK,
        mode=request.mode,
        input_path=str(prepared["input_path"]),
        provider_models=prepared["provider_models"],
        invocation_budget=_invocation_budget(request.mode),
        invocation_counts=_zero_invocation_counts(),
        dry_run=True,
        repair_enabled=request.mode == SMOKE_MODE_CONTROLLED_REPAIR,
        repair_executed=False,
        initial_outcome=None,
        final_outcome=None,
        accepted=False,
        final_post_text="",
        sanitized_result={
            "prompt_paths": copy.deepcopy(prepared["prompt_paths"]),
            "selected_evidence_ids": list(prepared["selected_evidence_ids"]),
            "role_diagnostics": prepared["execution_plan"].role_diagnostics(),
            "max_call_budget": _invocation_budget(request.mode),
            "manual_smoke_only": True,
            "api_call": "skipped",
        },
    )


def _result_from_standalone(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
    result: Any,
) -> FinalPostSmokeRunResult:
    outcome = getattr(getattr(result, "final_attempt_outcome", None), "outcome", None)
    accepted_payload = _accepted_payload_from_standalone(result)
    failure_code = getattr(result, "failure_code", None)
    status, exit_code, failure_message = _status_for_result(
        failure_code=failure_code,
        failure_message=getattr(result, "failure_message", ""),
        scenario_mismatch=False,
    )
    sanitized_result = _sanitize_standalone_result(
        result,
        include_raw_responses=request.include_raw_responses,
    )
    sanitized_result["role_diagnostics"] = prepared["execution_plan"].role_diagnostics()
    return FinalPostSmokeRunResult(
        status=status,
        exit_code=exit_code,
        mode=request.mode,
        input_path=str(prepared["input_path"]),
        provider_models=prepared["provider_models"],
        invocation_budget=_invocation_budget(request.mode),
        invocation_counts=_invocation_counts_from_standalone(result),
        dry_run=False,
        repair_enabled=False,
        repair_executed=False,
        initial_outcome=outcome,
        final_outcome=outcome,
        accepted=accepted_payload is not None,
        final_post_text=_post_text_from_payload(accepted_payload),
        safe_failure_code=failure_code,
        safe_failure_message=failure_message,
        deterministic_gate_passed=_gate_passed(getattr(result, "deterministic_gate_output", None)),
        quality_passed=_quality_passed(getattr(result, "quality_evaluation_state", None)),
        sanitized_result=sanitized_result,
    )


def _result_from_controlled_repair(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
    result: Any,
) -> FinalPostSmokeRunResult:
    accepted_payload = copy.deepcopy(getattr(result, "accepted_payload", None))
    repair_executed = bool(getattr(result, "repair_executed", False))
    scenario_mismatch = request.expect_repair and not repair_executed
    failure_code = getattr(result, "failure_code", None)
    status, exit_code, failure_message = _status_for_result(
        failure_code=failure_code,
        failure_message=getattr(result, "failure_message", ""),
        scenario_mismatch=scenario_mismatch,
    )
    if scenario_mismatch:
        failure_code = "expected_repair_not_executed"
        failure_message = "Expected controlled repair, but repair was not executed."

    initial_result = getattr(result, "initial_attempt_result", None)
    sanitized_result = _sanitize_controlled_result(
        result,
        include_raw_responses=request.include_raw_responses,
    )
    sanitized_result["role_diagnostics"] = prepared["execution_plan"].role_diagnostics()
    return FinalPostSmokeRunResult(
        status=status,
        exit_code=exit_code,
        mode=request.mode,
        input_path=str(prepared["input_path"]),
        provider_models=prepared["provider_models"],
        invocation_budget=_invocation_budget(request.mode),
        invocation_counts=_invocation_counts_from_controlled_repair(result),
        dry_run=False,
        repair_enabled=True,
        repair_executed=repair_executed,
        initial_outcome=getattr(
            getattr(initial_result, "final_attempt_outcome", None),
            "outcome",
            None,
        ),
        final_outcome=getattr(result, "terminal_outcome", None),
        accepted=accepted_payload is not None,
        final_post_text=_post_text_from_payload(accepted_payload),
        safe_failure_code=failure_code,
        safe_failure_message=failure_message,
        deterministic_gate_passed=_controlled_gate_passed(result),
        quality_passed=_controlled_quality_passed(result),
        sanitized_result=sanitized_result,
    )


def _config_error_result(
    request: FinalPostSmokeRunRequest,
    message: str,
) -> FinalPostSmokeRunResult:
    return FinalPostSmokeRunResult(
        status=SMOKE_STATUS_CONFIG_ERROR,
        exit_code=EXIT_CONFIG_ERROR,
        mode=request.mode,
        input_path=str(request.input_path),
        provider_models={},
        invocation_budget=_invocation_budget(request.mode),
        invocation_counts=_zero_invocation_counts(),
        dry_run=not request.allow_api,
        repair_enabled=request.mode == SMOKE_MODE_CONTROLLED_REPAIR,
        repair_executed=False,
        initial_outcome=None,
        final_outcome=None,
        accepted=False,
        final_post_text="",
        safe_failure_code="configuration_error",
        safe_failure_message=_safe_message(message),
    )


def _execution_error_result(
    request: FinalPostSmokeRunRequest,
    prepared: dict[str, Any],
) -> FinalPostSmokeRunResult:
    return FinalPostSmokeRunResult(
        status=SMOKE_STATUS_EXECUTION_FAILED,
        exit_code=EXIT_EXECUTION_FAILURE,
        mode=request.mode,
        input_path=str(prepared["input_path"]),
        provider_models=prepared["provider_models"],
        invocation_budget=_invocation_budget(request.mode),
        invocation_counts=_zero_invocation_counts(),
        dry_run=False,
        repair_enabled=request.mode == SMOKE_MODE_CONTROLLED_REPAIR,
        repair_executed=False,
        initial_outcome=None,
        final_outcome=None,
        accepted=False,
        final_post_text="",
        safe_failure_code="smoke_orchestration_failure",
        safe_failure_message="smoke orchestration failed",
    )


def _status_for_result(
    *,
    failure_code: str | None,
    failure_message: str,
    scenario_mismatch: bool,
) -> tuple[str, int, str]:
    if scenario_mismatch:
        return (
            SMOKE_STATUS_SCENARIO_MISMATCH,
            EXIT_SCENARIO_MISMATCH,
            "Expected smoke scenario did not occur.",
        )
    if failure_code:
        return (
            SMOKE_STATUS_EXECUTION_FAILED,
            EXIT_EXECUTION_FAILURE,
            _safe_message(failure_message),
        )
    return SMOKE_STATUS_COMPLETED, EXIT_OK, ""


def _load_smoke_input(input_path: Path) -> dict[str, Any]:
    if not input_path.exists():
        raise FinalPostSmokeInputError(f"smoke input does not exist: {input_path}")
    if not input_path.is_file():
        raise FinalPostSmokeInputError(f"smoke input is not a file: {input_path}")
    try:
        with input_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
    except OSError as exc:
        raise FinalPostSmokeInputError(
            f"smoke input could not be opened: {input_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise FinalPostSmokeInputError(
            f"smoke input is not valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(payload, dict):
        raise FinalPostSmokeInputError("smoke input must be a JSON object")
    return payload


def _validate_mode(mode: str) -> None:
    if mode not in SMOKE_MODES:
        raise FinalPostSmokeInputError(f"unsupported smoke mode: {mode}")


def _reject_secret_bearing_input(value: Any, path: str = "input") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).casefold()
            if any(fragment in key_text for fragment in SECRET_FIELD_FRAGMENTS):
                raise FinalPostSmokeInputError(
                    f"smoke input must not contain secret-bearing field: {path}.{key}"
                )
            _reject_secret_bearing_input(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secret_bearing_input(item, f"{path}[{index}]")


def _reject_runtime_fixture_fields(payload: dict[str, Any]) -> None:
    forbidden_fields = {
        "articles",
        "content_package",
        "api_key",
        "openai_api_key",
        "client",
        "headers",
    }
    present = sorted(forbidden_fields.intersection(payload))
    if present:
        raise FinalPostSmokeInputError(
            f"smoke input contains forbidden runtime fields: {present}"
        )


def _require_mapping(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise FinalPostSmokeInputError(f"smoke input {field_name} must be an object")
    return copy.deepcopy(value)


def _require_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise FinalPostSmokeInputError(f"smoke input {field_name} must be a string")
    return value


def _selected_evidence_ids_from_post_brief(post_brief: dict[str, Any]) -> tuple[str, ...]:
    evidence_to_use = post_brief.get("evidence_to_use")
    if not isinstance(evidence_to_use, list) or not evidence_to_use:
        raise FinalPostSmokeInputError(
            "PostBrief.evidence_to_use must be a non-empty list"
        )
    evidence_ids = []
    for item in evidence_to_use:
        if not isinstance(item, dict):
            raise FinalPostSmokeInputError(
                "PostBrief.evidence_to_use items must be objects"
            )
        evidence_id = item.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise FinalPostSmokeInputError(
                "PostBrief.evidence_to_use.evidence_id must be a non-empty string"
            )
        evidence_text = item.get("evidence_text")
        if not isinstance(evidence_text, str) or not evidence_text.strip():
            raise FinalPostSmokeInputError(
                "PostBrief.evidence_to_use.evidence_text must be a non-empty string"
            )
        role_in_post = item.get("role_in_post")
        if not isinstance(role_in_post, str) or not role_in_post.strip():
            raise FinalPostSmokeInputError(
                "PostBrief.evidence_to_use.role_in_post must be a non-empty string"
            )
        evidence_ids.append(evidence_id)
    if len(set(evidence_ids)) != len(evidence_ids):
        raise FinalPostSmokeInputError("PostBrief.evidence_to_use evidence IDs must be unique")
    return tuple(evidence_ids)


def _read_prompt_text(prompt_path: str) -> str:
    path = Path(settings.BASE_DIR) / prompt_path
    if not path.exists():
        raise FinalPostSmokeInputError(f"prompt file does not exist: {prompt_path}")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise FinalPostSmokeInputError(f"prompt file is empty: {prompt_path}")
    return text


def _resolve_provider(provider: str | None) -> str:
    return str(provider if provider is not None else settings.POSTFLOW_POST_PROVIDER).strip().lower()


def _resolve_model(model: str | None) -> str:
    return str(model if model is not None else settings.POSTFLOW_POST_MODEL).strip()


def _preflight_provider_models(
    provider_models: dict[str, dict[str, str]],
    *,
    validate_keys: bool,
) -> FinalPostExecutionPlanPreflightResult:
    result = preflight_final_post_execution_plan(
        _role_selections_from_provider_models(
            provider_models,
            validate_keys=validate_keys,
        )
    )
    failure = result.first_failure()
    if failure is not None:
        raise FinalPostSmokeInputError(failure.message)
    return result


def _role_selections_from_provider_models(
    provider_models: dict[str, dict[str, str]],
    *,
    validate_keys: bool,
) -> tuple[FinalPostExecutionRoleSelection, ...]:
    selections = [
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_CANDIDATE_WRITER,
            provider=provider_models["candidate_writer"]["provider"],
            model=provider_models["candidate_writer"]["model"],
            stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_code=FAILURE_CANDIDATE_WRITER_REQUEST,
            stage_label="Candidate Writer",
            validate_key=validate_keys,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
            provider=provider_models["semantic_grounding"]["provider"],
            model=provider_models["semantic_grounding"]["model"],
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            stage_label="Semantic Grounding Evaluator",
            validate_key=validate_keys,
        ),
        FinalPostExecutionRoleSelection(
            role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
            provider=provider_models["quality_evaluator"]["provider"],
            model=provider_models["quality_evaluator"]["model"],
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            stage_label="Quality Evaluator",
            validate_key=validate_keys,
        ),
    ]
    repair_model = provider_models.get("repair_writer")
    if repair_model is not None:
        selections.append(
            FinalPostExecutionRoleSelection(
                role=FINAL_POST_ROLE_REPAIR_WRITER,
                provider=repair_model["provider"],
                model=repair_model["model"],
                stage="repair_writer_request",
                failure_code=FAILURE_REPAIR_WRITER_REQUEST,
                stage_label="Repair Writer",
                validate_key=validate_keys,
            )
        )
    return tuple(selections)


def _invocation_budget(mode: str) -> dict[str, int]:
    return {
        "candidate_writer": MAX_CANDIDATE_WRITER_CALLS,
        "semantic_grounding": (
            MAX_QUALITY_EVALUATOR_CALLS_CONTROLLED_REPAIR
            if mode == SMOKE_MODE_CONTROLLED_REPAIR
            else MAX_QUALITY_EVALUATOR_CALLS_STANDALONE
        ),
        "quality_evaluator": (
            MAX_QUALITY_EVALUATOR_CALLS_CONTROLLED_REPAIR
            if mode == SMOKE_MODE_CONTROLLED_REPAIR
            else MAX_QUALITY_EVALUATOR_CALLS_STANDALONE
        ),
        "repair_writer": (
            MAX_REPAIR_WRITER_CALLS
            if mode == SMOKE_MODE_CONTROLLED_REPAIR
            else 0
        ),
    }


def _zero_invocation_counts() -> dict[str, int]:
    return {
        "candidate_writer": 0,
        "semantic_grounding": 0,
        "quality_evaluator": 0,
        "repair_writer": 0,
    }


def _invocation_counts_from_standalone(result: Any) -> dict[str, int]:
    return {
        "candidate_writer": int(getattr(result, "candidate_writer_invocation_count", 0) or 0),
        "semantic_grounding": int(getattr(result, "semantic_grounding_invocation_count", 0) or 0),
        "quality_evaluator": int(getattr(result, "quality_evaluator_invocation_count", 0) or 0),
        "repair_writer": int(getattr(result, "repair_invocation_count", 0) or 0),
    }


def _invocation_counts_from_controlled_repair(result: Any) -> dict[str, int]:
    return {
        "candidate_writer": int(getattr(result, "candidate_writer_invocation_count", 0) or 0),
        "semantic_grounding": _controlled_semantic_grounding_invocation_count(result),
        "quality_evaluator": int(getattr(result, "quality_evaluator_invocation_count", 0) or 0),
        "repair_writer": int(getattr(result, "repair_invocation_count", 0) or 0),
    }


def _controlled_semantic_grounding_invocation_count(result: Any) -> int:
    explicit_count = getattr(result, "semantic_grounding_invocation_count", None)
    if explicit_count is not None:
        return int(explicit_count or 0)
    initial_result = getattr(result, "initial_attempt_result", None)
    initial_count = int(
        getattr(initial_result, "semantic_grounding_invocation_count", 0) or 0
    )
    repaired_count = 1 if getattr(result, "repaired_semantic_grounding_raw_response", None) else 0
    return initial_count + repaired_count


def _accepted_payload_from_standalone(result: Any) -> dict[str, Any] | None:
    outcome = getattr(result, "final_attempt_outcome", None)
    accepted_result = getattr(outcome, "accepted_result", None)
    return copy.deepcopy(getattr(accepted_result, "accepted_payload", None))


def _post_text_from_payload(payload: dict[str, Any] | None) -> str:
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("post_text") or "")


def _gate_passed(gate_output: Any) -> bool | None:
    if gate_output is None:
        return None
    diagnostics = getattr(gate_output, "diagnostics", None)
    return bool(
        getattr(gate_output, "validation_passed", False)
        and getattr(diagnostics, "system_linkedin_ready", False)
        and getattr(diagnostics, "deterministic_checks_passed", False)
    )


def _quality_passed(quality_state: Any) -> bool | None:
    if quality_state is None:
        return None
    quality_review = getattr(quality_state, "quality_review", None)
    if not isinstance(quality_review, dict):
        return False
    return quality_review.get("pass") is True


def _controlled_gate_passed(result: Any) -> bool | None:
    repaired_gate = getattr(result, "repaired_deterministic_gate_output", None)
    if repaired_gate is not None:
        return _gate_passed(repaired_gate)
    initial_result = getattr(result, "initial_attempt_result", None)
    return _gate_passed(getattr(initial_result, "deterministic_gate_output", None))


def _controlled_quality_passed(result: Any) -> bool | None:
    repaired_quality = getattr(result, "repaired_quality_evaluation_state", None)
    if repaired_quality is not None:
        return _quality_passed(repaired_quality)
    initial_result = getattr(result, "initial_attempt_result", None)
    return _quality_passed(getattr(initial_result, "quality_evaluation_state", None))


def _sanitize_standalone_result(
    result: Any,
    *,
    include_raw_responses: bool,
) -> dict[str, Any]:
    return {
        "completed_stage": getattr(result, "completed_stage", None),
        "failure_stage": getattr(result, "failure_stage", None),
        "failure_code": getattr(result, "failure_code", None),
        "failure_message": _safe_message(getattr(result, "failure_message", "")),
        "stage_statuses": _stage_statuses(result),
        "candidate_writer_invocation_count": getattr(
            result,
            "candidate_writer_invocation_count",
            0,
        ),
        "quality_evaluator_invocation_count": getattr(
            result,
            "quality_evaluator_invocation_count",
            0,
        ),
        "semantic_grounding_invocation_count": getattr(
            result,
            "semantic_grounding_invocation_count",
            0,
        ),
        "repair_invocation_count": getattr(result, "repair_invocation_count", 0),
        "final_attempt_outcome": _final_attempt_outcome_summary(result),
        "candidate_payload": _candidate_payload_summary(
            result,
            include_post_text=include_raw_responses,
        ),
        "quality_review": _quality_review_summary(result),
        "semantic_grounding_review": _semantic_grounding_review_summary(result),
        "provider_response_diagnostics": _provider_response_diagnostics_summary(result),
        **(
            {"raw_texts": _raw_texts_from_standalone(result)}
            if include_raw_responses
            else {}
        ),
    }


def _sanitize_controlled_result(
    result: Any,
    *,
    include_raw_responses: bool,
) -> dict[str, Any]:
    return {
        "repair_eligibility": _to_json_safe(getattr(result, "repair_eligibility", None)),
        "repair_executed": getattr(result, "repair_executed", False),
        "terminal_outcome": getattr(result, "terminal_outcome", None),
        "terminal_reason": _safe_message(getattr(result, "terminal_reason", "")),
        "failure_stage": getattr(result, "failure_stage", None),
        "failure_code": getattr(result, "failure_code", None),
        "failure_message": _safe_message(getattr(result, "failure_message", "")),
        "candidate_writer_invocation_count": getattr(
            result,
            "candidate_writer_invocation_count",
            0,
        ),
        "quality_evaluator_invocation_count": getattr(
            result,
            "quality_evaluator_invocation_count",
            0,
        ),
        "semantic_grounding_invocation_count": _controlled_semantic_grounding_invocation_count(
            result
        ),
        "repair_invocation_count": getattr(result, "repair_invocation_count", 0),
        "initial_attempt": _sanitize_standalone_result(
            getattr(result, "initial_attempt_result", None),
            include_raw_responses=include_raw_responses,
        ),
        "repaired_candidate_payload": _payload_summary(
            getattr(getattr(result, "repaired_candidate_output", None), "payload", None),
            include_post_text=include_raw_responses,
        ),
        "repaired_quality_review": _quality_review_summary_from_state(
            getattr(result, "repaired_quality_evaluation_state", None)
        ),
        "repaired_semantic_grounding_review": _semantic_grounding_review_summary_from_state(
            getattr(result, "repaired_semantic_grounding_state", None)
        ),
        **(
            {"raw_texts": _raw_texts_from_controlled(result)}
            if include_raw_responses
            else {}
        ),
    }


def _stage_statuses(result: Any) -> list[dict[str, Any]]:
    statuses = getattr(result, "stage_statuses", None) or []
    return [
        {
            "stage": getattr(status, "stage", None),
            "status": getattr(status, "status", None),
            "error_code": getattr(status, "error_code", None),
            "error_message": _safe_message(getattr(status, "error_message", "")),
            "metadata": _safe_stage_metadata(getattr(status, "metadata", None)),
        }
        for status in statuses
    ]


def _final_attempt_outcome_summary(result: Any) -> dict[str, Any] | None:
    outcome = getattr(result, "final_attempt_outcome", None)
    if outcome is None:
        return None
    decision = getattr(outcome, "decision", None)
    return {
        "outcome": getattr(outcome, "outcome", None),
        "reason": _safe_message(getattr(outcome, "reason", "")),
        "repair_required": getattr(outcome, "repair_required", None),
        "decision": {
            "action": getattr(decision, "action", None),
            "reason": _safe_message(getattr(decision, "reason", "")),
            "repair_type": getattr(decision, "repair_type", None),
            "needs_human_review": getattr(decision, "needs_human_review", None),
        }
        if decision is not None
        else None,
    }


def _candidate_payload_summary(
    result: Any,
    *,
    include_post_text: bool,
) -> dict[str, Any] | None:
    candidate_output = getattr(result, "candidate_writer_output", None)
    return _payload_summary(
        getattr(candidate_output, "payload", None),
        include_post_text=include_post_text,
    )


def _payload_summary(
    payload: Any,
    *,
    include_post_text: bool = False,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    summary = {
        "hook_variants_count": len(payload.get("hook_variants") or []),
        "cta_variants_count": len(payload.get("cta_variants") or []),
        "hashtags": copy.deepcopy(payload.get("hashtags") or []),
        "quality_checks": copy.deepcopy(payload.get("quality_checks") or {}),
        "carousel_outline_count": len(payload.get("carousel_outline") or []),
    }
    if include_post_text:
        summary["post_text"] = payload.get("post_text")
    return summary


def _quality_review_summary(result: Any) -> dict[str, Any] | None:
    return _quality_review_summary_from_state(
        getattr(result, "quality_evaluation_state", None)
    )


def _quality_review_summary_from_state(quality_state: Any) -> dict[str, Any] | None:
    quality_review = getattr(quality_state, "quality_review", None)
    if not isinstance(quality_review, dict):
        return None
    return {
        "pass": quality_review.get("pass"),
        "total_score": quality_review.get("total_score"),
        "failed_criteria": copy.deepcopy(quality_review.get("failed_criteria") or []),
        "automatic_fail_reason": quality_review.get("automatic_fail_reason"),
        "criterion_rationales": copy.deepcopy(
            quality_review.get("criterion_rationales") or {}
        ),
    }


def _semantic_grounding_review_summary(result: Any) -> dict[str, Any] | None:
    return _semantic_grounding_review_summary_from_state(
        getattr(result, "semantic_grounding_state", None)
    )


def _semantic_grounding_review_summary_from_state(
    semantic_state: Any,
) -> dict[str, Any] | None:
    grounding_review = getattr(semantic_state, "grounding_review", None)
    if grounding_review is None:
        return None
    return {
        "pass": getattr(grounding_review, "passed", None),
        "blocking_claim_ids": list(getattr(grounding_review, "blocking_claim_ids", ()) or ()),
        "automatic_fail_reason": getattr(
            grounding_review,
            "automatic_fail_reason",
            "",
        ),
        "requires_human_review": getattr(
            grounding_review,
            "requires_human_review",
            None,
        ),
    }


def _safe_stage_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    safe_metadata: dict[str, Any] = {}
    adaptation_error_code = metadata.get("adaptation_error_code")
    if isinstance(adaptation_error_code, str) and adaptation_error_code.strip():
        safe_metadata["adaptation_error_code"] = adaptation_error_code.strip()[:120]
    safe_details = metadata.get("safe_details")
    if isinstance(safe_details, dict):
        safe_metadata["safe_details"] = _safe_adaptation_details(safe_details)
    structural_diagnostics = structural_diagnostics_from_dict(
        metadata.get(METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS)
    )
    if structural_diagnostics is not None:
        safe_metadata[METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS] = (
            structural_diagnostics.to_dict()
        )
    return safe_metadata or None


def _safe_adaptation_details(details: dict[str, Any]) -> dict[str, Any]:
    allowed_top_level = {
        "error_code",
        "validation_error",
        "missing_fields",
        "required_fields",
        "post_text",
        "hook_variants",
        "cta_variants",
        "hashtags",
        "quality_checks",
        "carousel_outline",
    }
    return {
        key: _to_json_safe(value)
        for key, value in details.items()
        if key in allowed_top_level
    }


def _provider_response_diagnostics_summary(result: Any) -> dict[str, Any] | None:
    if (
        getattr(result, "failure_stage", None) != STAGE_CANDIDATE_WRITER_EXECUTION
        or getattr(result, "failure_code", None) != FAILURE_CANDIDATE_WRITER_EMPTY_RESPONSE
    ):
        return None
    candidate_raw_response = getattr(result, "candidate_writer_raw_response", None)
    provider_metadata = getattr(
        candidate_raw_response,
        "provider_response_metadata",
        None,
    )
    classification = getattr(
        candidate_raw_response,
        "empty_text_classification",
        None,
    )
    diagnostics = _sanitize_provider_diagnostics(
        provider_metadata,
        empty_text_classification=classification,
    )
    if diagnostics:
        return {"candidate_writer": diagnostics}
    return None


def _sanitize_provider_diagnostics(
    metadata: Any,
    *,
    empty_text_classification: Any,
) -> dict[str, Any] | None:
    diagnostics: dict[str, Any] = {}
    if isinstance(metadata, dict):
        for field_name in PROVIDER_DIAGNOSTIC_FIELDS:
            if field_name == "empty_text_classification":
                continue
            value = metadata.get(field_name)
            if field_name == "content_block_types":
                values = _sanitize_content_block_types(value)
                if values:
                    diagnostics[field_name] = values
            elif field_name in ("input_tokens", "output_tokens", "thinking_tokens"):
                if _is_non_negative_int(value):
                    diagnostics[field_name] = value
            elif isinstance(value, str) and value.strip():
                diagnostics[field_name] = value.strip()[:120]
    if isinstance(empty_text_classification, str) and empty_text_classification.strip():
        diagnostics["empty_text_classification"] = empty_text_classification.strip()[:120]
    return diagnostics or None


def _sanitize_content_block_types(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    sanitized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        block_type = item.strip()[:120]
        if block_type and block_type not in sanitized:
            sanitized.append(block_type)
    return sanitized[:20]


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _raw_texts_from_standalone(result: Any) -> dict[str, str]:
    return {
        "candidate_writer": str(
            getattr(getattr(result, "candidate_writer_raw_response", None), "raw_text", "")
            or ""
        ),
        "semantic_grounding": str(
            getattr(getattr(result, "semantic_grounding_raw_response", None), "raw_text", "")
            or ""
        ),
        "quality_evaluator": str(
            getattr(getattr(result, "quality_evaluator_raw_response", None), "raw_text", "")
            or ""
        ),
    }


def _raw_texts_from_controlled(result: Any) -> dict[str, Any]:
    return {
        "initial": _raw_texts_from_standalone(
            getattr(result, "initial_attempt_result", None)
        ),
        "repair_writer": str(
            getattr(getattr(result, "repair_writer_raw_response", None), "raw_text", "")
            or ""
        ),
        "repaired_semantic_grounding": str(
            getattr(getattr(result, "repaired_semantic_grounding_raw_response", None), "raw_text", "")
            or ""
        ),
        "repaired_quality_evaluator": str(
            getattr(getattr(result, "repaired_quality_evaluator_raw_response", None), "raw_text", "")
            or ""
        ),
    }


def _safe_message(message: Any) -> str:
    text = str(message or "")
    for marker in ("sk-", "Bearer ", "Authorization:"):
        if marker in text:
            return "redacted provider/configuration message"
    return text


def _to_json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if not math.isfinite(value):
            raise TypeError("Non-finite smoke result float is not JSON-safe")
        return value
    if isinstance(value, (str, int, bool)):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _to_json_safe(value.to_dict())
    if is_dataclass(value):
        return _to_json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(item) for item in value]
    return str(value)


def _save_smoke_output(
    result: FinalPostSmokeRunResult,
    output_dir: Path | None,
) -> str:
    base_dir = output_dir or Path(settings.BASE_DIR) / "debug_outputs" / "final_post_smoke_runs"
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / f"final_post_smoke_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    output_path.write_text(
        json.dumps(
            result.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    return str(output_path)


def _replace_saved_output_path(
    result: FinalPostSmokeRunResult,
    saved_path: str,
) -> FinalPostSmokeRunResult:
    return FinalPostSmokeRunResult(
        status=result.status,
        exit_code=result.exit_code,
        mode=result.mode,
        input_path=result.input_path,
        provider_models=result.provider_models,
        invocation_budget=result.invocation_budget,
        invocation_counts=result.invocation_counts,
        dry_run=result.dry_run,
        repair_enabled=result.repair_enabled,
        repair_executed=result.repair_executed,
        initial_outcome=result.initial_outcome,
        final_outcome=result.final_outcome,
        accepted=result.accepted,
        final_post_text=result.final_post_text,
        safe_failure_code=result.safe_failure_code,
        safe_failure_message=result.safe_failure_message,
        deterministic_gate_passed=result.deterministic_gate_passed,
        quality_passed=result.quality_passed,
        saved_output_path=saved_path,
        sanitized_result=result.sanitized_result,
    )
