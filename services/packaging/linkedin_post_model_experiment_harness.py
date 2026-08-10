"""Controlled local experiment harness for final LinkedIn post model plans.

The harness is additive infrastructure around the manual final-post smoke runner.
It validates case and role-plan specifications before execution, delegates each
run to ``run_final_post_smoke(...)``, and writes sanitized local artifacts only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import copy
import csv
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from django.conf import settings

from apps.ai.client import AI_PROVIDER_ANTHROPIC, AI_PROVIDER_GEMINI, AI_PROVIDER_OPENAI
from services.packaging.linkedin_post_controlled_repair_contract import (
    FAILURE_REPAIR_WRITER_REQUEST,
)
from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS,
    structural_diagnostics_from_dict,
)
from services.packaging.linkedin_post_final_post_attempt_contract import (
    FAILURE_CANDIDATE_WRITER_REQUEST,
    FAILURE_QUALITY_EVALUATOR_REQUEST,
    FAILURE_SEMANTIC_GROUNDING_REQUEST,
    STAGE_CANDIDATE_WRITER_REQUEST,
    STAGE_QUALITY_EVALUATOR_REQUEST,
    STAGE_SEMANTIC_GROUNDING_REQUEST,
)
from services.packaging.linkedin_post_final_post_execution_plan import (
    FinalPostExecutionRoleSelection,
    preflight_final_post_execution_plan,
)
from services.packaging.linkedin_post_final_post_smoke_runner import (
    EXIT_CONFIG_ERROR,
    EXIT_EXECUTION_FAILURE,
    EXIT_OK,
    FinalPostSmokeRunRequest,
    FinalPostSmokeRunResult,
    SMOKE_MODE_CONTROLLED_REPAIR,
    SMOKE_MODE_STANDALONE,
    SMOKE_MODES,
    run_final_post_smoke,
)
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    OPENAI_FINAL_POST_MODEL,
    get_final_post_role_thinking_mode,
)


EXPERIMENT_STATUS_COMPLETED = "completed"
EXPERIMENT_STATUS_CONFIG_ERROR = "config_error"
EXPERIMENT_STATUS_EXECUTION_FAILED = "execution_failed"
EXPERIMENT_SCHEMA_VERSION = "2026-08-06"
DEFAULT_EXPERIMENT_OUTPUT_ROOT = Path("debug_outputs") / "final_post_model_experiments"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")

ROLE_CANDIDATE_WRITER = FINAL_POST_ROLE_CANDIDATE_WRITER
ROLE_SEMANTIC_GROUNDING = FINAL_POST_ROLE_SEMANTIC_GROUNDING
ROLE_QUALITY_EVALUATOR = FINAL_POST_ROLE_QUALITY_EVALUATOR
ROLE_REPAIR_WRITER = FINAL_POST_ROLE_REPAIR_WRITER

REPAIR_WRITER_DIAGNOSTIC_FIELDS = (
    "repair_enabled",
    "repair_executed",
    "repair_invocation_count",
    "repair_eligibility_status",
    "repair_eligibility_reason",
    "terminal_outcome",
    "terminal_reason",
    "failure_stage",
    "failure_code",
    "failure_message",
)

FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "api_key",
    "secret",
    "password",
    "token",
    "credential",
    "header",
    "prompt_text",
    "provider_payload",
    "provider_reply",
)


@dataclass(frozen=True)
class FinalPostExperimentRoleModel:
    provider: str
    model: str

    def to_dict(self) -> dict[str, str]:
        return {"provider": self.provider, "model": self.model}


@dataclass(frozen=True)
class FinalPostExperimentCase:
    case_id: str
    input_path: Path

    def to_dict(self) -> dict[str, str]:
        return {"case_id": self.case_id, "input_path": str(self.input_path)}


@dataclass(frozen=True)
class FinalPostExperimentPlan:
    plan_id: str
    mode: str
    candidate_writer: FinalPostExperimentRoleModel
    semantic_grounding: FinalPostExperimentRoleModel
    quality_evaluator: FinalPostExperimentRoleModel
    repair_writer: FinalPostExperimentRoleModel | None = None

    def role_models(self) -> dict[str, FinalPostExperimentRoleModel]:
        roles = {
            ROLE_CANDIDATE_WRITER: self.candidate_writer,
            ROLE_SEMANTIC_GROUNDING: self.semantic_grounding,
            ROLE_QUALITY_EVALUATOR: self.quality_evaluator,
        }
        if self.repair_writer is not None:
            roles[ROLE_REPAIR_WRITER] = self.repair_writer
        return roles

    def to_dict(self) -> dict[str, Any]:
        roles = {
            role: role_model.to_dict()
            for role, role_model in self.role_models().items()
        }
        return {"plan_id": self.plan_id, "mode": self.mode, "roles": roles}


@dataclass(frozen=True)
class FinalPostModelExperimentRequest:
    experiment_id: str
    cases: tuple[FinalPostExperimentCase, ...]
    plans: tuple[FinalPostExperimentPlan, ...]
    runs_per_plan: int = 1
    allow_api: bool = False
    output_root: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "cases": [case.to_dict() for case in self.cases],
            "plans": [plan.to_dict() for plan in self.plans],
            "runs_per_plan": self.runs_per_plan,
            "allow_api": self.allow_api,
            "output_root": str(self.output_root) if self.output_root is not None else None,
        }


@dataclass(frozen=True)
class FinalPostModelExperimentArtifacts:
    output_dir: str
    runs_jsonl: str
    summary_csv: str
    report_md: str
    manifest_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "runs_jsonl": self.runs_jsonl,
            "summary_csv": self.summary_csv,
            "report_md": self.report_md,
            "manifest_json": self.manifest_json,
        }


@dataclass(frozen=True)
class FinalPostModelExperimentResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    artifacts: FinalPostModelExperimentArtifacts | None
    safe_failure_code: str | None = None
    safe_failure_message: str = ""
    run_records: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "experiment_id": self.experiment_id,
            "run_count": self.run_count,
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "safe_failure_code": self.safe_failure_code,
            "safe_failure_message": self.safe_failure_message,
            "run_records": copy.deepcopy(list(self.run_records)),
        }


class FinalPostModelExperimentValidationError(ValueError):
    """Raised when an experiment specification is unsafe or unsupported."""


SmokeRunner = Callable[[FinalPostSmokeRunRequest], FinalPostSmokeRunResult]


def default_final_post_experiment_plans() -> tuple[FinalPostExperimentPlan, ...]:
    """Return the fixed pilot matrix without expanding a Cartesian product."""

    gpt = FinalPostExperimentRoleModel(AI_PROVIDER_OPENAI, OPENAI_FINAL_POST_MODEL)
    claude = FinalPostExperimentRoleModel(AI_PROVIDER_ANTHROPIC, "claude-sonnet-5")
    gemini = FinalPostExperimentRoleModel(AI_PROVIDER_GEMINI, "gemini-3.6-flash")
    return (
        FinalPostExperimentPlan("gpt_writer_gpt_grounding", SMOKE_MODE_STANDALONE, gpt, gpt, gpt),
        FinalPostExperimentPlan("claude_writer_gpt_grounding", SMOKE_MODE_STANDALONE, claude, gpt, gpt),
        FinalPostExperimentPlan("gemini_writer_gpt_grounding", SMOKE_MODE_STANDALONE, gemini, gpt, gpt),
        FinalPostExperimentPlan("gpt_writer_claude_grounding", SMOKE_MODE_STANDALONE, gpt, claude, gpt),
        FinalPostExperimentPlan("claude_writer_claude_grounding", SMOKE_MODE_STANDALONE, claude, claude, gpt),
        FinalPostExperimentPlan("gemini_writer_gemini_grounding", SMOKE_MODE_STANDALONE, gemini, gemini, gpt),
    )


def run_linkedin_final_post_model_experiment(
    request: FinalPostModelExperimentRequest,
    *,
    smoke_runner: SmokeRunner = run_final_post_smoke,
    now_factory: Callable[[], datetime] | None = None,
) -> FinalPostModelExperimentResult:
    """Run a local model experiment by delegating each run to the smoke runner."""

    try:
        validated = _validate_request(request)
    except FinalPostModelExperimentValidationError as exc:
        return FinalPostModelExperimentResult(
            status=EXPERIMENT_STATUS_CONFIG_ERROR,
            exit_code=EXIT_CONFIG_ERROR,
            experiment_id=str(request.experiment_id or ""),
            run_count=0,
            artifacts=None,
            safe_failure_code="experiment_configuration_error",
            safe_failure_message=_safe_text(str(exc)),
        )

    now = now_factory or (lambda: datetime.now(UTC))
    started = _isoformat(now())
    output_dir = validated["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = _artifact_paths(output_dir)

    run_records: list[dict[str, Any]] = []
    status = EXPERIMENT_STATUS_COMPLETED
    exit_code = EXIT_OK
    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                smoke_request = _smoke_request_for(case, plan, request.allow_api)
                smoke_result = smoke_runner(smoke_request)
                completed = _isoformat(now())
                record = _run_record_from_smoke_result(
                    request=request,
                    case=case,
                    plan=plan,
                    run_index=run_index,
                    started_at=started,
                    completed_at=completed,
                    smoke_result=smoke_result,
                )
                run_records.append(record)
                if smoke_result.exit_code != EXIT_OK and status == EXPERIMENT_STATUS_COMPLETED:
                    status = EXPERIMENT_STATUS_EXECUTION_FAILED
                    exit_code = EXIT_EXECUTION_FAILURE

    manifest = _manifest(request, output_dir, started)
    _write_artifacts(artifacts, manifest, tuple(run_records))
    return FinalPostModelExperimentResult(
        status=status,
        exit_code=exit_code,
        experiment_id=request.experiment_id,
        run_count=len(run_records),
        artifacts=artifacts,
        safe_failure_code=None if exit_code == EXIT_OK else "experiment_run_failure",
        safe_failure_message="" if exit_code == EXIT_OK else "one or more experiment runs failed",
        run_records=tuple(copy.deepcopy(run_records)),
    )


def _validate_request(request: FinalPostModelExperimentRequest) -> dict[str, Any]:
    _validate_identifier(request.experiment_id, "experiment_id")
    if not request.cases:
        raise FinalPostModelExperimentValidationError("at least one case is required")
    if not request.plans:
        raise FinalPostModelExperimentValidationError("at least one plan is required")
    if request.runs_per_plan < 1:
        raise FinalPostModelExperimentValidationError("runs_per_plan must be at least 1")

    case_ids: set[str] = set()
    for case in request.cases:
        _validate_identifier(case.case_id, "case_id")
        if case.case_id in case_ids:
            raise FinalPostModelExperimentValidationError(f"duplicate case_id: {case.case_id}")
        case_ids.add(case.case_id)
        input_path = Path(case.input_path)
        if not input_path.exists() or not input_path.is_file():
            raise FinalPostModelExperimentValidationError(
                f"case input path does not exist: {input_path}"
            )

    plan_ids: set[str] = set()
    for plan in request.plans:
        _validate_plan(plan)
        if plan.plan_id in plan_ids:
            raise FinalPostModelExperimentValidationError(f"duplicate plan_id: {plan.plan_id}")
        plan_ids.add(plan.plan_id)

    output_root = _resolve_output_root(request.output_root)
    output_dir = output_root / request.experiment_id
    if output_dir.exists():
        raise FinalPostModelExperimentValidationError(
            f"experiment output directory already exists: {output_dir}"
        )
    return {"output_dir": output_dir}


def _validate_plan(plan: FinalPostExperimentPlan) -> None:
    _validate_identifier(plan.plan_id, "plan_id")
    if plan.mode not in SMOKE_MODES:
        raise FinalPostModelExperimentValidationError(f"unsupported mode: {plan.mode}")
    if plan.mode == SMOKE_MODE_STANDALONE and plan.repair_writer is not None:
        raise FinalPostModelExperimentValidationError(
            "repair_writer is not allowed for standalone plans"
        )
    if plan.mode == SMOKE_MODE_CONTROLLED_REPAIR and plan.repair_writer is None:
        raise FinalPostModelExperimentValidationError(
            "repair_writer is required for controlled-repair plans"
        )
    preflight = preflight_final_post_execution_plan(
        _role_selections_for_plan(plan, validate_keys=False)
    )
    failure = preflight.first_failure()
    if failure is not None:
        raise FinalPostModelExperimentValidationError(failure.message)


def _role_selections_for_plan(
    plan: FinalPostExperimentPlan,
    *,
    validate_keys: bool,
) -> tuple[FinalPostExecutionRoleSelection, ...]:
    selections = [
        FinalPostExecutionRoleSelection(
            role=ROLE_CANDIDATE_WRITER,
            provider=plan.candidate_writer.provider,
            model=plan.candidate_writer.model,
            stage=STAGE_CANDIDATE_WRITER_REQUEST,
            failure_code=FAILURE_CANDIDATE_WRITER_REQUEST,
            stage_label="Candidate Writer",
            validate_key=validate_keys,
        ),
        FinalPostExecutionRoleSelection(
            role=ROLE_SEMANTIC_GROUNDING,
            provider=plan.semantic_grounding.provider,
            model=plan.semantic_grounding.model,
            stage=STAGE_SEMANTIC_GROUNDING_REQUEST,
            failure_code=FAILURE_SEMANTIC_GROUNDING_REQUEST,
            stage_label="Semantic Grounding Evaluator",
            validate_key=validate_keys,
        ),
        FinalPostExecutionRoleSelection(
            role=ROLE_QUALITY_EVALUATOR,
            provider=plan.quality_evaluator.provider,
            model=plan.quality_evaluator.model,
            stage=STAGE_QUALITY_EVALUATOR_REQUEST,
            failure_code=FAILURE_QUALITY_EVALUATOR_REQUEST,
            stage_label="Quality Evaluator",
            validate_key=validate_keys,
        ),
    ]
    if plan.repair_writer is not None:
        selections.append(
            FinalPostExecutionRoleSelection(
                role=ROLE_REPAIR_WRITER,
                provider=plan.repair_writer.provider,
                model=plan.repair_writer.model,
                stage="repair_writer_request",
                failure_code=FAILURE_REPAIR_WRITER_REQUEST,
                stage_label="Repair Writer",
                validate_key=validate_keys,
            )
        )
    return tuple(selections)


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise FinalPostModelExperimentValidationError(
            f"{field_name} must be a non-empty safe identifier"
        )


def _resolve_output_root(output_root: Path | None) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    default_root = (base_dir / DEFAULT_EXPERIMENT_OUTPUT_ROOT).resolve()
    if output_root is None:
        _ensure_default_output_root_ignored(base_dir)
        return default_root

    resolved = Path(output_root).resolve()
    if _is_relative_to(resolved, base_dir) and not _is_relative_to(resolved, default_root):
        raise FinalPostModelExperimentValidationError(
            "output_root inside the repository must be under debug_outputs/final_post_model_experiments"
        )
    if _is_relative_to(resolved, default_root):
        _ensure_default_output_root_ignored(base_dir)
    return resolved


def _ensure_default_output_root_ignored(base_dir: Path) -> None:
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists():
        raise FinalPostModelExperimentValidationError("debug_outputs is not ignored")
    ignored_patterns = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    if "debug_outputs/" not in ignored_patterns and "debug_outputs" not in ignored_patterns:
        raise FinalPostModelExperimentValidationError("debug_outputs is not ignored")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _smoke_request_for(
    case: FinalPostExperimentCase,
    plan: FinalPostExperimentPlan,
    allow_api: bool,
) -> FinalPostSmokeRunRequest:
    return FinalPostSmokeRunRequest(
        input_path=case.input_path,
        mode=plan.mode,
        allow_api=allow_api,
        save_output=False,
        include_raw_responses=False,
        candidate_provider=plan.candidate_writer.provider,
        candidate_model=plan.candidate_writer.model,
        semantic_grounding_provider=plan.semantic_grounding.provider,
        semantic_grounding_model=plan.semantic_grounding.model,
        quality_evaluator_provider=plan.quality_evaluator.provider,
        quality_evaluator_model=plan.quality_evaluator.model,
        repair_provider=plan.repair_writer.provider if plan.repair_writer else None,
        repair_model=plan.repair_writer.model if plan.repair_writer else None,
    )


def _run_record_from_smoke_result(
    *,
    request: FinalPostModelExperimentRequest,
    case: FinalPostExperimentCase,
    plan: FinalPostExperimentPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    smoke_result: FinalPostSmokeRunResult,
) -> dict[str, Any]:
    sanitized = smoke_result.sanitized_result if isinstance(smoke_result.sanitized_result, dict) else {}
    quality = sanitized.get("quality_review") if isinstance(sanitized.get("quality_review"), dict) else {}
    grounding = sanitized.get("semantic_grounding_review") if isinstance(sanitized.get("semantic_grounding_review"), dict) else {}
    candidate_payload = sanitized.get("candidate_payload") if isinstance(sanitized.get("candidate_payload"), dict) else {}
    publication_package = sanitized.get("publication_package") if isinstance(sanitized.get("publication_package"), dict) else {}
    final_outcome_summary = sanitized.get("final_attempt_outcome") if isinstance(sanitized.get("final_attempt_outcome"), dict) else {}
    provider_diagnostics = sanitized.get("provider_response_diagnostics") if isinstance(sanitized.get("provider_response_diagnostics"), dict) else {}
    candidate_writer_diagnostics = _candidate_writer_structural_diagnostics(sanitized)
    candidate_writer_primary_violation = _candidate_writer_primary_field_violation(
        candidate_writer_diagnostics
    )
    role_diagnostics = _role_diagnostics_for_plan(plan, sanitized)
    repair_diagnostics = _repair_diagnostics(smoke_result, sanitized, plan)
    record = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "case_id": case.case_id,
        "plan_id": plan.plan_id,
        "run_index": run_index,
        "mode": plan.mode,
        "started_at": started_at,
        "completed_at": completed_at,
        "role_plan": plan.to_dict(),
        "role_diagnostics": role_diagnostics,
        "status": smoke_result.status,
        "exit_code": smoke_result.exit_code,
        "accepted": smoke_result.accepted,
        "deterministic_gate_pass": smoke_result.deterministic_gate_passed,
        "semantic_grounding_pass": grounding.get("pass"),
        "semantic_grounding_blocking_claim_count": _length_or_none(grounding.get("blocking_claim_ids")),
        "quality_pass": smoke_result.quality_passed,
        "quality_total_score": quality.get("total_score"),
        "quality_failed_criteria": copy.deepcopy(quality.get("failed_criteria")) if isinstance(quality.get("failed_criteria"), list) else None,
        "repair_executed": smoke_result.repair_executed,
        "repair_diagnostics": repair_diagnostics,
        "final_attempt_disposition": smoke_result.final_outcome,
        "post_length": len(smoke_result.final_post_text or ""),
        "candidate_post_length": candidate_payload.get("post_text_length"),
        "hook_variant_count": _length_or_none(publication_package.get("hook_variants")),
        "cta_variant_count": _length_or_none(publication_package.get("cta_variants")),
        "hashtag_count": _length_or_none(publication_package.get("hashtags")),
        "provider_invocation_counts": copy.deepcopy(smoke_result.invocation_counts),
        "invocation_budget": copy.deepcopy(smoke_result.invocation_budget),
        "provider_models": copy.deepcopy(smoke_result.provider_models),
        "failure_code": smoke_result.safe_failure_code,
        "failure_message": _safe_text(smoke_result.safe_failure_message),
        "terminal_outcome": final_outcome_summary.get("terminal_outcome"),
        "terminal_reason": final_outcome_summary.get("terminal_reason"),
        "latency_ms": None,
        "input_tokens": None,
        "output_tokens": None,
        "finish_reason": None,
        "provider_diagnostic_category": _provider_diagnostic_category(provider_diagnostics),
        "candidate_writer_structural_diagnostics": candidate_writer_diagnostics,
        "candidate_writer_failure_stage": (
            candidate_writer_diagnostics.get("failure_stage")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_parser_error_code": (
            candidate_writer_diagnostics.get("parser_error_code")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_parser_error_detail_code": (
            candidate_writer_diagnostics.get("parser_error_detail_code")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_parser_error_line": (
            candidate_writer_diagnostics.get("parser_error_line")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_parser_error_column": (
            candidate_writer_diagnostics.get("parser_error_column")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_parser_error_position": (
            candidate_writer_diagnostics.get("parser_error_position")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_adapter_error_code": (
            candidate_writer_diagnostics.get("adapter_error_code")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_top_level_json_type": (
            candidate_writer_diagnostics.get("top_level_json_type")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_candidate_text_length": (
            candidate_writer_diagnostics.get("candidate_text_length")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_missing_field_count": _diagnostic_list_count(
            candidate_writer_diagnostics,
            "missing_required_fields",
        ),
        "candidate_writer_unexpected_field_count": _diagnostic_list_count(
            candidate_writer_diagnostics,
            "unexpected_fields",
        ),
        "candidate_writer_invalid_field_count": _diagnostic_list_count(
            candidate_writer_diagnostics,
            "invalid_field_names",
        ),
        "candidate_writer_field_violation_count": _diagnostic_list_count(
            candidate_writer_diagnostics,
            "field_violations",
        ),
        "candidate_writer_primary_invalid_field": (
            candidate_writer_primary_violation.get("field_name")
            if candidate_writer_primary_violation
            else None
        ),
        "candidate_writer_primary_violation_reason": (
            candidate_writer_primary_violation.get("reason_code")
            if candidate_writer_primary_violation
            else None
        ),
        "candidate_writer_primary_actual_length": (
            candidate_writer_primary_violation.get("actual_length")
            if candidate_writer_primary_violation
            else None
        ),
        "candidate_writer_primary_minimum_required": (
            candidate_writer_primary_violation.get("minimum_required")
            if candidate_writer_primary_violation
            else None
        ),
        "candidate_writer_primary_maximum_allowed": (
            candidate_writer_primary_violation.get("maximum_allowed")
            if candidate_writer_primary_violation
            else None
        ),
        "candidate_writer_diagnostics_truncated": (
            candidate_writer_diagnostics.get("diagnostics_truncated")
            if candidate_writer_diagnostics
            else None
        ),
        "candidate_writer_redacted_key_count": (
            candidate_writer_diagnostics.get("redacted_key_count")
            if candidate_writer_diagnostics
            else None
        ),
        "estimated_cost": None,
        "human_editing_distance": None,
        "human_reviewer_notes": None,
        "final_post_text": smoke_result.final_post_text,
    }
    return _sanitize_artifact_value(record)


def _role_diagnostics_for_plan(plan: FinalPostExperimentPlan, sanitized: dict[str, Any]) -> list[dict[str, Any]]:
    existing = sanitized.get("role_diagnostics")
    diagnostics = copy.deepcopy(existing) if isinstance(existing, list) else []
    roles_present = {
        item.get("role")
        for item in diagnostics
        if isinstance(item, dict) and isinstance(item.get("role"), str)
    }
    for role, role_model in plan.role_models().items():
        if role not in roles_present:
            diagnostics.append(
                {
                    "role": role,
                    "provider": role_model.provider,
                    "model": role_model.model,
                    "thinking_mode": get_final_post_role_thinking_mode(
                        role=role,
                        provider=role_model.provider,
                        model=role_model.model,
                    ),
                    "validation_status": "valid",
                }
            )
    if ROLE_REPAIR_WRITER not in roles_present and plan.repair_writer is None:
        diagnostics.append(
            {
                "role": ROLE_REPAIR_WRITER,
                "provider": None,
                "model": None,
                "thinking_mode": None,
                "validation_status": "not_applicable",
            }
        )
    return diagnostics


def _repair_diagnostics(
    smoke_result: FinalPostSmokeRunResult,
    sanitized: dict[str, Any],
    plan: FinalPostExperimentPlan,
) -> dict[str, Any]:
    outcome = sanitized.get("final_attempt_outcome") if isinstance(sanitized.get("final_attempt_outcome"), dict) else {}
    diagnostics = {
        "repair_enabled": smoke_result.repair_enabled,
        "repair_executed": smoke_result.repair_executed,
        "repair_invocation_count": smoke_result.invocation_counts.get("repair_writer"),
        "repair_eligibility_status": sanitized.get("repair_eligibility_status"),
        "repair_eligibility_reason": sanitized.get("repair_eligibility_reason"),
        "terminal_outcome": outcome.get("terminal_outcome"),
        "terminal_reason": outcome.get("terminal_reason"),
        "failure_stage": sanitized.get("failure_stage"),
        "failure_code": smoke_result.safe_failure_code,
        "failure_message": _safe_text(smoke_result.safe_failure_message),
        "schema_status": "applicable" if plan.repair_writer is not None else "not_applicable",
    }
    return {key: diagnostics.get(key) for key in (*REPAIR_WRITER_DIAGNOSTIC_FIELDS, "schema_status")}


def _provider_diagnostic_category(provider_diagnostics: dict[str, Any]) -> str | None:
    candidate = provider_diagnostics.get("candidate_writer")
    if isinstance(candidate, dict):
        value = candidate.get("empty_text_classification") or candidate.get("stop_reason")
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    return None


def _candidate_writer_structural_diagnostics(
    sanitized: dict[str, Any],
) -> dict[str, Any] | None:
    statuses = sanitized.get("stage_statuses")
    if not isinstance(statuses, list):
        return None
    for status in statuses:
        if not isinstance(status, dict):
            continue
        metadata = status.get("metadata")
        if not isinstance(metadata, dict):
            continue
        diagnostics = structural_diagnostics_from_dict(
            metadata.get(METADATA_KEY_CANDIDATE_WRITER_STRUCTURAL_DIAGNOSTICS)
        )
        if diagnostics is not None:
            return diagnostics.to_dict()
    return None


def _candidate_writer_primary_field_violation(
    diagnostics: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not diagnostics:
        return None
    violations = diagnostics.get("field_violations")
    if not isinstance(violations, list) or not violations:
        return None
    first = violations[0]
    return first if isinstance(first, dict) else None


def _diagnostic_list_count(
    diagnostics: dict[str, Any] | None,
    field_name: str,
) -> int | None:
    if diagnostics is None:
        return None
    value = diagnostics.get(field_name)
    if isinstance(value, list):
        return len(value)
    return None


def _length_or_none(value: Any) -> int | None:
    if isinstance(value, (list, tuple)):
        return len(value)
    return None


def _manifest(
    request: FinalPostModelExperimentRequest,
    output_dir: Path,
    created_at: str,
) -> dict[str, Any]:
    return _sanitize_artifact_value(
        {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": request.experiment_id,
            "created_at": created_at,
            "allow_api": request.allow_api,
            "runs_per_plan": request.runs_per_plan,
            "cases": [case.to_dict() for case in request.cases],
            "plans": [plan.to_dict() for plan in request.plans],
            "output_dir": str(output_dir),
            "git_head": _git_head_or_none(),
        }
    )


def _artifact_paths(output_dir: Path) -> FinalPostModelExperimentArtifacts:
    return FinalPostModelExperimentArtifacts(
        output_dir=str(output_dir),
        runs_jsonl=str(output_dir / "runs.jsonl"),
        summary_csv=str(output_dir / "summary.csv"),
        report_md=str(output_dir / "report.md"),
        manifest_json=str(output_dir / "manifest.json"),
    )


def _write_artifacts(
    artifacts: FinalPostModelExperimentArtifacts,
    manifest: dict[str, Any],
    run_records: tuple[dict[str, Any], ...],
) -> None:
    Path(artifacts.manifest_json).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    Path(artifacts.runs_jsonl).write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for record in run_records
        ),
        encoding="utf-8",
    )
    _write_summary_csv(Path(artifacts.summary_csv), run_records)
    Path(artifacts.report_md).write_text(_report_text(manifest, run_records), encoding="utf-8")


def _write_summary_csv(path: Path, run_records: tuple[dict[str, Any], ...]) -> None:
    fieldnames = (
        "case_id",
        "plan_id",
        "run_index",
        "status",
        "accepted",
        "final_attempt_disposition",
        "repair_executed",
        "failure_code",
        "candidate_writer_calls",
        "semantic_grounding_calls",
        "quality_evaluator_calls",
        "repair_writer_calls",
        "post_length",
        "candidate_writer_failure_stage",
        "candidate_writer_parser_error_code",
        "candidate_writer_parser_error_detail_code",
        "candidate_writer_parser_error_line",
        "candidate_writer_parser_error_column",
        "candidate_writer_parser_error_position",
        "candidate_writer_adapter_error_code",
        "candidate_writer_missing_field_count",
        "candidate_writer_unexpected_field_count",
        "candidate_writer_invalid_field_count",
        "candidate_writer_field_violation_count",
        "candidate_writer_primary_invalid_field",
        "candidate_writer_primary_violation_reason",
        "candidate_writer_primary_actual_length",
        "candidate_writer_primary_minimum_required",
        "candidate_writer_primary_maximum_allowed",
        "candidate_writer_diagnostics_truncated",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in run_records:
            counts = record.get("provider_invocation_counts") or {}
            writer.writerow(
                {
                    "case_id": record.get("case_id"),
                    "plan_id": record.get("plan_id"),
                    "run_index": record.get("run_index"),
                    "status": record.get("status"),
                    "accepted": record.get("accepted"),
                    "final_attempt_disposition": record.get("final_attempt_disposition"),
                    "repair_executed": record.get("repair_executed"),
                    "failure_code": record.get("failure_code"),
                    "candidate_writer_calls": counts.get("candidate_writer"),
                    "semantic_grounding_calls": counts.get("semantic_grounding"),
                    "quality_evaluator_calls": counts.get("quality_evaluator"),
                    "repair_writer_calls": counts.get("repair_writer"),
                    "post_length": record.get("post_length"),
                    "candidate_writer_failure_stage": record.get(
                        "candidate_writer_failure_stage"
                    ),
                    "candidate_writer_parser_error_code": record.get(
                        "candidate_writer_parser_error_code"
                    ),
                    "candidate_writer_parser_error_detail_code": record.get(
                        "candidate_writer_parser_error_detail_code"
                    ),
                    "candidate_writer_parser_error_line": record.get(
                        "candidate_writer_parser_error_line"
                    ),
                    "candidate_writer_parser_error_column": record.get(
                        "candidate_writer_parser_error_column"
                    ),
                    "candidate_writer_parser_error_position": record.get(
                        "candidate_writer_parser_error_position"
                    ),
                    "candidate_writer_adapter_error_code": record.get(
                        "candidate_writer_adapter_error_code"
                    ),
                    "candidate_writer_missing_field_count": record.get(
                        "candidate_writer_missing_field_count"
                    ),
                    "candidate_writer_unexpected_field_count": record.get(
                        "candidate_writer_unexpected_field_count"
                    ),
                    "candidate_writer_invalid_field_count": record.get(
                        "candidate_writer_invalid_field_count"
                    ),
                    "candidate_writer_field_violation_count": record.get(
                        "candidate_writer_field_violation_count"
                    ),
                    "candidate_writer_primary_invalid_field": record.get(
                        "candidate_writer_primary_invalid_field"
                    ),
                    "candidate_writer_primary_violation_reason": record.get(
                        "candidate_writer_primary_violation_reason"
                    ),
                    "candidate_writer_primary_actual_length": record.get(
                        "candidate_writer_primary_actual_length"
                    ),
                    "candidate_writer_primary_minimum_required": record.get(
                        "candidate_writer_primary_minimum_required"
                    ),
                    "candidate_writer_primary_maximum_allowed": record.get(
                        "candidate_writer_primary_maximum_allowed"
                    ),
                    "candidate_writer_diagnostics_truncated": record.get(
                        "candidate_writer_diagnostics_truncated"
                    ),
                }
            )


def _report_text(manifest: dict[str, Any], run_records: tuple[dict[str, Any], ...]) -> str:
    accepted_count = sum(1 for record in run_records if record.get("accepted") is True)
    lines = [
        "# Final Post Model Experiment",
        "",
        f"Experiment: `{manifest['experiment_id']}`",
        f"Runs: {len(run_records)}",
        f"Accepted: {accepted_count}",
        "",
        "| Case | Plan | Run | Status | Accepted | Failure | Candidate Writer diagnostic |",
        "| --- | --- | ---: | --- | --- | --- | --- |",
    ]
    for record in run_records:
        lines.append(
            "| {case} | {plan} | {run} | {status} | {accepted} | {failure} | {diagnostic} |".format(
                case=record.get("case_id"),
                plan=record.get("plan_id"),
                run=record.get("run_index"),
                status=record.get("status"),
                accepted=record.get("accepted"),
                failure=record.get("failure_code") or "",
                diagnostic=_report_candidate_writer_diagnostic(record),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _report_candidate_writer_diagnostic(record: dict[str, Any]) -> str:
    parts = [
        record.get("candidate_writer_failure_stage"),
        record.get("candidate_writer_parser_error_code"),
        record.get("candidate_writer_parser_error_detail_code"),
        record.get("candidate_writer_adapter_error_code"),
    ]
    counts = (
        f"missing={record.get('candidate_writer_missing_field_count') or 0}",
        f"unexpected={record.get('candidate_writer_unexpected_field_count') or 0}",
        f"invalid={record.get('candidate_writer_invalid_field_count') or 0}",
        f"field_violations={record.get('candidate_writer_field_violation_count') or 0}",
    )
    primary_violation = _format_primary_candidate_writer_violation(record)
    filtered = [str(part) for part in parts if isinstance(part, str) and part]
    if not filtered and not any(record.get(key) is not None for key in (
        "candidate_writer_missing_field_count",
        "candidate_writer_unexpected_field_count",
        "candidate_writer_invalid_field_count",
        "candidate_writer_field_violation_count",
    )):
        return ""
    if primary_violation:
        filtered.append(primary_violation)
    if record.get("candidate_writer_diagnostics_truncated") is True:
        filtered.append("truncated")
    return " ".join((*filtered, *counts))


def _format_primary_candidate_writer_violation(record: dict[str, Any]) -> str:
    field_name = record.get("candidate_writer_primary_invalid_field")
    reason = record.get("candidate_writer_primary_violation_reason")
    if not isinstance(field_name, str) or not isinstance(reason, str):
        return ""
    measurements: list[str] = []
    actual_length = record.get("candidate_writer_primary_actual_length")
    minimum_required = record.get("candidate_writer_primary_minimum_required")
    maximum_allowed = record.get("candidate_writer_primary_maximum_allowed")
    if isinstance(actual_length, int):
        measurements.append(f"actual_length={actual_length}")
    if isinstance(minimum_required, int):
        measurements.append(f"minimum_required={minimum_required}")
    if isinstance(maximum_allowed, int):
        measurements.append(f"maximum_allowed={maximum_allowed}")
    suffix = f" ({', '.join(measurements)})" if measurements else ""
    return f"{field_name}:{reason}{suffix}"


def _sanitize_artifact_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            if any(fragment in lowered for fragment in FORBIDDEN_ARTIFACT_KEY_FRAGMENTS):
                continue
            sanitized[key_text] = _sanitize_artifact_value(item)
        return sanitized
    if isinstance(value, (list, tuple)):
        return [_sanitize_artifact_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _safe_text(value: Any) -> str:
    text = str(value or "")
    lowered = text.lower()
    if any(marker in lowered for marker in ("sk-", "bearer ", "x-api-key", "secret")):
        return "redacted safe message"
    return text[:500]


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_head_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=settings.BASE_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    return value or None
