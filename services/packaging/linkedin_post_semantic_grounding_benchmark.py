"""Isolated dry-run benchmark infrastructure for Semantic Grounding."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable

from django.conf import settings

from apps.ai.client import (
    AI_PROVIDER_GEMINI,
    AI_PROVIDER_OPENAI,
    AI_REASONING_EFFORT_LOW,
    AI_REASONING_EFFORT_MINIMAL,
    GEMINI_SUPPORTED_MODELS,
    get_ai_provider_reasoning_effort_error,
)
from services.packaging.linkedin_post_deterministic_gate import run_candidate_post_deterministic_gate
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput
from services.packaging.linkedin_post_flow_input_builders import build_post_editorial_input
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    OPENAI_FINAL_POST_MODEL,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_prompt_renderers import render_semantic_grounding_prompt_input
from services.packaging.linkedin_post_semantic_grounding_structural_diagnostics import (
    build_semantic_grounding_raw_response_diagnostics,
)

BENCHMARK_SCHEMA_VERSION = "2026-08-12"
SOURCE_EXPERIMENT_ID = "writer-claude-vs-gpt-candidatepost-v5"
SOURCE_GIT_COMMIT = "b830bb95e84acb43f00a3f7abdb508d89a584beb"
FIXED_WRITER_PROVIDER = "anthropic"
FIXED_WRITER_MODEL = "claude-sonnet-5"
DEFAULT_FIXTURE_ROOT = Path("tests/fixtures/linkedin_post_semantic_grounding_benchmark/claude_sonnet_5_v5")
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/final_post_semantic_grounding_benchmarks")
DEFAULT_EXPERIMENT_ID = "semantic-grounding-gpt-vs-gemini-v1"
BENCHMARK_STATUS_DRY_RUN = "dry_run"
BENCHMARK_STATUS_COMPLETED = "completed"
BENCHMARK_STATUS_CONFIG_ERROR = "config_error"

FAILURE_PROVIDER_EXECUTION = "provider_execution_failure"
FAILURE_EMPTY_RESPONSE = "empty_response"
FAILURE_PARSE = "parse_failure"
FAILURE_NORMALIZATION = "normalization_failure"
GROUNDING_COMPLETED_PASS = "grounding_completed_pass"
GROUNDING_COMPLETED_BLOCK = "grounding_completed_block"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
PLAN_GPT_4_1_SEMANTIC_GROUNDING = "gpt_4_1_semantic_grounding"
PLAN_GEMINI_3_6_FLASH_SEMANTIC_GROUNDING = "gemini_3_6_flash_semantic_grounding"
PLAN_GROUNDING_DEFAULT = "grounding_default"
PLAN_GROUNDING_MINIMAL = "grounding_minimal"
PLAN_GROUNDING_LOW = "grounding_low"
SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT = "grounding_provider_default"
SEMANTIC_GROUNDING_EXECUTION_PROFILE_MINIMAL_REASONING = "grounding_minimal_reasoning"
SEMANTIC_GROUNDING_EXECUTION_PROFILE_LOW_REASONING = "grounding_low_reasoning"
SEMANTIC_GROUNDING_EXECUTION_PROFILE_REASONING_EFFORTS = {
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT: None,
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_MINIMAL_REASONING: AI_REASONING_EFFORT_MINIMAL,
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_LOW_REASONING: AI_REASONING_EFFORT_LOW,
}
REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION = "GENERICIZATION_IS_A_SELECTION_BLOCKER"
REPAIR_WRITER_ANTI_GENERIC_SELECTION_CRITERIA = (
    "structural_reliability",
    "grounding_fidelity",
    "target_repair_success",
    "quality_evaluator_result",
    "human_voice",
    "distinctive_voice_preservation",
    "anti_genericness",
)
CANONICAL_BENCHMARK_CASE_IDS = ("topic_200_digest_134", "topic_140_digest_126")
DIAGNOSTIC_EXCLUDED_CASE_IDS = ("topic_214_digest_128",)
SEMANTIC_GROUNDING_PARSER_PATH = "services.packaging.linkedin_post_semantic_grounding_parser.parse_and_normalize_semantic_grounding_response"
SEMANTIC_GROUNDING_NORMALIZATION_PATH = "services.packaging.linkedin_post_semantic_grounding_contract.normalize_semantic_grounding_review_result"
DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS = 2400
GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS = 4800

FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "api_key", "secret", "password", "credential", "header",
    "provider_payload", "provider_reply", "prompt_text",
)


@dataclass(frozen=True)
class SemanticGroundingBenchmarkCase:
    case_id: str
    fixture_path: Path
    source_experiment_id: str
    source_git_commit: str
    writer_provider: str
    writer_model: str
    candidate_payload: dict[str, Any]
    candidate_post_character_length: int
    canonical_candidate_valid: bool
    post_brief: dict[str, Any]
    angle_decision: dict[str, Any]
    selected_evidence: tuple[dict[str, Any], ...]
    prompt_metadata: PromptMetadata
    candidate_validity_status: str
    canonical_exclusion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fixture_path": str(self.fixture_path),
            "source_experiment_id": self.source_experiment_id,
            "source_git_commit": self.source_git_commit,
            "writer_provider": self.writer_provider,
            "writer_model": self.writer_model,
            "candidate_payload": copy.deepcopy(self.candidate_payload),
            "candidate_post_character_length": self.candidate_post_character_length,
            "canonical_candidate_valid": self.canonical_candidate_valid,
            "post_brief": copy.deepcopy(self.post_brief),
            "angle_decision": copy.deepcopy(self.angle_decision),
            "selected_evidence": copy.deepcopy(list(self.selected_evidence)),
            "prompt_metadata": self.prompt_metadata.to_dict(),
            "candidate_validity_status": self.candidate_validity_status,
            "canonical_exclusion": copy.deepcopy(self.canonical_exclusion),
        }


@dataclass(frozen=True)
class SemanticGroundingBenchmarkPlan:
    plan_id: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS
    json_mode: bool = True
    execution_profile: str = SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT

    @property
    def reasoning_effort(self) -> str | None:
        return semantic_grounding_reasoning_effort_for_execution_profile(
            self.execution_profile
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "json_mode": self.json_mode,
            "execution_profile": self.execution_profile,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class SemanticGroundingBenchmarkRequest:
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    cases: tuple[SemanticGroundingBenchmarkCase, ...] = ()
    plans: tuple[SemanticGroundingBenchmarkPlan, ...] = ()
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
            "output_root": str(self.output_root) if self.output_root else None,
        }


@dataclass(frozen=True)
class SemanticGroundingBenchmarkArtifacts:
    output_dir: str
    runs_jsonl: str
    summary_csv: str
    report_md: str
    manifest_json: str
    grounding_comparison_md: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "runs_jsonl": self.runs_jsonl,
            "summary_csv": self.summary_csv,
            "report_md": self.report_md,
            "manifest_json": self.manifest_json,
            "grounding_comparison_md": self.grounding_comparison_md,
        }


@dataclass(frozen=True)
class SemanticGroundingBenchmarkResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    provider_call_count: int
    artifacts: SemanticGroundingBenchmarkArtifacts | None
    safe_failure_code: str | None = None
    safe_failure_message: str = ""
    run_records: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "experiment_id": self.experiment_id,
            "run_count": self.run_count,
            "provider_call_count": self.provider_call_count,
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "safe_failure_code": self.safe_failure_code,
            "safe_failure_message": self.safe_failure_message,
            "run_records": copy.deepcopy(list(self.run_records)),
        }


class SemanticGroundingBenchmarkConfigurationError(ValueError):
    pass


NowFactory = Callable[[], datetime]
SemanticGroundingExecutor = Callable[[Any], Any]


def load_semantic_grounding_benchmark_case(path: str | Path) -> SemanticGroundingBenchmarkCase:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    _validate_case_payload(payload, fixture_path)
    return SemanticGroundingBenchmarkCase(
        case_id=payload["case_id"],
        fixture_path=fixture_path,
        source_experiment_id=payload["source_experiment_id"],
        source_git_commit=payload["source_git_commit"],
        writer_provider=payload["writer_provider"],
        writer_model=payload["writer_model"],
        candidate_payload=copy.deepcopy(payload["candidate_payload"]),
        candidate_post_character_length=payload["candidate_post_character_length"],
        canonical_candidate_valid=payload["canonical_candidate_valid"],
        post_brief=copy.deepcopy(payload["post_brief"]),
        angle_decision=copy.deepcopy(payload["angle_decision"]),
        selected_evidence=tuple(copy.deepcopy(payload["selected_evidence"])),
        prompt_metadata=PromptMetadata(**payload["prompt_metadata"]),
        candidate_validity_status=payload["candidate_validity_status"],
        canonical_exclusion=copy.deepcopy(payload.get("canonical_exclusion")),
    )


def default_semantic_grounding_benchmark_cases(fixture_root: Path | None = None) -> tuple[SemanticGroundingBenchmarkCase, ...]:
    root = DEFAULT_FIXTURE_ROOT if fixture_root is None else fixture_root
    return tuple(load_semantic_grounding_benchmark_case(root / f"{case_id}.json") for case_id in CANONICAL_BENCHMARK_CASE_IDS)


def default_semantic_grounding_benchmark_plans() -> tuple[SemanticGroundingBenchmarkPlan, ...]:
    return (
        SemanticGroundingBenchmarkPlan(PLAN_GPT_4_1_SEMANTIC_GROUNDING, AI_PROVIDER_OPENAI, OPENAI_FINAL_POST_MODEL),
        SemanticGroundingBenchmarkPlan(
            PLAN_GEMINI_3_6_FLASH_SEMANTIC_GROUNDING,
            AI_PROVIDER_GEMINI,
            GEMINI_SUPPORTED_MODELS[0],
            max_output_tokens=semantic_grounding_benchmark_max_output_tokens_for_provider(
                AI_PROVIDER_GEMINI
            ),
        ),
    )


def default_semantic_grounding_reasoning_calibration_plans() -> tuple[SemanticGroundingBenchmarkPlan, ...]:
    return (
        SemanticGroundingBenchmarkPlan(
            PLAN_GROUNDING_DEFAULT,
            AI_PROVIDER_GEMINI,
            GEMINI_SUPPORTED_MODELS[0],
            max_output_tokens=GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT,
        ),
        SemanticGroundingBenchmarkPlan(
            PLAN_GROUNDING_MINIMAL,
            AI_PROVIDER_GEMINI,
            GEMINI_SUPPORTED_MODELS[0],
            max_output_tokens=GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_MINIMAL_REASONING,
        ),
        SemanticGroundingBenchmarkPlan(
            PLAN_GROUNDING_LOW,
            AI_PROVIDER_GEMINI,
            GEMINI_SUPPORTED_MODELS[0],
            max_output_tokens=GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_LOW_REASONING,
        ),
    )


def semantic_grounding_reasoning_effort_for_execution_profile(
    execution_profile: str,
) -> str | None:
    normalized = str(execution_profile or "").strip().lower()
    if normalized not in SEMANTIC_GROUNDING_EXECUTION_PROFILE_REASONING_EFFORTS:
        raise SemanticGroundingBenchmarkConfigurationError(
            f"unsupported semantic grounding execution_profile: {execution_profile}"
        )
    return SEMANTIC_GROUNDING_EXECUTION_PROFILE_REASONING_EFFORTS[normalized]


def semantic_grounding_benchmark_max_output_tokens_for_provider(provider: str) -> int:
    if str(provider or "").strip().lower() == AI_PROVIDER_GEMINI:
        return GEMINI_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS
    return DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS


def build_semantic_grounding_benchmark_prompt_render(case: SemanticGroundingBenchmarkCase):
    editorial_input = _post_editorial_input_for_case(case)
    return render_semantic_grounding_prompt_input(editorial_input, prompt_metadata=case.prompt_metadata)


def run_semantic_grounding_benchmark(
    request: SemanticGroundingBenchmarkRequest,
    *,
    now_factory: NowFactory | None = None,
    semantic_grounding_executor: SemanticGroundingExecutor | None = None,
) -> SemanticGroundingBenchmarkResult:
    try:
        output_dir = _validate_request(request)["output_dir"]
    except SemanticGroundingBenchmarkConfigurationError as exc:
        return SemanticGroundingBenchmarkResult(
            status=BENCHMARK_STATUS_CONFIG_ERROR,
            exit_code=1,
            experiment_id=str(request.experiment_id or ""),
            run_count=0,
            provider_call_count=0,
            artifacts=None,
            safe_failure_code="semantic_grounding_benchmark_configuration_error",
            safe_failure_message=_safe_text(str(exc)),
        )
    now = now_factory or (lambda: datetime.now(UTC))
    started_at = _isoformat(now())
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = _artifact_paths(output_dir)
    records: list[dict[str, Any]] = []
    provider_call_count = 0
    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                render = build_semantic_grounding_benchmark_prompt_render(case)
                if request.allow_api:
                    record = _live_run_record(
                        request,
                        case,
                        plan,
                        run_index,
                        started_at,
                        _isoformat(now()),
                        render,
                        semantic_grounding_executor=semantic_grounding_executor,
                    )
                    provider_call_count += record.get("provider_invocation_counts", {}).get("semantic_grounding", 0)
                    records.append(record)
                else:
                    records.append(_dry_run_record(request, case, plan, run_index, started_at, _isoformat(now()), render))
    manifest = _manifest(request, output_dir, started_at)
    _write_artifacts(artifacts, manifest, tuple(records))
    status = BENCHMARK_STATUS_COMPLETED if request.allow_api else BENCHMARK_STATUS_DRY_RUN
    return SemanticGroundingBenchmarkResult(status, 0, request.experiment_id, len(records), provider_call_count, artifacts, run_records=tuple(copy.deepcopy(records)))


def _post_editorial_input_for_case(case: SemanticGroundingBenchmarkCase):
    selected_evidence_ids = tuple(item["evidence_id"] for item in case.selected_evidence)
    candidate_output = CandidateWriterOutput(
        payload=copy.deepcopy(case.candidate_payload), raw_output=None,
        provider=case.writer_provider, model=case.writer_model,
        prompt_name="fixed_claude_sonnet_5_v5_candidate",
        prompt_version=case.source_experiment_id, token_usage=None, cost_metadata=None,
    )
    gate_output = run_candidate_post_deterministic_gate(candidate_output, selected_evidence_ids=selected_evidence_ids)
    return build_post_editorial_input(
        post_brief=copy.deepcopy(case.post_brief), angle_decision=copy.deepcopy(case.angle_decision),
        candidate_output=candidate_output, gate_output=gate_output,
    )


def _validate_request(request: SemanticGroundingBenchmarkRequest) -> dict[str, Any]:
    _validate_identifier(request.experiment_id, "experiment_id")
    if not request.cases:
        raise SemanticGroundingBenchmarkConfigurationError("at least one case is required")
    if not request.plans:
        raise SemanticGroundingBenchmarkConfigurationError("at least one plan is required")
    if request.runs_per_plan < 1:
        raise SemanticGroundingBenchmarkConfigurationError("runs_per_plan must be at least 1")
    seen_cases: set[str] = set()
    for case in request.cases:
        _validate_identifier(case.case_id, "case_id")
        if case.case_id in seen_cases:
            raise SemanticGroundingBenchmarkConfigurationError(f"duplicate case_id: {case.case_id}")
        seen_cases.add(case.case_id)
        if case.case_id not in CANONICAL_BENCHMARK_CASE_IDS:
            if case.case_id in DIAGNOSTIC_EXCLUDED_CASE_IDS:
                raise SemanticGroundingBenchmarkConfigurationError(f"case is diagnostic-only and excluded from the primary Semantic Grounding benchmark: {case.case_id}")
            raise SemanticGroundingBenchmarkConfigurationError(f"case is not an approved canonical Semantic Grounding benchmark case: {case.case_id}")
        if not case.canonical_candidate_valid:
            raise SemanticGroundingBenchmarkConfigurationError(f"case is not canonical-valid for Semantic Grounding benchmark: {case.case_id}")
    seen_plans: set[str] = set()
    for plan in request.plans:
        _validate_plan(plan)
        if plan.plan_id in seen_plans:
            raise SemanticGroundingBenchmarkConfigurationError(f"duplicate plan_id: {plan.plan_id}")
        seen_plans.add(plan.plan_id)
    if request.allow_api:
        _validate_live_prompt_paths(request)
    output_root = _resolve_output_root(request.output_root)
    output_dir = output_root / request.experiment_id
    if output_dir.exists():
        raise SemanticGroundingBenchmarkConfigurationError(f"benchmark output directory already exists: {output_dir}")
    return {"output_dir": output_dir}


def _validate_case_payload(payload: dict[str, Any], fixture_path: Path) -> None:
    if not isinstance(payload, dict):
        raise SemanticGroundingBenchmarkConfigurationError(f"benchmark fixture must be a JSON object: {fixture_path}")
    required = (
        "case_id", "source_experiment_id", "source_git_commit", "writer_provider",
        "writer_model", "candidate_payload", "candidate_post_character_length",
        "canonical_candidate_valid", "candidate_validity_status", "post_brief",
        "angle_decision", "selected_evidence", "prompt_metadata",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise SemanticGroundingBenchmarkConfigurationError("benchmark fixture missing required fields: " + ", ".join(missing))
    if payload["source_experiment_id"] != SOURCE_EXPERIMENT_ID:
        raise SemanticGroundingBenchmarkConfigurationError("unexpected source_experiment_id")
    if payload["source_git_commit"] != SOURCE_GIT_COMMIT:
        raise SemanticGroundingBenchmarkConfigurationError("unexpected source_git_commit")
    if payload["writer_provider"] != FIXED_WRITER_PROVIDER:
        raise SemanticGroundingBenchmarkConfigurationError("unexpected writer_provider")
    if payload["writer_model"] != FIXED_WRITER_MODEL:
        raise SemanticGroundingBenchmarkConfigurationError("unexpected writer_model")
    candidate_payload = payload["candidate_payload"]
    if not isinstance(candidate_payload, dict) or set(candidate_payload) != {"post_text"}:
        raise SemanticGroundingBenchmarkConfigurationError("candidate_payload must contain only post_text")
    post_text = candidate_payload["post_text"]
    if not isinstance(post_text, str) or not post_text.strip():
        raise SemanticGroundingBenchmarkConfigurationError("candidate_payload.post_text must be a non-empty string")
    if payload["candidate_post_character_length"] != len(post_text):
        raise SemanticGroundingBenchmarkConfigurationError("candidate_post_character_length must match candidate_payload.post_text")
    if not isinstance(payload["canonical_candidate_valid"], bool):
        raise SemanticGroundingBenchmarkConfigurationError("canonical_candidate_valid must be a boolean")
    selected_evidence = payload["selected_evidence"]
    post_brief = payload["post_brief"]
    angle_decision = payload["angle_decision"]
    if not isinstance(post_brief, dict) or not isinstance(angle_decision, dict):
        raise SemanticGroundingBenchmarkConfigurationError("post_brief and angle_decision must be objects")
    if selected_evidence != post_brief.get("evidence_to_use"):
        raise SemanticGroundingBenchmarkConfigurationError("selected_evidence must match post_brief.evidence_to_use")
    evidence_ids = [item.get("evidence_id") for item in selected_evidence]
    if evidence_ids != angle_decision.get("supporting_evidence_ids"):
        raise SemanticGroundingBenchmarkConfigurationError("selected evidence IDs must match angle_decision.supporting_evidence_ids")
    if not payload["canonical_candidate_valid"] and not payload.get("canonical_exclusion"):
        raise SemanticGroundingBenchmarkConfigurationError("noncanonical fixtures require canonical_exclusion metadata")


def _validate_live_prompt_paths(request: SemanticGroundingBenchmarkRequest) -> None:
    for case in request.cases:
        prompt_path = case.prompt_metadata.prompt_path
        if not prompt_path:
            raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt path is required for live benchmark execution")
        path = (Path(settings.BASE_DIR) / prompt_path).resolve()
        base_dir = Path(settings.BASE_DIR).resolve()
        if not _is_relative_to(path, base_dir):
            raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt path must stay inside the repository")
        if not path.exists():
            raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt path does not exist")
        if not path.read_text(encoding="utf-8").strip():
            raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt text must be non-empty")


def _validate_plan(plan: SemanticGroundingBenchmarkPlan) -> None:
    _validate_identifier(plan.plan_id, "plan_id")
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_SEMANTIC_GROUNDING, provider=plan.provider, model=plan.model,
    )
    if policy_failure is not None:
        raise SemanticGroundingBenchmarkConfigurationError(str(policy_failure))
    if isinstance(plan.max_output_tokens, bool) or not isinstance(plan.max_output_tokens, int) or plan.max_output_tokens <= 0:
        raise SemanticGroundingBenchmarkConfigurationError("max_output_tokens must be a positive integer")
    if not isinstance(plan.json_mode, bool):
        raise SemanticGroundingBenchmarkConfigurationError("json_mode must be a boolean")
    try:
        reasoning_effort = plan.reasoning_effort
    except SemanticGroundingBenchmarkConfigurationError:
        raise
    reasoning_effort_error = get_ai_provider_reasoning_effort_error(
        provider=plan.provider,
        reasoning_effort=reasoning_effort,
        stage_name="semantic grounding",
    )
    if reasoning_effort_error is not None:
        raise SemanticGroundingBenchmarkConfigurationError(reasoning_effort_error)


def _live_run_record(
    request: SemanticGroundingBenchmarkRequest,
    case: SemanticGroundingBenchmarkCase,
    plan: SemanticGroundingBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: Any,
    *,
    semantic_grounding_executor: SemanticGroundingExecutor | None,
) -> dict[str, Any]:
    # Keep provider-capable imports out of the dry-run/import path.
    from services.packaging.linkedin_post_semantic_grounding_execution import (
        build_semantic_grounding_execution_request,
        execute_semantic_grounding_prompt,
        get_semantic_grounding_execution_request_error,
    )
    from services.packaging.linkedin_post_semantic_grounding_parser import (
        ERROR_EMPTY_RAW_RESPONSE,
        ERROR_EXECUTION_FAILED,
        ERROR_NORMALIZATION_FAILED,
        SemanticGroundingResponseParseError,
        parse_and_normalize_semantic_grounding_response,
    )

    selected_evidence_ids = tuple(item["evidence_id"] for item in case.selected_evidence)
    prompt_text = _semantic_grounding_prompt_text(render)
    execution_request = build_semantic_grounding_execution_request(
        render,
        prompt_text=prompt_text,
        provider=plan.provider,
        model=plan.model,
        max_output_tokens=plan.max_output_tokens,
        json_mode=plan.json_mode,
        reasoning_effort=plan.reasoning_effort,
        execution_metadata={
            "experiment_id": request.experiment_id,
            "case_id": case.case_id,
            "plan_id": plan.plan_id,
            "run_index": run_index,
            "execution_profile": plan.execution_profile,
            "reasoning_effort": plan.reasoning_effort,
        },
    )
    request_error = get_semantic_grounding_execution_request_error(execution_request)
    if request_error is not None:
        return _benchmark_record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status="failed", failure_stage="request_validation",
            failure_code=FAILURE_PROVIDER_EXECUTION, execution_request_error=request_error,
            execution_success=False, parse_success=False, normalization_success=False,
            semantic_grounding_calls=0,
        )

    executor = semantic_grounding_executor or execute_semantic_grounding_prompt
    raw_response = executor(execution_request)
    execution_error = getattr(raw_response, "execution_error", None)
    raw_text = getattr(raw_response, "raw_text", "")
    if execution_error:
        if execution_error == "empty provider response":
            failure_code = FAILURE_EMPTY_RESPONSE
        else:
            failure_code = FAILURE_PROVIDER_EXECUTION
        return _benchmark_record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status="failed", failure_stage="execution", failure_code=failure_code,
            execution_success=False, parse_success=False, normalization_success=False,
            canonical_error_code=ERROR_EXECUTION_FAILED,
            response_diagnostics=_raw_response_diagnostics(raw_response),
            semantic_grounding_calls=1,
        )
    if not str(raw_text or "").strip():
        return _benchmark_record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status="failed", failure_stage="execution", failure_code=FAILURE_EMPTY_RESPONSE,
            execution_success=False, parse_success=False, normalization_success=False,
            canonical_error_code=ERROR_EMPTY_RAW_RESPONSE,
            response_diagnostics=_raw_response_diagnostics(raw_response),
            semantic_grounding_calls=1,
        )

    try:
        grounding_review = parse_and_normalize_semantic_grounding_response(
            raw_response,
            selected_evidence_ids=selected_evidence_ids,
        )
    except SemanticGroundingResponseParseError as exc:
        normalization_failed = exc.code == ERROR_NORMALIZATION_FAILED
        return _benchmark_record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status="failed",
            failure_stage="normalization" if normalization_failed else "parse",
            failure_code=FAILURE_NORMALIZATION if normalization_failed else FAILURE_PARSE,
            execution_success=True,
            parse_success=normalization_failed,
            normalization_success=False,
            canonical_error_code=exc.code,
            parser_error_details=_parser_error_details(exc),
            normalization_error_details=(
                _normalization_error_details(exc) if normalization_failed else None
            ),
            response_diagnostics=_raw_response_diagnostics(raw_response),
            semantic_grounding_calls=1,
        )

    review_payload = grounding_review.to_dict()
    passed = bool(review_payload["pass"])
    blocking_claim_count = len(review_payload["blocking_claim_ids"])
    return _benchmark_record(
        request, case, plan, run_index, started_at, completed_at, render,
        execution_status=GROUNDING_COMPLETED_PASS if passed else GROUNDING_COMPLETED_BLOCK,
        failure_stage=None,
        failure_code=GROUNDING_COMPLETED_PASS if passed else GROUNDING_COMPLETED_BLOCK,
        execution_success=True,
        parse_success=True,
        normalization_success=True,
        grounding_pass=passed,
        blocking_claim_count=blocking_claim_count,
        human_review_required=bool(review_payload["requires_human_review"]),
        blocking_claim_ids=review_payload["blocking_claim_ids"],
        claim_reviews=review_payload["claim_reviews"],
        response_diagnostics=_raw_response_diagnostics(raw_response),
        semantic_grounding_calls=1,
    )


def _semantic_grounding_prompt_text(render: Any) -> str:
    prompt_path = getattr(render, "prompt_path", None)
    if not prompt_path:
        raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt path is required for live benchmark execution")
    path = (Path(settings.BASE_DIR) / prompt_path).resolve()
    base_dir = Path(settings.BASE_DIR).resolve()
    if not _is_relative_to(path, base_dir):
        raise SemanticGroundingBenchmarkConfigurationError("semantic grounding prompt path must stay inside the repository")
    return path.read_text(encoding="utf-8")


def _dry_run_record(
    request: SemanticGroundingBenchmarkRequest,
    case: SemanticGroundingBenchmarkCase,
    plan: SemanticGroundingBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: Any,
) -> dict[str, Any]:
    return _benchmark_record(
        request, case, plan, run_index, started_at, completed_at, render,
        execution_status=BENCHMARK_STATUS_DRY_RUN, failure_stage=None, failure_code=None,
        execution_success=None, parse_success=None, normalization_success=None,
        semantic_grounding_calls=0,
    )


def _benchmark_record(
    request: SemanticGroundingBenchmarkRequest,
    case: SemanticGroundingBenchmarkCase,
    plan: SemanticGroundingBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: Any,
    *,
    execution_status: str,
    failure_stage: str | None,
    failure_code: str | None,
    execution_success: bool | None,
    parse_success: bool | None,
    normalization_success: bool | None,
    semantic_grounding_calls: int,
    execution_request_error: str | None = None,
    canonical_error_code: str | None = None,
    grounding_pass: bool | None = None,
    blocking_claim_count: int | None = None,
    human_review_required: bool | None = None,
    blocking_claim_ids: list[str] | None = None,
    claim_reviews: list[dict[str, Any]] | None = None,
    parser_error_details: dict[str, Any] | None = None,
    normalization_error_details: dict[str, Any] | None = None,
    response_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected_evidence_ids = [item["evidence_id"] for item in case.selected_evidence]
    return _sanitize_artifact_value({
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "case_id": case.case_id,
        "plan_id": plan.plan_id,
        "run_index": run_index,
        "run_id": f"{case.case_id}__{plan.plan_id}__{run_index}",
        "started_at": started_at,
        "completed_at": completed_at,
        "execution_status": execution_status,
        "execution_success": execution_success,
        "provider": plan.provider,
        "model": plan.model,
        "max_output_tokens": plan.max_output_tokens,
        "json_mode": plan.json_mode,
        "execution_profile": plan.execution_profile,
        "reasoning_effort": plan.reasoning_effort,
        "source_experiment_id": case.source_experiment_id,
        "source_git_commit": case.source_git_commit,
        "writer_provider": case.writer_provider,
        "writer_model": case.writer_model,
        "candidate_post_character_length": case.candidate_post_character_length,
        "canonical_candidate_valid": case.canonical_candidate_valid,
        "selected_evidence_ids": selected_evidence_ids,
        "semantic_input_summary": _semantic_input_summary(render),
        "rendered_variable_names": list(render.variables),
        "prompt_metadata": {
            "prompt_name": render.prompt_name,
            "prompt_version": render.prompt_version,
            "prompt_path": render.prompt_path,
        },
        "execution_request_error": execution_request_error,
        "parser_path": SEMANTIC_GROUNDING_PARSER_PATH,
        "normalization_path": SEMANTIC_GROUNDING_NORMALIZATION_PATH,
        "parse_success": parse_success,
        "normalization_success": normalization_success,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "canonical_error_code": canonical_error_code,
        "parser_error_details": copy.deepcopy(parser_error_details),
        "normalization_error_details": copy.deepcopy(normalization_error_details),
        "response_diagnostics": copy.deepcopy(response_diagnostics),
        "grounding_pass": grounding_pass,
        "blocking_claim_count": blocking_claim_count,
        "blocking_claim_ids": copy.deepcopy(blocking_claim_ids or []),
        "human_review_required": human_review_required,
        "claim_reviews": copy.deepcopy(claim_reviews or []),
        "provider_invocation_counts": {
            "candidate_writer": 0,
            "semantic_grounding": semantic_grounding_calls,
            "quality_evaluator": 0,
            "repair_writer": 0,
            "publication_packaging": 0,
        },
    })


def _parser_error_details(error: Exception) -> dict[str, Any]:
    return {
        "line": getattr(error, "line", None),
        "column": getattr(error, "column", None),
        "position": getattr(error, "position", None),
    }


def _normalization_error_details(error: Exception) -> dict[str, Any]:
    return {
        "message": _safe_text(getattr(error, "normalization_error", "")),
    }


def _raw_response_diagnostics(raw_response: Any) -> dict[str, Any]:
    return build_semantic_grounding_raw_response_diagnostics(raw_response)


def _safe_provider_response_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in ("provider", "model"):
        value = metadata.get(key)
        if isinstance(value, str):
            safe[key] = value[:100]
    value = metadata.get("choices_count")
    if isinstance(value, int) and not isinstance(value, bool):
        safe["choices_count"] = value
    for key in ("finish_reasons", "message_content_types"):
        value = metadata.get(key)
        if isinstance(value, (list, tuple)):
            safe[key] = [str(item)[:100] for item in value[:20] if isinstance(item, str)]
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = metadata.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            safe[key] = value
    return safe


def _safe_token_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    safe: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            safe[key] = value
    return safe


def _semantic_input_summary(render: Any) -> dict[str, Any]:
    return {
        name: {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "character_count": len(value),
        }
        for name, value in sorted(render.variables.items())
    }


def _artifact_paths(output_dir: Path) -> SemanticGroundingBenchmarkArtifacts:
    return SemanticGroundingBenchmarkArtifacts(
        output_dir=str(output_dir),
        runs_jsonl=str(output_dir / "runs.jsonl"),
        summary_csv=str(output_dir / "summary.csv"),
        report_md=str(output_dir / "report.md"),
        manifest_json=str(output_dir / "manifest.json"),
        grounding_comparison_md=str(output_dir / "grounding_comparison.md"),
    )


def _write_artifacts(artifacts: SemanticGroundingBenchmarkArtifacts, manifest: dict[str, Any], run_records: tuple[dict[str, Any], ...]) -> None:
    Path(artifacts.manifest_json).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    Path(artifacts.runs_jsonl).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) for record in run_records) + ("\n" if run_records else ""),
        encoding="utf-8",
    )
    _write_summary_csv(Path(artifacts.summary_csv), run_records)
    Path(artifacts.report_md).write_text(_report_text(manifest, run_records), encoding="utf-8")
    Path(artifacts.grounding_comparison_md).write_text(_comparison_text(manifest, run_records), encoding="utf-8")


def _manifest(request: SemanticGroundingBenchmarkRequest, output_dir: Path, started_at: str) -> dict[str, Any]:
    return _sanitize_artifact_value({
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "benchmark_type": "isolated_semantic_grounding",
        "source_experiment_id": SOURCE_EXPERIMENT_ID,
        "source_git_commit": SOURCE_GIT_COMMIT,
        "created_at": started_at,
        "git_branch": _git_branch_or_none(),
        "git_head": _git_head_or_none(),
        "output_dir": str(output_dir),
        "allow_api": request.allow_api,
        "dry_run": not request.allow_api,
        "runs_per_plan": request.runs_per_plan,
        "cases": [
            {
                "case_id": case.case_id,
                "fixture_path": str(case.fixture_path),
                "candidate_post_character_length": case.candidate_post_character_length,
                "canonical_candidate_valid": case.canonical_candidate_valid,
            }
            for case in request.cases
        ],
        "plans": [plan.to_dict() for plan in request.plans],
        "grounding_calibration_selection": {
            "technical_viability": [
                "provider executions complete",
                "responses parse",
                "no output-limit truncations",
                "strict JSON contract preserved",
                "hidden output consumption materially lower than provider default",
                "no provider execution regressions",
            ],
            "semantic_quality": [
                "no new false positives",
                "no new false negatives",
                "causal-overreach detection preserved",
                "forecast qualification detection preserved",
                "unsupported fact detection preserved",
                "source-bounded author synthesis preserved",
                "rhetorical non-claim handling preserved",
            ],
            "preferred_order_when_semantically_equal": [
                "minimal",
                "low",
                "provider_default",
            ],
        },
        "repair_writer_selection_rule": {
            "genericization": REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION,
            "criteria": list(REPAIR_WRITER_ANTI_GENERIC_SELECTION_CRITERIA),
            "production_selection_changed": False,
        },
        "roles_not_invoked": ["Candidate Writer", "Quality Evaluator", "Repair Writer", "publication packaging"],
    })


def _write_summary_csv(path: Path, run_records: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "case_id", "plan_id", "provider", "model", "execution_profile",
            "reasoning_effort", "run_index", "execution_status", "parse_success", "normalization_success",
            "grounding_pass", "blocking_claim_count", "human_review_required", "provider_calls",
        ))
        writer.writeheader()
        for record in run_records:
            counts = record.get("provider_invocation_counts") or {}
            writer.writerow({
                "case_id": record.get("case_id"),
                "plan_id": record.get("plan_id"),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "execution_profile": record.get("execution_profile"),
                "reasoning_effort": record.get("reasoning_effort"),
                "run_index": record.get("run_index"),
                "execution_status": record.get("execution_status"),
                "parse_success": record.get("parse_success"),
                "normalization_success": record.get("normalization_success"),
                "grounding_pass": record.get("grounding_pass"),
                "blocking_claim_count": record.get("blocking_claim_count"),
                "human_review_required": record.get("human_review_required"),
                "provider_calls": sum(value for value in counts.values() if isinstance(value, int)),
            })


def _report_text(manifest: dict[str, Any], run_records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Isolated Semantic Grounding Benchmark", "",
        f"Experiment: `{manifest['experiment_id']}`", f"Runs: {len(run_records)}", f"Provider calls: {_provider_calls(run_records)}", "",
        _report_scope_sentence(manifest), "",
        "## Runs", "",
        "| Case | Plan | Profile | Reasoning | Provider | Model | Status | Grounding pass | Blocking claims | Human review |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | --- |",
    ]
    for record in run_records:
        lines.append(
            f"| {record.get('case_id')} | {record.get('plan_id')} | {record.get('execution_profile')} | {record.get('reasoning_effort')} | {record.get('provider')} | {record.get('model')} | {record.get('execution_status')} | {record.get('grounding_pass')} | {record.get('blocking_claim_count')} | {record.get('human_review_required')} |"
        )
    return "\n".join(lines) + "\n"


def _report_scope_sentence(manifest: dict[str, Any]) -> str:
    if manifest.get("dry_run"):
        return "This dry-run artifact fixes CandidatePost text and varies only future Semantic Grounding provider/model/execution-profile plans."
    return "This live benchmark artifact fixes CandidatePost text and varies only Semantic Grounding provider/model/execution-profile plans."


def _comparison_text(manifest: dict[str, Any], run_records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Semantic Grounding Comparison", "", f"Experiment: `{manifest['experiment_id']}`", "",
        "Grounding labels are local to each case. This artifact does not identify a winner.", "",
    ]
    model_mapping: list[tuple[str, str, str, str]] = []
    for case_id in sorted({str(record.get("case_id")) for record in run_records}):
        lines.extend([f"## Case `{case_id}`", ""])
        for index, record in enumerate([item for item in run_records if item.get("case_id") == case_id], start=1):
            label = f"Grounding {chr(ord('A') + index - 1)}"
            model_mapping.append((case_id, label, str(record.get("provider")), str(record.get("model"))))
            lines.extend([
                f"### {label}", "",
                f"execution_profile: {record.get('execution_profile')}",
                f"reasoning_effort: {record.get('reasoning_effort')}",
                f"execution_status: {record.get('execution_status')}",
                f"parse_success: {record.get('parse_success')}",
                f"normalization_success: {record.get('normalization_success')}",
                f"grounding_pass: {record.get('grounding_pass')}",
                f"blocking_claim_count: {record.get('blocking_claim_count')}",
                f"human_review_required: {record.get('human_review_required')}",
                "",
                "| Claim | Support | Severity | Blocking | Evidence | Human Review | Finding |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ])
            for claim in record.get("claim_reviews") or []:
                evidence = ", ".join(claim.get("supported_evidence_ids") or [])
                blocking = claim.get("claim_id") in (record.get("blocking_claim_ids") or [])
                lines.append(
                    f"| {claim.get('claim_id')} | {claim.get('support_status')} | {claim.get('severity')} | {blocking} | {evidence} | {record.get('human_review_required')} | {claim.get('rationale')} |"
                )
            if not record.get("claim_reviews"):
                lines.append("|  |  |  |  |  |  |  |")
            lines.append("")
    lines.extend(["## Model Mapping", ""])
    for case_id, label, provider, model in model_mapping:
        lines.append(f"- `{case_id}` {label}: `{provider}` / `{model}`")
    return "\n".join(lines) + "\n"


def _provider_calls(run_records: tuple[dict[str, Any], ...]) -> int:
    total = 0
    for record in run_records:
        counts = record.get("provider_invocation_counts") or {}
        total += sum(value for value in counts.values() if isinstance(value, int))
    return total


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise SemanticGroundingBenchmarkConfigurationError(f"{field_name} must be a non-empty safe identifier")


def _resolve_output_root(output_root: Path | None) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    default_root = (base_dir / DEFAULT_OUTPUT_ROOT).resolve()
    if output_root is None:
        _ensure_default_output_root_ignored(base_dir)
        return default_root
    resolved = Path(output_root).resolve()
    if _is_relative_to(resolved, base_dir) and not _is_relative_to(resolved, default_root):
        raise SemanticGroundingBenchmarkConfigurationError("output_root inside the repository must be under debug_outputs/final_post_semantic_grounding_benchmarks")
    if _is_relative_to(resolved, default_root):
        _ensure_default_output_root_ignored(base_dir)
    return resolved


def _ensure_default_output_root_ignored(base_dir: Path) -> None:
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists():
        raise SemanticGroundingBenchmarkConfigurationError("debug_outputs is not ignored")
    ignored_patterns = {line.strip() for line in gitignore.read_text(encoding="utf-8").splitlines() if line.strip() and not line.strip().startswith("#")}
    if "debug_outputs/" not in ignored_patterns and "debug_outputs" not in ignored_patterns:
        raise SemanticGroundingBenchmarkConfigurationError("debug_outputs is not ignored")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sanitize_artifact_value(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if any(fragment in key_text.lower() for fragment in FORBIDDEN_ARTIFACT_KEY_FRAGMENTS):
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
    if any(marker in text.lower() for marker in ("sk-", "bearer ", "x-api-key", "secret")):
        return "redacted safe message"
    return text[:500]


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _git_branch_or_none() -> str | None:
    try:
        result = subprocess.run(["git", "branch", "--show-current"], cwd=settings.BASE_DIR, text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


def _git_head_or_none() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=settings.BASE_DIR, text=True, capture_output=True, check=False)
    except Exception:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None
