"""Isolated benchmark harness for comparing Repair Writer models.

This module fixes the initial candidate/post context and varies only the
Repair Writer provider/model. It does not invoke Candidate Writer, run smoke
tests, save publication package objects, or connect to publication packaging.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import re
from typing import Any, Callable

from django.conf import settings

from apps.ai.client import (
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_GEMINI,
    AI_PROVIDER_OPENAI,
    AI_REASONING_EFFORT_LOW,
    AI_REASONING_EFFORT_MINIMAL,
)
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_grounding_and_quality,
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
    run_candidate_post_deterministic_gate,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_contracts import FinalPostAttemptHistory
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput
from services.packaging.linkedin_post_flow_input_builders import build_post_editorial_input
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    OPENAI_FINAL_POST_MODEL,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    get_prompt_contract,
    prompt_contract_to_prompt_metadata,
)
from services.packaging.linkedin_post_provider_diagnostics import (
    sanitize_provider_error_diagnostics,
)
from services.packaging.linkedin_post_prompt_renderers import (
    RepairWriterPromptRender,
    render_quality_evaluator_prompt_input,
    render_repair_writer_prompt_input,
    render_semantic_grounding_prompt_input,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorExecutionRequest,
    QualityEvaluatorRawResponse,
    build_quality_evaluator_execution_request,
    execute_quality_evaluator_prompt,
    get_quality_evaluator_execution_request_error,
)
from services.packaging.linkedin_post_quality_evaluator_parser import (
    ERROR_NORMALIZATION_FAILED as QUALITY_ERROR_NORMALIZATION_FAILED,
    QualityEvaluatorResponseParseError,
    parse_and_normalize_quality_evaluator_response,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_repair_writer_execution import (
    DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    RepairWriterExecutionRequest,
    RepairWriterRawResponse,
    build_repair_writer_execution_request,
    execute_repair_writer_prompt,
)
from services.packaging.linkedin_post_repair_writer_structural_diagnostics import (
    build_repair_writer_response_structure_diagnostics,
    build_repair_writer_structural_diagnostics,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_NEEDS_HUMAN_REVIEW,
    GROUNDING_STATUS_NOT_READY,
    GROUNDING_STATUS_PASS,
    FinalPostSemanticGroundingState,
)
from services.packaging.linkedin_post_semantic_grounding_structural_diagnostics import (
    build_semantic_grounding_raw_response_diagnostics,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingExecutionRequest,
    SemanticGroundingRawResponse,
    build_semantic_grounding_execution_request,
    execute_semantic_grounding_prompt,
    get_semantic_grounding_execution_request_error,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    ERROR_NORMALIZATION_FAILED as GROUNDING_ERROR_NORMALIZATION_FAILED,
    SemanticGroundingResponseParseError,
    parse_and_normalize_semantic_grounding_response,
)


BENCHMARK_SCHEMA_VERSION = "2026-08-14"
DEFAULT_EXPERIMENT_ID = "repair-writer-gpt-vs-claude-vs-gemini-v1"
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/final_post_repair_writer_benchmarks")
DEFAULT_FIXTURE_ROOT = Path("tests/fixtures/linkedin_post_repair_writer_benchmark")
BENCHMARK_STATUS_DRY_RUN = "dry_run"
BENCHMARK_STATUS_COMPLETED = "completed"
BENCHMARK_STATUS_CONFIG_ERROR = "config_error"
PLAN_GPT_REPAIR = "gpt_repair"
PLAN_CLAUDE_REPAIR = "claude_repair"
PLAN_GEMINI_REPAIR = "gemini_repair"
REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT = "provider_default"
REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_MINIMAL_REASONING = "gemini_repair_minimal_reasoning"
REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING = "gemini_repair_low_reasoning"
REPAIR_WRITER_EXECUTION_PROFILE_REASONING_EFFORTS = {
    REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_MINIMAL_REASONING: AI_REASONING_EFFORT_MINIMAL,
    REPAIR_WRITER_EXECUTION_PROFILE_GEMINI_LOW_REASONING: AI_REASONING_EFFORT_LOW,
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
REPAIR_WRITER_VOICE_PRESERVATION_DIAGNOSTICS = (
    "original_character_count",
    "repaired_character_count",
    "character_delta_percent",
    "quality_author_point_of_view",
    "quality_human_voice",
    "only_post_text_changed",
)
RECONSTRUCTION_DIRECT = "DIRECT"
RECONSTRUCTION_DETERMINISTIC = "DETERMINISTIC_RECONSTRUCTION"
REPAIR_WRITER_JSON_MODE = False
REPAIR_WRITER_MAX_OUTPUT_TOKENS = DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS
FIXED_SEMANTIC_GROUNDING_PROVIDER = AI_PROVIDER_GEMINI
FIXED_SEMANTIC_GROUNDING_MODEL = "gemini-3.6-flash"
FIXED_QUALITY_EVALUATOR_PROVIDER = AI_PROVIDER_OPENAI
FIXED_QUALITY_EVALUATOR_MODEL = OPENAI_FINAL_POST_MODEL
DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS = 4800
DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS = 2400
REPAIR_PROMPT_NAME = "fixture_repair_writer_prompt"
REPAIR_PROMPT_VERSION = "benchmark_fixture_v1"
REPAIR_PROMPT_TEXT = (
    "You are the FinalPostRepairWriter. Rewrite only the CandidatePost post_text "
    "according to the supplied repair instruction. Preserve selected evidence "
    "boundaries, do not add facts, and return only CandidatePost-compatible JSON "
    "with exactly one field: post_text. Return exactly one plain JSON object "
    'with this shape: {"post_text":"..."}. Do not wrap the JSON in markdown '
    "or code fences. Do not include prose before or after the JSON. The "
    "complete response must end immediately after the closing JSON brace."
)
DEFAULT_BENCHMARK_CASE_FIXTURES = (
    DEFAULT_FIXTURE_ROOT / "topic_140_digest_126.json",
    DEFAULT_FIXTURE_ROOT / "topic_214_digest_128__claude_v3.json",
    DEFAULT_FIXTURE_ROOT / "topic_200_digest_134__gpt_v2.json",
)
CANONICAL_BENCHMARK_CASE_IDS = (
    "topic_140_digest_126",
    "topic_214_digest_128__claude_v3",
    "topic_200_digest_134__gpt_v2",
)
DIAGNOSTIC_EXCLUDED_CASE_IDS = ("topic_214_digest_128__gpt_v5",)
FAILURE_REPAIR_EXECUTION = "repair_writer_execution_failure"
FAILURE_REPAIR_EMPTY_RESPONSE = "repair_writer_empty_response"
FAILURE_REPAIR_PARSE = "repair_writer_parse_failure"
FAILURE_REPAIR_ADAPTATION = "repair_writer_adaptation_failure"
FAILURE_REPAIRED_DETERMINISTIC_GATE = "repaired_deterministic_gate_failure"
FAILURE_GROUNDING_EXECUTION = "semantic_grounding_execution_failure"
FAILURE_GROUNDING_EMPTY_RESPONSE = "semantic_grounding_empty_response"
FAILURE_GROUNDING_PARSE = "semantic_grounding_parse_failure"
FAILURE_GROUNDING_NORMALIZATION = "semantic_grounding_normalization_failure"
FAILURE_GROUNDING_DOMAIN = "semantic_grounding_domain_failure"
FAILURE_QUALITY_EXECUTION = "quality_evaluator_execution_failure"
FAILURE_QUALITY_EMPTY_RESPONSE = "quality_evaluator_empty_response"
FAILURE_QUALITY_PARSE = "quality_evaluator_parse_failure"
FAILURE_QUALITY_NORMALIZATION = "quality_evaluator_normalization_failure"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "api_key",
    "secret",
    "password",
    "credential",
    "header",
    "provider_payload",
    "provider_reply",
    "prompt_text",
    "raw_text",
    "raw_provider_response",
)
SAFE_ARTIFACT_DIAGNOSTIC_KEYS = (
    "raw_text_sha256",
    "raw_text_length",
)

RepairWriterExecutor = Callable[[RepairWriterExecutionRequest], RepairWriterRawResponse]
SemanticGroundingExecutor = Callable[
    [SemanticGroundingExecutionRequest],
    SemanticGroundingRawResponse,
]
QualityEvaluatorExecutor = Callable[
    [QualityEvaluatorExecutionRequest],
    QualityEvaluatorRawResponse,
]
NowFactory = Callable[[], datetime]


class RepairWriterBenchmarkConfigurationError(ValueError):
    """Raised for benchmark configuration errors before provider execution."""


@dataclass(frozen=True)
class RepairWriterBenchmarkCase:
    case_id: str
    fixture_path: Path
    source_fixture_path: Path
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
    known_quality_result: dict[str, Any]
    repair_instruction: dict[str, Any]
    repair_prompt_text: str
    editorial_classification: str
    frozen_input_reconstruction: str
    case_selection_note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "fixture_path": str(self.fixture_path),
            "source_fixture_path": str(self.source_fixture_path),
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
            "known_quality_result": copy.deepcopy(self.known_quality_result),
            "repair_instruction": copy.deepcopy(self.repair_instruction),
            "repair_prompt_text_sha256": _sha256(self.repair_prompt_text),
            "editorial_classification": self.editorial_classification,
            "frozen_input_reconstruction": self.frozen_input_reconstruction,
            "case_selection_note": self.case_selection_note,
        }


@dataclass(frozen=True)
class RepairWriterBenchmarkPlan:
    plan_id: str
    provider: str
    model: str
    max_output_tokens: int = REPAIR_WRITER_MAX_OUTPUT_TOKENS
    json_mode: bool = REPAIR_WRITER_JSON_MODE
    execution_profile: str = REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "json_mode": self.json_mode,
            "execution_profile": self.execution_profile,
        }


@dataclass(frozen=True)
class RepairWriterBenchmarkRequest:
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    cases: tuple[RepairWriterBenchmarkCase, ...] = ()
    plans: tuple[RepairWriterBenchmarkPlan, ...] = ()
    runs_per_plan: int = 1
    allow_api: bool = False
    output_root: Path | None = None


@dataclass(frozen=True)
class RepairWriterBenchmarkArtifacts:
    output_dir: str
    manifest_json: str
    runs_jsonl: str
    summary_csv: str
    report_md: str
    repair_comparison_md: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "manifest_json": self.manifest_json,
            "runs_jsonl": self.runs_jsonl,
            "summary_csv": self.summary_csv,
            "report_md": self.report_md,
            "repair_comparison_md": self.repair_comparison_md,
        }


@dataclass(frozen=True)
class RepairWriterBenchmarkResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    provider_call_count: int
    artifacts: RepairWriterBenchmarkArtifacts | None
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


def load_repair_writer_benchmark_case(
    path: str | Path,
) -> RepairWriterBenchmarkCase:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    source_fixture_path = Path(payload.get("source_fixture_path", ""))
    if not source_fixture_path:
        raise RepairWriterBenchmarkConfigurationError(
            "repair benchmark fixture missing source_fixture_path"
        )
    source_payload = json.loads(source_fixture_path.read_text(encoding="utf-8"))
    merged = {**source_payload, **payload}
    _validate_case_payload(merged, fixture_path)
    return RepairWriterBenchmarkCase(
        case_id=merged["case_id"],
        fixture_path=fixture_path,
        source_fixture_path=source_fixture_path,
        source_experiment_id=merged["source_experiment_id"],
        source_git_commit=merged["source_git_commit"],
        writer_provider=merged["writer_provider"],
        writer_model=merged["writer_model"],
        candidate_payload=copy.deepcopy(merged["candidate_payload"]),
        candidate_post_character_length=merged["candidate_post_character_length"],
        canonical_candidate_valid=merged["canonical_candidate_valid"],
        post_brief=copy.deepcopy(merged["post_brief"]),
        angle_decision=copy.deepcopy(merged["angle_decision"]),
        selected_evidence=tuple(copy.deepcopy(merged["selected_evidence"])),
        known_quality_result=copy.deepcopy(merged["known_historical_quality_result"]),
        repair_instruction=copy.deepcopy(merged["repair_instruction"]),
        repair_prompt_text=merged.get("repair_prompt_text", REPAIR_PROMPT_TEXT),
        editorial_classification=merged["editorial_classification"],
        frozen_input_reconstruction=merged["frozen_input_reconstruction"],
        case_selection_note=merged["case_selection_note"],
    )


def default_repair_writer_benchmark_cases(
    fixture_root: Path | None = None,
) -> tuple[RepairWriterBenchmarkCase, ...]:
    if fixture_root is None:
        return tuple(
            load_repair_writer_benchmark_case(path)
            for path in DEFAULT_BENCHMARK_CASE_FIXTURES
        )
    root = Path(fixture_root)
    return tuple(
        load_repair_writer_benchmark_case(root / path.name)
        for path in DEFAULT_BENCHMARK_CASE_FIXTURES
    )


def default_repair_writer_benchmark_plans() -> tuple[RepairWriterBenchmarkPlan, ...]:
    return (
        RepairWriterBenchmarkPlan(
            PLAN_GPT_REPAIR,
            AI_PROVIDER_OPENAI,
            OPENAI_FINAL_POST_MODEL,
        ),
        RepairWriterBenchmarkPlan(
            PLAN_CLAUDE_REPAIR,
            AI_PROVIDER_ANTHROPIC,
            "claude-sonnet-5",
        ),
        RepairWriterBenchmarkPlan(
            PLAN_GEMINI_REPAIR,
            AI_PROVIDER_GEMINI,
            "gemini-3.6-flash",
        ),
    )


def build_repair_writer_benchmark_prompt_render(
    case: RepairWriterBenchmarkCase,
) -> RepairWriterPromptRender:
    return render_repair_writer_prompt_input(
        original_candidate_payload=case.candidate_payload,
        post_brief=case.post_brief,
        angle_decision=case.angle_decision,
        selected_evidence=case.selected_evidence,
        deterministic_findings={"pass": True, "repair_reasons": []},
        quality_findings=case.known_quality_result,
        repair_instruction=case.repair_instruction,
        attempt_index=2,
        max_attempts=2,
        prompt_metadata=PromptMetadata(
            prompt_name=REPAIR_PROMPT_NAME,
            prompt_version=REPAIR_PROMPT_VERSION,
            prompt_path=None,
        ),
    )


def run_repair_writer_benchmark(
    request: RepairWriterBenchmarkRequest,
    *,
    now_factory: NowFactory | None = None,
    repair_writer_executor: RepairWriterExecutor | None = None,
    semantic_grounding_executor: SemanticGroundingExecutor | None = None,
    quality_evaluator_executor: QualityEvaluatorExecutor | None = None,
) -> RepairWriterBenchmarkResult:
    try:
        output_dir = _validate_request(request)
    except RepairWriterBenchmarkConfigurationError as exc:
        return RepairWriterBenchmarkResult(
            status=BENCHMARK_STATUS_CONFIG_ERROR,
            exit_code=1,
            experiment_id=str(request.experiment_id or ""),
            run_count=0,
            provider_call_count=0,
            artifacts=None,
            safe_failure_code="repair_writer_benchmark_configuration_error",
            safe_failure_message=_safe_text(str(exc)),
        )
    now = now_factory or (lambda: datetime.now(UTC))
    started_at = _isoformat(now())
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = _artifact_paths(output_dir)
    records: list[dict[str, Any]] = []
    provider_calls = 0
    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                render = build_repair_writer_benchmark_prompt_render(case)
                if request.allow_api:
                    record = _live_record(
                        request,
                        case,
                        plan,
                        run_index,
                        started_at,
                        _isoformat(now()),
                        render,
                        repair_writer_executor=repair_writer_executor,
                        semantic_grounding_executor=semantic_grounding_executor,
                        quality_evaluator_executor=quality_evaluator_executor,
                    )
                    provider_calls += _provider_calls_for_record(record)
                else:
                    record = _dry_record(
                        request,
                        case,
                        plan,
                        run_index,
                        started_at,
                        _isoformat(now()),
                        render,
                    )
                records.append(record)
    manifest = _manifest(request, output_dir, started_at)
    _write_artifacts(artifacts, manifest, tuple(records))
    status = (
        BENCHMARK_STATUS_COMPLETED if request.allow_api else BENCHMARK_STATUS_DRY_RUN
    )
    return RepairWriterBenchmarkResult(
        status=status,
        exit_code=0,
        experiment_id=request.experiment_id,
        run_count=len(records),
        provider_call_count=provider_calls,
        artifacts=artifacts,
        run_records=tuple(copy.deepcopy(records)),
    )


def _live_record(
    request: RepairWriterBenchmarkRequest,
    case: RepairWriterBenchmarkCase,
    plan: RepairWriterBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: RepairWriterPromptRender,
    *,
    repair_writer_executor: RepairWriterExecutor | None,
    semantic_grounding_executor: SemanticGroundingExecutor | None,
    quality_evaluator_executor: QualityEvaluatorExecutor | None,
) -> dict[str, Any]:
    repair_request = _repair_execution_request(request, case, plan, run_index, render)
    raw_repair = (repair_writer_executor or execute_repair_writer_prompt)(
        repair_request
    )
    if raw_repair.execution_error:
        return _record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            failure_stage="repair_writer_execution",
            failure_code=_repair_execution_failure_code(raw_repair),
            repair_writer_attempts=1,
            repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
            semantic_grounding_provider_api_calls=0,
            quality_evaluator_provider_api_calls=0,
            response_diagnostics=_raw_response_diagnostics(raw_repair),
        )
    try:
        parsed = parse_candidate_writer_raw_response(raw_repair)
    except CandidateWriterResponseParseError as exc:
        return _record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            failure_stage="repair_writer_parse",
            failure_code=FAILURE_REPAIR_PARSE,
            repair_writer_attempts=1,
            repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
            semantic_grounding_provider_api_calls=0,
            quality_evaluator_provider_api_calls=0,
            parser_error_details={"code": exc.code, "message": _safe_text(str(exc))},
            response_diagnostics=_raw_response_diagnostics(raw_repair),
        )
    try:
        repaired_output = build_candidate_writer_output_from_parsed_response(
            parsed_candidate=parsed,
            raw_response=raw_repair,
        )
    except CandidateWriterOutputAdaptationError as exc:
        repair_adapter_diagnostics = build_repair_writer_structural_diagnostics(
            parsed_repair_candidate=parsed,
            adaptation_error=exc,
            repair_raw_response=raw_repair,
        )
        return _record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            failure_stage="repair_writer_adaptation",
            failure_code=FAILURE_REPAIR_ADAPTATION,
            repair_writer_attempts=1,
            repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
            semantic_grounding_provider_api_calls=0,
            quality_evaluator_provider_api_calls=0,
            parser_error_details={"code": exc.code, "message": _safe_text(str(exc))},
            adaptation_error_details={"code": exc.code, "message": _safe_text(str(exc))},
            repair_adapter_diagnostics=repair_adapter_diagnostics.to_dict(),
            response_diagnostics=_raw_response_diagnostics(raw_repair),
        )
    repair_adapter_diagnostics = build_repair_writer_structural_diagnostics(
        parsed_repair_candidate=parsed,
        repair_raw_response=raw_repair,
    )
    gate_output = run_candidate_post_deterministic_gate(
        repaired_output,
        selected_evidence_ids=_selected_evidence_ids(case),
    )
    preservation = _payload_preservation(case.candidate_payload, repaired_output.payload)
    if not _gate_passed(gate_output):
        return _record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            failure_stage="repaired_deterministic_gate",
            failure_code=FAILURE_REPAIRED_DETERMINISTIC_GATE,
            repair_writer_attempts=1,
            repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
            semantic_grounding_provider_api_calls=0,
            quality_evaluator_provider_api_calls=0,
            repaired_payload=repaired_output.payload,
            repaired_gate=gate_output.to_dict(),
            payload_preservation=preservation,
            repair_adapter_diagnostics=repair_adapter_diagnostics.to_dict(),
            response_diagnostics=_raw_response_diagnostics(raw_repair),
        )
    post_editorial_input = build_post_editorial_input(
        post_brief=copy.deepcopy(case.post_brief),
        angle_decision=copy.deepcopy(case.angle_decision),
        candidate_output=repaired_output,
        gate_output=gate_output,
    )
    grounding_state, grounding_calls, grounding_failure = _run_semantic_grounding(
        case,
        request,
        run_index,
        post_editorial_input,
        semantic_grounding_executor,
    )
    if grounding_state.status != GROUNDING_STATUS_PASS:
        return _record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            failure_stage=grounding_failure.get("failure_stage"),
            failure_code=grounding_failure.get("failure_code"),
            repair_writer_attempts=1,
            repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
            semantic_grounding_provider_api_calls=grounding_calls,
            quality_evaluator_provider_api_calls=0,
            repaired_payload=repaired_output.payload,
            repaired_gate=gate_output.to_dict(),
            semantic_grounding=grounding_state.to_dict(),
            payload_preservation=preservation,
            repair_adapter_diagnostics=repair_adapter_diagnostics.to_dict(),
            response_diagnostics=_raw_response_diagnostics(raw_repair),
        )
    quality_state, quality_calls, quality_failure = _run_quality_evaluator(
        case,
        request,
        run_index,
        post_editorial_input,
        quality_evaluator_executor,
    )
    outcome = build_final_post_attempt_outcome_from_gate_grounding_and_quality(
        post_brief=copy.deepcopy(case.post_brief),
        candidate_output=repaired_output,
        gate_output=gate_output,
        semantic_grounding=grounding_state,
        quality_evaluation=quality_state,
        attempt_index=2,
        attempt_history=FinalPostAttemptHistory(attempts=[]),
        repair_plan=copy.deepcopy(case.repair_instruction),
        parent_attempt_index=1,
    )
    return _record(
        request,
        case,
        plan,
        run_index,
        started_at,
        completed_at,
        render,
        execution_status=BENCHMARK_STATUS_COMPLETED,
        failure_stage=quality_failure.get("failure_stage"),
        failure_code=quality_failure.get("failure_code"),
        repair_writer_attempts=1,
        repair_writer_provider_api_calls=_confirmed_provider_api_call(raw_repair),
        semantic_grounding_provider_api_calls=grounding_calls,
        quality_evaluator_provider_api_calls=quality_calls,
        repaired_payload=repaired_output.payload,
        repaired_gate=gate_output.to_dict(),
        semantic_grounding=grounding_state.to_dict(),
        quality_evaluation=quality_state.to_dict(),
        adjudication_projection=outcome.to_dict(),
        payload_preservation=preservation,
        repair_adapter_diagnostics=repair_adapter_diagnostics.to_dict(),
        response_diagnostics=_raw_response_diagnostics(raw_repair),
    )


def _run_semantic_grounding(
    case: RepairWriterBenchmarkCase,
    request: RepairWriterBenchmarkRequest,
    run_index: int,
    post_editorial_input: Any,
    executor: SemanticGroundingExecutor | None,
) -> tuple[FinalPostSemanticGroundingState, int, dict[str, str | None]]:
    render = render_semantic_grounding_prompt_input(
        post_editorial_input,
        prompt_metadata=PromptMetadata(
            "final_post_semantic_grounding_evaluator",
            "1.0",
            "prompts/linkedin/final_post_semantic_grounding_evaluator.txt",
        ),
    )
    execution_request = build_semantic_grounding_execution_request(
        render,
        prompt_text=_prompt_text(render.prompt_path, "semantic grounding"),
        provider=FIXED_SEMANTIC_GROUNDING_PROVIDER,
        model=FIXED_SEMANTIC_GROUNDING_MODEL,
        max_output_tokens=DEFAULT_SEMANTIC_GROUNDING_MAX_OUTPUT_TOKENS,
        json_mode=True,
        execution_metadata=_execution_metadata(request, case, "semantic_grounding", run_index),
    )
    request_error = get_semantic_grounding_execution_request_error(execution_request)
    if request_error is not None:
        return (
            FinalPostSemanticGroundingState(
                status=GROUNDING_STATUS_NOT_READY,
                grounding_review=None,
                error_code=FAILURE_GROUNDING_EXECUTION,
                error_message=request_error,
            ),
            0,
            {
                "failure_stage": "semantic_grounding_request",
                "failure_code": FAILURE_GROUNDING_EXECUTION,
            },
        )
    raw = (executor or execute_semantic_grounding_prompt)(execution_request)
    if raw.execution_error:
        return (
            FinalPostSemanticGroundingState(
                status=GROUNDING_STATUS_NOT_READY,
                grounding_review=None,
                error_code=_semantic_execution_failure_code(raw),
                error_message=raw.execution_error,
                metadata=_semantic_grounding_failure_metadata(raw),
            ),
            _confirmed_provider_api_call(raw),
            {
                "failure_stage": "semantic_grounding_execution",
                "failure_code": _semantic_execution_failure_code(raw),
            },
        )
    try:
        review = parse_and_normalize_semantic_grounding_response(
            raw,
            selected_evidence_ids=_selected_evidence_ids(case),
        )
    except SemanticGroundingResponseParseError as exc:
        failure_code = (
            FAILURE_GROUNDING_NORMALIZATION
            if exc.code == GROUNDING_ERROR_NORMALIZATION_FAILED
            else FAILURE_GROUNDING_PARSE
        )
        return (
            FinalPostSemanticGroundingState(
                status=GROUNDING_STATUS_NOT_READY,
                grounding_review=None,
                error_code=failure_code,
                error_message=_safe_text(str(exc)),
                metadata=_semantic_grounding_failure_metadata(raw, exc),
            ),
            _confirmed_provider_api_call(raw),
            {
                "failure_stage": "semantic_grounding_normalization"
                if exc.code == GROUNDING_ERROR_NORMALIZATION_FAILED
                else "semantic_grounding_parse",
                "failure_code": failure_code,
            },
        )
    state = FinalPostSemanticGroundingState(
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
    )
    if state.status != GROUNDING_STATUS_PASS:
        return (
            state,
            _confirmed_provider_api_call(raw),
            {
                "failure_stage": "semantic_grounding_domain",
                "failure_code": FAILURE_GROUNDING_DOMAIN,
            },
        )
    return state, _confirmed_provider_api_call(raw), {"failure_stage": None, "failure_code": None}


def _run_quality_evaluator(
    case: RepairWriterBenchmarkCase,
    request: RepairWriterBenchmarkRequest,
    run_index: int,
    post_editorial_input: Any,
    executor: QualityEvaluatorExecutor | None,
) -> tuple[FinalPostQualityEvaluationState, int, dict[str, str | None]]:
    prompt_metadata = prompt_contract_to_prompt_metadata(
        get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    )
    render = render_quality_evaluator_prompt_input(
        post_editorial_input,
        get_quality_evaluator_rubric_payload(),
        prompt_metadata=prompt_metadata,
    )
    execution_request = build_quality_evaluator_execution_request(
        render,
        prompt_text=_prompt_text(render.prompt_path, "quality evaluator"),
        provider=FIXED_QUALITY_EVALUATOR_PROVIDER,
        model=FIXED_QUALITY_EVALUATOR_MODEL,
        max_output_tokens=DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
        json_mode=True,
        execution_metadata=_execution_metadata(request, case, "quality_evaluator", run_index),
    )
    request_error = get_quality_evaluator_execution_request_error(execution_request)
    if request_error is not None:
        return (
            FinalPostQualityEvaluationState(
                status="not_run",
                quality_review=None,
                error_code=FAILURE_QUALITY_EXECUTION,
                error_message=request_error,
            ),
            0,
            {
                "failure_stage": "quality_evaluator_request",
                "failure_code": FAILURE_QUALITY_EXECUTION,
            },
        )
    raw = (executor or execute_quality_evaluator_prompt)(execution_request)
    if raw.execution_error:
        return (
            FinalPostQualityEvaluationState(
                status="execution_failed",
                quality_review=None,
                error_code=_quality_execution_failure_code(raw),
                error_message=raw.execution_error,
                metadata={
                    "provider_error_diagnostics": sanitize_provider_error_diagnostics(
                        getattr(raw, "execution_diagnostics", None)
                    )
                }
                if getattr(raw, "execution_diagnostics", None)
                else None,
            ),
            _confirmed_provider_api_call(raw),
            {
                "failure_stage": "quality_evaluator_execution",
                "failure_code": _quality_execution_failure_code(raw),
            },
        )
    try:
        review = parse_and_normalize_quality_evaluator_response(raw)
    except QualityEvaluatorResponseParseError as exc:
        failure_code = (
            FAILURE_QUALITY_NORMALIZATION
            if exc.code == QUALITY_ERROR_NORMALIZATION_FAILED
            else FAILURE_QUALITY_PARSE
        )
        return (
            FinalPostQualityEvaluationState(
                status="normalization_failed"
                if exc.code == QUALITY_ERROR_NORMALIZATION_FAILED
                else "parse_failed",
                quality_review=None,
                error_code=failure_code,
                error_message=_safe_text(str(exc)),
            ),
            _confirmed_provider_api_call(raw),
            {
                "failure_stage": "quality_evaluator_normalization"
                if exc.code == QUALITY_ERROR_NORMALIZATION_FAILED
                else "quality_evaluator_parse",
                "failure_code": failure_code,
            },
        )
    return (
        FinalPostQualityEvaluationState(
            status=QUALITY_EVALUATION_READY,
            quality_review=review,
            metadata={"benchmark_case_id": case.case_id},
        ),
        _confirmed_provider_api_call(raw),
        {"failure_stage": None, "failure_code": None},
    )


def _dry_record(
    request: RepairWriterBenchmarkRequest,
    case: RepairWriterBenchmarkCase,
    plan: RepairWriterBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: RepairWriterPromptRender,
) -> dict[str, Any]:
    return _record(
        request,
        case,
        plan,
        run_index,
        started_at,
        completed_at,
        render,
        execution_status=BENCHMARK_STATUS_DRY_RUN,
        failure_stage=None,
        failure_code=None,
        repair_writer_attempts=0,
        repair_writer_provider_api_calls=0,
        semantic_grounding_provider_api_calls=0,
        quality_evaluator_provider_api_calls=0,
    )


def _record(
    request: RepairWriterBenchmarkRequest,
    case: RepairWriterBenchmarkCase,
    plan: RepairWriterBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: RepairWriterPromptRender,
    *,
    execution_status: str,
    failure_stage: str | None,
    failure_code: str | None,
    repair_writer_attempts: int,
    repair_writer_provider_api_calls: int,
    semantic_grounding_provider_api_calls: int,
    quality_evaluator_provider_api_calls: int,
    repaired_payload: dict[str, Any] | None = None,
    repaired_gate: dict[str, Any] | None = None,
    semantic_grounding: dict[str, Any] | None = None,
    quality_evaluation: dict[str, Any] | None = None,
    adjudication_projection: dict[str, Any] | None = None,
    payload_preservation: dict[str, Any] | None = None,
    parser_error_details: dict[str, Any] | None = None,
    adaptation_error_details: dict[str, Any] | None = None,
    repair_adapter_diagnostics: dict[str, Any] | None = None,
    response_diagnostics: dict[str, Any] | None = None,
    execution_request_error: str | None = None,
) -> dict[str, Any]:
    provider_counts = {
        "candidate_writer_provider_api_calls": 0,
        "repair_writer_provider_api_calls": repair_writer_provider_api_calls,
        "semantic_grounding_provider_api_calls": semantic_grounding_provider_api_calls,
        "quality_evaluator_provider_api_calls": quality_evaluator_provider_api_calls,
        "publication_packaging_invocations": 0,
    }
    record = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "case_id": case.case_id,
        "source_experiment_id": case.source_experiment_id,
        "source_git_commit": case.source_git_commit,
        "writer_provider": case.writer_provider,
        "writer_model": case.writer_model,
        "candidate_post_sha256": _sha256(case.candidate_payload["post_text"]),
        "candidate_post_character_length": case.candidate_post_character_length,
        "editorial_classification": case.editorial_classification,
        "frozen_input_reconstruction": case.frozen_input_reconstruction,
        "plan_id": plan.plan_id,
        "provider": plan.provider,
        "model": plan.model,
        "run_index": run_index,
        "started_at": started_at,
        "completed_at": completed_at,
        "execution_status": execution_status,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "repair_writer_config": {
            "json_mode": plan.json_mode,
            "max_output_tokens": plan.max_output_tokens,
            "execution_profile": plan.execution_profile,
            "reasoning_effort": _repair_writer_reasoning_effort(plan),
        },
        "fixed_downstream_roles": {
            "semantic_grounding": {
                "provider": FIXED_SEMANTIC_GROUNDING_PROVIDER,
                "model": FIXED_SEMANTIC_GROUNDING_MODEL,
            },
            "quality_evaluator": {
                "provider": FIXED_QUALITY_EVALUATOR_PROVIDER,
                "model": FIXED_QUALITY_EVALUATOR_MODEL,
            },
        },
        "input_parity": _input_parity(case, render),
        "repair_instruction": copy.deepcopy(case.repair_instruction),
        "render_diagnostics": _render_diagnostics(render),
        "provider_invocation_counts": provider_counts,
        "attempt_counts": {
            "candidate_writer_attempts": 0,
            "repair_writer_attempts": repair_writer_attempts,
            "semantic_grounding_attempts": semantic_grounding_provider_api_calls,
            "quality_evaluator_attempts": quality_evaluator_provider_api_calls,
        },
        "repaired_payload": copy.deepcopy(repaired_payload),
        "repaired_gate": copy.deepcopy(repaired_gate),
        "semantic_grounding": copy.deepcopy(semantic_grounding),
        "quality_evaluation": copy.deepcopy(quality_evaluation),
        "adjudication_projection": copy.deepcopy(adjudication_projection),
        "payload_preservation": copy.deepcopy(payload_preservation),
        "parser_error_details": copy.deepcopy(parser_error_details),
        "adaptation_error_details": copy.deepcopy(adaptation_error_details),
        "repair_adapter_diagnostics": copy.deepcopy(repair_adapter_diagnostics),
        "response_diagnostics": copy.deepcopy(response_diagnostics),
        "execution_request_error": execution_request_error,
    }
    return _sanitize_artifact_value(record)


def _validate_request(request: RepairWriterBenchmarkRequest) -> Path:
    _validate_identifier(request.experiment_id, "experiment_id")
    if not request.cases:
        raise RepairWriterBenchmarkConfigurationError("at least one case is required")
    if not request.plans:
        raise RepairWriterBenchmarkConfigurationError("at least one plan is required")
    if request.runs_per_plan < 1:
        raise RepairWriterBenchmarkConfigurationError("runs_per_plan must be at least 1")
    seen_cases: set[str] = set()
    for case in request.cases:
        _validate_identifier(case.case_id, "case_id")
        if case.case_id in seen_cases:
            raise RepairWriterBenchmarkConfigurationError(
                f"duplicate case_id: {case.case_id}"
            )
        seen_cases.add(case.case_id)
        if case.case_id in DIAGNOSTIC_EXCLUDED_CASE_IDS:
            raise RepairWriterBenchmarkConfigurationError(
                f"diagnostic-only case is excluded from Repair Writer benchmark: {case.case_id}"
            )
        if case.case_id not in CANONICAL_BENCHMARK_CASE_IDS:
            raise RepairWriterBenchmarkConfigurationError(
                f"case is not approved for Repair Writer benchmark: {case.case_id}"
            )
        if not case.canonical_candidate_valid:
            raise RepairWriterBenchmarkConfigurationError(
                f"case is not canonical-valid for Repair Writer benchmark: {case.case_id}"
            )
    seen_plans: set[str] = set()
    for plan in request.plans:
        _validate_plan(plan)
        if plan.plan_id in seen_plans:
            raise RepairWriterBenchmarkConfigurationError(
                f"duplicate plan_id: {plan.plan_id}"
            )
        seen_plans.add(plan.plan_id)
    if request.allow_api:
        _validate_live_prompt_paths()
    output_root = _resolve_output_root(request.output_root)
    output_dir = output_root / request.experiment_id
    if output_dir.exists():
        raise RepairWriterBenchmarkConfigurationError(
            f"benchmark output directory already exists: {output_dir}"
        )
    return output_dir


def _validate_case_payload(payload: dict[str, Any], fixture_path: Path) -> None:
    required = (
        "case_id",
        "source_fixture_path",
        "source_experiment_id",
        "source_git_commit",
        "writer_provider",
        "writer_model",
        "candidate_payload",
        "candidate_post_character_length",
        "canonical_candidate_valid",
        "post_brief",
        "angle_decision",
        "selected_evidence",
        "known_historical_quality_result",
        "repair_instruction",
        "editorial_classification",
        "frozen_input_reconstruction",
        "case_selection_note",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise RepairWriterBenchmarkConfigurationError(
            f"repair benchmark fixture missing required fields: {', '.join(missing)}"
        )
    if payload["case_id"] not in CANONICAL_BENCHMARK_CASE_IDS:
        raise RepairWriterBenchmarkConfigurationError(
            f"case is not approved for Repair Writer benchmark: {payload['case_id']}"
        )
    candidate_payload = payload["candidate_payload"]
    if not isinstance(candidate_payload, dict) or set(candidate_payload) != {"post_text"}:
        raise RepairWriterBenchmarkConfigurationError(
            "candidate_payload must contain only post_text"
        )
    if payload["candidate_post_character_length"] != len(candidate_payload["post_text"]):
        raise RepairWriterBenchmarkConfigurationError(
            "candidate_post_character_length must match post_text"
        )
    if payload["selected_evidence"] != payload["post_brief"].get("evidence_to_use"):
        raise RepairWriterBenchmarkConfigurationError(
            "selected_evidence must match post_brief.evidence_to_use"
        )
    evidence_ids = [item.get("evidence_id") for item in payload["selected_evidence"]]
    if evidence_ids != payload["angle_decision"].get("supporting_evidence_ids"):
        raise RepairWriterBenchmarkConfigurationError(
            "selected evidence IDs must match angle_decision.supporting_evidence_ids"
        )
    quality = payload["known_historical_quality_result"]
    repair_instruction = payload["repair_instruction"]
    if not isinstance(quality, dict) or quality.get("pass") is not False:
        raise RepairWriterBenchmarkConfigurationError(
            f"case must carry a failed historical quality result: {fixture_path}"
        )
    failed_criteria = quality.get("failed_criteria")
    if not isinstance(failed_criteria, list) or not failed_criteria:
        raise RepairWriterBenchmarkConfigurationError(
            "known_historical_quality_result.failed_criteria is required"
        )
    if not isinstance(repair_instruction, dict):
        raise RepairWriterBenchmarkConfigurationError(
            "repair_instruction must be an object"
        )
    if repair_instruction.get("failed_criterion") not in failed_criteria:
        raise RepairWriterBenchmarkConfigurationError(
            "repair_instruction.failed_criterion must match historical failed criteria"
        )
    if not str(repair_instruction.get("repair_instruction") or "").strip():
        raise RepairWriterBenchmarkConfigurationError(
            "repair_instruction.repair_instruction must be non-empty"
        )


def _validate_plan(plan: RepairWriterBenchmarkPlan) -> None:
    _validate_identifier(plan.plan_id, "plan_id")
    failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_REPAIR_WRITER,
        provider=plan.provider,
        model=plan.model,
    )
    if failure is not None:
        raise RepairWriterBenchmarkConfigurationError(str(failure))
    if isinstance(plan.max_output_tokens, bool) or not isinstance(plan.max_output_tokens, int):
        raise RepairWriterBenchmarkConfigurationError("max_output_tokens must be an integer")
    if plan.max_output_tokens != REPAIR_WRITER_MAX_OUTPUT_TOKENS:
        raise RepairWriterBenchmarkConfigurationError(
            f"repair writer max_output_tokens must be {REPAIR_WRITER_MAX_OUTPUT_TOKENS}"
        )
    if plan.json_mode is not REPAIR_WRITER_JSON_MODE:
        raise RepairWriterBenchmarkConfigurationError("repair writer json_mode must be False")
    if plan.execution_profile == REPAIR_WRITER_EXECUTION_PROFILE_PROVIDER_DEFAULT:
        return
    if plan.execution_profile not in REPAIR_WRITER_EXECUTION_PROFILE_REASONING_EFFORTS:
        raise RepairWriterBenchmarkConfigurationError(
            f"unsupported repair writer execution_profile: {plan.execution_profile}"
        )
    if plan.provider != AI_PROVIDER_GEMINI:
        raise RepairWriterBenchmarkConfigurationError(
            "repair writer reasoning execution profiles are supported only for gemini"
        )


def _validate_live_prompt_paths() -> None:
    _prompt_text("prompts/linkedin/final_post_semantic_grounding_evaluator.txt", "semantic grounding")
    quality_contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    _prompt_text(quality_contract.prompt_path, "quality evaluator")


def _repair_execution_request(
    request: RepairWriterBenchmarkRequest,
    case: RepairWriterBenchmarkCase,
    plan: RepairWriterBenchmarkPlan,
    run_index: int,
    render: RepairWriterPromptRender,
) -> RepairWriterExecutionRequest:
    return build_repair_writer_execution_request(
        render,
        prompt_text=case.repair_prompt_text,
        provider=plan.provider,
        model=plan.model,
        max_output_tokens=plan.max_output_tokens,
        reasoning_effort=_repair_writer_reasoning_effort(plan),
        execution_metadata=_execution_metadata(
            request,
            case,
            "repair_writer",
            run_index,
            plan=plan,
        ),
    )


def _execution_metadata(
    request: RepairWriterBenchmarkRequest,
    case: RepairWriterBenchmarkCase,
    stage: str,
    run_index: int,
    *,
    plan: RepairWriterBenchmarkPlan | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "experiment_id": request.experiment_id,
        "case_id": case.case_id,
        "stage": stage,
        "run_index": run_index,
    }
    if stage == "repair_writer" and plan is not None:
        metadata["repair_writer_execution_profile"] = plan.execution_profile
        metadata["repair_writer_reasoning_effort"] = _repair_writer_reasoning_effort(plan)
    return metadata


def _repair_writer_reasoning_effort(plan: RepairWriterBenchmarkPlan) -> str | None:
    return REPAIR_WRITER_EXECUTION_PROFILE_REASONING_EFFORTS.get(
        plan.execution_profile
    )


def _prompt_text(prompt_path: str | None, label: str) -> str:
    if not prompt_path:
        raise RepairWriterBenchmarkConfigurationError(f"{label} prompt path is required")
    base_dir = Path(settings.BASE_DIR).resolve()
    path = (base_dir / prompt_path).resolve()
    if not _is_relative_to(path, base_dir):
        raise RepairWriterBenchmarkConfigurationError(f"{label} prompt path must stay inside repository")
    if not path.exists():
        raise RepairWriterBenchmarkConfigurationError(f"{label} prompt path does not exist")
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise RepairWriterBenchmarkConfigurationError(f"{label} prompt text must be non-empty")
    return text


def _input_parity(
    case: RepairWriterBenchmarkCase,
    render: RepairWriterPromptRender,
) -> dict[str, Any]:
    return {
        "candidate_payload_sha256": _sha256_json(case.candidate_payload),
        "post_brief_sha256": _sha256_json(case.post_brief),
        "angle_decision_sha256": _sha256_json(case.angle_decision),
        "selected_evidence_sha256": _sha256_json(list(case.selected_evidence)),
        "selected_evidence_ids": list(_selected_evidence_ids(case)),
        "repair_prompt_input_sha256": _sha256(render.input_text),
    }


def _render_diagnostics(render: RepairWriterPromptRender) -> dict[str, Any]:
    return {
        "prompt_name": render.prompt_name,
        "prompt_version": render.prompt_version,
        "prompt_path": render.prompt_path,
        "variable_names": sorted(render.variables),
        "input_text_sha256": _sha256(render.input_text),
        "input_text_length": len(render.input_text),
    }


def _payload_preservation(
    original_payload: dict[str, Any],
    repaired_payload: dict[str, Any],
) -> dict[str, Any]:
    original_keys = set(original_payload)
    repaired_keys = set(repaired_payload)
    non_post_text_original = {
        key: value for key, value in original_payload.items() if key != "post_text"
    }
    non_post_text_repaired = {
        key: value for key, value in repaired_payload.items() if key != "post_text"
    }
    original_post_text = str(original_payload.get("post_text") or "")
    repaired_post_text = str(repaired_payload.get("post_text") or "")
    original_length = len(original_post_text)
    repaired_length = len(repaired_post_text)
    character_delta = repaired_length - original_length
    return {
        "only_post_text_changed": (
            original_keys == repaired_keys == {"post_text"}
            and original_payload.get("post_text") != repaired_payload.get("post_text")
        ),
        "non_post_text_fields_identical": non_post_text_original == non_post_text_repaired,
        "original_keys": sorted(original_keys),
        "repaired_keys": sorted(repaired_keys),
        "original_character_count": original_length,
        "repaired_character_count": repaired_length,
        "character_delta": character_delta,
        "character_delta_percent": (
            round((character_delta / original_length) * 100, 2)
            if original_length
            else None
        ),
    }


def _gate_passed(gate_output: Any) -> bool:
    diagnostics = gate_output.diagnostics
    return (
        gate_output.validation_passed
        and diagnostics.deterministic_checks_passed
        and diagnostics.system_linkedin_ready
    )


def _selected_evidence_ids(case: RepairWriterBenchmarkCase) -> tuple[str, ...]:
    return tuple(item["evidence_id"] for item in case.selected_evidence)


def _repair_execution_failure_code(raw_response: RepairWriterRawResponse) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_REPAIR_EMPTY_RESPONSE
    return FAILURE_REPAIR_EXECUTION


def _semantic_execution_failure_code(raw_response: SemanticGroundingRawResponse) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_GROUNDING_EMPTY_RESPONSE
    return FAILURE_GROUNDING_EXECUTION


def _semantic_grounding_failure_metadata(
    raw_response: SemanticGroundingRawResponse,
    error: SemanticGroundingResponseParseError | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "response_diagnostics": build_semantic_grounding_raw_response_diagnostics(
            raw_response
        )
    }
    if error is not None:
        metadata["parser_error_details"] = {
            "code": error.code,
            "line": error.line,
            "column": error.column,
            "position": error.position,
        }
        if error.normalization_error:
            metadata["normalization_error_details"] = {
                "message": _safe_text(error.normalization_error)
            }
    return metadata


def _quality_execution_failure_code(raw_response: QualityEvaluatorRawResponse) -> str:
    if raw_response.execution_error == "empty provider response":
        return FAILURE_QUALITY_EMPTY_RESPONSE
    return FAILURE_QUALITY_EXECUTION


def _raw_response_diagnostics(raw_response: Any) -> dict[str, Any]:
    raw_text = str(raw_response.raw_text or "")
    metadata = _safe_provider_response_metadata(
        getattr(raw_response, "provider_response_metadata", None)
    )
    structure = build_repair_writer_response_structure_diagnostics(
        raw_text,
        provider_output_limit_reached=metadata.get("provider_output_limit_reached"),
    )
    return {
        "provider": raw_response.provider,
        "model": raw_response.model,
        "raw_text_sha256": _sha256(raw_text),
        "raw_text_length": len(raw_text),
        "usage": _safe_token_usage(raw_response.usage or {}),
        "execution_error": raw_response.execution_error,
        "provider_error_diagnostics": sanitize_provider_error_diagnostics(
            getattr(raw_response, "execution_diagnostics", None)
        ),
        **metadata,
        "response_structure_diagnostics": structure,
    }


def _safe_provider_response_metadata(metadata: Any) -> dict[str, Any]:
    result = {
        "provider_finish_reason": None,
        "provider_stop_reason": None,
        "provider_max_output_tokens": None,
        "provider_reported_output_tokens": None,
        "provider_output_limit_reached": None,
        "provider_prompt_tokens": None,
        "provider_visible_output_tokens": None,
        "provider_total_tokens": None,
        "provider_hidden_output_tokens": None,
        "provider_combined_output_tokens": None,
        "provider_output_budget_utilization_percent": None,
        "provider_reasoning_tokens": None,
        "provider_thinking_tokens": None,
    }
    if not isinstance(metadata, dict):
        return result
    for key in result:
        value = metadata.get(key)
        if key == "provider_output_limit_reached":
            if value is None or isinstance(value, bool):
                result[key] = value
        elif key in (
            "provider_max_output_tokens",
            "provider_reported_output_tokens",
            "provider_prompt_tokens",
            "provider_visible_output_tokens",
            "provider_total_tokens",
            "provider_hidden_output_tokens",
            "provider_combined_output_tokens",
            "provider_reasoning_tokens",
            "provider_thinking_tokens",
        ):
            if value is None or (isinstance(value, int) and not isinstance(value, bool)):
                result[key] = value
        elif key == "provider_output_budget_utilization_percent":
            if value is None:
                result[key] = None
            elif isinstance(value, (int, float)) and not isinstance(value, bool):
                result[key] = round(float(value), 2)
        elif value is None or isinstance(value, str):
            result[key] = value[:120] if isinstance(value, str) else None
    return result


def _safe_token_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    safe: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            safe[key] = value
    return safe


def _confirmed_provider_api_call(raw_response: Any) -> int:
    if getattr(raw_response, "usage", None) or getattr(raw_response, "raw_provider_response", None):
        return 1
    return 0


def _provider_calls_for_record(record: dict[str, Any]) -> int:
    counts = record.get("provider_invocation_counts") or {}
    return (
        counts.get("repair_writer_provider_api_calls", 0)
        + counts.get("semantic_grounding_provider_api_calls", 0)
        + counts.get("quality_evaluator_provider_api_calls", 0)
    )


def _artifact_paths(output_dir: Path) -> RepairWriterBenchmarkArtifacts:
    return RepairWriterBenchmarkArtifacts(
        output_dir=str(output_dir),
        manifest_json=str(output_dir / "manifest.json"),
        runs_jsonl=str(output_dir / "runs.jsonl"),
        summary_csv=str(output_dir / "summary.csv"),
        report_md=str(output_dir / "report.md"),
        repair_comparison_md=str(output_dir / "repair_comparison.md"),
    )


def _manifest(
    request: RepairWriterBenchmarkRequest,
    output_dir: Path,
    started_at: str,
) -> dict[str, Any]:
    logical_calls = len(request.cases) * len(request.plans) * request.runs_per_plan
    max_calls = logical_calls * 3
    return _sanitize_artifact_value(
        {
            "schema_version": BENCHMARK_SCHEMA_VERSION,
            "experiment_id": request.experiment_id,
            "benchmark_type": "isolated_repair_writer",
            "created_at": started_at,
            "output_dir": str(output_dir),
            "allow_api": request.allow_api,
            "dry_run": not request.allow_api,
            "runs_per_plan": request.runs_per_plan,
            "cases": [
                {
                    "case_id": case.case_id,
                    "fixture_path": str(case.fixture_path),
                    "source_fixture_path": str(case.source_fixture_path),
                    "editorial_classification": case.editorial_classification,
                    "frozen_input_reconstruction": case.frozen_input_reconstruction,
                    "candidate_post_character_length": case.candidate_post_character_length,
                }
                for case in request.cases
            ],
            "plans": [plan.to_dict() for plan in request.plans],
            "fixed_downstream_roles": {
                "semantic_grounding": {
                    "provider": FIXED_SEMANTIC_GROUNDING_PROVIDER,
                    "model": FIXED_SEMANTIC_GROUNDING_MODEL,
                },
                "quality_evaluator": {
                    "provider": FIXED_QUALITY_EVALUATOR_PROVIDER,
                    "model": FIXED_QUALITY_EVALUATOR_MODEL,
                },
            },
            "repair_writer_selection_rule": {
                "genericization": REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION,
                "criteria": list(REPAIR_WRITER_ANTI_GENERIC_SELECTION_CRITERIA),
                "voice_preservation_diagnostics": list(
                    REPAIR_WRITER_VOICE_PRESERVATION_DIAGNOSTICS
                ),
                "production_selection_changed": False,
            },
            "planned_live_accounting": {
                "logical_repair_runs": logical_calls,
                "planned_repair_writer_calls": logical_calls,
                "planned_semantic_grounding_calls": logical_calls,
                "planned_quality_evaluator_calls": logical_calls,
                "planned_max_provider_calls": max_calls,
            },
            "roles_not_invoked": [
                "Candidate Writer",
                "publication packaging",
            ],
        }
    )


def _write_artifacts(
    artifacts: RepairWriterBenchmarkArtifacts,
    manifest: dict[str, Any],
    records: tuple[dict[str, Any], ...],
) -> None:
    Path(artifacts.manifest_json).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    Path(artifacts.runs_jsonl).write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for record in records
        )
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    _write_summary_csv(Path(artifacts.summary_csv), records)
    Path(artifacts.report_md).write_text(_report_text(manifest, records), encoding="utf-8")
    Path(artifacts.repair_comparison_md).write_text(
        _comparison_text(manifest, records),
        encoding="utf-8",
    )


def _write_summary_csv(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    fields = (
        "case_id",
        "plan_id",
        "provider",
        "model",
        "run_index",
        "execution_status",
        "failure_stage",
        "failure_code",
        "repair_writer_attempts",
        "semantic_grounding_attempts",
        "quality_evaluator_attempts",
        "provider_api_calls",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            counts = record.get("attempt_counts") or {}
            provider_counts = record.get("provider_invocation_counts") or {}
            writer.writerow(
                {
                    "case_id": record.get("case_id"),
                    "plan_id": record.get("plan_id"),
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                    "run_index": record.get("run_index"),
                    "execution_status": record.get("execution_status"),
                    "failure_stage": record.get("failure_stage"),
                    "failure_code": record.get("failure_code"),
                    "repair_writer_attempts": counts.get("repair_writer_attempts", 0),
                    "semantic_grounding_attempts": counts.get("semantic_grounding_attempts", 0),
                    "quality_evaluator_attempts": counts.get("quality_evaluator_attempts", 0),
                    "provider_api_calls": (
                        provider_counts.get("repair_writer_provider_api_calls", 0)
                        + provider_counts.get("semantic_grounding_provider_api_calls", 0)
                        + provider_counts.get("quality_evaluator_provider_api_calls", 0)
                    ),
                }
            )


def _report_text(manifest: dict[str, Any], records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Isolated Repair Writer Benchmark",
        "",
        f"Experiment: `{manifest['experiment_id']}`",
        f"Runs: {len(records)}",
        f"Provider calls: {sum(_provider_calls_for_record(record) for record in records)}",
        "",
        "This artifact fixes initial candidates and varies only Repair Writer provider/model plans.",
        "",
        f"Genericization rule: {REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION}.",
        "A Repair Writer model cannot win solely through accept count, QE total, cost, or structural reliability if it consistently genericizes strong source prose.",
        "",
        "| Case | Plan | Provider | Model | Status | Failure | Outcome |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for record in records:
        projection = record.get("adjudication_projection") or {}
        decision = ((projection.get("decision_ready_result") or {}).get("decision") or {})
        lines.append(
            f"| {record.get('case_id')} | {record.get('plan_id')} | {record.get('provider')} | "
            f"{record.get('model')} | {record.get('execution_status')} | "
            f"{record.get('failure_code') or ''} | {decision.get('action') or ''} |"
        )
    return "\n".join(lines) + "\n"


def _comparison_text(manifest: dict[str, Any], records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Repair Writer Comparison",
        "",
        f"Experiment: `{manifest['experiment_id']}`",
        "",
        "Downstream Semantic Grounding and Quality Evaluator roles are fixed for parity.",
        "",
        f"Genericization rule: {REPAIR_WRITER_SELECTION_BLOCKER_GENERICIZATION}.",
        "Voice preservation and anti-genericness are first-class selection axes for future final comparison.",
        "",
    ]
    for case_id in sorted({str(record.get("case_id")) for record in records}):
        lines.extend([f"## Case `{case_id}`", ""])
        for record in [item for item in records if item.get("case_id") == case_id]:
            preservation = record.get("payload_preservation") or {}
            repair_diagnostics = record.get("repair_adapter_diagnostics") or {}
            grounding = record.get("semantic_grounding") or {}
            quality = record.get("quality_evaluation") or {}
            quality_scores = quality.get("scores") or {}
            projection = record.get("adjudication_projection") or {}
            decision = ((projection.get("decision_ready_result") or {}).get("decision") or {})
            lines.extend(
                [
                    f"### {record.get('plan_id')}",
                    "",
                    f"provider/model: `{record.get('provider')}` / `{record.get('model')}`",
                    f"status: {record.get('execution_status')}",
                    f"failure_code: {record.get('failure_code')}",
                    f"repair_adaptation_success: {record.get('repair_adapter_diagnostics') is not None and record.get('failure_stage') != 'repair_writer_adaptation'}",
                    f"post_text_within_limit: {repair_diagnostics.get('post_text_within_candidate_max_length')}",
                    f"deterministic_gate_passed: {_record_gate_passed(record)}",
                    f"semantic_grounding_passed: {grounding.get('pass')}",
                    f"blocking_claims: {len(grounding.get('failed_claim_ids') or [])}",
                    f"quality_passed: {quality.get('pass')}",
                    f"quality_total_score: {quality.get('total_score')}",
                    f"quality_hook: {quality_scores.get('hook')}",
                    f"quality_controlling_angle: {quality_scores.get('controlling_angle')}",
                    f"quality_evidence: {quality_scores.get('evidence')}",
                    f"quality_author_point_of_view: {quality_scores.get('author_point_of_view')}",
                    f"quality_human_voice: {quality_scores.get('human_voice')}",
                    f"quality_practical_value: {quality_scores.get('practical_value')}",
                    f"quality_cta: {quality_scores.get('cta')}",
                    f"final_action: {decision.get('action')}",
                    f"only_post_text_changed: {preservation.get('only_post_text_changed')}",
                    f"original_character_count: {preservation.get('original_character_count')}",
                    f"repaired_character_count: {preservation.get('repaired_character_count')}",
                    f"character_delta: {preservation.get('character_delta')}",
                    f"character_delta_percent: {preservation.get('character_delta_percent')}",
                    "voice_preservation_note: compare distinctive phrasing, sentence rhythm, authorial specificity, rhetorical structure, and generic transition drift; this is reporting-only and not a deterministic AI detector.",
                    f"repair_target: {record.get('repair_instruction', {}).get('failed_criterion')}",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def _record_gate_passed(record: dict[str, Any]) -> bool | None:
    gate = record.get("repaired_gate")
    if not isinstance(gate, dict):
        return None
    diagnostics = gate.get("diagnostics") or {}
    return (
        gate.get("validation_passed") is True
        and diagnostics.get("deterministic_checks_passed") is True
        and diagnostics.get("system_linkedin_ready") is True
    )


def _resolve_output_root(output_root: Path | None) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    default_root = (base_dir / DEFAULT_OUTPUT_ROOT).resolve()
    if output_root is None:
        _ensure_default_output_root_ignored(base_dir)
        return default_root
    resolved = Path(output_root).resolve()
    if _is_relative_to(resolved, base_dir) and not _is_relative_to(resolved, default_root):
        raise RepairWriterBenchmarkConfigurationError(
            "output_root inside the repository must be under debug_outputs/final_post_repair_writer_benchmarks"
        )
    if _is_relative_to(resolved, default_root):
        _ensure_default_output_root_ignored(base_dir)
    return resolved


def _ensure_default_output_root_ignored(base_dir: Path) -> None:
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists():
        return
    text = gitignore.read_text(encoding="utf-8")
    if "debug_outputs/" not in text:
        raise RepairWriterBenchmarkConfigurationError(
            "debug_outputs/ must be ignored before writing benchmark artifacts"
        )


def _validate_identifier(value: str, field_name: str) -> None:
    if not SAFE_IDENTIFIER_RE.match(str(value or "")):
        raise RepairWriterBenchmarkConfigurationError(f"invalid {field_name}")


def _sanitize_artifact_value(value: Any) -> Any:
    if isinstance(value, dict):
        safe = {}
        for key, item in value.items():
            key_text = str(key)
            if (
                key_text not in SAFE_ARTIFACT_DIAGNOSTIC_KEYS
                and any(fragment in key_text.lower() for fragment in FORBIDDEN_ARTIFACT_KEY_FRAGMENTS)
            ):
                continue
            safe[key_text] = _sanitize_artifact_value(item)
        return safe
    if isinstance(value, list):
        return [_sanitize_artifact_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_artifact_value(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    return value


def _safe_text(value: str) -> str:
    text = " ".join(str(value or "").split())
    lowered = text.lower()
    if any(marker in lowered for marker in ("sk-", "authorization", "api_key", "secret", "password", "credential")):
        return "redacted safe error"
    return text[:500]


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    return _sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False))


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
