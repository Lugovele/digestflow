
"""Isolated benchmark harness for comparing Quality Evaluator models."""
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

from apps.ai.client import AI_PROVIDER_GEMINI, AI_PROVIDER_OPENAI
from services.packaging.linkedin_post_attempt_adjudication import (
    QUALITY_EVALUATION_EXECUTION_FAILED,
    QUALITY_EVALUATION_NORMALIZATION_FAILED,
    QUALITY_EVALUATION_NOT_RUN,
    QUALITY_EVALUATION_PARSE_FAILED,
    QUALITY_EVALUATION_READY,
    FinalPostQualityEvaluationState,
    build_final_post_attempt_outcome_from_gate_and_quality,
)
from services.packaging.linkedin_post_deterministic_gate import run_candidate_post_deterministic_gate
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_contracts import FinalPostAttemptHistory
from services.packaging.linkedin_post_flow_handoffs import CandidateWriterOutput
from services.packaging.linkedin_post_flow_input_builders import build_post_editorial_input
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    OPENAI_FINAL_POST_MODEL,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_provider_diagnostics import (
    sanitize_provider_error_diagnostics,
)
from services.packaging.linkedin_post_prompt_registry import (
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    get_prompt_contract,
    prompt_contract_to_prompt_metadata,
)
from services.packaging.linkedin_post_prompt_renderers import QualityEvaluatorPromptRender
from services.packaging.linkedin_post_prompt_renderers import render_quality_evaluator_prompt_input
from services.packaging.linkedin_post_quality_evaluator_execution import (
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
    QUALITY_CRITERIA,
    get_quality_evaluator_rubric_payload,
)

BENCHMARK_SCHEMA_VERSION = "2026-08-14"
DEFAULT_FIXTURE_ROOT = Path("tests/fixtures/linkedin_post_semantic_grounding_benchmark/claude_sonnet_5_v5")
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/final_post_quality_evaluator_benchmarks")
DEFAULT_EXPERIMENT_ID = "quality-evaluator-gpt-vs-gemini-v2"
BENCHMARK_STATUS_DRY_RUN = "dry_run"
BENCHMARK_STATUS_COMPLETED = "completed"
BENCHMARK_STATUS_CONFIG_ERROR = "config_error"
FAILURE_EXECUTION = "quality_evaluator_execution_failure"
FAILURE_EMPTY_RESPONSE = "quality_evaluator_empty_response"
FAILURE_PARSE = "quality_evaluator_parse_failure"
FAILURE_NORMALIZATION = "quality_evaluator_normalization_failure"
PLAN_GPT_QUALITY = "gpt_quality"
PLAN_GEMINI_QUALITY = "gemini_quality"
CLASSIFICATION_STRONG = "STRONG"
CLASSIFICATION_BORDERLINE = "BORDERLINE"
CLASSIFICATION_WEAK = "WEAK"
CLASSIFICATION_GENERIC = "GENERIC"
CLASSIFICATION_CTA_DIAGNOSTIC = "CTA_DIAGNOSTIC"
CLASSIFICATIONS = (
    CLASSIFICATION_STRONG,
    CLASSIFICATION_BORDERLINE,
    CLASSIFICATION_WEAK,
    CLASSIFICATION_GENERIC,
    CLASSIFICATION_CTA_DIAGNOSTIC,
)
RECONSTRUCTION_DIRECT = "DIRECT"
RECONSTRUCTION_DETERMINISTIC = "DETERMINISTIC_RECONSTRUCTION"
CANONICAL_BENCHMARK_CASE_IDS = (
    "topic_200_digest_134",
    "topic_140_digest_126",
    "topic_140_digest_126__gpt_v3",
    "topic_200_digest_134__gpt_v2",
    "topic_214_digest_128__gpt_v5",
    "topic_214_digest_128__claude_v3",
)
DIAGNOSTIC_EXCLUDED_CASE_IDS = ("topic_214_digest_128",)
DEFAULT_BENCHMARK_CASE_FIXTURES = (
    DEFAULT_FIXTURE_ROOT / "topic_200_digest_134.json",
    DEFAULT_FIXTURE_ROOT / "topic_140_digest_126.json",
    Path("tests/fixtures/linkedin_post_quality_evaluator_benchmark/writer_variants/topic_140_digest_126__gpt_v3.json"),
    Path("tests/fixtures/linkedin_post_quality_evaluator_benchmark/writer_variants/topic_200_digest_134__gpt_v2.json"),
    Path("tests/fixtures/linkedin_post_quality_evaluator_benchmark/writer_variants/topic_214_digest_128__gpt_v5.json"),
    Path("tests/fixtures/linkedin_post_quality_evaluator_benchmark/writer_variants/topic_214_digest_128__claude_v3.json"),
)
APPROVED_CASE_METADATA: dict[str, dict[str, str]] = {
    "topic_200_digest_134": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v5",
        "source_git_commit": "b830bb95e84acb43f00a3f7abdb508d89a584beb",
        "writer_provider": "anthropic",
        "writer_model": "claude-sonnet-5",
        "candidate_post_sha256": "ea906b00cac9764267b8aea65026ea701d9718a47e93ae297a9c999e78a884dc",
        "editorial_classification": CLASSIFICATION_STRONG,
        "frozen_input_reconstruction": RECONSTRUCTION_DIRECT,
        "case_selection_note": "Strong Claude crypto anchor retained from the v1 comparison.",
    },
    "topic_140_digest_126": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v5",
        "source_git_commit": "b830bb95e84acb43f00a3f7abdb508d89a584beb",
        "writer_provider": "anthropic",
        "writer_model": "claude-sonnet-5",
        "candidate_post_sha256": "776d84ea1c136ba9ca01673e8ffc35576278ead861def801421d0ae905aea15a",
        "editorial_classification": CLASSIFICATION_STRONG,
        "frozen_input_reconstruction": RECONSTRUCTION_DIRECT,
        "case_selection_note": "Strong Claude education anchor retained from the v1 comparison.",
    },
    "topic_140_digest_126__gpt_v3": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v3",
        "source_git_commit": "725ce6a78f2e63c6ffcf0ec0ef135240f74fe369",
        "writer_provider": "openai",
        "writer_model": "gpt-4.1-2025-04-14",
        "candidate_post_sha256": "f586d13759b8c03708c78c5d56497a53dfcacf5b1395a615614a615e9ab22c82",
        "editorial_classification": CLASSIFICATION_WEAK,
        "frozen_input_reconstruction": RECONSTRUCTION_DETERMINISTIC,
        "case_selection_note": (
            "Short, generic education post with confused impact/adoption framing; "
            "useful weak mechanically valid contrast against the strong Claude education anchor."
        ),
    },
    "topic_200_digest_134__gpt_v2": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v2",
        "source_git_commit": "20330a07c285fa6f1970ba873127516f92503369",
        "writer_provider": "openai",
        "writer_model": "gpt-4.1-2025-04-14",
        "candidate_post_sha256": "7861d8dbe6aaabd2be992490aab62cd7a88a6c765489eb1fc9d3b60b6b754562",
        "editorial_classification": CLASSIFICATION_BORDERLINE,
        "frozen_input_reconstruction": RECONSTRUCTION_DETERMINISTIC,
        "case_selection_note": (
            "Polished crypto post that historically scored high while still failing author point of view; "
            "useful for detecting over-scoring of smooth generic prose."
        ),
    },
    "topic_214_digest_128__gpt_v5": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v5",
        "source_git_commit": "b830bb95e84acb43f00a3f7abdb508d89a584beb",
        "writer_provider": "openai",
        "writer_model": "gpt-4.1-2025-04-14",
        "candidate_post_sha256": "1fee44a96dce31cd0a8e45f87480004725bfa1bdc75d186560b55827302222dc",
        "editorial_classification": CLASSIFICATION_GENERIC,
        "frozen_input_reconstruction": RECONSTRUCTION_DETERMINISTIC,
        "case_selection_note": (
            "Corporate future-of-work post with weak author presence and unsupported broad claims; "
            "useful weak/generic workplace contrast."
        ),
    },
    "topic_214_digest_128__claude_v3": {
        "source_experiment_id": "writer-claude-vs-gpt-candidatepost-v3",
        "source_git_commit": "725ce6a78f2e63c6ffcf0ec0ef135240f74fe369",
        "writer_provider": "anthropic",
        "writer_model": "claude-sonnet-5",
        "candidate_post_sha256": "b60cf818e00c1ea2f449a44ac989af03490ea1a407e5b48a3046f55b468c6873",
        "editorial_classification": CLASSIFICATION_CTA_DIAGNOSTIC,
        "frozen_input_reconstruction": RECONSTRUCTION_DETERMINISTIC,
        "case_selection_note": (
            "Strong human-voice workplace post with historical CTA failure; "
            "useful diagnostic for literal versus reflective CTA interpretation."
        ),
    },
}
DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS = 2400
GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS = 4800
QUALITY_EVALUATOR_PARSER_PATH = "services.packaging.linkedin_post_quality_evaluator_parser.parse_and_normalize_quality_evaluator_response"
QUALITY_REVIEW_NORMALIZATION_PATH = "services.packaging.linkedin_post_quality_review_contract.normalize_quality_review_result"
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "api_key", "secret", "password", "credential", "header",
    "provider_payload", "provider_reply", "prompt_text", "raw_text",
    "raw_provider_response",
)


@dataclass(frozen=True)
class QualityEvaluatorBenchmarkCase:
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
    editorial_classification: str
    frozen_input_reconstruction: str
    case_selection_note: str
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
            "editorial_classification": self.editorial_classification,
            "frozen_input_reconstruction": self.frozen_input_reconstruction,
            "case_selection_note": self.case_selection_note,
            "canonical_exclusion": copy.deepcopy(self.canonical_exclusion),
        }


@dataclass(frozen=True)
class QualityEvaluatorBenchmarkPlan:
    plan_id: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS
    json_mode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "json_mode": self.json_mode,
        }

@dataclass(frozen=True)
class QualityEvaluatorBenchmarkRequest:
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    cases: tuple[QualityEvaluatorBenchmarkCase, ...] = ()
    plans: tuple[QualityEvaluatorBenchmarkPlan, ...] = ()
    runs_per_plan: int = 1
    allow_api: bool = False
    output_root: Path | None = None


@dataclass(frozen=True)
class QualityEvaluatorBenchmarkArtifacts:
    output_dir: str
    runs_jsonl: str
    summary_csv: str
    report_md: str
    manifest_json: str
    quality_comparison_md: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "runs_jsonl": self.runs_jsonl,
            "summary_csv": self.summary_csv,
            "report_md": self.report_md,
            "manifest_json": self.manifest_json,
            "quality_comparison_md": self.quality_comparison_md,
        }


@dataclass(frozen=True)
class QualityEvaluatorBenchmarkResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    provider_call_count: int
    artifacts: QualityEvaluatorBenchmarkArtifacts | None
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


class QualityEvaluatorBenchmarkConfigurationError(ValueError):
    pass


NowFactory = Callable[[], datetime]
QualityEvaluatorExecutor = Callable[[QualityEvaluatorExecutionRequest], QualityEvaluatorRawResponse]


def load_quality_evaluator_benchmark_case(path: str | Path) -> QualityEvaluatorBenchmarkCase:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    _validate_case_payload(payload, fixture_path)
    metadata_payload = APPROVED_CASE_METADATA[payload["case_id"]]
    metadata = prompt_contract_to_prompt_metadata(
        get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
    )
    return QualityEvaluatorBenchmarkCase(
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
        prompt_metadata=metadata,
        candidate_validity_status=payload["candidate_validity_status"],
        editorial_classification=payload.get(
            "editorial_classification",
            metadata_payload["editorial_classification"],
        ),
        frozen_input_reconstruction=payload.get(
            "frozen_input_reconstruction",
            metadata_payload["frozen_input_reconstruction"],
        ),
        case_selection_note=payload.get(
            "case_selection_note",
            metadata_payload["case_selection_note"],
        ),
        canonical_exclusion=copy.deepcopy(payload.get("canonical_exclusion")),
    )


def default_quality_evaluator_benchmark_cases(
    fixture_root: Path | None = None,
) -> tuple[QualityEvaluatorBenchmarkCase, ...]:
    if fixture_root is None:
        return tuple(
            load_quality_evaluator_benchmark_case(path)
            for path in DEFAULT_BENCHMARK_CASE_FIXTURES
        )
    root = Path(fixture_root)
    return tuple(
        load_quality_evaluator_benchmark_case(root / path.relative_to("tests/fixtures"))
        for path in DEFAULT_BENCHMARK_CASE_FIXTURES
    )


def quality_evaluator_benchmark_max_output_tokens_for_provider(provider: str) -> int:
    if str(provider or "").strip().lower() == AI_PROVIDER_GEMINI:
        return GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS
    return DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS


def default_quality_evaluator_benchmark_plans() -> tuple[QualityEvaluatorBenchmarkPlan, ...]:
    return (
        QualityEvaluatorBenchmarkPlan(PLAN_GPT_QUALITY, AI_PROVIDER_OPENAI, OPENAI_FINAL_POST_MODEL),
        QualityEvaluatorBenchmarkPlan(
            PLAN_GEMINI_QUALITY,
            AI_PROVIDER_GEMINI,
            "gemini-3.6-flash",
            max_output_tokens=GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
        ),
    )


def build_quality_evaluator_benchmark_prompt_render(
    case: QualityEvaluatorBenchmarkCase,
) -> QualityEvaluatorPromptRender:
    return render_quality_evaluator_prompt_input(
        _post_editorial_input_for_case(case),
        get_quality_evaluator_rubric_payload(),
        prompt_metadata=case.prompt_metadata,
    )


def run_quality_evaluator_benchmark(
    request: QualityEvaluatorBenchmarkRequest,
    *,
    now_factory: NowFactory | None = None,
    quality_evaluator_executor: QualityEvaluatorExecutor | None = None,
) -> QualityEvaluatorBenchmarkResult:
    try:
        output_dir = _validate_request(request)
    except QualityEvaluatorBenchmarkConfigurationError as exc:
        return QualityEvaluatorBenchmarkResult(
            BENCHMARK_STATUS_CONFIG_ERROR,
            1,
            str(request.experiment_id or ""),
            0,
            0,
            None,
            "quality_evaluator_benchmark_configuration_error",
            _safe_text(str(exc)),
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
                render = build_quality_evaluator_benchmark_prompt_render(case)
                if request.allow_api:
                    record = _live_record(
                        request, case, plan, run_index, started_at,
                        _isoformat(now()), render, quality_evaluator_executor,
                    )
                    provider_calls += record["provider_invocation_counts"]["quality_evaluator_provider_api_calls"]
                else:
                    record = _dry_record(
                        request, case, plan, run_index, started_at,
                        _isoformat(now()), render,
                    )
                records.append(record)
    manifest = _manifest(request, output_dir, started_at)
    _write_artifacts(artifacts, manifest, tuple(records))
    status = BENCHMARK_STATUS_COMPLETED if request.allow_api else BENCHMARK_STATUS_DRY_RUN
    return QualityEvaluatorBenchmarkResult(
        status, 0, request.experiment_id, len(records), provider_calls,
        artifacts, run_records=tuple(copy.deepcopy(records))
    )


def _post_editorial_input_for_case(case: QualityEvaluatorBenchmarkCase):
    candidate_output = _candidate_output_for_case(case)
    gate_output = run_candidate_post_deterministic_gate(
        candidate_output,
        selected_evidence_ids=_selected_evidence_ids(case),
    )
    return build_post_editorial_input(
        post_brief=copy.deepcopy(case.post_brief),
        angle_decision=copy.deepcopy(case.angle_decision),
        candidate_output=candidate_output,
        gate_output=gate_output,
    )


def _candidate_output_for_case(case: QualityEvaluatorBenchmarkCase) -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=copy.deepcopy(case.candidate_payload),
        raw_output=None,
        provider=case.writer_provider,
        model=case.writer_model,
        prompt_name="frozen_candidate_writer_output",
        prompt_version=case.source_experiment_id,
        token_usage=None,
        cost_metadata=None,
    )


def _validate_request(request: QualityEvaluatorBenchmarkRequest) -> Path:
    _validate_identifier(request.experiment_id, "experiment_id")
    if not request.cases:
        raise QualityEvaluatorBenchmarkConfigurationError("at least one case is required")
    if not request.plans:
        raise QualityEvaluatorBenchmarkConfigurationError("at least one plan is required")
    if request.runs_per_plan < 1:
        raise QualityEvaluatorBenchmarkConfigurationError("runs_per_plan must be at least 1")
    seen_cases: set[str] = set()
    for case in request.cases:
        _validate_identifier(case.case_id, "case_id")
        if case.case_id in seen_cases:
            raise QualityEvaluatorBenchmarkConfigurationError(f"duplicate case_id: {case.case_id}")
        seen_cases.add(case.case_id)
        if case.case_id not in CANONICAL_BENCHMARK_CASE_IDS:
            if case.case_id in DIAGNOSTIC_EXCLUDED_CASE_IDS:
                raise QualityEvaluatorBenchmarkConfigurationError(
                    f"diagnostic-only case is excluded from Quality Evaluator benchmark: {case.case_id}"
                )
            raise QualityEvaluatorBenchmarkConfigurationError(
                f"case is not approved for Quality Evaluator benchmark: {case.case_id}"
            )
        if not case.canonical_candidate_valid:
            raise QualityEvaluatorBenchmarkConfigurationError(
                f"case is not canonical-valid for Quality Evaluator benchmark: {case.case_id}"
            )
        _validate_case_matches_approved_metadata(
            case_id=case.case_id,
            source_experiment_id=case.source_experiment_id,
            source_git_commit=case.source_git_commit,
            writer_provider=case.writer_provider,
            writer_model=case.writer_model,
            candidate_post_text=str(case.candidate_payload.get("post_text") or ""),
            editorial_classification=case.editorial_classification,
            frozen_input_reconstruction=case.frozen_input_reconstruction,
            case_selection_note=case.case_selection_note,
        )
    seen_plans: set[str] = set()
    for plan in request.plans:
        _validate_plan(plan)
        if plan.plan_id in seen_plans:
            raise QualityEvaluatorBenchmarkConfigurationError(f"duplicate plan_id: {plan.plan_id}")
        seen_plans.add(plan.plan_id)
    if request.allow_api:
        _validate_live_prompt_paths(request)
    output_root = _resolve_output_root(request.output_root)
    output_dir = output_root / request.experiment_id
    if output_dir.exists():
        raise QualityEvaluatorBenchmarkConfigurationError(
            f"benchmark output directory already exists: {output_dir}"
        )
    return output_dir


def _validate_case_payload(payload: dict[str, Any], fixture_path: Path) -> None:
    if not isinstance(payload, dict):
        raise QualityEvaluatorBenchmarkConfigurationError(f"benchmark fixture must be an object: {fixture_path}")
    required = (
        "case_id", "source_experiment_id", "source_git_commit", "writer_provider",
        "writer_model", "candidate_payload", "candidate_post_character_length",
        "canonical_candidate_valid", "candidate_validity_status", "post_brief",
        "angle_decision", "selected_evidence",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise QualityEvaluatorBenchmarkConfigurationError("benchmark fixture missing required fields: " + ", ".join(missing))
    if payload["case_id"] not in APPROVED_CASE_METADATA:
        raise QualityEvaluatorBenchmarkConfigurationError(
            f"case is not approved for Quality Evaluator benchmark: {payload['case_id']}"
        )
    candidate_payload = payload["candidate_payload"]
    if not isinstance(candidate_payload, dict) or set(candidate_payload) != {"post_text"}:
        raise QualityEvaluatorBenchmarkConfigurationError("candidate_payload must contain only post_text")
    post_text = candidate_payload["post_text"]
    if not isinstance(post_text, str) or not post_text.strip():
        raise QualityEvaluatorBenchmarkConfigurationError("candidate_payload.post_text must be non-empty")
    if payload["candidate_post_character_length"] != len(post_text):
        raise QualityEvaluatorBenchmarkConfigurationError("candidate_post_character_length must match post_text")
    metadata = APPROVED_CASE_METADATA[payload["case_id"]]
    _validate_case_matches_approved_metadata(
        case_id=payload["case_id"],
        source_experiment_id=payload["source_experiment_id"],
        source_git_commit=payload["source_git_commit"],
        writer_provider=payload["writer_provider"],
        writer_model=payload["writer_model"],
        candidate_post_text=post_text,
        editorial_classification=payload.get("editorial_classification", metadata["editorial_classification"]),
        frozen_input_reconstruction=payload.get("frozen_input_reconstruction", metadata["frozen_input_reconstruction"]),
        case_selection_note=payload.get("case_selection_note", metadata["case_selection_note"]),
    )
    if not isinstance(payload["canonical_candidate_valid"], bool):
        raise QualityEvaluatorBenchmarkConfigurationError("canonical_candidate_valid must be a boolean")
    selected_evidence = payload["selected_evidence"]
    post_brief = payload["post_brief"]
    angle_decision = payload["angle_decision"]
    if not isinstance(selected_evidence, list) or not selected_evidence:
        raise QualityEvaluatorBenchmarkConfigurationError("selected_evidence is required")
    if not isinstance(post_brief, dict) or not isinstance(angle_decision, dict):
        raise QualityEvaluatorBenchmarkConfigurationError("post_brief and angle_decision must be objects")
    if selected_evidence != post_brief.get("evidence_to_use"):
        raise QualityEvaluatorBenchmarkConfigurationError("selected_evidence must match post_brief.evidence_to_use")
    evidence_ids = [item.get("evidence_id") for item in selected_evidence]
    if evidence_ids != angle_decision.get("supporting_evidence_ids"):
        raise QualityEvaluatorBenchmarkConfigurationError("selected evidence IDs must match AngleDecision")


def _validate_case_matches_approved_metadata(
    *,
    case_id: str,
    source_experiment_id: str,
    source_git_commit: str,
    writer_provider: str,
    writer_model: str,
    candidate_post_text: str,
    editorial_classification: str,
    frozen_input_reconstruction: str,
    case_selection_note: str,
) -> None:
    metadata = APPROVED_CASE_METADATA[case_id]
    expected_fields = {
        "source_experiment_id": source_experiment_id,
        "source_git_commit": source_git_commit,
        "writer_provider": writer_provider,
        "writer_model": writer_model,
        "editorial_classification": editorial_classification,
        "frozen_input_reconstruction": frozen_input_reconstruction,
        "case_selection_note": case_selection_note,
    }
    for field, value in expected_fields.items():
        if value != metadata[field]:
            raise QualityEvaluatorBenchmarkConfigurationError(f"unexpected {field}")
    if _sha256(candidate_post_text) != metadata["candidate_post_sha256"]:
        raise QualityEvaluatorBenchmarkConfigurationError("unexpected candidate_payload.post_text")


def _validate_live_prompt_paths(request: QualityEvaluatorBenchmarkRequest) -> None:
    base_dir = Path(settings.BASE_DIR).resolve()
    for case in request.cases:
        prompt_path = case.prompt_metadata.prompt_path
        if not prompt_path:
            raise QualityEvaluatorBenchmarkConfigurationError("quality evaluator prompt path is required")
        path = (base_dir / prompt_path).resolve()
        if not _is_relative_to(path, base_dir):
            raise QualityEvaluatorBenchmarkConfigurationError("quality evaluator prompt path must stay inside repository")
        if not path.exists() or not path.read_text(encoding="utf-8").strip():
            raise QualityEvaluatorBenchmarkConfigurationError("quality evaluator prompt text must exist")


def _validate_plan(plan: QualityEvaluatorBenchmarkPlan) -> None:
    _validate_identifier(plan.plan_id, "plan_id")
    failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
        provider=plan.provider,
        model=plan.model,
    )
    if failure is not None:
        raise QualityEvaluatorBenchmarkConfigurationError(str(failure))
    if isinstance(plan.max_output_tokens, bool) or not isinstance(plan.max_output_tokens, int):
        raise QualityEvaluatorBenchmarkConfigurationError("max_output_tokens must be an integer")
    if plan.max_output_tokens < 2000:
        raise QualityEvaluatorBenchmarkConfigurationError("max_output_tokens must be at least 2000")
    if not isinstance(plan.json_mode, bool):
        raise QualityEvaluatorBenchmarkConfigurationError("json_mode must be a boolean")


def _dry_record(
    request: QualityEvaluatorBenchmarkRequest,
    case: QualityEvaluatorBenchmarkCase,
    plan: QualityEvaluatorBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: QualityEvaluatorPromptRender,
) -> dict[str, Any]:
    request_error = get_quality_evaluator_execution_request_error(_execution_request(render, plan))
    return _record(
        request, case, plan, run_index, started_at, completed_at, render,
        execution_status=BENCHMARK_STATUS_DRY_RUN,
        execution_success=None,
        parse_success=None,
        normalization_success=None,
        failure_stage=None,
        failure_code=None,
        execution_request_error=request_error,
        quality_evaluator_attempts=0,
        provider_api_calls=0,
    )


def _live_record(
    request: QualityEvaluatorBenchmarkRequest,
    case: QualityEvaluatorBenchmarkCase,
    plan: QualityEvaluatorBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: QualityEvaluatorPromptRender,
    executor: QualityEvaluatorExecutor | None,
) -> dict[str, Any]:
    raw_response = (executor or execute_quality_evaluator_prompt)(_execution_request(render, plan))
    if raw_response.execution_error:
        return _record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            execution_success=False,
            parse_success=False,
            normalization_success=False,
            failure_stage="execution",
            failure_code=_execution_failure_code(raw_response),
            response_diagnostics=_raw_response_diagnostics(raw_response),
            quality_evaluator_attempts=1,
            provider_api_calls=_confirmed_provider_api_call(raw_response),
        )
    try:
        quality_review = parse_and_normalize_quality_evaluator_response(raw_response)
    except QualityEvaluatorResponseParseError as exc:
        normalization = exc.code == ERROR_NORMALIZATION_FAILED
        return _record(
            request, case, plan, run_index, started_at, completed_at, render,
            execution_status=BENCHMARK_STATUS_COMPLETED,
            execution_success=True,
            parse_success=normalization,
            normalization_success=False,
            failure_stage="normalization" if normalization else "parse",
            failure_code=FAILURE_NORMALIZATION if normalization else FAILURE_PARSE,
            parser_error_details={"code": exc.code, "message": _safe_text(str(exc))},
            response_diagnostics=_raw_response_diagnostics(raw_response),
            quality_evaluator_attempts=1,
            provider_api_calls=_confirmed_provider_api_call(raw_response),
        )
    quality_evaluation = FinalPostQualityEvaluationState(
        status=QUALITY_EVALUATION_READY,
        quality_review=quality_review,
        metadata={"plan_id": plan.plan_id},
    )
    return _record(
        request, case, plan, run_index, started_at, completed_at, render,
        execution_status=BENCHMARK_STATUS_COMPLETED,
        execution_success=True,
        parse_success=True,
        normalization_success=True,
        failure_stage=None,
        failure_code=None,
        quality_review=quality_review,
        adjudication_projection=_adjudication_projection(case, quality_evaluation),
        response_diagnostics=_raw_response_diagnostics(raw_response),
        quality_evaluator_attempts=1,
        provider_api_calls=_confirmed_provider_api_call(raw_response),
    )


def _execution_request(render: QualityEvaluatorPromptRender, plan: QualityEvaluatorBenchmarkPlan) -> QualityEvaluatorExecutionRequest:
    return build_quality_evaluator_execution_request(
        render,
        prompt_text=_quality_evaluator_prompt_text(render),
        provider=plan.provider,
        model=plan.model,
        max_output_tokens=plan.max_output_tokens,
        json_mode=plan.json_mode,
        execution_metadata={"benchmark_plan_id": plan.plan_id},
    )


def _quality_evaluator_prompt_text(render: QualityEvaluatorPromptRender) -> str:
    if not render.prompt_path:
        raise QualityEvaluatorBenchmarkConfigurationError("quality evaluator prompt path is required")
    base_dir = Path(settings.BASE_DIR).resolve()
    path = (base_dir / render.prompt_path).resolve()
    if not _is_relative_to(path, base_dir):
        raise QualityEvaluatorBenchmarkConfigurationError("quality evaluator prompt path must stay inside repository")
    return path.read_text(encoding="utf-8")


def _adjudication_projection(
    case: QualityEvaluatorBenchmarkCase,
    quality_evaluation: FinalPostQualityEvaluationState,
) -> dict[str, Any]:
    candidate_output = _candidate_output_for_case(case)
    gate_output = run_candidate_post_deterministic_gate(
        candidate_output,
        selected_evidence_ids=_selected_evidence_ids(case),
    )
    outcome = build_final_post_attempt_outcome_from_gate_and_quality(
        post_brief=copy.deepcopy(case.post_brief),
        candidate_output=candidate_output,
        gate_output=gate_output,
        quality_evaluation=quality_evaluation,
        attempt_index=1,
        attempt_history=FinalPostAttemptHistory(attempts=[]),
    )
    return {
        "decision": outcome.decision.to_dict(),
        "accepted": outcome.accepted_result is not None,
        "terminal": outcome.terminal_result is not None,
        "repair_required": outcome.accepted_result is None and outcome.terminal_result is None,
    }


def _record(
    request: QualityEvaluatorBenchmarkRequest,
    case: QualityEvaluatorBenchmarkCase,
    plan: QualityEvaluatorBenchmarkPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: QualityEvaluatorPromptRender,
    *,
    execution_status: str,
    execution_success: bool | None,
    parse_success: bool | None,
    normalization_success: bool | None,
    failure_stage: str | None,
    failure_code: str | None,
    quality_evaluator_attempts: int,
    provider_api_calls: int,
    execution_request_error: str | None = None,
    quality_review: dict[str, Any] | None = None,
    adjudication_projection: dict[str, Any] | None = None,
    parser_error_details: dict[str, Any] | None = None,
    response_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        "source_experiment_id": case.source_experiment_id,
        "source_git_commit": case.source_git_commit,
        "writer_provider": case.writer_provider,
        "writer_model": case.writer_model,
        "editorial_classification": case.editorial_classification,
        "frozen_input_reconstruction": case.frozen_input_reconstruction,
        "case_selection_note": case.case_selection_note,
        "candidate_post_character_length": case.candidate_post_character_length,
        "selected_evidence_ids": list(_selected_evidence_ids(case)),
        "quality_input_summary": _quality_input_summary(render),
        "candidate_payload_summary": _candidate_payload_summary(case),
        "rendered_variable_names": sorted(render.variables),
        "prompt_metadata": {
            "prompt_name": render.prompt_name,
            "prompt_version": render.prompt_version,
            "prompt_path": render.prompt_path,
        },
        "execution_request_error": execution_request_error,
        "parser_path": QUALITY_EVALUATOR_PARSER_PATH,
        "normalization_path": QUALITY_REVIEW_NORMALIZATION_PATH,
        "parse_success": parse_success,
        "normalization_success": normalization_success,
        "failure_stage": failure_stage,
        "failure_code": failure_code,
        "quality_evaluation_status": _quality_status(quality_review, failure_stage),
        "parser_error_details": copy.deepcopy(parser_error_details),
        "response_diagnostics": copy.deepcopy(response_diagnostics),
        "quality_review": _quality_review_summary(quality_review),
        "adjudication_projection": _adjudication_projection_summary(
            adjudication_projection
        ),
        "provider_invocation_counts": {
            "candidate_writer": 0,
            "semantic_grounding": 0,
            "quality_evaluator": provider_api_calls,
            "quality_evaluator_attempts": quality_evaluator_attempts,
            "quality_evaluator_provider_api_calls": provider_api_calls,
            "repair_writer": 0,
            "publication_packaging": 0,
        },
    })


def _quality_status(review: dict[str, Any] | None, failure_stage: str | None) -> str:
    if review is not None:
        return QUALITY_EVALUATION_READY
    if failure_stage == "execution":
        return QUALITY_EVALUATION_EXECUTION_FAILED
    if failure_stage == "parse":
        return QUALITY_EVALUATION_PARSE_FAILED
    if failure_stage == "normalization":
        return QUALITY_EVALUATION_NORMALIZATION_FAILED
    return QUALITY_EVALUATION_NOT_RUN


def _quality_review_summary(review: dict[str, Any] | None) -> dict[str, Any] | None:
    if review is None:
        return None
    return {
        "scores": {key: review["scores"].get(key) for key in QUALITY_CRITERIA},
        "total_score": review.get("total_score"),
        "pass": review.get("pass"),
        "failed_criteria": copy.deepcopy(review.get("failed_criteria", [])),
        "automatic_fail_reason_summary": _text_digest(
            review.get("automatic_fail_reason", "")
        ),
        "criterion_rationales": _criterion_rationale_summaries(
            review.get("criterion_rationales", {})
        ),
        "notes_summary": _text_list_digest(review.get("notes", [])),
        "requires_human_review": review.get("requires_human_review"),
        "human_review_reason_summary": _text_digest(
            review.get("human_review_reason", "")
        ),
    }


def _adjudication_projection_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    decision = value.get("decision")
    accepted = bool(value.get("accepted"))
    terminal = bool(value.get("terminal"))
    repair_required = bool(value.get("repair_required"))
    return {
        "outcome_status": _adjudication_outcome_status(
            accepted=accepted,
            terminal=terminal,
            repair_required=repair_required,
        ),
        "accepted": accepted,
        "terminal": terminal,
        "repair_required": repair_required,
        "decision": _decision_summary(decision),
    }


def _adjudication_outcome_status(
    *, accepted: bool, terminal: bool, repair_required: bool
) -> str:
    if accepted:
        return "accepted"
    if repair_required:
        return "repair_required"
    if terminal:
        return "terminal"
    return "unknown"


def _decision_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    return {
        "action": value.get("action"),
        "reason_summary": _text_digest(value.get("reason", "")),
        "repair_type": value.get("repair_type"),
        "target_model_provider": value.get("target_model_provider"),
        "target_model_name": value.get("target_model_name"),
        "needs_human_review": value.get("needs_human_review"),
    }


def _criterion_rationale_summaries(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    summaries: dict[str, Any] = {}
    for criterion in QUALITY_CRITERIA:
        item = value.get(criterion)
        if not isinstance(item, dict):
            continue
        summaries[criterion] = {
            "score": item.get("score"),
            "max_score": item.get("max_score"),
            "rationale_summary": _text_digest(item.get("rationale", "")),
            "post_text_evidence_summary": _text_digest(
                item.get("post_text_evidence", "")
            ),
            "failure_reason_summary": _text_digest(item.get("failure_reason", "")),
        }
    return summaries


def _text_list_digest(value: Any) -> dict[str, Any]:
    if not isinstance(value, (list, tuple)):
        return {"count": 0, "items": []}
    return {
        "count": len(value),
        "items": [_text_digest(item) for item in value],
    }


def _text_digest(value: Any) -> dict[str, Any]:
    text = str(value or "")
    return {
        "character_count": len(text),
        "sha256": _sha256(text) if text else "",
        "present": bool(text.strip()),
    }


def _raw_response_diagnostics(raw_response: QualityEvaluatorRawResponse) -> dict[str, Any]:
    raw = str(raw_response.raw_text or "")
    stripped = raw.strip()
    provider_metadata = _safe_provider_response_metadata(
        getattr(raw_response, "provider_response_metadata", None)
    )
    diagnostics: dict[str, Any] = {
        "raw_response_character_count": len(raw),
        "stripped_response_character_count": len(stripped),
        "starts_with_json_object": stripped.startswith("{"),
        "starts_with_code_fence": stripped.startswith("```"),
        "ends_with_json_object": stripped.endswith("}"),
        "ends_with_code_fence": stripped.endswith("```"),
        "brace_balance": raw.count("{") - raw.count("}"),
        "provider_error_diagnostics": sanitize_provider_error_diagnostics(
            getattr(raw_response, "execution_diagnostics", None)
        ),
        **provider_metadata,
    }
    if isinstance(raw_response.usage, dict):
        diagnostics["usage"] = _safe_token_usage(raw_response.usage)
    return diagnostics


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


def _quality_input_summary(render: QualityEvaluatorPromptRender) -> dict[str, Any]:
    return {
        name: {"sha256": _sha256(value), "character_count": len(value)}
        for name, value in sorted(render.variables.items())
    }


def _candidate_payload_summary(case: QualityEvaluatorBenchmarkCase) -> dict[str, Any]:
    post_text = str(case.candidate_payload.get("post_text", ""))
    return {"post_text_sha256": _sha256(post_text), "post_text_character_count": len(post_text)}



def _confirmed_provider_api_call(raw_response: QualityEvaluatorRawResponse) -> int:
    if raw_response.usage or raw_response.raw_provider_response or str(raw_response.raw_text or "").strip():
        return 1
    return 0


def _execution_failure_code(raw_response: QualityEvaluatorRawResponse) -> str:
    if not str(raw_response.raw_text or "").strip():
        return FAILURE_EMPTY_RESPONSE
    return FAILURE_EXECUTION


def _artifact_paths(output_dir: Path) -> QualityEvaluatorBenchmarkArtifacts:
    return QualityEvaluatorBenchmarkArtifacts(
        output_dir=str(output_dir),
        runs_jsonl=str(output_dir / "runs.jsonl"),
        summary_csv=str(output_dir / "summary.csv"),
        report_md=str(output_dir / "report.md"),
        manifest_json=str(output_dir / "manifest.json"),
        quality_comparison_md=str(output_dir / "quality_comparison.md"),
    )


def _manifest(request: QualityEvaluatorBenchmarkRequest, output_dir: Path, started_at: str) -> dict[str, Any]:
    return _sanitize_artifact_value({
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "experiment_id": request.experiment_id,
        "benchmark_type": "isolated_quality_evaluator",
        "source_experiment_ids": sorted({case.source_experiment_id for case in request.cases}),
        "source_git_commits": sorted({case.source_git_commit for case in request.cases}),
        "created_at": started_at,
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
                "editorial_classification": case.editorial_classification,
                "frozen_input_reconstruction": case.frozen_input_reconstruction,
                "case_selection_note": case.case_selection_note,
                "writer_provider": case.writer_provider,
                "writer_model": case.writer_model,
            }
            for case in request.cases
        ],
        "plans": [plan.to_dict() for plan in request.plans],
        "planned_live_quality_evaluator_calls": _planned_live_provider_calls(request),
        "roles_not_invoked": [
            "Candidate Writer", "Semantic Grounding", "Repair Writer", "publication packaging",
        ],
    })


def _write_artifacts(
    artifacts: QualityEvaluatorBenchmarkArtifacts,
    manifest: dict[str, Any],
    records: tuple[dict[str, Any], ...],
) -> None:
    Path(artifacts.manifest_json).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    Path(artifacts.runs_jsonl).write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) for record in records)
        + ("\n" if records else ""),
        encoding="utf-8",
    )
    _write_summary_csv(Path(artifacts.summary_csv), records)
    Path(artifacts.report_md).write_text(_report_text(manifest, records), encoding="utf-8")
    Path(artifacts.quality_comparison_md).write_text(_comparison_text(manifest, records), encoding="utf-8")


def _write_summary_csv(path: Path, records: tuple[dict[str, Any], ...]) -> None:
    fields = (
        "case_id", "editorial_classification", "plan_id", "provider", "model",
        "run_index", "execution_status", "parse_success",
        "normalization_success", "total_score", "pass",
        "failed_criteria", "quality_evaluator_attempts", "provider_api_calls",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            review = record.get("quality_review") or {}
            counts = record.get("provider_invocation_counts") or {}
            writer.writerow({
                "case_id": record.get("case_id"),
                "editorial_classification": record.get("editorial_classification"),
                "plan_id": record.get("plan_id"),
                "provider": record.get("provider"),
                "model": record.get("model"),
                "run_index": record.get("run_index"),
                "execution_status": record.get("execution_status"),
                "parse_success": record.get("parse_success"),
                "normalization_success": record.get("normalization_success"),
                "total_score": review.get("total_score"),
                "pass": review.get("pass"),
                "failed_criteria": ",".join(review.get("failed_criteria") or []),
                "quality_evaluator_attempts": counts.get("quality_evaluator_attempts", 0),
                "provider_api_calls": counts.get("quality_evaluator_provider_api_calls", 0),
            })


def _report_text(manifest: dict[str, Any], records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Isolated Quality Evaluator Benchmark", "",
        f"Experiment: `{manifest['experiment_id']}`",
        f"Runs: {len(records)}",
        f"Provider calls: {_provider_calls(records)}", "",
        "This artifact fixes candidate final post text and varies only Quality Evaluator provider/model plans.", "",
        "| Case | Classification | Plan | Provider | Model | Status | Total | Pass | Decision |",
        "| --- | --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for record in records:
        review = record.get("quality_review") or {}
        decision = record.get("adjudication_projection") or {}
        action = (decision.get("decision") or {}).get("action")
        lines.append(
            f"| {record.get('case_id')} | {record.get('editorial_classification')} | "
            f"{record.get('plan_id')} | {record.get('provider')} | "
            f"{record.get('model')} | {record.get('execution_status')} | {review.get('total_score')} | "
            f"{review.get('pass')} | {action} |"
        )
    lines.extend(_discrimination_summary_lines(records))
    return "\n".join(lines) + "\n"


def _comparison_text(manifest: dict[str, Any], records: tuple[dict[str, Any], ...]) -> str:
    lines = [
        "# Quality Evaluator Comparison", "",
        f"Experiment: `{manifest['experiment_id']}`", "",
        "Quality labels are local to each case. This artifact does not select a winner.", "",
    ]
    for case_id in sorted({str(record.get("case_id")) for record in records}):
        case_records = [record for record in records if record.get("case_id") == case_id]
        lines.extend([f"## Case `{case_id}`", ""])
        for record in case_records:
            review = record.get("quality_review") or {}
            decision = record.get("adjudication_projection") or {}
            lines.extend([
                f"### {record.get('plan_id')}", "",
                f"provider/model: `{record.get('provider')}` / `{record.get('model')}`",
                f"total_score: {review.get('total_score')}",
                f"pass: {review.get('pass')}",
                f"failed_criteria: {review.get('failed_criteria')}",
                f"adjudication: {(decision.get('decision') or {}).get('action')}", "",
                "| Criterion | Score | Rationale |",
                "| --- | ---: | --- |",
            ])
            scores = review.get("scores") or {}
            rationales = review.get("criterion_rationales") or {}
            for criterion in QUALITY_CRITERIA:
                rationale = rationales.get(criterion) or {}
                lines.append(
                    f"| {criterion} | {scores.get(criterion)} | {rationale.get('rationale_summary', {}).get('sha256', '')[:12]} |"
                )
            lines.append("")
        lines.extend(_delta_lines(case_records))
    return "\n".join(lines) + "\n"


def _discrimination_summary_lines(records: tuple[dict[str, Any], ...]) -> list[str]:
    lines = [
        "",
        "## Discrimination Summary",
        "",
        "| Case | Classification | GPT total | Gemini total | GPT action | Gemini action |",
        "| --- | --- | ---: | ---: | --- | --- |",
    ]
    for case_id in sorted({str(record.get("case_id")) for record in records}):
        case_records = [record for record in records if record.get("case_id") == case_id]
        classification = str(case_records[0].get("editorial_classification") or "")
        gpt = _record_for_plan(case_records, PLAN_GPT_QUALITY)
        gemini = _record_for_plan(case_records, PLAN_GEMINI_QUALITY)
        lines.append(
            f"| {case_id} | {classification} | {_total_score(gpt)} | "
            f"{_total_score(gemini)} | {_decision_action(gpt)} | {_decision_action(gemini)} |"
        )
    lines.extend([
        "",
        "### Aggregate Averages",
        "",
        "| Classification group | GPT average total | Gemini average total |",
        "| --- | ---: | ---: |",
        f"| STRONG | {_average_total(records, (CLASSIFICATION_STRONG,), PLAN_GPT_QUALITY)} | "
        f"{_average_total(records, (CLASSIFICATION_STRONG,), PLAN_GEMINI_QUALITY)} |",
        f"| WEAK/GENERIC | {_average_total(records, (CLASSIFICATION_WEAK, CLASSIFICATION_GENERIC), PLAN_GPT_QUALITY)} | "
        f"{_average_total(records, (CLASSIFICATION_WEAK, CLASSIFICATION_GENERIC), PLAN_GEMINI_QUALITY)} |",
    ])
    return lines


def _record_for_plan(records: list[dict[str, Any]], plan_id: str) -> dict[str, Any] | None:
    for record in records:
        if record.get("plan_id") == plan_id:
            return record
    return None


def _total_score(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    score = (record.get("quality_review") or {}).get("total_score")
    return "" if score is None else str(score)


def _decision_action(record: dict[str, Any] | None) -> str:
    if not record:
        return ""
    projection = record.get("adjudication_projection") or {}
    return str((projection.get("decision") or {}).get("action") or "")


def _average_total(
    records: tuple[dict[str, Any], ...],
    classifications: tuple[str, ...],
    plan_id: str,
) -> str:
    scores: list[int] = []
    for record in records:
        if record.get("plan_id") != plan_id:
            continue
        if record.get("editorial_classification") not in classifications:
            continue
        score = (record.get("quality_review") or {}).get("total_score")
        if isinstance(score, int) and not isinstance(score, bool):
            scores.append(score)
    if not scores:
        return ""
    return f"{sum(scores) / len(scores):.1f}"


def _delta_lines(records: list[dict[str, Any]]) -> list[str]:
    if len(records) < 2:
        return []
    first, second = records[0], records[1]
    first_review = first.get("quality_review") or {}
    second_review = second.get("quality_review") or {}
    first_scores = first_review.get("scores") or {}
    second_scores = second_review.get("scores") or {}
    lines = ["### Delta", "", "| Field | Delta |", "| --- | ---: |"]
    for criterion in QUALITY_CRITERIA:
        if criterion in first_scores and criterion in second_scores:
            lines.append(f"| {criterion} | {second_scores[criterion] - first_scores[criterion]} |")
    if "total_score" in first_review and "total_score" in second_review:
        lines.append(f"| total_score | {second_review['total_score'] - first_review['total_score']} |")
    first_action = ((first.get("adjudication_projection") or {}).get("decision") or {}).get("action")
    second_action = ((second.get("adjudication_projection") or {}).get("decision") or {}).get("action")
    lines.extend([
        f"| pass_changed | {first_review.get('pass') != second_review.get('pass')} |",
        f"| adjudication_changed | {first_action != second_action} |", "",
    ])
    return lines


def _planned_live_provider_calls(request: QualityEvaluatorBenchmarkRequest) -> int:
    return len(request.cases) * len(request.plans) * request.runs_per_plan


def _provider_calls(records: tuple[dict[str, Any], ...]) -> int:
    total = 0
    for record in records:
        counts = record.get("provider_invocation_counts") or {}
        total += counts.get("quality_evaluator_provider_api_calls", 0)
    return total


def _selected_evidence_ids(case: QualityEvaluatorBenchmarkCase) -> tuple[str, ...]:
    return tuple(item["evidence_id"] for item in case.selected_evidence)


def _safe_token_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    safe: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            safe[key] = value
    return safe


def _resolve_output_root(output_root: Path | None) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    default_root = (base_dir / DEFAULT_OUTPUT_ROOT).resolve()
    if output_root is None:
        _ensure_default_output_root_ignored(base_dir)
        return default_root
    resolved = Path(output_root).resolve()
    if _is_relative_to(resolved, base_dir) and not _is_relative_to(resolved, default_root):
        raise QualityEvaluatorBenchmarkConfigurationError(
            "output_root inside the repository must be under debug_outputs/final_post_quality_evaluator_benchmarks"
        )
    if _is_relative_to(resolved, default_root):
        _ensure_default_output_root_ignored(base_dir)
    return resolved


def _ensure_default_output_root_ignored(base_dir: Path) -> None:
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists():
        raise QualityEvaluatorBenchmarkConfigurationError("debug_outputs is not ignored")
    ignored = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    if "debug_outputs/" not in ignored and "debug_outputs" not in ignored:
        raise QualityEvaluatorBenchmarkConfigurationError("debug_outputs is not ignored")


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


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise QualityEvaluatorBenchmarkConfigurationError(f"{field_name} must be a safe identifier")


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _safe_text(value: Any) -> str:
    text = str(value or "")
    if any(marker in text.lower() for marker in ("sk-", "bearer ", "x-api-key", "secret")):
        return "redacted safe message"
    return text[:500]


def _markdown_cell(value: Any) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")[:500]


def _isoformat(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
