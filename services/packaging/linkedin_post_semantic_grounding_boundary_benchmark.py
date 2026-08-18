"""Boundary benchmark for Semantic Grounding model experiments.

This module is measurement-only. It does not execute Candidate Writer, Quality
Evaluator, Repair Writer, publication packaging, or production final-post
runtime wiring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any, Callable

from django.conf import settings

from apps.ai.client import get_ai_provider_reasoning_effort_error
from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics
from services.packaging.linkedin_post_editorial_boundary import (
    PostGenerationMetadata,
    PostEditorialInput,
    PromptMetadata,
)
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    OPENAI_FINAL_POST_MODEL,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_semantic_grounding_benchmark import (
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_LOW_REASONING,
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_MINIMAL_REASONING,
    SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT,
    semantic_grounding_reasoning_effort_for_execution_profile,
)
from services.packaging.linkedin_post_prompt_renderers import (
    render_semantic_grounding_prompt_input,
)


BOUNDARY_BENCHMARK_SCHEMA_VERSION = "2026-08-13"
BOUNDARY_BENCHMARK_TYPE = "semantic_grounding_boundary"
BOUNDARY_BENCHMARK_CORPUS_ID = "semantic_grounding_boundary_v1"
BOUNDARY_BENCHMARK_PROVENANCE = "hand-authored deterministic fixtures"
DEFAULT_EXPERIMENT_ID = "semantic-grounding-boundary-gpt-vs-gemini-v1"
DEFAULT_FIXTURE_PATH = Path(
    "tests/fixtures/linkedin_post_semantic_grounding_boundary/boundary_cases_v1.json"
)
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/final_post_semantic_grounding_boundary_benchmarks")

EXPECTED_VALID = "VALID_STRONG_VOICE"
EXPECTED_INVALID = "INVALID_SEMANTIC_OVERREACH"
EXPECTED_LABELS = (EXPECTED_VALID, EXPECTED_INVALID)

PLAN_GPT_GROUNDING = "gpt_grounding"
PLAN_GEMINI_GROUNDING = "gemini_grounding"
PLAN_GROUNDING_PROVIDER_DEFAULT = "grounding_default"
PLAN_GROUNDING_MINIMAL = "grounding_minimal"
PLAN_GROUNDING_LOW = "grounding_low"
PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"
GEMINI_MODEL = "gemini-3.6-flash"
OPENAI_GROUNDING_MAX_OUTPUT_TOKENS = 2400
GEMINI_GROUNDING_MAX_OUTPUT_TOKENS = 4800

FALSE_POSITIVE_WEIGHT = 1.0
FALSE_NEGATIVE_WEIGHT = 1.25

STATUS_DRY_RUN = "dry_run"
STATUS_COMPLETED = "completed"
STATUS_CONFIG_ERROR = "config_error"

RUN_STATUS_DRY_RUN = "dry_run"
RUN_STATUS_PASS = "grounding_completed_pass"
RUN_STATUS_BLOCK = "grounding_completed_block"
RUN_STATUS_FAILED = "failed"

FAILURE_PROVIDER_EXECUTION = "provider_execution_failure"
FAILURE_EMPTY_RESPONSE = "empty_response"
FAILURE_PARSE = "parse_failure"
FAILURE_NORMALIZATION = "normalization_failure"

FAILURE_CLASS_RATE_LIMIT = "PROVIDER_RATE_LIMIT_PATTERN"
FAILURE_CLASS_CLIENT_LIFECYCLE = "CLIENT_LIFECYCLE_PATTERN"
FAILURE_CLASS_SHARED_TRANSPORT = "SHARED_TRANSPORT_FAILURE"
FAILURE_CLASS_COMMAND_ORCHESTRATION = "COMMAND_ORCHESTRATION_FAILURE"
FAILURE_CLASS_AUTH = "AUTH_FAILURE"
FAILURE_CLASS_TIMEOUT = "TIMEOUT_PATTERN"
FAILURE_CLASS_OTHER = "OTHER"
FAILURE_CLASS_INCONCLUSIVE = "INCONCLUSIVE"

PROVENANCE_HISTORICAL_REUSED = "historical_reused"
PROVENANCE_LIVE_EXECUTED = "live_executed"
PROVENANCE_RETRY_PLANNED = "retry_planned"
PROVENANCE_PLANNED = "planned"

OUTCOME_VALID_PRESERVED = "valid_preserved"
OUTCOME_FALSE_POSITIVE = "false_positive"
OUTCOME_INVALID_BLOCKED = "invalid_blocked"
OUTCOME_FALSE_NEGATIVE = "false_negative"
OUTCOME_INFRASTRUCTURE_FAILURE = "infrastructure_failure"
OUTCOME_PLANNED = "planned"

DECISION_STRONG_CANDIDATE = "STRONG_CANDIDATE"
DECISION_ACCEPTABLE_WITH_RISK = "ACCEPTABLE_WITH_RISK"
DECISION_NOT_SUITABLE = "NOT_SUITABLE_FOR_GROUNDING"
DECISION_INCOMPLETE = "INCOMPLETE_RUN_MATRIX"

PARSER_PATH = (
    "services.packaging.linkedin_post_semantic_grounding_parser."
    "parse_and_normalize_semantic_grounding_response"
)
NORMALIZATION_PATH = (
    "services.packaging.linkedin_post_semantic_grounding_contract."
    "normalize_semantic_grounding_review_result"
)

SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,119}$")
WORD_RE = re.compile(r"[a-z0-9]+")
FORBIDDEN_ARTIFACT_KEY_FRAGMENTS = (
    "api_key",
    "secret",
    "password",
    "credential",
    "header",
    "provider_payload",
    "provider_reply",
    "prompt_text",
)
ALLOWED_PAIR_DIFFERENCE_FIELDS = {
    "case_id",
    "candidate_post",
    "expected_label",
    "expected_blocking",
    "expected_issue_type",
    "notes",
    "boundary_delta",
    "must_inspect_fragments",
}


@dataclass(frozen=True)
class SemanticGroundingBoundaryCase:
    case_id: str
    pair_id: str
    category: str
    semantic_topic: str
    rhetorical_intent: str
    expected_label: str
    expected_blocking: bool
    expected_issue_type: str
    evidence: tuple[dict[str, Any], ...]
    candidate_post: str
    post_brief: dict[str, Any]
    angle_decision: dict[str, Any]
    notes: str
    voice_sensitive: bool
    must_inspect_fragments: tuple[str, ...]
    boundary_delta: str
    fixture_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "pair_id": self.pair_id,
            "category": self.category,
            "semantic_topic": self.semantic_topic,
            "rhetorical_intent": self.rhetorical_intent,
            "expected_label": self.expected_label,
            "expected_blocking": self.expected_blocking,
            "expected_issue_type": self.expected_issue_type,
            "evidence": copy.deepcopy(list(self.evidence)),
            "candidate_post": self.candidate_post,
            "post_brief": copy.deepcopy(self.post_brief),
            "angle_decision": copy.deepcopy(self.angle_decision),
            "notes": self.notes,
            "voice_sensitive": self.voice_sensitive,
            "must_inspect_fragments": list(self.must_inspect_fragments),
            "boundary_delta": self.boundary_delta,
            "fixture_path": str(self.fixture_path) if self.fixture_path else None,
        }

    @property
    def selected_evidence_ids(self) -> tuple[str, ...]:
        return tuple(item["evidence_id"] for item in self.evidence)

    @property
    def candidate_payload(self) -> dict[str, str]:
        return {"post_text": self.candidate_post}


@dataclass(frozen=True)
class SemanticGroundingBoundaryPlan:
    plan_id: str
    provider: str
    model: str
    max_output_tokens: int
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
class SemanticGroundingBoundaryExecutionPolicy:
    inter_call_delay_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "inter_call_delay_seconds": self.inter_call_delay_seconds,
        }


@dataclass(frozen=True)
class SemanticGroundingBoundaryRequest:
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    cases: tuple[SemanticGroundingBoundaryCase, ...] = ()
    plans: tuple[SemanticGroundingBoundaryPlan, ...] = ()
    runs_per_plan: int = 1
    allow_api: bool = False
    output_root: Path | None = None
    false_positive_weight: float = FALSE_POSITIVE_WEIGHT
    false_negative_weight: float = FALSE_NEGATIVE_WEIGHT
    retry_failed_from: Path | None = None
    execution_policy: SemanticGroundingBoundaryExecutionPolicy = field(
        default_factory=SemanticGroundingBoundaryExecutionPolicy
    )


@dataclass(frozen=True)
class SemanticGroundingBoundaryArtifacts:
    output_dir: str
    manifest_json: str
    runs_jsonl: str
    summary_csv: str
    boundary_metrics_json: str
    boundary_comparison_md: str
    report_md: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "manifest_json": self.manifest_json,
            "runs_jsonl": self.runs_jsonl,
            "summary_csv": self.summary_csv,
            "boundary_metrics_json": self.boundary_metrics_json,
            "boundary_comparison_md": self.boundary_comparison_md,
            "report_md": self.report_md,
        }


@dataclass(frozen=True)
class SemanticGroundingBoundaryResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    provider_call_count: int
    planned_provider_call_count: int
    artifacts: SemanticGroundingBoundaryArtifacts | None
    safe_failure_code: str | None = None
    safe_failure_message: str = ""
    run_records: tuple[dict[str, Any], ...] = ()
    boundary_metrics: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "exit_code": self.exit_code,
            "experiment_id": self.experiment_id,
            "run_count": self.run_count,
            "provider_call_count": self.provider_call_count,
            "planned_provider_call_count": self.planned_provider_call_count,
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "safe_failure_code": self.safe_failure_code,
            "safe_failure_message": self.safe_failure_message,
            "run_records": copy.deepcopy(list(self.run_records)),
            "boundary_metrics": copy.deepcopy(self.boundary_metrics),
        }


class SemanticGroundingBoundaryBenchmarkConfigurationError(ValueError):
    pass


NowFactory = Callable[[], datetime]
SemanticGroundingExecutor = Callable[[Any], Any]


def load_semantic_grounding_boundary_cases(
    path: str | Path = DEFAULT_FIXTURE_PATH,
) -> tuple[SemanticGroundingBoundaryCase, ...]:
    fixture_path = Path(path)
    payload = json.loads(fixture_path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark fixture must be a JSON object"
        )
    _validate_fixture_metadata(payload)
    raw_cases = payload.get("cases") if isinstance(payload, dict) else None
    if not isinstance(raw_cases, list):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark fixture must contain a cases list"
        )
    cases = tuple(
        _case_from_payload(raw_case, fixture_path=fixture_path)
        for raw_case in raw_cases
    )
    validate_boundary_case_corpus(cases)
    return cases


def _validate_fixture_metadata(payload: dict[str, Any]) -> None:
    expected = {
        "schema_version": BOUNDARY_BENCHMARK_SCHEMA_VERSION,
        "corpus_id": BOUNDARY_BENCHMARK_CORPUS_ID,
        "provenance": BOUNDARY_BENCHMARK_PROVENANCE,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"boundary benchmark fixture must declare {key}={expected_value!r}"
            )


def default_semantic_grounding_boundary_plans() -> tuple[SemanticGroundingBoundaryPlan, ...]:
    return (
        SemanticGroundingBoundaryPlan(
            plan_id=PLAN_GPT_GROUNDING,
            provider=PROVIDER_OPENAI,
            model=OPENAI_FINAL_POST_MODEL,
            max_output_tokens=OPENAI_GROUNDING_MAX_OUTPUT_TOKENS,
        ),
        SemanticGroundingBoundaryPlan(
            plan_id=PLAN_GEMINI_GROUNDING,
            provider=PROVIDER_GEMINI,
            model=GEMINI_MODEL,
            max_output_tokens=GEMINI_GROUNDING_MAX_OUTPUT_TOKENS,
        ),
    )


def default_semantic_grounding_boundary_reasoning_calibration_plans() -> tuple[SemanticGroundingBoundaryPlan, ...]:
    return (
        SemanticGroundingBoundaryPlan(
            plan_id=PLAN_GROUNDING_PROVIDER_DEFAULT,
            provider=PROVIDER_GEMINI,
            model=GEMINI_MODEL,
            max_output_tokens=GEMINI_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_PROVIDER_DEFAULT,
        ),
        SemanticGroundingBoundaryPlan(
            plan_id=PLAN_GROUNDING_MINIMAL,
            provider=PROVIDER_GEMINI,
            model=GEMINI_MODEL,
            max_output_tokens=GEMINI_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_MINIMAL_REASONING,
        ),
        SemanticGroundingBoundaryPlan(
            plan_id=PLAN_GROUNDING_LOW,
            provider=PROVIDER_GEMINI,
            model=GEMINI_MODEL,
            max_output_tokens=GEMINI_GROUNDING_MAX_OUTPUT_TOKENS,
            execution_profile=SEMANTIC_GROUNDING_EXECUTION_PROFILE_LOW_REASONING,
        ),
    )


def build_boundary_semantic_grounding_prompt_render(case: SemanticGroundingBoundaryCase):
    return render_semantic_grounding_prompt_input(
        _post_editorial_input_for_boundary_case(case),
        prompt_metadata=_semantic_grounding_prompt_metadata(),
    )


def run_semantic_grounding_boundary_benchmark(
    request: SemanticGroundingBoundaryRequest,
    *,
    now_factory: NowFactory | None = None,
    semantic_grounding_executor: SemanticGroundingExecutor | None = None,
) -> SemanticGroundingBoundaryResult:
    try:
        output_dir = _validate_request(request)
    except SemanticGroundingBoundaryBenchmarkConfigurationError as exc:
        return SemanticGroundingBoundaryResult(
            status=STATUS_CONFIG_ERROR,
            exit_code=1,
            experiment_id=str(request.experiment_id or ""),
            run_count=0,
            provider_call_count=0,
            planned_provider_call_count=0,
            artifacts=None,
            safe_failure_code="semantic_grounding_boundary_configuration_error",
            safe_failure_message=_safe_text(str(exc)),
        )

    now = now_factory or (lambda: datetime.now(UTC))
    started_at = _isoformat(now())
    output_dir.mkdir(parents=True, exist_ok=False)
    artifacts = _artifact_paths(output_dir)
    records: list[dict[str, Any]] = []
    provider_call_count = 0
    resume_records = _load_resume_records(request.retry_failed_from)

    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                render = build_boundary_semantic_grounding_prompt_render(case)
                historical = _compatible_historical_record(
                    resume_records,
                    request,
                    case,
                    plan,
                    run_index,
                    render,
                )
                if historical is not None and _is_reusable_historical_record(historical):
                    record = _historical_reuse_record(
                        historical,
                        request,
                        started_at=started_at,
                        completed_at=_isoformat(now()),
                    )
                elif request.allow_api:
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
                    provider_call_count += record["provider_invocation_counts"][
                        "semantic_grounding"
                    ]
                    if request.execution_policy.inter_call_delay_seconds > 0:
                        time.sleep(request.execution_policy.inter_call_delay_seconds)
                else:
                    record = _dry_run_record(
                        request,
                        case,
                        plan,
                        run_index,
                        started_at,
                        _isoformat(now()),
                        render,
                        result_provenance=(
                            PROVENANCE_RETRY_PLANNED
                            if historical is not None
                            else PROVENANCE_PLANNED
                        ),
                        resume_source_run_id=(
                            historical.get("run_id") if historical else None
                        ),
                        resume_source_artifact=(
                            _resume_source_artifact(request.retry_failed_from)
                            if historical is not None
                            else None
                        ),
                        execution_failure_classification=(
                            _historical_failure_classification(historical)
                            if historical is not None
                            else None
                        ),
                    )
                records.append(record)

    manifest = _manifest(request, output_dir, started_at)
    metrics = compute_boundary_metrics(
        tuple(records),
        false_positive_weight=request.false_positive_weight,
        false_negative_weight=request.false_negative_weight,
    )
    _write_artifacts(artifacts, manifest, tuple(records), metrics)
    return SemanticGroundingBoundaryResult(
        status=STATUS_COMPLETED if request.allow_api else STATUS_DRY_RUN,
        exit_code=0,
        experiment_id=request.experiment_id,
        run_count=len(records),
        provider_call_count=provider_call_count,
        planned_provider_call_count=_planned_provider_calls(request),
        artifacts=artifacts,
        run_records=tuple(copy.deepcopy(records)),
        boundary_metrics=copy.deepcopy(metrics),
    )


def validate_boundary_case_corpus(
    cases: tuple[SemanticGroundingBoundaryCase, ...],
) -> None:
    if len(cases) != 16:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark corpus must contain exactly 16 cases"
        )
    seen_case_ids: set[str] = set()
    pairs: dict[str, list[SemanticGroundingBoundaryCase]] = {}
    for case in cases:
        _validate_identifier(case.case_id, "case_id")
        _validate_identifier(case.pair_id, "pair_id")
        if case.case_id in seen_case_ids:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"duplicate boundary case_id: {case.case_id}"
            )
        seen_case_ids.add(case.case_id)
        if case.expected_label not in EXPECTED_LABELS:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"unsupported expected_label: {case.expected_label}"
            )
        if case.expected_blocking != (case.expected_label == EXPECTED_INVALID):
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"expected_blocking inconsistent with expected_label: {case.case_id}"
            )
        if case.expected_label == EXPECTED_INVALID and not case.expected_issue_type:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"invalid boundary cases require expected_issue_type: {case.case_id}"
            )
        _validate_case_evidence(case)
        _validate_must_inspect_fragments(case)
        pairs.setdefault(case.pair_id, []).append(case)

    if len(pairs) != 8:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark corpus must contain exactly 8 pairs"
        )
    for pair_id, pair_cases in pairs.items():
        _validate_pair(pair_id, tuple(pair_cases))


def classify_boundary_run(record: dict[str, Any]) -> str:
    if record.get("execution_status") == RUN_STATUS_DRY_RUN:
        return OUTCOME_PLANNED
    if not _is_evaluable(record):
        return OUTCOME_INFRASTRUCTURE_FAILURE
    expected_label = record.get("expected_label")
    grounding_pass = record.get("grounding_pass") is True
    blocking_ids = record.get("blocking_claim_ids") or []
    has_blocking = bool(blocking_ids)
    if expected_label == EXPECTED_VALID:
        if grounding_pass and not has_blocking:
            return OUTCOME_VALID_PRESERVED
        return OUTCOME_FALSE_POSITIVE
    if expected_label == EXPECTED_INVALID:
        if not grounding_pass and has_blocking:
            return OUTCOME_INVALID_BLOCKED
        return OUTCOME_FALSE_NEGATIVE
    return OUTCOME_INFRASTRUCTURE_FAILURE


def compute_boundary_metrics(
    run_records: tuple[dict[str, Any], ...],
    *,
    false_positive_weight: float = FALSE_POSITIVE_WEIGHT,
    false_negative_weight: float = FALSE_NEGATIVE_WEIGHT,
) -> dict[str, Any]:
    by_plan = {}
    for plan_id in sorted({str(record["plan_id"]) for record in run_records}):
        records = tuple(record for record in run_records if record["plan_id"] == plan_id)
        by_plan[plan_id] = _metrics_for_records(
            records,
            false_positive_weight=false_positive_weight,
            false_negative_weight=false_negative_weight,
        )
    return {
        "weights": {
            "false_positive_weight": false_positive_weight,
            "false_negative_weight": false_negative_weight,
        },
        "plans": by_plan,
    }


def _case_from_payload(
    payload: Any,
    *,
    fixture_path: Path,
) -> SemanticGroundingBoundaryCase:
    if not isinstance(payload, dict):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark case entries must be objects"
        )
    required = (
        "case_id",
        "pair_id",
        "category",
        "semantic_topic",
        "rhetorical_intent",
        "expected_label",
        "expected_blocking",
        "expected_issue_type",
        "evidence",
        "candidate_post",
        "post_brief",
        "angle_decision",
        "notes",
        "voice_sensitive",
        "must_inspect_fragments",
        "boundary_delta",
    )
    missing = [field for field in required if field not in payload]
    if missing:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark case missing required fields: " + ", ".join(missing)
        )
    evidence = payload["evidence"]
    fragments = payload["must_inspect_fragments"]
    if not isinstance(evidence, list) or not evidence:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary benchmark evidence must be a non-empty list"
        )
    if not isinstance(fragments, list) or not fragments:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "must_inspect_fragments must be a non-empty list"
        )
    return SemanticGroundingBoundaryCase(
        case_id=_required_string(payload, "case_id"),
        pair_id=_required_string(payload, "pair_id"),
        category=_required_string(payload, "category"),
        semantic_topic=_required_string(payload, "semantic_topic"),
        rhetorical_intent=_required_string(payload, "rhetorical_intent"),
        expected_label=_required_string(payload, "expected_label"),
        expected_blocking=_required_bool(payload, "expected_blocking"),
        expected_issue_type=_optional_string(payload, "expected_issue_type"),
        evidence=tuple(copy.deepcopy(evidence)),
        candidate_post=_required_string(payload, "candidate_post"),
        post_brief=_required_dict(payload, "post_brief"),
        angle_decision=_required_dict(payload, "angle_decision"),
        notes=_required_string(payload, "notes"),
        voice_sensitive=_required_bool(payload, "voice_sensitive"),
        must_inspect_fragments=tuple(_required_string(item, "fragment") for item in fragments),
        boundary_delta=_required_string(payload, "boundary_delta"),
        fixture_path=fixture_path,
    )


def _validate_case_evidence(case: SemanticGroundingBoundaryCase) -> None:
    evidence_ids = []
    for item in case.evidence:
        if not isinstance(item, dict):
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"evidence items must be objects: {case.case_id}"
            )
        evidence_id = item.get("evidence_id")
        evidence_text = item.get("evidence_text")
        role_in_post = item.get("role_in_post")
        for field_name, value in (
            ("evidence_id", evidence_id),
            ("evidence_text", evidence_text),
            ("role_in_post", role_in_post),
        ):
            if not isinstance(value, str) or not value.strip():
                raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                    f"evidence.{field_name} must be non-empty: {case.case_id}"
                )
        evidence_ids.append(evidence_id)
    if len(evidence_ids) != len(set(evidence_ids)):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"duplicate evidence_id in case: {case.case_id}"
        )
    if case.post_brief.get("evidence_to_use") != list(case.evidence):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"post_brief.evidence_to_use must match evidence: {case.case_id}"
        )
    if case.angle_decision.get("supporting_evidence_ids") != evidence_ids:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"angle_decision.supporting_evidence_ids must match evidence order: {case.case_id}"
        )


def _validate_must_inspect_fragments(case: SemanticGroundingBoundaryCase) -> None:
    normalized_post = _normalize_fragment_text(case.candidate_post)
    for fragment in case.must_inspect_fragments:
        if _normalize_fragment_text(fragment) not in normalized_post:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                "must_inspect_fragments must appear in candidate_post: "
                f"{case.case_id}"
            )


def _validate_pair(
    pair_id: str,
    pair_cases: tuple[SemanticGroundingBoundaryCase, ...],
) -> None:
    if len(pair_cases) != 2:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"boundary pair must contain exactly two cases: {pair_id}"
        )
    labels = {case.expected_label for case in pair_cases}
    if labels != {EXPECTED_VALID, EXPECTED_INVALID}:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"boundary pair must contain one valid and one invalid case: {pair_id}"
        )
    first, second = pair_cases
    for field_name in (
        "category",
        "semantic_topic",
        "rhetorical_intent",
        "evidence",
        "post_brief",
        "angle_decision",
    ):
        if getattr(first, field_name) != getattr(second, field_name):
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"boundary pair {pair_id} has mismatched {field_name}"
            )
    first_dict = first.to_dict()
    second_dict = second.to_dict()
    changed_fields = {
        field_name
        for field_name in first_dict
        if first_dict[field_name] != second_dict[field_name]
    }
    changed_fields.discard("fixture_path")
    if not changed_fields <= ALLOWED_PAIR_DIFFERENCE_FIELDS:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"boundary pair {pair_id} has unsupported differing fields: "
            + ", ".join(sorted(changed_fields - ALLOWED_PAIR_DIFFERENCE_FIELDS))
        )


def _post_editorial_input_for_boundary_case(
    case: SemanticGroundingBoundaryCase,
) -> PostEditorialInput:
    diagnostics = _passing_benchmark_diagnostics()
    return PostEditorialInput(
        candidate_payload=copy.deepcopy(case.candidate_payload),
        post_brief=copy.deepcopy(case.post_brief),
        angle_decision=copy.deepcopy(case.angle_decision),
        selected_evidence=copy.deepcopy(list(case.evidence)),
        final_payload_validation_passed=True,
        final_payload_validation_error="",
        diagnostics=diagnostics,
        repair_reasons=[],
        generation_metadata=PostGenerationMetadata(
            provider="benchmark_fixture",
            model="hand_authored_candidate_post",
            run_id=case.case_id,
            created_at=None,
            token_usage=None,
            cost_metadata=None,
        ),
        prompt_metadata=_semantic_grounding_prompt_metadata(),
    )


def _passing_benchmark_diagnostics() -> FinalPostDiagnostics:
    return FinalPostDiagnostics(
        schema_validation_passed=True,
        schema_validation_error="",
        deterministic_checks_passed=True,
        system_linkedin_ready=True,
        model_claimed_linkedin_ready=True,
        missing_quality_check_keys=[],
        non_boolean_quality_check_keys=[],
        evidence_id_leaks=[],
        scaffold_phrase_leaks=[],
        source_summary_phrase_leaks=[],
        repair_reasons=[],
    )


def _semantic_grounding_prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_semantic_grounding_evaluator",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_semantic_grounding_evaluator.txt",
    )


def _validate_request(request: SemanticGroundingBoundaryRequest) -> Path:
    _validate_identifier(request.experiment_id, "experiment_id")
    if not request.cases:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "at least one boundary case is required"
        )
    validate_boundary_case_corpus(request.cases)
    if not request.plans:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "at least one boundary plan is required"
        )
    for plan in request.plans:
        _validate_plan(plan)
    if request.runs_per_plan < 1:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "runs_per_plan must be at least 1"
        )
    if request.false_positive_weight <= 0 or request.false_negative_weight <= 0:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "boundary scoring weights must be positive"
        )
    _validate_execution_policy(request.execution_policy)
    if request.retry_failed_from is not None:
        _load_resume_records(request.retry_failed_from)
        _planned_provider_calls(request)
    if request.allow_api:
        _validate_live_prompt_paths(request.cases)
    output_root = _resolve_output_root(request.output_root)
    output_dir = output_root / request.experiment_id
    if output_dir.exists():
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"benchmark output directory already exists: {output_dir}"
        )
    return output_dir


def _validate_plan(plan: SemanticGroundingBoundaryPlan) -> None:
    _validate_identifier(plan.plan_id, "plan_id")
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
        provider=plan.provider,
        model=plan.model,
    )
    if policy_failure is not None:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(str(policy_failure))
    if (
        isinstance(plan.max_output_tokens, bool)
        or not isinstance(plan.max_output_tokens, int)
        or plan.max_output_tokens <= 0
    ):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "max_output_tokens must be a positive integer"
        )
    if not isinstance(plan.json_mode, bool):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "json_mode must be a boolean"
        )
    try:
        reasoning_effort = plan.reasoning_effort
    except ValueError as exc:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(str(exc)) from exc
    reasoning_effort_error = get_ai_provider_reasoning_effort_error(
        provider=plan.provider,
        reasoning_effort=reasoning_effort,
    )
    if reasoning_effort_error is not None:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(reasoning_effort_error)


def _validate_execution_policy(
    policy: SemanticGroundingBoundaryExecutionPolicy,
) -> None:
    if (
        isinstance(policy.inter_call_delay_seconds, bool)
        or not isinstance(policy.inter_call_delay_seconds, (int, float))
        or policy.inter_call_delay_seconds < 0
    ):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "inter_call_delay_seconds must be a non-negative number"
        )


def _live_run_record(
    request: SemanticGroundingBoundaryRequest,
    case: SemanticGroundingBoundaryCase,
    plan: SemanticGroundingBoundaryPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: Any,
    *,
    semantic_grounding_executor: SemanticGroundingExecutor | None,
) -> dict[str, Any]:
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

    selected_evidence_ids = case.selected_evidence_ids
    execution_request = build_semantic_grounding_execution_request(
        render,
        prompt_text=_semantic_grounding_prompt_text(render),
        provider=plan.provider,
        model=plan.model,
        max_output_tokens=plan.max_output_tokens,
        json_mode=plan.json_mode,
        reasoning_effort=plan.reasoning_effort,
        execution_metadata={
            "experiment_id": request.experiment_id,
            "benchmark_type": BOUNDARY_BENCHMARK_TYPE,
            "case_id": case.case_id,
            "plan_id": plan.plan_id,
            "run_index": run_index,
            "run_id": _run_id(case, plan, run_index),
            "semantic_grounding_execution_profile": plan.execution_profile,
            "semantic_grounding_reasoning_effort": plan.reasoning_effort,
        },
    )
    request_error = get_semantic_grounding_execution_request_error(execution_request)
    if request_error:
        return _boundary_record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=RUN_STATUS_FAILED,
            failure_stage="request_validation",
            failure_code=FAILURE_PROVIDER_EXECUTION,
            canonical_error_code=ERROR_EXECUTION_FAILED,
            execution_request_error=request_error,
            execution_success=False,
            parse_success=False,
            normalization_success=False,
            semantic_grounding_calls=0,
        )

    executor = semantic_grounding_executor or execute_semantic_grounding_prompt
    raw_response = executor(execution_request)
    execution_error = getattr(raw_response, "execution_error", None)
    raw_text = getattr(raw_response, "raw_text", "")
    if execution_error:
        failure_code = (
            FAILURE_EMPTY_RESPONSE
            if execution_error == "empty provider response"
            else FAILURE_PROVIDER_EXECUTION
        )
        return _boundary_record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=RUN_STATUS_FAILED,
            failure_stage="execution",
            failure_code=failure_code,
            canonical_error_code=ERROR_EXECUTION_FAILED,
            execution_success=False,
            parse_success=False,
            normalization_success=False,
            response_diagnostics=_raw_response_diagnostics(raw_response),
            execution_failure_classification=_execution_failure_classification(
                raw_response
            ),
            semantic_grounding_calls=1,
            result_provenance=PROVENANCE_LIVE_EXECUTED,
        )
    if not str(raw_text or "").strip():
        return _boundary_record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=RUN_STATUS_FAILED,
            failure_stage="execution",
            failure_code=FAILURE_EMPTY_RESPONSE,
            canonical_error_code=ERROR_EMPTY_RAW_RESPONSE,
            execution_success=False,
            parse_success=False,
            normalization_success=False,
            response_diagnostics=_raw_response_diagnostics(raw_response),
            execution_failure_classification=_execution_failure_classification(
                raw_response
            ),
            semantic_grounding_calls=1,
            result_provenance=PROVENANCE_LIVE_EXECUTED,
        )

    try:
        grounding_review = parse_and_normalize_semantic_grounding_response(
            raw_response,
            selected_evidence_ids=selected_evidence_ids,
        )
    except SemanticGroundingResponseParseError as exc:
        normalization_failed = exc.code == ERROR_NORMALIZATION_FAILED
        return _boundary_record(
            request,
            case,
            plan,
            run_index,
            started_at,
            completed_at,
            render,
            execution_status=RUN_STATUS_FAILED,
            failure_stage="normalization" if normalization_failed else "parse",
            failure_code=FAILURE_NORMALIZATION if normalization_failed else FAILURE_PARSE,
            canonical_error_code=exc.code,
            execution_success=True,
            parse_success=normalization_failed,
            normalization_success=False,
            parser_error_details=_parser_error_details(exc),
            normalization_error_details=(
                _normalization_error_details(exc) if normalization_failed else None
            ),
            response_diagnostics=_raw_response_diagnostics(raw_response),
            semantic_grounding_calls=1,
            result_provenance=PROVENANCE_LIVE_EXECUTED,
        )

    review_payload = grounding_review.to_dict()
    passed = bool(review_payload["pass"])
    blocking_claim_ids = review_payload["blocking_claim_ids"]
    return _boundary_record(
        request,
        case,
        plan,
        run_index,
        started_at,
        completed_at,
        render,
        execution_status=RUN_STATUS_PASS if passed else RUN_STATUS_BLOCK,
        failure_stage=None,
        failure_code=None,
        execution_success=True,
        parse_success=True,
        normalization_success=True,
        grounding_pass=passed,
        blocking_claim_count=len(blocking_claim_ids),
        human_review_required=bool(review_payload["requires_human_review"]),
        blocking_claim_ids=blocking_claim_ids,
        claim_reviews=review_payload["claim_reviews"],
        response_diagnostics=_raw_response_diagnostics(raw_response),
        semantic_grounding_calls=1,
        result_provenance=PROVENANCE_LIVE_EXECUTED,
    )


def _dry_run_record(
    request: SemanticGroundingBoundaryRequest,
    case: SemanticGroundingBoundaryCase,
    plan: SemanticGroundingBoundaryPlan,
    run_index: int,
    started_at: str,
    completed_at: str,
    render: Any,
    result_provenance: str = PROVENANCE_PLANNED,
    resume_source_run_id: str | None = None,
    resume_source_artifact: str | None = None,
    execution_failure_classification: str | None = None,
) -> dict[str, Any]:
    return _boundary_record(
        request,
        case,
        plan,
        run_index,
        started_at,
        completed_at,
        render,
        execution_status=RUN_STATUS_DRY_RUN,
        failure_stage=None,
        failure_code=None,
        execution_success=None,
        parse_success=None,
        normalization_success=None,
        semantic_grounding_calls=0,
        result_provenance=result_provenance,
        resume_source_run_id=resume_source_run_id,
        resume_source_artifact=resume_source_artifact,
        execution_failure_classification=execution_failure_classification,
    )


def _boundary_record(
    request: SemanticGroundingBoundaryRequest,
    case: SemanticGroundingBoundaryCase,
    plan: SemanticGroundingBoundaryPlan,
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
    execution_failure_classification: str | None = None,
    result_provenance: str = PROVENANCE_LIVE_EXECUTED,
    resume_source_run_id: str | None = None,
    resume_source_artifact: str | None = None,
) -> dict[str, Any]:
    fragments = _fragment_coverage(
        case.must_inspect_fragments,
        claim_reviews or [],
    )
    record = _sanitize_artifact_value(
        {
            "schema_version": BOUNDARY_BENCHMARK_SCHEMA_VERSION,
            "experiment_id": request.experiment_id,
            "benchmark_type": BOUNDARY_BENCHMARK_TYPE,
            "case_id": case.case_id,
            "pair_id": case.pair_id,
            "category": case.category,
            "semantic_topic": case.semantic_topic,
            "rhetorical_intent": case.rhetorical_intent,
            "expected_label": case.expected_label,
            "expected_blocking": case.expected_blocking,
            "expected_issue_type": case.expected_issue_type,
            "voice_sensitive": case.voice_sensitive,
            "boundary_delta": case.boundary_delta,
            "plan_id": plan.plan_id,
            "provider": plan.provider,
            "model": plan.model,
            "max_output_tokens": plan.max_output_tokens,
            "json_mode": plan.json_mode,
            "execution_profile": plan.execution_profile,
            "reasoning_effort": plan.reasoning_effort,
            "run_index": run_index,
            "run_id": f"{case.case_id}__{plan.plan_id}__{run_index}",
            "started_at": started_at,
            "completed_at": completed_at,
            "execution_status": execution_status,
            "execution_success": execution_success,
            "parse_success": parse_success,
            "normalization_success": normalization_success,
            "failure_stage": failure_stage,
            "failure_code": failure_code,
            "execution_failure_classification": execution_failure_classification,
            "canonical_error_code": canonical_error_code,
            "execution_request_error": execution_request_error,
            "parser_path": PARSER_PATH,
            "normalization_path": NORMALIZATION_PATH,
            "grounding_pass": grounding_pass,
            "blocking_claim_count": blocking_claim_count,
            "blocking_claim_ids": copy.deepcopy(blocking_claim_ids or []),
            "human_review_required": human_review_required,
            "claim_reviews": copy.deepcopy(claim_reviews or []),
            "must_inspect_fragments": list(case.must_inspect_fragments),
            "must_inspect_fragment_coverage": fragments,
            "semantic_input_summary": _semantic_input_summary(render),
            "rendered_variable_names": list(render.variables),
            "prompt_metadata": {
                "prompt_name": render.prompt_name,
                "prompt_version": render.prompt_version,
                "prompt_path": render.prompt_path,
            },
            "parser_error_details": copy.deepcopy(parser_error_details),
            "normalization_error_details": copy.deepcopy(normalization_error_details),
            "response_diagnostics": copy.deepcopy(response_diagnostics),
            "result_provenance": result_provenance,
            "resume_source_run_id": resume_source_run_id,
            "resume_source_artifact": resume_source_artifact,
            "execution_policy": request.execution_policy.to_dict(),
            "execution_attempt_index": 1,
            "execution_attempt_count": 1,
            "provider_invocation_counts": {
                "candidate_writer": 0,
                "semantic_grounding": semantic_grounding_calls,
                "quality_evaluator": 0,
                "repair_writer": 0,
                "publication_packaging": 0,
            },
        }
    )
    record["boundary_outcome"] = classify_boundary_run(record)
    return record


def _run_id(
    case: SemanticGroundingBoundaryCase,
    plan: SemanticGroundingBoundaryPlan,
    run_index: int,
) -> str:
    return f"{case.case_id}__{plan.plan_id}__{run_index}"


def _load_resume_records(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    resume_path = Path(path)
    if resume_path.is_dir():
        resume_path = resume_path / "runs.jsonl"
    if not resume_path.exists():
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"retry_failed_from runs artifact does not exist: {resume_path}"
        )
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(resume_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"retry_failed_from runs artifact contains invalid JSONL at line {line_number}"
            ) from exc
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"retry_failed_from record missing run_id at line {line_number}"
            )
        if run_id in records:
            raise SemanticGroundingBoundaryBenchmarkConfigurationError(
                f"retry_failed_from contains duplicate run_id: {run_id}"
            )
        records[run_id] = record
    return records


def _compatible_historical_record(
    resume_records: dict[str, dict[str, Any]],
    request: SemanticGroundingBoundaryRequest,
    case: SemanticGroundingBoundaryCase,
    plan: SemanticGroundingBoundaryPlan,
    run_index: int,
    render: Any,
) -> dict[str, Any] | None:
    if not resume_records:
        return None
    run_id = _run_id(case, plan, run_index)
    historical = resume_records.get(run_id)
    if historical is None:
        return None
    mismatches = []
    expected_scalars = {
        "case_id": case.case_id,
        "plan_id": plan.plan_id,
        "run_index": run_index,
        "provider": plan.provider,
        "model": plan.model,
        "max_output_tokens": plan.max_output_tokens,
        "json_mode": plan.json_mode,
        "execution_profile": plan.execution_profile,
        "reasoning_effort": plan.reasoning_effort,
        "expected_label": case.expected_label,
        "expected_blocking": case.expected_blocking,
    }
    for key, expected in expected_scalars.items():
        if historical.get(key) != expected:
            mismatches.append(key)
    if historical.get("semantic_input_summary") != _semantic_input_summary(render):
        mismatches.append("semantic_input_summary")
    if mismatches:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "retry_failed_from record is incompatible for "
            f"{run_id}: " + ", ".join(sorted(mismatches))
        )
    return historical


def _is_reusable_historical_record(record: dict[str, Any]) -> bool:
    return _is_evaluable(record) or record.get("failure_stage") in {
        "parse",
        "normalization",
    }


def _is_retryable_historical_failure(record: dict[str, Any]) -> bool:
    return not _is_reusable_historical_record(record)


def _historical_reuse_record(
    historical: dict[str, Any],
    request: SemanticGroundingBoundaryRequest,
    *,
    started_at: str,
    completed_at: str,
) -> dict[str, Any]:
    allowed_keys = {
        "schema_version",
        "benchmark_type",
        "case_id",
        "pair_id",
        "category",
        "semantic_topic",
        "rhetorical_intent",
        "expected_label",
        "expected_blocking",
        "expected_issue_type",
        "voice_sensitive",
        "boundary_delta",
        "plan_id",
        "provider",
        "model",
        "max_output_tokens",
        "json_mode",
        "execution_profile",
        "reasoning_effort",
        "run_index",
        "run_id",
        "execution_status",
        "execution_success",
        "parse_success",
        "normalization_success",
        "failure_stage",
        "failure_code",
        "canonical_error_code",
        "execution_request_error",
        "parser_path",
        "normalization_path",
        "grounding_pass",
        "blocking_claim_count",
        "blocking_claim_ids",
        "human_review_required",
        "claim_reviews",
        "must_inspect_fragments",
        "must_inspect_fragment_coverage",
        "semantic_input_summary",
        "rendered_variable_names",
        "prompt_metadata",
        "parser_error_details",
        "normalization_error_details",
        "boundary_outcome",
    }
    record = _sanitize_artifact_value(
        {key: copy.deepcopy(historical.get(key)) for key in allowed_keys if key in historical}
    )
    record["experiment_id"] = request.experiment_id
    record["started_at"] = started_at
    record["completed_at"] = completed_at
    record["result_provenance"] = PROVENANCE_HISTORICAL_REUSED
    record["resume_source_run_id"] = historical.get("run_id")
    record["resume_source_artifact"] = _resume_source_artifact(request.retry_failed_from)
    record["execution_failure_classification"] = _historical_failure_classification(
        historical
    )
    record["execution_policy"] = request.execution_policy.to_dict()
    record["execution_attempt_index"] = 0
    record["execution_attempt_count"] = 0
    record["provider_invocation_counts"] = {
        "candidate_writer": 0,
        "semantic_grounding": 0,
        "quality_evaluator": 0,
        "repair_writer": 0,
        "publication_packaging": 0,
    }
    record["boundary_outcome"] = classify_boundary_run(record)
    return record


def _resume_source_artifact(path: Path | None) -> str | None:
    if path is None:
        return None
    resume_path = Path(path)
    if resume_path.is_dir():
        resume_path = resume_path / "runs.jsonl"
    return str(resume_path)


def _historical_failure_classification(record: dict[str, Any]) -> str | None:
    value = record.get("execution_failure_classification")
    allowed = {
        FAILURE_CLASS_RATE_LIMIT,
        FAILURE_CLASS_CLIENT_LIFECYCLE,
        FAILURE_CLASS_SHARED_TRANSPORT,
        FAILURE_CLASS_COMMAND_ORCHESTRATION,
        FAILURE_CLASS_AUTH,
        FAILURE_CLASS_TIMEOUT,
        FAILURE_CLASS_OTHER,
        FAILURE_CLASS_INCONCLUSIVE,
    }
    if isinstance(value, str) and value in allowed:
        return value
    if isinstance(value, str) and value:
        return FAILURE_CLASS_INCONCLUSIVE
    if record.get("failure_stage") == "execution":
        return FAILURE_CLASS_INCONCLUSIVE
    return None


def _execution_failure_classification(raw_response: Any) -> str:
    diagnostics = getattr(raw_response, "execution_diagnostics", None)
    if not isinstance(diagnostics, dict) or not diagnostics:
        return FAILURE_CLASS_INCONCLUSIVE
    http_status = diagnostics.get("http_status")
    message = str(diagnostics.get("message") or "").lower()
    exception_class = str(diagnostics.get("exception_class") or "").lower()
    if diagnostics.get("rate_limited") is True or http_status == 429:
        return FAILURE_CLASS_RATE_LIMIT
    if diagnostics.get("timeout") is True:
        return FAILURE_CLASS_TIMEOUT
    if http_status in {401, 403}:
        return FAILURE_CLASS_AUTH
    if isinstance(http_status, int) and 500 <= http_status <= 599:
        return FAILURE_CLASS_SHARED_TRANSPORT
    if any(fragment in exception_class or fragment in message for fragment in ("connection", "network", "transport")):
        return FAILURE_CLASS_SHARED_TRANSPORT
    if any(fragment in exception_class or fragment in message for fragment in ("closed", "lifecycle", "event loop")):
        return FAILURE_CLASS_CLIENT_LIFECYCLE
    if "command" in exception_class or "subprocess" in exception_class:
        return FAILURE_CLASS_COMMAND_ORCHESTRATION
    return FAILURE_CLASS_OTHER


def _metrics_for_records(
    records: tuple[dict[str, Any], ...],
    *,
    false_positive_weight: float,
    false_negative_weight: float,
) -> dict[str, Any]:
    attempted = len(records)
    evaluated_records = tuple(record for record in records if _is_evaluable(record))
    valid_records = tuple(
        record for record in evaluated_records if record["expected_label"] == EXPECTED_VALID
    )
    invalid_records = tuple(
        record for record in evaluated_records if record["expected_label"] == EXPECTED_INVALID
    )
    false_positive_count = sum(
        1 for record in valid_records if classify_boundary_run(record) == OUTCOME_FALSE_POSITIVE
    )
    false_negative_count = sum(
        1 for record in invalid_records if classify_boundary_run(record) == OUTCOME_FALSE_NEGATIVE
    )
    valid_preserved = len(valid_records) - false_positive_count
    invalid_blocked = len(invalid_records) - false_negative_count
    infrastructure_records = tuple(record for record in records if _is_infrastructure_failure(record))
    voice_sensitive_valid = tuple(
        record for record in valid_records if record.get("voice_sensitive") is True
    )
    voice_sensitive_preserved = sum(
        1
        for record in voice_sensitive_valid
        if classify_boundary_run(record) == OUTCOME_VALID_PRESERVED
    )
    coverage = _aggregate_fragment_coverage(evaluated_records)
    accuracy_denominator = len(valid_records) + len(invalid_records)
    weighted_denominator = len(evaluated_records)
    weighted_error_score = (
        (
            false_positive_count * false_positive_weight
            + false_negative_count * false_negative_weight
        )
        / weighted_denominator
        if weighted_denominator
        else None
    )
    result = {
        "attempted_cases_total": attempted,
        "evaluated_cases_total": len(evaluated_records),
        "infrastructure_failure_count": len(infrastructure_records),
        "provider_failure_count": sum(
            1
            for record in infrastructure_records
            if record.get("failure_stage") in {"execution", "request_validation"}
        ),
        "parse_failure_count": sum(
            1 for record in infrastructure_records if record.get("failure_stage") == "parse"
        ),
        "normalization_failure_count": sum(
            1
            for record in infrastructure_records
            if record.get("failure_stage") == "normalization"
        ),
        "valid_cases_total": len(valid_records),
        "valid_cases_preserved": valid_preserved,
        "false_positive_count": false_positive_count,
        "false_positive_rate": _ratio(false_positive_count, len(valid_records)),
        "invalid_cases_total": len(invalid_records),
        "invalid_cases_blocked": invalid_blocked,
        "false_negative_count": false_negative_count,
        "false_negative_rate": _ratio(false_negative_count, len(invalid_records)),
        "overall_boundary_accuracy": _ratio(
            valid_preserved + invalid_blocked,
            accuracy_denominator,
        ),
        "strong_voice_preservation_rate": _ratio(valid_preserved, len(valid_records)),
        "semantic_overreach_detection_rate": _ratio(invalid_blocked, len(invalid_records)),
        "voice_sensitive_valid_total": len(voice_sensitive_valid),
        "voice_sensitive_preserved": voice_sensitive_preserved,
        "voice_sensitive_false_positive_rate": _ratio(
            len(voice_sensitive_valid) - voice_sensitive_preserved,
            len(voice_sensitive_valid),
        ),
        "required_fragment_coverage_rate": coverage["rate"],
        "required_fragments_total": coverage["total"],
        "required_fragments_covered": coverage["covered"],
        "weighted_error_score": (
            round(weighted_error_score, 6) if weighted_error_score is not None else None
        ),
        "result_completeness": (
            "complete" if len(evaluated_records) == attempted else "incomplete"
        ),
    }
    result["decision_band"] = _decision_band(result)
    return result


def _is_evaluable(record: dict[str, Any]) -> bool:
    return (
        record.get("execution_success") is True
        and record.get("parse_success") is True
        and record.get("normalization_success") is True
        and isinstance(record.get("grounding_pass"), bool)
    )


def _is_infrastructure_failure(record: dict[str, Any]) -> bool:
    return record.get("execution_status") != RUN_STATUS_DRY_RUN and not _is_evaluable(record)


def _decision_band(metrics: dict[str, Any]) -> str:
    if metrics["result_completeness"] != "complete":
        return DECISION_INCOMPLETE
    coverage = metrics["required_fragment_coverage_rate"]
    if coverage is None:
        coverage = 0
    if (
        metrics["false_negative_count"] == 0
        and metrics["false_positive_count"] <= 1
        and metrics["overall_boundary_accuracy"] >= 0.9375
        and coverage >= 0.90
    ):
        return DECISION_STRONG_CANDIDATE
    if (
        metrics["false_negative_count"] <= 1
        and metrics["false_positive_count"] <= 2
        and metrics["overall_boundary_accuracy"] >= 0.8125
        and coverage >= 0.80
    ):
        return DECISION_ACCEPTABLE_WITH_RISK
    return DECISION_NOT_SUITABLE


def _fragment_coverage(
    fragments: tuple[str, ...],
    claim_reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    claim_texts = [
        _normalize_fragment_text(claim.get("claim_text"))
        for claim in claim_reviews
        if isinstance(claim, dict)
    ]
    return [
        {
            "fragment": fragment,
            "covered": any(
                _normalize_fragment_text(fragment) in claim_text
                for claim_text in claim_texts
            ),
        }
        for fragment in fragments
    ]


def _aggregate_fragment_coverage(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    total = 0
    covered = 0
    for record in records:
        for item in record.get("must_inspect_fragment_coverage") or []:
            total += 1
            if item.get("covered") is True:
                covered += 1
    return {
        "total": total,
        "covered": covered,
        "rate": _ratio(covered, total),
    }


def _normalize_fragment_text(value: Any) -> str:
    return " ".join(WORD_RE.findall(str(value or "").lower()))


def _write_artifacts(
    artifacts: SemanticGroundingBoundaryArtifacts,
    manifest: dict[str, Any],
    run_records: tuple[dict[str, Any], ...],
    metrics: dict[str, Any],
) -> None:
    Path(artifacts.manifest_json).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    Path(artifacts.runs_jsonl).write_text(
        "\n".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
            for record in run_records
        )
        + ("\n" if run_records else ""),
        encoding="utf-8",
    )
    _write_summary_csv(Path(artifacts.summary_csv), run_records)
    Path(artifacts.boundary_metrics_json).write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    Path(artifacts.boundary_comparison_md).write_text(
        _comparison_text(manifest, run_records, metrics),
        encoding="utf-8",
    )
    Path(artifacts.report_md).write_text(
        _report_text(manifest, run_records, metrics),
        encoding="utf-8",
    )


def _artifact_paths(output_dir: Path) -> SemanticGroundingBoundaryArtifacts:
    return SemanticGroundingBoundaryArtifacts(
        output_dir=str(output_dir),
        manifest_json=str(output_dir / "manifest.json"),
        runs_jsonl=str(output_dir / "runs.jsonl"),
        summary_csv=str(output_dir / "summary.csv"),
        boundary_metrics_json=str(output_dir / "boundary_metrics.json"),
        boundary_comparison_md=str(output_dir / "boundary_comparison.md"),
        report_md=str(output_dir / "report.md"),
    )


def _manifest(
    request: SemanticGroundingBoundaryRequest,
    output_dir: Path,
    started_at: str,
) -> dict[str, Any]:
    return _sanitize_artifact_value(
        {
            "schema_version": BOUNDARY_BENCHMARK_SCHEMA_VERSION,
            "experiment_id": request.experiment_id,
            "benchmark_type": BOUNDARY_BENCHMARK_TYPE,
            "created_at": started_at,
            "git_branch": _git_branch_or_none(),
            "git_head": _git_head_or_none(),
            "output_dir": str(output_dir),
            "allow_api": request.allow_api,
            "dry_run": not request.allow_api,
            "runs_per_plan": request.runs_per_plan,
            "case_count": len(request.cases),
            "valid_count": sum(
                1 for case in request.cases if case.expected_label == EXPECTED_VALID
            ),
            "invalid_count": sum(
                1 for case in request.cases if case.expected_label == EXPECTED_INVALID
            ),
            "category_count": len({case.category for case in request.cases}),
            "planned_provider_call_count": _planned_provider_calls(request),
            "execution_policy": request.execution_policy.to_dict(),
            "retry_failed_from": (
                str(request.retry_failed_from) if request.retry_failed_from else None
            ),
            "historical_reused_cell_count": _historical_reused_cell_count(request),
            "retry_eligible_cell_count": _retry_eligible_cell_count(request),
            "scoring": {
                "false_positive_weight": request.false_positive_weight,
                "false_negative_weight": request.false_negative_weight,
                "decision_bands": {
                    DECISION_STRONG_CANDIDATE: {
                        "false_negative_count": 0,
                        "false_positive_count_max": 1,
                        "overall_boundary_accuracy_min": 0.9375,
                        "required_fragment_coverage_rate_min": 0.90,
                    },
                    DECISION_ACCEPTABLE_WITH_RISK: {
                        "false_negative_count_max": 1,
                        "false_positive_count_max": 2,
                        "overall_boundary_accuracy_min": 0.8125,
                        "required_fragment_coverage_rate_min": 0.80,
                    },
                    DECISION_INCOMPLETE: "any incomplete run matrix",
                },
            },
            "cases": [
                {
                    "case_id": case.case_id,
                    "pair_id": case.pair_id,
                    "category": case.category,
                    "expected_label": case.expected_label,
                    "expected_blocking": case.expected_blocking,
                    "voice_sensitive": case.voice_sensitive,
                }
                for case in request.cases
            ],
            "plans": [plan.to_dict() for plan in request.plans],
            "roles_not_invoked": [
                "Candidate Writer",
                "Quality Evaluator",
                "Repair Writer",
                "publication packaging",
            ],
        }
    )


def _write_summary_csv(path: Path, run_records: tuple[dict[str, Any], ...]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "case_id",
                "pair_id",
                "category",
                "expected_label",
                "plan_id",
                "provider",
                "model",
                "execution_profile",
                "reasoning_effort",
                "execution_status",
                "grounding_pass",
                "blocking_claim_count",
                "boundary_outcome",
                "result_provenance",
                "execution_failure_classification",
                "resume_source_run_id",
                "provider_calls",
            ),
        )
        writer.writeheader()
        for record in run_records:
            counts = record.get("provider_invocation_counts") or {}
            writer.writerow(
                {
                    "case_id": record.get("case_id"),
                    "pair_id": record.get("pair_id"),
                    "category": record.get("category"),
                    "expected_label": record.get("expected_label"),
                    "plan_id": record.get("plan_id"),
                    "provider": record.get("provider"),
                    "model": record.get("model"),
                    "execution_profile": record.get("execution_profile"),
                    "reasoning_effort": record.get("reasoning_effort"),
                    "execution_status": record.get("execution_status"),
                    "grounding_pass": record.get("grounding_pass"),
                    "blocking_claim_count": record.get("blocking_claim_count"),
                    "boundary_outcome": record.get("boundary_outcome"),
                    "result_provenance": record.get("result_provenance"),
                    "execution_failure_classification": record.get(
                        "execution_failure_classification"
                    ),
                    "resume_source_run_id": record.get("resume_source_run_id"),
                    "provider_calls": sum(
                        value for value in counts.values() if isinstance(value, int)
                    ),
                }
            )


def _comparison_text(
    manifest: dict[str, Any],
    run_records: tuple[dict[str, Any], ...],
    metrics: dict[str, Any],
) -> str:
    lines = [
        "# Semantic Grounding Boundary Comparison",
        "",
        f"Experiment: `{manifest['experiment_id']}`",
        "",
        "This artifact reports boundary metrics only and does not select a winner.",
        "",
    ]
    for pair_id in sorted({record["pair_id"] for record in run_records}):
        pair_records = [record for record in run_records if record["pair_id"] == pair_id]
        category = pair_records[0]["category"]
        lines.extend([f"## {pair_id}: {category}", ""])
        for record in pair_records:
            lines.extend(
                [
                    f"### {record['case_id']} / {record['plan_id']}",
                    "",
                    f"expected_label: {record['expected_label']}",
                    f"execution_profile: {record.get('execution_profile')}",
                    f"reasoning_effort: {record.get('reasoning_effort')}",
                    f"execution_status: {record['execution_status']}",
                    f"boundary_outcome: {record['boundary_outcome']}",
                    f"grounding_pass: {record.get('grounding_pass')}",
                    f"blocking_claim_ids: {record.get('blocking_claim_ids')}",
                    f"fragment_coverage: {record.get('must_inspect_fragment_coverage')}",
                    "",
                ]
            )
    lines.extend(["## Metrics", ""])
    for plan_id, plan_metrics in metrics["plans"].items():
        lines.extend(
            [
                f"### {plan_id}",
                "",
                f"decision_band: {plan_metrics['decision_band']}",
                f"false_positive_rate: {plan_metrics['false_positive_rate']}",
                f"false_negative_rate: {plan_metrics['false_negative_rate']}",
                f"required_fragment_coverage_rate: {plan_metrics['required_fragment_coverage_rate']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _report_text(
    manifest: dict[str, Any],
    run_records: tuple[dict[str, Any], ...],
    metrics: dict[str, Any],
) -> str:
    lines = [
        "# Semantic Grounding Boundary Benchmark",
        "",
        f"Experiment: `{manifest['experiment_id']}`",
        f"Runs: {len(run_records)}",
        f"Provider calls: {_provider_calls(run_records)}",
        f"Planned live provider calls: {manifest['planned_provider_call_count']}",
        "",
        "Semantic Grounding is a faithfulness boundary, not a prose sterilizer.",
        "",
        "## Model Metrics",
        "",
    ]
    for plan_id, plan_metrics in metrics["plans"].items():
        lines.extend(
            [
                f"### {plan_id}",
                "",
                f"- decision_band: `{plan_metrics['decision_band']}`",
                f"- overall_boundary_accuracy: {plan_metrics['overall_boundary_accuracy']}",
                f"- strong_voice_preservation_rate: {plan_metrics['strong_voice_preservation_rate']}",
                f"- semantic_overreach_detection_rate: {plan_metrics['semantic_overreach_detection_rate']}",
                f"- weighted_error_score: {plan_metrics['weighted_error_score']}",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def _semantic_grounding_prompt_text(render: Any) -> str:
    prompt_path = getattr(render, "prompt_path", None)
    if not prompt_path:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "semantic grounding prompt path is required for live benchmark execution"
        )
    path = (Path(settings.BASE_DIR) / prompt_path).resolve()
    base_dir = Path(settings.BASE_DIR).resolve()
    if not _is_relative_to(path, base_dir):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "semantic grounding prompt path must stay inside the repository"
        )
    return path.read_text(encoding="utf-8")


def _validate_live_prompt_paths(
    cases: tuple[SemanticGroundingBoundaryCase, ...],
) -> None:
    for case in cases:
        render = build_boundary_semantic_grounding_prompt_render(case)
        _semantic_grounding_prompt_text(render)


def _planned_provider_calls(request: SemanticGroundingBoundaryRequest) -> int:
    if request.retry_failed_from is None:
        return len(request.cases) * len(request.plans) * request.runs_per_plan
    resume_records = _load_resume_records(request.retry_failed_from)
    planned = 0
    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                render = build_boundary_semantic_grounding_prompt_render(case)
                historical = _compatible_historical_record(
                    resume_records,
                    request,
                    case,
                    plan,
                    run_index,
                    render,
                )
                if historical is None or not _is_reusable_historical_record(historical):
                    planned += 1
    return planned


def _historical_reused_cell_count(request: SemanticGroundingBoundaryRequest) -> int:
    if request.retry_failed_from is None:
        return 0
    return _historical_resume_counts(request)["reused"]


def _retry_eligible_cell_count(request: SemanticGroundingBoundaryRequest) -> int:
    if request.retry_failed_from is None:
        return 0
    return _historical_resume_counts(request)["retry_eligible"]


def _historical_resume_counts(request: SemanticGroundingBoundaryRequest) -> dict[str, int]:
    resume_records = _load_resume_records(request.retry_failed_from)
    counts = {"reused": 0, "retry_eligible": 0}
    for case in request.cases:
        for plan in request.plans:
            for run_index in range(1, request.runs_per_plan + 1):
                render = build_boundary_semantic_grounding_prompt_render(case)
                historical = _compatible_historical_record(
                    resume_records,
                    request,
                    case,
                    plan,
                    run_index,
                    render,
                )
                if historical is None:
                    counts["retry_eligible"] += 1
                elif _is_reusable_historical_record(historical):
                    counts["reused"] += 1
                else:
                    counts["retry_eligible"] += 1
    return counts


def _provider_calls(run_records: tuple[dict[str, Any], ...]) -> int:
    total = 0
    for record in run_records:
        counts = record.get("provider_invocation_counts") or {}
        total += sum(value for value in counts.values() if isinstance(value, int))
    return total


def _semantic_input_summary(render: Any) -> dict[str, Any]:
    return {
        name: {
            "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            "character_count": len(value),
        }
        for name, value in sorted(render.variables.items())
    }


def _raw_response_diagnostics(raw_response: Any) -> dict[str, Any]:
    raw_text = str(getattr(raw_response, "raw_text", "") or "")
    stripped = raw_text.strip()
    diagnostics = {
        "raw_response_character_count": len(raw_text),
        "stripped_response_character_count": len(stripped),
        "starts_with_json_object": stripped.startswith("{"),
        "starts_with_json_array": stripped.startswith("["),
        "starts_with_code_fence": stripped.startswith("```"),
        "ends_with_json_object": stripped.endswith("}"),
        "ends_with_json_array": stripped.endswith("]"),
        "ends_with_code_fence": stripped.endswith("```"),
        "brace_balance": raw_text.count("{") - raw_text.count("}"),
        "bracket_balance": raw_text.count("[") - raw_text.count("]"),
    }
    metadata = getattr(raw_response, "provider_response_metadata", None)
    if isinstance(metadata, dict):
        diagnostics["provider_response_metadata"] = _safe_provider_response_metadata(
            metadata
        )
    usage = getattr(raw_response, "usage", None)
    if isinstance(usage, dict):
        diagnostics["usage"] = _safe_token_usage(usage)
    execution_diagnostics = getattr(raw_response, "execution_diagnostics", None)
    if isinstance(execution_diagnostics, dict):
        diagnostics["execution_diagnostics"] = _safe_execution_diagnostics(
            execution_diagnostics
        )
    return diagnostics


def _safe_execution_diagnostics(diagnostics: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "exception_class",
        "provider",
        "model",
        "http_status",
        "error_code",
        "retryable",
        "timeout",
        "rate_limited",
        "message",
    ):
        value = diagnostics.get(key)
        if isinstance(value, str):
            safe[key] = (
                "provider execution failed; raw diagnostic message omitted"
                if key == "message" and value.strip()
                else _safe_text(value)
            )
        elif isinstance(value, bool) or value is None:
            safe[key] = value
        elif isinstance(value, int):
            safe[key] = value
    return _sanitize_artifact_value(safe)


def _safe_provider_response_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key in (
        "provider",
        "model",
        "stop_reason",
        "choices_count",
        "finish_reasons",
        "message_content_types",
        "content_block_types",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
        "thinking_tokens",
    ):
        value = metadata.get(key)
        if isinstance(value, str):
            safe[key] = value[:120]
        elif isinstance(value, int) and not isinstance(value, bool):
            safe[key] = value
        elif isinstance(value, list):
            safe[key] = [str(item)[:120] for item in value[:20]]
        elif value is None:
            safe[key] = None
    return safe


def _safe_token_usage(usage: dict[str, Any]) -> dict[str, int | None]:
    safe: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(key)
        if value is None or (isinstance(value, int) and not isinstance(value, bool)):
            safe[key] = value
    return safe


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


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def _validate_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not SAFE_IDENTIFIER_RE.fullmatch(value):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"{field_name} must be a non-empty safe identifier"
        )


def _required_string(payload: Any, field_name: str) -> str:
    if isinstance(payload, dict):
        value = payload.get(field_name)
    else:
        value = payload
    if not isinstance(value, str) or not value.strip():
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"{field_name} must be a non-empty string"
        )
    return value


def _optional_string(payload: dict[str, Any], field_name: str) -> str:
    value = payload.get(field_name, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"{field_name} must be a string"
        )
    return value


def _required_bool(payload: dict[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"{field_name} must be a boolean"
        )
    return value


def _required_dict(payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    value = payload.get(field_name)
    if not isinstance(value, dict):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            f"{field_name} must be an object"
        )
    return copy.deepcopy(value)


def _resolve_output_root(output_root: Path | None) -> Path:
    base_dir = Path(settings.BASE_DIR).resolve()
    default_root = (base_dir / DEFAULT_OUTPUT_ROOT).resolve()
    if output_root is None:
        _ensure_default_output_root_ignored(base_dir)
        return default_root
    resolved = Path(output_root).resolve()
    if _is_relative_to(resolved, base_dir) and not _is_relative_to(resolved, default_root):
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "output_root inside the repository must be under "
            "debug_outputs/final_post_semantic_grounding_boundary_benchmarks"
        )
    if _is_relative_to(resolved, default_root):
        _ensure_default_output_root_ignored(base_dir)
    return resolved


def _ensure_default_output_root_ignored(base_dir: Path) -> None:
    gitignore = base_dir / ".gitignore"
    if not gitignore.exists():
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "debug_outputs is not ignored"
        )
    ignored_patterns = {
        line.strip()
        for line in gitignore.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    if "debug_outputs/" not in ignored_patterns and "debug_outputs" not in ignored_patterns:
        raise SemanticGroundingBoundaryBenchmarkConfigurationError(
            "debug_outputs is not ignored"
        )


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
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace(
        "+00:00",
        "Z",
    )


def _git_branch_or_none() -> str | None:
    try:
        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=settings.BASE_DIR,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return None
    return result.stdout.strip() or None if result.returncode == 0 else None


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
    return result.stdout.strip() or None if result.returncode == 0 else None
