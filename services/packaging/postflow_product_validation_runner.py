"""PostFlow product validation benchmark runner."""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
)
from services.packaging.postflow_product_validation_corpus import (
    DEFAULT_MANIFEST_PATH,
    ProductValidationCorpusError,
    ProductValidationCorpusManifest,
    load_product_validation_corpus_manifest,
    manifest_content_hash,
    resolve_product_validation_case,
    sha256_file,
)
from services.packaging.postflow_product_validation_metrics import (
    compare_product_validation_metrics,
    compute_product_validation_metrics,
)
from services.packaging.postflow_product_validation_live_execution import (
    PRODUCT_CORPUS_MINIMUM_CASES,
    PRODUCT_CORPUS_PREFERRED_CASES,
    ProductValidationLiveExecutors,
    execute_product_validation_live_case,
    planned_provider_calls_for_case_family,
    runtime_input_for_case,
)
from services.packaging.postflow_product_validation_reports import (
    build_product_validation_report_markdown,
    write_cases_csv,
    write_human_review_csv,
    write_human_review_markdown,
    write_json,
    write_jsonl,
)

PRODUCT_VALIDATION_BENCHMARK_SCHEMA_VERSION = "2026-08-21"
DEFAULT_EXPERIMENT_ID = "scope-26a-product-validation-v1"
DEFAULT_OUTPUT_ROOT = Path("debug_outputs/postflow_product_validation_benchmarks")
STATUS_DRY_RUN = "dry_run"
STATUS_COMPLETED = "completed"
STATUS_CONFIG_ERROR = "config_error"

FROZEN_ROLE_CONFIGURATION = {
    FINAL_POST_ROLE_CANDIDATE_WRITER: {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
    },
    FINAL_POST_ROLE_SEMANTIC_GROUNDING: {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "execution_profile": "grounding_minimal_reasoning",
        "reasoning_effort": "minimal",
        "max_output_tokens": 4800,
        "json_mode": True,
    },
    FINAL_POST_ROLE_QUALITY_EVALUATOR: {
        "provider": "openai",
        "model": "gpt-4.1-2025-04-14",
    },
    FINAL_POST_ROLE_REPAIR_WRITER: {
        "provider": "gemini",
        "model": "gemini-3.6-flash",
        "execution_profile": "gemini_repair_minimal_reasoning",
        "reasoning_effort": "minimal",
        "max_output_tokens": 2800,
    },
}
FINGERPRINT_FILE_PATHS = (
    "prompts/linkedin/final_post_from_brief.txt",
    "prompts/linkedin/final_post_semantic_grounding_evaluator.txt",
    "prompts/linkedin/final_post_quality_evaluator.txt",
    "services/packaging/linkedin_post_controlled_repair_execution.py",
    "services/packaging/linkedin_post_prompt_renderers.py",
    "services/packaging/linkedin_post_repair_writer_execution.py",
    "services/packaging/linkedin_post_final_post_payload_contract.py",
    "services/packaging/linkedin_post_quality_review_contract.py",
    "services/packaging/linkedin_post_semantic_grounding_contract.py",
)


@dataclass(frozen=True)
class ProductValidationRequest:
    experiment_id: str = DEFAULT_EXPERIMENT_ID
    manifest_path: Path = DEFAULT_MANIFEST_PATH
    output_root: Path = DEFAULT_OUTPUT_ROOT
    compare_to: Path | None = None
    export_human_review: bool = True
    fail_on_corpus_invalid: bool = True
    allow_api: bool = False
    dry_run: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "manifest_path": str(self.manifest_path),
            "output_root": str(self.output_root),
            "compare_to": str(self.compare_to) if self.compare_to else None,
            "export_human_review": self.export_human_review,
            "fail_on_corpus_invalid": self.fail_on_corpus_invalid,
            "allow_api": self.allow_api,
            "dry_run": self.dry_run,
        }


@dataclass(frozen=True)
class ProductValidationArtifacts:
    output_dir: str
    corpus_manifest_json: str
    run_records_jsonl: str
    cases_csv: str
    metrics_json: str
    report_md: str
    human_review_csv: str
    human_review_md: str
    configuration_fingerprint_json: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_dir": self.output_dir,
            "corpus_manifest_json": self.corpus_manifest_json,
            "run_records_jsonl": self.run_records_jsonl,
            "cases_csv": self.cases_csv,
            "metrics_json": self.metrics_json,
            "report_md": self.report_md,
            "human_review_csv": self.human_review_csv,
            "human_review_md": self.human_review_md,
            "configuration_fingerprint_json": self.configuration_fingerprint_json,
        }


@dataclass(frozen=True)
class ProductValidationResult:
    status: str
    exit_code: int
    experiment_id: str
    run_count: int
    provider_call_count: int
    corpus_case_count: int
    artifacts: ProductValidationArtifacts | None
    metrics: dict[str, Any]
    configuration_fingerprint: dict[str, Any]
    baseline_comparison: dict[str, Any] | None = None
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
            "corpus_case_count": self.corpus_case_count,
            "artifacts": self.artifacts.to_dict() if self.artifacts else None,
            "metrics": copy.deepcopy(self.metrics),
            "configuration_fingerprint": copy.deepcopy(self.configuration_fingerprint),
            "baseline_comparison": copy.deepcopy(self.baseline_comparison),
            "safe_failure_code": self.safe_failure_code,
            "safe_failure_message": self.safe_failure_message,
            "run_records": copy.deepcopy(list(self.run_records)),
        }


def run_product_validation_benchmark(
    request: ProductValidationRequest,
    *,
    now_factory: Callable[[], datetime] | None = None,
    live_executors: ProductValidationLiveExecutors | None = None,
) -> ProductValidationResult:
    now = (now_factory or _utc_now)()
    try:
        _validate_execution_mode(request)
        manifest = load_product_validation_corpus_manifest(request.manifest_path)
        _validate_frozen_role_configuration(manifest)
        started_at = now.isoformat()
        records = tuple(
            _build_record(
                case,
                index,
                request=request,
                started_at=started_at,
                completed_at=(now_factory or _utc_now)().isoformat(),
                live_executors=live_executors,
            )
            for index, case in enumerate(manifest.cases, 1)
        )
        metrics = compute_product_validation_metrics(records)
        baseline_metrics = _load_baseline_metrics(request.compare_to)
        baseline_comparison = compare_product_validation_metrics(metrics, baseline_metrics)
        fingerprint = build_configuration_fingerprint(
            manifest,
            now=now,
            provider_call_count=_provider_calls(records),
            live_mode=request.allow_api,
        )
        artifacts = _write_artifacts(
            request=request,
            manifest=manifest,
            records=records,
            metrics=metrics,
            fingerprint=fingerprint,
            baseline_comparison=baseline_comparison,
        )
    except (OSError, json.JSONDecodeError, ProductValidationCorpusError, ValueError) as exc:
        if request.fail_on_corpus_invalid:
            return ProductValidationResult(
                status=STATUS_CONFIG_ERROR,
                exit_code=1,
                experiment_id=request.experiment_id,
                run_count=0,
                provider_call_count=0,
                corpus_case_count=0,
                artifacts=None,
                metrics={},
                configuration_fingerprint={},
                safe_failure_code="product_validation_corpus_invalid",
                safe_failure_message=str(exc),
            )
        raise
    return ProductValidationResult(
        status=STATUS_COMPLETED if request.allow_api else STATUS_DRY_RUN,
        exit_code=0,
        experiment_id=request.experiment_id,
        run_count=len(records),
        provider_call_count=_provider_calls(records),
        corpus_case_count=len(records),
        artifacts=artifacts,
        metrics=metrics,
        configuration_fingerprint=fingerprint,
        baseline_comparison=baseline_comparison,
        run_records=records,
    )


def build_configuration_fingerprint(
    manifest: ProductValidationCorpusManifest,
    *,
    now: datetime | None = None,
    provider_call_count: int = 0,
    live_mode: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": PRODUCT_VALIDATION_BENCHMARK_SCHEMA_VERSION,
        "created_at": (now or _utc_now()).isoformat(),
        "git_branch": _git_value("branch", "--show-current"),
        "git_head": _git_value("rev-parse", "HEAD"),
        "manifest_hash": manifest_content_hash(manifest),
        "fixture_hashes": {
            case.case_id: case.sha256 for case in manifest.cases
        },
        "frozen_model_role_configuration": copy.deepcopy(FROZEN_ROLE_CONFIGURATION),
        "contract_file_hashes": {
            path: sha256_file(Path(path))
            for path in FINGERPRINT_FILE_PATHS
            if Path(path).exists()
        },
        "missing_contract_file_paths": [
            path for path in FINGERPRINT_FILE_PATHS if not Path(path).exists()
        ],
        "provider_call_count": provider_call_count,
        "live_mode": live_mode,
        "benchmark_runner_version": PRODUCT_VALIDATION_BENCHMARK_SCHEMA_VERSION,
    }


def _validate_frozen_role_configuration(manifest: ProductValidationCorpusManifest) -> None:
    if manifest.frozen_model_role_configuration != FROZEN_ROLE_CONFIGURATION:
        raise ProductValidationCorpusError(
            "manifest frozen_model_role_configuration does not match runner policy"
        )


def _validate_execution_mode(request: ProductValidationRequest) -> None:
    if request.allow_api and request.dry_run:
        raise ProductValidationCorpusError("--allow-api and --dry-run are mutually exclusive")
    if not request.allow_api and not request.dry_run:
        raise ProductValidationCorpusError("live execution requires --allow-api")


def _build_record(
    case,
    ordinal: int,
    *,
    request: ProductValidationRequest,
    started_at: str,
    completed_at: str,
    live_executors: ProductValidationLiveExecutors | None,
) -> dict[str, Any]:
    payload = resolve_product_validation_case(case)
    runtime_input = runtime_input_for_case(case, payload)
    planned_counts = planned_provider_calls_for_case_family(case.family)
    live_fields = (
        execute_product_validation_live_case(
            case=case,
            fixture_payload=payload,
            experiment_id=request.experiment_id,
            started_at=started_at,
            completed_at=completed_at,
            executors=live_executors,
        )
        if request.allow_api
        else _dry_live_fields(planned_counts)
    )
    total_provider_calls = int(live_fields.get("total_provider_calls", 0) or 0)
    return {
        "ordinal": ordinal,
        "case_id": case.case_id,
        "family": case.family,
        "topic": case.topic,
        "risk_type": case.risk_type,
        "fixture_path": str(case.fixture_path),
        "boundary_case_id": case.boundary_case_id,
        "fixture_valid": True,
        "expected_outcome": case.expected_outcome,
        "historical_expected_outcome": case.expected_outcome,
        "historical_classification": case.failure_category,
        "historical_notes": case.notes,
        "historical_outcome": case.expected_outcome,
        "failure_category": case.failure_category,
        "human_review_required": case.human_review_required,
        "provenance": copy.deepcopy(case.provenance),
        "notes": case.notes,
        "runtime_input_fingerprint": _sha256_json(runtime_input),
        "runtime_input_keys": sorted(runtime_input),
        "planned_provider_invocation_counts": planned_counts,
        "provider_invocation_counts": copy.deepcopy(
            live_fields.get("provider_invocation_counts", {})
        ),
        "provider_call_count": total_provider_calls,
        "total_provider_calls": total_provider_calls,
        "publication_packaging_invocations": int(
            live_fields.get("publication_packaging_invocations", 0) or 0
        ),
        "product_case_distribution": _product_distribution(case, payload),
        **copy.deepcopy(live_fields),
    }


def _dry_live_fields(planned_counts: dict[str, int]) -> dict[str, Any]:
    return {
        "live_execution_status": STATUS_DRY_RUN,
        "live_outcome": None,
        "live_failure_category": None,
        "live_failure_code": None,
        "live_failure_stage": None,
        "live_stage_outcomes": {},
        "provider_invocation_counts": {key: 0 for key in planned_counts},
        "total_provider_calls": 0,
        "publication_packaging_invocations": 0,
    }


def _product_distribution(case, payload: dict[str, Any]) -> dict[str, Any]:
    candidate = payload.get("candidate_payload") if isinstance(payload, dict) else None
    post_text = candidate.get("post_text", "") if isinstance(candidate, dict) else ""
    length = len(post_text) if isinstance(post_text, str) else None
    return {
        "topic": case.topic,
        "writer_provider": payload.get("writer_provider") if isinstance(payload, dict) else None,
        "writer_model": payload.get("writer_model") if isinstance(payload, dict) else None,
        "length": length,
        "length_band": _length_band(length or 0),
        "repair_target": payload.get("repair_target") if isinstance(payload, dict) else None,
        "quality_classification": (
            payload.get("editorial_classification") if isinstance(payload, dict) else None
        ),
        "regression_anchor": case.case_id in {
            "topic_140_digest_126",
            "topic_214_digest_128__claude_v3",
            "topic_200_digest_134__gpt_v2",
        },
        "inclusion_reason": case.notes,
    }


def _length_band(length: int) -> str:
    if length >= 1200:
        return "near_limit"
    if length >= 900:
        return "long"
    if length >= 500:
        return "medium"
    return "short"


def _write_artifacts(
    *,
    request: ProductValidationRequest,
    manifest: ProductValidationCorpusManifest,
    records: tuple[dict[str, Any], ...],
    metrics: dict[str, Any],
    fingerprint: dict[str, Any],
    baseline_comparison: dict[str, Any] | None,
) -> ProductValidationArtifacts:
    output_dir = request.output_root / request.experiment_id
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ProductValidationCorpusError(
            f"output directory already contains artifacts: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "corpus_manifest.json"
    records_path = output_dir / "product_validation_runs.jsonl"
    cases_path = output_dir / "product_validation_cases.csv"
    metrics_path = output_dir / "product_validation_metrics.json"
    report_path = output_dir / "product_validation_report.md"
    human_csv_path = output_dir / "human_review.csv"
    human_md_path = output_dir / "human_review.md"
    fingerprint_path = output_dir / "configuration_fingerprint.json"

    manifest_payload = manifest.to_dict()
    manifest_payload["configuration_fingerprint"] = fingerprint
    manifest_payload["product_corpus_adequacy"] = _product_corpus_adequacy(manifest)
    write_json(manifest_path, manifest_payload)
    write_jsonl(records_path, records)
    write_cases_csv(cases_path, records)
    write_json(metrics_path, {"metrics": metrics, "baseline_comparison": baseline_comparison})
    report_path.write_text(
        build_product_validation_report_markdown(
            experiment_id=request.experiment_id,
            manifest=manifest.to_dict(),
            metrics=metrics,
            baseline_comparison=baseline_comparison,
        ),
        encoding="utf-8",
    )
    if request.export_human_review:
        write_human_review_csv(human_csv_path, records)
        write_human_review_markdown(human_md_path, records)
        human_csv_artifact = str(human_csv_path)
        human_md_artifact = str(human_md_path)
    else:
        human_csv_artifact = ""
        human_md_artifact = ""
    write_json(fingerprint_path, fingerprint)
    return ProductValidationArtifacts(
        output_dir=str(output_dir),
        corpus_manifest_json=str(manifest_path),
        run_records_jsonl=str(records_path),
        cases_csv=str(cases_path),
        metrics_json=str(metrics_path),
        report_md=str(report_path),
        human_review_csv=human_csv_artifact,
        human_review_md=human_md_artifact,
        configuration_fingerprint_json=str(fingerprint_path),
    )


def _load_baseline_metrics(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "metrics" in data and isinstance(data["metrics"], dict):
        _validate_baseline_corpus_compatibility(data)
        return data["metrics"]
    if isinstance(data, dict):
        return data
    raise ValueError("baseline metrics must be a JSON object")


def _validate_baseline_corpus_compatibility(data: dict[str, Any]) -> None:
    manifest = data.get("manifest") or data.get("corpus_manifest")
    if not isinstance(manifest, dict):
        return
    prior_fingerprint = manifest.get("configuration_fingerprint")
    prior_hash = (
        prior_fingerprint.get("manifest_hash")
        if isinstance(prior_fingerprint, dict)
        else None
    )
    if not prior_hash:
        return
    current_hash = manifest_content_hash(load_product_validation_corpus_manifest())
    if prior_hash != current_hash:
        raise ValueError("CORPUS_MISMATCH")


def _product_corpus_adequacy(manifest: ProductValidationCorpusManifest) -> dict[str, Any]:
    product_count = sum(
        1 for case in manifest.cases if case.family == "candidate_quality_case"
    )
    return {
        "status": (
            "PRODUCT_CORPUS_INSUFFICIENT"
            if product_count < PRODUCT_CORPUS_MINIMUM_CASES
            else "PRODUCT_CORPUS_ADEQUATE"
        ),
        "current_product_case_count": product_count,
        "minimum_product_case_count": PRODUCT_CORPUS_MINIMUM_CASES,
        "preferred_product_case_count": PRODUCT_CORPUS_PREFERRED_CASES,
        "gap_to_minimum": max(PRODUCT_CORPUS_MINIMUM_CASES - product_count, 0),
        "gap_to_preferred": max(PRODUCT_CORPUS_PREFERRED_CASES - product_count, 0),
    }


def _provider_calls(records: tuple[dict[str, Any], ...]) -> int:
    return sum(int(record.get("total_provider_calls", 0) or 0) for record in records)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ("git", *args),
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _utc_now() -> datetime:
    return datetime.now(UTC)
