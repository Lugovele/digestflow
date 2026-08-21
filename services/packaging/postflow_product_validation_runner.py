"""Provider-free PostFlow product validation benchmark runner."""
from __future__ import annotations

import copy
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

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "manifest_path": str(self.manifest_path),
            "output_root": str(self.output_root),
            "compare_to": str(self.compare_to) if self.compare_to else None,
            "export_human_review": self.export_human_review,
            "fail_on_corpus_invalid": self.fail_on_corpus_invalid,
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
) -> ProductValidationResult:
    now = (now_factory or _utc_now)()
    try:
        manifest = load_product_validation_corpus_manifest(request.manifest_path)
        _validate_frozen_role_configuration(manifest)
        records = tuple(_build_record(case, index) for index, case in enumerate(manifest.cases, 1))
        metrics = compute_product_validation_metrics(records)
        baseline_metrics = _load_baseline_metrics(request.compare_to)
        baseline_comparison = compare_product_validation_metrics(metrics, baseline_metrics)
        fingerprint = build_configuration_fingerprint(manifest, now=now)
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
        status=STATUS_DRY_RUN,
        exit_code=0,
        experiment_id=request.experiment_id,
        run_count=len(records),
        provider_call_count=0,
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
        "provider_call_count": 0,
    }


def _validate_frozen_role_configuration(manifest: ProductValidationCorpusManifest) -> None:
    if manifest.frozen_model_role_configuration != FROZEN_ROLE_CONFIGURATION:
        raise ProductValidationCorpusError(
            "manifest frozen_model_role_configuration does not match runner policy"
        )


def _build_record(case, ordinal: int) -> dict[str, Any]:
    resolve_product_validation_case(case)
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
        "historical_outcome": case.expected_outcome,
        "failure_category": case.failure_category,
        "human_review_required": case.human_review_required,
        "provenance": copy.deepcopy(case.provenance),
        "notes": case.notes,
        "provider_call_count": 0,
    }


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
        return data["metrics"]
    if isinstance(data, dict):
        return data
    raise ValueError("baseline metrics must be a JSON object")


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
