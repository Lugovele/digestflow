"""Corpus contracts for provider-free PostFlow product validation benchmarks.

This module is benchmark infrastructure only. It does not execute prompts,
call providers, import provider executors, mutate production inputs, or persist
publication packages.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRODUCT_VALIDATION_SCHEMA_VERSION = "2026-08-21"
PRODUCT_VALIDATION_CORPUS_ID = "postflow_product_validation_v1"
DEFAULT_MANIFEST_PATH = Path(
    "tests/fixtures/postflow_product_validation_benchmark/manifest_v1.json"
)
TARGET_CASE_COUNT = 27
FROZEN_ANCHOR_CASE_IDS = (
    "topic_140_digest_126",
    "topic_214_digest_128__claude_v3",
    "topic_200_digest_134__gpt_v2",
)

CASE_FAMILY_SOURCE_ARTICLE = "source_article_case"
CASE_FAMILY_CANDIDATE_QE = "candidate_quality_case"
CASE_FAMILY_SEMANTIC_BOUNDARY = "semantic_boundary_case"
CASE_FAMILIES = (
    CASE_FAMILY_SOURCE_ARTICLE,
    CASE_FAMILY_CANDIDATE_QE,
    CASE_FAMILY_SEMANTIC_BOUNDARY,
)

OUTCOME_FIXTURE_VALID = "fixture_valid"
OUTCOME_SOURCE_READY = "source_ready"
OUTCOME_CANDIDATE_SHAPE_VALID = "candidate_shape_valid"
OUTCOME_KNOWN_ACCEPTED_PROJECTION = "known_accepted_projection"
OUTCOME_KNOWN_REPAIR_REQUIRED = "known_repair_required"
OUTCOME_KNOWN_GROUNDING_BLOCKED = "known_grounding_blocked"
OUTCOME_KNOWN_QUALITY_REJECTED = "known_quality_rejected"
OUTCOME_BOUNDARY_VALID_PRESERVED = "boundary_valid_preserved"
OUTCOME_BOUNDARY_INVALID_BLOCKED = "boundary_invalid_blocked"
OUTCOME_HISTORICAL_INCONCLUSIVE = "historical_inconclusive"
OUTCOME_BENCHMARK_INFRA_INVALID = "benchmark_infra_invalid"
OUTCOMES = (
    OUTCOME_FIXTURE_VALID,
    OUTCOME_SOURCE_READY,
    OUTCOME_CANDIDATE_SHAPE_VALID,
    OUTCOME_KNOWN_ACCEPTED_PROJECTION,
    OUTCOME_KNOWN_REPAIR_REQUIRED,
    OUTCOME_KNOWN_GROUNDING_BLOCKED,
    OUTCOME_KNOWN_QUALITY_REJECTED,
    OUTCOME_BOUNDARY_VALID_PRESERVED,
    OUTCOME_BOUNDARY_INVALID_BLOCKED,
    OUTCOME_HISTORICAL_INCONCLUSIVE,
    OUTCOME_BENCHMARK_INFRA_INVALID,
)

FAILURE_CATEGORY_PRODUCT = "product_failure"
FAILURE_CATEGORY_INFRASTRUCTURE = "infrastructure_failure"
FAILURE_CATEGORY_PRODUCT_BEHAVIOR = "product_behavior"
FAILURE_CATEGORIES = (
    FAILURE_CATEGORY_PRODUCT,
    FAILURE_CATEGORY_INFRASTRUCTURE,
    FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
)

SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,119}$")


@dataclass(frozen=True)
class ProductValidationCorpusCase:
    case_id: str
    family: str
    fixture_path: Path
    expected_outcome: str
    failure_category: str
    topic: str
    risk_type: str
    provenance: dict[str, Any]
    sha256: str
    boundary_case_id: str | None = None
    human_review_required: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "family": self.family,
            "fixture_path": str(self.fixture_path),
            "expected_outcome": self.expected_outcome,
            "failure_category": self.failure_category,
            "topic": self.topic,
            "risk_type": self.risk_type,
            "provenance": copy.deepcopy(self.provenance),
            "sha256": self.sha256,
            "boundary_case_id": self.boundary_case_id,
            "human_review_required": self.human_review_required,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ProductValidationCorpusManifest:
    corpus_id: str
    schema_version: str
    target_case_count: int
    frozen_anchor_case_ids: tuple[str, ...]
    cases: tuple[ProductValidationCorpusCase, ...]
    frozen_model_role_configuration: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "corpus_id": self.corpus_id,
            "schema_version": self.schema_version,
            "target_case_count": self.target_case_count,
            "frozen_anchor_case_ids": list(self.frozen_anchor_case_ids),
            "cases": [case.to_dict() for case in self.cases],
            "frozen_model_role_configuration": copy.deepcopy(
                self.frozen_model_role_configuration
            ),
            "provenance": copy.deepcopy(self.provenance),
        }


class ProductValidationCorpusError(ValueError):
    """Raised when the benchmark manifest or referenced fixture corpus is invalid."""


def load_product_validation_corpus_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST_PATH,
) -> ProductValidationCorpusManifest:
    path = Path(manifest_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = tuple(_case_from_dict(raw) for raw in data.get("cases", ()))
    manifest = ProductValidationCorpusManifest(
        corpus_id=str(data.get("corpus_id", "")),
        schema_version=str(data.get("schema_version", "")),
        target_case_count=int(data.get("target_case_count", 0)),
        frozen_anchor_case_ids=tuple(data.get("frozen_anchor_case_ids", ())),
        cases=cases,
        frozen_model_role_configuration=dict(
            data.get("frozen_model_role_configuration", {})
        ),
        provenance=dict(data.get("provenance", {})),
    )
    validate_product_validation_corpus_manifest(manifest)
    return manifest


def validate_product_validation_corpus_manifest(
    manifest: ProductValidationCorpusManifest,
) -> None:
    if manifest.corpus_id != PRODUCT_VALIDATION_CORPUS_ID:
        raise ProductValidationCorpusError("unexpected product validation corpus_id")
    if manifest.schema_version != PRODUCT_VALIDATION_SCHEMA_VERSION:
        raise ProductValidationCorpusError("unexpected product validation schema_version")
    if manifest.target_case_count != TARGET_CASE_COUNT:
        raise ProductValidationCorpusError("unexpected target_case_count")
    if len(manifest.cases) != TARGET_CASE_COUNT:
        raise ProductValidationCorpusError("manifest must contain exactly 27 cases")
    case_ids = [case.case_id for case in manifest.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ProductValidationCorpusError("case_id values must be unique")
    for anchor_id in FROZEN_ANCHOR_CASE_IDS:
        if anchor_id not in case_ids:
            raise ProductValidationCorpusError(f"missing frozen anchor case_id: {anchor_id}")
    for case in manifest.cases:
        _validate_case(case)


def resolve_product_validation_case(case: ProductValidationCorpusCase) -> dict[str, Any]:
    if not case.fixture_path.exists():
        raise ProductValidationCorpusError(f"missing fixture: {case.fixture_path}")
    actual_hash = sha256_file(case.fixture_path)
    if actual_hash != case.sha256:
        raise ProductValidationCorpusError(
            f"fixture hash mismatch for {case.case_id}: expected {case.sha256} got {actual_hash}"
        )
    payload = json.loads(case.fixture_path.read_text(encoding="utf-8-sig"))
    if case.family == CASE_FAMILY_SEMANTIC_BOUNDARY:
        boundary_cases = payload.get("cases") if isinstance(payload, dict) else None
        if not isinstance(boundary_cases, list):
            raise ProductValidationCorpusError("semantic boundary fixture missing cases list")
        match = [item for item in boundary_cases if item.get("case_id") == case.boundary_case_id]
        if len(match) != 1:
            raise ProductValidationCorpusError(
                f"boundary case not found exactly once: {case.boundary_case_id}"
            )
        payload = match[0]
    return copy.deepcopy(payload)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_content_hash(manifest: ProductValidationCorpusManifest) -> str:
    payload = json.dumps(
        manifest.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _case_from_dict(raw: dict[str, Any]) -> ProductValidationCorpusCase:
    return ProductValidationCorpusCase(
        case_id=str(raw.get("case_id", "")),
        family=str(raw.get("family", "")),
        fixture_path=Path(str(raw.get("fixture_path", ""))),
        expected_outcome=str(raw.get("expected_outcome", "")),
        failure_category=str(raw.get("failure_category", "")),
        topic=str(raw.get("topic", "")),
        risk_type=str(raw.get("risk_type", "")),
        provenance=dict(raw.get("provenance", {})),
        sha256=str(raw.get("sha256", "")),
        boundary_case_id=raw.get("boundary_case_id"),
        human_review_required=bool(raw.get("human_review_required", False)),
        notes=str(raw.get("notes", "")),
    )


def _validate_case(case: ProductValidationCorpusCase) -> None:
    if not SAFE_IDENTIFIER_RE.match(case.case_id):
        raise ProductValidationCorpusError(f"invalid case_id: {case.case_id}")
    if case.family not in CASE_FAMILIES:
        raise ProductValidationCorpusError(f"invalid case family: {case.family}")
    if case.expected_outcome not in OUTCOMES:
        raise ProductValidationCorpusError(
            f"invalid expected_outcome for {case.case_id}: {case.expected_outcome}"
        )
    if case.failure_category not in FAILURE_CATEGORIES:
        raise ProductValidationCorpusError(
            f"invalid failure_category for {case.case_id}: {case.failure_category}"
        )
    if case.family == CASE_FAMILY_SEMANTIC_BOUNDARY and not case.boundary_case_id:
        raise ProductValidationCorpusError("semantic boundary cases require boundary_case_id")
    if not case.sha256:
        raise ProductValidationCorpusError(f"missing sha256 for {case.case_id}")
