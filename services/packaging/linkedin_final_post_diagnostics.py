"""Deterministic diagnostics for final LinkedIn post candidates."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


REQUIRED_QUALITY_CHECK_KEYS = {
    "linkedin_ready",
    "uses_only_provided_facts",
    "has_clear_point_of_view",
}
HUMAN_FACING_FIELDS = ("post_text", "hook_variants", "cta_variants")
SCAFFOLD_PHRASES = (
    "selected evidence",
    "evidence above",
    "source-grounded",
    "separate signals",
    "keep claims attributed",
    "avoid unsupported conclusions",
)
SOURCE_SUMMARY_PHRASES = (
    "according to one source",
    "another source",
    "research cited in a third source",
    "sources matter",
)


@dataclass(frozen=True)
class TextLeak:
    field_name: str
    value_index: int | None
    phrase: str
    leak_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "field_name": self.field_name,
            "value_index": self.value_index,
            "phrase": self.phrase,
            "leak_type": self.leak_type,
        }


@dataclass(frozen=True)
class FinalPostDiagnostics:
    schema_validation_passed: bool
    schema_validation_error: str
    missing_quality_check_keys: list[str]
    non_boolean_quality_check_keys: list[str]
    evidence_id_leaks: list[TextLeak]
    scaffold_phrase_leaks: list[TextLeak]
    source_summary_phrase_leaks: list[TextLeak]
    model_claimed_linkedin_ready: bool | None
    deterministic_checks_passed: bool
    system_linkedin_ready: bool
    repair_reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_validation_passed": self.schema_validation_passed,
            "schema_validation_error": self.schema_validation_error,
            "missing_quality_check_keys": list(self.missing_quality_check_keys),
            "non_boolean_quality_check_keys": list(self.non_boolean_quality_check_keys),
            "evidence_id_leaks": [leak.to_dict() for leak in self.evidence_id_leaks],
            "scaffold_phrase_leaks": [leak.to_dict() for leak in self.scaffold_phrase_leaks],
            "source_summary_phrase_leaks": [
                leak.to_dict() for leak in self.source_summary_phrase_leaks
            ],
            "model_claimed_linkedin_ready": self.model_claimed_linkedin_ready,
            "deterministic_checks_passed": self.deterministic_checks_passed,
            "system_linkedin_ready": self.system_linkedin_ready,
            "repair_reasons": list(self.repair_reasons),
        }


def diagnose_candidate_post_payload(
    payload: dict,
    *,
    selected_evidence_ids: list[str],
    schema_validation_passed: bool,
    schema_validation_error: str = "",
) -> FinalPostDiagnostics:
    """Diagnose the core CandidatePost payload without packaging self-checks."""

    evidence_id_leaks = _find_phrase_leaks(
        payload,
        [evidence_id for evidence_id in selected_evidence_ids if evidence_id],
        leak_type="evidence_id",
    )
    scaffold_phrase_leaks = _find_phrase_leaks(
        payload,
        SCAFFOLD_PHRASES,
        leak_type="scaffold_phrase",
    )
    source_summary_phrase_leaks = _find_phrase_leaks(
        payload,
        SOURCE_SUMMARY_PHRASES,
        leak_type="source_summary_phrase",
    )

    deterministic_checks_passed = not (
        evidence_id_leaks
        or scaffold_phrase_leaks
        or source_summary_phrase_leaks
    )
    system_linkedin_ready = schema_validation_passed and deterministic_checks_passed
    repair_reasons = _build_candidate_post_repair_reasons(
        schema_validation_passed=schema_validation_passed,
        evidence_id_leaks=evidence_id_leaks,
        scaffold_phrase_leaks=scaffold_phrase_leaks,
        source_summary_phrase_leaks=source_summary_phrase_leaks,
    )

    return FinalPostDiagnostics(
        schema_validation_passed=schema_validation_passed,
        schema_validation_error=schema_validation_error,
        missing_quality_check_keys=[],
        non_boolean_quality_check_keys=[],
        evidence_id_leaks=evidence_id_leaks,
        scaffold_phrase_leaks=scaffold_phrase_leaks,
        source_summary_phrase_leaks=source_summary_phrase_leaks,
        model_claimed_linkedin_ready=None,
        deterministic_checks_passed=deterministic_checks_passed,
        system_linkedin_ready=system_linkedin_ready,
        repair_reasons=repair_reasons,
    )


def diagnose_final_post_payload(
    payload: dict,
    *,
    selected_evidence_ids: list[str],
    schema_validation_passed: bool,
    schema_validation_error: str = "",
) -> FinalPostDiagnostics:
    quality_checks = payload.get("quality_checks") if isinstance(payload, dict) else None
    if not isinstance(quality_checks, dict):
        quality_checks = {}

    missing_quality_check_keys = sorted(REQUIRED_QUALITY_CHECK_KEYS - quality_checks.keys())
    non_boolean_quality_check_keys = sorted(
        key
        for key in REQUIRED_QUALITY_CHECK_KEYS
        if key in quality_checks and not isinstance(quality_checks[key], bool)
    )
    model_claimed_linkedin_ready = quality_checks.get("linkedin_ready")
    if not isinstance(model_claimed_linkedin_ready, bool):
        model_claimed_linkedin_ready = None

    evidence_id_leaks = _find_phrase_leaks(
        payload,
        [evidence_id for evidence_id in selected_evidence_ids if evidence_id],
        leak_type="evidence_id",
    )
    scaffold_phrase_leaks = _find_phrase_leaks(
        payload,
        SCAFFOLD_PHRASES,
        leak_type="scaffold_phrase",
    )
    source_summary_phrase_leaks = _find_phrase_leaks(
        payload,
        SOURCE_SUMMARY_PHRASES,
        leak_type="source_summary_phrase",
    )

    deterministic_checks_passed = not (
        missing_quality_check_keys
        or non_boolean_quality_check_keys
        or evidence_id_leaks
        or scaffold_phrase_leaks
        or source_summary_phrase_leaks
    )
    system_linkedin_ready = (
        schema_validation_passed
        and deterministic_checks_passed
        and model_claimed_linkedin_ready is True
    )
    repair_reasons = _build_repair_reasons(
        schema_validation_passed=schema_validation_passed,
        missing_quality_check_keys=missing_quality_check_keys,
        non_boolean_quality_check_keys=non_boolean_quality_check_keys,
        evidence_id_leaks=evidence_id_leaks,
        scaffold_phrase_leaks=scaffold_phrase_leaks,
        source_summary_phrase_leaks=source_summary_phrase_leaks,
        model_claimed_linkedin_ready=model_claimed_linkedin_ready,
    )

    return FinalPostDiagnostics(
        schema_validation_passed=schema_validation_passed,
        schema_validation_error=schema_validation_error,
        missing_quality_check_keys=missing_quality_check_keys,
        non_boolean_quality_check_keys=non_boolean_quality_check_keys,
        evidence_id_leaks=evidence_id_leaks,
        scaffold_phrase_leaks=scaffold_phrase_leaks,
        source_summary_phrase_leaks=source_summary_phrase_leaks,
        model_claimed_linkedin_ready=model_claimed_linkedin_ready,
        deterministic_checks_passed=deterministic_checks_passed,
        system_linkedin_ready=system_linkedin_ready,
        repair_reasons=repair_reasons,
    )


def _find_phrase_leaks(payload: dict, phrases: list[str] | tuple[str, ...], *, leak_type: str) -> list[TextLeak]:
    leaks: list[TextLeak] = []
    for field_name, value_index, text in _human_facing_text_values(payload):
        normalized_text = text.lower()
        for phrase in phrases:
            normalized_phrase = phrase.lower()
            if normalized_phrase and normalized_phrase in normalized_text:
                leaks.append(
                    TextLeak(
                        field_name=field_name,
                        value_index=value_index,
                        phrase=phrase,
                        leak_type=leak_type,
                    )
                )
    return leaks


def _human_facing_text_values(payload: dict) -> list[tuple[str, int | None, str]]:
    values: list[tuple[str, int | None, str]] = []
    if not isinstance(payload, dict):
        return values

    for field_name in HUMAN_FACING_FIELDS:
        field_value = payload.get(field_name)
        if isinstance(field_value, str):
            values.append((field_name, None, field_value))
            continue
        if isinstance(field_value, list):
            for index, item in enumerate(field_value):
                if isinstance(item, str):
                    values.append((field_name, index, item))
    return values


def _build_repair_reasons(
    *,
    schema_validation_passed: bool,
    missing_quality_check_keys: list[str],
    non_boolean_quality_check_keys: list[str],
    evidence_id_leaks: list[TextLeak],
    scaffold_phrase_leaks: list[TextLeak],
    source_summary_phrase_leaks: list[TextLeak],
    model_claimed_linkedin_ready: bool | None,
) -> list[str]:
    reasons: list[str] = []
    if not schema_validation_passed:
        reasons.append("schema_validation_failed")
    if missing_quality_check_keys:
        reasons.append("missing_quality_checks")
    if non_boolean_quality_check_keys:
        reasons.append("non_boolean_quality_checks")
    if evidence_id_leaks:
        reasons.append("evidence_ids_in_human_text")
    if scaffold_phrase_leaks:
        reasons.append("scaffold_language_in_human_text")
    if source_summary_phrase_leaks:
        reasons.append("source_summary_language_in_human_text")
    if model_claimed_linkedin_ready is not True:
        reasons.append("model_not_linkedin_ready")
    return reasons


def _build_candidate_post_repair_reasons(
    *,
    schema_validation_passed: bool,
    evidence_id_leaks: list[TextLeak],
    scaffold_phrase_leaks: list[TextLeak],
    source_summary_phrase_leaks: list[TextLeak],
) -> list[str]:
    reasons: list[str] = []
    if not schema_validation_passed:
        reasons.append("schema_validation_failed")
    if evidence_id_leaks:
        reasons.append("evidence_ids_in_human_text")
    if scaffold_phrase_leaks:
        reasons.append("scaffold_language_in_human_text")
    if source_summary_phrase_leaks:
        reasons.append("source_summary_language_in_human_text")
    return reasons
