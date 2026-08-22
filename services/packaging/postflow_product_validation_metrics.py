"""Metrics for PostFlow product validation benchmarks."""
from __future__ import annotations

from collections import Counter
from typing import Any

from services.packaging.postflow_product_validation_corpus import (
    CASE_FAMILY_CANDIDATE_QE,
    CASE_FAMILY_SEMANTIC_BOUNDARY,
    CASE_FAMILY_SOURCE_ARTICLE,
    FAILURE_CATEGORY_INFRASTRUCTURE,
    FAILURE_CATEGORY_PRODUCT,
    FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
)
from services.packaging.postflow_product_validation_live_execution import (
    LIVE_ACCEPTED_FIRST_ATTEMPT,
    LIVE_BOUNDARY_FALSE_NEGATIVE,
    LIVE_BOUNDARY_FALSE_POSITIVE,
    LIVE_BOUNDARY_INVALID_BLOCKED,
    LIVE_BOUNDARY_VALID_PRESERVED,
    LIVE_DETERMINISTIC_BLOCK,
    LIVE_GROUNDING_BLOCK,
    LIVE_GROUNDING_INFRA_FAILURE,
    LIVE_OVER_LENGTH,
    LIVE_QE_INFRA_FAILURE,
    LIVE_QE_REJECTED,
    LIVE_REPAIR_ACCEPTED,
    LIVE_REPAIR_EXCESSIVE_REWRITE,
    LIVE_REPAIR_REQUIRED_NOT_EXECUTED,
    LIVE_REPAIR_REGRESSION,
    LIVE_REPAIR_TARGET_NOT_FIXED,
    LIVE_SOURCE_READY,
    PRODUCT_CORPUS_MINIMUM_CASES,
    PRODUCT_CORPUS_PREFERRED_CASES,
    SYSTEMIC_FAILURE_MIN_CASES,
)


def classify_product_validation_record(record: dict[str, Any]) -> str:
    return str(record.get("expected_outcome") or "historical_inconclusive")


def compute_product_validation_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    family_counts = Counter(str(record.get("family", "")) for record in records)
    outcome_counts = Counter(classify_product_validation_record(record) for record in records)
    failure_category_counts = Counter(
        str(record.get("failure_category", "")) for record in records
    )
    topic_counts = Counter(str(record.get("topic", "")) for record in records)
    risk_type_counts = Counter(str(record.get("risk_type", "")) for record in records)
    human_review_count = sum(1 for record in records if record.get("human_review_required"))
    provider_counts = _provider_counts(records)
    product_records = tuple(
        record for record in records if record.get("family") == CASE_FAMILY_CANDIDATE_QE
    )
    boundary_records = tuple(
        record for record in records if record.get("family") == CASE_FAMILY_SEMANTIC_BOUNDARY
    )
    source_records = tuple(
        record for record in records if record.get("family") == CASE_FAMILY_SOURCE_ARTICLE
    )
    live_outcome_counts = Counter(
        str(record.get("live_outcome") or "not_executed") for record in records
    )
    live_failure_category_counts = Counter(
        str(record.get("live_failure_category") or "none") for record in records
    )
    return {
        "case_count": len(records),
        "provider_call_count": sum(provider_counts.values()),
        "provider_invocation_counts": provider_counts,
        "family_counts": dict(sorted(family_counts.items())),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "failure_category_counts": dict(sorted(failure_category_counts.items())),
        "live_outcome_counts": dict(sorted(live_outcome_counts.items())),
        "live_failure_category_counts": dict(sorted(live_failure_category_counts.items())),
        "topic_counts": dict(sorted(topic_counts.items())),
        "risk_type_counts": dict(sorted(risk_type_counts.items())),
        "human_review_case_count": human_review_count,
        "product_failure_case_count": failure_category_counts.get(FAILURE_CATEGORY_PRODUCT, 0),
        "infrastructure_failure_case_count": failure_category_counts.get(
            FAILURE_CATEGORY_INFRASTRUCTURE, 0
        ),
        "product_behavior_case_count": failure_category_counts.get(
            FAILURE_CATEGORY_PRODUCT_BEHAVIOR, 0
        ),
        "product_metrics": _product_metrics(product_records),
        "boundary_metrics": _boundary_metrics(boundary_records),
        "source_metrics": _source_metrics(source_records),
        "critical_invariants": _critical_invariants(records),
        "human_ground_truth_metrics": _human_ground_truth_metrics(product_records),
        "product_case_distribution": _product_case_distribution(product_records),
        "repeated_failure_signatures": _repeated_failure_signatures(product_records),
    }


def compare_product_validation_metrics(
    current_metrics: dict[str, Any], baseline_metrics: dict[str, Any] | None
) -> dict[str, Any] | None:
    if baseline_metrics is None:
        return None
    comparable_keys = (
        "case_count",
        "human_review_case_count",
        "product_failure_case_count",
        "infrastructure_failure_case_count",
        "product_behavior_case_count",
        "provider_call_count",
    )
    return {
        key: int(current_metrics.get(key, 0)) - int(baseline_metrics.get(key, 0))
        for key in comparable_keys
    }


def _provider_counts(records: tuple[dict[str, Any], ...]) -> dict[str, int]:
    result = Counter()
    for record in records:
        counts = record.get("provider_invocation_counts")
        if not isinstance(counts, dict):
            continue
        for key in (
            "candidate_writer_provider_api_calls",
            "semantic_grounding_provider_api_calls",
            "quality_evaluator_provider_api_calls",
            "repair_writer_provider_api_calls",
            "publication_packaging_invocations",
        ):
            result[key] += int(counts.get(key, 0) or 0)
    return dict(sorted(result.items()))


def _product_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    live_records = [
        record for record in records
        if record.get("live_execution_status") != "dry_run" and record.get("live_outcome")
    ]
    evaluable = [
        record for record in live_records
        if record.get("live_failure_category") != FAILURE_CATEGORY_INFRASTRUCTURE
    ]
    accepted = _count(tuple(live_records), LIVE_ACCEPTED_FIRST_ATTEMPT)
    repair_accepted = _count(tuple(live_records), LIVE_REPAIR_ACCEPTED)
    repair_attempts = sum(1 for record in live_records if _repair_executed(record))
    repair_required_not_executed = _count(tuple(live_records), LIVE_REPAIR_REQUIRED_NOT_EXECUTED)
    target_repair_successes = sum(
        1 for record in live_records
        if _repair_executed(record) and record.get("target_repair_success") is True
    )
    quality_reached = sum(
        1 for record in live_records
        if "quality_evaluation" in (record.get("live_stage_outcomes") or {})
    )
    return {
        "product_case_count": len(records),
        "product_corpus_adequacy_status": _product_corpus_adequacy_status(len(records)),
        "minimum_product_case_count": PRODUCT_CORPUS_MINIMUM_CASES,
        "preferred_product_case_count": PRODUCT_CORPUS_PREFERRED_CASES,
        "gap_to_minimum_product_cases": max(PRODUCT_CORPUS_MINIMUM_CASES - len(records), 0),
        "gap_to_preferred_product_cases": max(PRODUCT_CORPUS_PREFERRED_CASES - len(records), 0),
        "product_rates_are_exploratory": len(records) < PRODUCT_CORPUS_PREFERRED_CASES,
        "live_product_case_count": len(live_records),
        "live_product_not_measured_count": len(records) - len(live_records),
        "product_evaluable_case_count": len(evaluable),
        "product_infrastructure_failure_count": len(live_records) - len(evaluable),
        "first_attempt_acceptance_count": accepted,
        "first_attempt_acceptance_rate": _rate(accepted, len(evaluable)),
        "repair_required_not_executed_count": repair_required_not_executed,
        "repair_required_not_executed_rate": _rate(
            repair_required_not_executed,
            len(evaluable),
        ),
        "repair_attempt_count": repair_attempts,
        "repair_attempt_rate": _rate(repair_attempts, len(evaluable)),
        "repair_accepted_count": repair_accepted,
        "repair_accepted_rate": _rate(repair_accepted, repair_attempts),
        "evaluable_final_acceptance_count": accepted + repair_accepted,
        "evaluable_final_acceptance_rate": _rate(accepted + repair_accepted, len(evaluable)),
        "final_not_ready_count": _count(tuple(live_records), LIVE_QE_REJECTED),
        "final_not_ready_rate": _rate(_count(tuple(live_records), LIVE_QE_REJECTED), len(evaluable)),
        "target_repair_success_count": target_repair_successes,
        "target_repair_success_rate": _rate(target_repair_successes, repair_attempts),
        "TARGET_FIXED_CLEAN_count": repair_accepted,
        "TARGET_FIXED_CLEAN_rate": _rate(repair_accepted, repair_attempts),
        "target_not_fixed_count": _count(tuple(live_records), LIVE_REPAIR_TARGET_NOT_FIXED),
        "target_not_fixed_rate": _rate(
            _count(tuple(live_records), LIVE_REPAIR_TARGET_NOT_FIXED),
            repair_attempts,
        ),
        "repair_regression_count": _count(tuple(live_records), LIVE_REPAIR_REGRESSION),
        "broad_rewrite_count": _count(tuple(live_records), LIVE_REPAIR_EXCESSIVE_REWRITE),
        "excessive_rewrite_count": _count(tuple(live_records), LIVE_REPAIR_EXCESSIVE_REWRITE),
        "generic_marker_introduction_count": _generic_marker_count(tuple(live_records)),
        "within_1300_count": sum(1 for record in records if _candidate_length(record) <= 1300),
        "within_1300_rate": _rate(
            sum(1 for record in records if _candidate_length(record) <= 1300),
            len(records),
        ),
        "qe_reached_count": quality_reached,
        "qe_reached_rate": _rate(quality_reached, len(live_records)),
        "qe_pass_count": accepted + repair_accepted,
        "qe_pass_rate": _rate(accepted + repair_accepted, quality_reached),
        "failed_criterion_frequencies": _failed_criterion_frequencies(tuple(live_records)),
        "automatic_fail_frequency": sum(
            1 for record in live_records
            if (record.get("quality_review_summary") or {}).get("automatic_fail_reason")
        ),
        "human_review_frequency": sum(
            1 for record in live_records if record.get("live_outcome") == "LIVE_HUMAN_REVIEW"
        ),
        "median_distinctive_preservation": None,
        "p25_distinctive_preservation": None,
        "median_character_delta": None,
        "changed_sentence_distribution": {},
    }


def _boundary_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    evaluable = [
        record for record in records
        if record.get("live_failure_category") != FAILURE_CATEGORY_INFRASTRUCTURE
    ]
    correct = _count(records, LIVE_BOUNDARY_VALID_PRESERVED) + _count(
        records, LIVE_BOUNDARY_INVALID_BLOCKED
    )
    return {
        "boundary_case_count": len(records),
        "valid_preserved": _count(records, LIVE_BOUNDARY_VALID_PRESERVED),
        "invalid_blocked": _count(records, LIVE_BOUNDARY_INVALID_BLOCKED),
        "false_positive": _count(records, LIVE_BOUNDARY_FALSE_POSITIVE),
        "false_negative": _count(records, LIVE_BOUNDARY_FALSE_NEGATIVE),
        "infrastructure_failure": _count(records, LIVE_GROUNDING_INFRA_FAILURE),
        "accuracy_on_evaluable_cases": _rate(correct, len(evaluable)),
    }


def _source_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    ready = sum(
        1 for record in records
        if record.get("live_outcome") == LIVE_SOURCE_READY
        or (record.get("live_execution_status") == "dry_run" and record.get("fixture_valid"))
    )
    return {
        "source_case_count": len(records),
        "source_valid": ready,
        "source_invalid": len(records) - ready,
        "reconstruction_ready": ready,
        "reconstruction_failure": len(records) - ready,
    }


def _critical_invariants(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    accepted = {
        LIVE_ACCEPTED_FIRST_ATTEMPT,
        LIVE_REPAIR_ACCEPTED,
    }
    invariants: dict[str, Any] = {
        "accepted_with_target_fixed_false": sum(
            1 for record in records
            if record.get("live_outcome") in accepted
            and record.get("target_repair_success") is False
        ),
        "accepted_with_grounding_block": sum(
            1 for record in records
            if record.get("live_outcome") in accepted
            and record.get("live_failure_stage") == "semantic_grounding"
        ),
        "accepted_after_deterministic_gate_failure": sum(
            1 for record in records
            if record.get("live_outcome") in accepted
            and record.get("live_failure_stage") == "deterministic_gate"
        ),
        "accepted_with_over_length_candidate": sum(
            1 for record in records
            if record.get("live_outcome") in accepted and _candidate_length(record) > 1300
        ),
        "accepted_with_material_genericization": sum(
            1 for record in records
            if record.get("live_outcome") in accepted
            and record.get("material_genericization") is True
        ),
        "accepted_with_missing_required_author_ownership": sum(
            1 for record in records
            if record.get("live_outcome") in accepted
            and record.get("missing_required_author_ownership") is True
        ),
    }
    not_measured = []
    for field_name, invariant_name in (
        ("target_repair_success", "accepted_with_target_fixed_false"),
        ("material_genericization", "accepted_with_material_genericization"),
        (
            "missing_required_author_ownership",
            "accepted_with_missing_required_author_ownership",
        ),
    ):
        if not any(field_name in record for record in records):
            not_measured.append(invariant_name)
    invariants["not_measured_invariants"] = not_measured
    return invariants


def _human_ground_truth_metrics(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    reviewed = [record for record in records if isinstance(record.get("human_ground_truth"), dict)]
    matrix: Counter[str] = Counter()
    for record in reviewed:
        human = (record.get("human_ground_truth") or {}).get("final_disposition", "")
        runtime = _runtime_disposition(record)
        matrix[f"runtime_{runtime.lower()}_vs_human_{str(human).lower()}"] += 1
    unsafe_accept_count = sum(
        1
        for record in reviewed
        if _runtime_disposition(record) == "ACCEPT"
        and (record.get("human_ground_truth") or {}).get("final_disposition") == "NOT_READY"
    )
    unnecessary_repair_count = sum(
        1
        for record in reviewed
        if _runtime_disposition(record) == "REPAIR"
        and (record.get("human_ground_truth") or {}).get("final_disposition") == "ACCEPT"
    )
    missed_repair_count = sum(
        1
        for record in reviewed
        if _runtime_disposition(record) in {"ACCEPT", "NOT_READY"}
        and (record.get("human_ground_truth") or {}).get("final_disposition") == "REPAIR"
    )
    human_genericization_block_count = sum(
        1
        for record in reviewed
        if (record.get("human_ground_truth") or {}).get("genericization") == "MATERIAL"
    )
    runtime_accepted_material_genericization_count = sum(
        1
        for record in reviewed
        if _runtime_disposition(record) == "ACCEPT"
        and (record.get("human_ground_truth") or {}).get("genericization") == "MATERIAL"
    )
    repair_introduced_material_genericization_count = sum(
        1
        for record in reviewed
        if _repair_executed(record) and record.get("material_genericization") is True
    )
    return {
        "human_ground_truth_case_count": len(reviewed),
        "human_ground_truth_coverage_rate": _rate(len(reviewed), len(records)),
        **{key: matrix.get(key, 0) for key in _human_runtime_matrix_keys()},
        "unsafe_accept_count": unsafe_accept_count,
        "unsafe_accept_rate": _rate(unsafe_accept_count, len(reviewed)),
        "unnecessary_repair_count": unnecessary_repair_count,
        "missed_repair_count": missed_repair_count,
        "human_material_genericization_count": human_genericization_block_count,
        "human_genericization_block_count": human_genericization_block_count,
        "runtime_accepted_material_genericization_count": runtime_accepted_material_genericization_count,
        "accepted_human_genericization_block_count": runtime_accepted_material_genericization_count,
        "repair_introduced_material_genericization_count": repair_introduced_material_genericization_count,
        "grounding_disagreement_count": _grounding_disagreement_count(tuple(reviewed)),
    }


def _human_runtime_matrix_keys() -> tuple[str, ...]:
    runtime = ("accept", "repair", "not_ready", "not_measured")
    human = ("accept", "repair", "not_ready")
    return tuple(f"runtime_{r}_vs_human_{h}" for r in runtime for h in human)


def _runtime_disposition(record: dict[str, Any]) -> str:
    outcome = record.get("live_outcome")
    if not outcome:
        return "NOT_MEASURED"
    if outcome in {LIVE_ACCEPTED_FIRST_ATTEMPT, LIVE_REPAIR_ACCEPTED}:
        return "ACCEPT"
    if outcome in {
        LIVE_REPAIR_REQUIRED_NOT_EXECUTED,
        LIVE_REPAIR_TARGET_NOT_FIXED,
        LIVE_REPAIR_REGRESSION,
        LIVE_REPAIR_EXCESSIVE_REWRITE,
    }:
        return "REPAIR"
    return "NOT_READY"


def _grounding_disagreement_count(records: tuple[dict[str, Any], ...]) -> int:
    count = 0
    for record in records:
        truth = record.get("human_ground_truth") or {}
        human_grounding = truth.get("grounding_fidelity")
        live_stage = record.get("live_stage_outcomes") or {}
        runtime_grounding = live_stage.get("semantic_grounding")
        if not isinstance(runtime_grounding, dict) or human_grounding not in {"PASS", "FAIL"}:
            continue
        runtime_pass = runtime_grounding.get("status") == "pass" or (
            isinstance(runtime_grounding.get("grounding_review"), dict)
            and runtime_grounding["grounding_review"].get("pass") is True
        )
        if (human_grounding == "PASS") != runtime_pass:
            count += 1
    return count


def _product_case_distribution(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    return {
        "topic": dict(sorted(Counter(str(record.get("topic", "")) for record in records).items())),
        "candidate_provider_model": dict(sorted(Counter(_provider_model(record) for record in records).items())),
        "human_disposition": dict(sorted(Counter(str((record.get("human_ground_truth") or {}).get("final_disposition", "")) for record in records).items())),
        "repair_target": dict(sorted(Counter(str((record.get("human_ground_truth") or {}).get("repair_target", "")) for record in records).items())),
        "genericization": dict(sorted(Counter(str((record.get("human_ground_truth") or {}).get("genericization", "")) for record in records).items())),
        "grounding": dict(sorted(Counter(str((record.get("human_ground_truth") or {}).get("grounding_fidelity", "")) for record in records).items())),
        "length_bucket": dict(sorted(Counter(str((record.get("product_case_distribution") or {}).get("length_band", "")) for record in records).items())),
        "independence_group_count": len({str(record.get("independence_group", "")) for record in records if record.get("independence_group")}),
        "duplicate_candidate_hash_count": _duplicate_candidate_hash_count(records),
        "raw_product_case_count": len(records),
        "rates_should_be_read_with_independence_groups": True,
    }


def _provider_model(record: dict[str, Any]) -> str:
    provider = record.get("candidate_provider") or (record.get("product_case_distribution") or {}).get("writer_provider") or "unknown"
    model = record.get("candidate_model") or (record.get("product_case_distribution") or {}).get("writer_model") or "unknown"
    return f"{provider}/{model}"


def _duplicate_candidate_hash_count(records: tuple[dict[str, Any], ...]) -> int:
    hashes = Counter()
    for record in records:
        distribution = record.get("product_case_distribution") or {}
        text_hash = distribution.get("candidate_text_hash")
        if text_hash:
            hashes[str(text_hash)] += 1
            continue
        candidate = record.get("final_candidate") or {}
        text = candidate.get("post_text", "") if isinstance(candidate, dict) else ""
        if text:
            hashes[text.strip().lower()] += 1
    return sum(count - 1 for count in hashes.values() if count > 1)


def _product_corpus_adequacy_status(product_count: int) -> str:
    if product_count < PRODUCT_CORPUS_MINIMUM_CASES:
        return "PRODUCT_CORPUS_INSUFFICIENT"
    if product_count < PRODUCT_CORPUS_PREFERRED_CASES:
        return "PRODUCT_CORPUS_MINIMUM_MET"
    return "PRODUCT_CORPUS_PREFERRED_MET"


def _repeated_failure_signatures(records: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    groups_by_signature: dict[str, set[str]] = {}
    raw_counts = Counter()
    for record in records:
        signature = _failure_signature(record)
        if not signature:
            continue
        raw_counts[signature] += 1
        group = str(record.get("independence_group") or record.get("case_id") or "")
        groups_by_signature.setdefault(signature, set()).add(group)
    independent_counts = {
        signature: len(groups) for signature, groups in sorted(groups_by_signature.items())
    }
    systemic = {
        signature: count
        for signature, count in independent_counts.items()
        if count >= SYSTEMIC_FAILURE_MIN_CASES
    }
    return {
        "systemic_threshold": SYSTEMIC_FAILURE_MIN_CASES,
        "signature_counts": dict(sorted(raw_counts.items())),
        "independent_group_counts": independent_counts,
        "candidate_systemic_issues": systemic,
    }


def _repair_executed(record: dict[str, Any]) -> bool:
    repair_stage = (record.get("live_stage_outcomes") or {}).get("repair") or {}
    if isinstance(repair_stage, dict) and repair_stage.get("repair_executed") is True:
        return True
    counts = record.get("provider_invocation_counts") or {}
    return int(counts.get("repair_writer_provider_api_calls", 0) or 0) > 0


def _live_repair_target(record: dict[str, Any]) -> str:
    diagnostics = record.get("repair_target_enforcement") or {}
    target = diagnostics.get("initiating_failed_criterion")
    return str(target or "")

def _failure_signature(record: dict[str, Any]) -> str:
    outcome = record.get("live_outcome")
    if not outcome:
        return ""
    if outcome == LIVE_REPAIR_REQUIRED_NOT_EXECUTED:
        target = (record.get("product_case_distribution") or {}).get("repair_target")
        return f"repair_required_not_executed:{target or 'unknown'}"
    if outcome == LIVE_REPAIR_TARGET_NOT_FIXED:
        target = _live_repair_target(record)
        return f"repair_target_not_fixed:{target or 'unknown'}"
    if outcome in {LIVE_DETERMINISTIC_BLOCK, LIVE_OVER_LENGTH, LIVE_GROUNDING_BLOCK, LIVE_QE_REJECTED}:
        return str(outcome).removeprefix("LIVE_").lower()
    if record.get("live_failure_category") == FAILURE_CATEGORY_INFRASTRUCTURE:
        return f"provider_execution_failure:{record.get('live_failure_stage') or 'unknown'}"
    return ""


def _failed_criterion_frequencies(records: tuple[dict[str, Any], ...]) -> dict[str, int]:
    counter = Counter()
    for record in records:
        summary = record.get("quality_review_summary") or {}
        for criterion in summary.get("failed_criteria", []) or []:
            counter[str(criterion)] += 1
    return dict(sorted(counter.items()))


def _candidate_length(record: dict[str, Any]) -> int:
    distribution = record.get("product_case_distribution") or {}
    value = distribution.get("length")
    return int(value or 0)


def _generic_marker_count(records: tuple[dict[str, Any], ...]) -> int:
    return sum(1 for record in records if record.get("generic_marker_introduced"))


def _count(records: tuple[dict[str, Any], ...], outcome: str) -> int:
    return sum(1 for record in records if record.get("live_outcome") == outcome)


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)
