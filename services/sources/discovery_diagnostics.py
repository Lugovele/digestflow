from __future__ import annotations

from collections import Counter
from typing import Any

from services.sources.discovery_constants import (
    DISCOVERY_DECISION_MAX_ROUNDS_REACHED,
    DISCOVERY_DECISION_PARTIAL_NO_UNUSED_SURFACES,
    DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR,
    DISCOVERY_DECISION_PARTIAL_TARGET_NOT_REACHED,
    DISCOVERY_DECISION_PROVIDER_UNAVAILABLE,
    DISCOVERY_DECISION_TARGET_REACHED,
)


def build_discovery_cycle_payload(
    *,
    cycle_id: str,
    target_visible_new_suggestions: int,
    max_immediate_rounds: int,
    round_count: int,
    accumulated_visible_suggestions: int,
    decision: str,
    rounds: list[dict],
    cycle_diagnosis: dict,
    repair_plan: dict,
) -> dict:
    return {
        "cycle_id": cycle_id,
        "target_visible_suggestions": int(target_visible_new_suggestions),
        "target_visible_new_suggestions": int(target_visible_new_suggestions),
        "max_immediate_rounds": int(max_immediate_rounds),
        "rounds_run": int(round_count),
        "round_count": int(round_count),
        "accumulated_visible_suggestions": int(accumulated_visible_suggestions),
        "decision": str(decision or "").strip() or DISCOVERY_DECISION_PARTIAL_TARGET_NOT_REACHED,
        "rounds": rounds,
        "cycle_diagnosis": dict(cycle_diagnosis or {}),
        "repair_plan": dict(repair_plan or {}),
    }


def classify_discovery_cycle_round_reason(
    *,
    provider_error_count: int,
    raw_result_count: int,
    visible_new_suggestions_count: int,
    quality_rejected_count: int,
    known_or_duplicate_count: int,
    target_visible_new_suggestions: int,
) -> str:
    if visible_new_suggestions_count >= target_visible_new_suggestions:
        return DISCOVERY_DECISION_TARGET_REACHED
    if provider_error_count > 0 and raw_result_count == 0 and visible_new_suggestions_count == 0:
        return "provider_error"
    if quality_rejected_count > 0 and quality_rejected_count >= max(known_or_duplicate_count, 1):
        return "quality_heavy"
    if known_or_duplicate_count > 0 and known_or_duplicate_count >= max(quality_rejected_count, 1):
        return "duplicate_heavy"
    if visible_new_suggestions_count == 0:
        return "zero_visible"
    return "mixed_low_yield"


def build_discovery_cycle_round_diagnosis(
    *,
    round_result: dict,
    returned_count: int,
    visible_new_suggestions: int,
    target_visible_new_suggestions: int,
) -> dict:
    provider_unavailable = bool(round_result.get("provider_unavailable"))
    provider_error_count = int(round_result.get("provider_error_count") or 0)
    quality_rejected_count = int(round_result.get("quality_rejected_count") or 0)
    known_or_duplicate_count = int(round_result.get("known_or_duplicate_count") or 0)
    run = round_result.get("discovery_run")
    diagnostics = dict(getattr(run, "diagnostics", {}) or {})
    query_rows = diagnostics.get("per_query_result_counts") or []
    quality_feedback = diagnostics.get("source_quality_feedback") if isinstance(diagnostics.get("source_quality_feedback"), dict) else {}

    duplicate_ratio = known_or_duplicate_count / max(returned_count, 1)
    quality_ratio = quality_rejected_count / max(returned_count, 1)
    zero_result_queries = sum(
        1
        for item in query_rows
        if isinstance(item, dict) and int(item.get("result_count") or 0) == 0
    )
    weak_domains_count = sum(
        int(item.get("count") or 0)
        for item in quality_feedback.get("weak_domains") or []
        if isinstance(item, dict)
    )
    weak_material_types = {
        str(item.get("material_type") or "").strip()
        for item in quality_feedback.get("weak_material_types") or []
        if isinstance(item, dict) and str(item.get("material_type") or "").strip()
    }
    stale_rejection_count = sum(
        int(item.get("count") or 0)
        for item in quality_feedback.get("dominant_rejection_reasons") or []
        if isinstance(item, dict) and is_stale_rejection_reason(str(item.get("reason") or "").strip())
    )
    over_broad_signals = {
        "social_profile_forum",
        "price_prediction_live_price",
        "beginner_seo_guide",
        "generic_live_price_page",
        "generic_seo",
    }

    primary_cause = "mixed_low_yield"
    secondary_causes: list[str] = []
    recommended_next_action = "reframe_search_strategy"
    severity = "medium"

    if visible_new_suggestions >= target_visible_new_suggestions:
        primary_cause = "target_reached"
        recommended_next_action = "stop"
        severity = "low"
    elif provider_unavailable:
        primary_cause = "provider_unavailable"
        recommended_next_action = "show_provider_unavailable_message"
        severity = "high"
    elif provider_error_count > 0 and returned_count == 0 and quality_rejected_count == 0 and known_or_duplicate_count == 0:
        primary_cause = "provider_partial_error"
        recommended_next_action = "retry_or_rephrase_failed_queries"
        severity = "high"
    elif returned_count == 0 and zero_result_queries > 0:
        primary_cause = "over_narrow_query" if zero_result_queries >= max(2, len(query_rows) // 2 or 1) else "zero_return"
        recommended_next_action = "broaden_query"
        severity = "high"
    elif stale_rejection_count > 0 and stale_rejection_count >= max(quality_rejected_count // 2, 1):
        primary_cause = "stale_heavy"
        recommended_next_action = "tighten_recency_or_use_current_terms"
        severity = "high" if visible_new_suggestions == 0 else "medium"
    elif known_or_duplicate_count > 0 and (
        duplicate_ratio >= 0.5
        or known_or_duplicate_count >= visible_new_suggestions + quality_rejected_count
    ):
        primary_cause = "duplicate_heavy"
        recommended_next_action = "pivot_to_new_subangles"
        severity = "high" if visible_new_suggestions == 0 else "medium"
    elif quality_rejected_count > 0 and (
        quality_ratio >= 0.5
        or quality_rejected_count >= 2 * max(visible_new_suggestions, 1)
    ):
        if weak_material_types.intersection(over_broad_signals) and returned_count >= max(6, quality_rejected_count):
            primary_cause = "over_broad_query"
            recommended_next_action = "narrow_by_material_type"
        else:
            primary_cause = "quality_heavy"
            recommended_next_action = "pivot_to_stronger_material_types"
        severity = "high" if visible_new_suggestions == 0 else "medium"
    elif weak_domains_count >= 3:
        primary_cause = "domain_repetition"
        recommended_next_action = "diversify_domains"
        severity = "medium"
    elif visible_new_suggestions == 0:
        primary_cause = "zero_return" if returned_count == 0 else "mixed_low_yield"
        recommended_next_action = "broaden_query" if returned_count == 0 else "reframe_search_strategy"
        severity = "high"

    if provider_error_count > 0 and primary_cause != "provider_unavailable":
        secondary_causes.append("provider_partial_error")
    if known_or_duplicate_count > 0 and primary_cause != "duplicate_heavy" and duplicate_ratio >= 0.5:
        secondary_causes.append("duplicate_heavy")
    if quality_rejected_count > 0 and primary_cause not in {"quality_heavy", "over_broad_query"} and (
        quality_ratio >= 0.5 or quality_rejected_count >= 2 * max(visible_new_suggestions, 1)
    ):
        secondary_causes.append("quality_heavy")
    if stale_rejection_count > 0 and primary_cause != "stale_heavy":
        secondary_causes.append("stale_heavy")
    if weak_domains_count >= 3 and primary_cause != "domain_repetition":
        secondary_causes.append("domain_repetition")
    if returned_count == 0 and primary_cause not in {"zero_return", "over_narrow_query", "provider_unavailable"}:
        secondary_causes.append("zero_return")
    secondary_causes = dedupe_string_list(secondary_causes)

    explanation = build_discovery_cycle_diagnosis_explanation(
        primary_cause=primary_cause,
        secondary_causes=secondary_causes,
        provider_error_count=provider_error_count,
        returned_count=returned_count,
        quality_rejected_count=quality_rejected_count,
        known_or_duplicate_count=known_or_duplicate_count,
    )

    return {
        "primary_cause": primary_cause,
        "secondary_causes": secondary_causes,
        "severity": severity,
        "explanation": explanation,
        "recommended_next_action": recommended_next_action,
    }


def build_discovery_cycle_overall_diagnosis(
    *,
    decision: str,
    rounds: list[dict],
    accumulated_visible_suggestions: int,
    target_visible_new_suggestions: int,
) -> dict:
    if decision == DISCOVERY_DECISION_TARGET_REACHED:
        return {
            "primary_cause": DISCOVERY_DECISION_TARGET_REACHED,
            "secondary_causes": [],
            "severity": "low",
            "explanation": (
                f"The discovery cycle reached the {target_visible_new_suggestions}-source target "
                f"with {accumulated_visible_suggestions} visible suggestions."
            ),
            "recommended_next_action": "stop",
        }
    if decision == DISCOVERY_DECISION_PROVIDER_UNAVAILABLE:
        return {
            "primary_cause": DISCOVERY_DECISION_PROVIDER_UNAVAILABLE,
            "secondary_causes": [],
            "severity": "high",
            "explanation": "The search provider was unavailable, so the cycle could not process meaningful provider results.",
            "recommended_next_action": "show_provider_unavailable_message",
        }

    diagnoses = [
        item.get("diagnosis")
        for item in rounds
        if isinstance(item, dict) and isinstance(item.get("diagnosis"), dict)
    ]
    primary_counts = Counter(
        str(item.get("primary_cause") or "").strip()
        for item in diagnoses
        if str(item.get("primary_cause") or "").strip() and str(item.get("primary_cause") or "").strip() != "target_reached"
    )
    secondary_counts = Counter()
    for item in diagnoses:
        for cause in item.get("secondary_causes") or []:
            cleaned = str(cause or "").strip()
            if cleaned:
                secondary_counts[cleaned] += 1

    priority = [
        "duplicate_heavy",
        "quality_heavy",
        "over_broad_query",
        "stale_heavy",
        "domain_repetition",
        "provider_partial_error",
        "over_narrow_query",
        "zero_return",
        "mixed_low_yield",
    ]
    primary_cause = "mixed_low_yield"
    if primary_counts:
        primary_cause = sorted(
            primary_counts.items(),
            key=lambda item: (-item[1], priority.index(item[0]) if item[0] in priority else len(priority), item[0]),
        )[0][0]

    secondary_causes = dedupe_string_list(
        [
            cause
            for cause in priority
            if cause != primary_cause and (secondary_counts.get(cause) or primary_counts.get(cause))
        ]
    )
    recommended_next_action = recommended_next_action_for_diagnosis(primary_cause)
    severity = "high" if accumulated_visible_suggestions == 0 else "medium"
    explanation = build_discovery_cycle_diagnosis_explanation(
        primary_cause=primary_cause,
        secondary_causes=secondary_causes,
        provider_error_count=sum(int(item.get("provider_error_count") or 0) for item in rounds if isinstance(item, dict)),
        returned_count=sum(int(item.get("returned_count") or 0) for item in rounds if isinstance(item, dict)),
        quality_rejected_count=sum(int(item.get("quality_rejected_count") or 0) for item in rounds if isinstance(item, dict)),
        known_or_duplicate_count=sum(int(item.get("known_or_duplicate_count") or 0) for item in rounds if isinstance(item, dict)),
    )
    return {
        "primary_cause": primary_cause,
        "secondary_causes": secondary_causes,
        "severity": severity,
        "explanation": explanation,
        "recommended_next_action": recommended_next_action,
    }


def is_stale_rejection_reason(reason: str) -> bool:
    lowered = str(reason or "").strip().casefold()
    return any(
        needle in lowered
        for needle in (
            "stale",
            "very stale",
            "outside recency",
            "publication year outside recency",
            "freshness",
        )
    )


def recommended_next_action_for_diagnosis(primary_cause: str) -> str:
    mapping = {
        "target_reached": "stop",
        "provider_unavailable": "show_provider_unavailable_message",
        "provider_partial_error": "retry_or_rephrase_failed_queries",
        "zero_return": "broaden_query",
        "over_narrow_query": "broaden_query",
        "duplicate_heavy": "pivot_to_new_subangles",
        "quality_heavy": "pivot_to_stronger_material_types",
        "stale_heavy": "tighten_recency_or_use_current_terms",
        "domain_repetition": "diversify_domains",
        "over_broad_query": "narrow_by_material_type",
        "mixed_low_yield": "reframe_search_strategy",
    }
    return mapping.get(primary_cause, "reframe_search_strategy")


def build_discovery_cycle_diagnosis_explanation(
    *,
    primary_cause: str,
    secondary_causes: list[str],
    provider_error_count: int,
    returned_count: int,
    quality_rejected_count: int,
    known_or_duplicate_count: int,
) -> str:
    primary_label = format_discovery_cycle_diagnosis_label(primary_cause)
    if primary_cause == "target_reached":
        return "The discovery cycle reached the visible-source target."
    if primary_cause == "provider_unavailable":
        return "The search provider was unavailable, so PostFlow could not process meaningful provider data."
    fragments = []
    if known_or_duplicate_count > 0:
        fragments.append(f"{known_or_duplicate_count} returned URL{'s were' if known_or_duplicate_count != 1 else ' was'} already known or duplicate")
    if quality_rejected_count > 0:
        fragments.append(f"{quality_rejected_count} result{'s were' if quality_rejected_count != 1 else ' was'} rejected by quality filters")
    if provider_error_count > 0:
        fragments.append(f"{provider_error_count} provider quer{'ies failed' if provider_error_count != 1 else 'y failed'}")
    if returned_count == 0:
        fragments.append("provider queries returned no meaningful results")
    if not fragments:
        fragments.append("the round stayed below the visible-source target")
    explanation = f"Primary cause: {primary_label}. " + "; ".join(fragments).capitalize() + "."
    if secondary_causes:
        explanation += (
            " Secondary causes: "
            + ", ".join(format_discovery_cycle_diagnosis_label(item).lower() for item in secondary_causes)
            + "."
        )
    return explanation


def dedupe_string_list(items) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items or []:
        cleaned = str(item or "").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return result


def format_discovery_cycle_decision_label(decision: str) -> str:
    mapping = {
        DISCOVERY_DECISION_TARGET_REACHED: "Target reached",
        DISCOVERY_DECISION_PARTIAL_TARGET_NOT_REACHED: "Partial target not reached",
        DISCOVERY_DECISION_PARTIAL_NO_UNUSED_SURFACES: "Partial target not reached - no unused surfaces",
        DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR: "Partial target not reached - no usable repair queries",
        DISCOVERY_DECISION_MAX_ROUNDS_REACHED: "Max rounds reached",
        DISCOVERY_DECISION_PROVIDER_UNAVAILABLE: "Provider unavailable",
    }
    return mapping.get(str(decision or "").strip().lower(), "Discovery cycle update")


def format_discovery_cycle_diagnosis_label(cause: str) -> str:
    mapping = {
        "provider_unavailable": "Provider unavailable",
        "provider_partial_error": "Provider partial errors",
        "zero_return": "Zero-return queries",
        "duplicate_heavy": "Duplicate-heavy results",
        "quality_heavy": "Quality-heavy results",
        "stale_heavy": "Stale-heavy results",
        "domain_repetition": "Domain repetition",
        "over_narrow_query": "Over-narrow query",
        "over_broad_query": "Over-broad query",
        "mixed_low_yield": "Mixed low-yield results",
        "target_reached": "Target reached",
    }
    return mapping.get(str(cause or "").strip().lower(), "Search diagnosis update")


def format_discovery_cycle_next_action_label(action: str) -> str:
    mapping = {
        "stop": "Stop",
        "show_provider_unavailable_message": "Show provider unavailable message",
        "retry_or_rephrase_failed_queries": "Retry or rephrase failed queries",
        "broaden_query": "Broaden query",
        "pivot_to_new_subangles": "Pivot to new sub-angles",
        "pivot_to_stronger_material_types": "Pivot to stronger material types",
        "tighten_recency_or_use_current_terms": "Tighten recency or use current terms",
        "diversify_domains": "Diversify domains",
        "narrow_by_material_type": "Narrow by material type",
        "reframe_search_strategy": "Reframe search strategy",
    }
    return mapping.get(str(action or "").strip().lower(), "Review search strategy")
