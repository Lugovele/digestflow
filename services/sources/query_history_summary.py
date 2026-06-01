"""Compact history summaries for history-aware query planning."""

from __future__ import annotations

from collections import Counter
from typing import Any
import re

from django.utils import timezone


MAX_SUMMARY_RUNS = 5
MAX_SUMMARY_QUERIES_PER_BUCKET = 3
MAX_SUMMARY_ANGLES_PER_BUCKET = 3
MAX_SUMMARY_GUIDANCE_ITEMS = 8


def build_query_history_summary(topic) -> dict[str, Any]:
    runs = _get_recent_source_discovery_runs(topic)
    if not runs:
        return _empty_summary()

    useful_queries: list[dict[str, Any]] = []
    weak_queries: list[dict[str, Any]] = []
    duplicate_heavy_queries: list[dict[str, Any]] = []
    provider_error_queries: list[dict[str, Any]] = []
    quality_rejected_queries: list[dict[str, Any]] = []
    useful_angles: Counter[str] = Counter()
    weak_angles: Counter[str] = Counter()
    provider_error_angles: Counter[str] = Counter()
    malformed_run_count = 0
    total_query_rows = 0
    stale_year_counter: Counter[str] = Counter()
    weak_material_types: Counter[str] = Counter()
    preferred_material_types: Counter[str] = Counter()
    weak_domains: Counter[str] = Counter()
    dominant_rejection_reasons: Counter[str] = Counter()
    quality_guidance: list[str] = []

    for run in runs:
        diagnostics = getattr(run, "diagnostics", {}) or {}
        if not isinstance(diagnostics, dict):
            malformed_run_count += 1
            continue
        _merge_quality_feedback_summary(
            diagnostics.get("source_quality_feedback"),
            weak_material_types=weak_material_types,
            preferred_material_types=preferred_material_types,
            weak_domains=weak_domains,
            dominant_rejection_reasons=dominant_rejection_reasons,
            quality_guidance=quality_guidance,
        )
        query_rows = diagnostics.get("query_performance")
        if not isinstance(query_rows, list):
            continue

        for item in query_rows:
            if not isinstance(item, dict):
                continue
            row = _normalize_query_history_row(item)
            if row is None:
                continue
            total_query_rows += 1
            status = row["status"]
            angle = row["angle"]
            for year in _extract_stale_years(row["query"]):
                stale_year_counter[year] += 1
            if status == "useful":
                useful_queries.append(row)
                if angle:
                    useful_angles[angle] += 1
            elif status == "duplicate_heavy":
                duplicate_heavy_queries.append(row)
                if angle:
                    weak_angles[angle] += 1
            elif status == "partial_error":
                provider_error_queries.append(row)
                if angle:
                    provider_error_angles[angle] += 1
            elif status in {"weak", "no_visible_results"}:
                weak_queries.append(row)
                if angle:
                    weak_angles[angle] += 1
                if int(row.get("rejected_count") or 0) > 0 and int(row.get("accepted_count") or 0) == 0:
                    quality_rejected_queries.append(row)

    summary = {
        "history_available": total_query_rows > 0,
        "recent_run_count": len(runs),
        "malformed_run_count": malformed_run_count,
        "total_query_rows": total_query_rows,
        "useful_queries": _dedupe_and_limit_rows(useful_queries),
        "weak_queries": _dedupe_and_limit_rows(weak_queries),
        "duplicate_heavy_queries": _dedupe_and_limit_rows(duplicate_heavy_queries),
        "provider_error_queries": _dedupe_and_limit_rows(provider_error_queries),
        "quality_rejected_queries": _dedupe_and_limit_rows(quality_rejected_queries),
        "useful_angles": _limit_counter(useful_angles),
        "weak_angles": _limit_counter(weak_angles),
        "provider_error_angles": _limit_counter(provider_error_angles),
        "stale_year_patterns": _limit_counter(stale_year_counter, label_key="pattern"),
        "weak_material_types": _limit_counter(weak_material_types, label_key="material_type"),
        "preferred_material_types_found": _limit_counter(preferred_material_types, label_key="material_type"),
        "weak_domains": _limit_counter(weak_domains, label_key="domain"),
        "dominant_rejection_reasons": _limit_counter(dominant_rejection_reasons, label_key="reason"),
        "quality_guidance": quality_guidance[:MAX_SUMMARY_GUIDANCE_ITEMS],
        "planning_guidance": [],
    }
    summary["planning_guidance"] = _build_planning_guidance(summary)
    return summary


def render_query_history_summary_for_prompt(summary: dict[str, Any] | None) -> str:
    normalized = summary if isinstance(summary, dict) else _empty_summary()
    if not normalized.get("history_available"):
        return "No prior query performance history is available for this topic."

    lines = [
        "Recent query performance summary:",
        f"- Recent discovery runs considered: {int(normalized.get('recent_run_count') or 0)}",
        f"- Query rows summarized: {int(normalized.get('total_query_rows') or 0)}",
    ]
    lines.extend(_render_summary_bucket("Useful queries", normalized.get("useful_queries")))
    lines.extend(_render_summary_bucket("Weak or no-result queries", normalized.get("weak_queries")))
    lines.extend(_render_summary_bucket("Duplicate-heavy queries", normalized.get("duplicate_heavy_queries")))
    lines.extend(_render_summary_bucket("Provider/API error queries", normalized.get("provider_error_queries")))
    lines.extend(_render_summary_bucket("Low-quality or rejected-result queries", normalized.get("quality_rejected_queries")))
    lines.extend(_render_angle_bucket("Useful angles", normalized.get("useful_angles")))
    lines.extend(_render_angle_bucket("Weak or exhausted angles", normalized.get("weak_angles")))
    lines.extend(_render_angle_bucket("Provider-error angles", normalized.get("provider_error_angles")))
    lines.extend(_render_pattern_bucket("Stale year patterns", normalized.get("stale_year_patterns")))
    lines.extend(_render_pattern_bucket("Weak material types", normalized.get("weak_material_types"), label_key="material_type"))
    lines.extend(_render_pattern_bucket("Preferred material types found", normalized.get("preferred_material_types_found"), label_key="material_type"))
    lines.extend(_render_pattern_bucket("Weak domains", normalized.get("weak_domains"), label_key="domain"))
    lines.extend(_render_pattern_bucket("Dominant rejection reasons", normalized.get("dominant_rejection_reasons"), label_key="reason"))
    quality_guidance = normalized.get("quality_guidance") or []
    if quality_guidance:
        lines.append("- Quality guidance:")
        for item in quality_guidance[:MAX_SUMMARY_GUIDANCE_ITEMS]:
            lines.append(f"  - {str(item).strip()}")
    guidance = normalized.get("planning_guidance") or []
    if guidance:
        lines.append("- Planning guidance:")
        for item in guidance[:MAX_SUMMARY_GUIDANCE_ITEMS]:
            lines.append(f"  - {str(item).strip()}")
    return "\n".join(lines)


def _get_recent_source_discovery_runs(topic) -> list[Any]:
    runs = getattr(topic, "source_discovery_runs", None)
    if runs is None:
        return []
    try:
        return list(runs.exclude(status="started").order_by("-created_at", "-id")[:MAX_SUMMARY_RUNS])
    except Exception:
        return []


def _empty_summary() -> dict[str, Any]:
    return {
        "history_available": False,
        "recent_run_count": 0,
        "malformed_run_count": 0,
        "total_query_rows": 0,
        "useful_queries": [],
        "weak_queries": [],
        "duplicate_heavy_queries": [],
        "provider_error_queries": [],
        "quality_rejected_queries": [],
        "useful_angles": [],
        "weak_angles": [],
        "provider_error_angles": [],
        "stale_year_patterns": [],
        "weak_material_types": [],
        "preferred_material_types_found": [],
        "weak_domains": [],
        "dominant_rejection_reasons": [],
        "quality_guidance": [],
        "planning_guidance": [],
    }


def _normalize_query_history_row(item: dict[str, Any]) -> dict[str, Any] | None:
    query = str(item.get("query") or "").strip()
    if not query:
        return None
    status = str(item.get("status") or "").strip() or "no_visible_results"
    angle = str(item.get("angle") or "").strip()
    purpose = str(item.get("purpose") or "").strip()
    return {
        "query": query,
        "angle": angle,
        "purpose": purpose,
        "status": status,
        "returned_count": int(item.get("returned_count") or 0),
        "accepted_count": int(item.get("accepted_count") or 0),
        "visible_new_suggestions_count": int(item.get("visible_new_suggestions_count") or 0),
        "duplicate_count": int(item.get("duplicate_count") or 0),
        "rejected_count": int(item.get("rejected_count") or 0),
        "error_message": str(item.get("error_message") or "").strip(),
    }


def _dedupe_and_limit_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    best_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = str(row.get("query") or "").casefold()
        if not key:
            continue
        previous = best_rows.get(key)
        if previous is None or _row_score(row) > _row_score(previous):
            best_rows[key] = row
    ordered_rows = sorted(
        best_rows.values(),
        key=lambda row: (
            -_row_score(row)[0],
            -_row_score(row)[1],
            -_row_score(row)[2],
            -_row_score(row)[3],
            row["query"].casefold(),
        ),
    )
    return ordered_rows[:MAX_SUMMARY_QUERIES_PER_BUCKET]


def _row_score(row: dict[str, Any]) -> tuple[int, int, int, int]:
    return (
        int(row.get("visible_new_suggestions_count") or 0),
        int(row.get("accepted_count") or 0),
        int(row.get("duplicate_count") or 0),
        int(row.get("returned_count") or 0),
    )


def _limit_counter(counter: Counter[str], *, label_key: str = "angle") -> list[dict[str, Any]]:
    rows = [
        {label_key: value, "count": count}
        for value, count in counter.most_common(MAX_SUMMARY_ANGLES_PER_BUCKET)
        if str(value).strip()
    ]
    return rows


def _build_planning_guidance(summary: dict[str, Any]) -> list[str]:
    guidance: list[str] = []
    useful_queries = summary.get("useful_queries") or []
    weak_queries = summary.get("weak_queries") or []
    duplicate_heavy_queries = summary.get("duplicate_heavy_queries") or []
    provider_error_queries = summary.get("provider_error_queries") or []
    quality_rejected_queries = summary.get("quality_rejected_queries") or []
    useful_angles = summary.get("useful_angles") or []
    weak_angles = summary.get("weak_angles") or []
    stale_year_patterns = summary.get("stale_year_patterns") or []
    weak_material_type_rows = summary.get("weak_material_types") or []
    preferred_material_type_rows = summary.get("preferred_material_types_found") or []
    weak_domain_rows = summary.get("weak_domains") or []
    rejection_reason_rows = summary.get("dominant_rejection_reasons") or []
    quality_guidance = summary.get("quality_guidance") or []

    if useful_queries:
        useful_angle_names = ", ".join(str(item.get("angle") or "").strip() for item in useful_angles if str(item.get("angle") or "").strip())
        if useful_angle_names:
            guidance.append(f"Useful directions so far: {useful_angle_names}. Create fresh variants around those angles instead of reusing the same wording.")
        else:
            guidance.append("Useful queries exist. Reframe their strongest patterns into fresh variants instead of repeating them word-for-word.")
    if weak_queries:
        weak_angle_names = ", ".join(str(item.get("angle") or "").strip() for item in weak_angles if str(item.get("angle") or "").strip())
        if weak_angle_names:
            guidance.append(f"Weak or no-result directions include: {weak_angle_names}. Reframe them substantially or replace them with fresher angles.")
        else:
            guidance.append("Some queries returned no useful results. Avoid repeating them unless the framing changes materially.")
    if duplicate_heavy_queries:
        guidance.append("Duplicate-heavy directions look exhausted. Pivot to a different material type, narrower subtopic, or fresher angle.")
    if provider_error_queries:
        guidance.append("Provider or API failures affected some queries. Do not treat those rows as proof that the angle is weak until they run cleanly.")
    if stale_year_patterns:
        stale_year_labels = ", ".join(str(item.get("pattern") or "").strip() for item in stale_year_patterns if str(item.get("pattern") or "").strip())
        guidance.append(f"Avoid stale explicit years in fresh searches ({stale_year_labels}). Prefer latest, current, recent, or this month phrasing.")
    for item in quality_guidance:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in guidance:
            guidance.append(cleaned)
        if len(guidance) >= MAX_SUMMARY_GUIDANCE_ITEMS:
            break
    if weak_material_type_rows and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
        labels = ", ".join(str(item.get("material_type") or "").strip() for item in weak_material_type_rows if str(item.get("material_type") or "").strip())
        guidance.append(f"Recent low-quality material types include: {labels}. Adjust queries to avoid attracting those patterns.")
    if preferred_material_type_rows and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
        labels = ", ".join(str(item.get("material_type") or "").strip() for item in preferred_material_type_rows if str(item.get("material_type") or "").strip())
        guidance.append(f"Preferred material types already found include: {labels}. Lean harder into those patterns in the next run.")
    if weak_domain_rows and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
        labels = ", ".join(str(item.get("domain") or "").strip() for item in weak_domain_rows if str(item.get("domain") or "").strip())
        guidance.append(f"Low-quality domains keep appearing: {labels}. Avoid broad phrasing that attracts forum, social, or profile pages.")
    if quality_rejected_queries and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
        guidance.append("Some queries returned results but mostly low-quality or off-topic candidates. Tighten source intent and specificity before retrying those directions.")
    if rejection_reason_rows and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
        labels = ", ".join(str(item.get("reason") or "").strip() for item in rejection_reason_rows if str(item.get("reason") or "").strip())
        guidance.append(f"Dominant rejection reasons include: {labels}. Use that feedback to steer toward higher-substance source types.")

    return guidance[:MAX_SUMMARY_GUIDANCE_ITEMS]


def _render_summary_bucket(label: str, rows: Any) -> list[str]:
    normalized_rows = rows if isinstance(rows, list) else []
    if not normalized_rows:
        return [f"- {label}: none"]
    lines = [f"- {label}:"]
    for row in normalized_rows[:MAX_SUMMARY_QUERIES_PER_BUCKET]:
        query = str(row.get("query") or "").strip()
        angle = str(row.get("angle") or "").strip()
        status = str(row.get("status") or "").strip()
        counts = (
            f"returned={int(row.get('returned_count') or 0)}, "
            f"accepted={int(row.get('accepted_count') or 0)}, "
            f"visible={int(row.get('visible_new_suggestions_count') or 0)}, "
            f"duplicates={int(row.get('duplicate_count') or 0)}, "
            f"rejected={int(row.get('rejected_count') or 0)}"
        )
        suffix = f" | angle={angle}" if angle else ""
        error_suffix = f" | error={str(row.get('error_message') or '').strip()}" if str(row.get("error_message") or "").strip() else ""
        lines.append(f"  - {query} | status={status} | {counts}{suffix}{error_suffix}")
    return lines


def _render_angle_bucket(label: str, rows: Any) -> list[str]:
    normalized_rows = rows if isinstance(rows, list) else []
    if not normalized_rows:
        return [f"- {label}: none"]
    lines = [f"- {label}:"]
    for row in normalized_rows[:MAX_SUMMARY_ANGLES_PER_BUCKET]:
        angle = str(row.get("angle") or "").strip()
        count = int(row.get("count") or 0)
        if not angle:
            continue
        lines.append(f"  - {angle} ({count})")
    return lines


def _render_pattern_bucket(label: str, rows: Any, *, label_key: str = "pattern") -> list[str]:
    normalized_rows = rows if isinstance(rows, list) else []
    if not normalized_rows:
        return [f"- {label}: none"]
    lines = [f"- {label}:"]
    for row in normalized_rows[:MAX_SUMMARY_ANGLES_PER_BUCKET]:
        pattern = str(row.get(label_key) or "").strip()
        count = int(row.get("count") or 0)
        if not pattern:
            continue
        lines.append(f"  - {pattern} ({count})")
    return lines


def _extract_stale_years(query: str) -> list[str]:
    current_year = timezone.localdate().year
    years: list[str] = []
    for match in re.finditer(r"\b20\d{2}\b", str(query or "")):
        if int(match.group(0)) < current_year:
            years.append(match.group(0))
    return years


def _merge_quality_feedback_summary(
    feedback: Any,
    *,
    weak_material_types: Counter[str],
    preferred_material_types: Counter[str],
    weak_domains: Counter[str],
    dominant_rejection_reasons: Counter[str],
    quality_guidance: list[str],
) -> None:
    if not isinstance(feedback, dict):
        return
    for item in feedback.get("dominant_rejection_reasons") or []:
        if isinstance(item, dict):
            reason = str(item.get("reason") or "").strip()
            if reason:
                dominant_rejection_reasons[reason] += int(item.get("count") or 0)
    for item in feedback.get("weak_domains") or []:
        if isinstance(item, dict):
            domain = str(item.get("domain") or "").strip()
            if domain:
                weak_domains[domain] += int(item.get("count") or 0)
    for item in feedback.get("weak_material_types") or []:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("material_type") or "").strip()
            if label:
                weak_material_types[label] += int(item.get("count") or 0)
    for item in feedback.get("preferred_material_types_found") or []:
        if isinstance(item, dict):
            label = str(item.get("label") or item.get("material_type") or "").strip()
            if label:
                preferred_material_types[label] += int(item.get("count") or 0)
    for item in feedback.get("planner_quality_guidance") or []:
        cleaned = str(item or "").strip()
        if cleaned and cleaned not in quality_guidance:
            quality_guidance.append(cleaned)
