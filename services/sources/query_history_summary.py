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
MAX_SURFACE_MEMORY_ROWS = 8
MAX_RECENT_QUERY_TEXTS = 12

_SURFACE_KEY_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("on_chain_exchange_reserves_analysis", ("exchange reserves",)),
    ("on_chain_weekly_report", ("on-chain weekly report", "on chain weekly report")),
    ("on_chain_research_paper", ("on-chain research paper", "on chain research paper")),
    ("market_structure_research_paper", ("market structure research paper",)),
    ("research_paper", ("research paper",)),
    ("etf_flow_data_market_report", ("etf flow data", "spot etf fund flows")),
    ("etf_flows_report", ("etf flows weekly report", "etf flows")),
    ("institutional_demand_report", ("treasury holdings", "treasury activity", "institutional demand")),
    ("institutional_flows_report", ("institutional fund flows", "institutional flows")),
    ("funding_open_interest_report", ("funding rates open interest", "open interest funding rates")),
    ("funding_rates_analysis", ("funding rates",)),
    ("open_interest_futures_positioning", ("open interest futures positioning", "open interest positioning")),
    ("derivatives_positioning_market_structure", ("derivatives positioning", "futures positioning")),
    ("market_structure_report", ("market structure",)),
    ("analyst_report", ("analyst report", "market outlook report")),
    ("on_chain_analysis", ("on-chain analysis", "on chain analysis")),
    ("volatility_market_structure_report", ("volatility market structure",)),
    ("volatility_drawdown_risk_analysis", ("drawdown risk", "volatility risk", "market volatility")),
)

_PREFERRED_MATERIAL_TYPE_SURFACE_KEYS: dict[str, tuple[str, ...]] = {
    "institutional / analyst report": (
        "analyst_report",
        "institutional_demand_report",
        "institutional_flows_report",
    ),
    "market data / flow analysis": (
        "market_structure_report",
        "etf_flows_report",
        "institutional_flows_report",
        "funding_rates_analysis",
        "open_interest_futures_positioning",
    ),
    "on-chain analysis": (
        "on_chain_exchange_reserves_analysis",
        "on_chain_weekly_report",
        "on_chain_analysis",
    ),
    "market structure analysis": (
        "market_structure_report",
        "derivatives_positioning_market_structure",
        "funding_rates_analysis",
    ),
    "research paper": (
        "research_paper",
        "market_structure_research_paper",
        "on_chain_research_paper",
    ),
}

_SURFACE_KEY_QUERY_TERMS: dict[str, str] = {
    "etf_flows_report": "ETF flows weekly report",
    "etf_flow_data_market_report": "spot ETF fund flows analysis",
    "institutional_demand_report": "treasury holdings institutional demand",
    "institutional_flows_report": "institutional fund flows report",
    "funding_open_interest_report": "funding rates open interest report",
    "funding_rates_analysis": "funding rates analysis",
    "open_interest_futures_positioning": "open interest futures positioning",
    "derivatives_positioning_market_structure": "derivatives positioning market structure",
    "market_structure_report": "market structure report",
    "market_structure_research_paper": "market structure research paper",
    "research_paper": "research paper",
    "on_chain_exchange_reserves_analysis": "on-chain exchange reserves analysis",
    "on_chain_weekly_report": "on-chain weekly report",
    "on_chain_analysis": "on-chain analysis recent report",
    "analyst_report": "analyst report market outlook",
    "volatility_market_structure_report": "volatility market structure report",
    "volatility_drawdown_risk_analysis": "volatility drawdown risk analysis",
}


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
    all_rows: list[dict[str, Any]] = []

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
            row = _normalize_query_history_row(item, created_at=getattr(run, "created_at", None))
            if row is None:
                continue
            total_query_rows += 1
            all_rows.append(row)
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
        "recent_query_texts": _limit_recent_query_texts(all_rows),
        "search_surface_memory": _build_recent_search_surface_memory(
            rows=all_rows,
            recent_run_count=len(runs),
            preferred_material_types_rows=_limit_counter(preferred_material_types, label_key="material_type"),
            quality_guidance=quality_guidance[:MAX_SUMMARY_GUIDANCE_ITEMS],
        ),
        "planning_guidance": [],
    }
    summary["planning_guidance"] = _build_planning_guidance(summary)
    return summary


def render_query_history_summary_for_prompt(summary: dict[str, Any] | None) -> str:
    normalized = summary if isinstance(summary, dict) else _empty_summary()
    if not normalized.get("history_available"):
        return "No prior query performance history is available for this topic."

    lines = [
        "Recent query history summary:",
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
    lines.extend(_render_search_surface_memory_bucket(normalized.get("search_surface_memory")))
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
        "recent_query_texts": [],
        "search_surface_memory": {
            "recent_run_count": 0,
            "surfaces": [],
            "avoided_surfaces": [],
            "preferred_surfaces": [],
            "underexplored_surfaces": [],
        },
        "planning_guidance": [],
    }


def _normalize_query_history_row(item: dict[str, Any], *, created_at=None) -> dict[str, Any] | None:
    query = str(item.get("query") or "").strip()
    if not query:
        return None
    status = str(item.get("status") or "").strip() or "no_visible_results"
    angle = str(item.get("angle") or "").strip()
    purpose = str(item.get("purpose") or "").strip()
    surface_key = str(item.get("surface_key") or "").strip() or _surface_key_for_query_text(query)
    material_type = str(item.get("material_type") or "").strip()
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
        "surface_key": surface_key,
        "material_type": material_type,
        "created_at": created_at.isoformat() if created_at else "",
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
    search_surface_memory = summary.get("search_surface_memory") if isinstance(summary.get("search_surface_memory"), dict) else {}

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
    if search_surface_memory:
        avoided_surfaces = [str(item or "").strip() for item in search_surface_memory.get("avoided_surfaces") or [] if str(item or "").strip()]
        preferred_surfaces = [str(item or "").strip() for item in search_surface_memory.get("preferred_surfaces") or [] if str(item or "").strip()]
        underexplored_surfaces = [str(item or "").strip() for item in search_surface_memory.get("underexplored_surfaces") or [] if str(item or "").strip()]
        uncertain_surfaces = [
            str(item.get("surface_key") or "").strip()
            for item in search_surface_memory.get("surfaces") or []
            if isinstance(item, dict) and str(item.get("status") or "").strip() == "unknown"
        ]
        if avoided_surfaces:
            guidance.append(
                f"Avoid starting with exhausted surfaces from recent clicks: {', '.join(_humanize_surface_key(item) for item in avoided_surfaces[:3])}."
            )
        if preferred_surfaces:
            guidance.append(
                f"Prefer useful surfaces that still have room: {', '.join(_humanize_surface_key(item) for item in preferred_surfaces[:3])}."
            )
        if underexplored_surfaces:
            guidance.append(
                f"Try underexplored adjacent surfaces next: {', '.join(_humanize_surface_key(item) for item in underexplored_surfaces[:3])}."
            )
        if uncertain_surfaces and len(guidance) < MAX_SUMMARY_GUIDANCE_ITEMS:
            guidance.append(
                f"Provider-error-only surfaces remain uncertain, not exhausted: {', '.join(_humanize_surface_key(item) for item in uncertain_surfaces[:2])}."
            )
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


def _limit_recent_query_texts(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    recent_queries: list[str] = []
    for row in rows:
        query = str(row.get("query") or "").strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        recent_queries.append(query)
        if len(recent_queries) >= MAX_RECENT_QUERY_TEXTS:
            break
    return recent_queries


def _build_recent_search_surface_memory(
    *,
    rows: list[dict[str, Any]],
    recent_run_count: int,
    preferred_material_types_rows: list[dict[str, Any]],
    quality_guidance: list[str],
) -> dict[str, Any]:
    surface_stats: dict[str, dict[str, Any]] = {}
    for row in rows:
        surface_key = str(row.get("surface_key") or "").strip()
        if not surface_key:
            continue
        stats = surface_stats.setdefault(
            surface_key,
            {
                "surface_key": surface_key,
                "visible_count": 0,
                "known_duplicate_count": 0,
                "quality_rejected_count": 0,
                "returned_count": 0,
                "provider_error_count": 0,
                "duplicate_heavy_count": 0,
                "quality_heavy_count": 0,
                "last_seen": "",
            },
        )
        visible = int(row.get("visible_new_suggestions_count") or 0)
        duplicates = int(row.get("duplicate_count") or 0)
        rejected = int(row.get("rejected_count") or 0)
        returned = int(row.get("returned_count") or 0)
        status = str(row.get("status") or "").strip()
        stats["visible_count"] += visible
        stats["known_duplicate_count"] += duplicates
        stats["quality_rejected_count"] += rejected
        stats["returned_count"] += returned
        if status == "partial_error":
            stats["provider_error_count"] += 1
        if status == "duplicate_heavy":
            stats["duplicate_heavy_count"] += 1
        if status in {"weak", "no_visible_results"} and rejected > 0 and visible == 0:
            stats["quality_heavy_count"] += 1
        last_seen = str(row.get("created_at") or "").strip()
        if last_seen and last_seen > str(stats.get("last_seen") or ""):
            stats["last_seen"] = last_seen

    observed_surface_keys = set(surface_stats)
    underexplored_keys = _select_underexplored_surfaces(
        preferred_material_types_rows=preferred_material_types_rows,
        quality_guidance=quality_guidance,
        observed_surface_keys=observed_surface_keys,
    )

    surfaces: list[dict[str, Any]] = []
    avoided_surfaces: list[str] = []
    preferred_surfaces: list[str] = []
    for surface_key, stats in surface_stats.items():
        status, reason = _classify_search_surface_outcomes(stats)
        row = {
            "surface_key": surface_key,
            "status": status,
            "visible_count": int(stats.get("visible_count") or 0),
            "known_duplicate_count": int(stats.get("known_duplicate_count") or 0),
            "quality_rejected_count": int(stats.get("quality_rejected_count") or 0),
            "returned_count": int(stats.get("returned_count") or 0),
            "last_seen": str(stats.get("last_seen") or "") or None,
            "reason": reason,
        }
        surfaces.append(row)
        if status == "exhausted":
            avoided_surfaces.append(surface_key)
        elif status == "useful":
            preferred_surfaces.append(surface_key)

    for surface_key in underexplored_keys:
        surfaces.append(
            {
                "surface_key": surface_key,
                "status": "underexplored",
                "visible_count": 0,
                "known_duplicate_count": 0,
                "quality_rejected_count": 0,
                "returned_count": 0,
                "last_seen": None,
                "reason": "Preferred adjacent surface has little or no recent coverage.",
            }
        )

    surfaces.sort(key=lambda item: (_surface_status_rank(str(item.get("status") or "").strip()), -int(item.get("visible_count") or 0), str(item.get("surface_key") or "")))
    return {
        "recent_run_count": recent_run_count,
        "surfaces": surfaces[:MAX_SURFACE_MEMORY_ROWS],
        "avoided_surfaces": avoided_surfaces[:4],
        "preferred_surfaces": preferred_surfaces[:4],
        "underexplored_surfaces": underexplored_keys[:4],
    }


def _classify_search_surface_outcomes(stats: dict[str, Any]) -> tuple[str, str]:
    visible = int(stats.get("visible_count") or 0)
    duplicates = int(stats.get("known_duplicate_count") or 0)
    rejected = int(stats.get("quality_rejected_count") or 0)
    returned = int(stats.get("returned_count") or 0)
    provider_errors = int(stats.get("provider_error_count") or 0)
    duplicate_heavy_count = int(stats.get("duplicate_heavy_count") or 0)
    quality_heavy_count = int(stats.get("quality_heavy_count") or 0)
    evidence_total = max(returned, visible + duplicates + rejected, 1)
    duplicate_ratio = duplicates / evidence_total
    quality_ratio = rejected / evidence_total

    if provider_errors > 0 and returned == 0 and visible == 0 and duplicates == 0 and rejected == 0:
        return "unknown", "Recent attempts hit provider errors only, so this surface remains uncertain."
    if (visible == 0 and duplicates >= 5) or (duplicate_ratio >= 0.6 and visible <= 1) or (duplicate_heavy_count >= 2 and visible <= 1):
        return "exhausted", "Recent clicks mostly hit already-known or duplicate URLs."
    if (quality_ratio >= 0.6 and visible == 0) or (quality_heavy_count >= 2 and visible == 0):
        return "weak", "Recent clicks mostly produced low-quality or low-substance results."
    if visible >= 2 and duplicate_ratio < 0.5:
        return "useful", "Recent clicks still surfaced visible new suggestions without heavy duplication."
    if visible >= 1 and duplicates <= rejected:
        return "useful", "Recent clicks still produced some visible suggestions and are not clearly exhausted."
    return "unknown", "Recent evidence is mixed, so this surface is neither clearly exhausted nor clearly useful."


def _select_underexplored_surfaces(
    *,
    preferred_material_types_rows: list[dict[str, Any]],
    quality_guidance: list[str],
    observed_surface_keys: set[str],
) -> list[str]:
    candidate_keys: list[str] = []
    seen: set[str] = set()
    for item in preferred_material_types_rows:
        material_type = str(item.get("material_type") or "").strip()
        for surface_key in _PREFERRED_MATERIAL_TYPE_SURFACE_KEYS.get(material_type, ()):
            if surface_key in seen:
                continue
            seen.add(surface_key)
            candidate_keys.append(surface_key)
    for surface_key in _extract_surface_keys_from_quality_guidance(quality_guidance):
        if surface_key in seen:
            continue
        seen.add(surface_key)
        candidate_keys.append(surface_key)
    underexplored: list[str] = []
    for surface_key in candidate_keys:
        if surface_key in observed_surface_keys:
            continue
        underexplored.append(surface_key)
        if len(underexplored) >= 4:
            break
    return underexplored


def _extract_surface_keys_from_quality_guidance(guidance: list[str]) -> list[str]:
    surface_keys: list[str] = []
    seen: set[str] = set()
    for item in guidance:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        match = re.search(r"Use query terms such as (.+?)(?:\.|$)", cleaned, flags=re.IGNORECASE)
        if not match:
            continue
        for part in match.group(1).split(","):
            candidate = re.sub(r"\s+", " ", part.strip(" .")).strip()
            surface_key = _surface_key_for_query_text(candidate)
            if not surface_key or surface_key in seen:
                continue
            seen.add(surface_key)
            surface_keys.append(surface_key)
    return surface_keys


def _surface_key_for_query_text(query: str) -> str:
    normalized_query = str(query or "").casefold()
    for surface_key, needles in _SURFACE_KEY_PATTERNS:
        if any(needle in normalized_query for needle in needles):
            return surface_key
    return ""


def _render_search_surface_memory_bucket(memory: Any) -> list[str]:
    normalized = memory if isinstance(memory, dict) else {}
    surfaces = normalized.get("surfaces") or []
    if not surfaces:
        return ["- Search surface memory: none"]
    lines = ["- Search surface memory:"]
    avoided = normalized.get("avoided_surfaces") or []
    preferred = normalized.get("preferred_surfaces") or []
    underexplored = normalized.get("underexplored_surfaces") or []
    if avoided:
        lines.append(f"  - Avoided surfaces: {', '.join(_humanize_surface_key(item) for item in avoided)}")
    if preferred:
        lines.append(f"  - Preferred surfaces: {', '.join(_humanize_surface_key(item) for item in preferred)}")
    if underexplored:
        lines.append(f"  - Underexplored surfaces: {', '.join(_humanize_surface_key(item) for item in underexplored)}")
    for item in surfaces[:MAX_SUMMARY_ANGLES_PER_BUCKET]:
        if not isinstance(item, dict):
            continue
        surface_key = str(item.get("surface_key") or "").strip()
        status = str(item.get("status") or "").strip()
        if not surface_key or not status:
            continue
        lines.append(
            "  - "
            f"{_humanize_surface_key(surface_key)} | status={status} | "
            f"visible={int(item.get('visible_count') or 0)}, "
            f"duplicates={int(item.get('known_duplicate_count') or 0)}, "
            f"rejected={int(item.get('quality_rejected_count') or 0)}"
        )
    return lines


def _surface_status_rank(status: str) -> int:
    order = {
        "exhausted": 0,
        "useful": 1,
        "underexplored": 2,
        "weak": 3,
        "unknown": 4,
    }
    return order.get(status, 9)


def _humanize_surface_key(surface_key: str) -> str:
    return _SURFACE_KEY_QUERY_TERMS.get(surface_key, str(surface_key or "").replace("_", " ")).strip()


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
