import re
from collections import Counter

from django.utils import timezone

from apps.digests.models import SourceDiscoveryHistory, SourceDiscoveryRun
from apps.topics.models import Topic, TopicSourceOrigin
from services.sources.discovery_diagnostics import (
    format_discovery_cycle_decision_label,
    format_discovery_cycle_diagnosis_label,
    format_discovery_cycle_next_action_label,
)


def _build_research_history_run_entries(topic: Topic) -> list[dict]:
    runs = list(topic.source_discovery_runs.order_by("-created_at", "-id"))
    entries: list[dict] = []
    for run in runs:
        diagnostics = dict(run.diagnostics or {})
        provider_errors = _build_research_history_provider_errors(diagnostics)
        compact_metrics = _build_research_history_compact_metrics(run, diagnostics)
        stage_diagnostics = _build_research_history_stage_diagnostics(run, diagnostics)
        entries.append(
            {
                "run": run,
                "title": _format_research_history_status_title(run.status, run=run),
                "subtitle": _build_research_history_status_subtitle(run),
                "completed_label": _format_research_history_timestamp(run),
                "compact_metrics": compact_metrics,
                "stage_diagnostics": stage_diagnostics,
                "strategy_rows": _build_research_history_strategy_rows(run, diagnostics),
                "technical_rows": _build_research_history_detail_rows(run, diagnostics, provider_errors),
                "query_rows": _build_discovery_query_rows(diagnostics),
                "quality_feedback": _build_research_history_quality_feedback(diagnostics),
                "cycle_info": _build_research_history_cycle_info(diagnostics),
                "provider_errors": provider_errors,
                "warning_title": _build_research_history_warning_title(run, provider_errors),
                "warning_body": _build_research_history_warning_body(run, provider_errors),
            }
        )
    return entries


def _build_seen_source_history_entries(history_rows: list[SourceDiscoveryHistory]) -> list[dict]:
    entries: list[dict] = []
    for row in history_rows:
        display_url = row.url or row.normalized_url
        details: list[dict[str, str]] = []
        freshness_label = _format_source_history_freshness_label(row.freshness_status)
        if freshness_label and freshness_label not in {"Unknown date", "Unknown"}:
            details.append({"label": "Freshness", "value": freshness_label, "kind": "text"})
        if row.detected_publication_year:
            details.append({"label": "Publication year", "value": str(row.detected_publication_year), "kind": "text"})
        if str(row.quality_rejection_reason or "").strip():
            details.append({"label": "Quality note", "value": row.quality_rejection_reason.strip(), "kind": "text"})
        if details:
            details.insert(
                0,
                {"label": "First seen", "value": _format_history_timestamp(row.first_seen_at), "kind": "text"},
            )
        entries.append(
            {
                "title": row.title or row.normalized_url or row.url,
                "url": display_url,
                "domain": row.domain or "unknown",
                "status_label": _format_source_history_status_label(row.status),
                "outcome_label": _format_source_history_outcome_label(row.last_run_outcome),
                "seen_count": str(row.seen_count or 0),
                "last_seen": _format_history_timestamp(row.last_seen_at),
                "details": details,
                "has_details": bool(details),
            }
        )
    return entries


def _build_query_performance_section(topic: Topic) -> dict:
    entries: list[dict] = []
    for run in topic.source_discovery_runs.order_by("-completed_at", "-created_at", "-id"):
        diagnostics = run.diagnostics if isinstance(run.diagnostics, dict) else {}
        query_rows = diagnostics.get("query_performance") or _build_legacy_query_performance_rows(run, diagnostics)
        for item in query_rows:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            entries.append(
                {
                    "query": query,
                    "provider": str(item.get("provider") or run.provider_name or "").strip() or "unknown",
                    "purpose": _build_query_performance_purpose(item),
                    "returned_count": _format_query_metric_value(item.get("returned_count")),
                    "accepted_count": _format_query_metric_value(item.get("accepted_count")),
                    "visible_count": _format_query_metric_value(item.get("visible_new_suggestions_count")),
                    "rejected_count": _format_query_metric_value(item.get("rejected_count")),
                    "duplicate_count": _format_query_metric_value(item.get("duplicate_count")),
                    "status_label": _format_query_performance_status_label(str(item.get("status") or "").strip()),
                    "last_used": _format_research_history_timestamp(run),
                }
            )
    return {"entries": entries}


def _build_source_quality_feedback_section(topic: Topic) -> dict:
    aggregate = {
        "quality_rejected_count": 0,
        "known_or_duplicate_count": 0,
        "shown_count": 0,
        "dominant_rejection_reasons": Counter(),
        "weak_domains": Counter(),
        "weak_material_types": Counter(),
        "preferred_material_types_found": Counter(),
        "planner_quality_guidance": [],
    }
    latest_main_quality_issue = ""

    for run in topic.source_discovery_runs.order_by("-completed_at", "-created_at", "-id")[:5]:
        diagnostics = run.diagnostics if isinstance(run.diagnostics, dict) else {}
        feedback = diagnostics.get("source_quality_feedback")
        if not isinstance(feedback, dict):
            continue
        aggregate["quality_rejected_count"] += int(feedback.get("quality_rejected_count") or 0)
        aggregate["known_or_duplicate_count"] += int(feedback.get("known_or_duplicate_count") or 0)
        aggregate["shown_count"] += int(feedback.get("shown_count") or 0)
        if not latest_main_quality_issue:
            latest_main_quality_issue = str(feedback.get("main_quality_issue") or "").strip()
        for item in feedback.get("dominant_rejection_reasons") or []:
            if isinstance(item, dict):
                reason = str(item.get("reason") or "").strip()
                if reason:
                    aggregate["dominant_rejection_reasons"][reason] += int(item.get("count") or 0)
        for item in feedback.get("weak_domains") or []:
            if isinstance(item, dict):
                domain = str(item.get("domain") or "").strip()
                reason = str(item.get("reason") or "").strip()
                if domain:
                    aggregate["weak_domains"][(domain, reason)] += int(item.get("count") or 0)
        for item in feedback.get("weak_material_types") or []:
            if isinstance(item, dict):
                material_type = str(item.get("material_type") or "").strip()
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if material_type:
                    aggregate["weak_material_types"][(material_type, label)] += int(item.get("count") or 0)
        for item in feedback.get("preferred_material_types_found") or []:
            if isinstance(item, dict):
                material_type = str(item.get("material_type") or "").strip()
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if material_type:
                    aggregate["preferred_material_types_found"][(material_type, label)] += int(item.get("count") or 0)
        for item in feedback.get("planner_quality_guidance") or []:
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in aggregate["planner_quality_guidance"]:
                aggregate["planner_quality_guidance"].append(cleaned)

    has_feedback = bool(
        aggregate["quality_rejected_count"]
        or aggregate["known_or_duplicate_count"]
        or aggregate["shown_count"]
        or aggregate["dominant_rejection_reasons"]
        or aggregate["weak_domains"]
        or aggregate["weak_material_types"]
        or aggregate["preferred_material_types_found"]
        or aggregate["planner_quality_guidance"]
    )
    return {
        "has_feedback": has_feedback,
        "main_quality_issue": latest_main_quality_issue or "No strong quality pattern detected yet.",
        "quality_rejected_count": str(aggregate["quality_rejected_count"]),
        "known_or_duplicate_count": str(aggregate["known_or_duplicate_count"]),
        "shown_count": str(aggregate["shown_count"]),
        "dominant_rejection_reasons": [
            {"reason": reason, "count": str(count)}
            for reason, count in aggregate["dominant_rejection_reasons"].most_common(3)
        ],
        "weak_domains": [
            {"domain": domain, "reason": reason, "count": str(count)}
            for (domain, reason), count in aggregate["weak_domains"].most_common(3)
        ],
        "weak_material_types": [
            {"material_type": material_type, "label": label, "count": str(count)}
            for (material_type, label), count in aggregate["weak_material_types"].most_common(4)
        ],
        "preferred_material_types_found": [
            {"material_type": material_type, "label": label, "count": str(count)}
            for (material_type, label), count in aggregate["preferred_material_types_found"].most_common(4)
        ],
        "planner_quality_guidance": aggregate["planner_quality_guidance"][:4],
    }


def _build_search_surface_memory_section(topic: Topic) -> dict:
    latest_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    if latest_run is None or not isinstance(latest_run.diagnostics, dict):
        return {"has_memory": False}
    history_summary = latest_run.diagnostics.get("query_history_summary")
    if not isinstance(history_summary, dict):
        return {"has_memory": False}
    memory = history_summary.get("search_surface_memory")
    if not isinstance(memory, dict):
        return {"has_memory": False}

    surfaces = [
        item
        for item in memory.get("surfaces") or []
        if isinstance(item, dict) and str(item.get("surface_key") or "").strip()
    ]
    if not surfaces and not any(memory.get(key) for key in ("avoided_surfaces", "preferred_surfaces", "underexplored_surfaces")):
        return {"has_memory": False}

    return {
        "has_memory": True,
        "recent_run_count": int(memory.get("recent_run_count") or 0),
        "avoided_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("avoided_surfaces") or []
            if str(item or "").strip()
        ],
        "preferred_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("preferred_surfaces") or []
            if str(item or "").strip()
        ],
        "underexplored_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("underexplored_surfaces") or []
            if str(item or "").strip()
        ],
        "surfaces": [
            {
                "label": _humanize_surface_key_label(item.get("surface_key")),
                "status": _humanize_surface_status_label(item.get("status")),
                "reason": str(item.get("reason") or "").strip(),
            }
            for item in surfaces[:4]
        ],
    }


def _build_legacy_query_performance_rows(run: SourceDiscoveryRun, diagnostics: dict) -> list[dict]:
    rows: list[dict] = []
    for item in diagnostics.get("per_query_result_counts", []) or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        rows.append(
            {
                "query": query,
                "provider": str(item.get("provider_name") or run.provider_name or "").strip(),
                "angle": str(item.get("angle") or "").strip(),
                "purpose": str(item.get("purpose") or item.get("query_reason") or "").strip(),
                "returned_count": int(item.get("result_count") or 0),
                "accepted_count": None,
                "visible_new_suggestions_count": None,
                "rejected_count": None,
                "duplicate_count": int(item.get("duplicate_url_count") or 0),
                "status": "partial_error" if str(item.get("error") or "").strip() else "no_visible_results",
            }
        )
    return rows


def _build_query_performance_purpose(item: dict) -> str:
    angle = str(item.get("angle") or "").strip()
    purpose = str(item.get("purpose") or "").strip()
    if angle and purpose:
        return f"{angle} вЂ” {purpose}"
    if angle:
        return angle
    if purpose:
        return purpose
    return "вЂ”"


def _format_query_metric_value(value) -> str:
    if value is None:
        return "вЂ”"
    return str(int(value))


def _format_query_performance_status_label(status: str) -> str:
    mapping = {
        "useful": "Useful",
        "weak": "Weak",
        "duplicate_heavy": "Duplicate-heavy",
        "no_visible_results": "No visible results",
        "partial_error": "Partial/error",
    }
    return mapping.get(str(status or "").strip(), "No visible results")


def _build_research_history_compact_metrics(run: SourceDiscoveryRun, diagnostics: dict) -> str:
    status = str(run.status or "").strip().lower()
    status_label = "partial run" if status == SourceDiscoveryRun.STATUS_PARTIAL_FAILED else (
        "failed run" if status == SourceDiscoveryRun.STATUS_FAILED else (
            "blocked run" if status == SourceDiscoveryRun.STATUS_BLOCKED else "completed run"
        )
    )
    return " В· ".join(
        [
            f"{int(run.provider_result_count or 0)} URLs returned",
            f"{int(run.new_suggestions_count or 0)} visible new suggestions",
            status_label,
        ]
    )


def _build_research_history_stage_diagnostics(run: SourceDiscoveryRun, diagnostics: dict) -> dict:
    duplicate_count = int(diagnostics.get("duplicate_url_count") or 0) + int(run.already_known_count or 0)
    return {
        "intro": "These counts describe different pipeline checks and may overlap; they are not an additive breakdown.",
        "rows": [
            {"label": "Passed filtering", "value": str(int(run.accepted_count or 0))},
            {"label": "Rejected by filters", "value": str(int(run.rejected_count or 0))},
            {"label": "Already known or duplicate", "value": str(duplicate_count)},
        ],
    }


def _build_research_history_strategy_rows(run: SourceDiscoveryRun, diagnostics: dict) -> list[dict[str, str]]:
    query_angle_key = str(diagnostics.get("selected_query_angle_key") or "").strip()
    query_angle_suffix = str(diagnostics.get("selected_query_angle_suffix") or "").strip()
    rows = [
        {"label": "Recency", "value": _format_recency_label(int(run.search_recency_months or 1))},
        {"label": "Queries used", "value": str(run.query_count)},
    ]
    if query_angle_key:
        rows.insert(
            0,
            {"label": "Search angle", "value": _format_query_angle_label(query_angle_key, query_angle_suffix)},
        )
    return rows


def _build_research_history_detail_rows(
    run: SourceDiscoveryRun,
    diagnostics: dict,
    provider_errors: list[str],
) -> list[dict[str, str]]:
    query_angle_reason = str(diagnostics.get("selected_query_angle_reason") or "").strip()
    previous_run_count = int(diagnostics.get("previous_discovery_run_count") or 0)

    rows = [
        {"label": "Raw status", "value": str(run.status or "unknown")},
        {"label": "Provider", "value": run.provider_name or "unknown"},
    ]
    if query_angle_reason:
        rows.append({"label": "Angle reason", "value": query_angle_reason})
    rows.extend(
        [
            {"label": "Previous discovery runs", "value": str(previous_run_count)},
            {"label": "Recency", "value": _format_recency_label(int(run.search_recency_months or 1))},
            {"label": "Provider filter", "value": run.search_time_filter or str(diagnostics.get("provider_tbs") or "").strip() or "none"},
        ]
    )
    if provider_errors:
        rows.append({"label": "Technical reason", "value": provider_errors[0]})
    return rows


def _build_research_history_quality_feedback(diagnostics: dict) -> dict:
    feedback = diagnostics.get("source_quality_feedback")
    if not isinstance(feedback, dict):
        return {"has_feedback": False}
    return {
        "has_feedback": True,
        "main_quality_issue": str(feedback.get("main_quality_issue") or "").strip(),
        "rows": [
            {"label": "Quality rejected", "value": str(int(feedback.get("quality_rejected_count") or 0))},
            {"label": "Known / duplicate", "value": str(int(feedback.get("known_or_duplicate_count") or 0))},
            {"label": "Shown", "value": str(int(feedback.get("shown_count") or 0))},
        ],
        "weak_material_types": [
            {
                "label": str(item.get("label") or item.get("material_type") or "").strip(),
                "count": str(int(item.get("count") or 0)),
            }
            for item in feedback.get("weak_material_types") or []
            if isinstance(item, dict)
        ],
        "preferred_material_types_found": [
            {
                "label": str(item.get("label") or item.get("material_type") or "").strip(),
                "count": str(int(item.get("count") or 0)),
            }
            for item in feedback.get("preferred_material_types_found") or []
            if isinstance(item, dict)
        ],
        "planner_quality_guidance": [
            str(item).strip()
            for item in feedback.get("planner_quality_guidance") or []
            if str(item).strip()
        ][:4],
    }


def _build_research_history_cycle_info(diagnostics: dict) -> dict:
    cycle = diagnostics.get("discovery_cycle")
    if not isinstance(cycle, dict):
        return {"has_cycle": False}
    cycle_diagnosis = cycle.get("cycle_diagnosis") if isinstance(cycle.get("cycle_diagnosis"), dict) else {}
    repair_plan = cycle.get("repair_plan") if isinstance(cycle.get("repair_plan"), dict) else {}
    current_round_item = {}
    current_round_index = int(cycle.get("round_index") or 0)
    for item in cycle.get("rounds") or []:
        if isinstance(item, dict) and int(item.get("round_index") or 0) == current_round_index:
            current_round_item = item
            break
    return {
        "has_cycle": True,
        "summary": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip()),
        "rows": [
            {"label": "Cycle round", "value": f"{int(cycle.get('round_index') or 0)} of {int(cycle.get('round_count') or 0)}"},
            {"label": "Cycle target", "value": str(int(cycle.get("target_visible_new_suggestions") or 0))},
            {"label": "Max immediate rounds", "value": str(int(cycle.get("max_immediate_rounds") or 0))},
            {"label": "Rounds run", "value": str(int(cycle.get("round_count") or 0))},
            {"label": "Accumulated visible suggestions", "value": str(int(cycle.get("accumulated_visible_suggestions") or 0))},
            {"label": "Cycle decision", "value": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip())},
        ],
        "decision": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip()),
        "diagnosis": _build_research_history_cycle_diagnosis(cycle_diagnosis),
        "repair": _build_research_history_cycle_repair(repair_plan),
        "repair_used": _build_research_history_repair_usage(current_round_item),
    }


def _build_current_research_state(topic: Topic) -> dict:
    history_qs = topic.source_discovery_history.all()
    status_counts = Counter(history_qs.values_list("status", flat=True))
    current_discovered_sources = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED)
    last_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    last_cycle = {}
    if last_run is not None and isinstance(last_run.diagnostics, dict):
        last_cycle = last_run.diagnostics.get("discovery_cycle") or {}
    return {
        "cards": [
            {"label": "Kept", "value": str(current_discovered_sources.filter(is_pinned=True).count())},
            {"label": "Shown now", "value": str(current_discovered_sources.filter(is_pinned=False).count())},
            {"label": "Rejected", "value": str(int(status_counts.get(SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY, 0)))},
            {"label": "Seen only", "value": str(int(status_counts.get(SourceDiscoveryHistory.STATUS_SEEN, 0)))},
        ],
        "last_run_status": _format_research_history_status_title(last_run.status, run=last_run) if last_run else "No discovery runs yet",
        "last_run_subtitle": _build_current_research_feedback_note(last_run) if last_run else "Run Find sources to start source discovery for this topic.",
        "last_cycle_summary": _build_current_research_cycle_summary(last_cycle),
    }


def _build_current_research_cycle_summary(cycle: dict) -> str:
    if not isinstance(cycle, dict) or not cycle:
        return ""
    target = int(cycle.get("target_visible_new_suggestions") or 0)
    visible = int(cycle.get("accumulated_visible_suggestions") or 0)
    decision = format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip())
    cycle_diagnosis = cycle.get("cycle_diagnosis") if isinstance(cycle.get("cycle_diagnosis"), dict) else {}
    diagnosis_note = ""
    if cycle_diagnosis:
        primary_cause = str(cycle_diagnosis.get("primary_cause") or "").strip()
        if primary_cause not in {"", "target_reached", "provider_unavailable"}:
            diagnosis_note = f" вЂ” {format_discovery_cycle_diagnosis_label(primary_cause).lower()}."
    if target > 0:
        return f"Last discovery cycle: {decision.lower()} ({visible} of {target} visible suggestions).{diagnosis_note}"
    return f"Last discovery cycle: {decision.lower()}.{diagnosis_note}"


def _build_current_research_feedback_note(run: SourceDiscoveryRun) -> str:
    diagnostics = dict(getattr(run, "diagnostics", {}) or {})
    cycle = diagnostics.get("discovery_cycle")
    if not isinstance(cycle, dict) or not cycle:
        return _build_research_history_status_subtitle(run)

    decision = str(cycle.get("decision") or "").strip()
    visible = int(cycle.get("accumulated_visible_suggestions") or 0)
    rounds_run = int(cycle.get("rounds_run") or cycle.get("round_count") or 0)
    if decision == "provider_unavailable":
        return _build_research_history_status_subtitle(run)
    if decision == "target_reached" and visible > 0 and rounds_run > 0:
        return (
            f"Target reached: {visible} new source suggestion{'s' if visible != 1 else ''} "
            f"after {rounds_run} search round{'s' if rounds_run != 1 else ''}."
        )
    if visible > 0 and rounds_run > 0:
        return (
            f"{visible} new source suggestion{'s' if visible != 1 else ''} were found "
            f"after {rounds_run} search round{'s' if rounds_run != 1 else ''}."
        )
    return _build_research_history_status_subtitle(run)


def _build_full_research_history_copy_report(
    *,
    topic: Topic,
    current_research_state: dict,
    query_performance_entries: list[dict],
    source_quality_feedback: dict,
    search_surface_memory: dict,
    history_runs: list[dict],
    seen_source_history: list[dict],
) -> str:
    lines: list[str] = []
    active_source_count = topic.sources.filter(is_active=True).count()
    active_discovered_count = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED, is_active=True).count()
    active_manual_count = topic.sources.filter(is_active=True).exclude(origin=TopicSourceOrigin.DISCOVERED).count()
    current_discovered_sources = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED)
    latest_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    latest_history_summary = {}
    if latest_run is not None and isinstance(latest_run.diagnostics, dict):
        latest_history_summary = latest_run.diagnostics.get("query_history_summary") or {}

    def add_section(title: str) -> None:
        if lines:
            lines.append("")
        lines.append(title)

    add_section("Topic")
    lines.append(f"- id: {topic.id}")
    lines.append(f"- name: {topic.name}")

    add_section("Current research state")
    lines.append(f"- source mode: {topic.source_mode}")
    for metric in current_research_state.get("cards", []):
        lines.append(f"- {metric.get('label')}: {metric.get('value')}")
    lines.append(f"- active selected sources: {active_source_count}")
    lines.append(f"- active selected research sources: {active_discovered_count}")
    lines.append(f"- active selected my sources: {active_manual_count}")
    lines.append(f"- current kept discovered sources: {current_discovered_sources.filter(is_pinned=True).count()}")
    lines.append(f"- current shown discovered sources: {current_discovered_sources.filter(is_pinned=False).count()}")
    lines.append(f"- last discovery run status: {current_research_state.get('last_run_status')}")
    lines.append(f"- last discovery run note: {current_research_state.get('last_run_subtitle')}")

    add_section("Query performance")
    if query_performance_entries:
        for item in query_performance_entries:
            lines.append(f"- query: {item.get('query')}")
            lines.append(f"  provider: {item.get('provider')}")
            lines.append(f"  purpose / angle: {item.get('purpose')}")
            lines.append(f"  returned: {item.get('returned_count')}")
            lines.append(f"  accepted: {item.get('accepted_count')}")
            lines.append(f"  visible: {item.get('visible_count')}")
            lines.append(f"  rejected: {item.get('rejected_count')}")
            lines.append(f"  known / duplicates: {item.get('duplicate_count')}")
            lines.append(f"  status: {item.get('status_label')}")
            lines.append(f"  last used: {item.get('last_used')}")
    else:
        lines.append("- No query performance data yet.")

    add_section("Source quality feedback")
    if source_quality_feedback.get("has_feedback"):
        lines.append(f"- quality rejected count: {source_quality_feedback.get('quality_rejected_count')}")
        lines.append(f"- known or duplicate count: {source_quality_feedback.get('known_or_duplicate_count')}")
        lines.append(f"- shown count: {source_quality_feedback.get('shown_count')}")
        lines.append(f"- main quality issue: {source_quality_feedback.get('main_quality_issue')}")
        if source_quality_feedback.get("dominant_rejection_reasons"):
            lines.append("- dominant rejection reasons:")
            for item in source_quality_feedback.get("dominant_rejection_reasons", []):
                lines.append(f"  - {item.get('reason')}: {item.get('count')}")
        if source_quality_feedback.get("weak_domains"):
            lines.append("- weak domains:")
            for item in source_quality_feedback.get("weak_domains", []):
                lines.append(f"  - {item.get('domain')}: {item.get('count')} ({item.get('reason')})")
        if source_quality_feedback.get("weak_material_types"):
            lines.append("- weak material types:")
            for item in source_quality_feedback.get("weak_material_types", []):
                lines.append(f"  - {item.get('label')}: {item.get('count')}")
        if source_quality_feedback.get("preferred_material_types_found"):
            lines.append("- preferred material types found:")
            for item in source_quality_feedback.get("preferred_material_types_found", []):
                lines.append(f"  - {item.get('label')}: {item.get('count')}")
        if source_quality_feedback.get("planner_quality_guidance"):
            lines.append("- planner quality guidance:")
            for item in source_quality_feedback.get("planner_quality_guidance", []):
                lines.append(f"  - {item}")
    else:
        lines.append("- No source quality feedback yet.")

    add_section("Search surface memory")
    if search_surface_memory.get("has_memory"):
        lines.append(f"- recent run count: {search_surface_memory.get('recent_run_count')}")
        if search_surface_memory.get("avoided_surfaces"):
            lines.append(f"- avoided surfaces: {', '.join(str(item) for item in search_surface_memory.get('avoided_surfaces') or [])}")
        if search_surface_memory.get("preferred_surfaces"):
            lines.append(f"- preferred surfaces: {', '.join(str(item) for item in search_surface_memory.get('preferred_surfaces') or [])}")
        if search_surface_memory.get("underexplored_surfaces"):
            lines.append(f"- underexplored surfaces: {', '.join(str(item) for item in search_surface_memory.get('underexplored_surfaces') or [])}")
        if search_surface_memory.get("surfaces"):
            lines.append("- surfaces:")
            for item in search_surface_memory.get("surfaces", []):
                lines.append(f"  - key: {item.get('label')}")
                lines.append(f"    status: {item.get('status')}")
                lines.append(f"    reason: {item.get('reason')}")
    else:
        lines.append("- No search surface memory yet.")

    add_section("Discovery runs")
    if history_runs:
        for item in history_runs:
            run = item.get("run")
            lines.append(f"- run id: {getattr(run, 'id', '')}")
            lines.append(f"  timestamp: {item.get('completed_label') or 'вЂ”'}")
            lines.append(f"  status: {item.get('title')}")
            if item.get("subtitle"):
                lines.append(f"  note: {item.get('subtitle')}")
            if item.get("compact_metrics"):
                lines.append(f"  compact metrics: {item.get('compact_metrics')}")
            if item.get("warning_title") or item.get("warning_body"):
                lines.append(f"  provider warning: {item.get('warning_body') or item.get('warning_title')}")
            stage_diagnostics = item.get("stage_diagnostics") or {}
            if stage_diagnostics.get("rows"):
                lines.append("  stage diagnostics:")
                for metric in stage_diagnostics.get("rows", []):
                    lines.append(f"    - {metric.get('label')}: {metric.get('value')}")
            if item.get("strategy_rows"):
                lines.append("  search strategy:")
                for detail in item.get("strategy_rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
            if item.get("query_rows"):
                lines.append("  query rows:")
                for query in item.get("query_rows", []):
                    lines.append(f"    - {query.get('label')}: {query.get('value')}")
                    if query.get("query"):
                        lines.append(f"      query: {query.get('query')}")
            if item.get("technical_rows"):
                lines.append("  technical details:")
                for detail in item.get("technical_rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
            quality_feedback = item.get("quality_feedback") or {}
            if quality_feedback.get("has_feedback"):
                lines.append("  quality diagnostics:")
                for detail in quality_feedback.get("rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
                if quality_feedback.get("main_quality_issue"):
                    lines.append(f"    - Main issue: {quality_feedback.get('main_quality_issue')}")
                for weak_item in quality_feedback.get("weak_material_types", []):
                    lines.append(f"    - Weak material type: {weak_item.get('label')} ({weak_item.get('count')})")
                for preferred_item in quality_feedback.get("preferred_material_types_found", []):
                    lines.append(f"    - Preferred material type: {preferred_item.get('label')} ({preferred_item.get('count')})")
    else:
        lines.append("- No discovery runs yet.")

    add_section("Seen sources")
    if seen_source_history:
        for item in seen_source_history:
            lines.append(f"- title: {item.get('title')}")
            lines.append(f"  url: {item.get('url') or 'вЂ”'}")
            lines.append(f"  domain: {item.get('domain')}")
            lines.append(f"  status: {item.get('status_label')}")
            lines.append(f"  last outcome: {item.get('outcome_label')}")
            lines.append(f"  seen count: {item.get('seen_count')}")
            lines.append(f"  last seen: {item.get('last_seen')}")
            for detail in item.get("details", []):
                lines.append(f"  {str(detail.get('label') or '').lower()}: {detail.get('value')}")
    else:
        lines.append("- No seen sources yet.")

    add_section("Planner history guidance")
    if latest_history_summary:
        lines.append(f"- history available: {latest_history_summary.get('history_available')}")
        lines.append(f"- recent run count: {latest_history_summary.get('recent_run_count')}")
        lines.append(f"- malformed run count: {latest_history_summary.get('malformed_run_count')}")
        lines.append(f"- total query rows: {latest_history_summary.get('total_query_rows')}")
        useful_angles = latest_history_summary.get("useful_angles") or []
        weak_angles = latest_history_summary.get("weak_angles") or []
        quality_guidance = _dedupe_guidance_strings(latest_history_summary.get("quality_guidance") or [])
        planning_guidance = _dedupe_guidance_strings(
            latest_history_summary.get("planning_guidance") or [],
            skip_existing=quality_guidance,
        )
        if useful_angles:
            lines.append("- useful angles:")
            for item in useful_angles:
                lines.append(f"  - {item.get('angle')}: {item.get('count')}")
        if weak_angles:
            lines.append("- weak angles:")
            for item in weak_angles:
                lines.append(f"  - {item.get('angle')}: {item.get('count')}")
        if quality_guidance:
            lines.append("- quality guidance used for next run:")
            for item in quality_guidance:
                lines.append(f"  - {item}")
        if planning_guidance:
            lines.append("- planning guidance:")
            for item in planning_guidance:
                lines.append(f"  - {item}")
    else:
        lines.append("- No compact planner history guidance available.")

    add_section("Discovery cycle")
    latest_cycle = latest_run.diagnostics.get("discovery_cycle") if latest_run and isinstance(latest_run.diagnostics, dict) else {}
    if isinstance(latest_cycle, dict) and latest_cycle:
        lines.append(f"- target visible suggestions: {latest_cycle.get('target_visible_new_suggestions')}")
        lines.append(f"- max immediate rounds: {latest_cycle.get('max_immediate_rounds')}")
        lines.append(f"- rounds run: {latest_cycle.get('round_count')}")
        lines.append(f"- accumulated visible suggestions: {latest_cycle.get('accumulated_visible_suggestions')}")
        lines.append(f"- decision: {latest_cycle.get('decision')}")
        if str(latest_cycle.get("decision") or "").strip() not in {"", "target_reached"}:
            lines.append(f"- stop reason: {latest_cycle.get('decision')}")
        repair_plan = latest_cycle.get("repair_plan") if isinstance(latest_cycle.get("repair_plan"), dict) else {}
        if repair_plan:
            lines.append("- Strategy repair")
            lines.append(f"  - strategy: {repair_plan.get('strategy')}")
            lines.append(f"  - reason: {repair_plan.get('reason')}")
            constraints = repair_plan.get("constraints") if isinstance(repair_plan.get("constraints"), dict) else {}
            if constraints:
                lines.append("  - constraints:")
                for key in (
                    "avoid_repeating_queries",
                    "avoid_verbatim_failed_queries",
                    "avoid_duplicate_repaired_queries",
                    "avoid_near_duplicate_repaired_queries",
                    "avoid_long_natural_language_queries",
                    "prefer_compact_search_grade_queries",
                    "require_semantic_distance_from_failed_query",
                    "require_query_surface_diversity",
                ):
                    lines.append(f"    - {key}: {constraints.get(key)}")
                if constraints.get("prefer_material_types"):
                    lines.append(f"    - prefer_material_types: {', '.join(str(item) for item in constraints.get('prefer_material_types') or [])}")
                if constraints.get("avoid_material_types"):
                    lines.append(f"    - avoid_material_types: {', '.join(str(item) for item in constraints.get('avoid_material_types') or [])}")
                if constraints.get("avoid_domains"):
                    lines.append(f"    - avoid_domains: {', '.join(str(item) for item in constraints.get('avoid_domains') or [])}")
            lines.append("  - query repair plan:")
            for item in repair_plan.get("query_repair_plan") or []:
                if not isinstance(item, dict):
                    continue
                lines.append(f"    - old: {item.get('old_query')}")
                lines.append(f"      action: {item.get('action')}")
                lines.append(f"      semantic shift: {item.get('semantic_shift_type')}")
                lines.append(f"      new: {item.get('new_query')}")
                lines.append(f"      repair reason: {item.get('repair_reason')}")
                lines.append(f"      angle: {item.get('angle')}")
                lines.append(f"      material type: {item.get('material_type')}")
                if item.get("surface_key"):
                    lines.append(f"      surface key: {item.get('surface_key')}")
                if item.get("diversity_reason"):
                    lines.append(f"      diversity reason: {item.get('diversity_reason')}")
        latest_round_index = int(latest_cycle.get("round_index") or 0)
        latest_round_item = {}
        for round_item in latest_cycle.get("rounds") or []:
            if isinstance(round_item, dict) and int(round_item.get("round_index") or 0) == latest_round_index:
                latest_round_item = round_item
                break
        latest_repair_usage = latest_round_item.get("repair_plan_usage") if isinstance(latest_round_item.get("repair_plan_usage"), dict) else {}
        if latest_round_item.get("used_repair_plan"):
            lines.append("- Repair plan used")
            lines.append(f"  - source round: {latest_repair_usage.get('repair_plan_source_round')}")
            lines.append(f"  - queries used: {latest_repair_usage.get('queries_used_count') or len(latest_repair_usage.get('repair_queries_used') or [])}")
            lines.append(f"  - strategy: {latest_repair_usage.get('strategy')}")
            for item in latest_repair_usage.get("repair_queries_used") or []:
                if not isinstance(item, dict):
                    continue
                lines.append(f"  - query: {item.get('query')}")
                lines.append(f"    old query: {item.get('old_query')}")
                lines.append(f"    action: {item.get('action')}")
                lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                lines.append(f"    material type: {item.get('material_type')}")
        cycle_diagnosis = latest_cycle.get("cycle_diagnosis") if isinstance(latest_cycle.get("cycle_diagnosis"), dict) else {}
        if cycle_diagnosis:
            lines.append("- Search diagnosis")
            lines.append(f"  - primary cause: {cycle_diagnosis.get('primary_cause')}")
            secondary = cycle_diagnosis.get("secondary_causes") or []
            if secondary:
                lines.append(f"  - secondary causes: {', '.join(str(item) for item in secondary)}")
            lines.append(f"  - severity: {cycle_diagnosis.get('severity')}")
            lines.append(f"  - explanation: {cycle_diagnosis.get('explanation')}")
            lines.append(f"  - recommended next action: {cycle_diagnosis.get('recommended_next_action')}")
        for round_item in latest_cycle.get("rounds") or []:
            if not isinstance(round_item, dict):
                continue
            lines.append("")
            lines.append(f"Round {round_item.get('round_index')}")
            lines.append(f"- run id: {round_item.get('run_id')}")
            lines.append(f"- visible new suggestions: {round_item.get('visible_new_suggestions')}")
            lines.append(f"- accepted count: {round_item.get('accepted_count')}")
            lines.append(f"- quality rejected: {round_item.get('quality_rejected_count')}")
            lines.append(f"- known / duplicate: {round_item.get('known_or_duplicate_count')}")
            lines.append(f"- provider errors: {round_item.get('provider_error_count')}")
            lines.append(f"- returned count: {round_item.get('returned_count')}")
            lines.append(f"- reason summary: {round_item.get('reason_summary')}")
            if round_item.get("used_repair_plan"):
                round_usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
                lines.append("- repair plan used:")
                lines.append(f"  - source round: {round_usage.get('repair_plan_source_round')}")
                lines.append(f"  - queries used: {round_usage.get('queries_used_count') or len(round_usage.get('repair_queries_used') or [])}")
                lines.append(f"  - strategy: {round_usage.get('strategy')}")
                for item in round_usage.get("repair_queries_used") or []:
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"  - query: {item.get('query')}")
                    lines.append(f"    old query: {item.get('old_query')}")
                    lines.append(f"    action: {item.get('action')}")
                    lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                    lines.append(f"    material type: {item.get('material_type')}")
            round_repair = round_item.get("repair_plan_for_next_round") if isinstance(round_item.get("repair_plan_for_next_round"), dict) else {}
            if round_repair:
                lines.append("- strategy repair:")
                lines.append(f"  - strategy: {round_repair.get('strategy')}")
                lines.append(f"  - reason: {round_repair.get('reason')}")
                for item in round_repair.get("query_repair_plan") or []:
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"  - old: {item.get('old_query')}")
                    lines.append(f"    action: {item.get('action')}")
                    lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                    lines.append(f"    new: {item.get('new_query')}")
            round_diagnosis = round_item.get("diagnosis") if isinstance(round_item.get("diagnosis"), dict) else {}
            if round_diagnosis:
                lines.append("- diagnosis:")
                lines.append(f"  - primary cause: {round_diagnosis.get('primary_cause')}")
                secondary = round_diagnosis.get("secondary_causes") or []
                if secondary:
                    lines.append(f"  - secondary causes: {', '.join(str(item) for item in secondary)}")
                lines.append(f"  - severity: {round_diagnosis.get('severity')}")
                lines.append(f"  - explanation: {round_diagnosis.get('explanation')}")
                lines.append(f"  - recommended next action: {round_diagnosis.get('recommended_next_action')}")
    else:
        lines.append("- No discovery cycle diagnostics available yet.")

    return "\n".join(str(line) for line in lines if line is not None)


def _dedupe_guidance_strings(items, *, skip_existing=None) -> list[str]:
    normalized_items = items if isinstance(items, (list, tuple)) else []
    deduped: list[str] = []
    seen: set[str] = {
        re.sub(r"\s+", " ", str(item or "").strip()).casefold()
        for item in (skip_existing or [])
        if re.sub(r"\s+", " ", str(item or "").strip())
    }
    for item in normalized_items:
        cleaned = re.sub(r"\s+", " ", str(item or "").strip())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _humanize_surface_key_label(surface_key: str) -> str:
    value = str(surface_key or "").strip()
    if not value:
        return ""
    labels = {
        "etf_flows_report": "ETF flows",
        "etf_flow_data_market_report": "ETF flow data",
        "institutional_demand_report": "institutional demand",
        "institutional_flows_report": "institutional flows",
        "funding_open_interest_report": "funding rates / open interest",
        "funding_rates_analysis": "funding rates",
        "open_interest_futures_positioning": "open interest",
        "derivatives_positioning_market_structure": "derivatives positioning",
        "market_structure_report": "market structure",
        "market_structure_research_paper": "market structure research paper",
        "research_paper": "research paper",
        "on_chain_exchange_reserves_analysis": "on-chain exchange reserves",
        "on_chain_weekly_report": "on-chain weekly report",
        "on_chain_analysis": "on-chain analysis",
        "analyst_report": "analyst report",
        "volatility_market_structure_report": "volatility market structure",
        "volatility_drawdown_risk_analysis": "volatility drawdown risk",
    }
    return labels.get(value, value.replace("_", " "))


def _humanize_surface_status_label(status: str) -> str:
    value = str(status or "").strip().replace("_", " ")
    if not value:
        return ""
    return value.capitalize()


def _format_research_history_timestamp(run: SourceDiscoveryRun) -> str:
    timestamp = run.completed_at or run.started_at
    return _format_history_timestamp(timestamp)


def _format_history_timestamp(timestamp) -> str:
    if timestamp is None:
        return ""
    return timezone.localtime(timestamp).strftime("%Y-%m-%d %H:%M")


def _build_research_history_provider_errors(diagnostics: dict) -> list[str]:
    errors: list[str] = []
    for item in diagnostics.get("provider_errors", []) or []:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or "").strip()
        if message:
            errors.append(message)
    return errors


def _format_query_angle_label(query_angle_key: str, query_angle_suffix: str) -> str:
    if query_angle_key == "base":
        return "base"
    if query_angle_suffix:
        return query_angle_suffix
    return query_angle_key.replace("_", " ")


def _format_query_intent_label(intent: str) -> str:
    if not intent:
        return ""
    return intent.replace("_", " ")


def _format_recency_label(recency_months: int) -> str:
    if recency_months == 1:
        return "last 1 month"
    return f"last {recency_months} months"


def _format_source_history_status_label(status: str) -> str:
    mapping = {
        SourceDiscoveryHistory.STATUS_SEEN: "Seen only",
        SourceDiscoveryHistory.STATUS_SHOWN: "Shown as suggestion",
        SourceDiscoveryHistory.STATUS_KEPT: "Kept",
        SourceDiscoveryHistory.STATUS_REMOVED_BY_USER: "Removed by user",
        SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY: "Rejected by quality",
    }
    return mapping.get(str(status or "").strip(), str(status or "").strip() or "Unknown")


def _format_source_history_outcome_label(outcome: str) -> str:
    mapping = {
        SourceDiscoveryHistory.OUTCOME_NEW_SHOWN: "New suggestion",
        SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN: "Already known",
        SourceDiscoveryHistory.OUTCOME_DUPLICATE_URL: "Duplicate URL",
        SourceDiscoveryHistory.OUTCOME_DUPLICATE_DOMAIN: "Duplicate domain",
        SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED: "Previously removed",
        SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED: "Previously rejected",
        SourceDiscoveryHistory.OUTCOME_QUALITY_REJECTED: "Quality rejected",
        SourceDiscoveryHistory.OUTCOME_STALE_REJECTED: "Stale rejected",
        SourceDiscoveryHistory.OUTCOME_COMMERCIAL_REJECTED: "Commercial rejected",
        SourceDiscoveryHistory.OUTCOME_NONE: "None recorded",
    }
    return mapping.get(str(outcome or "").strip(), str(outcome or "").strip() or "None recorded")


def _format_source_history_freshness_label(freshness_status: str) -> str:
    mapping = {
        "fresh": "Fresh",
        "acceptable": "Acceptable",
        "unknown": "Unknown date",
        "stale": "Stale",
        "very_stale": "Very stale",
    }
    return mapping.get(str(freshness_status or "").strip().lower(), str(freshness_status or "").strip() or "Unknown")


def _format_research_history_status_title(status: str, *, run: SourceDiscoveryRun) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == SourceDiscoveryRun.STATUS_COMPLETED:
        if int(run.new_suggestions_count or 0) == 0:
            return "No new suggestions found"
        return "Discovery completed"
    if normalized == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Discovery partially completed"
    if normalized == SourceDiscoveryRun.STATUS_FAILED:
        return "Discovery failed"
    if normalized == SourceDiscoveryRun.STATUS_BLOCKED:
        return "Discovery did not run"
    if normalized == SourceDiscoveryRun.STATUS_STARTED:
        return "Discovery started"
    return "Discovery update"


def _build_research_history_status_subtitle(run: SourceDiscoveryRun) -> str:
    status = str(run.status or "").strip().lower()
    if status == SourceDiscoveryRun.STATUS_COMPLETED:
        if int(run.new_suggestions_count or 0) > 0:
            count = int(run.new_suggestions_count or 0)
            return f"{count} new source suggestion{'s' if count != 1 else ''} were added." if count != 1 else "1 new source suggestion was added."
        if int(run.provider_result_count or 0) == 0:
            return "No new suggestions were found."
        return "The search ran, but all usable results were already known, rejected, or duplicates."
    if status == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some provider queries returned results, but at least one query failed."
    if status == SourceDiscoveryRun.STATUS_FAILED:
        return "Provider results could not be loaded."
    if status == SourceDiscoveryRun.STATUS_BLOCKED:
        return "Research provider was unavailable, so discovery did not run."
    if status == SourceDiscoveryRun.STATUS_STARTED:
        return "Source discovery started for this topic."
    return ""


def _build_research_history_cycle_diagnosis(cycle_diagnosis: dict) -> dict:
    if not isinstance(cycle_diagnosis, dict) or not cycle_diagnosis:
        return {"has_diagnosis": False}
    secondary = [
        format_discovery_cycle_diagnosis_label(str(item).strip())
        for item in cycle_diagnosis.get("secondary_causes") or []
        if str(item).strip()
    ]
    return {
        "has_diagnosis": True,
        "primary_cause": format_discovery_cycle_diagnosis_label(str(cycle_diagnosis.get("primary_cause") or "").strip()),
        "secondary_causes": secondary,
        "severity": str(cycle_diagnosis.get("severity") or "").strip() or "unknown",
        "explanation": str(cycle_diagnosis.get("explanation") or "").strip(),
        "recommended_next_action": format_discovery_cycle_next_action_label(
            str(cycle_diagnosis.get("recommended_next_action") or "").strip()
        ),
    }


def _build_research_history_cycle_repair(repair_plan: dict) -> dict:
    if not isinstance(repair_plan, dict) or not repair_plan:
        return {"has_repair": False}
    items = [item for item in repair_plan.get("query_repair_plan") or [] if isinstance(item, dict)]
    changed_count = sum(1 for item in items if str(item.get("action") or "").strip() == "replace_query")
    recovered_failed_area_count = sum(
        1 for item in items if str(item.get("repair_reason") or "").strip().casefold().find("failed search area") >= 0
    )
    strategy = str(repair_plan.get("strategy") or "").strip()
    return {
        "has_repair": True,
        "strategy": strategy.replace("_", " ").strip().title() if strategy else "Repair plan",
        "reason": str(repair_plan.get("reason") or "").strip(),
        "rows": [
            {"label": "Changed queries", "value": str(changed_count)},
            {"label": "Kept queries", "value": str(max(len(items) - changed_count, 0))},
            {"label": "Recovered failed search areas", "value": str(recovered_failed_area_count)},
        ],
        "items": items[:3],
    }


def _build_research_history_repair_usage(round_item: dict) -> dict:
    if not isinstance(round_item, dict) or not round_item.get("used_repair_plan"):
        return {"has_used_repair": False}
    usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
    return {
        "has_used_repair": True,
        "source_round": int(usage.get("repair_plan_source_round") or 0),
        "queries_used_count": int(usage.get("queries_used_count") or len(usage.get("repair_queries_used") or [])),
        "strategy": str(usage.get("strategy") or "").replace("_", " ").strip().title() or "Repair plan",
    }


def _build_research_history_warning_title(run: SourceDiscoveryRun, provider_errors: list[str]) -> str:
    if not provider_errors:
        return ""
    if str(run.status or "").strip().lower() == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some provider queries failed."
    return "Provider results could not be fully loaded."


def _build_research_history_warning_body(run: SourceDiscoveryRun, provider_errors: list[str]) -> str:
    if not provider_errors:
        return ""
    if str(run.status or "").strip().lower() == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some searches could not be completed. Other searches still returned results."
    return "Some searches could not be completed."
