import re

from django.utils import timezone

from apps.topics.models import Topic
from services.sources.discovery_constants import (
    DISCOVERY_DECISION_PARTIAL_NO_UNUSED_SURFACES,
    DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR,
    DISCOVERY_REPAIR_PLAN_MAX_ITEMS,
)
from services.sources.research_queries import build_research_query_plan_from_repair_items
DISCOVERY_REPAIR_CONSTRAINTS = {
    "avoid_repeating_queries": True,
    "avoid_verbatim_failed_queries": True,
    "avoid_duplicate_repaired_queries": True,
    "avoid_near_duplicate_repaired_queries": True,
    "avoid_long_natural_language_queries": True,
    "prefer_compact_search_grade_queries": True,
    "do_not_retry_failed_query_verbatim": True,
    "do_not_use_only_synonym_rewrite": True,
    "require_semantic_distance_from_failed_query": True,
    "prefer_compact_search_grade_query": True,
    "require_query_surface_diversity": True,
}


def _build_next_round_repair_override(
    *,
    topic: Topic,
    round_summary: dict,
    prior_rounds: list[dict],
    query_limit: int,
):
    if _should_stop_after_zero_yield_rounds(prior_rounds):
        return None, None, DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR
    repair_plan = round_summary.get("repair_plan_for_next_round") if isinstance(round_summary.get("repair_plan_for_next_round"), dict) else {}
    if not repair_plan:
        return None, None, DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR
    source_round_index = int(round_summary.get("round_index") or 0)
    repair_queries_used, stop_decision = _select_repair_queries_for_next_round(
        repair_plan=repair_plan,
        prior_rounds=prior_rounds,
        query_limit=query_limit,
    )
    if not repair_queries_used:
        return None, None, stop_decision or DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR
    query_plan_override = build_research_query_plan_from_repair_items(
        topic,
        repair_queries_used,
        source_round_index=source_round_index,
    )
    return query_plan_override, {
        "used_repair_plan": True,
        "repair_plan_source_round": source_round_index,
        "strategy": str(repair_plan.get("strategy") or "").strip(),
        "queries_used_count": len(repair_queries_used),
        "repair_queries_used": repair_queries_used,
    }, None


def _select_repair_queries_for_next_round(
    *,
    repair_plan: dict,
    prior_rounds: list[dict],
    query_limit: int,
) -> tuple[list[dict], str | None]:
    if not isinstance(repair_plan, dict) or not repair_plan:
        return [], DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR
    selected: list[dict] = []
    seen_queries: set[str] = set()
    limit = int(query_limit or 0)
    used_repair_query_keys = _collect_used_repair_query_keys(prior_rounds)
    used_surface_keys = _collect_used_surface_keys(prior_rounds)
    used_provider_query_texts = _collect_used_provider_query_texts(prior_rounds)
    has_usable_candidate = False

    for item in repair_plan.get("query_repair_plan") or []:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip()
        old_query = str(item.get("old_query") or "").strip()
        new_query = compact_search_query(str(item.get("new_query") or "").strip())
        if not new_query or action == "drop":
            continue
        if new_query.casefold() == compact_search_query(old_query).casefold():
            continue
        has_usable_candidate = True
        query_key = _normalized_repaired_query_key(new_query)
        surface_key = str(item.get("surface_key") or "").strip()
        if query_key in seen_queries or query_key in used_repair_query_keys:
            continue
        if new_query.casefold() in used_provider_query_texts:
            continue
        if surface_key and surface_key in used_surface_keys:
            continue
        seen_queries.add(query_key)
        selected.append(
            {
                "query": new_query,
                "old_query": old_query,
                "action": action or "replace_query",
                "semantic_shift_type": str(item.get("semantic_shift_type") or "").strip(),
                "material_type": str(item.get("material_type") or "").strip(),
                "angle": str(item.get("angle") or "").strip(),
                "source": "repair_plan",
                "surface_key": surface_key,
                "diversity_reason": str(item.get("diversity_reason") or "").strip(),
                "repair_reason": str(item.get("repair_reason") or "").strip(),
            }
        )
        if limit > 0 and len(selected) >= limit:
            break
    if selected:
        return selected, None
    if has_usable_candidate:
        return [], DISCOVERY_DECISION_PARTIAL_NO_UNUSED_SURFACES
    return [], DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR


def _should_stop_after_zero_yield_rounds(prior_rounds: list[dict]) -> bool:
    if len(prior_rounds) < 2:
        return False
    recent_rounds = [item for item in prior_rounds[-2:] if isinstance(item, dict)]
    if len(recent_rounds) < 2:
        return False
    return all(
        int(item.get("returned_count") or 0) == 0
        and int(item.get("visible_new_suggestions") or 0) == 0
        for item in recent_rounds
    )


def _collect_used_repair_query_keys(rounds: list[dict]) -> set[str]:
    used_query_keys: set[str] = set()
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        repair_usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
        for item in repair_usage.get("repair_queries_used") or []:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if query:
                used_query_keys.add(_normalized_repaired_query_key(query))
    return used_query_keys


def _collect_used_surface_keys(rounds: list[dict]) -> set[str]:
    used_surface_keys: set[str] = set()
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        repair_usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
        for item in repair_usage.get("repair_queries_used") or []:
            if not isinstance(item, dict):
                continue
            surface_key = str(item.get("surface_key") or "").strip()
            if surface_key:
                used_surface_keys.add(surface_key)
    return used_surface_keys


def _collect_used_provider_query_texts(rounds: list[dict]) -> set[str]:
    used_queries: set[str] = set()
    for round_item in rounds:
        if not isinstance(round_item, dict):
            continue
        query_rows = round_item.get("query_rows")
        if not isinstance(query_rows, list):
            query_rows = _extract_repair_query_rows_from_round_summary(round_item)
        for item in query_rows or []:
            if not isinstance(item, dict):
                continue
            query = compact_search_query(str(item.get("query") or "").strip()).casefold()
            if query:
                used_queries.add(query)
    return used_queries


def _build_round_repair_plan(
    *,
    topic: Topic,
    round_result: dict,
    diagnosis: dict,
) -> dict:
    run = round_result.get("discovery_run")
    diagnostics = dict(getattr(run, "diagnostics", {}) or {})
    return _build_discovery_repair_plan(
        topic=topic,
        diagnosis=diagnosis,
        rounds=[
            {
                "run_id": getattr(run, "id", None),
                "round_index": 1,
                "returned_count": int(round_result.get("returned_count") or 0),
                "visible_new_suggestions": len(round_result.get("new_visible_candidates") or []),
                "diagnosis": diagnosis,
                "query_rows": _extract_repair_query_rows(diagnostics),
                "quality_feedback": diagnostics.get("source_quality_feedback") if isinstance(diagnostics.get("source_quality_feedback"), dict) else {},
            }
        ],
    )


def _build_discovery_repair_plan(
    *,
    topic: Topic,
    diagnosis: dict,
    rounds: list[dict],
) -> dict:
    primary_cause = str(diagnosis.get("primary_cause") or "").strip()
    strategy = _strategy_for_diagnosis(primary_cause)
    constraints = _build_repair_constraints(rounds)
    if strategy in {"stop", "stop_provider_unavailable"}:
        return {
            "strategy": strategy,
            "reason": str(diagnosis.get("explanation") or "").strip() or _reason_for_strategy(strategy),
            "query_repair_plan": [],
            "constraints": constraints,
        }

    repair_items: list[dict] = []
    used_query_keys: set[str] = set()
    used_surface_keys: set[str] = set()
    for round_item in rounds:
        round_diagnosis = round_item.get("diagnosis") if isinstance(round_item.get("diagnosis"), dict) else diagnosis
        query_rows = round_item.get("query_rows")
        if not isinstance(query_rows, list):
            query_rows = _extract_repair_query_rows_from_round_summary(round_item)
        for query_row in query_rows[:DISCOVERY_REPAIR_PLAN_MAX_ITEMS]:
            repair_item = _repair_query_from_diagnosis(
                topic=topic,
                query_row=query_row,
                diagnosis=round_diagnosis,
                constraints=constraints,
                used_query_keys=used_query_keys,
                used_surface_keys=used_surface_keys,
            )
            if repair_item is None:
                continue
            repair_items.append(repair_item)
            used_query_keys.add(_normalized_repaired_query_key(str(repair_item.get("new_query") or "")))
            surface_key = str(repair_item.get("surface_key") or "").strip()
            if surface_key:
                used_surface_keys.add(surface_key)
            if len(repair_items) >= DISCOVERY_REPAIR_PLAN_MAX_ITEMS:
                break
        if len(repair_items) >= DISCOVERY_REPAIR_PLAN_MAX_ITEMS:
            break

    return {
        "strategy": strategy,
        "reason": str(diagnosis.get("explanation") or "").strip() or _reason_for_strategy(strategy),
        "query_repair_plan": repair_items,
        "constraints": constraints,
    }


def _extract_repair_query_rows_from_round_summary(round_item: dict) -> list[dict]:
    return []


def _extract_repair_query_rows(diagnostics: dict) -> list[dict]:
    rows: list[dict] = []
    for item in diagnostics.get("query_performance") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        rows.append(
            {
                "query": query,
                "status": str(item.get("status") or "").strip(),
                "returned_count": int(item.get("returned_count") or 0),
                "accepted_count": int(item.get("accepted_count") or 0),
                "visible_new_suggestions_count": int(item.get("visible_new_suggestions_count") or 0),
            }
        )
    if rows:
        return rows
    for item in diagnostics.get("per_query_result_counts") or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        rows.append(
            {
                "query": query,
                "status": "",
                "returned_count": int(item.get("result_count") or 0),
                "accepted_count": 0,
                "visible_new_suggestions_count": 0,
            }
        )
    return rows


def _strategy_for_diagnosis(primary_cause: str) -> str:
    mapping = {
        "target_reached": "stop",
        "provider_unavailable": "stop_provider_unavailable",
        "provider_partial_error": "recover_failed_search_area",
        "zero_return": "adjacent_scope_shift",
        "over_narrow_query": "adjacent_scope_shift",
        "duplicate_heavy": "pivot_exhausted_angle",
        "quality_heavy": "pivot_to_stronger_material_types",
        "over_broad_query": "narrow_by_material_type",
        "stale_heavy": "tighten_recency_or_current_terms",
        "domain_repetition": "diversify_search_surface",
        "mixed_low_yield": "mixed_repair",
    }
    return mapping.get(primary_cause, "mixed_repair")


def _reason_for_strategy(strategy: str) -> str:
    mapping = {
        "stop": "Target reached.",
        "stop_provider_unavailable": "Provider unavailable is a technical issue, not a weak search strategy.",
        "recover_failed_search_area": "Some provider queries failed, so the next search should preserve the angle but change the search surface.",
        "adjacent_scope_shift": "The previous search surface was too narrow or returned no results, so the next search should move to an adjacent evidence layer.",
        "pivot_exhausted_angle": "The previous angle was duplicate-heavy, so the next search should pivot to a fresh adjacent angle.",
        "pivot_to_stronger_material_types": "Low-substance or weak material types dominated, so the next search should pivot to stronger evidence-rich materials.",
        "narrow_by_material_type": "Broad query phrasing attracted generic low-quality pages, so the next search should narrow by material type.",
        "tighten_recency_or_current_terms": "Stale results dominated, so the next search should use compact current-term framing.",
        "diversify_search_surface": "Repeated weak domains suggest the next search should diversify the evidence surface.",
        "mixed_repair": "The previous round underperformed for mixed reasons, so the next search should use a balanced repair mix.",
    }
    return mapping.get(strategy, "Prepare a compact adjacent search strategy.")


def _build_repair_constraints(rounds: list[dict]) -> dict:
    preferred_material_types: list[str] = []
    avoid_material_types: list[str] = []
    avoid_domains: list[str] = []
    for round_item in rounds:
        quality_feedback = round_item.get("quality_feedback") if isinstance(round_item.get("quality_feedback"), dict) else {}
        for item in quality_feedback.get("preferred_material_types_found") or []:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if label and label not in preferred_material_types:
                    preferred_material_types.append(label)
        for item in quality_feedback.get("weak_material_types") or []:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if label and label not in avoid_material_types:
                    avoid_material_types.append(label)
        for item in quality_feedback.get("weak_domains") or []:
            if isinstance(item, dict):
                domain = str(item.get("domain") or "").strip()
                if domain and domain not in avoid_domains:
                    avoid_domains.append(domain)
    return {
        **DISCOVERY_REPAIR_CONSTRAINTS,
        "prefer_material_types": preferred_material_types[:6],
        "avoid_material_types": avoid_material_types[:6],
        "avoid_domains": avoid_domains[:6],
    }


def _repair_query_from_diagnosis(
    *,
    topic: Topic,
    query_row: dict,
    diagnosis: dict,
    constraints: dict,
    used_query_keys: set[str],
    used_surface_keys: set[str],
) -> dict | None:
    old_query = str(query_row.get("query") or "").strip()
    if not old_query:
        return None
    primary_cause = str(diagnosis.get("primary_cause") or "").strip()
    strategy = _strategy_for_diagnosis(primary_cause)
    subject = _infer_repair_subject(topic, old_query)
    preferred_terms = _preferred_repair_terms(constraints)
    semantic_shift_type = choose_semantic_shift_type(primary_cause)
    action = "replace_query"
    if strategy in {"stop", "stop_provider_unavailable"}:
        return None

    candidates = _build_repair_candidates(
        subject=subject,
        old_query=old_query,
        strategy=strategy,
        preferred_terms=preferred_terms,
    )
    selected_candidate = _select_repair_candidate(
        candidates=candidates,
        old_query=old_query,
        used_query_keys=used_query_keys,
        used_surface_keys=used_surface_keys,
    )
    if selected_candidate is None:
        new_query = old_query
        repair_reason = "No deterministic repair available."
        action = "keep_query"
        semantic_shift_type = "none"
        angle = None
        material_type = None
        surface_key = ""
        diversity_reason = ""
    else:
        new_query = str(selected_candidate.get("query") or "").strip()
        angle = str(selected_candidate.get("angle") or "").strip() or None
        material_type = str(selected_candidate.get("material_type") or "").strip() or None
        repair_reason = str(selected_candidate.get("repair_reason") or "").strip() or _reason_for_strategy(strategy)
        surface_key = str(selected_candidate.get("surface_key") or "").strip()
        diversity_reason = str(selected_candidate.get("diversity_reason") or "").strip()

    compacted_query = compact_search_query(new_query)
    if compacted_query.casefold() == compact_search_query(old_query).casefold() and strategy not in {"stop", "stop_provider_unavailable"}:
        compacted_query = compact_search_query(f"{subject} {_next_repair_term(preferred_terms, old_query, fallback=['market structure report'])}")
    if strategy == "tighten_recency_or_current_terms":
        compacted_query = _ensure_compact_current_year_query(compacted_query)
    return {
        "old_query": old_query,
        "action": action,
        "semantic_shift_type": semantic_shift_type,
        "repair_reason": repair_reason,
        "new_query": compacted_query,
        "angle": angle,
        "material_type": material_type,
        "surface_key": surface_key,
        "diversity_reason": diversity_reason,
    }


def _build_repair_candidates(
    *,
    subject: str,
    old_query: str,
    strategy: str,
    preferred_terms: list[str],
) -> list[dict]:
    repair_reason = _reason_for_strategy(strategy)
    candidates: list[dict] = []

    def add_candidate(
        query: str,
        *,
        angle: str,
        material_type: str,
        surface_key: str,
        diversity_reason: str,
        reason_override: str | None = None,
    ) -> None:
        compacted = compact_search_query(query)
        if not compacted:
            return
        candidates.append(
            {
                "query": compacted,
                "angle": angle,
                "material_type": material_type,
                "surface_key": surface_key,
                "diversity_reason": diversity_reason,
                "repair_reason": reason_override or repair_reason,
            }
        )

    if _contains_query_term(old_query, "etf", "fund flows"):
        add_candidate(
            f"{subject} ETF flows weekly report",
            angle="ETF flows",
            material_type="report",
            surface_key="etf_flows_report",
            diversity_reason="Keep ETF flow intent, but shift to a compact report surface.",
            reason_override="Preserve the ETF flow angle while using a tighter report-style search surface.",
        )
        add_candidate(
            f"spot {subject} ETF fund flows analysis",
            angle="ETF flows",
            material_type="analysis",
            surface_key="etf_flows_analysis",
            diversity_reason="Use an alternate ETF flow surface if the first ETF report surface is already used.",
        )
    if _contains_query_term(old_query, "institutional", "treasury", "fund flows"):
        add_candidate(
            f"{subject} treasury holdings institutional demand",
            angle="institutional flows",
            material_type="market report",
            surface_key="institutional_demand_report",
            diversity_reason="Keep institutional intent while shifting away from reused flow surfaces.",
            reason_override="Preserve the institutional angle, but move to a compact holdings/demand surface.",
        )
        add_candidate(
            f"{subject} institutional fund flows report",
            angle="institutional flows",
            material_type="report",
            surface_key="institutional_flows_report",
            diversity_reason="Use a second institutional evidence surface when another institutional repair is already present.",
        )
    if _contains_query_term(old_query, "funding rates", "funding"):
        add_candidate(
            f"{subject} funding rates open interest report",
            angle="derivatives / market structure",
            material_type="market structure",
            surface_key="funding_open_interest_report",
            diversity_reason="Preserve the derivatives signal while tightening into a compact funding/open-interest report.",
            reason_override="Preserve the derivatives angle while moving to a compact funding/open-interest search surface.",
        )
        add_candidate(
            f"{subject} derivatives funding rates analysis",
            angle="derivatives / market structure",
            material_type="analysis",
            surface_key="funding_rates_analysis",
            diversity_reason="Use an alternate derivatives surface if the first funding-rate surface is already used.",
        )
    if _contains_query_term(old_query, "open interest", "futures"):
        add_candidate(
            f"{subject} derivatives positioning market structure",
            angle="derivatives / market structure",
            material_type="market structure",
            surface_key="derivatives_positioning_market_structure",
            diversity_reason="Shift from repeated open-interest phrasing to a broader positioning surface.",
            reason_override="Preserve the open-interest angle while shifting to a compact positioning/market-structure surface.",
        )
        add_candidate(
            f"{subject} open interest futures positioning",
            angle="derivatives / market structure",
            material_type="analysis",
            surface_key="open_interest_futures_positioning",
            diversity_reason="Use a second open-interest surface when another positioning-style repair is already present.",
        )
    if _contains_query_term(old_query, "research paper", "paper", "empirical"):
        add_candidate(
            f"{subject} market structure research paper",
            angle="research evidence",
            material_type="research paper",
            surface_key="market_structure_research_paper",
            diversity_reason="Preserve the evidence-seeking intent and move to a compact research-paper surface.",
            reason_override="Preserve the research-evidence angle while shifting to a compact paper-oriented search surface.",
        )
        add_candidate(
            f"{subject} on-chain analysis research paper",
            angle="research evidence",
            material_type="research paper",
            surface_key="on_chain_research_paper",
            diversity_reason="Use an alternate research surface if the first paper surface is already used.",
        )
    if _contains_query_term(old_query, "on-chain", "exchange reserves", "reserves"):
        add_candidate(
            f"{subject} on-chain exchange reserves analysis",
            angle="on-chain analysis",
            material_type="analysis",
            surface_key="on_chain_exchange_reserves_analysis",
            diversity_reason="Preserve the on-chain angle while tightening to a compact reserves analysis surface.",
        )
        add_candidate(
            f"{subject} on-chain analysis weekly report",
            angle="on-chain analysis",
            material_type="report",
            surface_key="on_chain_weekly_report",
            diversity_reason="Use a second on-chain surface if another on-chain repair already exists.",
        )
    if _contains_query_term(old_query, "volatility", "risk", "drawdown"):
        add_candidate(
            f"{subject} volatility market structure report",
            angle="risk / volatility",
            material_type="report",
            surface_key="volatility_market_structure_report",
            diversity_reason="Keep the volatility theme while shifting into a more evidence-rich market-structure surface.",
        )
        add_candidate(
            f"{subject} volatility drawdown risk analysis",
            angle="risk / volatility",
            material_type="analysis",
            surface_key="volatility_drawdown_risk_analysis",
            diversity_reason="Use a second volatility evidence surface if needed.",
        )
    if _contains_query_term(old_query, "market analysis", "latest analysis", "market impacts", "fresh evidence"):
        add_candidate(
            f"{subject} institutional demand market report",
            angle="institutional flows",
            material_type="market report",
            surface_key="institutional_demand_market_report",
            diversity_reason="Replace a broad market-analysis surface with a more material-rich institutional report surface.",
        )
        add_candidate(
            f"{subject} ETF flow data market report",
            angle="ETF flows",
            material_type="market data",
            surface_key="etf_flow_data_market_report",
            diversity_reason="Use a data-rich ETF surface instead of repeating generic market-analysis wording.",
        )
        add_candidate(
            f"{subject} on-chain exchange reserves analysis",
            angle="on-chain analysis",
            material_type="analysis",
            surface_key="on_chain_exchange_reserves_analysis",
            diversity_reason="Use an on-chain evidence surface instead of repeating generic market-analysis wording.",
        )

    for preferred_term in preferred_terms:
        cleaned = str(preferred_term or "").strip()
        if not cleaned:
            continue
        surface_key = _surface_key_from_term(cleaned)
        add_candidate(
            f"{subject} {cleaned}",
            angle=_angle_from_surface_key(surface_key),
            material_type=cleaned,
            surface_key=surface_key,
            diversity_reason="Use an unused preferred material-rich surface from quality guidance.",
        )

    fallback_candidates = [
        ("institutional demand market report", "institutional flows", "market report", "institutional_demand_market_report"),
        ("ETF flow data market report", "ETF flows", "market data", "etf_flow_data_market_report"),
        ("funding rates open interest report", "derivatives / market structure", "market structure", "funding_open_interest_report"),
        ("derivatives positioning market structure", "derivatives / market structure", "market structure", "derivatives_positioning_market_structure"),
        ("on-chain exchange reserves analysis", "on-chain analysis", "analysis", "on_chain_exchange_reserves_analysis"),
        ("market structure research paper", "research evidence", "research paper", "market_structure_research_paper"),
    ]
    for term, angle, material_type, surface_key in fallback_candidates:
        add_candidate(
            f"{subject} {term}",
            angle=angle,
            material_type=material_type,
            surface_key=surface_key,
            diversity_reason="Fallback to a compact adjacent surface that keeps topic relevance while diversifying the repair plan.",
        )

    return _dedupe_repair_candidates(candidates)


def _dedupe_repair_candidates(candidates: list[dict]) -> list[dict]:
    deduped: list[dict] = []
    seen_keys: set[tuple[str, str]] = set()
    for candidate in candidates:
        query = str(candidate.get("query") or "").strip()
        surface_key = str(candidate.get("surface_key") or "").strip()
        key = (_normalized_repaired_query_key(query), surface_key)
        if not query or key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(candidate)
    return deduped


def _select_repair_candidate(
    *,
    candidates: list[dict],
    old_query: str,
    used_query_keys: set[str],
    used_surface_keys: set[str],
) -> dict | None:
    compact_old_query = compact_search_query(old_query).casefold()

    for candidate in candidates:
        query = str(candidate.get("query") or "").strip()
        if not query:
            continue
        query_key = _normalized_repaired_query_key(query)
        surface_key = str(candidate.get("surface_key") or "").strip()
        if compact_search_query(query).casefold() == compact_old_query:
            continue
        if query_key in used_query_keys:
            continue
        if surface_key and surface_key in used_surface_keys:
            continue
        return candidate

    for candidate in candidates:
        query = str(candidate.get("query") or "").strip()
        if not query:
            continue
        query_key = _normalized_repaired_query_key(query)
        if compact_search_query(query).casefold() == compact_old_query or query_key in used_query_keys:
            continue
        return candidate
    return None


def _normalized_repaired_query_key(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", compact_search_query(query).casefold())
    stopwords = {
        "bitcoin",
        "latest",
        str(timezone.now().year),
        "report",
        "analysis",
        "weekly",
        "market",
    }
    filtered = sorted(token for token in tokens if token not in stopwords)
    if not filtered:
        filtered = sorted(tokens)
    return " ".join(filtered)


def _surface_key_from_term(term: str) -> str:
    lowered = str(term or "").casefold()
    if "etf" in lowered and "flow" in lowered:
        return "etf_flow_data_market_report"
    if "institutional" in lowered:
        return "institutional_demand_market_report"
    if "funding" in lowered:
        return "funding_open_interest_report"
    if "open interest" in lowered:
        return "derivatives_positioning_market_structure"
    if "research paper" in lowered or "paper" in lowered:
        return "market_structure_research_paper"
    if "on-chain" in lowered:
        return "on_chain_exchange_reserves_analysis"
    if "market structure" in lowered:
        return "derivatives_positioning_market_structure"
    return re.sub(r"[^a-z0-9]+", "_", lowered).strip("_") or "adjacent_surface"


def _angle_from_surface_key(surface_key: str) -> str:
    mapping = {
        "etf_flows_report": "ETF flows",
        "etf_flows_analysis": "ETF flows",
        "etf_flow_data_market_report": "ETF flows",
        "institutional_demand_report": "institutional flows",
        "institutional_flows_report": "institutional flows",
        "institutional_demand_market_report": "institutional flows",
        "funding_open_interest_report": "derivatives / market structure",
        "funding_rates_analysis": "derivatives / market structure",
        "derivatives_positioning_market_structure": "derivatives / market structure",
        "open_interest_futures_positioning": "derivatives / market structure",
        "market_structure_research_paper": "research evidence",
        "on_chain_research_paper": "research evidence",
        "on_chain_exchange_reserves_analysis": "on-chain analysis",
        "on_chain_weekly_report": "on-chain analysis",
        "volatility_market_structure_report": "risk / volatility",
        "volatility_drawdown_risk_analysis": "risk / volatility",
    }
    return mapping.get(surface_key, "adjacent evidence layer")


def compact_search_query(query: str) -> str:
    cleaned = re.sub(r"[\.,:;!?()\[\]{}]+", " ", str(query or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    tokens = [token for token in cleaned.split(" ") if token]
    stopwords = {
        "latest",
        "current",
        "recent",
        "what",
        "should",
        "know",
        "about",
        "for",
        "their",
        "and",
        "the",
        "this",
        "month",
        "in",
        "of",
        "different",
        "effectiveness",
        "landscape",
        "investing",
    }
    compacted: list[str] = []
    for token in tokens:
        normalized = token.casefold()
        if normalized in stopwords and len(tokens) > 6:
            continue
        compacted.append(token)
    if len(compacted) > 8:
        compacted = compacted[:8]
    return " ".join(compacted)


def _ensure_compact_current_year_query(query: str) -> str:
    compacted = compact_search_query(query)
    current_year = str(timezone.now().year)
    tokens = [token for token in compacted.split(" ") if token]
    if any(token == current_year for token in tokens):
        return compacted
    if len(tokens) >= 8:
        tokens = tokens[:7]
    tokens.append(current_year)
    return " ".join(tokens)


def choose_semantic_shift_type(primary_cause: str) -> str:
    mapping = {
        "provider_partial_error": "query_compression",
        "zero_return": "adjacent_angle_shift",
        "over_narrow_query": "adjacent_angle_shift",
        "duplicate_heavy": "adjacent_angle_shift",
        "quality_heavy": "material_type_shift",
        "over_broad_query": "material_type_shift",
        "stale_heavy": "timeframe_shift",
        "domain_repetition": "evidence_layer_shift",
        "mixed_low_yield": "evidence_layer_shift",
        "target_reached": "none",
        "provider_unavailable": "none",
    }
    return mapping.get(primary_cause, "adjacent_angle_shift")


def _infer_repair_subject(topic: Topic, old_query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", old_query)
    for token in tokens:
        if token and token[0].isupper() and len(token) > 2:
            return token
    keyword_tokens = re.findall(r"[A-Za-z0-9]+", " ".join(topic.keywords or []))
    for token in keyword_tokens:
        if len(token) > 2:
            return token.title()
    topic_tokens = re.findall(r"[A-Za-z0-9]+", topic.name)
    for token in topic_tokens:
        if len(token) > 2:
            return token.title()
    return "Topic"


def _preferred_repair_terms(constraints: dict) -> list[str]:
    terms: list[str] = []
    mapping = {
        "market data / flow analysis": "ETF flows weekly report",
        "research paper": "research paper",
        "on-chain analysis": "on-chain analysis report",
        "market structure analysis": "market structure report",
        "institutional / analyst report": "institutional demand report",
        "analyst report": "analyst report",
    }
    for item in constraints.get("prefer_material_types") or []:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        terms.append(mapping.get(cleaned.casefold(), cleaned))
    defaults = [
        "ETF flows weekly report",
        "institutional flows report",
        "funding rates report",
        "open interest market structure",
        "on-chain analysis report",
        "research paper",
    ]
    for item in defaults:
        if item not in terms:
            terms.append(item)
    return terms


def _next_repair_term(preferred_terms: list[str], old_query: str, *, fallback: list[str]) -> str:
    lowered_query = str(old_query or "").casefold()
    for term in preferred_terms:
        if term.casefold() not in lowered_query:
            return term
    for term in fallback:
        if term.casefold() not in lowered_query:
            return term
    return fallback[0]


def _contains_query_term(query: str, *needles: str) -> bool:
    lowered = str(query or "").casefold()
    return any(needle.casefold() in lowered for needle in needles)
