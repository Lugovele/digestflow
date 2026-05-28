"""Pure orchestration for the dry source-research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from services.sources.candidate_review import CandidateReviewItem, build_candidate_review_items
from services.sources.candidates import (
    EvaluatedSourceCandidate,
    SourceCandidateInput,
    SourceCandidateStatus,
    evaluate_source_candidates,
)
from services.sources.research_queries import ResearchQueryPlan, build_research_query_plan
from services.sources.search_config import (
    build_explicit_search_provider_diagnostics,
    resolve_configured_search_provider,
)
from services.sources.search_candidates import search_provider_result_to_candidate_inputs
from services.sources.search_provider import SearchProvider, SearchProviderResult, search_research_query_plan


@dataclass(frozen=True)
class SourceResearchResult:
    query_plan: ResearchQueryPlan
    provider_result: SearchProviderResult
    candidate_inputs: tuple[SourceCandidateInput, ...]
    evaluated_candidates: tuple[EvaluatedSourceCandidate, ...]
    review_items: tuple[CandidateReviewItem, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)


def run_source_research(topic, provider: SearchProvider | None = None) -> SourceResearchResult:
    query_plan = build_research_query_plan(topic)
    if provider is None:
        provider_resolution = resolve_configured_search_provider(topic)
        if provider_resolution.provider is None:
            return _build_source_research_result(
                query_plan=query_plan,
                provider_result=SearchProviderResult(
                    provider_name=str(provider_resolution.diagnostics.get("search_provider_name") or "unconfigured"),
                    results=(),
                    diagnostics=dict(provider_resolution.diagnostics),
                ),
            )
        provider = provider_resolution.provider
        provider_diagnostics = dict(provider_resolution.diagnostics)
    else:
        provider_diagnostics = build_explicit_search_provider_diagnostics(provider, topic)

    provider_result = search_research_query_plan(query_plan, provider)
    provider_result = SearchProviderResult(
        provider_name=provider_result.provider_name,
        results=provider_result.results,
        diagnostics={
            **provider_result.diagnostics,
            **provider_diagnostics,
        },
    )
    return _build_source_research_result(query_plan=query_plan, provider_result=provider_result)


def _build_source_research_result(
    *,
    query_plan: ResearchQueryPlan,
    provider_result: SearchProviderResult,
) -> SourceResearchResult:
    candidate_inputs = tuple(search_provider_result_to_candidate_inputs(provider_result))
    evaluated_candidates = tuple(
        evaluate_source_candidates(
            candidate_inputs,
            topic=query_plan.topic_name,
            focus_terms=query_plan.topic_keywords,
        )
    )
    review_items = tuple(build_candidate_review_items(evaluated_candidates))

    accepted_candidate_count = sum(1 for candidate in evaluated_candidates if candidate.status == SourceCandidateStatus.ACCEPTED)
    needs_review_candidate_count = sum(
        1 for candidate in evaluated_candidates if candidate.status == SourceCandidateStatus.NEEDS_REVIEW
    )
    non_accepted_candidate_count = len(evaluated_candidates) - accepted_candidate_count
    rejected_candidate_count = sum(
        1
        for candidate in evaluated_candidates
        if candidate.status
        not in {
            SourceCandidateStatus.ACCEPTED,
            SourceCandidateStatus.NEEDS_REVIEW,
        }
    )
    selectable_review_item_count = sum(1 for item in review_items if item.is_selectable)

    diagnostics = {
        "query_count": len(query_plan.query_items),
        "raw_result_count": len(provider_result.results),
        "candidate_input_count": len(candidate_inputs),
        "evaluated_candidate_count": len(evaluated_candidates),
        "review_item_count": len(review_items),
        "accepted_candidate_count": accepted_candidate_count,
        "needs_review_candidate_count": needs_review_candidate_count,
        "rejected_candidate_count": rejected_candidate_count,
        "non_accepted_candidate_count": non_accepted_candidate_count,
        "selectable_review_item_count": selectable_review_item_count,
        "provider_name": provider_result.provider_name,
        "topic_domain": query_plan.topic_domain,
        "query_performance": _build_query_performance_diagnostics(
            query_plan=query_plan,
            provider_result=provider_result,
            evaluated_candidates=evaluated_candidates,
        ),
        **query_plan.diagnostics,
        **provider_result.diagnostics,
    }

    return SourceResearchResult(
        query_plan=query_plan,
        provider_result=provider_result,
        candidate_inputs=candidate_inputs,
        evaluated_candidates=evaluated_candidates,
        review_items=review_items,
        diagnostics=diagnostics,
    )


def _build_query_performance_diagnostics(
    *,
    query_plan: ResearchQueryPlan,
    provider_result: SearchProviderResult,
    evaluated_candidates: tuple[EvaluatedSourceCandidate, ...],
) -> list[dict[str, Any]]:
    performance_by_query: dict[str, dict[str, Any]] = {}

    for index, query_item in enumerate(query_plan.query_items):
        query = str(query_item.query or "").strip()
        if not query:
            continue
        performance_by_query[query] = {
            "query": query,
            "intent": query_item.intent.value,
            "provider": provider_result.provider_name,
            "angle": str((query_item.diagnostics or {}).get("query_angle_suffix") or "").strip(),
            "purpose": str(query_item.reason or "").strip(),
            "returned_count": 0,
            "accepted_count": 0,
            "rejected_count": 0,
            "duplicate_count": 0,
            "visible_new_suggestions_count": 0,
            "status": "no_visible_results",
            "query_index": index,
        }

    for item in provider_result.diagnostics.get("per_query_result_counts", []) or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        row = performance_by_query.setdefault(
            query,
            {
                "query": query,
                "intent": str(item.get("intent") or "").strip(),
                "provider": str(item.get("provider_name") or provider_result.provider_name or "").strip(),
                "angle": str(item.get("angle") or "").strip(),
                "purpose": str(item.get("purpose") or item.get("query_reason") or "").strip(),
                "returned_count": 0,
                "accepted_count": 0,
                "rejected_count": 0,
                "duplicate_count": 0,
                "visible_new_suggestions_count": 0,
                "status": "no_visible_results",
                "query_index": len(performance_by_query),
            },
        )
        row["returned_count"] = int(item.get("result_count") or 0)
        row["duplicate_count"] = int(item.get("duplicate_url_count") or 0)
        if str(item.get("provider_name") or "").strip():
            row["provider"] = str(item.get("provider_name") or "").strip()
        if str(item.get("angle") or "").strip():
            row["angle"] = str(item.get("angle") or "").strip()
        if str(item.get("purpose") or item.get("query_reason") or "").strip():
            row["purpose"] = str(item.get("purpose") or item.get("query_reason") or "").strip()
        if str(item.get("error") or "").strip():
            row["status"] = "partial_error"
            row["error_message"] = str(item.get("error") or "").strip()

    for candidate in evaluated_candidates:
        query = str(candidate.diagnostics.get("query") or "").strip()
        if not query:
            continue
        row = performance_by_query.get(query)
        if row is None:
            continue
        if candidate.status == SourceCandidateStatus.ACCEPTED:
            row["accepted_count"] = int(row["accepted_count"]) + 1
        elif candidate.status not in {SourceCandidateStatus.NEEDS_REVIEW}:
            row["rejected_count"] = int(row["rejected_count"]) + 1

    query_performance = sorted(
        performance_by_query.values(),
        key=lambda item: int(item.get("query_index") or 0),
    )
    for item in query_performance:
        item["status"] = _derive_query_performance_status(item)
        item.pop("query_index", None)
    return query_performance


def _derive_query_performance_status(item: dict[str, Any]) -> str:
    if str(item.get("error_message") or "").strip() or str(item.get("status") or "").strip() == "partial_error":
        return "partial_error"
    if int(item.get("visible_new_suggestions_count") or 0) > 0 or int(item.get("accepted_count") or 0) > 0:
        return "useful"
    if int(item.get("duplicate_count") or 0) > 0:
        return "duplicate_heavy"
    if int(item.get("returned_count") or 0) == 0:
        return "no_visible_results"
    if int(item.get("rejected_count") or 0) > 0:
        return "weak"
    return "no_visible_results"
