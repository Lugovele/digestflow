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


def run_source_research(topic, provider: SearchProvider) -> SourceResearchResult:
    query_plan = build_research_query_plan(topic)
    provider_result = search_research_query_plan(query_plan, provider)
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
    }

    return SourceResearchResult(
        query_plan=query_plan,
        provider_result=provider_result,
        candidate_inputs=candidate_inputs,
        evaluated_candidates=evaluated_candidates,
        review_items=review_items,
        diagnostics=diagnostics,
    )
