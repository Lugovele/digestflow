"""Adapters that convert raw search-provider results into candidate inputs."""

from __future__ import annotations

from typing import Sequence

from services.sources.candidates import SourceCandidateInput
from services.sources.search_provider import RawSearchResult, SearchProviderResult


def search_result_to_candidate_input(raw_result: RawSearchResult) -> SourceCandidateInput:
    provider_name = str(raw_result.provider_name or "").strip() or "unknown"
    query = str(raw_result.query or "").strip()
    intent = raw_result.intent.value if getattr(raw_result, "intent", None) else ""
    rank = int(raw_result.rank or 0)

    return SourceCandidateInput(
        url=str(raw_result.url or "").strip(),
        title=str(raw_result.title or "").strip(),
        snippet=str(raw_result.snippet or "").strip(),
        origin_reason=f"found by {provider_name} search provider for query: {query}",
        diagnostics={
            "provider_name": provider_name,
            "query": query,
            "intent": intent,
            "rank": rank,
            "raw_result_diagnostics": dict(raw_result.diagnostics or {}),
        },
    )


def search_results_to_candidate_inputs(raw_results: Sequence[RawSearchResult]) -> list[SourceCandidateInput]:
    return [search_result_to_candidate_input(raw_result) for raw_result in raw_results]


def search_provider_result_to_candidate_inputs(provider_result: SearchProviderResult) -> list[SourceCandidateInput]:
    return search_results_to_candidate_inputs(provider_result.results)
