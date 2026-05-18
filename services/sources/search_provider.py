"""Pure search-provider boundary for deterministic research query execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from services.sources.research_queries import ResearchQueryIntent, ResearchQueryPlan


@dataclass(frozen=True)
class RawSearchResult:
    query: str
    title: str
    url: str
    snippet: str
    rank: int
    provider_name: str
    intent: ResearchQueryIntent
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchProviderResult:
    provider_name: str
    results: tuple[RawSearchResult, ...]
    diagnostics: dict[str, Any]


class SearchProvider(Protocol):
    provider_name: str

    def search(self, query: str, *, intent: ResearchQueryIntent) -> Sequence[Mapping[str, Any]]:
        ...


class FakeSearchProvider:
    provider_name = "fake"

    def __init__(self, responses: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        self._responses = {str(query): list(items) for query, items in responses.items()}

    def search(self, query: str, *, intent: ResearchQueryIntent) -> Sequence[Mapping[str, Any]]:
        return list(self._responses.get(str(query), ()))


def search_research_query_plan(plan: ResearchQueryPlan, provider: SearchProvider) -> SearchProviderResult:
    results: list[RawSearchResult] = []
    seen_urls: set[str] = set()
    per_query_counts: list[dict[str, Any]] = []
    duplicate_url_count = 0
    skipped_queries = 0

    for query_item in plan.query_items:
        query = str(query_item.query or "").strip()
        if not query:
            skipped_queries += 1
            per_query_counts.append(
                {
                    "query": query,
                    "intent": query_item.intent.value,
                    "result_count": 0,
                    "skipped": True,
                }
            )
            continue

        provider_results = list(provider.search(query, intent=query_item.intent) or ())
        emitted_count = 0
        for index, item in enumerate(provider_results, start=1):
            url = str(item.get("url") or "").strip()
            if not url:
                continue
            if url in seen_urls:
                duplicate_url_count += 1
                continue
            seen_urls.add(url)
            emitted_count += 1
            results.append(
                RawSearchResult(
                    query=query,
                    title=str(item.get("title") or "").strip(),
                    url=url,
                    snippet=str(item.get("snippet") or "").strip(),
                    rank=int(item.get("rank") or index),
                    provider_name=str(getattr(provider, "provider_name", "") or "unknown"),
                    intent=query_item.intent,
                    diagnostics={
                        "query_reason": query_item.reason,
                        "source_type_hint": query_item.source_type_hint,
                        "query_diagnostics": dict(query_item.diagnostics or {}),
                        "provider_rank": int(item.get("rank") or index),
                    },
                )
            )

        per_query_counts.append(
            {
                "query": query,
                "intent": query_item.intent.value,
                "result_count": emitted_count,
                "skipped": False,
            }
        )

    diagnostics = {
        "query_count": len(plan.query_items),
        "raw_result_count": len(results),
        "per_query_result_counts": per_query_counts,
        "skipped_query_count": skipped_queries,
        "duplicate_url_count": duplicate_url_count,
        "topic_domain": plan.topic_domain,
    }

    return SearchProviderResult(
        provider_name=str(getattr(provider, "provider_name", "") or "unknown"),
        results=tuple(results),
        diagnostics=diagnostics,
    )
