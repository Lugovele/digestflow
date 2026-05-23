"""Search provider configuration and readiness diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings

from services.sources.serpapi_provider import SerpApiSearchProvider
from services.sources.search_provider import FakeSearchProvider, SearchProvider


READY_STATUS = "ready"
DISABLED_STATUS = "disabled"
MISSING_CONFIG_STATUS = "missing_config"
NOT_IMPLEMENTED_STATUS = "not_implemented"

_DISCOVERY_SOURCE_MODES = {"discovery_only", "hybrid"}
_SUPPORTED_PROVIDER_NAMES = {"fake", "serpapi"}
_PROVIDERS_REQUIRING_API_KEY = {"serpapi", "tavily", "brave", "bing"}


@dataclass(frozen=True)
class SearchProviderResolution:
    provider: SearchProvider | None
    diagnostics: dict[str, Any]


def resolve_configured_search_provider(topic=None) -> SearchProviderResolution:
    provider_enabled = bool(getattr(settings, "SEARCH_PROVIDER_ENABLED", False))
    provider_name = str(getattr(settings, "SEARCH_PROVIDER", "") or "").strip().lower()
    provider_api_key = str(getattr(settings, "SEARCH_PROVIDER_API_KEY", "") or "").strip()
    search_recency_months = get_search_recency_months()
    search_time_filter = build_search_time_filter(search_recency_months)
    research_required_for_topic = _topic_requires_research_sources(topic)

    missing_settings: list[str] = []
    status = READY_STATUS
    error = ""
    provider: SearchProvider | None = None

    if not provider_enabled:
        status = DISABLED_STATUS
        error = "Search provider is disabled."
    elif not provider_name:
        status = MISSING_CONFIG_STATUS
        missing_settings.append("SEARCH_PROVIDER")
        error = "Search provider is enabled but SEARCH_PROVIDER is not set."
    else:
        if provider_name in _PROVIDERS_REQUIRING_API_KEY and not provider_api_key:
            status = MISSING_CONFIG_STATUS
            missing_settings.append("SEARCH_PROVIDER_API_KEY")
            error = f"Search provider '{provider_name}' is missing required credentials."
        elif provider_name not in _SUPPORTED_PROVIDER_NAMES:
            status = NOT_IMPLEMENTED_STATUS
            error = f"Search provider '{provider_name}' is not implemented yet."
        elif provider_name == "fake":
            provider = FakeSearchProvider({})
        elif provider_name == "serpapi":
            provider = SerpApiSearchProvider(
                api_key=provider_api_key,
                recency_months=search_recency_months,
                time_filter=search_time_filter,
            )

    research_execution_status = "completed"
    if status != READY_STATUS:
        research_execution_status = "blocked" if research_required_for_topic else "skipped_not_required"

    diagnostics = {
        "research_required_for_topic": research_required_for_topic,
        "research_execution_status": research_execution_status,
        "search_provider_enabled": provider_enabled,
        "search_provider_name": provider_name or "unconfigured",
        "search_provider_configured": status == READY_STATUS,
        "search_provider_status": status,
        "search_provider_error": error,
        "search_provider_missing_settings": tuple(missing_settings),
        "search_recency_months": search_recency_months,
        "search_time_filter": search_time_filter,
        "provider_tbs": search_time_filter,
    }

    return SearchProviderResolution(provider=provider, diagnostics=diagnostics)


def build_explicit_search_provider_diagnostics(provider: SearchProvider, topic=None) -> dict[str, Any]:
    return {
        "research_required_for_topic": _topic_requires_research_sources(topic),
        "research_execution_status": "completed",
        "search_provider_enabled": True,
        "search_provider_name": str(getattr(provider, "provider_name", "") or "unknown"),
        "search_provider_configured": True,
        "search_provider_status": READY_STATUS,
        "search_provider_error": "",
        "search_provider_missing_settings": (),
        "search_recency_months": int(getattr(provider, "recency_months", get_search_recency_months()) or get_search_recency_months()),
        "search_time_filter": str(getattr(provider, "time_filter", "") or build_search_time_filter(get_search_recency_months())),
        "provider_tbs": str(getattr(provider, "time_filter", "") or build_search_time_filter(get_search_recency_months())),
    }


def get_search_recency_months() -> int:
    raw_value = getattr(settings, "SEARCH_RECENCY_MONTHS", 1)
    try:
        months = int(raw_value)
    except (TypeError, ValueError):
        months = 1
    return max(1, months)


def build_search_time_filter(recency_months: int) -> str:
    months = max(1, int(recency_months or 1))
    if months == 1:
        return "qdr:m"
    return f"qdr:m{months}"


def _topic_requires_research_sources(topic) -> bool:
    if topic is None:
        return True

    uses_source_discovery = getattr(topic, "uses_source_discovery", None)
    if uses_source_discovery is not None:
        return bool(uses_source_discovery)

    source_mode = str(getattr(topic, "source_mode", "") or "").strip().lower()
    if not source_mode:
        return True
    return source_mode in _DISCOVERY_SOURCE_MODES
