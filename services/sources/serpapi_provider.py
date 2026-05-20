"""Thin SerpAPI adapter for the research search-provider boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from services.sources.research_queries import ResearchQueryIntent


class SearchProviderRuntimeError(RuntimeError):
    """Structured runtime failure for provider adapters."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics or {})


@dataclass(frozen=True)
class SerpApiSearchProvider:
    api_key: str
    timeout_seconds: float = 8.0

    provider_name = "serpapi"
    base_url = "https://serpapi.com/search.json"

    def search(self, query: str, *, intent: ResearchQueryIntent) -> Sequence[dict[str, Any]]:
        params = {
            "engine": "google",
            "q": str(query or "").strip(),
            "api_key": self.api_key,
            "num": 5,
        }
        request_url = f"{self.base_url}?{urlencode(params)}"
        request = Request(
            request_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "DigestFlow/1.0",
            },
        )

        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            raise SearchProviderRuntimeError(
                "SerpAPI request failed.",
                diagnostics={
                    "provider_http_status": getattr(exc, "code", None),
                    "provider_error_type": "http_error",
                    "provider_intent": intent.value,
                },
            ) from exc
        except URLError as exc:
            raise SearchProviderRuntimeError(
                "SerpAPI request failed.",
                diagnostics={
                    "provider_error_type": "url_error",
                    "provider_error_reason": str(getattr(exc, "reason", "") or "").strip(),
                    "provider_intent": intent.value,
                },
            ) from exc
        except (TimeoutError, json.JSONDecodeError, ValueError) as exc:
            error_type = "timeout" if isinstance(exc, TimeoutError) else "invalid_response"
            raise SearchProviderRuntimeError(
                "SerpAPI response could not be processed.",
                diagnostics={
                    "provider_error_type": error_type,
                    "provider_intent": intent.value,
                },
            ) from exc

        error = payload.get("error")
        if error:
            raise SearchProviderRuntimeError(
                "SerpAPI returned an API error.",
                diagnostics={
                    "provider_error_type": "api_error",
                    "provider_api_error": str(error).strip(),
                    "provider_intent": intent.value,
                },
            )

        organic_results = payload.get("organic_results")
        if not isinstance(organic_results, list):
            return ()

        mapped_results: list[dict[str, Any]] = []
        for index, item in enumerate(organic_results, start=1):
            if not isinstance(item, dict):
                continue
            url = str(item.get("link") or "").strip()
            if not url:
                continue
            mapped_results.append(
                {
                    "title": str(item.get("title") or "").strip(),
                    "url": url,
                    "snippet": str(item.get("snippet") or item.get("snippet_highlighted_words") or "").strip(),
                    "rank": int(item.get("position") or index),
                    "source": str(item.get("source") or item.get("displayed_link") or "").strip(),
                }
            )

        return mapped_results
