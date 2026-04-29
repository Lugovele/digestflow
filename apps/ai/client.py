"""Небольшой OpenAI-адаптер для сервисного слоя."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings
from openai import OpenAI


@dataclass(frozen=True)
class AIResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, int | None]


INPUT_COST_PER_1K_TOKENS = 0.005
OUTPUT_COST_PER_1K_TOKENS = 0.015


class OpenAIClient:
    """Минимальный синхронный клиент с явным timeout и выбором модели."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        self.model = model or settings.OPENAI_MODEL
        self.client = OpenAI(
            api_key=api_key or settings.OPENAI_API_KEY,
            timeout=settings.OPENAI_TIMEOUT_SECONDS,
        )

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = 1200,
        json_mode: bool = False,
    ) -> AIResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "max_output_tokens": max_output_tokens,
        }
        if json_mode:
            request_kwargs["text"] = {"format": {"type": "json_object"}}

        try:
            response = self.client.responses.create(**request_kwargs)
        except Exception:
            if not json_mode:
                raise
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )

        raw = response.model_dump()
        text = response.output_text
        return AIResponse(text=text, raw=raw, usage=_extract_usage(response, raw))


def estimate_cost_usd(prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = (prompt_tokens / 1000) * INPUT_COST_PER_1K_TOKENS
    output_cost = (completion_tokens / 1000) * OUTPUT_COST_PER_1K_TOKENS
    return round(input_cost + output_cost, 6)


def _extract_usage(response: Any, raw: dict[str, Any]) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    if prompt_tokens is None:
        prompt_tokens = raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens")
    if completion_tokens is None:
        completion_tokens = raw_usage.get("output_tokens") or raw_usage.get("completion_tokens")
    if total_tokens is None:
        total_tokens = raw_usage.get("total_tokens")

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
