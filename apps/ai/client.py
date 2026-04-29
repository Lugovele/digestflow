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

        text = response.output_text
        return AIResponse(text=text, raw=response.model_dump())
