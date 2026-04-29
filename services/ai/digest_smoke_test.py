"""Minimal AI smoke test helpers for digest generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import OpenAIClient, estimate_cost_usd
from services.ai.prompt_builder import build_prompt
from services.ai.validators import DigestPayloadValidationError, validate_digest_payload
from services.sources import get_demo_articles_for_topic


class DigestSmokeTestError(DigestPayloadValidationError):
    """Structured validation or smoke test error for digest payloads."""


@dataclass(frozen=True)
class DigestSmokeTestResult:
    prompt: str
    response_text: str
    payload: dict[str, Any] | None
    is_mock: bool
    validation_passed: bool
    provider: str
    fallback_reason: str
    error_message: str


@dataclass(frozen=True)
class DigestGenerationPayload:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    is_mock: bool
    provider: str
    fallback_reason: str
    tokens: dict[str, int | None] | None
    estimated_cost_usd: float | None


def run_digest_smoke_test(topic_name: str) -> DigestSmokeTestResult:
    """Build a prompt, get a real or mock response, and validate the payload."""
    articles = get_demo_articles_for_topic(topic_name)
    try:
        generation = generate_digest_payload(topic_name, articles)
        return DigestSmokeTestResult(
            prompt=generation.prompt,
            response_text=generation.response_text,
            payload=generation.payload,
            is_mock=generation.is_mock,
            validation_passed=True,
            provider=generation.provider,
            fallback_reason=generation.fallback_reason,
            error_message="",
        )
    except Exception as exc:  # noqa: BLE001 - smoke test should finish in a controlled way
        prompt = build_prompt(
            "digest/generate_digest.txt",
            topic_name=topic_name,
            articles=_format_articles_for_prompt(articles),
        )
        return DigestSmokeTestResult(
            prompt=prompt,
            response_text="",
            payload=None,
            is_mock=False,
            validation_passed=False,
            provider="failed",
            fallback_reason="",
            error_message=str(exc),
        )


def generate_digest_payload(topic_name: str, articles: list[dict[str, Any]]) -> DigestGenerationPayload:
    """Build a prompt and return digest payload via real AI or deterministic mock."""
    prompt = build_prompt(
        "digest/generate_digest.txt",
        topic_name=topic_name,
        articles=_format_articles_for_prompt(articles),
    )

    fallback_reason = ""
    is_mock = False
    provider = "openai"
    tokens: dict[str, int | None] | None = None
    estimated_cost: float | None = None

    if _should_use_mock():
        response_text = _build_mock_response(topic_name, articles)
        is_mock = True
        provider = "mock"
        fallback_reason = "OPENAI_API_KEY не задан или содержит placeholder."
    else:
        try:
            response = OpenAIClient().generate_text(
                prompt=prompt,
                max_output_tokens=700,
                json_mode=True,
            )
            response_text = response.text.strip()
            if not response_text:
                raise DigestSmokeTestError("Модель вернула пустой ответ.")
            payload = _parse_json_response(response_text)
            validate_digest_payload(payload)
            tokens = response.usage
            estimated_cost = estimate_cost_usd(
                response.usage.get("prompt_tokens"),
                response.usage.get("completion_tokens"),
            )
            return DigestGenerationPayload(
                prompt=prompt,
                response_text=response_text,
                payload=payload,
                is_mock=is_mock,
                provider=provider,
                fallback_reason=fallback_reason,
                tokens=tokens,
                estimated_cost_usd=estimated_cost,
            )
        except Exception as exc:  # noqa: BLE001 - controlled fallback for smoke test and digest stage
            raw_response_text = locals().get("response_text", "")
            response_text = _build_mock_response(topic_name, articles)
            is_mock = True
            provider = "mock"
            fallback_reason = (
                "Fallback на mock из-за ошибки реального AI call или невалидного JSON: "
                f"{exc}. Raw response: {raw_response_text or '<empty>'}"
            )

    payload = _parse_json_response(response_text)
    validate_digest_payload(payload)

    return DigestGenerationPayload(
        prompt=prompt,
        response_text=response_text,
        payload=payload,
        is_mock=is_mock,
        provider=provider,
        fallback_reason=fallback_reason,
        tokens=tokens,
        estimated_cost_usd=estimated_cost,
    )

def _should_use_mock() -> bool:
    api_key = settings.OPENAI_API_KEY.strip()
    return not api_key or api_key == "sk-your-key"


def _format_articles_for_prompt(articles: list[dict[str, Any]]) -> str:
    lines = []
    for index, article in enumerate(articles, start=1):
        lines.append(
            f"{index}. {article['title']} | source={article['source']} | "
            f"url={article['url']} | snippet={article['snippet']}"
        )
    return "\n".join(lines)


def _build_mock_response(topic_name: str, articles: list[dict[str, Any]]) -> str:
    payload = {
        "title": f"Digest for {topic_name}",
        "summary": f"Mock digest summary for {topic_name} based on {len(articles)} demo articles.",
        "key_points": [article["title"] for article in articles[:3]],
        "sources": [article["url"] for article in articles[:3]],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_json_response(response_text: str) -> dict[str, Any]:
    normalized = _extract_json_candidate(response_text)
    if not normalized:
        raise DigestSmokeTestError("Ответ модели пустой или не содержит JSON-объект.")

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise DigestSmokeTestError(f"Ответ модели не является валидным JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DigestSmokeTestError("Digest payload должен быть JSON-объектом.")

    return payload


def _extract_json_candidate(response_text: str) -> str:
    text = response_text.strip()
    if not text:
        return ""

    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace == -1 or last_brace == -1 or last_brace <= first_brace:
        return ""
    return text[first_brace : last_brace + 1]
