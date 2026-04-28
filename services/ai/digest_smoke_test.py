"""Минимальный AI smoke test для digest generation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import OpenAIClient
from services.ai.prompt_builder import build_prompt
from services.sources import get_demo_articles_for_topic


class DigestSmokeTestError(ValueError):
    """Ошибка smoke test или валидации digest payload."""


@dataclass(frozen=True)
class DigestSmokeTestResult:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    is_mock: bool
    validation_passed: bool
    provider: str
    fallback_reason: str


@dataclass(frozen=True)
class DigestGenerationPayload:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    is_mock: bool
    provider: str
    fallback_reason: str


def run_digest_smoke_test(topic_name: str) -> DigestSmokeTestResult:
    """Собрать prompt, получить ответ модели или mock и провалидировать payload."""
    articles = get_demo_articles_for_topic(topic_name)
    generation = generate_digest_payload(topic_name, articles)

    return DigestSmokeTestResult(
        prompt=generation.prompt,
        response_text=generation.response_text,
        payload=generation.payload,
        is_mock=generation.is_mock,
        validation_passed=True,
        provider=generation.provider,
        fallback_reason=generation.fallback_reason,
    )


def generate_digest_payload(topic_name: str, articles: list[dict[str, Any]]) -> DigestGenerationPayload:
    """Собрать prompt и получить digest payload через реальный AI или mock."""
    prompt = build_prompt(
        "digest/generate_digest.txt",
        topic_name=topic_name,
        articles=_format_articles_for_prompt(articles),
    )

    fallback_reason = ""
    is_mock = False
    provider = "openai"

    if _should_use_mock():
        response_text = _build_mock_response(topic_name, articles)
        is_mock = True
        provider = "mock"
        fallback_reason = "OPENAI_API_KEY не задан или содержит placeholder."
    else:
        try:
            response = OpenAIClient().generate_text(prompt=prompt, max_output_tokens=700)
            response_text = response.text.strip()
            if not response_text:
                raise DigestSmokeTestError("Модель вернула пустой ответ.")
        except Exception as exc:  # noqa: BLE001 - для явного fallback в MVP AI stage
            response_text = _build_mock_response(topic_name, articles)
            is_mock = True
            provider = "mock"
            fallback_reason = f"Fallback на mock из-за ошибки реального AI call: {exc}"

    payload = _parse_json_response(response_text)
    validate_digest_payload(payload)

    return DigestGenerationPayload(
        prompt=prompt,
        response_text=response_text,
        payload=payload,
        is_mock=is_mock,
        provider=provider,
        fallback_reason=fallback_reason,
    )


def validate_digest_payload(payload: dict[str, Any]) -> None:
    """Проверить минимальный контракт digest payload."""
    required_fields = ["title", "summary", "key_points", "sources"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise DigestSmokeTestError(f"В digest payload отсутствуют поля: {missing}")

    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise DigestSmokeTestError("Поле title должно быть непустой строкой.")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise DigestSmokeTestError("Поле summary должно быть непустой строкой.")
    if not isinstance(payload["key_points"], list) or not payload["key_points"]:
        raise DigestSmokeTestError("Поле key_points должно быть непустым списком.")
    if not all(isinstance(item, str) and item.strip() for item in payload["key_points"]):
        raise DigestSmokeTestError("Каждый элемент key_points должен быть непустой строкой.")
    if not isinstance(payload["sources"], list) or not payload["sources"]:
        raise DigestSmokeTestError("Поле sources должно быть непустым списком.")
    if not all(isinstance(item, str) and item.strip() for item in payload["sources"]):
        raise DigestSmokeTestError("Каждый элемент sources должен быть непустой строкой.")


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
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise DigestSmokeTestError(f"Ответ модели не является валидным JSON: {exc}") from exc

    if not isinstance(payload, dict):
        raise DigestSmokeTestError("Digest payload должен быть JSON-объектом.")

    return payload
