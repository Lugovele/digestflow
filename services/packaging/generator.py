"""First working LinkedIn packaging stage for the MVP."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from django.conf import settings
from django.db import transaction

from apps.ai.client import OpenAIClient, estimate_cost_usd
from apps.digests.models import Digest
from apps.packaging.models import ContentPackage
from services.ai import build_prompt


class PackagingValidationError(ValueError):
    """Structured validation error for ContentPackage payloads."""


@dataclass(frozen=True)
class PackagingGenerationResult:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    provider: str
    is_mock: bool
    fallback_reason: str
    tokens: dict[str, int | None] | None
    estimated_cost_usd: float | None


def generate_content_package_for_digest(digest: Digest) -> tuple[ContentPackage, dict[str, Any]]:
    """Generate and save a ContentPackage for a ready Digest."""
    _debug(digest.run.id, "INFO", f"digest loaded -> {digest.id}")
    _debug(digest.run.id, "INFO", f"digest title -> {digest.title}")

    generation = _generate_packaging_payload(digest)

    _debug(digest.run.id, "INFO", f"provider -> {generation.provider}")
    _debug(digest.run.id, "INFO", f"is_mock -> {generation.is_mock}")
    if generation.fallback_reason:
        _debug(digest.run.id, "INFO", f"fallback_reason -> {generation.fallback_reason}")
    if generation.tokens and generation.tokens.get("total_tokens") is not None:
        _debug(digest.run.id, "INFO", f"tokens -> total: {generation.tokens['total_tokens']}")
    if generation.estimated_cost_usd is not None:
        _debug(digest.run.id, "INFO", f"estimated cost -> ${generation.estimated_cost_usd:.6f}")

    payload = generation.payload
    validation_report = _build_validation_report(payload)

    with transaction.atomic():
        ContentPackage.objects.filter(digest=digest).delete()
        content_package = ContentPackage.objects.create(
            digest=digest,
            post_text=payload["post_text"],
            hook_variants=payload["hook_variants"],
            cta_variants=payload["cta_variants"],
            hashtags=payload["hashtags"],
            carousel_outline=payload.get("carousel_outline", []),
            validation_report=validation_report,
        )

    _debug(digest.run.id, "OK", f"package saved -> {content_package.id}")

    debug_info = {
        "prompt": generation.prompt,
        "response_text": generation.response_text,
        "provider": generation.provider,
        "is_mock": generation.is_mock,
        "fallback_reason": generation.fallback_reason,
        "validation_report": validation_report,
        "tokens": generation.tokens,
        "estimated_cost_usd": generation.estimated_cost_usd,
    }
    return content_package, debug_info


def _generate_packaging_payload(digest: Digest) -> PackagingGenerationResult:
    prompt = build_prompt(
        "linkedin/generate_post.txt",
        topic_name=digest.run.topic.name,
        digest_title=digest.title,
        digest_summary=digest.summary,
        key_points=_format_list_for_prompt(digest.key_points),
        sources=_format_list_for_prompt(digest.sources),
    )

    fallback_reason = ""
    provider = "openai"
    is_mock = False
    tokens: dict[str, int | None] | None = None
    estimated_cost: float | None = None

    if _should_use_mock():
        response_text = _build_mock_response(digest)
        provider = "mock"
        is_mock = True
        fallback_reason = "OPENAI_API_KEY не задан или содержит placeholder."
    else:
        try:
            response = OpenAIClient().generate_text(
                prompt=prompt,
                max_output_tokens=900,
                json_mode=True,
            )
            response_text = response.text.strip()
            if not response_text:
                raise PackagingValidationError("Модель вернула пустой ответ для packaging stage.")
            payload = _parse_json_response(response_text)
            validate_content_package_payload(payload)
            tokens = response.usage
            estimated_cost = estimate_cost_usd(
                response.usage.get("prompt_tokens"),
                response.usage.get("completion_tokens"),
            )
            return PackagingGenerationResult(
                prompt=prompt,
                response_text=response_text,
                payload=payload,
                provider=provider,
                is_mock=is_mock,
                fallback_reason=fallback_reason,
                tokens=tokens,
                estimated_cost_usd=estimated_cost,
            )
        except Exception as exc:  # noqa: BLE001 - explicit fallback for the MVP stage
            raw_response_text = locals().get("response_text", "")
            response_text = _build_mock_response(digest)
            provider = "mock"
            is_mock = True
            fallback_reason = (
                "Fallback на mock из-за ошибки реального AI call или невалидного JSON: "
                f"{exc}. Raw response: {raw_response_text or '<empty>'}"
            )

    payload = _parse_json_response(response_text)
    validate_content_package_payload(payload)

    return PackagingGenerationResult(
        prompt=prompt,
        response_text=response_text,
        payload=payload,
        provider=provider,
        is_mock=is_mock,
        fallback_reason=fallback_reason,
        tokens=tokens,
        estimated_cost_usd=estimated_cost,
    )


def validate_content_package_payload(payload: dict[str, Any]) -> None:
    required_fields = ["post_text", "hook_variants", "cta_variants", "hashtags"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise PackagingValidationError(f"В ContentPackage payload отсутствуют поля: {missing}")

    if not isinstance(payload["post_text"], str) or not payload["post_text"].strip():
        raise PackagingValidationError("Поле post_text должно быть непустой строкой.")
    if len(payload["post_text"]) > 1300:
        raise PackagingValidationError("Поле post_text превышает лимит 1300 символов.")

    _validate_string_list(payload["hook_variants"], "hook_variants", min_items=3)
    _validate_string_list(payload["cta_variants"], "cta_variants", min_items=3)
    _validate_string_list(payload["hashtags"], "hashtags", min_items=1)

    carousel_outline = payload.get("carousel_outline", [])
    if carousel_outline is None:
        raise PackagingValidationError("Поле carousel_outline не должно быть null.")
    if not isinstance(carousel_outline, list):
        raise PackagingValidationError("Поле carousel_outline должно быть списком.")


def _validate_string_list(value: Any, field_name: str, min_items: int) -> None:
    if not isinstance(value, list) or len(value) < min_items:
        raise PackagingValidationError(
            f"Поле {field_name} должно быть списком минимум из {min_items} элементов."
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise PackagingValidationError(
            f"Каждый элемент поля {field_name} должен быть непустой строкой."
        )


def _build_validation_report(payload: dict[str, Any]) -> dict[str, Any]:
    quality_checks = payload.get("quality_checks", {})
    return {
        "status": "valid",
        "post_text_length": len(payload["post_text"]),
        "hook_variants_count": len(payload["hook_variants"]),
        "cta_variants_count": len(payload["cta_variants"]),
        "hashtags_count": len(payload["hashtags"]),
        "carousel_outline_count": len(payload.get("carousel_outline", [])),
        "quality_checks": quality_checks,
    }


def _format_list_for_prompt(items: list[Any]) -> str:
    return json.dumps(items, ensure_ascii=False, indent=2)


def _parse_json_response(response_text: str) -> dict[str, Any]:
    normalized = _extract_json_candidate(response_text)
    if not normalized:
        raise PackagingValidationError("Ответ модели пустой или не содержит JSON-объект.")

    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError as exc:
        raise PackagingValidationError(f"Ответ модели не является валидным JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise PackagingValidationError("ContentPackage payload должен быть JSON-объектом.")
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


def _should_use_mock() -> bool:
    api_key = settings.OPENAI_API_KEY.strip()
    return not api_key or api_key == "sk-your-key"


def _build_mock_response(digest: Digest) -> str:
    payload = {
        "post_text": (
            f"{digest.title}\n\n"
            f"{digest.summary}\n\n"
            "Three signals stood out to me:\n"
            + "\n".join(f"- {point}" for point in digest.key_points[:3])
        )[:1250],
        "hook_variants": [
            f"What changed in {digest.run.topic.name} this week?",
            f"Three practical signals from {digest.run.topic.name}.",
            f"If you follow {digest.run.topic.name}, watch these shifts.",
        ],
        "cta_variants": [
            "What would you add to this view?",
            "Which signal matters most for your team?",
            "Follow for more practical digests.",
        ],
        "hashtags": ["#LinkedIn", "#AI", "#ProductStrategy"],
        "carousel_outline": [
            {
                "slide": 1,
                "title": digest.title,
                "bullets": digest.key_points[:3],
            }
        ],
        "quality_checks": {
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
            "linkedin_ready": True,
        },
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _debug(run_id: int, level: str, message: str) -> None:
    print(f"[DigestRun {run_id}] {level}: {message}")
