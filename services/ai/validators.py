"""Simple structured validators for AI-generated digest payloads."""
from __future__ import annotations

from typing import Any


class DigestPayloadValidationError(ValueError):
    """Raised when the digest payload does not match the expected structure."""


def validate_digest_payload(payload: dict[str, Any]) -> None:
    """Validate the minimum contract for a digest payload."""
    required_fields = ["title", "summary", "key_points", "sources"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise DigestPayloadValidationError(f"В digest payload отсутствуют поля: {missing}")

    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise DigestPayloadValidationError("Поле title должно быть непустой строкой.")
    if not isinstance(payload["summary"], str) or not payload["summary"].strip():
        raise DigestPayloadValidationError("Поле summary должно быть непустой строкой.")
    if not isinstance(payload["key_points"], list) or len(payload["key_points"]) < 3:
        raise DigestPayloadValidationError(
            "Поле key_points должно быть списком минимум из 3 элементов."
        )
    if not all(isinstance(item, str) and item.strip() for item in payload["key_points"]):
        raise DigestPayloadValidationError("Каждый элемент key_points должен быть непустой строкой.")
    if not isinstance(payload["sources"], list) or len(payload["sources"]) < 3:
        raise DigestPayloadValidationError(
            "Поле sources должно быть списком минимум из 3 элементов."
        )
    if not all(isinstance(item, str) and item.strip() for item in payload["sources"]):
        raise DigestPayloadValidationError("Каждый элемент sources должен быть непустой строкой.")
