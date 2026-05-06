"""Structured validators for the article-based digest payload."""
from __future__ import annotations

from typing import Any


class DigestPayloadValidationError(ValueError):
    """Raised when the digest payload does not match the expected structure."""


def validate_digest_payload(payload: dict[str, Any]) -> None:
    """Validate the article-based digest payload contract."""
    required_fields = ["title", "articles"]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise DigestPayloadValidationError(f"Digest payload is missing required fields: {missing}")

    version = payload.get("version", 1)
    if not isinstance(version, int):
        raise DigestPayloadValidationError("Digest payload version must be an integer when provided.")

    if not isinstance(payload["title"], str) or not payload["title"].strip():
        raise DigestPayloadValidationError("Digest title must be a non-empty string.")

    articles = payload["articles"]
    if not isinstance(articles, list) or not articles:
        raise DigestPayloadValidationError("Digest articles must be a non-empty list.")

    for article in articles:
        if not isinstance(article, dict):
            raise DigestPayloadValidationError("Each digest article must be a JSON object.")
        if not isinstance(article.get("url"), str) or not article["url"].strip():
            raise DigestPayloadValidationError("Each digest article must include a non-empty url.")
        if not isinstance(article.get("title"), str) or not article["title"].strip():
            raise DigestPayloadValidationError("Each digest article must include a non-empty title.")
        if not isinstance(article.get("summary"), str) or not article["summary"].strip():
            raise DigestPayloadValidationError("Each digest article must include a non-empty summary.")
        if not isinstance(article.get("key_points"), list):
            raise DigestPayloadValidationError("Each digest article must include key_points as a list.")
        if not all(isinstance(item, str) and item.strip() for item in article["key_points"]):
            raise DigestPayloadValidationError("Each digest key point must be a non-empty string.")
