"""Simple structured validators for LinkedIn content package payloads."""
from __future__ import annotations

from typing import Any


class ContentPackageValidationError(ValueError):
    """Raised when the content package payload does not match the expected structure."""


def validate_content_package_payload(payload: dict[str, Any]) -> None:
    required_fields = [
        "post_text",
        "hook_variants",
        "cta_variants",
        "hashtags",
        "quality_checks",
    ]
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ContentPackageValidationError(f"В ContentPackage payload отсутствуют поля: {missing}")

    if not isinstance(payload["post_text"], str) or not payload["post_text"].strip():
        raise ContentPackageValidationError("Поле post_text должно быть непустой строкой.")
    if len(payload["post_text"]) > 1300:
        raise ContentPackageValidationError("Поле post_text превышает лимит 1300 символов.")

    _validate_string_list(payload["hook_variants"], "hook_variants", min_items=3)
    _validate_string_list(payload["cta_variants"], "cta_variants", min_items=3)
    _validate_string_list(payload["hashtags"], "hashtags", min_items=1)

    carousel_outline = payload.get("carousel_outline", [])
    if carousel_outline is None:
        raise ContentPackageValidationError("Поле carousel_outline не должно быть null.")
    if not isinstance(carousel_outline, list):
        raise ContentPackageValidationError("Поле carousel_outline должно быть списком.")

    quality_checks = payload["quality_checks"]
    if not isinstance(quality_checks, dict):
        raise ContentPackageValidationError("Поле quality_checks должно быть объектом.")

    required_quality_checks = [
        "uses_only_provided_facts",
        "has_clear_point_of_view",
        "linkedin_ready",
    ]
    missing_quality_checks = [
        field for field in required_quality_checks if field not in quality_checks
    ]
    if missing_quality_checks:
        raise ContentPackageValidationError(
            f"В quality_checks отсутствуют поля: {missing_quality_checks}"
        )
    for field in required_quality_checks:
        if not isinstance(quality_checks[field], bool):
            raise ContentPackageValidationError(
                f"Поле quality_checks.{field} должно быть boolean."
            )


def _validate_string_list(value: Any, field_name: str, min_items: int) -> None:
    if not isinstance(value, list) or len(value) < min_items:
        raise ContentPackageValidationError(
            f"Поле {field_name} должно быть списком минимум из {min_items} элементов."
        )
    if not all(isinstance(item, str) and item.strip() for item in value):
        raise ContentPackageValidationError(
            f"Каждый элемент поля {field_name} должен быть непустой строкой."
        )
