from __future__ import annotations

from datetime import date, datetime
from typing import Any


def make_json_safe(value: Any) -> Any:
    """Convert nested values to JSON-safe primitives for JSONField storage."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [make_json_safe(item) for item in value]

    return str(value)
