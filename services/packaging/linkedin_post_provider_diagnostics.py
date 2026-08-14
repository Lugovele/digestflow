"""Safe provider diagnostics for PostFlow prompt execution boundaries."""
from __future__ import annotations

from typing import Any


PROVIDER_ERROR_INVALID_REQUEST = "INVALID_REQUEST"
PROVIDER_ERROR_UNSUPPORTED_PARAMETER = "UNSUPPORTED_PARAMETER"
PROVIDER_ERROR_AUTHENTICATION = "AUTHENTICATION"
PROVIDER_ERROR_PERMISSION = "PERMISSION"
PROVIDER_ERROR_RATE_LIMIT = "RATE_LIMIT"
PROVIDER_ERROR_QUOTA = "QUOTA"
PROVIDER_ERROR_MODEL_NOT_FOUND = "MODEL_NOT_FOUND"
PROVIDER_ERROR_CONNECTION = "CONNECTION"
PROVIDER_ERROR_TIMEOUT = "TIMEOUT"
PROVIDER_ERROR_SERVER_ERROR = "SERVER_ERROR"
PROVIDER_ERROR_CONTENT_POLICY = "PROVIDER_CONTENT_POLICY"
PROVIDER_ERROR_OTHER = "OTHER"

PROVIDER_ERROR_CATEGORIES = (
    PROVIDER_ERROR_INVALID_REQUEST,
    PROVIDER_ERROR_UNSUPPORTED_PARAMETER,
    PROVIDER_ERROR_AUTHENTICATION,
    PROVIDER_ERROR_PERMISSION,
    PROVIDER_ERROR_RATE_LIMIT,
    PROVIDER_ERROR_QUOTA,
    PROVIDER_ERROR_MODEL_NOT_FOUND,
    PROVIDER_ERROR_CONNECTION,
    PROVIDER_ERROR_TIMEOUT,
    PROVIDER_ERROR_SERVER_ERROR,
    PROVIDER_ERROR_CONTENT_POLICY,
    PROVIDER_ERROR_OTHER,
)

PROVIDER_DIAGNOSTIC_FIELDS = (
    "provider_error_type",
    "provider_error_code",
    "provider_http_status",
    "provider_error_category",
    "provider_error_retryable",
    "provider_endpoint_family",
    "provider_model",
    "provider_error_message_safe",
)

SECRET_MARKERS = (
    "sk-",
    "bearer",
    "authorization",
    "x-api-key",
    "api_key",
    "secret",
    "password",
    "credential",
    "token",
)


def build_safe_provider_error_diagnostics(
    exc: BaseException,
    *,
    provider: str,
    model: str,
    endpoint_family: str,
) -> dict[str, Any]:
    """Return bounded, non-secret diagnostics for a provider exception."""
    http_status = _safe_http_status(exc)
    error_code = _safe_error_code(exc)
    error_type = _safe_text(exc.__class__.__name__)
    category = _provider_error_category(
        exc,
        http_status=http_status,
        error_code=error_code,
        error_type=error_type,
    )
    return {
        "provider_error_type": error_type,
        "provider_error_code": error_code,
        "provider_http_status": http_status,
        "provider_error_category": category,
        "provider_error_retryable": _is_retryable(category, http_status, exc),
        "provider_endpoint_family": _safe_text(endpoint_family),
        "provider_model": _safe_text(model),
        "provider_error_message_safe": _safe_provider_error_message(exc),
    }


def sanitize_provider_error_diagnostics(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    sanitized: dict[str, Any] = {}
    for field_name in PROVIDER_DIAGNOSTIC_FIELDS:
        field_value = value.get(field_name)
        if field_name == "provider_http_status":
            if _is_non_negative_int(field_value):
                sanitized[field_name] = field_value
        elif field_name == "provider_error_retryable":
            if field_value is None or isinstance(field_value, bool):
                sanitized[field_name] = field_value
        elif field_name == "provider_error_category":
            if field_value in PROVIDER_ERROR_CATEGORIES:
                sanitized[field_name] = field_value
        elif isinstance(field_value, str) and field_value.strip():
            sanitized[field_name] = _safe_text(field_value)
    return sanitized or None


def _provider_error_category(
    exc: BaseException,
    *,
    http_status: int | None,
    error_code: str | None,
    error_type: str | None,
) -> str:
    code = str(error_code or "").lower()
    exc_type = str(error_type or "").lower()
    if code == "timeout" or _is_timeout_error(exc):
        return PROVIDER_ERROR_TIMEOUT
    if code in ("url_error", "os_error"):
        return PROVIDER_ERROR_CONNECTION
    if "connection" in exc_type or "api_connection" in exc_type:
        return PROVIDER_ERROR_CONNECTION
    if "content" in code and "policy" in code:
        return PROVIDER_ERROR_CONTENT_POLICY
    if "unsupported" in code and "parameter" in code:
        return PROVIDER_ERROR_UNSUPPORTED_PARAMETER
    if "model_not_found" in code or "model_not_found" in exc_type:
        return PROVIDER_ERROR_MODEL_NOT_FOUND
    if "insufficient_quota" in code or "quota" in code:
        return PROVIDER_ERROR_QUOTA
    if "rate_limit" in code or "ratelimit" in exc_type or "rate_limit" in exc_type:
        return PROVIDER_ERROR_RATE_LIMIT
    if http_status == 400:
        return PROVIDER_ERROR_UNSUPPORTED_PARAMETER if "unsupported" in code else PROVIDER_ERROR_INVALID_REQUEST
    if http_status == 401:
        return PROVIDER_ERROR_AUTHENTICATION
    if http_status == 403:
        return PROVIDER_ERROR_PERMISSION
    if http_status == 404:
        return PROVIDER_ERROR_MODEL_NOT_FOUND
    if http_status == 429:
        return PROVIDER_ERROR_QUOTA if "quota" in code else PROVIDER_ERROR_RATE_LIMIT
    if http_status is not None and 500 <= http_status <= 599:
        return PROVIDER_ERROR_SERVER_ERROR
    return PROVIDER_ERROR_OTHER


def _is_retryable(
    category: str,
    http_status: int | None,
    exc: BaseException,
) -> bool | None:
    if category in (
        PROVIDER_ERROR_RATE_LIMIT,
        PROVIDER_ERROR_CONNECTION,
        PROVIDER_ERROR_TIMEOUT,
        PROVIDER_ERROR_SERVER_ERROR,
    ):
        return True
    if category in (
        PROVIDER_ERROR_INVALID_REQUEST,
        PROVIDER_ERROR_UNSUPPORTED_PARAMETER,
        PROVIDER_ERROR_AUTHENTICATION,
        PROVIDER_ERROR_PERMISSION,
        PROVIDER_ERROR_QUOTA,
        PROVIDER_ERROR_MODEL_NOT_FOUND,
        PROVIDER_ERROR_CONTENT_POLICY,
    ):
        return False
    if http_status is not None:
        if http_status in {408, 409, 425}:
            return True
        if 400 <= http_status < 500:
            return False
    if _is_timeout_error(exc):
        return True
    return None


def _safe_http_status(exc: BaseException) -> int | None:
    for attr_name in ("status_code", "status", "http_status"):
        value = getattr(exc, attr_name, None)
        if _is_non_negative_int(value):
            return value
    code = getattr(exc, "code", None)
    if _is_non_negative_int(code) and 100 <= code <= 599:
        return code
    response = getattr(exc, "response", None)
    value = getattr(response, "status_code", None)
    if _is_non_negative_int(value):
        return value
    return None

def _safe_error_code(exc: BaseException) -> str | None:
    value = getattr(exc, "code", None) or getattr(exc, "error_code", None)
    if value is None:
        body = getattr(exc, "body", None)
        if isinstance(body, dict):
            error = body.get("error")
            if isinstance(error, dict):
                value = error.get("code") or error.get("type")
    if value is None:
        return None
    if any(marker in str(value).lower() for marker in SECRET_MARKERS):
        return "redacted_error_code"
    text = _safe_text(value)
    if text is None:
        return None
    return "".join(char if char.isalnum() or char in "_.-" else "_" for char in text)[:120]


def _is_timeout_error(exc: BaseException) -> bool:
    text = f"{exc.__class__.__name__} {exc}".lower()
    return "timeout" in text or "timed out" in text


def _safe_provider_error_message(exc: BaseException) -> str:
    return "provider execution failed; raw exception message omitted"


def _safe_text(value: Any) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if not text:
        return None
    if any(marker in text.lower() for marker in SECRET_MARKERS):
        return "redacted"
    return text[:120]


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
