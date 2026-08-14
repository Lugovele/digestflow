from __future__ import annotations

import json
from urllib.error import HTTPError

from django.test import SimpleTestCase

from services.packaging.linkedin_post_provider_diagnostics import (
    PROVIDER_ERROR_AUTHENTICATION,
    PROVIDER_ERROR_CATEGORIES,
    PROVIDER_ERROR_CONNECTION,
    PROVIDER_ERROR_INVALID_REQUEST,
    PROVIDER_ERROR_MODEL_NOT_FOUND,
    PROVIDER_ERROR_OTHER,
    PROVIDER_ERROR_PERMISSION,
    PROVIDER_ERROR_QUOTA,
    PROVIDER_ERROR_RATE_LIMIT,
    PROVIDER_ERROR_SERVER_ERROR,
    PROVIDER_ERROR_TIMEOUT,
    PROVIDER_ERROR_UNSUPPORTED_PARAMETER,
    build_safe_provider_error_diagnostics,
    sanitize_provider_error_diagnostics,
)


class ProviderDiagnosticsTests(SimpleTestCase):
    def test_supported_error_categories_are_canonical(self) -> None:
        self.assertEqual(
            PROVIDER_ERROR_CATEGORIES,
            (
                "INVALID_REQUEST",
                "UNSUPPORTED_PARAMETER",
                "AUTHENTICATION",
                "PERMISSION",
                "RATE_LIMIT",
                "QUOTA",
                "MODEL_NOT_FOUND",
                "CONNECTION",
                "TIMEOUT",
                "SERVER_ERROR",
                "PROVIDER_CONTENT_POLICY",
                "OTHER",
            ),
        )

    def test_openai_invalid_request_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("bad request", status_code=400))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_INVALID_REQUEST)
        self.assertEqual(diagnostics["provider_http_status"], 400)
        self.assertFalse(diagnostics["provider_error_retryable"])

    def test_openai_unsupported_parameter_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(
            _ProviderError("bad request", status_code=400, code="unsupported_parameter")
        )

        self.assertEqual(
            diagnostics["provider_error_category"],
            PROVIDER_ERROR_UNSUPPORTED_PARAMETER,
        )

    def test_openai_authentication_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("auth", status_code=401))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_AUTHENTICATION)

    def test_openai_permission_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("permission", status_code=403))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_PERMISSION)

    def test_openai_model_not_found_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("missing", status_code=404))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_MODEL_NOT_FOUND)

    def test_openai_rate_limit_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(
            _ProviderError("rate", status_code=429, code="rate_limit_exceeded")
        )

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_RATE_LIMIT)
        self.assertTrue(diagnostics["provider_error_retryable"])

    def test_urllib_http_error_status_is_normalized(self) -> None:
        diagnostics = _diagnostics(
            HTTPError("https://api.anthropic.com/v1/messages", 429, "secret body", {}, None)
        )

        self.assertEqual(diagnostics["provider_http_status"], 429)
        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_RATE_LIMIT)
        self.assertTrue(diagnostics["provider_error_retryable"])
        self.assertNotIn("secret body", json.dumps(diagnostics))
    def test_openai_quota_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(
            _ProviderError("quota", status_code=429, code="insufficient_quota")
        )

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_QUOTA)
        self.assertFalse(diagnostics["provider_error_retryable"])

    def test_openai_timeout_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(TimeoutError("secret timeout prompt body"))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_TIMEOUT)
        self.assertTrue(diagnostics["provider_error_retryable"])

    def test_openai_connection_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_APIConnectionError("secret connection details"))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_CONNECTION)
        self.assertTrue(diagnostics["provider_error_retryable"])

    def test_sanitized_timeout_code_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("anthropic provider request failed", code="timeout"))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_TIMEOUT)
        self.assertTrue(diagnostics["provider_error_retryable"])

    def test_sanitized_network_codes_are_normalized(self) -> None:
        for code in ("url_error", "os_error"):
            with self.subTest(code=code):
                diagnostics = _diagnostics(
                    _ProviderError("anthropic provider request failed", code=code)
                )

                self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_CONNECTION)
                self.assertTrue(diagnostics["provider_error_retryable"])

    def test_openai_server_error_is_normalized(self) -> None:
        diagnostics = _diagnostics(_ProviderError("server", status_code=500))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_SERVER_ERROR)
        self.assertTrue(diagnostics["provider_error_retryable"])

    def test_unknown_exception_maps_to_other(self) -> None:
        diagnostics = _diagnostics(RuntimeError("ordinary opaque failure"))

        self.assertEqual(diagnostics["provider_error_category"], PROVIDER_ERROR_OTHER)

    def test_no_sensitive_request_content_is_persisted(self) -> None:
        diagnostics = _diagnostics(
            _ProviderError(
                "sk-secret prompt with Authorization header",
                status_code=400,
                code="sk-secret-code",
            )
        )
        serialized = json.dumps(diagnostics, sort_keys=True)

        self.assertEqual(diagnostics["provider_error_code"], "redacted_error_code")
        self.assertNotIn("sk-secret", serialized)
        self.assertNotIn("Authorization", serialized)
        self.assertEqual(
            diagnostics["provider_error_message_safe"],
            "provider execution failed; raw exception message omitted",
        )

    def test_sanitize_provider_error_diagnostics_filters_unknown_fields(self) -> None:
        diagnostics = sanitize_provider_error_diagnostics(
            {
                "provider_error_type": "BadRequestError",
                "provider_error_code": "unsupported_parameter",
                "provider_http_status": 400,
                "provider_error_category": "UNSUPPORTED_PARAMETER",
                "provider_error_retryable": False,
                "provider_endpoint_family": "responses",
                "provider_model": "gpt-4.1-2025-04-14",
                "provider_error_message_safe": "provider execution failed",
                "raw_prompt": "secret prompt",
            }
        )

        self.assertNotIn("raw_prompt", diagnostics)
        self.assertEqual(diagnostics["provider_error_category"], "UNSUPPORTED_PARAMETER")


    def test_sanitize_provider_error_diagnostics_drops_invalid_category(self) -> None:
        diagnostics = sanitize_provider_error_diagnostics(
            {
                "provider_error_type": "BadRequestError",
                "provider_error_category": "raw secret category",
                "provider_error_message_safe": "provider execution failed",
            }
        )

        self.assertNotIn("provider_error_category", diagnostics)

class _ProviderError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code


class _APIConnectionError(Exception):
    pass


def _diagnostics(exc: BaseException) -> dict:
    return build_safe_provider_error_diagnostics(
        exc,
        provider="openai",
        model="gpt-4.1-2025-04-14",
        endpoint_family="responses",
    )
