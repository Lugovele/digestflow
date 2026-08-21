"""Небольшой AI-адаптер для сервисного слоя."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from django.conf import settings
from openai import OpenAI


AI_PROVIDER_OPENAI = "openai"
AI_PROVIDER_GEMINI = "gemini"
AI_PROVIDER_ANTHROPIC = "anthropic"
SUPPORTED_AI_PROVIDERS = (
    AI_PROVIDER_OPENAI,
    AI_PROVIDER_GEMINI,
    AI_PROVIDER_ANTHROPIC,
)
AI_THINKING_MODE_PROVIDER_DEFAULT = "provider_default"
AI_THINKING_MODE_DISABLED = "disabled"
SUPPORTED_AI_THINKING_MODES = (
    AI_THINKING_MODE_PROVIDER_DEFAULT,
    AI_THINKING_MODE_DISABLED,
)
AI_REASONING_EFFORT_MINIMAL = "minimal"
AI_REASONING_EFFORT_LOW = "low"
AI_REASONING_EFFORT_MEDIUM = "medium"
AI_REASONING_EFFORT_HIGH = "high"
GEMINI_SUPPORTED_REASONING_EFFORTS = (
    AI_REASONING_EFFORT_MINIMAL,
    AI_REASONING_EFFORT_LOW,
    AI_REASONING_EFFORT_MEDIUM,
    AI_REASONING_EFFORT_HIGH,
)
GEMINI_OPENAI_COMPATIBLE_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_GENERATE_CONTENT_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_EXECUTION_PATH_OPENAI_COMPATIBLE = "openai_compatible"
GEMINI_EXECUTION_PATH_NATIVE = "native_gemini"
GEMINI_SUPPORTED_EXECUTION_PATHS = (
    GEMINI_EXECUTION_PATH_OPENAI_COMPATIBLE,
    GEMINI_EXECUTION_PATH_NATIVE,
)
GEMINI_SUPPORTED_MODELS = ("gemini-3.6-flash",)
ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_SUPPORTED_MODELS = ("claude-sonnet-5",)
PLACEHOLDER_API_KEYS = {
    "",
    "sk-your-key",
    "your-openai-api-key-here",
    "your-gemini-api-key-here",
    "your-anthropic-api-key-here",
}


@dataclass(frozen=True)
class AIResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, int | None]
    provider_response_metadata: dict[str, Any] | None = None


@dataclass(frozen=True)
class AIProviderConfig:
    provider: str
    api_key: str
    base_url: str | None = None


class AIProviderRequestError(RuntimeError):
    """Sanitized provider request failure with safe diagnostic attributes."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code


INPUT_COST_PER_1K_TOKENS = 0.005
OUTPUT_COST_PER_1K_TOKENS = 0.015


class OpenAICompatibleClient:
    """Synchronous OpenAI-compatible text client with explicit provider metadata."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        provider: str,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.provider = provider
        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": settings.OPENAI_TIMEOUT_SECONDS,
        }
        if base_url is not None:
            client_kwargs["base_url"] = base_url
        self.client = OpenAI(**client_kwargs)

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = 1200,
        json_mode: bool = False,
        allow_json_mode_fallback: bool = True,
        thinking_mode: str = AI_THINKING_MODE_PROVIDER_DEFAULT,
        reasoning_effort: str | None = None,
        execution_path: str | None = None,
        thinking_budget: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AIResponse:
        thinking_mode_error = get_ai_provider_thinking_mode_error(
            provider=self.provider,
            thinking_mode=thinking_mode,
        )
        if thinking_mode_error is not None:
            raise ValueError(thinking_mode_error)
        reasoning_effort_error = get_ai_provider_reasoning_effort_error(
            provider=self.provider,
            reasoning_effort=reasoning_effort,
        )
        if reasoning_effort_error is not None:
            raise ValueError(reasoning_effort_error)
        if execution_path not in (None, GEMINI_EXECUTION_PATH_OPENAI_COMPATIBLE):
            raise ValueError(
                f"unsupported AI execution_path for provider {self.provider}: "
                f"{execution_path}"
            )
        if thinking_budget is not None:
            raise ValueError("thinking_budget is supported only by native Gemini execution")
        if response_schema is not None:
            raise ValueError("response_schema is supported only by native Gemini execution")
        if self.provider == AI_PROVIDER_GEMINI:
            return self._generate_chat_completion(
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                json_mode=json_mode,
                reasoning_effort=reasoning_effort,
            )

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
            if not json_mode or not allow_json_mode_fallback:
                raise
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )

        raw = response.model_dump()
        text = response.output_text
        return AIResponse(
            text=text,
            raw=raw,
            usage=_extract_usage(response, raw),
            provider_response_metadata=_build_openai_responses_metadata(
                raw,
                provider=self.provider,
                model=self.model,
                max_output_tokens=max_output_tokens,
            ),
        )

    def _generate_chat_completion(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        json_mode: bool,
        reasoning_effort: str | None = None,
    ) -> AIResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}
        normalized_reasoning_effort = _normalize_optional_reasoning_effort(
            reasoning_effort
        )
        if normalized_reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = normalized_reasoning_effort

        response = self.client.chat.completions.create(**request_kwargs)
        raw = response.model_dump()
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            message = getattr(choices[0], "message", None)
            text = str(getattr(message, "content", "") or "")
        return AIResponse(
            text=text,
            raw=raw,
            usage=_extract_usage(response, raw),
            provider_response_metadata=_build_openai_compatible_chat_metadata(
                raw,
                provider=self.provider,
                model=self.model,
                max_output_tokens=max_output_tokens,
            ),
        )


class OpenAIClient(OpenAICompatibleClient):
    """Legacy OpenAI wrapper preserving existing OpenAI defaults."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key or settings.OPENAI_API_KEY,
            model=model or settings.OPENAI_MODEL,
            provider=AI_PROVIDER_OPENAI,
        )


class AnthropicMessagesClient:
    """Synchronous native Anthropic Messages client using AIResponse."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.provider = AI_PROVIDER_ANTHROPIC
        self.api_key = api_key
        self.timeout = (
            timeout
            if timeout is not None
            else int(
                getattr(
                    settings,
                    "ANTHROPIC_TIMEOUT_SECONDS",
                    settings.OPENAI_TIMEOUT_SECONDS,
                )
            )
        )

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = 1200,
        json_mode: bool = False,
        allow_json_mode_fallback: bool = True,
        thinking_mode: str = AI_THINKING_MODE_PROVIDER_DEFAULT,
        reasoning_effort: str | None = None,
        execution_path: str | None = None,
        thinking_budget: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AIResponse:
        normalized_thinking_mode = _normalize_thinking_mode(thinking_mode)
        thinking_mode_error = get_ai_provider_thinking_mode_error(
            provider=self.provider,
            thinking_mode=normalized_thinking_mode,
        )
        if thinking_mode_error is not None:
            raise ValueError(thinking_mode_error)
        reasoning_effort_error = get_ai_provider_reasoning_effort_error(
            provider=self.provider,
            reasoning_effort=reasoning_effort,
        )
        if reasoning_effort_error is not None:
            raise ValueError(reasoning_effort_error)
        if execution_path is not None:
            raise ValueError(
                f"unsupported AI execution_path for provider {self.provider}: "
                f"{execution_path}"
            )
        if thinking_budget is not None:
            raise ValueError("thinking_budget is supported only by native Gemini execution")
        if response_schema is not None:
            raise ValueError("response_schema is supported only by native Gemini execution")

        request_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if normalized_thinking_mode == AI_THINKING_MODE_DISABLED:
            request_body["thinking"] = {"type": "disabled"}
        if json_mode:
            request_body["system"] = (
                "Return only valid JSON. Do not include markdown fences or commentary."
            )

        request = Request(
            ANTHROPIC_MESSAGES_ENDPOINT,
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": ANTHROPIC_API_VERSION,
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_bytes = response.read()
        except HTTPError as exc:
            raise AIProviderRequestError(
                "anthropic provider request failed",
                status_code=exc.code,
                code=f"http_{exc.code}",
            ) from None
        except TimeoutError:
            raise AIProviderRequestError(
                "anthropic provider request failed",
                code="timeout",
            ) from None
        except URLError:
            raise AIProviderRequestError(
                "anthropic provider request failed",
                code="url_error",
            ) from None
        except OSError:
            raise AIProviderRequestError(
                "anthropic provider request failed",
                code="os_error",
            ) from None

        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError("anthropic provider response was not valid JSON") from None

        if not _is_valid_anthropic_response_shape(raw):
            raise RuntimeError("anthropic provider response shape was invalid") from None
        text = _extract_anthropic_text(raw)
        usage = _extract_anthropic_usage(raw)
        metadata = _build_anthropic_provider_response_metadata(
            raw,
            model=self.model,
            max_output_tokens=max_output_tokens,
        )
        return AIResponse(
            text=text,
            raw=raw if text else {},
            usage=usage,
            provider_response_metadata=metadata,
        )


class GeminiNativeGenerateContentClient:
    """Synchronous native Gemini GenerateContent client using AIResponse."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout: int | None = None,
    ) -> None:
        self.model = model
        self.provider = AI_PROVIDER_GEMINI
        self.execution_path = GEMINI_EXECUTION_PATH_NATIVE
        self.api_key = api_key
        self.timeout = (
            timeout
            if timeout is not None
            else int(
                getattr(
                    settings,
                    "GEMINI_TIMEOUT_SECONDS",
                    settings.OPENAI_TIMEOUT_SECONDS,
                )
            )
        )

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = 1200,
        json_mode: bool = False,
        allow_json_mode_fallback: bool = True,
        thinking_mode: str = AI_THINKING_MODE_PROVIDER_DEFAULT,
        reasoning_effort: str | None = None,
        execution_path: str | None = None,
        thinking_budget: int | None = None,
        response_schema: dict[str, Any] | None = None,
    ) -> AIResponse:
        if execution_path not in (None, GEMINI_EXECUTION_PATH_NATIVE):
            raise ValueError(
                f"unsupported AI execution_path for provider gemini: {execution_path}"
            )
        thinking_mode_error = get_ai_provider_thinking_mode_error(
            provider=self.provider,
            thinking_mode=thinking_mode,
        )
        if thinking_mode_error is not None:
            raise ValueError(thinking_mode_error)
        if reasoning_effort is not None:
            raise ValueError(
                "reasoning_effort is not supported by native Gemini execution; "
                "use thinking_budget"
            )
        if isinstance(thinking_budget, bool) or not isinstance(thinking_budget, int):
            raise ValueError("native Gemini thinking_budget must be an integer")
        if thinking_budget < 0:
            raise ValueError("native Gemini thinking_budget must be non-negative")

        generation_config: dict[str, Any] = {
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {
                "thinkingBudget": thinking_budget,
                "includeThoughts": False,
            },
        }
        if json_mode or response_schema is not None:
            generation_config["responseMimeType"] = "application/json"
            if response_schema is not None:
                generation_config["responseSchema"] = copy_json_value(response_schema)

        request_body: dict[str, Any] = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": generation_config,
        }
        request = Request(
            GEMINI_GENERATE_CONTENT_ENDPOINT_TEMPLATE.format(model=self.model),
            data=json.dumps(request_body).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_bytes = response.read()
        except HTTPError as exc:
            raise AIProviderRequestError(
                "gemini native provider request failed",
                status_code=exc.code,
                code=f"http_{exc.code}",
            ) from None
        except TimeoutError:
            raise AIProviderRequestError(
                "gemini native provider request failed",
                code="timeout",
            ) from None
        except URLError:
            raise AIProviderRequestError(
                "gemini native provider request failed",
                code="url_error",
            ) from None
        except OSError:
            raise AIProviderRequestError(
                "gemini native provider request failed",
                code="os_error",
            ) from None

        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise RuntimeError(
                "gemini native provider response was not valid JSON"
            ) from None

        if not _is_valid_gemini_native_response_shape(raw):
            raise RuntimeError("gemini native provider response shape was invalid") from None
        text = _extract_gemini_native_text(raw)
        return AIResponse(
            text=text,
            raw=raw if text else {},
            usage=_extract_gemini_native_usage(raw),
            provider_response_metadata=_build_gemini_native_provider_response_metadata(
                raw,
                model=self.model,
                max_output_tokens=max_output_tokens,
                thinking_budget=thinking_budget,
            ),
        )


def normalize_ai_provider(provider: str | None) -> str:
    return str(provider or "").strip().lower()


def get_ai_provider_config(provider: str) -> AIProviderConfig:
    normalized_provider = normalize_ai_provider(provider)
    if normalized_provider == AI_PROVIDER_OPENAI:
        return AIProviderConfig(
            provider=AI_PROVIDER_OPENAI,
            api_key=str(settings.OPENAI_API_KEY or "").strip(),
        )
    if normalized_provider == AI_PROVIDER_GEMINI:
        return AIProviderConfig(
            provider=AI_PROVIDER_GEMINI,
            api_key=str(getattr(settings, "GEMINI_API_KEY", "") or "").strip(),
            base_url=GEMINI_OPENAI_COMPATIBLE_BASE_URL,
        )
    if normalized_provider == AI_PROVIDER_ANTHROPIC:
        return AIProviderConfig(
            provider=AI_PROVIDER_ANTHROPIC,
            api_key=str(getattr(settings, "ANTHROPIC_API_KEY", "") or "").strip(),
        )
    raise ValueError(f"unsupported AI provider: {normalized_provider}")


def get_ai_provider_model_error(
    provider: str,
    model: str,
    *,
    stage_name: str = "AI",
) -> str | None:
    normalized_provider = normalize_ai_provider(provider)
    normalized_model = str(model or "").strip()
    if not normalized_provider:
        return f"missing {stage_name} provider"
    if normalized_provider not in SUPPORTED_AI_PROVIDERS:
        return f"unsupported {stage_name} provider: {normalized_provider}"
    if not normalized_model:
        return f"missing {stage_name} model"
    if (
        normalized_provider == AI_PROVIDER_GEMINI
        and normalized_model not in GEMINI_SUPPORTED_MODELS
    ):
        return (
            f"{stage_name} provider/model mismatch: provider gemini supports "
            + ", ".join(GEMINI_SUPPORTED_MODELS)
        )
    if (
        normalized_provider == AI_PROVIDER_ANTHROPIC
        and normalized_model not in ANTHROPIC_SUPPORTED_MODELS
    ):
        return (
            f"{stage_name} provider/model mismatch: provider anthropic supports "
            + ", ".join(ANTHROPIC_SUPPORTED_MODELS)
        )
    if normalized_provider == AI_PROVIDER_OPENAI and normalized_model.startswith("gemini-"):
        return (
            f"{stage_name} provider/model mismatch: provider openai cannot use "
            "a Gemini model"
        )
    if normalized_provider == AI_PROVIDER_OPENAI and normalized_model.startswith("claude-"):
        return (
            f"{stage_name} provider/model mismatch: provider openai cannot use "
            "an Anthropic model"
        )
    return None


def get_ai_provider_credential_error(
    provider: str,
    *,
    stage_name: str = "AI",
) -> str | None:
    normalized_provider = normalize_ai_provider(provider)
    try:
        config = get_ai_provider_config(normalized_provider)
    except ValueError:
        return f"unsupported {stage_name} provider: {normalized_provider}"
    if config.api_key in PLACEHOLDER_API_KEYS:
        env_name = _provider_api_key_env_name(normalized_provider)
        return (
            f"{env_name} must be configured with a real key before "
            f"{stage_name} provider execution"
        )
    return None


def get_ai_client_configuration_error(
    provider: str,
    model: str,
    *,
    stage_name: str = "AI",
) -> str | None:
    model_error = get_ai_provider_model_error(provider, model, stage_name=stage_name)
    if model_error is not None:
        return model_error
    return get_ai_provider_credential_error(provider, stage_name=stage_name)


def get_ai_provider_thinking_mode_error(
    *,
    provider: str,
    thinking_mode: str | None,
    stage_name: str = "AI",
) -> str | None:
    normalized_provider = normalize_ai_provider(provider)
    normalized_thinking_mode = _normalize_thinking_mode(thinking_mode)
    if normalized_thinking_mode not in SUPPORTED_AI_THINKING_MODES:
        return f"unsupported {stage_name} thinking_mode: {normalized_thinking_mode}"
    if (
        normalized_provider != AI_PROVIDER_ANTHROPIC
        and normalized_thinking_mode != AI_THINKING_MODE_PROVIDER_DEFAULT
    ):
        return (
            f"unsupported {stage_name} thinking_mode for provider "
            f"{normalized_provider}: {normalized_thinking_mode}"
        )
    return None


def get_ai_provider_reasoning_effort_error(
    *,
    provider: str,
    reasoning_effort: str | None,
    stage_name: str = "AI",
) -> str | None:
    normalized_reasoning_effort = _normalize_optional_reasoning_effort(
        reasoning_effort
    )
    if normalized_reasoning_effort is None:
        return None
    normalized_provider = normalize_ai_provider(provider)
    if normalized_provider != AI_PROVIDER_GEMINI:
        return (
            f"unsupported {stage_name} reasoning_effort for provider "
            f"{normalized_provider}: {normalized_reasoning_effort}"
        )
    if normalized_reasoning_effort not in GEMINI_SUPPORTED_REASONING_EFFORTS:
        return f"unsupported {stage_name} reasoning_effort: {normalized_reasoning_effort}"
    return None


def build_ai_client(
    provider: str,
    model: str,
    *,
    execution_path: str | None = None,
) -> (
    OpenAICompatibleClient
    | AnthropicMessagesClient
    | GeminiNativeGenerateContentClient
):
    normalized_model = str(model or "").strip()
    configuration_error = get_ai_client_configuration_error(provider, normalized_model)
    if configuration_error is not None:
        raise ValueError(configuration_error)
    config = get_ai_provider_config(provider)
    normalized_execution_path = _normalize_optional_execution_path(execution_path)
    if config.provider == AI_PROVIDER_GEMINI:
        if normalized_execution_path == GEMINI_EXECUTION_PATH_NATIVE:
            return GeminiNativeGenerateContentClient(
                api_key=config.api_key,
                model=normalized_model,
                timeout=int(
                    getattr(
                        settings,
                        "GEMINI_TIMEOUT_SECONDS",
                        settings.OPENAI_TIMEOUT_SECONDS,
                    )
                ),
            )
        if normalized_execution_path not in (None, GEMINI_EXECUTION_PATH_OPENAI_COMPATIBLE):
            raise ValueError(
                f"unsupported AI execution_path for provider gemini: {execution_path}"
            )
    elif normalized_execution_path is not None:
        raise ValueError(
            f"unsupported AI execution_path for provider {config.provider}: {execution_path}"
        )
    if config.provider == AI_PROVIDER_ANTHROPIC:
        return AnthropicMessagesClient(
            api_key=config.api_key,
            model=normalized_model,
            timeout=int(
                getattr(
                    settings,
                    "ANTHROPIC_TIMEOUT_SECONDS",
                    settings.OPENAI_TIMEOUT_SECONDS,
                )
            ),
        )
    return OpenAICompatibleClient(
        api_key=config.api_key,
        model=normalized_model,
        provider=config.provider,
        base_url=config.base_url,
    )


def estimate_cost_usd(prompt_tokens: int | None, completion_tokens: int | None) -> float | None:
    if prompt_tokens is None or completion_tokens is None:
        return None

    input_cost = (prompt_tokens / 1000) * INPUT_COST_PER_1K_TOKENS
    output_cost = (completion_tokens / 1000) * OUTPUT_COST_PER_1K_TOKENS
    return round(input_cost + output_cost, 6)


def _provider_api_key_env_name(provider: str) -> str:
    if provider == AI_PROVIDER_GEMINI:
        return "GEMINI_API_KEY"
    if provider == AI_PROVIDER_ANTHROPIC:
        return "ANTHROPIC_API_KEY"
    return "OPENAI_API_KEY"


def _normalize_thinking_mode(thinking_mode: str | None) -> str:
    return str(thinking_mode or AI_THINKING_MODE_PROVIDER_DEFAULT).strip().lower()


def _normalize_optional_reasoning_effort(reasoning_effort: str | None) -> str | None:
    if reasoning_effort is None:
        return None
    normalized = str(reasoning_effort).strip().lower()
    return normalized or None


def _normalize_optional_execution_path(execution_path: str | None) -> str | None:
    if execution_path is None:
        return None
    normalized = str(execution_path).strip().lower()
    return normalized or None


def copy_json_value(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _is_valid_gemini_native_response_shape(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    candidates = raw.get("candidates")
    if not isinstance(candidates, list):
        return False
    for candidate in candidates:
        if not isinstance(candidate, dict):
            return False
        content = candidate.get("content", {})
        if not isinstance(content, dict):
            return False
        parts = content.get("parts", [])
        if not isinstance(parts, list):
            return False
        for part in parts:
            if not isinstance(part, dict):
                return False
            if "text" in part and not isinstance(part.get("text"), str):
                return False
    usage = raw.get("usageMetadata", {})
    if not isinstance(usage, dict):
        return False
    for usage_key in (
        "promptTokenCount",
        "candidatesTokenCount",
        "thoughtsTokenCount",
        "totalTokenCount",
    ):
        value = usage.get(usage_key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            return False
    return True


def _extract_gemini_native_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    candidates = raw.get("candidates", [])
    if not candidates:
        return ""
    content = candidates[0].get("content", {})
    for part in content.get("parts", []):
        if part.get("thought") is True:
            continue
        text = part.get("text")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _extract_gemini_native_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    usage = raw.get("usageMetadata", {}) if isinstance(raw, dict) else {}
    prompt_tokens = usage.get("promptTokenCount") if isinstance(usage, dict) else None
    completion_tokens = (
        usage.get("candidatesTokenCount") if isinstance(usage, dict) else None
    )
    total_tokens = usage.get("totalTokenCount") if isinstance(usage, dict) else None
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_gemini_native_provider_response_metadata(
    raw: dict[str, Any],
    *,
    model: str,
    max_output_tokens: int,
    thinking_budget: int,
) -> dict[str, Any]:
    candidates = raw.get("candidates", [])
    candidate = candidates[0] if candidates and isinstance(candidates[0], dict) else {}
    finish_reason = _safe_metadata_text(candidate.get("finishReason"))
    usage = raw.get("usageMetadata", {})
    prompt_tokens = _safe_metadata_int(usage.get("promptTokenCount"))
    visible_output_tokens = _safe_metadata_int(usage.get("candidatesTokenCount"))
    thinking_tokens = _safe_metadata_int(usage.get("thoughtsTokenCount"))
    total_tokens = _safe_metadata_int(usage.get("totalTokenCount"))
    return {
        "provider": AI_PROVIDER_GEMINI,
        "model": model,
        "execution_path": GEMINI_EXECUTION_PATH_NATIVE,
        "provider_finish_reason": finish_reason,
        "provider_stop_reason": None,
        "provider_max_output_tokens": _safe_metadata_int(max_output_tokens),
        "thinking_budget_configured": _safe_metadata_int(thinking_budget),
        "provider_thinking_budget": _safe_metadata_int(thinking_budget),
        "provider_reported_output_tokens": visible_output_tokens,
        "provider_output_limit_reached": _provider_output_limit_reached(
            finish_reason=finish_reason,
            stop_reason=None,
        ),
        **_provider_token_accounting_metadata(
            prompt_tokens=prompt_tokens,
            visible_output_tokens=visible_output_tokens,
            total_tokens=total_tokens,
            max_output_tokens=max_output_tokens,
            thinking_tokens=thinking_tokens,
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": visible_output_tokens,
        "total_tokens": total_tokens,
        "thoughts_token_count": thinking_tokens,
    }


def _is_valid_anthropic_response_shape(raw: Any) -> bool:
    if not isinstance(raw, dict):
        return False
    content = raw.get("content")
    if not isinstance(content, list):
        return False
    for block in content:
        if not isinstance(block, dict):
            return False
        block_type = block.get("type")
        if not isinstance(block_type, str) or not block_type.strip():
            return False
        if block_type == "text" and not isinstance(block.get("text"), str):
            return False
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        return False
    for usage_key in ("input_tokens", "output_tokens", "thinking_tokens"):
        value = usage.get(usage_key)
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value < 0
        ):
            return False
    return True


def _extract_anthropic_text(raw: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in raw["content"]:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if text is not None:
            parts.append(str(text))
    return "".join(parts)


def _extract_anthropic_usage(raw: dict[str, Any]) -> dict[str, int | None]:
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    prompt_tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    completion_tokens = usage.get("output_tokens") if isinstance(usage, dict) else None
    total_tokens = None
    if prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_anthropic_provider_response_metadata(
    raw: dict[str, Any],
    *,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    usage = raw.get("usage", {})
    block_types: list[str] = []
    for block in raw.get("content", []):
        block_type = _safe_metadata_text(block.get("type"))
        if block_type and block_type not in block_types:
            block_types.append(block_type)
    stop_reason = _safe_metadata_text(raw.get("stop_reason"))
    input_tokens = _safe_metadata_int(usage.get("input_tokens"))
    output_tokens = _safe_metadata_int(usage.get("output_tokens"))
    thinking_tokens = _safe_metadata_int(usage.get("thinking_tokens"))
    return {
        "provider": AI_PROVIDER_ANTHROPIC,
        "model": _safe_metadata_text(raw.get("model")) or model,
        "stop_reason": stop_reason,
        "provider_stop_reason": stop_reason,
        "provider_finish_reason": None,
        "provider_max_output_tokens": _safe_metadata_int(max_output_tokens),
        "provider_reported_output_tokens": output_tokens,
        "provider_output_limit_reached": _provider_output_limit_reached(
            finish_reason=None,
            stop_reason=stop_reason,
        ),
        **_provider_token_accounting_metadata(
            prompt_tokens=input_tokens,
            visible_output_tokens=output_tokens,
            total_tokens=(
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            ),
            max_output_tokens=max_output_tokens,
            thinking_tokens=thinking_tokens,
        ),
        "content_block_types": block_types[:20],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
    }


def _build_openai_compatible_chat_metadata(
    raw: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    choices = raw.get("choices", [])
    if not isinstance(choices, list):
        choices = []
    finish_reasons: list[str] = []
    message_content_types: list[str] = []
    for choice in choices[:20]:
        if not isinstance(choice, dict):
            continue
        finish_reason = _safe_metadata_text(choice.get("finish_reason"))
        if finish_reason and finish_reason not in finish_reasons:
            finish_reasons.append(finish_reason)
        message = choice.get("message")
        if isinstance(message, dict):
            content = message.get("content")
            content_type = type(content).__name__
            if content_type not in message_content_types:
                message_content_types.append(content_type)
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    finish_reason = finish_reasons[0] if len(finish_reasons) == 1 else None
    prompt_tokens = _safe_metadata_int(
        usage.get("prompt_tokens", usage.get("input_tokens"))
    )
    completion_tokens = _safe_metadata_int(
        usage.get("completion_tokens", usage.get("output_tokens"))
    )
    total_tokens = _safe_metadata_int(usage.get("total_tokens"))
    completion_details = usage.get("completion_tokens_details")
    if not isinstance(completion_details, dict):
        completion_details = {}
    reasoning_tokens = _safe_metadata_int(
        completion_details.get("reasoning_tokens")
    )
    return {
        "provider": provider,
        "model": _safe_metadata_text(raw.get("model")) or model,
        "choices_count": len(choices),
        "finish_reasons": finish_reasons,
        "provider_finish_reason": finish_reason,
        "provider_stop_reason": None,
        "provider_max_output_tokens": _safe_metadata_int(max_output_tokens),
        "provider_reported_output_tokens": completion_tokens,
        "provider_output_limit_reached": _provider_output_limit_reached(
            finish_reason=finish_reason,
            stop_reason=None,
        ),
        **_provider_token_accounting_metadata(
            prompt_tokens=prompt_tokens,
            visible_output_tokens=completion_tokens,
            total_tokens=total_tokens,
            max_output_tokens=max_output_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
        "message_content_types": message_content_types,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _build_openai_responses_metadata(
    raw: dict[str, Any],
    *,
    provider: str,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    usage = raw.get("usage", {})
    if not isinstance(usage, dict):
        usage = {}
    incomplete_details = raw.get("incomplete_details")
    if not isinstance(incomplete_details, dict):
        incomplete_details = {}
    status = _safe_metadata_text(raw.get("status"))
    finish_reason = _safe_metadata_text(incomplete_details.get("reason")) or status
    prompt_tokens = _safe_metadata_int(
        usage.get("input_tokens", usage.get("prompt_tokens"))
    )
    output_tokens = _safe_metadata_int(
        usage.get("output_tokens", usage.get("completion_tokens"))
    )
    total_tokens = _safe_metadata_int(usage.get("total_tokens"))
    output_details = usage.get("output_tokens_details")
    if not isinstance(output_details, dict):
        output_details = {}
    reasoning_tokens = _safe_metadata_int(output_details.get("reasoning_tokens"))
    return {
        "provider": provider,
        "model": _safe_metadata_text(raw.get("model")) or model,
        "provider_finish_reason": finish_reason,
        "provider_stop_reason": None,
        "provider_max_output_tokens": _safe_metadata_int(max_output_tokens),
        "provider_reported_output_tokens": output_tokens,
        "provider_output_limit_reached": _provider_output_limit_reached(
            finish_reason=finish_reason,
            stop_reason=None,
        ),
        **_provider_token_accounting_metadata(
            prompt_tokens=prompt_tokens,
            visible_output_tokens=output_tokens,
            total_tokens=total_tokens,
            max_output_tokens=max_output_tokens,
            reasoning_tokens=reasoning_tokens,
        ),
    }


def _provider_output_limit_reached(
    *,
    finish_reason: str | None,
    stop_reason: str | None,
) -> bool | None:
    reason = (finish_reason or stop_reason or "").strip().lower()
    if not reason:
        return None
    if reason in {"length", "max_tokens", "max_output_tokens"}:
        return True
    if reason in {"stop", "end_turn", "completed", "complete"}:
        return False
    return None


def _provider_token_accounting_metadata(
    *,
    prompt_tokens: int | None,
    visible_output_tokens: int | None,
    total_tokens: int | None,
    max_output_tokens: int,
    reasoning_tokens: int | None = None,
    thinking_tokens: int | None = None,
) -> dict[str, int | float | None]:
    hidden_output_tokens = _provider_hidden_output_tokens(
        prompt_tokens=prompt_tokens,
        visible_output_tokens=visible_output_tokens,
        total_tokens=total_tokens,
    )
    combined_output_tokens = (
        visible_output_tokens + hidden_output_tokens
        if visible_output_tokens is not None and hidden_output_tokens is not None
        else None
    )
    return {
        "provider_prompt_tokens": prompt_tokens,
        "provider_visible_output_tokens": visible_output_tokens,
        "provider_total_tokens": total_tokens,
        "provider_hidden_output_tokens": hidden_output_tokens,
        "provider_combined_output_tokens": combined_output_tokens,
        "provider_output_budget_utilization_percent": (
            round((combined_output_tokens / max_output_tokens) * 100, 2)
            if combined_output_tokens is not None
            and not isinstance(max_output_tokens, bool)
            and isinstance(max_output_tokens, int)
            and max_output_tokens > 0
            else None
        ),
        "provider_reasoning_tokens": reasoning_tokens,
        "provider_thinking_tokens": thinking_tokens,
    }


def _provider_hidden_output_tokens(
    *,
    prompt_tokens: int | None,
    visible_output_tokens: int | None,
    total_tokens: int | None,
) -> int | None:
    if (
        prompt_tokens is None
        or visible_output_tokens is None
        or total_tokens is None
    ):
        return None
    hidden = total_tokens - prompt_tokens - visible_output_tokens
    return hidden if hidden >= 0 else None


def _safe_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    normalized = " ".join(str(value).split())
    if not normalized:
        return None
    return normalized[:120]


def _safe_metadata_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _extract_usage(response: Any, raw: dict[str, Any]) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "input_tokens", None)
    completion_tokens = getattr(usage, "output_tokens", None)
    total_tokens = getattr(usage, "total_tokens", None)

    raw_usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    if prompt_tokens is None:
        prompt_tokens = raw_usage.get("input_tokens") or raw_usage.get("prompt_tokens")
    if completion_tokens is None:
        completion_tokens = raw_usage.get("output_tokens") or raw_usage.get("completion_tokens")
    if total_tokens is None:
        total_tokens = raw_usage.get("total_tokens")

    if total_tokens is None and prompt_tokens is not None and completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }
