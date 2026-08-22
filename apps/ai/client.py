"""Небольшой OpenAI-адаптер для сервисного слоя."""
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
SUPPORTED_AI_PROVIDERS = (AI_PROVIDER_OPENAI, AI_PROVIDER_GEMINI, AI_PROVIDER_ANTHROPIC)
GEMINI_OPENAI_COMPATIBLE_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_SUPPORTED_MODELS = ("gemini-3.6-flash",)
ANTHROPIC_MESSAGES_ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_API_VERSION = "2023-06-01"
ANTHROPIC_SUPPORTED_MODELS = ("claude-sonnet-5",)
THINKING_MODE_PROVIDER_DEFAULT = "provider_default"
THINKING_MODE_DISABLED = "disabled"
THINKING_MODE_ADAPTIVE = "adaptive"
SUPPORTED_THINKING_MODES = (
    THINKING_MODE_PROVIDER_DEFAULT,
    THINKING_MODE_DISABLED,
    THINKING_MODE_ADAPTIVE,
)
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
        thinking_mode: str = THINKING_MODE_PROVIDER_DEFAULT,
    ) -> AIResponse:
        thinking_error = _get_provider_thinking_mode_error(
            provider=self.provider,
            thinking_mode=thinking_mode,
        )
        if thinking_error is not None:
            raise ValueError(thinking_error)
        if self.provider == AI_PROVIDER_GEMINI:
            return self._generate_chat_completion(
                prompt=prompt,
                max_output_tokens=max_output_tokens,
                json_mode=json_mode,
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
        usage = _extract_usage(response, raw)
        return AIResponse(
            text=text,
            raw=raw,
            usage=usage,
            provider_response_metadata=_build_provider_response_metadata(
                provider=self.provider,
                model=self.model,
                raw=raw,
                usage=usage,
                text=text,
            ),
        )

    def _generate_chat_completion(
        self,
        *,
        prompt: str,
        max_output_tokens: int,
        json_mode: bool,
    ) -> AIResponse:
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens,
        }
        if json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self.client.chat.completions.create(**request_kwargs)
        raw = response.model_dump()
        choices = getattr(response, "choices", None) or []
        text = ""
        if choices:
            message = getattr(choices[0], "message", None)
            text = str(getattr(message, "content", "") or "")
        usage = _extract_usage(response, raw)
        return AIResponse(
            text=text,
            raw=raw,
            usage=usage,
            provider_response_metadata=_build_provider_response_metadata(
                provider=self.provider,
                model=self.model,
                raw=raw,
                usage=usage,
                text=text,
            ),
        )


class OpenAIClient(OpenAICompatibleClient):
    """Legacy OpenAI wrapper; new PostFlow code should prefer build_ai_client."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key or settings.OPENAI_API_KEY,
            model=model or settings.OPENAI_MODEL,
            provider=AI_PROVIDER_OPENAI,
        )


class AnthropicMessagesClient:
    """Synchronous native Anthropic Messages client using the provider-neutral response."""

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
            else int(getattr(settings, "ANTHROPIC_TIMEOUT_SECONDS", settings.OPENAI_TIMEOUT_SECONDS))
        )

    def generate_text(
        self,
        prompt: str,
        max_output_tokens: int = 1200,
        json_mode: bool = False,
        allow_json_mode_fallback: bool = True,
        thinking_mode: str = THINKING_MODE_PROVIDER_DEFAULT,
    ) -> AIResponse:
        thinking_mode = _normalize_thinking_mode(thinking_mode)
        request_body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_output_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if thinking_mode == THINKING_MODE_DISABLED:
            request_body["thinking"] = {"type": "disabled"}
        elif thinking_mode == THINKING_MODE_ADAPTIVE:
            request_body["thinking"] = {"type": "adaptive"}
        elif thinking_mode != THINKING_MODE_PROVIDER_DEFAULT:
            raise ValueError(f"unsupported thinking_mode: {thinking_mode}")
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

        request_failed = False
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw_bytes = response.read()
        except (HTTPError, URLError, TimeoutError, OSError):
            request_failed = True
        if request_failed:
            raise RuntimeError("anthropic provider request failed") from None

        invalid_json = False
        try:
            raw = json.loads(raw_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_json = True
        if invalid_json:
            raise RuntimeError("anthropic provider response was not valid JSON") from None

        text = _extract_anthropic_text(raw)
        usage = _extract_anthropic_usage(raw)
        return AIResponse(
            text=text,
            raw=raw,
            usage=usage,
            provider_response_metadata=_build_anthropic_provider_response_metadata(
                raw=raw,
                model=self.model,
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
    if normalized_thinking_mode not in SUPPORTED_THINKING_MODES:
        return (
            f"unsupported {stage_name} thinking_mode: "
            f"{normalized_thinking_mode}"
        )
    if (
        normalized_provider != AI_PROVIDER_ANTHROPIC
        and normalized_thinking_mode != THINKING_MODE_PROVIDER_DEFAULT
    ):
        return (
            f"unsupported {stage_name} thinking_mode for provider "
            f"{normalized_provider}: {normalized_thinking_mode}"
        )
    return None


def build_ai_client(provider: str, model: str) -> OpenAICompatibleClient:
    normalized_model = str(model or "").strip()
    configuration_error = get_ai_client_configuration_error(provider, normalized_model)
    if configuration_error is not None:
        raise ValueError(configuration_error)
    config = get_ai_provider_config(provider)
    if config.provider == AI_PROVIDER_ANTHROPIC:
        return AnthropicMessagesClient(
            api_key=config.api_key,
            model=normalized_model,
            timeout=int(getattr(settings, "ANTHROPIC_TIMEOUT_SECONDS", settings.OPENAI_TIMEOUT_SECONDS)),
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
    return str(thinking_mode or THINKING_MODE_PROVIDER_DEFAULT).strip().lower()


def _get_provider_thinking_mode_error(
    *,
    provider: str,
    thinking_mode: str | None,
) -> str | None:
    return get_ai_provider_thinking_mode_error(
        provider=provider,
        thinking_mode=thinking_mode,
    )


def _extract_anthropic_text(raw: dict[str, Any]) -> str:
    content = raw.get("content", []) if isinstance(raw, dict) else []
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
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
    *,
    raw: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    usage = raw.get("usage", {}) if isinstance(raw, dict) else {}
    content = raw.get("content", []) if isinstance(raw, dict) else []
    content_block_types: list[str] = []
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = _safe_metadata_text(block.get("type"))
            if block_type is not None:
                content_block_types.append(block_type)
    return _provider_response_metadata(
        provider=AI_PROVIDER_ANTHROPIC,
        model=model,
        stop_reason=_safe_metadata_text(raw.get("stop_reason")),
        content_block_types=content_block_types,
        input_tokens=_safe_metadata_int(usage.get("input_tokens"))
        if isinstance(usage, dict)
        else None,
        output_tokens=_safe_metadata_int(usage.get("output_tokens"))
        if isinstance(usage, dict)
        else None,
        thinking_tokens=_extract_thinking_tokens(usage),
    )


def _build_provider_response_metadata(
    *,
    provider: str,
    model: str,
    raw: dict[str, Any],
    usage: dict[str, int | None],
    text: str,
) -> dict[str, Any]:
    return _provider_response_metadata(
        provider=provider,
        model=model,
        stop_reason=_extract_openai_compatible_stop_reason(raw),
        content_block_types=_extract_openai_compatible_content_block_types(raw, text),
        input_tokens=_safe_metadata_int(usage.get("prompt_tokens")),
        output_tokens=_safe_metadata_int(usage.get("completion_tokens")),
        thinking_tokens=None,
    )


def _provider_response_metadata(
    *,
    provider: str,
    model: str,
    stop_reason: str | None,
    content_block_types: list[str],
    input_tokens: int | None,
    output_tokens: int | None,
    thinking_tokens: int | None,
) -> dict[str, Any]:
    return {
        "provider": _safe_metadata_text(provider),
        "model": _safe_metadata_text(model),
        "stop_reason": stop_reason,
        "content_block_types": list(content_block_types[:20]),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
    }


def _extract_openai_compatible_stop_reason(raw: dict[str, Any]) -> str | None:
    choices = raw.get("choices") if isinstance(raw, dict) else None
    if isinstance(choices, list) and choices:
        first_choice = choices[0]
        if isinstance(first_choice, dict):
            return _safe_metadata_text(first_choice.get("finish_reason"))
    incomplete_reason = _extract_openai_responses_incomplete_reason(raw)
    if incomplete_reason is not None:
        return incomplete_reason
    return _safe_metadata_text(raw.get("status")) if isinstance(raw, dict) else None


def _extract_openai_responses_incomplete_reason(raw: dict[str, Any]) -> str | None:
    if not isinstance(raw, dict):
        return None
    if _safe_metadata_text(raw.get("status")) != "incomplete":
        return None
    incomplete_details = raw.get("incomplete_details")
    if not isinstance(incomplete_details, dict):
        return None
    return _safe_metadata_text(incomplete_details.get("reason"))


def _extract_openai_compatible_content_block_types(
    raw: dict[str, Any],
    text: str,
) -> list[str]:
    output = raw.get("output") if isinstance(raw, dict) else None
    block_types: list[str] = []
    if isinstance(output, list):
        for output_item in output:
            if not isinstance(output_item, dict):
                continue
            content = output_item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                block_type = _safe_metadata_text(block.get("type"))
                if block_type is not None:
                    block_types.append(block_type)
    if block_types:
        return block_types
    return ["text"] if str(text or "").strip() else []


def _extract_thinking_tokens(usage: Any) -> int | None:
    if not isinstance(usage, dict):
        return None
    direct_value = _safe_metadata_int(usage.get("thinking_tokens"))
    if direct_value is not None:
        return direct_value
    details = usage.get("output_tokens_details")
    if isinstance(details, dict):
        return _safe_metadata_int(details.get("thinking_tokens"))
    return None


def _safe_metadata_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    if value < 0:
        return None
    return value


def _safe_metadata_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text[:120]


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
