"""Небольшой AI-адаптер для сервисного слоя."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings
from openai import OpenAI


AI_PROVIDER_OPENAI = "openai"
AI_PROVIDER_GEMINI = "gemini"
SUPPORTED_AI_PROVIDERS = (AI_PROVIDER_OPENAI, AI_PROVIDER_GEMINI)
GEMINI_OPENAI_COMPATIBLE_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/openai/"
)
GEMINI_SUPPORTED_MODELS = ("gemini-3.6-flash",)
PLACEHOLDER_API_KEYS = {
    "",
    "sk-your-key",
    "your-openai-api-key-here",
    "your-gemini-api-key-here",
}


@dataclass(frozen=True)
class AIResponse:
    text: str
    raw: dict[str, Any]
    usage: dict[str, int | None]


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
    ) -> AIResponse:
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
        return AIResponse(text=text, raw=raw, usage=_extract_usage(response, raw))

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
        return AIResponse(text=text, raw=raw, usage=_extract_usage(response, raw))


class OpenAIClient(OpenAICompatibleClient):
    """Legacy OpenAI wrapper preserving existing OpenAI defaults."""

    def __init__(self, api_key: str | None = None, model: str | None = None) -> None:
        super().__init__(
            api_key=api_key or settings.OPENAI_API_KEY,
            model=model or settings.OPENAI_MODEL,
            provider=AI_PROVIDER_OPENAI,
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
    if normalized_provider == AI_PROVIDER_OPENAI and normalized_model.startswith("gemini-"):
        return (
            f"{stage_name} provider/model mismatch: provider openai cannot use "
            "a Gemini model"
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


def build_ai_client(provider: str, model: str) -> OpenAICompatibleClient:
    normalized_model = str(model or "").strip()
    configuration_error = get_ai_client_configuration_error(provider, normalized_model)
    if configuration_error is not None:
        raise ValueError(configuration_error)
    config = get_ai_provider_config(provider)
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
    return "OPENAI_API_KEY"


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
