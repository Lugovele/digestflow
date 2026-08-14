"""Provider execution boundary for the LinkedIn Repair Writer prompt.

This module executes an already-rendered repair prompt and captures the raw
provider response. It does not parse JSON, adapt payloads, run deterministic
gates, score quality, route decisions, persist data, or connect to runtime
packaging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import build_ai_client
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_REPAIR_WRITER,
    get_final_post_role_provider_model_policy_failure,
)
from services.packaging.linkedin_post_provider_diagnostics import (
    build_safe_provider_error_diagnostics,
)
from services.packaging.linkedin_post_prompt_renderers import RepairWriterPromptRender


DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS = 1800
STAGE_NAME = "repair writer"
PROVIDER_RESPONSE_METADATA_FIELDS = (
    "provider",
    "model",
    "provider_finish_reason",
    "provider_stop_reason",
    "provider_max_output_tokens",
    "provider_reported_output_tokens",
    "provider_output_limit_reached",
    "provider_prompt_tokens",
    "provider_visible_output_tokens",
    "provider_total_tokens",
    "provider_hidden_output_tokens",
    "provider_combined_output_tokens",
    "provider_output_budget_utilization_percent",
    "provider_reasoning_tokens",
    "provider_thinking_tokens",
)


@dataclass(frozen=True)
class RepairWriterExecutionRequest:
    rendered_prompt_input: RepairWriterPromptRender
    prompt_text: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_prompt_input": self.rendered_prompt_input.to_dict(),
            "prompt_text": self.prompt_text,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }


@dataclass(frozen=True)
class RepairWriterRawResponse:
    raw_text: str
    provider: str
    model: str
    prompt_metadata: PromptMetadata | None = None
    usage: dict[str, Any] | None = None
    raw_provider_response: dict[str, Any] | None = None
    provider_response_metadata: dict[str, Any] | None = None
    execution_diagnostics: dict[str, Any] | None = None
    execution_error: str | None = None
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "raw_text": self.raw_text,
            "provider": self.provider,
            "model": self.model,
        }
        if self.prompt_metadata is not None:
            result["prompt_metadata"] = self.prompt_metadata.to_dict()
        if self.usage is not None:
            result["usage"] = copy.deepcopy(self.usage)
        if self.raw_provider_response is not None:
            result["raw_provider_response"] = copy.deepcopy(self.raw_provider_response)
        if self.provider_response_metadata is not None:
            result["provider_response_metadata"] = copy.deepcopy(
                self.provider_response_metadata
            )
        if self.execution_diagnostics is not None:
            result["execution_diagnostics"] = copy.deepcopy(self.execution_diagnostics)
        if self.execution_error is not None:
            result["execution_error"] = self.execution_error
        if self.execution_metadata is not None:
            result["execution_metadata"] = copy.deepcopy(self.execution_metadata)
        return result


def build_repair_writer_execution_request(
    rendered_prompt_input: RepairWriterPromptRender,
    *,
    prompt_text: str,
    provider: str | None = None,
    model: str | None = None,
    max_output_tokens: int = DEFAULT_REPAIR_WRITER_MAX_OUTPUT_TOKENS,
    execution_metadata: dict[str, Any] | None = None,
) -> RepairWriterExecutionRequest:
    return RepairWriterExecutionRequest(
        rendered_prompt_input=rendered_prompt_input,
        prompt_text=prompt_text,
        provider=_resolve_provider(provider),
        model=_resolve_model(model),
        max_output_tokens=max_output_tokens,
        execution_metadata=copy.deepcopy(execution_metadata),
    )


def execute_repair_writer_prompt(
    request: RepairWriterExecutionRequest,
    *,
    client: Any | None = None,
) -> RepairWriterRawResponse:
    prompt_metadata = _prompt_metadata_from_render(request.rendered_prompt_input)
    execution_metadata = copy.deepcopy(request.execution_metadata)
    execution_error = _execution_request_error(request)
    if execution_error is not None:
        return RepairWriterRawResponse(
            raw_text="",
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            execution_error=execution_error,
            execution_metadata=execution_metadata,
        )

    prompt = _build_provider_prompt(request)
    if client is not None:
        text_client = client
    else:
        try:
            text_client = build_ai_client(
                provider=request.provider,
                model=request.model,
            )
        except ValueError as exc:
            return RepairWriterRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                prompt_metadata=prompt_metadata,
                execution_error=str(exc),
                execution_metadata=execution_metadata,
            )

    try:
        response = text_client.generate_text(
            prompt=prompt,
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
        )
    except Exception as exc:  # pragma: no cover - covered with fake failure.
        return RepairWriterRawResponse(
            raw_text="",
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            execution_diagnostics=build_safe_provider_error_diagnostics(
                exc,
                provider=request.provider,
                model=request.model,
                endpoint_family="responses"
                if request.provider == "openai"
                else "openai_compatible_chat"
                if request.provider == "gemini"
                else "messages"
                if request.provider == "anthropic"
                else "unknown",
            ),
            execution_error="provider invocation failed",
            execution_metadata=execution_metadata,
        )

    raw_text = response.text
    provider_response_metadata = _sanitize_provider_response_metadata(
        getattr(response, "provider_response_metadata", None)
    )
    if raw_text is None or not str(raw_text).strip():
        return RepairWriterRawResponse(
            raw_text=str(raw_text or ""),
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            usage=copy.deepcopy(response.usage),
            raw_provider_response=copy.deepcopy(response.raw),
            provider_response_metadata=provider_response_metadata,
            execution_error="empty provider response",
            execution_metadata=execution_metadata,
        )

    return RepairWriterRawResponse(
        raw_text=str(raw_text),
        provider=request.provider,
        model=request.model,
        prompt_metadata=prompt_metadata,
        usage=copy.deepcopy(response.usage),
        raw_provider_response=copy.deepcopy(response.raw),
        provider_response_metadata=provider_response_metadata,
        execution_metadata=execution_metadata,
    )


def _resolve_provider(provider: str | None) -> str:
    resolved = provider if provider is not None else settings.POSTFLOW_POST_PROVIDER
    return str(resolved or "").strip().lower()


def _resolve_model(model: str | None) -> str:
    resolved = model if model is not None else settings.POSTFLOW_POST_MODEL
    return str(resolved or "").strip()


def _execution_request_error(request: RepairWriterExecutionRequest) -> str | None:
    if not request.provider:
        return "missing repair writer provider"
    if not request.model:
        return "missing repair writer model"
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_REPAIR_WRITER,
        provider=request.provider,
        model=request.model,
    )
    if policy_failure is not None:
        return str(policy_failure)
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid repair writer max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid repair writer max_output_tokens: must be a positive integer"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing repair writer prompt text"
    rendered_input_text = request.rendered_prompt_input.input_text
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing repair writer rendered input text"
    return None


def _build_provider_prompt(request: RepairWriterExecutionRequest) -> str:
    return f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}"


def _prompt_metadata_from_render(
    render: RepairWriterPromptRender,
) -> PromptMetadata | None:
    if (
        render.prompt_name is None
        and render.prompt_version is None
        and render.prompt_path is None
    ):
        return None
    return PromptMetadata(
        prompt_name=render.prompt_name or "",
        prompt_version=render.prompt_version or "",
        prompt_path=render.prompt_path,
    )


def _sanitize_provider_response_metadata(metadata: Any) -> dict[str, Any] | None:
    if not isinstance(metadata, dict):
        return None
    sanitized: dict[str, Any] = {}
    for field_name in PROVIDER_RESPONSE_METADATA_FIELDS:
        value = metadata.get(field_name)
        if field_name in (
            "provider_max_output_tokens",
            "provider_reported_output_tokens",
            "provider_prompt_tokens",
            "provider_visible_output_tokens",
            "provider_total_tokens",
            "provider_hidden_output_tokens",
            "provider_combined_output_tokens",
            "provider_reasoning_tokens",
            "provider_thinking_tokens",
        ):
            if _is_non_negative_int(value):
                sanitized[field_name] = value
        elif field_name == "provider_output_budget_utilization_percent":
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                sanitized[field_name] = round(float(value), 2)
        elif field_name == "provider_output_limit_reached":
            if value is None or isinstance(value, bool):
                sanitized[field_name] = value
        elif isinstance(value, str) and value.strip():
            sanitized[field_name] = value.strip()[:120]
    return sanitized or None


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
