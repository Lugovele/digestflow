"""Provider execution boundary for the semantic grounding evaluator.

This module executes an already-rendered semantic-grounding prompt and captures
the raw provider response. It does not parse JSON, normalize grounding reviews,
route decisions, repair payloads, persist data, or connect to runtime packaging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import build_ai_client
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    get_final_post_role_provider_model_policy_failure,
)


DEFAULT_MAX_OUTPUT_TOKENS = 2400
MIN_MAX_OUTPUT_TOKENS = 2000
DEFAULT_JSON_MODE = True
STAGE_NAME = "semantic grounding"


@dataclass(frozen=True)
class SemanticGroundingExecutionRequest:
    rendered_prompt_input: object
    prompt_text: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    json_mode: bool = DEFAULT_JSON_MODE
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_prompt_input": self.rendered_prompt_input.to_dict(),
            "prompt_text": self.prompt_text,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "json_mode": self.json_mode,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }


@dataclass(frozen=True)
class SemanticGroundingRawResponse:
    raw_text: str
    provider: str
    model: str
    prompt_metadata: PromptMetadata | None = None
    usage: dict[str, Any] | None = None
    raw_provider_response: dict[str, Any] | None = None
    execution_error: str | None = None

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
        if self.execution_error is not None:
            result["execution_error"] = self.execution_error
        return result


def build_semantic_grounding_execution_request(
    rendered_prompt_input: object,
    *,
    prompt_text: str,
    provider: str | None = None,
    model: str | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    json_mode: bool = DEFAULT_JSON_MODE,
    execution_metadata: dict[str, Any] | None = None,
) -> SemanticGroundingExecutionRequest:
    return SemanticGroundingExecutionRequest(
        rendered_prompt_input=rendered_prompt_input,
        prompt_text=prompt_text,
        provider=_resolve_provider(provider),
        model=_resolve_model(model),
        max_output_tokens=max_output_tokens,
        json_mode=json_mode,
        execution_metadata=copy.deepcopy(execution_metadata),
    )


def execute_semantic_grounding_prompt(
    request: SemanticGroundingExecutionRequest,
    *,
    client: Any | None = None,
) -> SemanticGroundingRawResponse:
    prompt_metadata = _prompt_metadata_from_render(request.rendered_prompt_input)
    execution_error = get_semantic_grounding_execution_request_error(request)
    if execution_error is not None:
        return SemanticGroundingRawResponse(
            raw_text="",
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            execution_error=execution_error,
        )

    prompt = f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}"
    if client is not None:
        text_client = client
    else:
        try:
            text_client = build_ai_client(
                provider=request.provider,
                model=request.model,
            )
        except ValueError as exc:
            return SemanticGroundingRawResponse(
                raw_text="",
                provider=request.provider,
                model=request.model,
                prompt_metadata=prompt_metadata,
                execution_error=str(exc),
            )

    try:
        response = text_client.generate_text(
            prompt=prompt,
            max_output_tokens=request.max_output_tokens,
            json_mode=request.json_mode,
            allow_json_mode_fallback=False,
        )
    except Exception:  # pragma: no cover - covered with fake failure.
        return SemanticGroundingRawResponse(
            raw_text="",
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            execution_error="provider invocation failed",
        )

    raw_text = response.text
    if raw_text is None or not str(raw_text).strip():
        return SemanticGroundingRawResponse(
            raw_text=str(raw_text or ""),
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            usage=copy.deepcopy(response.usage),
            raw_provider_response=copy.deepcopy(response.raw),
            execution_error="empty provider response",
        )

    return SemanticGroundingRawResponse(
        raw_text=str(raw_text),
        provider=request.provider,
        model=request.model,
        prompt_metadata=prompt_metadata,
        usage=copy.deepcopy(response.usage),
        raw_provider_response=copy.deepcopy(response.raw),
    )


def get_semantic_grounding_execution_request_error(
    request: SemanticGroundingExecutionRequest,
) -> str | None:
    if not request.provider:
        return "missing semantic grounding provider"
    if not request.model:
        return "missing semantic grounding model"
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
        provider=request.provider,
        model=request.model,
    )
    if policy_failure is not None:
        return str(policy_failure)
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid semantic grounding max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid semantic grounding max_output_tokens: must be a positive integer"
    if request.max_output_tokens < MIN_MAX_OUTPUT_TOKENS:
        return (
            "invalid semantic grounding max_output_tokens: must be at least "
            f"{MIN_MAX_OUTPUT_TOKENS}"
        )
    if not isinstance(request.json_mode, bool):
        return "invalid semantic grounding json_mode: must be a boolean"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing semantic grounding prompt text"
    rendered_input_text = getattr(request.rendered_prompt_input, "input_text", None)
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing semantic grounding rendered input text"
    return None


def _resolve_provider(provider: str | None) -> str:
    resolved = provider if provider is not None else settings.POSTFLOW_POST_PROVIDER
    return str(resolved or "").strip().lower()


def _resolve_model(model: str | None) -> str:
    resolved = model if model is not None else settings.POSTFLOW_POST_MODEL
    return str(resolved or "").strip()


def _prompt_metadata_from_render(render: object) -> PromptMetadata | None:
    if (
        getattr(render, "prompt_name", None) is None
        and getattr(render, "prompt_version", None) is None
        and getattr(render, "prompt_path", None) is None
    ):
        return None
    return PromptMetadata(
        prompt_name=getattr(render, "prompt_name", "") or "",
        prompt_version=getattr(render, "prompt_version", "") or "",
        prompt_path=getattr(render, "prompt_path", None),
    )
