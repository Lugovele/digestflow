"""Provider execution boundary for the LinkedIn Candidate Writer prompt.

This module executes an already-rendered Candidate Writer prompt and captures
the raw provider response. It does not parse JSON, build CandidateWriterOutput,
validate FinalPostPayload, route decisions, repair payloads, persist data, or
connect to runtime packaging.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from django.conf import settings

from apps.ai.client import (
    AI_THINKING_MODE_PROVIDER_DEFAULT,
    build_ai_client,
    get_ai_provider_thinking_mode_error,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    get_final_post_role_provider_model_policy_failure,
    get_final_post_role_thinking_mode,
)
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)


DEFAULT_CANDIDATE_WRITER_MAX_OUTPUT_TOKENS = 1200
STAGE_NAME = "candidate writer"
EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT = "MAX_TOKENS_BEFORE_TEXT"
PROVIDER_RESPONSE_METADATA_FIELDS = (
    "provider",
    "model",
    "stop_reason",
    "content_block_types",
    "input_tokens",
    "output_tokens",
    "thinking_tokens",
)


@dataclass(frozen=True)
class CandidateWriterExecutionRequest:
    rendered_prompt_input: CandidateWriterPromptRender
    prompt_text: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_CANDIDATE_WRITER_MAX_OUTPUT_TOKENS
    thinking_mode: str = AI_THINKING_MODE_PROVIDER_DEFAULT
    execution_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rendered_prompt_input": self.rendered_prompt_input.to_dict(),
            "prompt_text": self.prompt_text,
            "provider": self.provider,
            "model": self.model,
            "max_output_tokens": self.max_output_tokens,
            "thinking_mode": self.thinking_mode,
            "execution_metadata": copy.deepcopy(self.execution_metadata),
        }


@dataclass(frozen=True)
class CandidateWriterRawResponse:
    raw_text: str
    provider: str
    model: str
    prompt_metadata: PromptMetadata | None = None
    usage: dict[str, Any] | None = None
    raw_provider_response: dict[str, Any] | None = None
    provider_response_metadata: dict[str, Any] | None = None
    empty_text_classification: str | None = None
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
        if self.empty_text_classification is not None:
            result["empty_text_classification"] = self.empty_text_classification
        if self.execution_error is not None:
            result["execution_error"] = self.execution_error
        if self.execution_metadata is not None:
            result["execution_metadata"] = copy.deepcopy(self.execution_metadata)
        return result


def build_candidate_writer_execution_request(
    rendered_prompt_input: CandidateWriterPromptRender,
    *,
    prompt_text: str,
    provider: str | None = None,
    model: str | None = None,
    max_output_tokens: int = DEFAULT_CANDIDATE_WRITER_MAX_OUTPUT_TOKENS,
    thinking_mode: str | None = None,
    execution_metadata: dict[str, Any] | None = None,
) -> CandidateWriterExecutionRequest:
    resolved_provider = _resolve_provider(provider)
    resolved_model = _resolve_model(model)
    resolved_thinking_mode = (
        str(thinking_mode or "").strip().lower()
        if thinking_mode is not None
        else _resolve_thinking_mode(
            provider=resolved_provider,
            model=resolved_model,
        )
    )
    return CandidateWriterExecutionRequest(
        rendered_prompt_input=rendered_prompt_input,
        prompt_text=prompt_text,
        provider=resolved_provider,
        model=resolved_model,
        max_output_tokens=max_output_tokens,
        thinking_mode=resolved_thinking_mode,
        execution_metadata=copy.deepcopy(execution_metadata),
    )


def execute_candidate_writer_prompt(
    request: CandidateWriterExecutionRequest,
    *,
    client: Any | None = None,
) -> CandidateWriterRawResponse:
    prompt_metadata = _prompt_metadata_from_render(request.rendered_prompt_input)
    execution_metadata = copy.deepcopy(request.execution_metadata)
    execution_error = _execution_request_error(request)
    if execution_error is not None:
        return CandidateWriterRawResponse(
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
            return CandidateWriterRawResponse(
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
            thinking_mode=request.thinking_mode,
        )
    except Exception:  # pragma: no cover - covered with fake failure.
        return CandidateWriterRawResponse(
            raw_text="",
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            execution_error="provider invocation failed",
            execution_metadata=execution_metadata,
        )

    raw_text = response.text
    if raw_text is None or not str(raw_text).strip():
        provider_response_metadata = _sanitize_provider_response_metadata(
            getattr(response, "provider_response_metadata", None)
        )
        return CandidateWriterRawResponse(
            raw_text=str(raw_text or ""),
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            usage=copy.deepcopy(response.usage),
            raw_provider_response=copy.deepcopy(response.raw),
            provider_response_metadata=provider_response_metadata,
            empty_text_classification=_classify_empty_text_response(
                provider_response_metadata
            ),
            execution_error="empty provider response",
            execution_metadata=execution_metadata,
        )

    return CandidateWriterRawResponse(
        raw_text=str(raw_text),
        provider=request.provider,
        model=request.model,
        prompt_metadata=prompt_metadata,
        usage=copy.deepcopy(response.usage),
        raw_provider_response=copy.deepcopy(response.raw),
        provider_response_metadata=_sanitize_provider_response_metadata(
            getattr(response, "provider_response_metadata", None)
        ),
        execution_metadata=execution_metadata,
    )


def _resolve_provider(provider: str | None) -> str:
    resolved = provider if provider is not None else settings.POSTFLOW_POST_PROVIDER
    return str(resolved or "").strip().lower()


def _resolve_model(model: str | None) -> str:
    resolved = model if model is not None else settings.POSTFLOW_POST_MODEL
    return str(resolved or "").strip()


def _resolve_thinking_mode(*, provider: str, model: str) -> str:
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_CANDIDATE_WRITER,
        provider=provider,
        model=model,
    )
    if policy_failure is not None:
        return AI_THINKING_MODE_PROVIDER_DEFAULT
    return get_final_post_role_thinking_mode(
        role=FINAL_POST_ROLE_CANDIDATE_WRITER,
        provider=provider,
        model=model,
    )


def _execution_request_error(request: CandidateWriterExecutionRequest) -> str | None:
    if not request.provider:
        return "missing candidate writer provider"
    if not request.model:
        return "missing candidate writer model"
    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=FINAL_POST_ROLE_CANDIDATE_WRITER,
        provider=request.provider,
        model=request.model,
    )
    if policy_failure is not None:
        return str(policy_failure)
    thinking_mode_error = get_ai_provider_thinking_mode_error(
        provider=request.provider,
        thinking_mode=request.thinking_mode,
        stage_name=STAGE_NAME,
    )
    if thinking_mode_error is not None:
        return thinking_mode_error
    if isinstance(request.max_output_tokens, bool) or not isinstance(
        request.max_output_tokens,
        int,
    ):
        return "invalid candidate writer max_output_tokens: must be a positive integer"
    if request.max_output_tokens <= 0:
        return "invalid candidate writer max_output_tokens: must be a positive integer"
    if not isinstance(request.prompt_text, str) or not request.prompt_text.strip():
        return "missing candidate writer prompt text"
    rendered_input_text = request.rendered_prompt_input.input_text
    if not isinstance(rendered_input_text, str) or not rendered_input_text.strip():
        return "missing candidate writer rendered input text"
    return None


def _build_provider_prompt(request: CandidateWriterExecutionRequest) -> str:
    return f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}"


def _prompt_metadata_from_render(
    render: CandidateWriterPromptRender,
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
        if field_name == "content_block_types":
            sanitized_list = _sanitize_content_block_types(value)
            if sanitized_list:
                sanitized[field_name] = sanitized_list
        elif field_name in ("input_tokens", "output_tokens", "thinking_tokens"):
            if _is_non_negative_int(value):
                sanitized[field_name] = value
        elif isinstance(value, str) and value.strip():
            sanitized[field_name] = value.strip()[:120]
    return sanitized or None


def _sanitize_content_block_types(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    sanitized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            continue
        block_type = item.strip()[:120]
        if block_type and block_type not in sanitized:
            sanitized.append(block_type)
    return sanitized[:20]


def _classify_empty_text_response(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    content_block_types = metadata.get("content_block_types")
    output_tokens = metadata.get("output_tokens")
    thinking_tokens = metadata.get("thinking_tokens")
    if (
        metadata.get("provider") == "anthropic"
        and metadata.get("stop_reason") == "max_tokens"
        and isinstance(content_block_types, list)
        and "thinking" in content_block_types
        and "text" not in content_block_types
        and _is_non_negative_int(output_tokens)
        and _is_non_negative_int(thinking_tokens)
        and thinking_tokens == output_tokens
        and output_tokens > 0
    ):
        return EMPTY_TEXT_CLASSIFICATION_MAX_TOKENS_BEFORE_TEXT
    return None


def _is_non_negative_int(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0
