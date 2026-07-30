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

from apps.ai.client import OpenAIClient
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)


DEFAULT_CANDIDATE_WRITER_MAX_OUTPUT_TOKENS = 1200
SUPPORTED_PROVIDER = "openai"


@dataclass(frozen=True)
class CandidateWriterExecutionRequest:
    rendered_prompt_input: CandidateWriterPromptRender
    prompt_text: str
    provider: str
    model: str
    max_output_tokens: int = DEFAULT_CANDIDATE_WRITER_MAX_OUTPUT_TOKENS
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
class CandidateWriterRawResponse:
    raw_text: str
    provider: str
    model: str
    prompt_metadata: PromptMetadata | None = None
    usage: dict[str, Any] | None = None
    raw_provider_response: dict[str, Any] | None = None
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
    execution_metadata: dict[str, Any] | None = None,
) -> CandidateWriterExecutionRequest:
    return CandidateWriterExecutionRequest(
        rendered_prompt_input=rendered_prompt_input,
        prompt_text=prompt_text,
        provider=_resolve_provider(provider),
        model=_resolve_model(model),
        max_output_tokens=max_output_tokens,
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
    try:
        text_client = client if client is not None else OpenAIClient(model=request.model)
        response = text_client.generate_text(
            prompt=prompt,
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
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
        return CandidateWriterRawResponse(
            raw_text=str(raw_text or ""),
            provider=request.provider,
            model=request.model,
            prompt_metadata=prompt_metadata,
            usage=copy.deepcopy(response.usage),
            raw_provider_response=copy.deepcopy(response.raw),
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
        execution_metadata=execution_metadata,
    )


def _resolve_provider(provider: str | None) -> str:
    resolved = provider if provider is not None else settings.POSTFLOW_POST_PROVIDER
    return str(resolved or "").strip().lower()


def _resolve_model(model: str | None) -> str:
    resolved = model if model is not None else settings.POSTFLOW_POST_MODEL
    return str(resolved or "").strip()


def _execution_request_error(request: CandidateWriterExecutionRequest) -> str | None:
    if not request.provider:
        return "missing candidate writer provider"
    if request.provider != SUPPORTED_PROVIDER:
        return f"unsupported candidate writer provider: {request.provider}"
    if not request.model:
        return "missing candidate writer model"
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
