"""Canonical provider/model policy for final LinkedIn post roles.

This module is intentionally declarative. It does not construct clients, execute
prompts, call providers, route attempts, or inspect payloads.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from apps.ai.client import (
    AI_PROVIDER_ANTHROPIC,
    THINKING_MODE_DISABLED,
    THINKING_MODE_PROVIDER_DEFAULT,
)


FINAL_POST_ROLE_CANDIDATE_WRITER = "candidate_writer"
FINAL_POST_ROLE_SEMANTIC_GROUNDING = "semantic_grounding"
FINAL_POST_ROLE_QUALITY_EVALUATOR = "quality_evaluator"
FINAL_POST_ROLE_REPAIR_WRITER = "repair_writer"

FINAL_POST_MODEL_ROLES = (
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
)

FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_CODE = (
    "final_post_role_provider_model_policy_failure"
)
FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX = (
    "unsupported PostFlow final post role/provider/model"
)
SAFE_POLICY_VALUE_MAX_LENGTH = 120

FINAL_POST_ROLE_PROVIDER_MODEL_POLICY: Mapping[str, Mapping[str, tuple[str, ...]]] = (
    MappingProxyType(
        {
            FINAL_POST_ROLE_CANDIDATE_WRITER: MappingProxyType(
                {
                    "openai": ("gpt-4.1-2025-04-14",),
                    "gemini": ("gemini-3.6-flash",),
                    "anthropic": ("claude-sonnet-5",),
                }
            ),
            FINAL_POST_ROLE_SEMANTIC_GROUNDING: MappingProxyType(
                {
                    "openai": ("gpt-4.1-2025-04-14",),
                    "gemini": ("gemini-3.6-flash",),
                    "anthropic": ("claude-sonnet-5",),
                }
            ),
            FINAL_POST_ROLE_QUALITY_EVALUATOR: MappingProxyType(
                {
                    "openai": ("gpt-4.1-2025-04-14",),
                }
            ),
            FINAL_POST_ROLE_REPAIR_WRITER: MappingProxyType(
                {
                    "openai": ("gpt-4.1-2025-04-14",),
                }
            ),
        }
    )
)


@dataclass(frozen=True)
class FinalPostRoleProviderModelPolicyFailure:
    code: str
    role: str
    provider: str
    model: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "message": self.message,
        }

    def __str__(self) -> str:
        return self.message


def _safe_policy_value(value: str) -> str:
    normalized = " ".join(str(value or "").split())
    if len(normalized) > SAFE_POLICY_VALUE_MAX_LENGTH:
        return normalized[: SAFE_POLICY_VALUE_MAX_LENGTH - 3] + "..."
    return normalized


def normalize_final_post_model_role(role: str) -> str:
    return _safe_policy_value(role).lower()


def get_allowed_final_post_provider_models(role: str) -> dict[str, tuple[str, ...]]:
    normalized_role = normalize_final_post_model_role(role)
    allowed = FINAL_POST_ROLE_PROVIDER_MODEL_POLICY.get(normalized_role)
    if allowed is None:
        return {}
    return {provider: tuple(models) for provider, models in allowed.items()}


def get_final_post_role_provider_model_policy_failure(
    *,
    role: str,
    provider: str,
    model: str,
) -> FinalPostRoleProviderModelPolicyFailure | None:
    normalized_role = normalize_final_post_model_role(role)
    normalized_provider = _safe_policy_value(provider).lower()
    normalized_model = _safe_policy_value(model)

    allowed_by_provider = FINAL_POST_ROLE_PROVIDER_MODEL_POLICY.get(normalized_role)
    allowed_models = (
        allowed_by_provider.get(normalized_provider)
        if allowed_by_provider is not None
        else None
    )
    if allowed_models is not None and normalized_model in allowed_models:
        return None

    return FinalPostRoleProviderModelPolicyFailure(
        code=FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_CODE,
        role=normalized_role,
        provider=normalized_provider,
        model=normalized_model,
        message=(
            f"{FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX}: "
            f"role={normalized_role} provider={normalized_provider} "
            f"model={normalized_model}"
        ),
    )


def validate_final_post_role_provider_model(
    *,
    role: str,
    provider: str,
    model: str,
) -> None:
    failure = get_final_post_role_provider_model_policy_failure(
        role=role,
        provider=provider,
        model=model,
    )
    if failure is not None:
        raise ValueError(str(failure))


def get_final_post_role_thinking_mode(
    *,
    role: str,
    provider: str,
    model: str,
) -> str:
    """Resolve provider-neutral thinking behavior for a final-post role.

    Candidate Writer disables Anthropic Sonnet 5 thinking so the provider does
    not spend the whole output budget before producing JSON text. Other roles
    keep the provider default until their own prompt runs justify a change.
    """

    normalized_role = normalize_final_post_model_role(role)
    normalized_provider = _safe_policy_value(provider).lower()
    normalized_model = _safe_policy_value(model)

    if (
        normalized_role == FINAL_POST_ROLE_CANDIDATE_WRITER
        and normalized_provider == AI_PROVIDER_ANTHROPIC
        and normalized_model == "claude-sonnet-5"
    ):
        return THINKING_MODE_DISABLED
    return THINKING_MODE_PROVIDER_DEFAULT
