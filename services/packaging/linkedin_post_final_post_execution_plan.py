"""Canonical execution-plan preflight for final LinkedIn post model roles.

This module validates role/provider/model/key selections before provider
execution. It does not execute prompts, call providers, route attempts, repair
text, inspect payloads, or persist data.
"""
from __future__ import annotations

from dataclasses import dataclass

from django.conf import settings

from apps.ai.client import (
    get_ai_client_configuration_error,
    get_ai_provider_model_error,
    get_ai_provider_thinking_mode_error,
)
from services.packaging.linkedin_post_model_role_policy import (
    get_final_post_role_provider_model_policy_failure,
    get_final_post_role_thinking_mode,
    normalize_final_post_model_role,
)


@dataclass(frozen=True)
class FinalPostExecutionRoleSelection:
    role: str
    provider: str | None
    model: str | None
    stage: str
    failure_code: str
    stage_label: str
    validate_key: bool = False


@dataclass(frozen=True)
class FinalPostExecutionRoleConfig:
    role: str
    provider: str
    model: str
    thinking_mode: str
    validation_status: str = "valid"

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "thinking_mode": self.thinking_mode,
            "validation_status": self.validation_status,
        }


@dataclass(frozen=True)
class FinalPostExecutionPlanPreflightFailure:
    role: str
    provider: str
    model: str
    stage: str
    failure_code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "stage": self.stage,
            "failure_code": self.failure_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class FinalPostExecutionPlanPreflightResult:
    valid: bool
    role_configs: tuple[FinalPostExecutionRoleConfig, ...]
    failures: tuple[FinalPostExecutionPlanPreflightFailure, ...]

    def first_failure(self) -> FinalPostExecutionPlanPreflightFailure | None:
        return self.failures[0] if self.failures else None

    def role_diagnostics(self) -> list[dict[str, str]]:
        diagnostics = [config.to_dict() for config in self.role_configs]
        diagnostics.extend(
            {
                "role": failure.role,
                "provider": failure.provider,
                "model": failure.model,
                "thinking_mode": "",
                "validation_status": "failed",
            }
            for failure in self.failures
        )
        return diagnostics

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "role_configs": [config.to_dict() for config in self.role_configs],
            "failures": [failure.to_dict() for failure in self.failures],
        }


def preflight_final_post_execution_plan(
    selections: tuple[FinalPostExecutionRoleSelection, ...]
    | list[FinalPostExecutionRoleSelection],
) -> FinalPostExecutionPlanPreflightResult:
    """Validate all selected final-post model roles before any provider call."""

    role_configs: list[FinalPostExecutionRoleConfig] = []
    failures: list[FinalPostExecutionPlanPreflightFailure] = []
    for selection in selections:
        failure = _validate_role_selection(selection)
        if failure is not None:
            failures.append(failure)
            continue
        provider = _normalize_provider(selection.provider)
        model = _normalize_model(selection.model)
        role_configs.append(
            FinalPostExecutionRoleConfig(
                role=normalize_final_post_model_role(selection.role),
                provider=provider,
                model=model,
                thinking_mode=get_final_post_role_thinking_mode(
                    role=selection.role,
                    provider=provider,
                    model=model,
                ),
            )
        )
    return FinalPostExecutionPlanPreflightResult(
        valid=not failures,
        role_configs=tuple(role_configs),
        failures=tuple(failures),
    )


def _validate_role_selection(
    selection: FinalPostExecutionRoleSelection,
) -> FinalPostExecutionPlanPreflightFailure | None:
    role = normalize_final_post_model_role(selection.role)
    provider = _normalize_provider(selection.provider)
    model = _normalize_model(selection.model)
    if not provider:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=f"missing {selection.stage_label} provider",
        )
    if not model:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=f"missing {selection.stage_label} model",
        )

    provider_model_error = get_ai_provider_model_error(
        provider,
        model,
        stage_name=selection.stage_label,
    )
    if provider_model_error is not None:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=provider_model_error,
        )

    policy_failure = get_final_post_role_provider_model_policy_failure(
        role=role,
        provider=provider,
        model=model,
    )
    if policy_failure is not None:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=str(policy_failure),
        )

    try:
        thinking_mode = get_final_post_role_thinking_mode(
            role=role,
            provider=provider,
            model=model,
        )
    except ValueError as exc:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=str(exc),
        )

    thinking_error = get_ai_provider_thinking_mode_error(
        provider=provider,
        thinking_mode=thinking_mode,
        stage_name=selection.stage_label,
    )
    if thinking_error is not None:
        return _failure(
            selection,
            role=role,
            provider=provider,
            model=model,
            message=thinking_error,
        )

    if selection.validate_key:
        configuration_error = get_ai_client_configuration_error(
            provider,
            model,
            stage_name=selection.stage_label,
        )
        if configuration_error is not None:
            return _failure(
                selection,
                role=role,
                provider=provider,
                model=model,
                message=configuration_error,
            )
    return None


def _failure(
    selection: FinalPostExecutionRoleSelection,
    *,
    role: str,
    provider: str,
    model: str,
    message: str,
) -> FinalPostExecutionPlanPreflightFailure:
    return FinalPostExecutionPlanPreflightFailure(
        role=role,
        provider=provider,
        model=model,
        stage=selection.stage,
        failure_code=selection.failure_code,
        message=_safe_message(message),
    )


def _normalize_provider(provider: str | None) -> str:
    resolved = provider if provider is not None else settings.POSTFLOW_POST_PROVIDER
    return str(resolved or "").strip().lower()


def _normalize_model(model: str | None) -> str:
    resolved = model if model is not None else settings.POSTFLOW_POST_MODEL
    return str(resolved or "").strip()


def _safe_message(message: object) -> str:
    text = str(message or "")
    for marker in ("sk-", "Bearer ", "Authorization:", "x-api-key"):
        if marker in text:
            return "redacted provider/configuration message"
    return text
