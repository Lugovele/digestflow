"""Internal orchestration for LinkedIn post synthesis artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from django.conf import settings

from apps.ai.client import estimate_cost_usd
from apps.digests.models import Digest
from services.packaging.validators import (
    ContentPackageValidationError,
    validate_content_package_payload,
)


class _PostSynthesisRepairFailure(ContentPackageValidationError):
    def __init__(
        self,
        message: str,
        *,
        repair_prompt: str,
        repair_response_text: str,
        repair_quality_gate: dict[str, Any] | None,
        repair_delta: dict[str, Any] | None,
        concrete_detail_diagnostics: dict[str, Any] | None,
        banned_phrase_diagnostics: dict[str, Any] | None,
    ) -> None:
        super().__init__(message)
        self.repair_prompt = repair_prompt
        self.repair_response_text = repair_response_text
        self.repair_quality_gate = repair_quality_gate
        self.repair_delta = repair_delta
        self.concrete_detail_diagnostics = concrete_detail_diagnostics
        self.banned_phrase_diagnostics = banned_phrase_diagnostics


@dataclass(frozen=True)
class PostSynthesisDependencies:
    generate_source_evidence_pack: Callable[..., Any]
    generate_author_take: Callable[..., Any]
    author_take_quality_issues: Callable[[dict[str, Any]], list[str]]
    author_take_requires_rejection: Callable[[dict[str, Any]], bool]
    generate_post_brief: Callable[..., Any]
    generate_payload: Callable[..., Any]
    normalize_payload: Callable[[dict[str, Any]], dict[str, Any]]
    collect_repairable_payload_issues: Callable[[dict[str, Any]], list[str]]
    evaluate_quality: Callable[[dict[str, Any]], dict[str, Any]]
    evaluate_brief_alignment: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]
    evaluate_post_mechanics: Callable[[dict[str, Any], dict[str, Any] | None], dict[str, Any]]
    build_concrete_detail_diagnostics: Callable[..., dict[str, Any]]
    combined_repair_reasons: Callable[..., list[str]]
    build_banned_phrase_diagnostics: Callable[..., dict[str, Any]]
    quality_gate_requires_repair: Callable[[dict[str, Any]], bool]
    brief_alignment_requires_repair: Callable[[dict[str, Any] | None], bool]
    post_mechanics_requires_repair: Callable[[dict[str, Any] | None], bool]
    generate_editorial_review: Callable[..., Any]
    repair_payload: Callable[..., Any]
    split_long_post_paragraphs: Callable[[str], str]
    evaluate_repair_rewrite_delta: Callable[[dict[str, Any], dict[str, Any], list[str]], dict[str, Any]]
    repair_delta_requires_failure: Callable[[dict[str, Any] | None], bool]
    editorial_review_requires_repair: Callable[[dict[str, Any] | None], bool]
    editorial_review_repair_reasons: Callable[[dict[str, Any] | None], list[str]]
    build_mock_payload: Callable[[Digest, list[dict[str, Any]]], dict[str, Any]]
    build_post_prompt: Callable[..., str]


@dataclass(frozen=True)
class PostSynthesisResult:
    prompt: str
    response_text: str
    payload: dict[str, Any]
    provider: str
    is_mock: bool
    fallback_reason: str
    tokens: dict[str, int | None] | None
    estimated_cost_usd: float | None
    quality_gate: dict[str, Any] | None = None
    repair_attempted: bool = False
    repair_succeeded: bool = False
    repair_reasons: list[str] | None = None
    repair_quality_gate: dict[str, Any] | None = None
    repair_delta: dict[str, Any] | None = None
    repair_prompt: str = ""
    repair_response_text: str = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt: str = ""
    post_brief_tokens: dict[str, int | None] | None = None
    source_evidence_pack: dict[str, Any] | None = None
    source_evidence_tokens: dict[str, int | None] | None = None
    source_evidence_error: str = ""
    author_take: dict[str, Any] | None = None
    author_take_tokens: dict[str, int | None] | None = None
    author_take_error: str = ""
    author_take_quality_issues: list[str] | None = None
    brief_alignment: dict[str, Any] | None = None
    post_mechanics: dict[str, Any] | None = None
    editorial_review: dict[str, Any] | None = None
    editorial_review_prompt: str = ""
    editorial_review_response_text: str = ""
    editorial_review_tokens: dict[str, int | None] | None = None
    editorial_review_error: str = ""
    editorial_review_used_for_repair: bool = False
    editorial_review_triggered_repair: bool = False
    editorial_repair_reasons: list[str] | None = None
    concrete_detail_diagnostics: dict[str, Any] | None = None
    banned_phrase_diagnostics: dict[str, Any] | None = None


def run_post_synthesis_pipeline(
    digest: Digest,
    articles: list[dict[str, Any]],
    profile: dict[str, Any],
    dependencies: PostSynthesisDependencies,
) -> PostSynthesisResult:
    """Run the current EvidencePack -> PostBrief -> FinalPost synthesis flow."""
    fallback_reason = ""
    provider = "openai"
    is_mock = False
    tokens: dict[str, int | None] | None = None
    estimated_cost: float | None = None
    quality_gate: dict[str, Any] | None = None
    repair_attempted = False
    repair_succeeded = False
    repair_reasons: list[str] = []
    repair_quality_gate: dict[str, Any] | None = None
    repair_delta: dict[str, Any] | None = None
    repair_prompt = ""
    repair_response_text = ""
    post_brief: dict[str, Any] | None = None
    post_brief_prompt = ""
    post_brief_tokens: dict[str, int | None] | None = None
    source_evidence_pack: dict[str, Any] | None = None
    source_evidence_tokens: dict[str, int | None] | None = None
    source_evidence_error = ""
    author_take: dict[str, Any] | None = None
    author_take_tokens: dict[str, int | None] | None = None
    author_take_error = ""
    author_take_quality_issues: list[str] = []
    brief_alignment: dict[str, Any] | None = None
    post_mechanics: dict[str, Any] | None = None
    editorial_review: dict[str, Any] | None = None
    editorial_review_prompt = ""
    editorial_review_response_text = ""
    editorial_review_tokens: dict[str, int | None] | None = None
    editorial_review_error = ""
    editorial_review_used_for_repair = False
    editorial_review_triggered_repair = False
    editorial_repair_reasons: list[str] = []
    concrete_detail_diagnostics: dict[str, Any] | None = None
    banned_phrase_diagnostics: dict[str, Any] | None = None
    prompt = dependencies.build_post_prompt(digest, articles, profile)
    response_text = ""

    try:
        if getattr(settings, "PACKAGING_SOURCE_EVIDENCE_ENABLED", True):
            try:
                (
                    source_evidence_pack,
                    _source_evidence_prompt,
                    _source_evidence_response,
                    source_evidence_tokens,
                ) = dependencies.generate_source_evidence_pack(digest, articles)
            except Exception as exc:  # noqa: BLE001 - evidence extraction is best-effort
                source_evidence_error = str(exc)
                source_evidence_pack = None

        if getattr(settings, "PACKAGING_AUTHOR_TAKE_ENABLED", True):
            try:
                author_take, _author_take_prompt, _author_take_response, author_take_tokens = (
                    dependencies.generate_author_take(
                        digest,
                        articles,
                        profile,
                        source_evidence_pack=source_evidence_pack,
                    )
                )
                author_take_quality_issues = dependencies.author_take_quality_issues(author_take)
                if dependencies.author_take_requires_rejection(author_take):
                    author_take_error = f"author take rejected: {', '.join(author_take_quality_issues)}"
                    author_take = None
            except Exception as exc:  # noqa: BLE001 - author take is best-effort
                author_take_error = str(exc)
                author_take = None

        try:
            post_brief, post_brief_prompt, _brief_response_text, post_brief_tokens = (
                dependencies.generate_post_brief(
                    digest,
                    articles,
                    profile,
                    source_evidence_pack=source_evidence_pack,
                    author_take=author_take,
                )
            )
        except Exception as exc:
            raise ContentPackageValidationError(
                f"LinkedIn post brief generation/validation failed: {exc}"
            ) from exc

        payload, prompt, response_text, tokens = dependencies.generate_payload(
            digest,
            articles,
            profile,
            post_brief=post_brief,
            source_evidence_pack=source_evidence_pack,
            author_take=author_take,
        )
        payload = dependencies.normalize_payload(payload)
        estimated_cost = estimate_cost_usd(
            tokens.get("prompt_tokens") if tokens else None,
            tokens.get("completion_tokens") if tokens else None,
        )
        repairable_payload_issues = dependencies.collect_repairable_payload_issues(payload)
        if not repairable_payload_issues:
            validate_content_package_payload(payload)
        quality_gate = dependencies.evaluate_quality(payload)
        brief_alignment = dependencies.evaluate_brief_alignment(payload, post_brief)
        post_mechanics = dependencies.evaluate_post_mechanics(payload, post_brief)
        concrete_detail_diagnostics = dependencies.build_concrete_detail_diagnostics(
            post_brief,
            payload,
            initial_alignment=brief_alignment,
        )
        repair_reasons = repairable_payload_issues + dependencies.combined_repair_reasons(
            quality_gate,
            brief_alignment,
            post_mechanics,
        )
        banned_phrase_diagnostics = dependencies.build_banned_phrase_diagnostics(
            repair_reasons,
            payload,
        )

        deterministic_repair_required = (
            repairable_payload_issues
            or dependencies.quality_gate_requires_repair(quality_gate)
            or dependencies.brief_alignment_requires_repair(brief_alignment)
            or dependencies.post_mechanics_requires_repair(post_mechanics)
        )
        if deterministic_repair_required:
            repair_attempted = True
            if getattr(settings, "PACKAGING_EDITORIAL_REVIEW_ENABLED", True):
                try:
                    (
                        editorial_review,
                        editorial_review_prompt,
                        editorial_review_response_text,
                        editorial_review_tokens,
                    ) = dependencies.generate_editorial_review(
                        digest,
                        payload,
                        profile,
                        post_brief,
                        quality_gate,
                        brief_alignment,
                        post_mechanics,
                        repair_delta=None,
                        repair_reasons=repair_reasons,
                    )
                    editorial_review_used_for_repair = True
                except Exception as exc:  # noqa: BLE001 - editorial guidance is diagnostic-only
                    editorial_review_error = str(exc)
            try:
                (
                    payload,
                    response_text,
                    repair_attempted,
                    repair_succeeded,
                    repair_prompt,
                    repair_response_text,
                    repair_quality_gate,
                    repair_delta,
                    brief_alignment,
                    post_mechanics,
                    concrete_detail_diagnostics,
                    banned_phrase_diagnostics,
                    editorial_review,
                    editorial_review_prompt,
                    editorial_review_response_text,
                    editorial_review_tokens,
                    editorial_review_error,
                    editorial_review_used_for_repair,
                ) = _run_repair_attempt(
                    digest=digest,
                    articles=articles,
                    profile=profile,
                    dependencies=dependencies,
                    payload=payload,
                    post_brief=post_brief,
                    quality_gate=quality_gate,
                    brief_alignment=brief_alignment,
                    post_mechanics=post_mechanics,
                    repair_reasons=repair_reasons,
                    editorial_review=editorial_review,
                    editorial_review_prompt=editorial_review_prompt,
                    editorial_review_response_text=editorial_review_response_text,
                    editorial_review_tokens=editorial_review_tokens,
                    editorial_review_error=editorial_review_error,
                    editorial_review_used_for_repair=editorial_review_used_for_repair,
                    failure_message="LinkedIn post quality repair did not resolve retry-trigger issues",
                )
            except _PostSynthesisRepairFailure as exc:
                repair_prompt = exc.repair_prompt
                repair_response_text = exc.repair_response_text
                repair_quality_gate = exc.repair_quality_gate
                repair_delta = exc.repair_delta
                concrete_detail_diagnostics = exc.concrete_detail_diagnostics
                banned_phrase_diagnostics = exc.banned_phrase_diagnostics
                raise
        if getattr(settings, "PACKAGING_EDITORIAL_REVIEW_ENABLED", True) and not repair_attempted:
            try:
                (
                    editorial_review,
                    editorial_review_prompt,
                    editorial_review_response_text,
                    editorial_review_tokens,
                ) = dependencies.generate_editorial_review(
                    digest,
                    payload,
                    profile,
                    post_brief,
                    quality_gate,
                    brief_alignment,
                    post_mechanics,
                    repair_delta=repair_delta,
                    repair_reasons=repair_reasons,
                )
            except Exception as exc:  # noqa: BLE001 - diagnostic-only review must not block saving
                editorial_review_error = str(exc)
            if dependencies.editorial_review_requires_repair(editorial_review):
                editorial_review_triggered_repair = True
                editorial_review_used_for_repair = True
                editorial_repair_reasons = dependencies.editorial_review_repair_reasons(editorial_review)
                repair_reasons = editorial_repair_reasons
                repair_attempted = True
                try:
                    (
                        payload,
                        response_text,
                        repair_attempted,
                        repair_succeeded,
                        repair_prompt,
                        repair_response_text,
                        repair_quality_gate,
                        repair_delta,
                        brief_alignment,
                        post_mechanics,
                        concrete_detail_diagnostics,
                        banned_phrase_diagnostics,
                        editorial_review,
                        editorial_review_prompt,
                        editorial_review_response_text,
                        editorial_review_tokens,
                        editorial_review_error,
                        editorial_review_used_for_repair,
                    ) = _run_repair_attempt(
                        digest=digest,
                        articles=articles,
                        profile=profile,
                        dependencies=dependencies,
                        payload=payload,
                        post_brief=post_brief,
                        quality_gate=quality_gate,
                        brief_alignment=brief_alignment,
                        post_mechanics=post_mechanics,
                        repair_reasons=repair_reasons,
                        editorial_review=editorial_review,
                        editorial_review_prompt=editorial_review_prompt,
                        editorial_review_response_text=editorial_review_response_text,
                        editorial_review_tokens=editorial_review_tokens,
                        editorial_review_error=editorial_review_error,
                        editorial_review_used_for_repair=editorial_review_used_for_repair,
                        failure_message="LinkedIn post editorial repair did not produce a valid package",
                    )
                except _PostSynthesisRepairFailure as exc:
                    repair_prompt = exc.repair_prompt
                    repair_response_text = exc.repair_response_text
                    repair_quality_gate = exc.repair_quality_gate
                    repair_delta = exc.repair_delta
                    concrete_detail_diagnostics = exc.concrete_detail_diagnostics
                    banned_phrase_diagnostics = exc.banned_phrase_diagnostics
                    raise
    except Exception as exc:  # noqa: BLE001 - explicit fallback for the MVP stage
        payload = dependencies.build_mock_payload(digest, articles)
        if post_brief is not None:
            prompt = dependencies.build_post_prompt(
                digest,
                articles,
                profile,
                post_brief=post_brief,
                source_evidence_pack=source_evidence_pack,
                author_take=author_take,
            )
        response_text = json.dumps(payload, ensure_ascii=False, indent=2)
        provider = "mock"
        is_mock = True
        fallback_reason = (
            "Fallback РЅР° mock РёР·-Р·Р° РѕС€РёР±РєРё СЂРµР°Р»СЊРЅРѕРіРѕ AI call РёР»Рё РЅРµРІР°Р»РёРґРЅРѕРіРѕ JSON: "
            f"{exc}. Raw response: <empty>"
        )

    return PostSynthesisResult(
        prompt=prompt,
        response_text=response_text,
        payload=payload,
        provider=provider,
        is_mock=is_mock,
        fallback_reason=fallback_reason,
        tokens=tokens,
        estimated_cost_usd=estimated_cost,
        quality_gate=quality_gate,
        repair_attempted=repair_attempted,
        repair_succeeded=repair_succeeded,
        repair_reasons=repair_reasons,
        repair_quality_gate=repair_quality_gate,
        repair_delta=repair_delta,
        repair_prompt=repair_prompt,
        repair_response_text=repair_response_text,
        post_brief=post_brief,
        post_brief_prompt=post_brief_prompt,
        post_brief_tokens=post_brief_tokens,
        source_evidence_pack=source_evidence_pack,
        source_evidence_tokens=source_evidence_tokens,
        source_evidence_error=source_evidence_error,
        author_take=author_take,
        author_take_tokens=author_take_tokens,
        author_take_error=author_take_error,
        author_take_quality_issues=author_take_quality_issues,
        brief_alignment=brief_alignment,
        post_mechanics=post_mechanics,
        editorial_review=editorial_review,
        editorial_review_prompt=editorial_review_prompt,
        editorial_review_response_text=editorial_review_response_text,
        editorial_review_tokens=editorial_review_tokens,
        editorial_review_error=editorial_review_error,
        editorial_review_used_for_repair=editorial_review_used_for_repair,
        editorial_review_triggered_repair=editorial_review_triggered_repair,
        editorial_repair_reasons=editorial_repair_reasons,
        concrete_detail_diagnostics=concrete_detail_diagnostics,
        banned_phrase_diagnostics=banned_phrase_diagnostics,
    )


def _run_repair_attempt(
    *,
    digest: Digest,
    articles: list[dict[str, Any]],
    profile: dict[str, Any],
    dependencies: PostSynthesisDependencies,
    payload: dict[str, Any],
    post_brief: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
    brief_alignment: dict[str, Any] | None,
    post_mechanics: dict[str, Any] | None,
    repair_reasons: list[str],
    editorial_review: dict[str, Any] | None,
    editorial_review_prompt: str,
    editorial_review_response_text: str,
    editorial_review_tokens: dict[str, int | None] | None,
    editorial_review_error: str,
    editorial_review_used_for_repair: bool,
    failure_message: str,
) -> tuple[
    dict[str, Any],
    str,
    bool,
    bool,
    str,
    str,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
    str,
    str,
    dict[str, int | None] | None,
    str,
    bool,
]:
    repair_attempted = True
    if (
        getattr(settings, "PACKAGING_EDITORIAL_REVIEW_ENABLED", True)
        and editorial_review is None
        and not editorial_review_error
    ):
        try:
            (
                editorial_review,
                editorial_review_prompt,
                editorial_review_response_text,
                editorial_review_tokens,
            ) = dependencies.generate_editorial_review(
                digest,
                payload,
                profile,
                post_brief,
                quality_gate,
                brief_alignment,
                post_mechanics,
                repair_delta=None,
                repair_reasons=repair_reasons,
            )
            editorial_review_used_for_repair = True
        except Exception as exc:  # noqa: BLE001 - editorial guidance is diagnostic-only
            editorial_review_error = str(exc)

    repair_report = {
        "status": "retry",
        "reasons": repair_reasons,
        "quality_gate": quality_gate,
        "brief_alignment": brief_alignment,
        "post_mechanics": post_mechanics,
    }
    if editorial_review is not None and any(str(reason).startswith("editorial_review:") for reason in repair_reasons):
        repair_report["editorial_review"] = editorial_review

    repaired_payload, repair_prompt, repair_response_text, _repair_tokens = dependencies.repair_payload(
        digest,
        articles,
        profile,
        weak_payload=payload,
        quality_report=repair_report,
        post_brief=post_brief,
        editorial_review=editorial_review,
        editorial_review_error=editorial_review_error,
    )
    repaired_payload = dependencies.normalize_payload(repaired_payload)
    repaired_payload["post_text"] = dependencies.split_long_post_paragraphs(repaired_payload["post_text"])
    validate_content_package_payload(repaired_payload)
    repair_quality_gate = dependencies.evaluate_quality(repaired_payload)
    repair_brief_alignment = dependencies.evaluate_brief_alignment(repaired_payload, post_brief)
    repair_post_mechanics = dependencies.evaluate_post_mechanics(repaired_payload, post_brief)
    repair_delta = dependencies.evaluate_repair_rewrite_delta(payload, repaired_payload, repair_reasons)
    concrete_detail_diagnostics = dependencies.build_concrete_detail_diagnostics(
        post_brief,
        payload,
        initial_alignment=brief_alignment,
        repaired_payload=repaired_payload,
        repair_alignment=repair_brief_alignment,
        repair_attempted=True,
    )
    banned_phrase_diagnostics = dependencies.build_banned_phrase_diagnostics(
        repair_reasons,
        payload,
        repaired_payload=repaired_payload,
        repair_attempted=True,
    )
    if (
        dependencies.quality_gate_requires_repair(repair_quality_gate)
        or dependencies.brief_alignment_requires_repair(repair_brief_alignment)
        or dependencies.post_mechanics_requires_repair(repair_post_mechanics)
        or dependencies.repair_delta_requires_failure(repair_delta)
    ):
        unresolved_reasons = dependencies.combined_repair_reasons(
            repair_quality_gate,
            repair_brief_alignment,
            repair_post_mechanics,
        )
        unresolved_reasons.extend(
            f"repair_delta:{issue}"
            for issue in repair_delta.get("issues", [])
        )
        raise _PostSynthesisRepairFailure(
            f"{failure_message}: {unresolved_reasons}",
            repair_prompt=repair_prompt,
            repair_response_text=repair_response_text,
            repair_quality_gate=repair_quality_gate,
            repair_delta=repair_delta,
            concrete_detail_diagnostics=concrete_detail_diagnostics,
            banned_phrase_diagnostics=banned_phrase_diagnostics,
        )

    return (
        repaired_payload,
        repair_response_text,
        repair_attempted,
        True,
        repair_prompt,
        repair_response_text,
        repair_quality_gate,
        repair_delta,
        repair_brief_alignment,
        repair_post_mechanics,
        concrete_detail_diagnostics,
        banned_phrase_diagnostics,
        editorial_review,
        editorial_review_prompt,
        editorial_review_response_text,
        editorial_review_tokens,
        editorial_review_error,
        editorial_review_used_for_repair,
    )
