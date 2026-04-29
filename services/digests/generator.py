"""First working AI digest stage for the MVP."""
from __future__ import annotations

from typing import Any

from django.db import transaction
from django.utils import timezone

from apps.digests.models import Digest, DigestRun
from services.ai import generate_digest_payload


def generate_digest_for_run(run: DigestRun, articles: list[dict[str, Any]]) -> tuple[Digest, dict[str, Any]]:
    """Generate and save a Digest for the selected DigestRun."""
    run.status = DigestRun.STATUS_GENERATING_DIGEST
    run.started_at = run.started_at or timezone.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    _debug(run.id, "STEP", "digest generating")
    _debug(run.id, "INFO", f"topic -> {run.topic.name}")
    _debug(run.id, "INFO", f"articles received -> {len(articles)}")

    generation = generate_digest_payload(run.topic.name, articles)

    _debug(run.id, "INFO", f"provider -> {generation.provider}")
    _debug(run.id, "INFO", f"is_mock -> {generation.is_mock}")
    if generation.fallback_reason:
        _debug(run.id, "INFO", f"fallback_reason -> {generation.fallback_reason}")
    if generation.tokens and generation.tokens.get("total_tokens") is not None:
        _debug(run.id, "INFO", f"tokens -> total: {generation.tokens['total_tokens']}")
    if generation.estimated_cost_usd is not None:
        _debug(run.id, "INFO", f"estimated cost -> ${generation.estimated_cost_usd:.6f}")

    payload = generation.payload

    with transaction.atomic():
        Digest.objects.filter(run=run).delete()
        digest = Digest.objects.create(
            run=run,
            title=payload["title"],
            summary=payload["summary"],
            key_points=payload["key_points"],
            sources=payload["sources"],
            quality_score=_score_digest_payload(payload),
        )

        run.metrics = {
            **run.metrics,
            "digest_stage": {
                "provider": generation.provider,
                "is_mock": generation.is_mock,
                "articles_in_prompt": len(articles),
                "key_points_count": len(payload["key_points"]),
                "sources_count": len(payload["sources"]),
                "tokens": {
                    "prompt": generation.tokens["prompt_tokens"] if generation.tokens else None,
                    "completion": generation.tokens["completion_tokens"] if generation.tokens else None,
                    "total": generation.tokens["total_tokens"] if generation.tokens else None,
                },
                "estimated_cost_usd": generation.estimated_cost_usd,
            },
        }
        run.save(update_fields=["metrics", "updated_at"])

    _debug(run.id, "OK", f"digest saved -> {digest.id}")

    debug_info = {
        "prompt": generation.prompt,
        "response_text": generation.response_text,
        "provider": generation.provider,
        "is_mock": generation.is_mock,
        "fallback_reason": generation.fallback_reason,
        "tokens": generation.tokens,
        "estimated_cost_usd": generation.estimated_cost_usd,
    }
    return digest, debug_info


def _score_digest_payload(payload: dict[str, Any]) -> float:
    key_points_count = len(payload.get("key_points", []))
    sources_count = len(payload.get("sources", []))
    score = 0.45 + min(key_points_count, 5) * 0.08 + min(sources_count, 5) * 0.04
    return min(1.0, round(score, 2))


def _debug(run_id: int, level: str, message: str) -> None:
    print(f"[DigestRun {run_id}] {level}: {message}")
