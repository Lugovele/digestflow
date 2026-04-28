"""Первый рабочий AI digest stage для MVP."""
from __future__ import annotations

from typing import Any

from django.db import transaction
from django.utils import timezone

from apps.digests.models import Digest, DigestRun
from services.ai import generate_digest_payload


def generate_digest_for_run(run: DigestRun, articles: list[dict[str, Any]]) -> tuple[Digest, dict[str, Any]]:
    """Сгенерировать и сохранить Digest для выбранного DigestRun."""
    run.status = DigestRun.STATUS_GENERATING_DIGEST
    run.started_at = run.started_at or timezone.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    print(f"[DigestRun {run.id}] Stage started: generating_digest")
    print(f"[DigestRun {run.id}] Topic: {run.topic.name}")
    print(f"[DigestRun {run.id}] Articles received: {len(articles)}")

    generation = generate_digest_payload(run.topic.name, articles)

    print(f"[DigestRun {run.id}] Provider: {generation.provider}")
    print(f"[DigestRun {run.id}] Is mock: {generation.is_mock}")
    if generation.fallback_reason:
        print(f"[DigestRun {run.id}] Fallback reason: {generation.fallback_reason}")

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
            },
        }
        run.save(update_fields=["metrics", "updated_at"])

    print(f"[DigestRun {run.id}] Digest saved: {digest.id}")
    print(f"[DigestRun {run.id}] Stage completed: generating_digest")

    debug_info = {
        "prompt": generation.prompt,
        "response_text": generation.response_text,
        "provider": generation.provider,
        "is_mock": generation.is_mock,
        "fallback_reason": generation.fallback_reason,
    }
    return digest, debug_info


def _score_digest_payload(payload: dict[str, Any]) -> float:
    key_points_count = len(payload.get("key_points", []))
    sources_count = len(payload.get("sources", []))
    score = 0.45 + min(key_points_count, 5) * 0.08 + min(sources_count, 5) * 0.04
    return min(1.0, round(score, 2))
