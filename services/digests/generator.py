"""First working AI digest stage for the MVP."""
from __future__ import annotations

import logging
from typing import Any

from django.db import transaction
from django.utils import timezone

from apps.digests.models import Digest, DigestRun
from services.ai import generate_digest_payload

logger = logging.getLogger(__name__)


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
        # Deprecated storage fields stay only at the persistence boundary until the DB schema is redesigned.
        digest = Digest.objects.create(
            run=run,
            title=payload["title"],
            payload=payload,
            quality_score=_score_digest_payload(payload),
        )

        run.metrics = {
            **run.metrics,
            "digest_stage": {
                "status": "completed",
                "provider": generation.provider,
                "is_mock": generation.is_mock,
                "articles_in_prompt": len(articles),
                "articles_count": len(payload["articles"]),
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
        "articles": generation.articles,
        "tokens": generation.tokens,
        "estimated_cost_usd": generation.estimated_cost_usd,
    }
    return digest, debug_info


def _score_digest_payload(payload: dict[str, Any]) -> float:
    articles_count = len(payload.get("articles", []))
    strong_articles = sum(
        1
        for article in payload.get("articles", [])
        if float(article.get("confidence", 0.0) or 0.0) >= 0.6
    )
    score = 0.4 + min(articles_count, 5) * 0.08 + min(strong_articles, 5) * 0.06
    return min(1.0, round(score, 2))


def _debug(run_id: int, level: str, message: str) -> None:
    logger.info("[DigestRun %s] %s: %s", run_id, level, message)
