"""Pipeline-first оркестрация одного DigestRun."""
from __future__ import annotations

import logging
from typing import Iterable

from django.db import transaction
from django.utils import timezone

from apps.digests.models import Digest, DigestRun
from apps.packaging.models import ContentPackage
from services.processing.cleaner import clean_source_items
from services.processing.deduper import dedupe_source_items


logger = logging.getLogger(__name__)


def run_digest_pipeline(run_id: int, raw_items: Iterable[dict]) -> DigestRun:
    """Выполнить MVP pipeline с детерминированной предобработкой.

    В MVP raw_items передаются вызывающим кодом. Позже сбор источников будет
    вынесен в search service, который отдаст сюда нормализованные словари.
    """
    run = DigestRun.objects.select_related("topic").get(pk=run_id)
    run.status = DigestRun.STATUS_PROCESSING
    run.started_at = run.started_at or timezone.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    try:
        raw_items_list = list(raw_items)
        cleaned_items = clean_source_items(raw_items_list)
        unique_items = dedupe_source_items(cleaned_items)

        with transaction.atomic():
            run.status = DigestRun.STATUS_GENERATING_DIGEST
            run.metrics = {
                "raw_items": len(raw_items_list),
                "cleaned_items": len(cleaned_items),
                "unique_items": len(unique_items),
            }
            run.save(update_fields=["status", "metrics", "updated_at"])

            digest = Digest.objects.create(
                run=run,
                title=f"Digest: {run.topic.name}",
                summary=_build_basic_summary(unique_items),
                key_points=[item["title"] for item in unique_items[:5]],
                sources=unique_items,
                quality_score=_score_digest(unique_items),
            )

            run.status = DigestRun.STATUS_PACKAGING
            run.save(update_fields=["status", "updated_at"])

            ContentPackage.objects.create(
                digest=digest,
                post_text=_build_placeholder_post(digest),
                hook_variants=[
                    f"What changed in {run.topic.name} this week?",
                    f"Three practical signals from {run.topic.name}.",
                    f"If you follow {run.topic.name}, watch this.",
                ],
                cta_variants=[
                    "What would you add to this list?",
                    "Which signal matters most for your team?",
                    "Follow for more practical digests.",
                ],
                hashtags=["#AI", "#Product", "#Strategy"],
                carousel_outline=[],
                validation_report={"status": "needs_ai_packaging"},
            )

            run.status = DigestRun.STATUS_COMPLETED
            run.finished_at = timezone.now()
            run.save(update_fields=["status", "finished_at", "updated_at"])

        logger.info("Digest pipeline completed", extra={"run_id": run.id})
        return run
    except Exception as exc:
        logger.exception("Digest pipeline failed", extra={"run_id": run.id})
        run.status = DigestRun.STATUS_FAILED
        run.error_message = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        raise


def _build_basic_summary(items: list[dict]) -> str:
    if not items:
        return "No high-quality source items were available for this run."
    titles = "; ".join(item["title"] for item in items[:5])
    return f"Processed {len(items)} unique source items. Main signals: {titles}."


def _score_digest(items: list[dict]) -> float:
    if not items:
        return 0.0
    return min(1.0, 0.45 + len(items) * 0.05)


def _build_placeholder_post(digest: Digest) -> str:
    return (
        f"{digest.title}\n\n"
        f"{digest.summary}\n\n"
        "This is an MVP package placeholder. Run AI packaging before publishing."
    )
