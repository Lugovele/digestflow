"""Pipeline-first оркестрация первого сквозного MVP pipeline."""
from __future__ import annotations

import logging
from typing import Iterable

from django.utils import timezone

from apps.digests.models import DigestRun
from services.digests import generate_digest_for_run
from services.packaging import generate_content_package_for_digest
from services.processing.deduper import dedupe_source_items
from services.processing.ranker import rank_source_items
from services.sources import save_articles_for_topic


logger = logging.getLogger(__name__)


def run_digest_pipeline(run_id: int, raw_items: Iterable[dict]) -> DigestRun:
    """Выполнить первый сквозной pipeline: Topic -> articles -> Digest -> ContentPackage."""
    run = DigestRun.objects.select_related("topic").get(pk=run_id)
    run.status = DigestRun.STATUS_COLLECTING
    run.started_at = run.started_at or timezone.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    _debug(run.id, "STEP", "pipeline started")
    _debug(run.id, "OK", f"topic loaded -> {run.topic.name}")

    try:
        raw_items_list = list(raw_items)
        _debug(run.id, "OK", f"demo articles loaded -> {len(raw_items_list)}")

        if not raw_items_list:
            raise ValueError("Source stage returned no articles.")

        deduped_items = dedupe_source_items(raw_items_list)
        duplicates_removed = len(raw_items_list) - len(deduped_items)
        _debug(run.id, "OK", f"articles deduplicated -> {len(deduped_items)}")
        _debug(run.id, "INFO", f"duplicates removed -> {duplicates_removed}")

        _debug(run.id, "STEP", "ranking")
        ranked_items, ranking_scores = rank_source_items(deduped_items)
        _debug(run.id, "INFO", f"ranked articles -> {len(deduped_items)}")
        _debug(run.id, "INFO", f"selected for prompt -> {len(ranked_items)}")

        saved_articles = save_articles_for_topic(run.topic, deduped_items)
        _debug(run.id, "OK", f"articles saved -> {len(saved_articles)}")

        run.metrics = {
            **run.metrics,
            "source_stage": {
                "status": "completed",
                "articles_count": len(raw_items_list),
                "articles_after_dedupe": len(deduped_items),
                "duplicates_removed": duplicates_removed,
                "saved_articles_count": len(saved_articles),
                "article_ids": [article.id for article in saved_articles],
            },
            "ranking_stage": {
                "status": "completed",
                "articles_after_dedupe": len(deduped_items),
                "ranked_articles_count": len(deduped_items),
                "selected_for_prompt": len(ranked_items),
                "ranking_scores": ranking_scores,
            },
        }
        run.save(update_fields=["metrics", "updated_at"])

        digest, digest_debug = generate_digest_for_run(run, ranked_items)

        run.status = DigestRun.STATUS_PACKAGING
        run.metrics = {
            **run.metrics,
            "packaging_stage": {"status": "started"},
        }
        run.save(update_fields=["status", "metrics", "updated_at"])
        _debug(run.id, "STEP", "package generating")

        try:
            content_package, packaging_debug = generate_content_package_for_digest(digest)
        except Exception as exc:
            logger.exception("Packaging stage failed", extra={"run_id": run.id})
            run.status = DigestRun.STATUS_PARTIAL_FAILED
            run.error_message = f"Packaging stage failed: {exc}"
            run.finished_at = timezone.now()
            run.metrics = {
                **run.metrics,
                "packaging_stage": {
                    "status": "failed",
                    "error": str(exc),
                    "digest_id": digest.id,
                },
            }
            run.save(
                update_fields=["status", "error_message", "finished_at", "metrics", "updated_at"]
            )
            _debug(run.id, "FAIL", "package generating")
            _debug(run.id, "INFO", f"error -> {run.error_message}")
            _debug(run.id, "INFO", f"digest preserved -> {digest.id}")
            _debug(run.id, "DONE", "run partial_failed")
            return run

        run.status = DigestRun.STATUS_COMPLETED
        run.finished_at = timezone.now()
        run.metrics = {
            **run.metrics,
            "packaging_stage": {
                "status": "completed",
                "content_package_id": content_package.id,
                "provider": packaging_debug["provider"],
                "is_mock": packaging_debug["is_mock"],
                "tokens": {
                    "prompt": packaging_debug["tokens"]["prompt_tokens"]
                    if packaging_debug.get("tokens")
                    else None,
                    "completion": packaging_debug["tokens"]["completion_tokens"]
                    if packaging_debug.get("tokens")
                    else None,
                    "total": packaging_debug["tokens"]["total_tokens"]
                    if packaging_debug.get("tokens")
                    else None,
                },
                "estimated_cost_usd": packaging_debug.get("estimated_cost_usd"),
            },
        }
        run.save(update_fields=["status", "finished_at", "metrics", "updated_at"])
        _debug(run.id, "DONE", "run completed")

        logger.info("Digest pipeline completed", extra={"run_id": run.id})
        return run
    except Exception as exc:
        logger.exception("Digest pipeline failed", extra={"run_id": run.id})
        run.status = DigestRun.STATUS_FAILED
        run.error_message = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        _debug(run.id, "FAIL", "run failed")
        _debug(run.id, "INFO", f"error -> {run.error_message}")
        return run


def _debug(run_id: int, level: str, message: str) -> None:
    print(f"[DigestRun {run_id}] {level}: {message}")
