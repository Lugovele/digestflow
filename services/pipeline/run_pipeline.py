"""Pipeline-first orchestration for the first end-to-end MVP pipeline."""
from __future__ import annotations

import logging
from typing import Iterable

from django.utils import timezone

from apps.digests.models import DigestRun
from apps.topics.models import DigestSettings
from services.digests import generate_digest_for_run
from services.config.author_profile import load_author_profile
from services.json_utils import make_json_safe
from services.packaging import generate_content_package_for_digest
from services.processing.cleaner import clean_source_items
from services.processing.deduper import dedupe_source_items_with_metrics
from services.processing.ranker import rank_source_items
from services.sources import get_demo_articles_for_topic, save_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

DEFAULT_RSS_FEED = "https://techcrunch.com/feed/"

logger = logging.getLogger(__name__)


def run_digest_pipeline(run_id: int, raw_items: Iterable[dict] | None = None) -> DigestRun:
    """Run the first end-to-end pipeline: Topic -> articles -> Digest -> ContentPackage."""
    run = DigestRun.objects.select_related("topic").get(pk=run_id)
    run.status = DigestRun.STATUS_COLLECTING
    run.started_at = run.started_at or timezone.now()
    run.save(update_fields=["status", "started_at", "updated_at"])

    _debug(run.id, "STEP", "pipeline started")
    _debug(run.id, "OK", f"topic loaded -> {run.topic.name}")

    try:
        if raw_items is None:
            try:
                rss_items = fetch_rss_articles(DEFAULT_RSS_FEED)
            except Exception:
                rss_items = []

            if rss_items:
                raw_items_list = list(rss_items)
                _debug(run.id, "OK", f"rss articles loaded -> {len(raw_items_list)}")
            else:
                _debug(run.id, "INFO", "fallback to demo source")
                raw_items_list = list(get_demo_articles_for_topic(run.topic.name))
                _debug(run.id, "OK", f"demo articles loaded -> {len(raw_items_list)}")
        else:
            raw_items_list = list(raw_items)
            _debug(run.id, "OK", f"source items loaded -> {len(raw_items_list)}")

        if not raw_items_list:
            raise ValueError("Source stage returned no articles.")

        cleaned_items = clean_source_items(raw_items_list)
        removed_during_cleaning = len(raw_items_list) - len(cleaned_items)
        _debug(run.id, "OK", f"articles cleaned -> {len(cleaned_items)}")
        _debug(run.id, "INFO", f"removed during cleaning -> {removed_during_cleaning}")

        run.metrics = make_json_safe({
            **run.metrics,
            "source_stage": {
                "status": "completed" if cleaned_items else "failed_cleaning",
                "raw_items_count": len(raw_items_list),
                "articles_count": len(raw_items_list),
                "articles_after_cleaning": len(cleaned_items),
                "removed_during_cleaning": removed_during_cleaning,
            },
        })
        run.save(update_fields=["metrics", "updated_at"])

        if not cleaned_items:
            raise ValueError("Source items were loaded but none passed cleaning.")

        deduped_items, dedupe_metrics = dedupe_source_items_with_metrics(cleaned_items)
        duplicates_removed = dedupe_metrics["duplicates_removed"]
        _debug(run.id, "OK", f"articles deduplicated -> {len(deduped_items)}")
        _debug(run.id, "INFO", f"duplicates removed -> {duplicates_removed}")

        topic_settings = _get_digest_settings(run)
        top_n = min(3, topic_settings.max_sources) if topic_settings else 3
        min_quality_score = topic_settings.min_source_quality_score if topic_settings else 0.0

        _debug(run.id, "STEP", "ranking")
        ranked_items, ranking_scores = rank_source_items(
            deduped_items,
            keywords=run.topic.keywords,
            excluded_keywords=run.topic.excluded_keywords,
            top_n=top_n,
            min_quality_score=min_quality_score,
        )
        _debug(run.id, "INFO", f"ranked articles -> {len(deduped_items)}")
        _debug(run.id, "INFO", f"selected for prompt -> {len(ranked_items)}")

        saved_articles = save_articles_for_topic(run.topic, deduped_items)
        _debug(run.id, "OK", f"articles saved -> {len(saved_articles)}")

        run.metrics = make_json_safe({
            **run.metrics,
            "source_stage": {
                **run.metrics.get("source_stage", {}),
                "status": "completed",
                "articles_after_dedupe": len(deduped_items),
                "duplicates_removed": duplicates_removed,
                "duplicate_urls_removed": dedupe_metrics["duplicate_urls_removed"],
                "duplicate_titles_removed": dedupe_metrics["duplicate_titles_removed"],
                "saved_articles_count": len(saved_articles),
                "article_ids": [article.id for article in saved_articles],
            },
            "ranking_stage": {
                "status": "completed",
                "articles_after_dedupe": len(deduped_items),
                "ranked_articles_count": len(deduped_items),
                "articles_after_rank": len(ranked_items),
                "selected_for_prompt": len(ranked_items),
                "min_quality_score": min_quality_score,
                "top_n": top_n,
                "ranking_scores": ranking_scores,
            },
        })
        run.save(update_fields=["metrics", "updated_at"])

        digest, digest_debug = generate_digest_for_run(run, ranked_items)

        run.status = DigestRun.STATUS_PACKAGING
        run.metrics = make_json_safe({
            **run.metrics,
            "packaging_stage": {"status": "started"},
        })
        run.save(update_fields=["status", "metrics", "updated_at"])
        _debug(run.id, "STEP", "package generating")
        author_profile = load_author_profile()

        try:
            content_package, packaging_debug = generate_content_package_for_digest(
                digest,
                author_profile=author_profile,
            )
        except Exception as exc:
            logger.exception("[DigestRun %s] Packaging stage failed", run.id)
            run.status = DigestRun.STATUS_PARTIAL_FAILED
            run.error_message = f"Packaging stage failed: {exc}"
            run.finished_at = timezone.now()
            run.metrics = make_json_safe({
                **run.metrics,
                "packaging_stage": {
                    "status": "failed",
                    "error": str(exc),
                    "digest_id": digest.id,
                },
            })
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
        run.error_message = ""
        run.metrics = make_json_safe({
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
        })
        run.save(update_fields=["status", "finished_at", "error_message", "metrics", "updated_at"])
        _debug(run.id, "DONE", "run completed")

        logger.info("[DigestRun %s] Digest pipeline completed", run.id)
        return run
    except Exception as exc:
        logger.exception("[DigestRun %s] Digest pipeline failed", run.id)
        run.status = DigestRun.STATUS_FAILED
        run.error_message = str(exc)
        run.finished_at = timezone.now()
        run.save(update_fields=["status", "error_message", "finished_at", "updated_at"])
        _debug(run.id, "FAIL", "run failed")
        _debug(run.id, "INFO", f"error -> {run.error_message}")
        return run


def _get_digest_settings(run: DigestRun) -> DigestSettings | None:
    try:
        return run.topic.digest_settings
    except DigestSettings.DoesNotExist:
        return None


def _debug(run_id: int, level: str, message: str) -> None:
    logger.info("[DigestRun %s] %s: %s", run_id, level, message)
