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
from services.processing.ranker import DEFAULT_MIN_QUALITY_SCORE, rank_source_items
from services.sources import get_demo_articles_for_topic, save_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

DEFAULT_RSS_FEED = "https://techcrunch.com/feed/"
MIN_ARTICLES_FOR_DIGEST = 2
INSUFFICIENT_QUALITY_MESSAGE = (
    "Недостаточно качественных статей для полноценного дайджеста. "
    "Источник обработан, но найденные материалы слишком слабые или разрозненные."
)

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

        source_metrics = _build_source_input_metrics(raw_items_list)
        cleaned_items = clean_source_items(raw_items_list)
        removed_during_cleaning = len(raw_items_list) - len(cleaned_items)
        _debug(run.id, "OK", f"articles cleaned -> {len(cleaned_items)}")
        _debug(run.id, "INFO", f"removed during cleaning -> {removed_during_cleaning}")

        run.metrics = make_json_safe({
            **run.metrics,
            "source_stage": {
                "status": "completed" if cleaned_items else "failed_cleaning",
                "raw_items_count": source_metrics["raw_items_count"],
                "article_links_extracted": source_metrics["article_links_extracted"],
                "article_contents_fetched": source_metrics["article_contents_fetched"],
                "content_unavailable_count": source_metrics["content_unavailable_count"],
                "normalized_source_type": source_metrics["normalized_source_type"],
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
        quality_threshold = (
            topic_settings.min_source_quality_score
            if topic_settings
            else DEFAULT_MIN_QUALITY_SCORE
        )

        _debug(run.id, "STEP", "ranking")
        ranked_items, ranking_scores = rank_source_items(
            deduped_items,
            keywords=run.topic.keywords,
            excluded_keywords=run.topic.excluded_keywords,
            top_n=top_n,
            min_quality_score=quality_threshold,
        )
        qualified_scores = [
            score_entry
            for score_entry in ranking_scores
            if float(score_entry.get("quality_score", 0.0) or 0.0) >= quality_threshold
        ]
        rejected_scores = [
            score_entry
            for score_entry in ranking_scores
            if float(score_entry.get("quality_score", 0.0) or 0.0) < quality_threshold
        ]
        quality_values = [float(score_entry.get("quality_score", 0.0) or 0.0) for score_entry in ranking_scores]
        max_quality_score = max(quality_values) if quality_values else None
        min_actual_quality_score = min(quality_values) if quality_values else None
        average_quality_score = (
            round(sum(quality_values) / len(quality_values), 2)
            if quality_values
            else None
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
                "articles_above_quality_threshold": len(qualified_scores),
                "selected_for_prompt": len(ranked_items),
                "quality_threshold": quality_threshold,
                "max_quality_score": max_quality_score,
                "min_actual_quality_score": min_actual_quality_score,
                "average_quality_score": average_quality_score,
                "rejected_low_quality_count": len(rejected_scores),
                "top_n": top_n,
                "ranking_scores": ranking_scores,
                "top_rejected_articles": rejected_scores[:5],
            },
        })
        run.save(update_fields=["metrics", "updated_at"])

        if len(ranked_items) < MIN_ARTICLES_FOR_DIGEST:
            _debug(run.id, "INFO", "insufficient article quality for digest generation")
            run.status = DigestRun.STATUS_INSUFFICIENT_QUALITY
            run.error_message = INSUFFICIENT_QUALITY_MESSAGE
            run.finished_at = timezone.now()
            run.metrics = make_json_safe({
                **run.metrics,
                "ranking_stage": {
                    **run.metrics.get("ranking_stage", {}),
                    "status": "insufficient_quality",
                    "insufficient_quality": True,
                    "insufficient_quality_message": INSUFFICIENT_QUALITY_MESSAGE,
                    "minimum_articles_required": MIN_ARTICLES_FOR_DIGEST,
                },
                "digest_stage": {
                    "status": "skipped",
                    "reason": "insufficient_quality",
                },
                "packaging_stage": {
                    "status": "skipped",
                    "reason": "insufficient_quality",
                },
            })
            run.save(
                update_fields=["status", "error_message", "finished_at", "metrics", "updated_at"]
            )
            _debug(run.id, "DONE", "run insufficient_quality")
            return run

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


def _build_source_input_metrics(raw_items: list[dict]) -> dict[str, int | str | None]:
    raw_items_count = len(raw_items)
    article_links_extracted = 0
    article_contents_fetched = 0
    content_unavailable_count = 0
    normalized_source_type = None

    for item in raw_items:
        if not isinstance(item, dict):
            continue
        if item.get("url"):
            article_links_extracted += 1
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        if normalized_source_type is None:
            normalized_source_type = (
                item.get("source_type")
                or metadata.get("source_type")
                or "raw_items"
            )
        if metadata.get("content_unavailable"):
            content_unavailable_count += 1
        elif str(item.get("content") or item.get("snippet") or "").strip():
            article_contents_fetched += 1

    return {
        "raw_items_count": raw_items_count,
        "article_links_extracted": article_links_extracted,
        "article_contents_fetched": article_contents_fetched,
        "content_unavailable_count": content_unavailable_count,
        "normalized_source_type": normalized_source_type or "raw_items",
    }


def _debug(run_id: int, level: str, message: str) -> None:
    logger.info("[DigestRun %s] %s: %s", run_id, level, message)
