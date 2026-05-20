from __future__ import annotations

from typing import Any

from apps.digests.models import DigestRun, UsedArticle
from django.utils import timezone
from services.sources.detector import classify_source_url


def get_used_article_urls_for_topic(topic) -> set[str]:
    return {
        str(url or "").strip()
        for url in UsedArticle.objects.filter(topic=topic).values_list("normalized_url", flat=True)
        if str(url or "").strip()
    }


def build_used_article_filter_diagnostics(
    ranking_scores: list[dict[str, Any]],
    *,
    used_article_count_for_topic: int,
) -> dict[str, Any]:
    excluded_used_articles = [
        {
            "title": str(score.get("title") or "").strip(),
            "url": str(score.get("url") or "").strip(),
        }
        for score in ranking_scores
        if score.get("excluded_as_used")
    ]
    remaining_after_used_filter = sum(
        1
        for score in ranking_scores
        if float(score.get("quality_score", 0.0) or 0.0) >= float(score.get("quality_threshold_used", 0.0) or 0.0)
        and not score.get("excluded_as_used")
    )
    return {
        "used_article_filter_enabled": True,
        "used_article_count_for_topic": used_article_count_for_topic,
        "articles_excluded_as_used": len(excluded_used_articles),
        "articles_remaining_after_used_filter": remaining_after_used_filter,
        "excluded_used_articles": excluded_used_articles,
    }


def record_used_articles_for_run(run: DigestRun, selected_articles: list[dict[str, Any]]) -> list[UsedArticle]:
    """Persist only the articles actually used in the completed digest run."""
    if not selected_articles:
        return []

    used_articles: list[UsedArticle] = []
    seen_normalized_urls: set[str] = set()

    for article in selected_articles:
        article_url = str(article.get("url") or "").strip()
        if not article_url:
            continue

        normalized_url = str(article.get("normalized_url") or "").strip()
        if not normalized_url:
            normalized_url = classify_source_url(article_url).normalized_url
        if not normalized_url or normalized_url in seen_normalized_urls:
            continue
        seen_normalized_urls.add(normalized_url)

        source_url = str(article.get("source_url") or "").strip() or str(article.get("source_api_url") or "").strip()
        now = timezone.now()
        used_article, created = UsedArticle.objects.get_or_create(
            topic=run.topic,
            normalized_url=normalized_url,
            defaults={
                "user": run.topic.user,
                "digest_run": run,
                "first_used_in_run": run,
                "last_used_in_run": run,
                "article_url": article_url,
                "title": str(article.get("title") or "").strip(),
                "source_url": source_url,
                "use_count": 1,
                "first_used_at": now,
                "last_used_at": now,
            },
        )
        if not created:
            used_article.user = run.topic.user
            used_article.digest_run = run
            used_article.last_used_in_run = run
            used_article.article_url = article_url
            used_article.title = str(article.get("title") or "").strip()
            used_article.source_url = source_url
            used_article.use_count += 1
            used_article.last_used_at = now
            if used_article.first_used_in_run_id is None:
                used_article.first_used_in_run = used_article.digest_run
            if used_article.first_used_at is None:
                used_article.first_used_at = now
            used_article.save(
                update_fields=[
                    "user",
                    "digest_run",
                    "last_used_in_run",
                    "article_url",
                    "title",
                    "source_url",
                    "use_count",
                    "first_used_in_run",
                    "first_used_at",
                    "last_used_at",
                    "used_at",
                ],
            )
        used_articles.append(used_article)

    return used_articles
