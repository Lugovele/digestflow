"""Сохранение source items в Article для локального MVP."""
from __future__ import annotations

from typing import Any

from apps.sources.models import Article
from apps.topics.models import Topic


def save_articles_for_topic(topic: Topic, raw_items: list[dict[str, Any]]) -> list[Article]:
    """Сохранить source items как Article, не дублируя записи по URL внутри Topic."""
    saved_articles: list[Article] = []
    seen_article_ids: set[int] = set()

    for item in raw_items:
        article, _created = Article.objects.update_or_create(
            topic=topic,
            url=item["url"],
            defaults={
                "title": item["title"],
                "source_name": item["source"],
                "snippet": item["snippet"],
                "published_at": item.get("published_at"),
                "raw_payload": item,
            },
        )
        if article.id not in seen_article_ids:
            saved_articles.append(article)
            seen_article_ids.add(article.id)

    return saved_articles
