"""РЎРѕС…СЂР°РЅРµРЅРёРµ source items РІ Article РґР»СЏ Р»РѕРєР°Р»СЊРЅРѕРіРѕ MVP."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from django.utils.dateparse import parse_datetime

from apps.sources.models import Article
from apps.topics.models import Topic
from services.json_utils import make_json_safe


def save_articles_for_topic(topic: Topic, raw_items: list[dict[str, Any]]) -> list[Article]:
    """РЎРѕС…СЂР°РЅРёС‚СЊ source items РєР°Рє Article, РЅРµ РґСѓР±Р»РёСЂСѓСЏ Р·Р°РїРёСЃРё РїРѕ URL РІРЅСѓС‚СЂРё Topic."""
    saved_articles: list[Article] = []
    seen_article_ids: set[int] = set()

    for item in raw_items:
        article, _created = Article.objects.update_or_create(
            topic=topic,
            url=item["url"],
            defaults={
                "title": item["title"],
                "source_name": item.get("source_name") or item["source"],
                "snippet": item["snippet"],
                "published_at": _coerce_published_at(item.get("published_at")),
                "raw_payload": make_json_safe(item),
            },
        )
        if article.id not in seen_article_ids:
            saved_articles.append(article)
            seen_article_ids.add(article.id)

    return saved_articles


def _coerce_published_at(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return parse_datetime(value)
    return None
