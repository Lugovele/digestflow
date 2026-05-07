import logging
from urllib.parse import urlparse

from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

from .forms import TopicInputForm

logger = logging.getLogger(__name__)


@require_GET
def topic_list_view(request: HttpRequest) -> HttpResponse:
    return render(request, "digestflow/topic_list.html", _build_topic_list_context())


@require_POST
def create_topic_and_run_view(request: HttpRequest) -> HttpResponse:
    form = TopicInputForm(request.POST)
    if not form.is_valid():
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = "Нужно указать тему перед запуском pipeline."
        return render(request, "digestflow/topic_list.html", context, status=400)

    topic_name = form.cleaned_data["topic_name"]
    source_url = form.cleaned_data.get("source_url") or ""
    topic = _get_or_create_ui_topic(topic_name, source_url=source_url)
    run = _create_ui_digest_run(topic, source="web_ui_form")

    _start_topic_run(run, topic, default_source="web_ui_form")
    return redirect("run-detail", run_id=run.id)


@require_POST
def run_pipeline_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    run = _create_ui_digest_run(topic, source="web_ui")

    _start_topic_run(run, topic, default_source="web_ui")
    return redirect("run-detail", run_id=run.id)


@require_GET
def run_detail_view(request: HttpRequest, run_id: int) -> HttpResponse:
    run = get_object_or_404(
        DigestRun.objects.select_related("topic").prefetch_related(),
        pk=run_id,
    )
    digest = getattr(run, "digest", None)
    content_package = getattr(digest, "content_package", None) if digest else None

    metrics = run.metrics if isinstance(run.metrics, dict) else {}
    source_stage = metrics.get("source_stage", {})
    ranking_stage = metrics.get("ranking_stage", {})
    digest_stage = metrics.get("digest_stage", {})
    packaging_stage = metrics.get("packaging_stage", {})
    digest_articles = _decorate_article_links(digest.get_articles() if digest else [])
    ranked_articles = _decorate_article_links(ranking_stage.get("ranking_scores", []))
    is_insufficient_quality = run.status == DigestRun.STATUS_INSUFFICIENT_QUALITY

    logger.info("[DigestRun %s] digest payload articles count -> %s", run.id, len(digest_articles))
    if not digest_articles:
        logger.warning("[DigestRun %s] digest payload articles are empty", run.id)

    digest_total_tokens = (digest_stage.get("tokens") or {}).get("total")
    packaging_total_tokens = (packaging_stage.get("tokens") or {}).get("total")
    total_tokens = _sum_metric_values(digest_total_tokens, packaging_total_tokens)
    total_estimated_cost = _sum_metric_values(
        digest_stage.get("estimated_cost_usd"),
        packaging_stage.get("estimated_cost_usd"),
    )
    validation_report = (
        content_package.validation_report
        if content_package and isinstance(content_package.validation_report, dict)
        else {}
    )

    digest_payload = {
        "title": digest.get_payload_title() if digest else "",
        "articles": digest_articles,
    }
    selected_ranked_articles = _select_ranked_articles_for_prompt(
        ranked_articles,
        ranking_stage.get("quality_threshold"),
        ranking_stage.get("selected_for_prompt"),
    )

    context = {
        "run": run,
        "digest_payload": digest_payload,
        "has_digest": digest is not None,
        "content_package": content_package,
        "is_insufficient_quality": is_insufficient_quality,
        "insufficient_quality_message": (
            ranking_stage.get("insufficient_quality_message") or run.error_message
        ),
        "metrics": metrics,
        "article_ids": source_stage.get("article_ids", []),
        "articles_after_dedupe": source_stage.get("articles_after_dedupe"),
        "selected_for_prompt": ranking_stage.get("selected_for_prompt"),
        "selected_for_prompt_display": _display_metric_value(ranking_stage.get("selected_for_prompt")),
        "quality_threshold": ranking_stage.get("quality_threshold"),
        "quality_threshold_display": _display_metric_value(ranking_stage.get("quality_threshold")),
        "max_quality_score": ranking_stage.get("max_quality_score"),
        "max_quality_score_display": _display_metric_value(ranking_stage.get("max_quality_score")),
        "min_actual_quality_score": ranking_stage.get("min_actual_quality_score"),
        "min_actual_quality_score_display": _display_metric_value(ranking_stage.get("min_actual_quality_score")),
        "average_quality_score": ranking_stage.get("average_quality_score"),
        "average_quality_score_display": _display_metric_value(ranking_stage.get("average_quality_score")),
        "articles_above_quality_threshold": ranking_stage.get("articles_above_quality_threshold"),
        "articles_above_quality_threshold_display": _display_metric_value(
            ranking_stage.get("articles_above_quality_threshold")
        ),
        "rejected_low_quality_count": ranking_stage.get("rejected_low_quality_count"),
        "rejected_low_quality_count_display": _display_metric_value(
            ranking_stage.get("rejected_low_quality_count")
        ),
        "top_rejected_articles": _decorate_article_links(ranking_stage.get("top_rejected_articles", [])),
        "ranked_articles": ranked_articles,
        "selected_ranked_articles": selected_ranked_articles,
        "total_tokens": total_tokens,
        "total_tokens_display": _display_metric_value(total_tokens),
        "total_estimated_cost": total_estimated_cost,
        "total_estimated_cost_display": _display_metric_value(total_estimated_cost),
        "digest_provider": digest_stage.get("provider"),
        "packaging_provider": packaging_stage.get("provider"),
        "has_digest_articles": bool(digest_articles),
        "has_ranked_articles": bool(ranked_articles),
        "validation_report": validation_report,
        "quality_checks": validation_report.get("quality_checks", {}),
        "hook_variants": content_package.hook_variants if content_package else [],
        "cta_variants": content_package.cta_variants if content_package else [],
        "primary_hook": content_package.primary_hook() if content_package else "",
        "hashtags_text": content_package.hashtags_text() if content_package else "",
        "validation_status": (
            content_package.validation_report.get("status", "unknown")
            if content_package
            else "not_available"
        ),
    }
    return render(request, "digestflow/run_detail.html", context)


def _sum_metric_values(*values):
    present_values = [value for value in values if value is not None]
    if not present_values:
        return None
    return round(sum(present_values), 6)


def _display_metric_value(value):
    if value is None:
        return "-"
    return value


def _select_ranked_articles_for_prompt(
    ranked_articles: list[dict],
    quality_threshold,
    selected_count,
) -> list[dict]:
    try:
        threshold_value = float(quality_threshold) if quality_threshold is not None else None
    except (TypeError, ValueError):
        threshold_value = None

    try:
        selected_limit = int(selected_count or 0)
    except (TypeError, ValueError):
        selected_limit = 0

    if selected_limit <= 0:
        return []

    eligible_articles = []
    for article in ranked_articles:
        quality_score = article.get("quality_score")
        try:
            quality_value = float(quality_score) if quality_score is not None else None
        except (TypeError, ValueError):
            quality_value = None

        if threshold_value is not None and quality_value is not None and quality_value >= threshold_value:
            eligible_articles.append(article)

    return eligible_articles[:selected_limit]


def _decorate_article_links(articles: list[dict]) -> list[dict]:
    decorated_articles = []
    for article in articles:
        url = str(article.get("url", "")).strip()
        domain = ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

        decorated_articles.append(
            {
                **article,
                "domain": domain,
                "link_label": str(article.get("title", "")).strip() or "Open article",
            }
        )
    return decorated_articles

def _build_topic_list_context(form: TopicInputForm | None = None) -> dict:
    topics = Topic.objects.order_by("name")
    recent_runs = DigestRun.objects.select_related("topic").order_by("-created_at")[:10]
    return {
        "topics": topics,
        "recent_runs": recent_runs,
        "topic_form": form or TopicInputForm(),
    }


def _get_or_create_ui_topic(topic_name: str, source_url: str = "") -> Topic:
    user = _get_or_create_ui_user()
    topic, created = Topic.objects.get_or_create(
        user=user,
        name=topic_name,
        defaults={
            "source_url": source_url or None,
            "description": "",
            "keywords": [topic_name],
            "excluded_keywords": [],
            "is_active": True,
        },
    )

    if not created and source_url and topic.source_url != source_url:
        topic.source_url = source_url
        topic.save(update_fields=["source_url", "updated_at"])

    return topic


def _get_or_create_ui_user():
    user_model = get_user_model()
    user = user_model.objects.order_by("id").first()
    if user is not None:
        return user

    user, created = user_model.objects.get_or_create(username="digestflow-local-ui")
    if created:
        user.set_unusable_password()
        user.save(update_fields=["password"])
    return user


def _create_ui_digest_run(topic: Topic, source: str) -> DigestRun:
    return DigestRun.objects.create(
        topic=topic,
        input_snapshot={
            "mode": "manual",
            "source": source,
            "topic_name": topic.name,
            "source_url": topic.source_url or "",
        },
    )


def _start_topic_run(run: DigestRun, topic: Topic, default_source: str) -> None:
    if topic.source_url:
        raw_items = fetch_rss_articles(topic.source_url)
        if not raw_items:
            _mark_run_failed_for_empty_rss(run, topic.source_url)
            return

        run.input_snapshot = {
            **run.input_snapshot,
            "source": "topic_rss",
            "source_url": topic.source_url,
            "raw_items_count": len(raw_items),
        }
        run.save(update_fields=["input_snapshot", "updated_at"])
        run_digest_pipeline(run.id, raw_items=raw_items)
        return

    raw_items = get_demo_articles_for_topic(topic.name)
    run.input_snapshot = {
        **run.input_snapshot,
        "source": default_source,
        "source_url": "",
        "raw_items_count": len(raw_items),
    }
    run.save(update_fields=["input_snapshot", "updated_at"])
    run_digest_pipeline(run.id, raw_items=raw_items)


def _mark_run_failed_for_empty_rss(run: DigestRun, source_url: str) -> None:
    run.status = DigestRun.STATUS_FAILED
    run.error_message = f"RSS source returned no valid items: {source_url}"
    run.finished_at = timezone.now()
    run.input_snapshot = {
        **run.input_snapshot,
        "source": "topic_rss",
        "source_url": source_url,
        "raw_items_count": 0,
    }
    run.save(
        update_fields=[
            "status",
            "error_message",
            "finished_at",
            "input_snapshot",
            "updated_at",
        ]
    )
