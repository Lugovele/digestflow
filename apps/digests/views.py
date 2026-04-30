from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_GET, require_POST

from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic

from .forms import TopicInputForm


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
    topic = _get_or_create_ui_topic(topic_name)
    run = _create_ui_digest_run(topic, source="web_ui_form")

    raw_items = get_demo_articles_for_topic(topic.name)
    run_digest_pipeline(run.id, raw_items)

    return redirect("run-detail", run_id=run.id)


@require_POST
def run_pipeline_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    run = _create_ui_digest_run(topic, source="web_ui")

    raw_items = get_demo_articles_for_topic(topic.name)
    run_digest_pipeline(run.id, raw_items)

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

    context = {
        "run": run,
        "digest": digest,
        "content_package": content_package,
        "metrics": metrics,
        "article_ids": source_stage.get("article_ids", []),
        "articles_after_dedupe": source_stage.get("articles_after_dedupe"),
        "selected_for_prompt": ranking_stage.get("selected_for_prompt"),
        "total_tokens": total_tokens,
        "total_estimated_cost": total_estimated_cost,
        "digest_provider": digest_stage.get("provider"),
        "packaging_provider": packaging_stage.get("provider"),
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


def _build_topic_list_context(form: TopicInputForm | None = None) -> dict:
    topics = Topic.objects.order_by("name")
    recent_runs = DigestRun.objects.select_related("topic").order_by("-created_at")[:10]
    return {
        "topics": topics,
        "recent_runs": recent_runs,
        "topic_form": form or TopicInputForm(),
    }


def _get_or_create_ui_topic(topic_name: str) -> Topic:
    user = _get_or_create_ui_user()
    topic, _created = Topic.objects.get_or_create(
        user=user,
        name=topic_name,
        defaults={
            "description": "",
            "keywords": [topic_name],
            "excluded_keywords": [],
            "is_active": True,
        },
    )
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
            "mode": "demo",
            "source": source,
        },
    )
