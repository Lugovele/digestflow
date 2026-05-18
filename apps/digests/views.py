import json
import logging
from collections import Counter
from datetime import timedelta
from urllib.parse import urlparse

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.validators import URLValidator
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.db.models import F
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.digests import result_messages
from apps.digests.models import DigestRun
from apps.topics.focus import FOCUS_VALIDATION_MESSAGE, clean_focus_terms, validate_new_focus_terms
from apps.topics.focus_suggestions import generate_focus_suggestions, should_seed_focus_terms
from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import (
    CuratedSourceSeed,
    TopicSourceDiscoveryRequest,
    filter_new_source_candidates,
    get_demo_articles_for_topic,
    is_new_research_source,
    resolve_source_candidates,
)
from services.sources.detector import classify_source_url
from services.sources.rss_adapter import (
    fetch_dev_to_article_content,
    fetch_generic_web_article,
    fetch_rss_articles,
    inspect_generic_web_article,
)

from .forms import TOPIC_NAME_REQUIRED_MESSAGE, TopicInputForm

logger = logging.getLogger(__name__)
INSUFFICIENT_QUALITY_ERROR_FALLBACK = "Insufficient-quality diagnostics are available in metrics."
INSUFFICIENT_QUALITY_GENERIC_FALLBACK = "Not enough high-quality articles were available for a full digest."


@require_GET
def topic_list_view(request: HttpRequest) -> HttpResponse:
    return render(request, "digestflow/topic_list.html", _build_topic_list_context())


@require_GET
def topic_workspace_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    return _render_topic_source_review(request, topic)


@require_POST
def discover_sources_view(request: HttpRequest) -> HttpResponse:
    form = TopicInputForm(request.POST)
    if not form.is_valid():
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = _get_topic_form_error(form)
        return render(request, "digestflow/topic_list.html", context, status=400)

    topic_name = form.cleaned_data["topic_name"]
    source_url = str(form.cleaned_data.get("source_url") or "").strip()
    source_mode = form.cleaned_data.get("source_mode") or TopicSourceMode.HYBRID
    try:
        topic = _get_or_create_ui_topic(
            topic_name,
            source_urls=[source_url] if source_url else [],
            source_mode=source_mode,
            topic_id=request.POST.get("topic_id"),
        )
    except ValidationError as exc:
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = str(exc)
        return render(request, "digestflow/topic_list.html", context, status=400)
    topic.manual_source_inputs = [source_url] if source_url else []
    return render(
        request,
        "digestflow/topic_list.html",
        _build_topic_list_context(
            form=form,
            discovered_topic=topic,
            discovered_source_candidates=_discover_and_prepare_candidates(topic),
        ),
    )


@require_POST
def create_topic_and_run_view(request: HttpRequest) -> HttpResponse:
    form = TopicInputForm(request.POST)
    if not form.is_valid():
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = _get_topic_form_error(form)
        return render(request, "digestflow/topic_list.html", context, status=400)

    topic_name = form.cleaned_data["topic_name"]
    source_url = str(form.cleaned_data.get("source_url") or "").strip()
    source_mode = form.cleaned_data.get("source_mode") or TopicSourceMode.HYBRID
    topic = _get_or_create_ui_topic(topic_name, source_urls=[source_url] if source_url else [], source_mode=source_mode)
    topic.manual_source_inputs = [source_url] if source_url else []
    run = _create_ui_digest_run(topic, source="web_ui_form")

    _start_topic_run(run, topic, default_source="web_ui_form")
    return redirect("run-detail", run_id=run.id)


@require_POST
def add_topic_source_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    raw_source_url = str(request.POST.get("source_url") or "")
    source_url = raw_source_url.strip()
    source_mode = str(request.POST.get("source_mode") or topic.source_mode or TopicSourceMode.HYBRID).strip()
    form = TopicInputForm(
        initial={
            "topic_name": topic.name,
            "source_url": "",
            "source_mode": source_mode,
        }
    )

    validation = _validate_topic_source_submission(topic, source_url)
    if not validation["ok"]:
        context = _build_topic_list_context(
            form=form,
            discovered_topic=topic,
            discovered_source_candidates=_discover_and_prepare_candidates(topic),
        )
        context["source_add_feedback"] = validation
        context["source_add_input_value"] = raw_source_url
        context["source_add_diagnostics_json"] = _serialize_source_add_diagnostics(validation)
        return render(request, "digestflow/topic_list.html", context, status=400)

    _ensure_manual_topic_source(
        topic,
        source_url,
        source_name=str(validation.get("resolved_title") or "").strip(),
    )
    topic.source_mode = source_mode
    topic.save(update_fields=["source_mode", "updated_at"])
    return _render_topic_source_review(
        request,
        topic,
        source_add_feedback=validation,
        source_add_diagnostics_json=_serialize_source_add_diagnostics(validation),
    )


@require_POST
def update_topic_focus_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    focus_terms = _parse_focus_terms(request.POST)
    validation_error = validate_new_focus_terms(_build_topic_focus_terms(topic), focus_terms)
    if validation_error:
        return _render_topic_source_review(
            request,
            topic,
            focus_feedback={
                "level": "error",
                "message": validation_error.message,
            },
            focus_input_value=validation_error.term,
            status=400,
        )
    if topic.keywords != focus_terms:
        topic.keywords = focus_terms
        topic.focus_initialized = True
        topic.save(update_fields=["keywords", "focus_initialized", "updated_at"])
    elif not topic.focus_initialized:
        topic.focus_initialized = True
        topic.save(update_fields=["focus_initialized", "updated_at"])
    return _render_topic_source_review(request, topic)


@require_POST
def run_pipeline_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    run = _create_ui_digest_run(topic, source="web_ui")

    _start_topic_run(run, topic, default_source="web_ui")
    return redirect("run-detail", run_id=run.id)


@require_POST
def run_with_selected_sources_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    discovered_source_candidates = _discover_and_prepare_candidates(topic)
    selected_source_urls = [str(raw_url).strip() for raw_url in request.POST.getlist("selected_source_urls") if str(raw_url).strip()]
    selected_candidates = _resolve_selected_source_candidates(
        selected_source_urls,
        discovered_source_candidates,
    )

    if not selected_candidates:
        context = _build_topic_list_context(
            discovered_topic=topic,
            discovered_source_candidates=discovered_source_candidates,
        )
        context["source_selection_error"] = "Select at least one source before generating the digest."
        return render(request, "digestflow/topic_list.html", context, status=400)

    run = _create_ui_digest_run(
        topic,
        source="selected_sources_web_ui",
        selected_source_urls=[str(candidate.get("url") or "").strip() for candidate in selected_candidates if str(candidate.get("url") or "").strip()],
    )
    _start_selected_source_run(run, topic, selected_candidates, default_source="selected_sources_web_ui")
    return redirect("run-detail", run_id=run.id)


@require_POST
def toggle_topic_source_view(request: HttpRequest, topic_id: int, source_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    source = get_object_or_404(TopicSource, pk=source_id, topic=topic)
    next_is_active = "is_active" in request.POST
    if source.is_active != next_is_active:
        source.is_active = next_is_active
        source.save(update_fields=["is_active", "updated_at"])
    return redirect("topic-workspace", topic_id=topic.id)


@require_POST
def remove_topic_source_view(request: HttpRequest, topic_id: int, source_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    source = get_object_or_404(TopicSource, pk=source_id, topic=topic)
    source.delete()
    return _render_topic_source_review(request, topic)


@require_POST
def delete_topic_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    topic.delete()
    return redirect("topic-list")


@require_POST
def reorder_topics_view(request: HttpRequest) -> JsonResponse:
    user = _get_or_create_ui_user()
    raw_topic_ids = request.POST.getlist("topic_ids")
    ordered_topic_ids: list[int] = []
    seen_topic_ids: set[int] = set()

    for raw_topic_id in raw_topic_ids:
        try:
            topic_id = int(str(raw_topic_id).strip())
        except (TypeError, ValueError):
            continue
        if topic_id in seen_topic_ids:
            continue
        seen_topic_ids.add(topic_id)
        ordered_topic_ids.append(topic_id)

    user_topics = list(Topic.objects.filter(user=user).order_by("display_order", "name"))
    if not ordered_topic_ids:
        return JsonResponse({"ok": False, "error": "No topics were provided."}, status=400)

    topic_by_id = {topic.id: topic for topic in user_topics}
    if set(ordered_topic_ids) != set(topic_by_id.keys()):
        return JsonResponse({"ok": False, "error": "Topic order did not match the saved topics list."}, status=400)

    for position, topic_id in enumerate(ordered_topic_ids, start=1):
        topic = topic_by_id[topic_id]
        if topic.display_order == position:
            continue
        topic.display_order = position
        topic.save(update_fields=["display_order", "updated_at"])

    return JsonResponse({"ok": True})


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
    raw_digest_articles = _get_digest_payload_articles(digest)
    selected_article_lookup = _build_selected_article_lookup(raw_digest_articles)
    digest_articles = _decorate_article_links(digest.get_articles() if digest else [])
    ranked_articles = _decorate_article_links(
        ranking_stage.get("ranking_scores", []),
        selected_article_lookup=selected_article_lookup,
    )
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
    decorated_top_rejected_articles = _decorate_article_links(
        ranking_stage.get("top_rejected_articles", []),
        selected_article_lookup=selected_article_lookup,
    )
    source_stage_report = _build_source_stage_report(source_stage)
    ranking_stage_report = _build_ranking_stage_report(
        ranking_stage,
        run.status,
        decorated_top_rejected_articles,
    )
    raw_metrics_json = json.dumps(metrics, ensure_ascii=False, indent=2, sort_keys=True) if metrics else "{}"

    digest_payload = {
        "title": digest.get_payload_title() if digest else "",
        "articles": digest_articles,
    }
    selected_ranked_articles = [article for article in ranked_articles if article.get("is_selected_for_digest")]
    content_package_report = _build_content_package_report(content_package, validation_report)
    copy_diagnostics_text = _build_copy_diagnostics_text(
        run=run,
        digest_payload=digest_payload,
        content_package=content_package,
        content_package_report=content_package_report,
        source_stage_report=source_stage_report,
        ranking_stage_report=ranking_stage_report,
        ranked_articles=ranked_articles,
        total_tokens=total_tokens,
        total_estimated_cost=total_estimated_cost,
        digest_provider=digest_stage.get("provider"),
        packaging_provider=packaging_stage.get("provider"),
        validation_status=(
            content_package.validation_report.get("status", "unknown")
            if content_package
            else "not_available"
        ),
        raw_metrics_json=raw_metrics_json,
        display_error_message=_get_display_error_message(run),
        insufficient_quality_message=_get_insufficient_quality_message(run),
        is_insufficient_quality=is_insufficient_quality,
    )

    context = {
        "run": run,
        "digest_payload": digest_payload,
        "has_digest": digest is not None,
        "content_package": content_package,
        "display_error_message": _get_display_error_message(run),
        "is_insufficient_quality": is_insufficient_quality,
        "insufficient_quality_message": _get_insufficient_quality_message(run),
        "metrics": metrics,
        "raw_metrics_json": raw_metrics_json,
        "source_stage_report": source_stage_report,
        "ranking_stage_report": ranking_stage_report,
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
        "top_rejected_articles": decorated_top_rejected_articles,
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
        "content_package_report": content_package_report,
        "copy_diagnostics_text": copy_diagnostics_text,
        "hook_variants": content_package.hook_variants if content_package else [],
        "cta_variants": content_package.cta_variants if content_package else [],
        "primary_hook": content_package.primary_hook() if content_package else "",
        "primary_cta": content_package.primary_cta() if content_package else "",
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


def _get_display_error_message(run: DigestRun) -> str:
    if run.status == DigestRun.STATUS_INSUFFICIENT_QUALITY:
        return INSUFFICIENT_QUALITY_ERROR_FALLBACK if run.error_message else ""
    return run.error_message


def _get_insufficient_quality_message(run: DigestRun) -> str:
    result_message = str(getattr(run, "result_message", "") or "").strip()
    if result_message:
        return result_message

    centralized_message = str(getattr(result_messages, "INSUFFICIENT_QUALITY", "") or "").strip()
    if centralized_message:
        return centralized_message

    return INSUFFICIENT_QUALITY_GENERIC_FALLBACK


def _display_metric_value(value):
    if value is None:
        return "-"
    return value


def _build_source_stage_report(source_stage: dict) -> dict:
    if not isinstance(source_stage, dict):
        source_stage = {}

    extraction_items = [
        ("Raw feed items", _display_metric_value(source_stage.get("raw_items_count"))),
        ("Article links extracted", _display_metric_value(source_stage.get("article_links_extracted"))),
        ("Article contents fetched", _display_metric_value(source_stage.get("article_contents_fetched"))),
    ]
    if (source_stage.get("content_unavailable_count") or 0) > 0:
        extraction_items.append(
            ("Content unavailable", _display_metric_value(source_stage.get("content_unavailable_count")))
        )

    cleaning_items = [
        ("Articles before cleaning", _display_metric_value(source_stage.get("articles_count"))),
        ("Articles after cleaning", _display_metric_value(source_stage.get("articles_after_cleaning"))),
    ]
    if (source_stage.get("removed_during_cleaning") or 0) > 0:
        cleaning_items.append(
            ("Removed during cleaning", _display_metric_value(source_stage.get("removed_during_cleaning")))
        )

    content_tier_items = [("Full articles", _display_metric_value(source_stage.get("full_article_count")))]
    optional_tier_items = [
        ("Summary-only items", source_stage.get("rich_summary_count")),
        ("Short snippets", source_stage.get("weak_snippet_count")),
        ("No extracted content", source_stage.get("missing_content_count")),
    ]
    for label, value in optional_tier_items:
        if (value or 0) > 0:
            content_tier_items.append((label, _display_metric_value(value)))

    return {
        "status": source_stage.get("status", "unknown"),
        "source_url": source_stage.get("source_url"),
        "detected_source_type": source_stage.get("detected_source_type")
        or source_stage.get("normalized_source_type"),
        "detection_reason": source_stage.get("detection_reason"),
        "extraction_items": extraction_items,
        "cleaning_items": cleaning_items,
        "content_tier_items": content_tier_items,
        "cleaning_rejections": _decorate_article_links(source_stage.get("cleaning_rejections", [])),
        "dedupe_items": [
            ("Articles after dedupe", _display_metric_value(source_stage.get("articles_after_dedupe"))),
            ("Duplicate URLs removed", _display_metric_value(source_stage.get("duplicate_urls_removed"))),
            ("Duplicate titles removed", _display_metric_value(source_stage.get("duplicate_titles_removed"))),
        ],
        "persistence_items": [
            ("Saved articles", _display_metric_value(source_stage.get("saved_articles_count"))),
        ],
    }


def _build_ranking_stage_report(
    ranking_stage: dict,
    run_status: str,
    top_rejected_articles: list[dict],
) -> dict:
    if not isinstance(ranking_stage, dict):
        ranking_stage = {}

    return {
        "status": ranking_stage.get("status", "unknown"),
        "pipeline_items": [
            ("Articles processed", _display_metric_value(ranking_stage.get("ranked_articles_count"))),
            ("Selected for digest", _display_metric_value(ranking_stage.get("selected_for_prompt"))),
        ],
        "summary_items": [
            ("Articles ranked", _display_metric_value(ranking_stage.get("ranked_articles_count"))),
            (
                "Articles above threshold",
                _display_metric_value(ranking_stage.get("articles_above_quality_threshold")),
            ),
            ("Quality threshold", _display_metric_value(ranking_stage.get("quality_threshold"))),
            ("Average quality score", _display_metric_value(ranking_stage.get("average_quality_score"))),
            ("Max quality score", _display_metric_value(ranking_stage.get("max_quality_score"))),
            ("Min actual quality score", _display_metric_value(ranking_stage.get("min_actual_quality_score"))),
            (
                "Rejected low quality articles",
                _display_metric_value(ranking_stage.get("rejected_low_quality_count")),
            ),
            ("Selected for prompt", _display_metric_value(ranking_stage.get("selected_for_prompt"))),
        ],
        "decision_message": _build_ranking_decision_message(ranking_stage, run_status),
        "top_rejected_article": top_rejected_articles[0] if top_rejected_articles else None,
    }


def _build_ranking_decision_message(ranking_stage: dict, run_status: str) -> str:
    if run_status == DigestRun.STATUS_INSUFFICIENT_QUALITY:
        return "Digest generation skipped because too few articles passed quality validation."
    if ranking_stage.get("selected_for_prompt"):
        return "Digest generation proceeded with the selected articles."
    return "No ranking decision details are available."


def _build_content_package_report(content_package, validation_report: dict) -> dict:
    if not content_package:
        return {
            "hook_variants": [],
            "cta_variants": [],
            "hashtags": [],
            "carousel_items": [],
            "validation_items": [],
            "quality_check_items": [],
            "validation_report_json": "{}",
        }

    hooks = _normalize_text_list(getattr(content_package, "hook_variants", []))
    ctas = _normalize_text_list(getattr(content_package, "cta_variants", []))
    hashtags = _normalize_text_list(getattr(content_package, "hashtags", []))
    carousel_outline = getattr(content_package, "carousel_outline", [])
    if not isinstance(carousel_outline, list):
        carousel_outline = []

    return {
        "hook_variants": hooks,
        "cta_variants": ctas,
        "hashtags": hashtags,
        "carousel_items": [
            _format_carousel_outline_item(index, item)
            for index, item in enumerate(carousel_outline, start=1)
        ],
        "validation_items": [
            ("Post length", _display_metric_value(validation_report.get("post_text_length"))),
            ("Hook variants", _display_metric_value(validation_report.get("hook_variants_count"))),
            ("CTA options", _display_metric_value(validation_report.get("cta_variants_count"))),
            ("Hashtags", _display_metric_value(validation_report.get("hashtags_count"))),
            ("Carousel slides", _display_metric_value(validation_report.get("carousel_outline_count"))),
        ],
        "quality_check_items": _build_quality_check_items(validation_report.get("quality_checks", {})),
        "validation_report_json": (
            json.dumps(validation_report, ensure_ascii=False, indent=2, sort_keys=True)
            if validation_report
            else "{}"
        ),
    }


def _normalize_text_list(values) -> list[str]:
    if not isinstance(values, list):
        return []

    normalized = []
    for value in values:
        text = str(value or "").strip()
        if text:
            normalized.append(text)
    return normalized


def _format_carousel_outline_item(index: int, item) -> str:
    if isinstance(item, str):
        text = item.strip()
        return f"Slide {index}: {text}" if text else f"Slide {index}"

    if isinstance(item, dict):
        for key in ("title", "headline", "hook", "slide_title", "body", "summary"):
            value = str(item.get(key, "") or "").strip()
            if value:
                return f"Slide {index}: {value}"
        return f"Slide {index}: {json.dumps(item, ensure_ascii=False, sort_keys=True)}"

    text = str(item or "").strip()
    return f"Slide {index}: {text}" if text else f"Slide {index}"


def _build_quality_check_items(quality_checks) -> list[tuple[str, str]]:
    if not isinstance(quality_checks, dict):
        return []

    items = []
    for key, value in quality_checks.items():
        label = str(key).replace("_", " ").capitalize()
        if isinstance(value, bool):
            rendered = "pass" if value else "fail"
        elif isinstance(value, list):
            rendered = ", ".join(str(item) for item in value if item) or "-"
        elif isinstance(value, dict):
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        else:
            rendered = str(value)
        items.append((label, rendered))
    return items


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


def _get_digest_payload_articles(digest) -> list[dict]:
    if not digest:
        return []

    payload = digest.payload if isinstance(digest.payload, dict) else {}
    raw_articles = payload.get("articles", [])
    if not isinstance(raw_articles, list):
        return []
    return [article for article in raw_articles if isinstance(article, dict)]


def _normalize_identity_text(value) -> str:
    return " ".join(str(value or "").strip().lower().split())


def _article_identity_keys(article: dict) -> set[str]:
    if not isinstance(article, dict):
        return set()

    keys: set[str] = set()
    for key_name in ("article_id", "id"):
        value = article.get(key_name)
        if value not in (None, ""):
            keys.add(f"id:{value}")

    url = _normalize_identity_text(article.get("url"))
    if url:
        keys.add(f"url:{url}")

    title = _normalize_identity_text(article.get("title"))
    if title:
        keys.add(f"title:{title}")

    return keys


def _build_selected_article_lookup(articles: list[dict]) -> set[str]:
    selected_lookup: set[str] = set()
    for article in articles:
        selected_lookup.update(_article_identity_keys(article))
    return selected_lookup


def _decorate_article_links(
    articles: list[dict],
    selected_article_lookup: set[str] | None = None,
) -> list[dict]:
    selected_article_lookup = selected_article_lookup or set()
    decorated_articles = []
    for article in articles:
        if not isinstance(article, dict):
            continue
        url = str(article.get("url", "")).strip()
        domain = ""
        if url:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]

        is_selected_for_digest = bool(_article_identity_keys(article) & selected_article_lookup)
        article_with_selection = {
            **article,
            "is_selected_for_digest": is_selected_for_digest,
        }

        decorated_articles.append(
            {
                **article_with_selection,
                "domain": domain,
                "link_label": str(article.get("title", "")).strip() or "Open article",
                "weighted_tag_details": _build_weighted_tag_details(article_with_selection.get("weighted_secondary_tags")),
                "heading_diagnostics_display": _build_heading_diagnostics_display(article_with_selection.get("heading_diagnostics")),
                "article_card": _build_article_card(article_with_selection, domain),
            }
        )
    return decorated_articles


def _build_article_card(article: dict, domain: str) -> dict:
    quality_reasons = article.get("quality_reasons")
    rejection_reasons = article.get("rejection_reasons")
    diagnostic_warnings = article.get("diagnostic_warnings")
    classification_signal_summary = article.get("classification_signal_summary")

    if not isinstance(quality_reasons, list):
        quality_reasons = []
    if not isinstance(rejection_reasons, list):
        rejection_reasons = []
    if not isinstance(diagnostic_warnings, list):
        diagnostic_warnings = []
    if not isinstance(classification_signal_summary, dict):
        classification_signal_summary = {}

    weighted_tag_details = _build_weighted_tag_details(article.get("weighted_secondary_tags"))
    heading_display = _build_heading_diagnostics_display(article.get("heading_diagnostics"))

    return {
        "source_label": str(article.get("source_name") or domain or "Unknown source"),
        "quality_label": _display_metric_value(article.get("quality_score")),
        "score_label": _display_metric_value(article.get("score")),
        "type_label": str(article.get("primary_article_type") or article.get("article_type") or "unknown"),
        "selection_label": "selected" if article.get("is_selected_for_digest") else "rejected",
        "summary_reason": str(
            article.get("topic_relevance_reason")
            or (quality_reasons[0] if quality_reasons else "")
            or article.get("article_type_reason")
            or ""
        ),
        "top_reasons": quality_reasons[:3],
        "dominant_tags": article.get("dominant_tags") or [],
        "supporting_tags": article.get("supporting_tags") or [],
        "weak_tags": article.get("weak_tags") or [],
        "dominant_tags_display": _build_tag_display(article.get("dominant_tags") or []),
        "supporting_tags_display": _build_tag_display(article.get("supporting_tags") or []),
        "weak_tags_display": _build_tag_display(article.get("weak_tags") or [], limit=4),
        "weak_tag_count": len(article.get("weak_tags") or []),
        "weighted_table_rows": _build_weighted_tag_table_rows(weighted_tag_details),
        "classification_sections": _build_classification_signal_sections(classification_signal_summary),
        "relevance_sections": _build_relevance_signal_sections(article),
        "heading_display": heading_display,
        "editorial_alignment_reason": _extract_editorial_alignment_reason(quality_reasons),
        "weighted_tag_details": weighted_tag_details,
        "diagnostic_warnings": diagnostic_warnings,
        "rejection_reasons": rejection_reasons,
    }


def _build_copy_diagnostics_text(
    *,
    run: DigestRun,
    digest_payload: dict,
    content_package,
    content_package_report: dict,
    source_stage_report: dict,
    ranking_stage_report: dict,
    ranked_articles: list[dict],
    total_tokens,
    total_estimated_cost,
    digest_provider,
    packaging_provider,
    validation_status: str,
    raw_metrics_json: str,
    display_error_message: str,
    insufficient_quality_message: str,
    is_insufficient_quality: bool,
) -> str:
    lines = [
        f"Run ID: {run.id}",
        f"Topic: {run.topic.name}",
        f"Status: {run.status}",
        (
            "Result: insufficient quality"
            if is_insufficient_quality
            else f"Result: {run.result_message or '-'}"
        ),
        (
            "Error: see diagnostics"
            if is_insufficient_quality and display_error_message
            else f"Error: {display_error_message or '-'}"
        ),
        "",
        "Digest",
        f"Title: {digest_payload.get('title') or '-'}",
        f"Digest articles: {len(digest_payload.get('articles') or [])}",
    ]

    digest_articles = digest_payload.get("articles") or []
    if digest_articles:
        lines.append("Digest article summaries:")
        for index, article in enumerate(digest_articles, start=1):
            lines.extend(
                [
                    f"{index}. {article.get('title') or 'Untitled article'}",
                    f"   URL: {article.get('url') or '-'}",
                    f"   Summary: {article.get('summary') or '-'}",
                ]
            )
            key_points = article.get("key_points") or []
            if key_points:
                lines.append("   Key points:")
                for point in key_points:
                    lines.append(f"   - {point}")
    else:
        lines.append("Digest article summaries: none")

    lines.extend(
        [
            "",
            "Content package",
            f"Validation status: {validation_status}",
            f"Generated post: {getattr(content_package, 'post_text', '') or '-'}",
            f"Primary hook: {content_package.primary_hook() if content_package else '-'}",
            f"Primary CTA: {content_package.primary_cta() if content_package else '-'}",
            "Hooks:",
            *_format_text_list_for_copy(content_package_report.get("hook_variants")),
            "CTA options:",
            *_format_text_list_for_copy(content_package_report.get("cta_variants")),
            "Hashtags:",
            *_format_text_list_for_copy(content_package_report.get("hashtags")),
            "Carousel outline:",
            *_format_text_list_for_copy(content_package_report.get("carousel_items")),
            "Validation report / quality checks:",
            *_format_label_value_items_for_copy(content_package_report.get("validation_items")),
            *_format_label_value_items_for_copy(content_package_report.get("quality_check_items")),
            "",
            "Pipeline diagnostics",
            f"Source stage status: {source_stage_report.get('status', 'unknown')}",
            f"Source URL: {source_stage_report.get('source_url') or '-'}",
            f"Detected source type: {source_stage_report.get('detected_source_type') or '-'}",
            f"Detection reason: {source_stage_report.get('detection_reason') or '-'}",
            "Source extraction:",
            *_format_label_value_items_for_copy(source_stage_report.get("extraction_items")),
            "Source cleaning:",
            *_format_label_value_items_for_copy(source_stage_report.get("cleaning_items")),
            "Content tiers:",
            *_format_label_value_items_for_copy(source_stage_report.get("content_tier_items")),
            "Deduplication:",
            *_format_label_value_items_for_copy(source_stage_report.get("dedupe_items")),
            "Persistence:",
            *_format_label_value_items_for_copy(source_stage_report.get("persistence_items")),
            "",
            "Ranking summary",
            f"Ranking status: {ranking_stage_report.get('status', 'unknown')}",
            *_format_label_value_items_for_copy(ranking_stage_report.get("summary_items")),
            f"Pipeline decision: {ranking_stage_report.get('decision_message') or '-'}",
            f"Total tokens: {total_tokens if total_tokens is not None else '-'}",
            f"Total estimated cost: {total_estimated_cost if total_estimated_cost is not None else '-'}",
            f"Digest provider: {digest_provider or '-'}",
            f"Packaging provider: {packaging_provider or '-'}",
            "",
            f"All ranked articles ({len(ranked_articles)})",
        ]
    )

    for index, article in enumerate(ranked_articles, start=1):
        lines.extend(_build_ranked_article_copy_lines(index, article))

    lines.extend(["", "Raw metrics JSON", raw_metrics_json or "{}"])
    return "\n".join(lines)


def _format_text_list_for_copy(values) -> list[str]:
    values = [str(value).strip() for value in (values or []) if str(value or "").strip()]
    if not values:
        return ["- none"]
    return [f"- {value}" for value in values]


def _format_label_value_items_for_copy(items) -> list[str]:
    formatted_lines = []
    for item in items or []:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        formatted_lines.append(f"- {item[0]}: {item[1]}")
    if not formatted_lines:
        return ["- none"]
    return formatted_lines


def _build_ranked_article_copy_lines(index: int, article: dict) -> list[str]:
    article_card = article.get("article_card") or {}
    lines = [
        f"{index}. {article.get('title') or article.get('link_label') or 'Untitled article'}",
        f"   URL: {article.get('url') or '-'}",
        f"   Source: {article_card.get('source_label') or '-'}",
        f"   Status: {'selected' if article.get('is_selected_for_digest') else 'rejected'}",
        f"   Quality score: {article.get('quality_score') if article.get('quality_score') is not None else '-'}",
        f"   Total score: {article.get('score') if article.get('score') is not None else '-'}",
        f"   Article type: {article.get('primary_article_type') or article.get('article_type') or 'unknown'}",
        f"   Topic relevance: {article.get('topic_relevance_score') if article.get('topic_relevance_score') is not None else '-'}",
        f"   Topic relevance reason: {article.get('topic_relevance_reason') or '-'}",
        f"   Topic specificity: {article.get('topic_specificity_score') if article.get('topic_specificity_score') is not None else '-'}",
        f"   Topic specificity reason: {article.get('topic_specificity_reason') or '-'}",
        f"   Dominant tags: {', '.join(article.get('dominant_tags') or []) or '-'}",
        f"   Supporting tags: {', '.join(article.get('supporting_tags') or []) or '-'}",
        f"   Weak tags: {', '.join(article.get('weak_tags') or []) or '-'}",
        "   Top reasons:",
    ]
    top_reasons = article_card.get("top_reasons") or []
    lines.extend([f"   - {reason}" for reason in top_reasons] or ["   - none"])

    heading_display = article_card.get("heading_display") or {}
    matched_heading_tags = heading_display.get("matched_heading_tags") or []
    lines.extend(
        [
            "   Heading diagnostics:",
            f"   - source: {heading_display.get('heading_source') or 'none'}",
            f"   - strategy: {heading_display.get('heading_extraction_strategy') or 'none'}",
            f"   - raw html heading count: {heading_display.get('raw_html_heading_count') if heading_display.get('raw_html_heading_count') is not None else 0}",
            f"   - extracted heading count: {heading_display.get('extracted_heading_count') if heading_display.get('extracted_heading_count') is not None else 0}",
            f"   - detected headings: {'; '.join(heading_display.get('detected_headings') or []) or 'none'}",
            f"   - sample headings: {'; '.join(heading_display.get('sample_detected_headings') or []) or 'none'}",
            f"   - normalized headings: {'; '.join(heading_display.get('normalized_headings') or []) or 'none'}",
        ]
    )
    if matched_heading_tags:
        lines.append("   - matched heading tags:")
        for tag in matched_heading_tags:
            lines.append(f"     * {tag.get('tag')}: {', '.join(tag.get('matches') or []) or 'none'}")

    lines.append("   Editorial weighting:")
    for row in article_card.get("weighted_table_rows") or []:
        lines.append(
            "   - "
            f"{row.get('tag')} | role={row.get('role')} | weight={row.get('weight')} | "
            f"title={'yes' if row.get('title') else 'no'} | intro={'yes' if row.get('intro') else 'no'} | "
            f"heading={'yes' if row.get('heading') else 'no'} | body={row.get('body')} | notes={row.get('notes') or '-'}"
        )
    if not article_card.get("weighted_table_rows"):
        lines.append("   - none")

    classification_sections = article_card.get("classification_sections") or []
    if classification_sections:
        lines.append("   Classification signals:")
        for section in classification_sections:
            label = section.get("label") or "Signals"
            items = section.get("items") or []
            lines.append(f"   - {label}: {', '.join(items) if items else 'none'}")

    relevance_sections = article_card.get("relevance_sections") or []
    if relevance_sections:
        lines.append("   Relevance diagnostics:")
        for section in relevance_sections:
            label = section.get("label") or "Signals"
            items = section.get("items") or []
            lines.append(f"   - {label}: {', '.join(items) if items else 'none'}")

    rejection_reasons = article_card.get("rejection_reasons") or []
    if rejection_reasons:
        lines.append(f"   Rejection reasons: {', '.join(rejection_reasons)}")

    diagnostic_warnings = article_card.get("diagnostic_warnings") or []
    if diagnostic_warnings:
        lines.append(f"   Diagnostic warnings: {', '.join(diagnostic_warnings)}")

    return lines


def _build_tag_display(tags: list[str], limit: int = 6) -> dict:
    tags = [str(tag) for tag in tags if str(tag).strip()]
    visible = tags[:limit]
    hidden_count = max(0, len(tags) - len(visible))
    return {
        "visible": visible,
        "hidden_count": hidden_count,
    }


def _build_weighted_tag_details(weighted_tags) -> dict[str, list[dict]]:
    if not isinstance(weighted_tags, dict):
        return {"dominant": [], "supporting": [], "weak": []}

    def to_detail(tag: str, payload: dict) -> dict:
        return {
            "tag": tag,
            "strength": payload.get("strength"),
            "reason": payload.get("reason", ""),
            "signals": payload.get("signals", []),
            "title_matches": payload.get("title_matches", []),
            "intro_matches": payload.get("intro_matches", []),
            "heading_matches": payload.get("heading_matches", []),
            "body_match_count": payload.get("body_match_count"),
            "editorial_weight": payload.get("editorial_weight"),
            "body_weight_component": payload.get("body_weight_component"),
            "body_saturation_applied": payload.get("body_saturation_applied"),
            "heading_weight_component": payload.get("heading_weight_component"),
            "centrality_reason": payload.get("centrality_reason", ""),
        }

    dominant: list[dict] = []
    supporting: list[dict] = []
    weak: list[dict] = []
    for tag, payload in weighted_tags.items():
        if not isinstance(payload, dict):
            continue
        strength = payload.get("strength")
        detail = to_detail(tag, payload)
        if strength == 2.0:
            dominant.append(detail)
        elif strength == 1.0:
            supporting.append(detail)
        elif strength == 0.5:
            weak.append(detail)
    return {"dominant": dominant, "supporting": supporting, "weak": weak}


def _build_weighted_tag_table_rows(weighted_tag_details: dict[str, list[dict]]) -> list[dict]:
    rows: list[dict] = []
    for role in ("dominant", "supporting"):
        for detail in weighted_tag_details.get(role, []):
            rows.append(
                {
                    "tag": detail.get("tag", ""),
                    "role": role,
                    "weight": detail.get("editorial_weight"),
                    "strength": detail.get("strength"),
                    "title": bool(detail.get("title_matches")),
                    "intro": bool(detail.get("intro_matches")),
                    "heading": bool(detail.get("heading_matches")),
                    "body": detail.get("body_match_count"),
                    "saturated": detail.get("body_saturation_applied"),
                    "notes": detail.get("centrality_reason") or detail.get("reason") or "",
                    "title_matches": detail.get("title_matches", []),
                    "intro_matches": detail.get("intro_matches", []),
                    "heading_matches": detail.get("heading_matches", []),
                    "body_weight_component": detail.get("body_weight_component"),
                    "heading_weight_component": detail.get("heading_weight_component"),
                }
            )
    return rows


def _build_classification_signal_sections(classification_signal_summary: dict) -> list[dict]:
    sections: list[dict] = []
    for key, label in (
        ("primary_signals", "Primary signals"),
        ("tag_signals", "Tag signals"),
    ):
        values = classification_signal_summary.get(key)
        if isinstance(values, list) and values:
            sections.append({"label": label, "items": values})
    return sections


def _build_relevance_signal_sections(article: dict) -> list[dict]:
    sections: list[dict] = []
    for key, label in (
        ("relevance_signals", "Relevance signals"),
        ("weak_relevance_signals", "Weak relevance signals"),
        ("missing_relevance_signals", "Missing relevance signals"),
        ("specificity_signals", "Specificity signals"),
        ("generic_topic_signals", "Generic topic signals"),
    ):
        values = article.get(key)
        if isinstance(values, list) and values:
            sections.append({"label": label, "items": values})
    return sections


def _extract_editorial_alignment_reason(quality_reasons: list[str]) -> str:
    for reason in quality_reasons:
        if reason in {
            "editorial center aligns with topic",
            "supporting editorial theme aligns with topic",
        }:
            return reason
    return ""


def _build_heading_diagnostics_display(heading_diagnostics) -> dict:
    if not isinstance(heading_diagnostics, dict):
        heading_diagnostics = {}

    detected_headings = heading_diagnostics.get("detected_headings")
    normalized_headings = heading_diagnostics.get("normalized_headings")
    matched_heading_tags = heading_diagnostics.get("matched_heading_tags")
    sample_detected_headings = heading_diagnostics.get("sample_detected_headings")
    if not isinstance(detected_headings, list):
        detected_headings = []
    if not isinstance(normalized_headings, list):
        normalized_headings = []
    if not isinstance(matched_heading_tags, dict):
        matched_heading_tags = {}
    if not isinstance(sample_detected_headings, list):
        sample_detected_headings = detected_headings[:5]

    matched_tag_details = []
    for tag, payload in matched_heading_tags.items():
        if not isinstance(payload, dict):
            continue
        matches = payload.get("matches")
        normalized_matches = payload.get("normalized_matches")
        matched_tag_details.append(
            {
                "tag": tag,
                "matches": matches if isinstance(matches, list) else [],
                "normalized_matches": normalized_matches if isinstance(normalized_matches, list) else [],
            }
        )

    return {
        "detected_headings": detected_headings[:12],
        "normalized_headings": normalized_headings[:12],
        "heading_count": heading_diagnostics.get("heading_count", 0),
        "raw_html_heading_count": heading_diagnostics.get("raw_html_heading_count", 0),
        "extracted_heading_count": heading_diagnostics.get("extracted_heading_count", len(detected_headings)),
        "heading_extraction_strategy": heading_diagnostics.get("heading_extraction_strategy", "none"),
        "sample_detected_headings": [str(value).strip() for value in sample_detected_headings if str(value).strip()][:5],
        "heading_source": heading_diagnostics.get("heading_source", "none"),
        "matched_heading_tags": matched_tag_details,
    }


def _get_topic_form_error(form: TopicInputForm) -> str:
    topic_errors = form.errors.get("topic_name", [])
    if topic_errors:
        return str(topic_errors[0])
    return TOPIC_NAME_REQUIRED_MESSAGE

def _build_topic_list_context(
    form: TopicInputForm | None = None,
    *,
    discovered_topic: Topic | None = None,
    discovered_source_candidates: list[dict] | None = None,
    focus_feedback: dict | None = None,
    focus_input_value: str = "",
) -> dict:
    user = _get_or_create_ui_user()
    all_candidate_records = discovered_source_candidates or []
    visible_new_source_candidates = _build_visible_new_source_candidates(all_candidate_records)
    topics = list(
        Topic.objects.filter(user=user)
        .order_by("display_order", "name")
        .prefetch_related("sources")
    )
    for topic in topics:
        topic.source_count = sum(1 for source in topic.sources.all() if source.origin != TopicSourceOrigin.DISCOVERED)
        topic.active_source_count = sum(
            1 for source in topic.sources.all() if source.is_active and source.origin != TopicSourceOrigin.DISCOVERED
        )
        topic.legacy_source_display = _build_legacy_source_display(topic)
    recent_runs = DigestRun.objects.filter(topic__user=user).select_related("topic").order_by("-created_at")[:10]
    for run in recent_runs:
        run.display_time = _format_recent_run_time(run.created_at)
    return {
        "topics": topics,
        "recent_runs": recent_runs,
        "topic_form": form or TopicInputForm(),
        "discovered_topic": discovered_topic,
        "focus_terms": _build_topic_focus_terms(discovered_topic),
        "focus_feedback": focus_feedback,
        "focus_input_value": focus_input_value,
        "discovered_source_candidates": visible_new_source_candidates,
        "source_review_summary": _build_source_review_summary(discovered_topic, visible_new_source_candidates),
        "topic_source_inventory": _build_topic_source_inventory(discovered_topic),
        "active_saved_source_urls": _build_active_saved_source_urls(discovered_topic),
        "active_selected_source_urls": _build_active_selected_source_urls(discovered_topic),
        "selected_source_count": _build_selected_source_count(discovered_topic, visible_new_source_candidates),
        "legacy_topic_source": _build_legacy_source_display(discovered_topic),
        "source_add_feedback": None,
    }


def _format_recent_run_time(created_at):
    local_created_at = timezone.localtime(created_at)
    now = timezone.localtime()
    today = now.date()
    created_date = local_created_at.date()
    if created_date == today:
        elapsed = max(now - local_created_at, timedelta(minutes=1))
        if elapsed < timedelta(hours=1):
            minutes = max(1, int(elapsed.total_seconds() // 60))
            unit = "minute" if minutes == 1 else "minutes"
            return f"{minutes} {unit} ago"
        hours = max(1, int(elapsed.total_seconds() // 3600))
        unit = "hour" if hours == 1 else "hours"
        return f"{hours} {unit} ago"
    if created_date == today - timedelta(days=1):
        return "Yesterday"
    return f"{local_created_at.strftime('%b')} {local_created_at.day}"


def _render_topic_source_review(
    request: HttpRequest,
    topic: Topic,
    *,
    status: int = 200,
    source_add_feedback: dict | None = None,
    source_add_input_value: str = "",
    source_add_diagnostics_json: str = "",
    focus_feedback: dict | None = None,
    focus_input_value: str = "",
) -> HttpResponse:
    if focus_feedback is None:
        _ensure_topic_focus_seeded(topic)
    form = TopicInputForm(
        initial={
            "topic_name": topic.name,
            "source_url": "",
            "source_mode": topic.source_mode,
        }
    )
    context = _build_topic_list_context(
        form=form,
        discovered_topic=topic,
        discovered_source_candidates=_discover_and_prepare_candidates(topic),
        focus_feedback=focus_feedback,
        focus_input_value=focus_input_value,
    )
    context["source_add_feedback"] = source_add_feedback
    context["source_add_input_value"] = source_add_input_value
    context["source_add_diagnostics_json"] = source_add_diagnostics_json
    return render(
        request,
        "digestflow/topic_list.html",
        context,
        status=status,
    )


def _build_topic_focus_terms(topic: Topic | None) -> list[str]:
    if topic is None:
        return []
    raw_terms = topic.keywords if isinstance(topic.keywords, list) else []
    return clean_focus_terms(raw_terms)


def _discover_and_prepare_candidates(topic: Topic) -> list[dict]:
    raw_candidate_records = resolve_source_candidates(
        TopicSourceDiscoveryRequest(
            topic=topic.name,
            focus_terms=_build_topic_focus_terms(topic),
            source_mode=topic.source_mode,
            manual_source_urls=list(getattr(topic, "manual_source_inputs", []) or []),
            curated_sources=_build_curated_source_seeds(topic),
        )
    )
    candidate_records = filter_new_source_candidates(raw_candidate_records, topic.sources.all())
    return _upsert_and_build_source_candidates(topic, candidate_records)


def _build_source_review_summary(
    topic: Topic | None,
    candidate_records: list[dict],
) -> dict:
    if topic is None:
        return {
            "mode": TopicSourceMode.HYBRID,
            "mode_label": TopicSourceMode(TopicSourceMode.HYBRID).label,
            "candidate_count": 0,
            "deduped_source_count": 0,
            "origin_counts": {},
            "discovered_candidates": [],
            "curated_candidates": [],
            "manual_candidates": [],
        }

    origin_counter = Counter()
    discovered_candidates: list[dict] = []
    curated_candidates: list[dict] = []
    manual_candidates: list[dict] = []

    for candidate in candidate_records:
        origin = str(candidate.get("candidate_origin") or "discovered")
        if candidate.get("is_manual"):
            origin = "manual"
        origin_counter[origin] += 1
        if origin == "manual":
            manual_candidates.append(candidate)
        elif origin == "curated":
            curated_candidates.append(candidate)
        else:
            discovered_candidates.append(candidate)

    mode = str(topic.source_mode or TopicSourceMode.HYBRID)
    try:
        mode_label = TopicSourceMode(mode).label
    except ValueError:
        mode_label = mode

    return {
        "mode": mode,
        "mode_label": mode_label,
        "candidate_count": len(candidate_records),
        "deduped_source_count": len(
            {
                str(candidate.get("normalized_url") or candidate.get("url") or "").strip()
                for candidate in candidate_records
                if str(candidate.get("normalized_url") or candidate.get("url") or "").strip()
            }
        ),
        "origin_counts": dict(origin_counter),
        "discovered_candidates": discovered_candidates,
        "curated_candidates": curated_candidates,
        "manual_candidates": manual_candidates,
    }


def _build_visible_new_source_candidates(candidate_records: list[dict]) -> list[dict]:
    visible_candidates: list[dict] = []
    for candidate in candidate_records:
        candidate_origin = str(candidate.get("candidate_origin") or "").strip().lower()
        if candidate_origin != TopicSourceOrigin.DISCOVERED:
            continue
        if candidate.get("is_pinned"):
            continue
        if candidate.get("persisted_source_id") and candidate_origin != TopicSourceOrigin.DISCOVERED:
            continue
        visible_candidates.append(candidate)
    return visible_candidates


def _build_active_saved_source_urls(topic: Topic | None) -> list[str]:
    if topic is None:
        return []
    return [
        str(source.url).strip()
        for source in topic.sources.filter(is_active=True).exclude(origin=TopicSourceOrigin.DISCOVERED).order_by("id")
        if str(source.url).strip()
    ]


def _build_active_selected_source_urls(topic: Topic | None) -> list[str]:
    if topic is None:
        return []
    return [
        str(source.url).strip()
        for source in topic.sources.filter(is_active=True).order_by("id")
        if str(source.url).strip()
    ]


def _build_selected_source_count(topic: Topic | None, candidate_records: list[dict]) -> int:
    saved_source_count = len(_build_active_saved_source_urls(topic))
    if topic is None or not topic.uses_source_discovery:
        return saved_source_count
    new_source_count = sum(1 for candidate in candidate_records if candidate.get("selected"))
    return saved_source_count + new_source_count


def _build_curated_source_seeds(topic: Topic) -> list[CuratedSourceSeed]:
    curated_sources: list[CuratedSourceSeed] = []
    for source in topic.sources.filter(is_active=True).exclude(origin=TopicSourceOrigin.DISCOVERED).order_by("id"):
        curated_sources.append(
            CuratedSourceSeed(
                url=source.url,
                title=_build_safe_saved_source_display_title(source.name, source.url),
                description=_build_topic_source_description(source),
                quality_estimate="manual" if source.origin == TopicSourceOrigin.MANUAL else "curated",
                is_manual=source.origin == TopicSourceOrigin.MANUAL,
                default_selected=source.is_active,
            )
        )
    return curated_sources


def _build_topic_source_inventory(topic: Topic | None) -> list[dict]:
    if topic is None:
        return []

    inventory: list[dict] = []
    for source in topic.sources.exclude(origin=TopicSourceOrigin.DISCOVERED).order_by("-id"):
        safe_display_title = _build_safe_saved_source_display_title(source.name, source.url)
        inventory.append(
            {
                "id": source.id,
                "name": safe_display_title,
                "url": source.url,
                "display_url": _build_compact_source_url(source.url),
                "source_type": source.source_type or "unknown",
                "origin": source.origin,
                "origin_label": source.get_origin_display(),
                "is_active": source.is_active,
                "validation_status": source.validation_status,
                "validation_status_label": source.get_validation_status_display(),
                "last_validation_error": source.last_validation_error,
            }
        )
    return inventory


def _build_legacy_source_display(topic: Topic | None) -> dict | None:
    if topic is None or not topic.source_url:
        return None

    normalized_source = classify_source_url(topic.source_url)
    if topic.sources.filter(normalized_url=normalized_source.normalized_url).exists():
        return None

    return {
        "url": topic.source_url,
        "normalized_url": normalized_source.normalized_url,
    }


def _validate_topic_source_submission(topic: Topic, source_url: str) -> dict:
    return _validate_topic_source_submission_v2(topic, source_url)

def _validate_topic_source_submission_v2(topic: Topic, source_url: str) -> dict:
    source_url = str(source_url or "").strip()
    if not source_url:
        return {
            "ok": False,
            "level": "error",
            "message": "Please add a source address before saving it to this topic.",
        }

    try:
        URLValidator()(source_url)
    except ValidationError:
        return {
            "ok": False,
            "level": "error",
            "message": "Please check the URL format.",
        }

    parsed_source = urlparse(source_url)
    if parsed_source.scheme not in {"http", "https"}:
        return {
            "ok": False,
            "level": "error",
            "message": "Use an http or https URL.",
        }

    try:
        normalized_source = classify_source_url(source_url)
    except Exception:
        return {
            "ok": False,
            "level": "error",
            "message": "Enter a valid URL before adding a source.",
        }

    if topic.sources.filter(normalized_url=normalized_source.normalized_url).exists():
        return {
            "ok": True,
            "level": "info",
            "message": "This source has already been added to this topic. Please check the address or use another source.",
            "normalized_source": normalized_source,
        }

    if normalized_source.source_type not in {
        "rss_feed",
        "devto_tag",
        "devto_article",
        "devto_author",
        "generic_html",
        "blog_index",
        "publication",
    }:
        return {
            "ok": False,
            "level": "error",
            "message": "Please check the address or try another article.",
        }

    availability = _validate_topic_source_availability(normalized_source, original_source_url=source_url)
    if not availability["ok"]:
        return availability

    return {
        "ok": True,
        "level": str(availability.get("level") or "success"),
        "message": str(
            availability.get("message")
            or "Source added and saved for this topic. It will be used when generating the digest."
        ),
        "normalized_source": normalized_source,
        "resolved_title": str(availability.get("resolved_title") or "").strip(),
        "diagnostics": availability.get("diagnostics"),
    }


def _validate_topic_source_availability(normalized_source, original_source_url: str = "") -> dict:
    source_type = normalized_source.source_type

    if source_type == "devto_article":
        try:
            article = fetch_dev_to_article_content(normalized_source.normalized_url)
        except Exception:
            article = None
        content = str(article.get("content") or "").strip() if isinstance(article, dict) else ""
        title = str(article.get("title") or "").strip() if isinstance(article, dict) else ""
        return {
            "ok": True,
            "resolved_title": title or _fallback_source_label(normalized_source.original_url),
            "diagnostics": {
                "normalized_url": normalized_source.normalized_url,
                "source_type": source_type,
                "fetch_status": 200 if content else None,
                "fetch_failure_reason": "" if content else "content unavailable during source entry validation",
                "extraction_strategy": "devto_article_fetch" if content else "devto_article_unverified",
                "usable_text_length": len(content),
                "rejection_reason": "" if content else "content unavailable during source entry validation",
            },
        }

    if source_type in {"generic_html", "blog_index", "publication"}:
        try:
            inspection = inspect_generic_web_article(original_source_url or normalized_source.normalized_url)
        except Exception:
            inspection = {
                "article": None,
                "diagnostics": {
                    "normalized_url": normalized_source.normalized_url,
                    "source_type": source_type,
                    "fetch_status": None,
                    "fetch_failure_reason": "source inspection failed during source entry validation",
                    "extraction_strategy": "inspection_failed",
                    "usable_text_length": 0,
                    "rejection_reason": "source inspection failed during source entry validation",
                },
            }
        article = inspection.get("article")
        diagnostics = inspection.get("diagnostics", {})
        content = str(article.get("content") or "").strip() if isinstance(article, dict) else ""
        title = str(article.get("title") or "").strip() if isinstance(article, dict) else ""
        if isinstance(article, dict) and content and title:
            return {"ok": True, "resolved_title": title, "diagnostics": diagnostics}

        return {
            "ok": True,
            "level": "success",
            "resolved_title": str(diagnostics.get("title") or "").strip() or _fallback_source_label(normalized_source.original_url),
            "diagnostics": diagnostics,
        }

    try:
        items = fetch_rss_articles(normalized_source.normalized_url)
    except Exception:
        items = []
    if items:
        return {"ok": True}

    return {
        "ok": True,
        "resolved_title": _fallback_source_label(normalized_source.original_url),
        "diagnostics": {
            "normalized_url": normalized_source.normalized_url,
            "source_type": source_type,
            "fetch_status": None,
            "fetch_failure_reason": "source content unavailable during source entry validation",
            "extraction_strategy": "unverified_source_entry",
            "usable_text_length": 0,
            "rejection_reason": "source content unavailable during source entry validation",
        },
    }


def _serialize_source_add_diagnostics(validation: dict | None) -> str:
    diagnostics = validation.get("diagnostics") if isinstance(validation, dict) else None
    if not isinstance(diagnostics, dict) or not diagnostics:
        return ""
    return json.dumps(diagnostics, ensure_ascii=False, sort_keys=True)


def _assess_generic_source_reachability(diagnostics: dict) -> dict:
    if not isinstance(diagnostics, dict):
        return {"ok": False, "message": "We could not reach this URL."}

    status = diagnostics.get("fetch_status")
    try:
        status_code = int(status) if status is not None else None
    except (TypeError, ValueError):
        status_code = None

    failure_reason = str(diagnostics.get("fetch_failure_reason") or "").strip().casefold()

    if status_code in {404, 410}:
        return {"ok": False, "message": "This page returned 404/410."}

    if "timed out" in failure_reason or "timeout" in failure_reason:
        return {"ok": False, "message": "We could not reach this URL."}

    if any(marker in failure_reason for marker in ("name or service not known", "nodename nor servname", "getaddrinfo", "dns", "failed to resolve", "temporary failure in name resolution")):
        return {"ok": False, "message": "We could not reach this URL."}

    if status_code is not None and status_code >= 500:
        return {"ok": False, "message": "We could not reach this URL."}

    if status_code is None and failure_reason:
        return {"ok": False, "message": "We could not reach this URL."}

    return {"ok": True, "message": ""}


def _build_topic_source_description(source: TopicSource) -> str:
    if source.origin == TopicSourceOrigin.MANUAL:
        return "User-added source for this topic."
    if source.origin == TopicSourceOrigin.DISCOVERED:
        return "Previously discovered source saved on this topic."
    return "Persistent curated source for this topic."


def _fallback_source_label(url: str) -> str:
    parsed = urlparse(str(url or ""))
    host = (parsed.netloc or "").strip()
    if host.lower().startswith("www."):
        host = host[4:]
    return host or str(url or "Source")


def _build_safe_saved_source_display_title(title: str, url: str) -> str:
    cleaned_title = str(title or "").strip()
    if cleaned_title and not _looks_like_blocked_or_error_page_title(cleaned_title):
        return cleaned_title
    return _fallback_source_label(url)


def _looks_like_blocked_or_error_page_title(title: str) -> bool:
    normalized = " ".join(str(title or "").strip().casefold().split())
    if not normalized:
        return True

    exact_blocked_titles = {
        "access denied",
        "forbidden",
        "403 forbidden",
        "not found",
        "404 not found",
        "just a moment...",
        "please enable javascript",
        "attention required",
        "request blocked",
        "service unavailable",
        "temporarily unavailable",
    }
    if normalized in exact_blocked_titles:
        return True

    blocked_title_markers = (
        "unable to give you access to our site",
        "access denied",
        "403 forbidden",
        "404 not found",
        "just a moment",
        "please enable javascript",
        "attention required",
        "request blocked",
        "service unavailable",
        "temporarily unavailable",
        "access to this page has been denied",
        "verify you are human",
        "checking your browser before accessing",
        "blocked by",
        "temporarily unavailable",
    )
    return any(marker in normalized for marker in blocked_title_markers)


def _build_compact_source_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if not parsed.scheme and not parsed.netloc:
        return str(url or "").strip()

    compact = f"{parsed.netloc}{parsed.path}".rstrip("/")
    if parsed.query:
        compact = f"{compact}?{parsed.query}"
    return compact or str(url or "").strip()


def _upsert_and_build_source_candidates(topic: Topic, candidate_records: list[dict]) -> list[dict]:
    existing_sources = list(topic.sources.all())
    existing_by_normalized = {source.normalized_url: source for source in existing_sources}
    prepared_candidates: list[dict] = []
    seen_discovered_normalized_urls: set[str] = set()

    for candidate in candidate_records:
        source_url = str(candidate.get("url") or "").strip()
        if not source_url:
            continue

        normalized = classify_source_url(source_url)
        seen_discovered_normalized_urls.add(normalized.normalized_url)
        source = existing_by_normalized.get(normalized.normalized_url)
        candidate_origin = _normalize_candidate_origin(
            candidate,
            source.origin if source is not None else TopicSourceOrigin.CURATED,
        )
        if source is not None and source.origin != TopicSourceOrigin.DISCOVERED:
            candidate_origin = source.origin
        if candidate_origin == TopicSourceOrigin.DISCOVERED:
            candidate_title = str(candidate.get("title") or "").strip()
            if source is None:
                source = TopicSource.objects.create(
                    topic=topic,
                    name=candidate_title,
                    url=source_url,
                    normalized_url=normalized.normalized_url,
                    source_type=normalized.source_type,
                    origin=TopicSourceOrigin.DISCOVERED,
                    platform=normalized.platform,
                    validation_status=TopicSource.VALIDATION_PENDING,
                    last_validation_error="",
                    is_active=True,
                )
                existing_by_normalized[normalized.normalized_url] = source
            elif source.origin == TopicSourceOrigin.DISCOVERED:
                source.name = candidate_title or source.name
                source.url = source_url
                source.source_type = normalized.source_type
                source.platform = normalized.platform
                source.save(
                    update_fields=[
                        "name",
                        "url",
                        "source_type",
                        "platform",
                        "updated_at",
                    ]
                )
        source_origin = source.origin if source is not None else TopicSourceOrigin.CURATED

        prepared_candidates.append(
            {
                **candidate,
                "display_url": _build_compact_source_url(source_url),
                "persisted_source_id": source.id if source is not None else None,
                "normalized_url": normalized.normalized_url,
                "candidate_origin": candidate_origin,
                "is_pinned": bool(source.is_pinned) if source is not None else False,
                "selected": source.is_active if source is not None else bool(candidate.get("default_selected")),
                "has_recent_article_count": candidate.get("has_recent_article_count")
                if "has_recent_article_count" in candidate
                else candidate.get("recent_article_count") is not None,
            }
        )

    topic.sources.filter(
        origin=TopicSourceOrigin.DISCOVERED,
        is_pinned=False,
    ).exclude(normalized_url__in=seen_discovered_normalized_urls).delete()

    return prepared_candidates


def _resolve_selected_source_candidates(
    selected_source_urls: list[str],
    candidate_records: list[dict],
) -> list[dict]:
    allowed_candidates: dict[str, dict] = {}
    for candidate in candidate_records:
        candidate_url = str(candidate.get("url") or "").strip()
        if not candidate_url:
            continue
        normalized_url = str(candidate.get("normalized_url") or classify_source_url(candidate_url).normalized_url).strip()
        if normalized_url:
            allowed_candidates[normalized_url] = candidate

    selected_candidates: list[dict] = []
    seen_normalized_urls: set[str] = set()
    for source_url in selected_source_urls:
        normalized_url = classify_source_url(source_url).normalized_url
        candidate = allowed_candidates.get(normalized_url)
        if candidate is None or normalized_url in seen_normalized_urls:
            continue
        seen_normalized_urls.add(normalized_url)
        selected_candidates.append(candidate)
    return selected_candidates


def _persist_selected_topic_sources(topic: Topic, selected_candidates: list[dict]) -> list[TopicSource]:
    selected_sources: list[TopicSource] = []
    for candidate in selected_candidates:
        source_url = str(candidate.get("url") or "").strip()
        if not source_url:
            continue

        normalized = classify_source_url(source_url)
        source = topic.sources.filter(normalized_url=normalized.normalized_url).first()
        if source is None:
            source = TopicSource.objects.create(
                topic=topic,
                name=str(candidate.get("title") or "").strip(),
                url=source_url,
                normalized_url=normalized.normalized_url,
                source_type=normalized.source_type,
                origin=_normalize_candidate_origin(candidate),
                platform=normalized.platform,
                validation_status=TopicSource.VALIDATION_PENDING,
                last_validation_error="",
                is_active=True,
            )
        else:
            source.name = str(candidate.get("title") or source.name or "").strip()
            source.url = source_url
            source.normalized_url = normalized.normalized_url
            source.source_type = normalized.source_type
            source.origin = _normalize_candidate_origin(candidate, source.origin)
            source.platform = normalized.platform
            source.is_active = True
            source.save(
                update_fields=[
                    "name",
                    "url",
                    "normalized_url",
                    "source_type",
                    "origin",
                    "platform",
                    "is_active",
                    "updated_at",
                ]
            )
        selected_sources.append(source)
    return selected_sources


def _normalize_candidate_origin(candidate: dict, fallback_origin: str = TopicSourceOrigin.CURATED) -> str:
    origin = str(candidate.get("candidate_origin") or "").strip().lower()
    if candidate.get("is_manual"):
        return TopicSourceOrigin.MANUAL
    if origin in {TopicSourceOrigin.MANUAL, TopicSourceOrigin.DISCOVERED, TopicSourceOrigin.CURATED}:
        return origin
    return fallback_origin


def _get_or_create_ui_topic(
    topic_name: str,
    source_urls: list[str] | None = None,
    source_mode: str = TopicSourceMode.HYBRID,
    topic_id: int | str | None = None,
) -> Topic:
    user = _get_or_create_ui_user()
    normalized_source_urls = _dedupe_source_urls(source_urls or [])
    primary_source_url = normalized_source_urls[0] if normalized_source_urls else ""

    if topic_id:
        topic = get_object_or_404(Topic, pk=topic_id, user=user)
        conflicting_topic_exists = (
            Topic.objects.filter(user=user, name=topic_name)
            .exclude(pk=topic.id)
            .exists()
        )
        if conflicting_topic_exists:
            raise ValidationError("A topic with this name already exists.")

        update_fields: list[str] = []
        if topic.name != topic_name:
            topic.name = topic_name
            update_fields.append("name")
        if primary_source_url and topic.source_url != primary_source_url:
            topic.source_url = primary_source_url
            update_fields.append("source_url")
        if source_mode and topic.source_mode != source_mode:
            topic.source_mode = source_mode
            update_fields.append("source_mode")
        if update_fields:
            update_fields.append("updated_at")
            topic.save(update_fields=update_fields)

        for source_url in normalized_source_urls:
            _ensure_manual_topic_source(topic, source_url)

        _ensure_topic_focus_seeded(topic)
        return topic

    topic, created = Topic.objects.get_or_create(
        user=user,
        name=topic_name,
        defaults={
            "source_url": primary_source_url or None,
            "source_mode": source_mode,
            "description": "",
            "keywords": [topic_name],
            "excluded_keywords": [],
            "focus_initialized": False,
            "is_active": True,
        },
    )

    if created and topic.display_order != 1:
        Topic.objects.filter(user=user).exclude(pk=topic.pk).update(display_order=F("display_order") + 1)
        topic.display_order = 1
        topic.save(update_fields=["display_order", "updated_at"])

    update_fields: list[str] = []
    if primary_source_url and topic.source_url != primary_source_url:
        topic.source_url = primary_source_url
        update_fields.append("source_url")
    if source_mode and topic.source_mode != source_mode:
        topic.source_mode = source_mode
        update_fields.append("source_mode")
    if update_fields:
        update_fields.append("updated_at")
        topic.save(update_fields=update_fields)

    for source_url in normalized_source_urls:
        _ensure_manual_topic_source(topic, source_url)

    _ensure_topic_focus_seeded(topic)
    return topic


def _dedupe_source_urls(source_urls: list[str]) -> list[str]:
    deduped_urls: list[str] = []
    seen_normalized_urls: set[str] = set()
    for raw_url in source_urls:
        source_url = str(raw_url or "").strip()
        if not source_url:
            continue
        normalized_url = classify_source_url(source_url).normalized_url
        if normalized_url in seen_normalized_urls:
            continue
        seen_normalized_urls.add(normalized_url)
        deduped_urls.append(source_url)
    return deduped_urls


def _parse_focus_terms(data) -> list[str]:
    raw_values: list[str] = []
    raw_focus_value = data.get("focus_terms")
    if raw_focus_value is not None:
        raw_values.extend(str(raw_focus_value).split("\n"))
    raw_values.extend(data.getlist("focus_terms[]"))
    return clean_focus_terms(raw_values)


def _ensure_topic_focus_seeded(topic: Topic) -> None:
    current_terms = topic.keywords if isinstance(topic.keywords, list) else []
    if not should_seed_focus_terms(topic.name, current_terms, focus_initialized=bool(topic.focus_initialized)):
        return

    suggested_terms = generate_focus_suggestions(topic.name, existing_terms=current_terms)
    if not suggested_terms:
        return

    topic.keywords = suggested_terms
    topic.focus_initialized = True
    topic.save(update_fields=["keywords", "focus_initialized", "updated_at"])


def _ensure_manual_topic_source(topic: Topic, source_url: str, source_name: str = "") -> TopicSource:
    normalized = classify_source_url(source_url)
    source = topic.sources.filter(normalized_url=normalized.normalized_url).first()
    cleaned_source_name = str(source_name or "").strip()
    if source is None:
        source = TopicSource.objects.create(
            topic=topic,
            name=cleaned_source_name,
            url=normalized.original_url,
            normalized_url=normalized.normalized_url,
            source_type=normalized.source_type,
            origin=TopicSourceOrigin.MANUAL,
            platform=normalized.platform,
            validation_status=TopicSource.VALIDATION_VALID,
            last_validation_error="",
            is_active=True,
        )
    else:
        source.url = normalized.original_url
        if cleaned_source_name:
            source.name = cleaned_source_name
        source.source_type = normalized.source_type
        source.origin = TopicSourceOrigin.MANUAL
        source.platform = normalized.platform
        source.validation_status = TopicSource.VALIDATION_VALID
        source.last_validation_error = ""
        if not source.is_active:
            source.is_active = True
        source.save(
            update_fields=[
                "url",
                "name",
                "source_type",
                "origin",
                "platform",
                "validation_status",
                "last_validation_error",
                "is_active",
                "updated_at",
            ]
        )
    return source


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


def _create_ui_digest_run(topic: Topic, source: str, selected_source_urls: list[str] | None = None) -> DigestRun:
    return DigestRun.objects.create(
        topic=topic,
        input_snapshot={
            "mode": "manual",
            "source": source,
            "topic_name": topic.name,
            "source_url": topic.source_url or "",
            "selected_source_urls": list(selected_source_urls or topic.sources.filter(is_active=True).values_list("url", flat=True)),
        },
    )


def _start_topic_run(run: DigestRun, topic: Topic, default_source: str) -> None:
    active_sources = list(topic.sources.filter(is_active=True).order_by("id"))
    if active_sources:
        raw_items = []
        selected_source_urls: list[str] = []
        valid_source_count = 0

        for source in active_sources:
            fetch_url = source.normalized_url or source.url
            source_items = fetch_rss_articles(fetch_url)
            selected_source_urls.append(source.url)

            if source_items:
                valid_source_count += 1
                raw_items.extend(source_items)
                source.validation_status = TopicSource.VALIDATION_VALID
                source.last_validation_error = ""
            else:
                source.validation_status = TopicSource.VALIDATION_INVALID
                source.last_validation_error = "Source returned no valid items."
            source.save(update_fields=["validation_status", "last_validation_error", "updated_at"])

        if not raw_items:
            _mark_run_failed_for_empty_selected_sources(run, selected_source_urls)
            return

        run.input_snapshot = {
            **run.input_snapshot,
            "source": "selected_sources",
            "source_url": "",
            "selected_source_urls": selected_source_urls,
            "selected_source_count": len(selected_source_urls),
            "validated_source_count": valid_source_count,
            "raw_items_count": len(raw_items),
        }
        run.save(update_fields=["input_snapshot", "updated_at"])
        run_digest_pipeline(run.id, raw_items=raw_items)
        return

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


def _start_selected_source_run(run: DigestRun, topic: Topic, selected_candidates: list[dict], default_source: str) -> None:
    if not selected_candidates:
        _mark_run_failed_for_empty_selected_sources(run, [])
        return

    raw_items: list[dict] = []
    selected_source_urls: list[str] = []
    valid_source_count = 0

    for candidate in selected_candidates:
        source_url = str(candidate.get("url") or "").strip()
        if not source_url:
            continue

        normalized_url = str(candidate.get("normalized_url") or classify_source_url(source_url).normalized_url).strip()
        source_items = fetch_rss_articles(normalized_url)
        selected_source_urls.append(source_url)

        persisted_source = topic.sources.filter(normalized_url=normalized_url).first()
        if source_items:
            valid_source_count += 1
            raw_items.extend(source_items)
            if persisted_source is not None:
                persisted_source.validation_status = TopicSource.VALIDATION_VALID
                persisted_source.last_validation_error = ""
                persisted_source.save(update_fields=["validation_status", "last_validation_error", "updated_at"])
        elif persisted_source is not None:
            persisted_source.validation_status = TopicSource.VALIDATION_INVALID
            persisted_source.last_validation_error = "Source returned no valid items."
            persisted_source.save(update_fields=["validation_status", "last_validation_error", "updated_at"])

    if not raw_items:
        _mark_run_failed_for_empty_selected_sources(run, selected_source_urls)
        return

    run.input_snapshot = {
        **run.input_snapshot,
        "source": "selected_sources",
        "source_url": "",
        "selected_source_urls": selected_source_urls,
        "selected_source_count": len(selected_source_urls),
        "validated_source_count": valid_source_count,
        "raw_items_count": len(raw_items),
    }
    run.save(update_fields=["input_snapshot", "updated_at"])
    run_digest_pipeline(run.id, raw_items=raw_items)


def _mark_run_failed_for_empty_selected_sources(run: DigestRun, source_urls: list[str]) -> None:
    joined_sources = ", ".join(source_urls) if source_urls else "selected sources"
    run.status = DigestRun.STATUS_FAILED
    run.error_message = f"Selected sources returned no valid items: {joined_sources}"
    run.result_message = result_messages.SOURCE_NO_USABLE_ARTICLES
    run.finished_at = timezone.now()
    run.input_snapshot = {
        **run.input_snapshot,
        "source": "selected_sources",
        "source_url": "",
        "selected_source_urls": source_urls,
        "selected_source_count": len(source_urls),
        "raw_items_count": 0,
    }
    run.save(
        update_fields=[
            "status",
            "error_message",
            "result_message",
            "finished_at",
            "input_snapshot",
            "updated_at",
        ]
    )


def _mark_run_failed_for_empty_rss(run: DigestRun, source_url: str) -> None:
    run.status = DigestRun.STATUS_FAILED
    run.error_message = f"RSS source returned no valid items: {source_url}"
    run.result_message = result_messages.SOURCE_NO_USABLE_ARTICLES
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
            "result_message",
            "finished_at",
            "input_snapshot",
            "updated_at",
        ]
    )

