import json
import base64
import logging
import re
from collections import Counter
from datetime import timedelta
from urllib.parse import urlencode, urlparse

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.paginator import Paginator
from django.core.validators import URLValidator
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.db.models import F, Q
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.digests import result_messages
from apps.digests.models import DigestRun, SourceDiscoveryHistory, SourceDiscoveryRun, UsedArticle
from apps.topics.focus import FOCUS_VALIDATION_MESSAGE, clean_focus_terms, validate_new_focus_terms
from apps.topics.focus_suggestions import generate_focus_suggestions, should_seed_focus_terms
from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import (
    build_research_review_context,
    build_discovery_cycle_overall_diagnosis,
    build_discovery_cycle_payload,
    build_discovery_cycle_round_diagnosis,
    build_source_quality_feedback,
    build_topic_source_payloads_from_review_items,
    CuratedSourceSeed,
    classify_discovery_cycle_round_reason,
    TopicSourceDiscoveryRequest,
    filter_new_source_candidates,
    format_discovery_cycle_decision_label,
    format_discovery_cycle_diagnosis_label,
    format_discovery_cycle_next_action_label,
    get_demo_articles_for_topic,
    is_new_research_source,
    resolve_configured_search_provider,
    run_source_research,
    split_topic_sources,
    resolve_source_candidates,
)
from services.sources.discovery_history import (
    build_topic_history_by_normalized_url,
    build_topic_known_url_set,
    finalize_source_discovery_run,
    mark_removed_discovered_sources_as_seen,
    record_source_discovery_history,
    record_source_discovery_run_started,
    sync_topic_discovered_sources_into_history,
    update_history_for_kept_source,
    update_history_for_removed_source,
    update_history_for_unpinned_source,
)
from services.sources.discovery_repair import (
    _build_discovery_repair_plan,
    _build_next_round_repair_override,
    _build_round_repair_plan,
    _extract_repair_query_rows,
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
VISIBLE_NEW_SOURCE_LIMIT = 12
DISCOVERY_CONTEXT_PARAM = "discovery_context"
SHOW_ALL_NEW_SUGGESTIONS_PARAM = "show_all_suggestions"
DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS = 6
DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS = 3


def _get_user_facing_source_mode_label(mode: str) -> str:
    if mode == TopicSourceMode.HYBRID:
        return "my sources & research"
    if mode == TopicSourceMode.CURATED_ONLY:
        return "my sources only"
    if mode == TopicSourceMode.DISCOVERY_ONLY:
        return "research only"
    try:
        return TopicSourceMode(mode).label
    except ValueError:
        return mode


@require_GET
def topic_list_view(request: HttpRequest) -> HttpResponse:
    return render(request, "digestflow/topic_list.html", _build_topic_list_context())


@require_GET
def topic_workspace_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    return _render_topic_source_review(request, topic)


@require_GET
def topic_research_history_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    sync_topic_discovered_sources_into_history(topic)
    current_research_state = _build_current_research_state(topic)
    query_performance = _build_query_performance_section(topic)
    source_quality_feedback = _build_source_quality_feedback_section(topic)
    search_surface_memory = _build_search_surface_memory_section(topic)
    history_runs = _build_research_history_run_entries(topic)
    seen_source_history = _build_seen_source_history_section(
        topic,
        status_filter=str(request.GET.get("status") or "").strip(),
        search_query=str(request.GET.get("q") or "").strip(),
        page_number=str(request.GET.get("page") or "").strip() or "1",
    )
    full_history_copy_report = _build_full_research_history_copy_report(
        topic=topic,
        current_research_state=current_research_state,
        query_performance_entries=query_performance["entries"],
        source_quality_feedback=source_quality_feedback,
        search_surface_memory=search_surface_memory,
        history_runs=history_runs,
        seen_source_history=seen_source_history["entries"],
    )
    return render(
        request,
        "digestflow/topic_research_history.html",
        {
            "topic": topic,
            "current_research_state": current_research_state,
            "query_performance_entries": query_performance["entries"],
            "source_quality_feedback": source_quality_feedback,
            "search_surface_memory": search_surface_memory,
            "history_runs": history_runs,
            "seen_source_history": seen_source_history["entries"],
            "seen_source_history_filters": seen_source_history["filters"],
            "seen_source_history_search_query": seen_source_history["search_query"],
            "seen_source_history_active_filter": seen_source_history["active_filter"],
            "seen_source_history_pagination": seen_source_history["pagination"],
            "full_history_copy_report": full_history_copy_report,
            "full_history_copy_report_b64": base64.b64encode(full_history_copy_report.encode("utf-8")).decode("ascii"),
        },
    )


@require_POST
def discover_sources_view(request: HttpRequest) -> HttpResponse:
    form = TopicInputForm(request.POST)
    if not form.is_valid():
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = _get_topic_form_error(form)
        return render(request, "digestflow/topic_list.html", context, status=400)

    topic_id = str(request.POST.get("topic_id") or "").strip() or None
    topic_name = form.cleaned_data["topic_name"]
    source_url = str(form.cleaned_data.get("source_url") or "").strip()
    source_mode = form.cleaned_data.get("source_mode") or TopicSourceMode.HYBRID
    try:
        topic = _get_or_create_ui_topic(
            topic_name,
            source_urls=[source_url] if source_url else [],
            source_mode=source_mode,
            topic_id=topic_id,
        )
    except ValidationError as exc:
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = str(exc)
        return render(request, "digestflow/topic_list.html", context, status=400)
    topic.manual_source_inputs = [source_url] if source_url else []
    discovery_summary = None
    discovered_source_candidates = None
    discovery_requested = _should_run_research_discovery(request, topic_id=topic_id)
    if discovery_requested:
        discovered_source_candidates, discovery_summary = _discover_and_prepare_candidates_with_summary(topic)
    return render(
        request,
        "digestflow/topic_list.html",
        _build_topic_list_context(
            form=form,
            discovered_topic=topic,
            discovered_source_candidates=discovered_source_candidates,
            discovery_summary=discovery_summary,
            discovery_context_active=discovery_requested,
            show_all_new_suggestions=_request_wants_all_new_suggestions(request),
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
    if topic.sources.exists():
        run_eligibility = _build_run_eligibility(topic)
        if not run_eligibility["is_eligible"]:
            return redirect("topic-workspace", topic_id=topic.id)
    run = _create_ui_digest_run(topic, source="web_ui")

    _start_topic_run(run, topic, default_source="web_ui")
    return redirect("run-detail", run_id=run.id)


@require_POST
def delete_used_article_view(request: HttpRequest, run_id: int, used_article_id: int) -> HttpResponse:
    run = get_object_or_404(
        DigestRun.objects.select_related("topic__user"),
        pk=run_id,
    )
    used_article = get_object_or_404(
        UsedArticle,
        pk=used_article_id,
        topic=run.topic,
        user=run.topic.user,
    )
    used_article.delete()
    return redirect(f"{reverse('run-detail', kwargs={'run_id': run.id})}#used-article-history")


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
    return redirect(
        _build_topic_workspace_url(
            topic.id,
            discovery_context_active=_request_has_discovery_context(request),
            show_all_new_suggestions=_request_wants_all_new_suggestions(request),
        )
    )


@require_POST
def pin_topic_source_view(request: HttpRequest, topic_id: int, source_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    source = get_object_or_404(TopicSource, pk=source_id, topic=topic)
    if source.origin == TopicSourceOrigin.DISCOVERED and not source.is_pinned:
        source.is_pinned = True
        source.is_active = True
        source.save(update_fields=["is_pinned", "is_active", "updated_at"])
        update_history_for_kept_source(source)
    return redirect(
        _build_topic_workspace_url(
            topic.id,
            discovery_context_active=_request_has_discovery_context(request),
            show_all_new_suggestions=_request_wants_all_new_suggestions(request),
        )
    )


@require_POST
def unpin_topic_source_view(request: HttpRequest, topic_id: int, source_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    source = get_object_or_404(TopicSource, pk=source_id, topic=topic)
    if source.origin == TopicSourceOrigin.DISCOVERED and source.is_pinned:
        source.is_pinned = False
        source.is_active = False
        source.save(update_fields=["is_pinned", "is_active", "updated_at"])
        update_history_for_unpinned_source(source)
    return redirect(
        _build_topic_workspace_url(
            topic.id,
            discovery_context_active=_request_has_discovery_context(request),
            show_all_new_suggestions=_request_wants_all_new_suggestions(request),
        )
    )


@require_POST
def remove_topic_source_view(request: HttpRequest, topic_id: int, source_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    source = get_object_or_404(TopicSource, pk=source_id, topic=topic)
    if source.origin == TopicSourceOrigin.DISCOVERED:
        source.is_pinned = False
        source.is_active = False
        source.save(update_fields=["is_pinned", "is_active", "updated_at"])
        update_history_for_unpinned_source(source)
    else:
        update_history_for_removed_source(source)
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
    used_articles = _build_used_article_history(run.topic)
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
        "used_article_count": len(used_articles),
        "used_articles": used_articles,
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
            (
                "Used article history for topic",
                _display_metric_value(ranking_stage.get("used_article_count_for_topic")),
            ),
            (
                "Excluded as already used",
                _display_metric_value(ranking_stage.get("articles_excluded_as_used")),
            ),
            (
                "Remaining after used filter",
                _display_metric_value(ranking_stage.get("articles_remaining_after_used_filter")),
            ),
            ("Selected for prompt", _display_metric_value(ranking_stage.get("selected_for_prompt"))),
        ],
        "decision_message": _build_ranking_decision_message(ranking_stage, run_status),
        "top_rejected_article": top_rejected_articles[0] if top_rejected_articles else None,
        "excluded_used_articles": ranking_stage.get("excluded_used_articles", []),
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
    discovery_summary: dict | None = None,
    discovery_context_active: bool = False,
    show_all_new_suggestions: bool = False,
    focus_feedback: dict | None = None,
    focus_input_value: str = "",
) -> dict:
    user = _get_or_create_ui_user()
    all_candidate_records = discovered_source_candidates
    if all_candidate_records is None:
        all_candidate_records = _build_persisted_new_source_candidates(discovered_topic)
    total_new_source_candidates = _build_visible_new_source_candidates(all_candidate_records)
    if show_all_new_suggestions:
        visible_new_source_candidates = total_new_source_candidates
    else:
        visible_new_source_candidates = total_new_source_candidates[:VISIBLE_NEW_SOURCE_LIMIT]
    rendered_discovery_summary = _finalize_discovery_summary(
        discovery_summary,
        total_visible_candidates=len(visible_new_source_candidates),
        total_new_source_candidates=len(total_new_source_candidates),
    )
    if (
        rendered_discovery_summary is None
        and discovery_context_active
        and discovered_topic is not None
        and discovered_topic.uses_source_discovery
    ):
        rendered_discovery_summary = _build_discovery_results_summary(
            total_visible_candidates=len(visible_new_source_candidates),
            total_new_source_candidates=len(total_new_source_candidates),
        )
    topics = list(
        Topic.objects.filter(user=user)
        .order_by("display_order", "name")
        .prefetch_related("sources")
    )
    for topic in topics:
        topic.source_count = sum(1 for source in topic.sources.all() if source.origin != TopicSourceOrigin.DISCOVERED)
        topic.research_source_count = sum(1 for source in topic.sources.all() if source.origin == TopicSourceOrigin.DISCOVERED)
        topic.active_source_count = sum(
            1 for source in topic.sources.all() if source.is_active and source.origin != TopicSourceOrigin.DISCOVERED
        )
        topic.run_eligibility = _build_run_eligibility(topic)
        topic.legacy_source_display = _build_legacy_source_display(topic)
    recent_runs = DigestRun.objects.filter(topic__user=user).select_related("topic").order_by("-created_at")[:10]
    for run in recent_runs:
        run.display_time = _format_recent_run_time(run.created_at)
    run_eligibility = _build_run_eligibility(discovered_topic)
    research_provider_state = _build_research_provider_state(discovered_topic)
    hidden_new_source_candidate_count = max(0, len(total_new_source_candidates) - len(visible_new_source_candidates))
    has_research_discovery_results = _topic_has_research_discovery_results(discovered_topic)
    return {
        "topics": topics,
        "recent_runs": recent_runs,
        "topic_form": form or TopicInputForm(),
        "discovered_topic": discovered_topic,
        "focus_terms": _build_topic_focus_terms(discovered_topic),
        "focus_feedback": focus_feedback,
        "focus_input_value": focus_input_value,
        "discovered_source_candidates": visible_new_source_candidates,
        "new_suggestions_visible_limit": VISIBLE_NEW_SOURCE_LIMIT,
        "total_new_source_candidate_count": len(total_new_source_candidates),
        "new_suggestions_total_count": len(total_new_source_candidates),
        "new_suggestions_hidden_count": hidden_new_source_candidate_count,
        "new_suggestions_is_truncated": hidden_new_source_candidate_count > 0,
        "show_all_new_suggestions": show_all_new_suggestions,
        "discovery_context_active": discovery_context_active,
        "source_review_summary": _build_source_review_summary(discovered_topic, all_candidate_records),
        "discovery_summary": rendered_discovery_summary,
        "topic_source_inventory": _build_topic_source_inventory(discovered_topic),
        "pinned_research_source_inventory": _build_pinned_research_source_inventory(discovered_topic),
        "active_saved_source_urls": _build_active_saved_source_urls(discovered_topic),
        "active_selected_source_urls": _build_active_selected_source_urls(discovered_topic),
        "selected_source_count": _build_selected_source_count(discovered_topic),
        "run_eligibility": run_eligibility,
        "research_provider_notice": research_provider_state["notice"],
        "research_provider_blocked": research_provider_state["blocked"],
        "find_sources_disabled_hint": research_provider_state["button_hint"],
        "has_research_discovery_results": has_research_discovery_results,
        "source_discovery_button_label": (
            "Find new sources" if has_research_discovery_results else "Find sources"
        ),
        "legacy_topic_source": _build_legacy_source_display(discovered_topic),
        "source_add_feedback": None,
        "can_find_research_sources": _can_find_research_sources(
            discovered_topic,
            provider_blocked=research_provider_state["blocked"],
        ),
        "show_all_new_suggestions_url": _build_topic_workspace_url(
            discovered_topic.id,
            discovery_context_active=True,
            show_all_new_suggestions=True,
        ) if discovered_topic is not None else "",
        "research_history_url": (
            reverse("topic-research-history", args=[discovered_topic.id])
            if discovered_topic is not None and discovered_topic.uses_source_discovery
            else ""
        ),
    }


def _build_research_provider_state(topic: Topic | None) -> dict:
    empty_state = {
        "notice": None,
        "blocked": False,
        "button_hint": "",
    }
    if topic is None or not topic.uses_source_discovery:
        return empty_state

    diagnostics = resolve_configured_search_provider(topic).diagnostics
    status = str(diagnostics.get("search_provider_status") or "").strip().lower()
    if not status or status == "ready":
        return empty_state

    button_hint = ""
    if status == "disabled":
        button_hint = "Research unavailable"
    elif status == "missing_config":
        button_hint = "Provider setup required"
    elif status == "not_implemented":
        button_hint = "Search adapter not connected"

    return {
        "notice": _build_research_provider_notice_from_diagnostics(diagnostics),
        "blocked": True,
        "button_hint": button_hint,
    }


def _build_research_provider_notice(topic: Topic | None) -> dict | None:
    if topic is None or not topic.uses_source_discovery:
        return None

    diagnostics = resolve_configured_search_provider(topic).diagnostics
    return _build_research_provider_notice_from_diagnostics(diagnostics)


def _build_research_provider_notice_from_diagnostics(diagnostics: dict) -> dict | None:
    status = str(diagnostics.get("search_provider_status") or "").strip().lower()
    if not status or status == "ready":
        return None

    title = ""
    body = ""
    if status == "disabled":
        title = "Research is currently disabled"
        body = "DigestFlow can still use your sources, but automatic research is turned off."
    elif status == "missing_config":
        title = "Research provider needs configuration"
        body = "Automatic research is enabled, but the selected provider is missing required settings."
    elif status == "not_implemented":
        title = "Research provider is not connected yet"
        body = "The selected provider is configured, but the real search adapter has not been implemented yet."
    else:
        return None

    missing_settings = tuple(str(value).strip() for value in diagnostics.get("search_provider_missing_settings", ()) if str(value).strip())
    return {
        "title": title,
        "body": body,
        "status": status,
        "provider_name": str(diagnostics.get("search_provider_name") or "").strip(),
        "missing_settings": missing_settings,
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
        discovery_context_active=_request_has_discovery_context(request),
        show_all_new_suggestions=_request_wants_all_new_suggestions(request),
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


def _should_run_research_discovery(request: HttpRequest, *, topic_id: str | None) -> bool:
    if not str(request.POST.get("run_research") or "").strip():
        return False
    if not topic_id:
        return True
    topic = Topic.objects.filter(pk=topic_id).first()
    return _can_find_research_sources(topic)


def _discover_and_prepare_candidates(topic: Topic) -> list[dict]:
    candidate_records, _ = _discover_and_prepare_candidates_with_summary(topic)
    return candidate_records


def _discover_and_prepare_candidates_with_summary(topic: Topic) -> tuple[list[dict], dict | None]:
    provider_resolution = resolve_configured_search_provider(topic)
    provider_diagnostics = provider_resolution.diagnostics
    provider_status = str(provider_diagnostics.get("search_provider_status") or "").strip().lower()
    provider_name = str(provider_diagnostics.get("search_provider_name") or "").strip().lower()
    provider_enabled = bool(provider_diagnostics.get("search_provider_enabled"))

    if provider_status == "ready" and provider_name == "serpapi":
        return _run_provider_discovery_cycle(
            topic=topic,
            provider_name=provider_name,
            provider_diagnostics=dict(provider_diagnostics),
        )

    if provider_status != "ready" and (provider_enabled or provider_name not in {"", "unconfigured"}):
        blocked_run = finalize_source_discovery_run(
            record_source_discovery_run_started(
                topic=topic,
                provider_name=provider_name,
                diagnostics=dict(provider_diagnostics),
            ),
            status=SourceDiscoveryRun.STATUS_BLOCKED,
            diagnostics={
                **dict(provider_diagnostics),
                "discovery_cycle": _build_discovery_cycle_payload(
                    topic=topic,
                    cycle_id=f"provider-unavailable-{topic.id}-{timezone.now().strftime('%Y%m%d%H%M%S%f')}",
                    target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
                    max_immediate_rounds=DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS,
                    round_count=0,
                    accumulated_visible_suggestions=0,
                    decision="provider_unavailable",
                    rounds=[],
                ),
            },
        )
        return _build_persisted_new_source_candidates(topic), {
            "title": "Source search is temporarily unavailable",
            "body": (
                "DigestFlow could not connect to the search provider. Existing suggestions were kept."
                if _count_existing_new_suggestions(topic) > 0
                else "DigestFlow could not connect to the search provider. Please try again later."
            ),
            "provider_name": provider_name or str(provider_diagnostics.get("search_provider_name") or "").strip(),
            "execution_status": "provider_unavailable",
            "provider_result_count": 0,
            "candidate_input_count": 0,
            "query_count": 0,
            "existing_new_suggestion_count": _count_existing_new_suggestions(topic),
            "discovery_cycle": blocked_run.diagnostics.get("discovery_cycle"),
        }

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
    candidate_records = _upsert_and_build_source_candidates(topic, candidate_records)
    return candidate_records, None


def _build_provider_backed_candidate_records(source_research_result) -> list[dict]:
    review_context = build_research_review_context(source_research_result)
    payloads = build_topic_source_payloads_from_review_items(review_context.persistable_items)
    payloads_by_url = {
        str(payload.get("url") or "").strip(): payload
        for payload in payloads
        if str(payload.get("url") or "").strip()
    }

    candidate_records: list[dict] = []
    for item in review_context.persistable_items:
        payload = payloads_by_url.get(str(item.url or "").strip())
        if payload is None:
            continue
        candidate_records.append(
            {
                "url": payload["url"],
                "title": payload["title"],
                "normalized_url": item.normalized_url,
                "query": str(item.diagnostics.get("query") or "").strip(),
                "source_type": payload.get("source_type") or item.source_type,
                "candidate_origin": payload.get("origin") or TopicSourceOrigin.DISCOVERED,
                "default_selected": item.default_selected,
                "description": str(item.diagnostics.get("origin_reason") or "").strip(),
                "provider_name": review_context.diagnostics.get("provider_name", ""),
                "has_recent_article_count": False,
                "recent_article_count": None,
            }
        )
    return candidate_records


def _run_provider_discovery_cycle(
    *,
    topic: Topic,
    provider_name: str,
    provider_diagnostics: dict,
) -> tuple[list[dict], dict]:
    cycle_id = f"cycle-{topic.id}-{timezone.now().strftime('%Y%m%d%H%M%S%f')}"
    accumulated_new_candidates: list[dict] = []
    accumulated_seen_normalized_urls: set[str] = set()
    round_results: list[dict] = []
    round_count = 0
    decision = "partial_target_not_reached"
    next_round_query_plan = None
    next_round_repair_usage = None

    for round_index in range(1, DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS + 1):
        round_repair_usage = next_round_repair_usage
        round_result = _run_provider_discovery_round(
            topic=topic,
            provider_name=provider_name,
            provider_diagnostics=provider_diagnostics,
            prune_missing_discovered=False,
            cycle_id=cycle_id,
            round_index=round_index,
            query_plan_override=next_round_query_plan,
            repair_usage=round_repair_usage,
        )
        next_round_query_plan = None
        next_round_repair_usage = None
        if round_repair_usage and not round_result.get("used_repair_plan"):
            round_result["used_repair_plan"] = True
            round_result["repair_plan_usage"] = dict(round_repair_usage)
        round_count = round_index
        for candidate in round_result["new_visible_candidates"]:
            normalized_url = str(candidate.get("normalized_url") or "").strip()
            if not normalized_url or normalized_url in accumulated_seen_normalized_urls:
                continue
            accumulated_seen_normalized_urls.add(normalized_url)
            accumulated_new_candidates.append(candidate)

        accumulated_visible_suggestions = len(accumulated_seen_normalized_urls)
        round_results.append(
            _build_discovery_cycle_round_summary(
                topic=topic,
                round_result=round_result,
                round_index=round_index,
                accumulated_visible_suggestions=accumulated_visible_suggestions,
            )
        )

        if accumulated_visible_suggestions >= DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS:
            decision = "target_reached"
            break
        if round_result["provider_unavailable"]:
            decision = "provider_unavailable"
            break
        if round_index >= DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS:
            decision = "max_rounds_reached"
            break
        next_round_query_plan, next_round_repair_usage, continuation_decision = _build_next_round_repair_override(
            topic=topic,
            round_summary=round_results[-1],
            prior_rounds=round_results,
            query_limit=int(getattr(round_result.get("discovery_run"), "query_count", 0) or 0),
        )
        if next_round_query_plan is None:
            decision = continuation_decision or "partial_target_not_reached_no_usable_repair_queries"
            break

    final_candidate_records = _finalize_discovery_cycle_candidate_records(
        topic=topic,
        accumulated_new_candidates=accumulated_new_candidates,
        prune_missing_discovered=(decision != "provider_unavailable"),
    )
    accumulated_visible_suggestions = len(accumulated_seen_normalized_urls)
    cycle_payload = _build_discovery_cycle_payload(
        topic=topic,
        cycle_id=cycle_id,
        target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
        max_immediate_rounds=DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS,
        round_count=round_count,
        accumulated_visible_suggestions=accumulated_visible_suggestions,
        decision=decision,
        rounds=round_results,
    )
    _attach_discovery_cycle_to_runs(round_results, cycle_payload)
    return final_candidate_records, _build_discovery_cycle_summary(
        topic=topic,
        candidate_records=final_candidate_records,
        accumulated_visible_suggestions=accumulated_visible_suggestions,
        round_count=round_count,
        decision=decision,
        round_results=round_results,
    )


def _run_provider_discovery_round(
    *,
    topic: Topic,
    provider_name: str,
    provider_diagnostics: dict,
    prune_missing_discovered: bool,
    cycle_id: str,
    round_index: int,
    query_plan_override=None,
    repair_usage: dict | None = None,
) -> dict:
    discovery_run = record_source_discovery_run_started(
        topic=topic,
        provider_name=provider_name,
        diagnostics=dict(provider_diagnostics),
    )
    known_normalized_urls = build_topic_known_url_set(topic)
    source_research_result = run_source_research(topic, query_plan_override=query_plan_override)
    provider_error_count = int(source_research_result.diagnostics.get("provider_error_count") or 0)
    accepted_count = int(source_research_result.diagnostics.get("accepted_candidate_count") or 0)
    rejected_count = int(source_research_result.diagnostics.get("rejected_candidate_count") or 0)
    raw_result_count = int(source_research_result.diagnostics.get("raw_result_count") or 0)
    candidate_input_count = int(source_research_result.diagnostics.get("candidate_input_count") or 0)
    already_known_count = _count_known_provider_results(
        source_research_result=source_research_result,
        known_normalized_urls=known_normalized_urls,
    )

    new_visible_candidates = _build_provider_backed_candidate_records(source_research_result)
    new_visible_candidates = filter_new_source_candidates(new_visible_candidates, topic.sources.all())
    new_visible_candidates = _filter_previously_handled_provider_candidates(topic, new_visible_candidates)
    has_new_visible_suggestions = _has_new_visible_suggestions(
        candidate_records=new_visible_candidates,
        known_normalized_urls=known_normalized_urls,
    )
    if provider_error_count > 0 and not has_new_visible_suggestions:
        discovery_diagnostics = _build_source_discovery_run_diagnostics(
            source_research_result=source_research_result,
            known_normalized_urls=known_normalized_urls,
            shown_candidates=[],
        )
        discovery_diagnostics["discovery_cycle"] = _build_discovery_cycle_round_stub(
            cycle_id=cycle_id,
            round_index=round_index,
        )
        if repair_usage:
            discovery_diagnostics["used_repair_plan"] = True
            discovery_diagnostics["repair_plan_usage"] = dict(repair_usage)
        run_status = (
            SourceDiscoveryRun.STATUS_PARTIAL_FAILED
            if raw_result_count > 0
            else SourceDiscoveryRun.STATUS_FAILED
        )
        finalized_run = finalize_source_discovery_run(
            discovery_run,
            status=run_status,
            diagnostics=discovery_diagnostics,
            known_url_count=already_known_count,
            accepted_count=accepted_count,
            rejected_count=rejected_count,
            new_suggestions_count=0,
            already_known_count=already_known_count,
        )
        record_source_discovery_history(
            topic=topic,
            discovery_run=finalized_run,
            source_research_result=source_research_result,
            shown_candidates=[],
            known_normalized_urls=known_normalized_urls,
        )
        return {
            "display_candidate_records": _build_persisted_new_source_candidates(topic),
            "new_visible_candidates": [],
            "source_research_result": source_research_result,
            "discovery_run": finalized_run,
            "execution_status": "failed",
            "provider_unavailable": raw_result_count == 0 and candidate_input_count == 0,
            "provider_error_count": provider_error_count,
            "accepted_count": accepted_count,
            "rejected_count": rejected_count,
            "known_or_duplicate_count": already_known_count,
            "quality_rejected_count": int(discovery_diagnostics["source_quality_feedback"].get("quality_rejected_count") or 0),
            "returned_count": raw_result_count,
            "used_repair_plan": bool(repair_usage and repair_usage.get("used_repair_plan")),
            "repair_plan_usage": dict(repair_usage or {}),
            "reason_summary": classify_discovery_cycle_round_reason(
            provider_error_count=provider_error_count,
            raw_result_count=raw_result_count,
            visible_new_suggestions_count=0,
            quality_rejected_count=int(discovery_diagnostics["source_quality_feedback"].get("quality_rejected_count") or 0),
            known_or_duplicate_count=already_known_count,
            target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
        ),
        }

    display_candidate_records = _upsert_and_build_source_candidates(
        topic,
        new_visible_candidates,
        prune_missing_discovered=prune_missing_discovered and has_new_visible_suggestions,
    )
    shown_candidate_records = _select_round_shown_candidate_records(
        display_candidate_records=display_candidate_records,
        new_visible_candidates=new_visible_candidates,
    )
    if not has_new_visible_suggestions:
        display_candidate_records = _build_persisted_new_source_candidates(topic)
        shown_candidate_records = []
    run_status = SourceDiscoveryRun.STATUS_COMPLETED
    execution_status = "completed"
    if provider_error_count > 0:
        run_status = (
            SourceDiscoveryRun.STATUS_PARTIAL_FAILED
            if raw_result_count > 0
            else SourceDiscoveryRun.STATUS_FAILED
        )
        execution_status = "failed"
    discovery_diagnostics = _build_source_discovery_run_diagnostics(
        source_research_result=source_research_result,
        known_normalized_urls=known_normalized_urls,
        shown_candidates=shown_candidate_records,
    )
    discovery_diagnostics["discovery_cycle"] = _build_discovery_cycle_round_stub(
        cycle_id=cycle_id,
        round_index=round_index,
    )
    if repair_usage:
        discovery_diagnostics["used_repair_plan"] = True
        discovery_diagnostics["repair_plan_usage"] = dict(repair_usage)
    finalized_run = finalize_source_discovery_run(
        discovery_run,
        status=run_status,
        diagnostics=discovery_diagnostics,
        known_url_count=already_known_count,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        new_suggestions_count=len(shown_candidate_records),
        already_known_count=already_known_count,
    )
    record_source_discovery_history(
        topic=topic,
        discovery_run=finalized_run,
        source_research_result=source_research_result,
        shown_candidates=shown_candidate_records,
        known_normalized_urls=known_normalized_urls,
    )
    return {
        "display_candidate_records": display_candidate_records,
        "new_visible_candidates": shown_candidate_records,
        "source_research_result": source_research_result,
        "discovery_run": finalized_run,
        "execution_status": execution_status,
        "provider_unavailable": False,
        "provider_error_count": provider_error_count,
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "known_or_duplicate_count": already_known_count,
        "quality_rejected_count": int(discovery_diagnostics["source_quality_feedback"].get("quality_rejected_count") or 0),
        "returned_count": raw_result_count,
        "used_repair_plan": bool(repair_usage and repair_usage.get("used_repair_plan")),
        "repair_plan_usage": dict(repair_usage or {}),
        "reason_summary": classify_discovery_cycle_round_reason(
            provider_error_count=provider_error_count,
            raw_result_count=raw_result_count,
            visible_new_suggestions_count=len(new_visible_candidates),
            quality_rejected_count=int(discovery_diagnostics["source_quality_feedback"].get("quality_rejected_count") or 0),
            known_or_duplicate_count=already_known_count,
            target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
        ),
    }


def _select_round_shown_candidate_records(
    *,
    display_candidate_records: list[dict],
    new_visible_candidates: list[dict],
) -> list[dict]:
    visible_normalized_urls = {
        str(candidate.get("normalized_url") or "").strip()
        for candidate in new_visible_candidates
        if str(candidate.get("normalized_url") or "").strip()
    }
    return [
        candidate
        for candidate in display_candidate_records
        if str(candidate.get("normalized_url") or "").strip() in visible_normalized_urls
    ]


def _finalize_discovery_cycle_candidate_records(
    *,
    topic: Topic,
    accumulated_new_candidates: list[dict],
    prune_missing_discovered: bool,
) -> list[dict]:
    if accumulated_new_candidates:
        return _upsert_and_build_source_candidates(
            topic,
            accumulated_new_candidates,
            prune_missing_discovered=prune_missing_discovered,
        )
    if prune_missing_discovered:
        _upsert_and_build_source_candidates(topic, [], prune_missing_discovered=True)
    return _build_persisted_new_source_candidates(topic)


def _build_discovery_cycle_round_summary(
    *,
    topic: Topic,
    round_result: dict,
    round_index: int,
    accumulated_visible_suggestions: int,
) -> dict:
    run = round_result["discovery_run"]
    visible_new_suggestions = len(round_result["new_visible_candidates"])
    returned_count = int(round_result.get("returned_count") or getattr(run, "provider_result_count", 0) or 0)
    diagnosis = build_discovery_cycle_round_diagnosis(
        round_result=round_result,
        returned_count=returned_count,
        visible_new_suggestions=visible_new_suggestions,
        target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
    )
    repair_plan_for_next_round = _build_round_repair_plan(
        topic=topic,
        round_result=round_result,
        diagnosis=diagnosis,
    )
    return {
        "run_id": run.id,
        "round_index": round_index,
        "visible_new_suggestions": visible_new_suggestions,
        "accepted_count": int(round_result.get("accepted_count") or 0),
        "quality_rejected_count": int(round_result.get("quality_rejected_count") or 0),
        "known_or_duplicate_count": int(round_result.get("known_or_duplicate_count") or 0),
        "provider_error_count": int(round_result.get("provider_error_count") or 0),
        "returned_count": returned_count,
        "reason_summary": str(round_result.get("reason_summary") or "").strip() or "mixed_low_yield",
        "accumulated_visible_suggestions": accumulated_visible_suggestions,
        "diagnosis": diagnosis,
        "repair_plan_for_next_round": repair_plan_for_next_round,
        "query_rows": _extract_repair_query_rows(dict(getattr(run, "diagnostics", {}) or {})),
        "quality_feedback": dict((dict(getattr(run, "diagnostics", {}) or {})).get("source_quality_feedback") or {}),
        "used_repair_plan": bool(round_result.get("used_repair_plan")),
        "repair_plan_usage": dict(round_result.get("repair_plan_usage") or {}),
    }


def _build_discovery_cycle_payload(
    *,
    topic: Topic,
    cycle_id: str,
    target_visible_new_suggestions: int,
    max_immediate_rounds: int,
    round_count: int,
    accumulated_visible_suggestions: int,
    decision: str,
    rounds: list[dict],
) -> dict:
    cycle_diagnosis = build_discovery_cycle_overall_diagnosis(
        decision=decision,
        rounds=rounds,
        accumulated_visible_suggestions=accumulated_visible_suggestions,
        target_visible_new_suggestions=DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
    )
    repair_plan = _build_discovery_repair_plan(
        topic=topic,
        diagnosis=cycle_diagnosis,
        rounds=rounds,
    )
    return build_discovery_cycle_payload(
        cycle_id=cycle_id,
        target_visible_new_suggestions=target_visible_new_suggestions,
        max_immediate_rounds=max_immediate_rounds,
        round_count=round_count,
        accumulated_visible_suggestions=accumulated_visible_suggestions,
        decision=decision,
        rounds=rounds,
        cycle_diagnosis=cycle_diagnosis,
        repair_plan=repair_plan,
    )


def _build_discovery_cycle_round_stub(*, cycle_id: str, round_index: int) -> dict:
    return {
        "cycle_id": cycle_id,
        "target_visible_new_suggestions": DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
        "max_immediate_rounds": DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS,
        "round_index": int(round_index),
    }


def _attach_discovery_cycle_to_runs(rounds: list[dict], cycle_payload: dict) -> None:
    round_count = int(cycle_payload.get("round_count") or len(rounds))
    for round_item in rounds:
        run = SourceDiscoveryRun.objects.filter(pk=round_item.get("run_id")).first()
        if run is None:
            continue
        diagnostics = dict(run.diagnostics or {})
        diagnostics["discovery_cycle"] = {
            **cycle_payload,
            "round_index": int(round_item.get("round_index") or 0),
            "round_count": round_count,
            "rounds": cycle_payload.get("rounds", []),
        }
        run.diagnostics = diagnostics
        run.save(update_fields=["diagnostics", "updated_at"])


def _build_discovery_cycle_summary(
    *,
    topic: Topic,
    candidate_records: list[dict],
    accumulated_visible_suggestions: int,
    round_count: int,
    decision: str,
    round_results: list[dict],
) -> dict:
    provider_name = ""
    provider_result_count = 0
    candidate_input_count = 0
    query_count = 0
    provider_error_count = 0
    if round_results:
        runs_by_id = {
            run.id: run
            for run in SourceDiscoveryRun.objects.filter(pk__in=[item.get("run_id") for item in round_results if item.get("run_id")])
        }
        last_run = runs_by_id.get(round_results[-1].get("run_id"))
        if last_run is not None and isinstance(last_run.diagnostics, dict):
            diagnostics = last_run.diagnostics
            provider_name = str(diagnostics.get("provider_name") or last_run.provider_name or "").strip() or "unknown"
        provider_result_count = sum(
            int((runs_by_id.get(item.get("run_id")).provider_result_count if runs_by_id.get(item.get("run_id")) else 0) or 0)
            for item in round_results
        )
        candidate_input_count = sum(
            int(
                (
                    (runs_by_id.get(item.get("run_id")).diagnostics or {}).get("candidate_input_count")
                    if runs_by_id.get(item.get("run_id"))
                    else 0
                )
                or 0
            )
            for item in round_results
        )
        query_count = sum(
            int((runs_by_id.get(item.get("run_id")).query_count if runs_by_id.get(item.get("run_id")) else 0) or 0)
            for item in round_results
        )
        provider_error_count = sum(int(item.get("provider_error_count") or 0) for item in round_results)

    existing_new_suggestion_count = _count_existing_new_suggestions(topic)

    if decision == "provider_unavailable":
        title = "Source search is temporarily unavailable"
        if existing_new_suggestion_count > 0:
            body = "DigestFlow could not connect to the search provider. Existing suggestions were kept."
        else:
            body = "DigestFlow could not connect to the search provider. Please try again later."
        execution_status = "provider_unavailable"
    elif provider_error_count > 0 and accumulated_visible_suggestions > 0:
        title = "Source discovery partially completed"
        body = (
            f"Some searches could not be completed. {accumulated_visible_suggestions} new source suggestion"
            f"{'s' if accumulated_visible_suggestions != 1 else ''} "
            f"{'are' if accumulated_visible_suggestions != 1 else 'is'} still available"
            f"{f' after {round_count} search rounds' if round_count > 1 else ''}."
        )
        execution_status = "failed"
    elif decision == "target_reached":
        title = "Source discovery completed"
        if round_count > 1:
            body = (
                f"Found {accumulated_visible_suggestions} new source suggestion"
                f"{'s' if accumulated_visible_suggestions != 1 else ''} after {round_count} search rounds."
            )
        else:
            body = (
                f"Found {accumulated_visible_suggestions} new source suggestion"
                f"{'s' if accumulated_visible_suggestions != 1 else ''}."
            )
        execution_status = "completed"
    else:
        title = "Source discovery partially completed"
        if accumulated_visible_suggestions > 0:
            body = (
                f"Found {accumulated_visible_suggestions} new source suggestion"
                f"{'s' if accumulated_visible_suggestions != 1 else ''} after {round_count} search rounds. "
                f"DigestFlow could not reach the {DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS}-source target "
                f"with the current search strategy."
            )
        else:
            body = (
                f"DigestFlow could not reach the {DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS}-source target "
                f"after {round_count} search rounds with the current search strategy."
            )
        execution_status = "partial"

    return {
        "title": title,
        "body": body,
        "provider_name": provider_name,
        "execution_status": execution_status,
        "provider_result_count": provider_result_count,
        "candidate_input_count": candidate_input_count,
        "query_count": query_count,
        "provider_error_count": provider_error_count,
        "existing_new_suggestion_count": existing_new_suggestion_count,
        "discovery_cycle": round_results and {
            "target_visible_suggestions": DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
            "target_visible_new_suggestions": DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
            "max_immediate_rounds": DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS,
            "rounds_run": round_count,
            "round_count": round_count,
            "accumulated_visible_suggestions": accumulated_visible_suggestions,
            "decision": decision,
        } or None,
    }


def _count_known_provider_results(*, source_research_result, known_normalized_urls: set[str]) -> int:
    known_count = 0
    for candidate in source_research_result.evaluated_candidates:
        normalized_url = str(candidate.normalized_url or "").strip()
        if normalized_url and normalized_url in known_normalized_urls:
            known_count += 1
    return known_count


def _build_source_discovery_run_diagnostics(
    *,
    source_research_result,
    known_normalized_urls: set[str],
    shown_candidates: list[dict],
) -> dict:
    diagnostics = dict(source_research_result.diagnostics)
    diagnostics["query_performance"] = _build_query_performance_entries_for_run(
        source_research_result=source_research_result,
        known_normalized_urls=known_normalized_urls,
        shown_candidates=shown_candidates,
    )
    diagnostics["source_quality_feedback"] = build_source_quality_feedback(
        source_research_result=source_research_result,
        known_normalized_urls=known_normalized_urls,
        shown_candidates=shown_candidates,
    )
    return diagnostics


def _build_query_performance_entries_for_run(
    *,
    source_research_result,
    known_normalized_urls: set[str],
    shown_candidates: list[dict],
) -> list[dict]:
    base_entries = source_research_result.diagnostics.get("query_performance") or []
    query_rows: list[dict] = []
    query_index: dict[str, dict] = {}

    for item in base_entries:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        row = {
            "query": query,
            "provider": str(item.get("provider") or source_research_result.provider_result.provider_name or "").strip(),
            "angle": str(item.get("angle") or "").strip(),
            "purpose": str(item.get("purpose") or "").strip(),
            "returned_count": int(item.get("returned_count") or 0),
            "accepted_count": int(item.get("accepted_count") or 0),
            "rejected_count": int(item.get("rejected_count") or 0),
            "duplicate_count": int(item.get("duplicate_count") or 0),
            "visible_new_suggestions_count": 0,
            "status": str(item.get("status") or "").strip(),
        }
        for key in (
            "source",
            "repair_action",
            "semantic_shift_type",
            "material_type",
            "old_query",
            "repair_plan_source_round",
            "surface_key",
            "diversity_reason",
        ):
            if str(item.get(key) or "").strip():
                row[key] = str(item.get(key) or "").strip()
        if str(item.get("error_message") or "").strip():
            row["error_message"] = str(item.get("error_message") or "").strip()
        query_rows.append(row)
        query_index[query] = row

    for candidate in source_research_result.evaluated_candidates:
        query = str(candidate.diagnostics.get("query") or "").strip()
        normalized_url = str(candidate.normalized_url or "").strip()
        if not query or not normalized_url or normalized_url not in known_normalized_urls:
            continue
        row = query_index.get(query)
        if row is None:
            continue
        row["duplicate_count"] = int(row.get("duplicate_count") or 0) + 1

    for candidate in shown_candidates:
        query = str(candidate.get("query") or "").strip()
        if not query:
            continue
        row = query_index.get(query)
        if row is None:
            continue
        row["visible_new_suggestions_count"] = int(row.get("visible_new_suggestions_count") or 0) + 1

    for row in query_rows:
        row["status"] = _derive_saved_query_performance_status(row)
    return query_rows


def _derive_saved_query_performance_status(item: dict) -> str:
    if str(item.get("error_message") or "").strip() or str(item.get("status") or "").strip() == "partial_error":
        return "partial_error"
    if int(item.get("visible_new_suggestions_count") or 0) > 0 or int(item.get("accepted_count") or 0) > 0:
        return "useful"
    if int(item.get("duplicate_count") or 0) > 0:
        return "duplicate_heavy"
    if int(item.get("returned_count") or 0) == 0:
        return "no_visible_results"
    if int(item.get("rejected_count") or 0) > 0:
        return "weak"
    return "no_visible_results"


def _count_existing_new_suggestions(topic: Topic) -> int:
    return topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED, is_pinned=False).count()


def _has_new_visible_suggestions(*, candidate_records: list[dict], known_normalized_urls: set[str]) -> bool:
    for candidate in candidate_records:
        normalized_url = str(candidate.get("normalized_url") or "").strip()
        if normalized_url and normalized_url not in known_normalized_urls:
            return True
    return False


def _filter_previously_handled_provider_candidates(topic: Topic, candidate_records: list[dict]) -> list[dict]:
    history_by_normalized = build_topic_history_by_normalized_url(topic)
    current_source_normalized_urls = {
        str(value or "").strip()
        for value in topic.sources.values_list("normalized_url", flat=True)
        if str(value or "").strip()
    }
    filtered_candidates: list[dict] = []
    seen_normalized_urls: set[str] = set()

    for candidate in candidate_records:
        normalized_url = str(candidate.get("normalized_url") or "").strip()
        source_url = str(candidate.get("url") or "").strip()
        if not normalized_url and source_url:
            normalized_url = classify_source_url(source_url).normalized_url
            candidate = {
                **candidate,
                "normalized_url": normalized_url,
            }
        if not normalized_url:
            continue
        if normalized_url in seen_normalized_urls:
            continue
        seen_normalized_urls.add(normalized_url)
        if normalized_url in current_source_normalized_urls:
            continue
        if normalized_url in history_by_normalized:
            continue
        filtered_candidates.append(candidate)

    return filtered_candidates


def _build_provider_discovery_summary(
    *,
    source_research_result,
    candidate_records: list[dict],
    execution_status: str,
    existing_new_suggestion_count: int = 0,
    had_new_visible_suggestions: bool = False,
) -> dict:
    diagnostics = dict(source_research_result.diagnostics or {})
    provider_name = str(diagnostics.get("provider_name") or "").strip() or "unknown"
    query_count = int(diagnostics.get("query_count") or 0)
    provider_result_count = int(diagnostics.get("raw_result_count") or 0)
    candidate_input_count = int(diagnostics.get("candidate_input_count") or 0)
    suggestion_count = len(candidate_records)
    provider_error_count = int(diagnostics.get("provider_error_count") or 0)
    preserved_existing_suggestions = (
        existing_new_suggestion_count > 0
        and suggestion_count == existing_new_suggestion_count
        and not had_new_visible_suggestions
    )

    if execution_status == "failed":
        if suggestion_count > 0:
            title = "Source discovery partially completed"
            body = (
                f"Some searches could not be completed. "
                f"{suggestion_count} new source suggestion{'s' if suggestion_count != 1 else ''} "
                f"{'are' if suggestion_count != 1 else 'is'} still available."
            )
        else:
            title = "Source discovery did not complete"
            body = (
                "Provider results could not be loaded for this research search. Existing suggestions were kept."
                if existing_new_suggestion_count > 0
                else "Provider results could not be loaded for this research search."
            )
    elif suggestion_count == 0:
        title = "No new sources found"
        body = "No new sources found."
    elif preserved_existing_suggestions:
        title = "No new sources found"
        body = "No new sources found. Existing suggestions were kept."
    else:
        title = "Source discovery completed"
        body = f"Found {suggestion_count} new source suggestion{'s' if suggestion_count != 1 else ''}."

    return {
        "title": title,
        "body": body,
        "provider_name": provider_name,
        "execution_status": execution_status,
        "provider_result_count": provider_result_count,
        "candidate_input_count": candidate_input_count,
        "query_count": query_count,
        "provider_error_count": provider_error_count,
        "existing_new_suggestion_count": existing_new_suggestion_count,
    }


def _build_discovery_results_summary(
    *,
    total_visible_candidates: int,
    total_new_source_candidates: int,
) -> dict:
    summary = {
        "title": "Source discovery results",
        "provider_name": "",
        "execution_status": "reviewing",
        "provider_result_count": 0,
        "candidate_input_count": 0,
        "query_count": 0,
        "provider_error_count": 0,
        "visible_suggestion_count": total_visible_candidates,
        "total_suggestion_count": total_new_source_candidates,
        "is_truncated": total_new_source_candidates > total_visible_candidates,
        "truncation_hint": "",
    }
    if total_new_source_candidates == 0:
        summary["body"] = "No new research suggestions remain from the current discovery results."
    elif total_new_source_candidates > total_visible_candidates:
        summary["body"] = (
            f"{total_visible_candidates} of {total_new_source_candidates} suggestions shown"
        )
        summary["truncation_hint"] = (
            f"Showing the first {total_visible_candidates} suggestions. "
            "Refine the research focus to narrow results."
        )
    else:
        summary["body"] = (
            f"{total_new_source_candidates} research suggestion"
            f"{'s' if total_new_source_candidates != 1 else ''} available"
        )
    return summary


def _finalize_discovery_summary(
    discovery_summary: dict | None,
    *,
    total_visible_candidates: int,
    total_new_source_candidates: int,
) -> dict | None:
    if discovery_summary is None:
        return None

    summary = dict(discovery_summary)
    summary["visible_suggestion_count"] = total_visible_candidates
    summary["total_suggestion_count"] = total_new_source_candidates
    summary["is_truncated"] = total_new_source_candidates > total_visible_candidates
    if summary["is_truncated"]:
        summary["truncation_hint"] = (
            f"Showing the first {total_visible_candidates} suggestions. "
            "Refine the research focus to narrow results."
        )
    else:
        summary["truncation_hint"] = ""
    return summary


def _build_discovery_query_rows(diagnostics: dict) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for item in diagnostics.get("per_query_result_counts", []) or []:
        if not isinstance(item, dict):
            continue
        intent = _format_query_intent_label(str(item.get("intent") or "").strip())
        result_count = int(item.get("result_count") or 0)
        query = str(item.get("query") or "").strip()
        if not intent and not query:
            continue
        rows.append(
            {
                "label": intent or "query",
                "value": f"{result_count} result{'s' if result_count != 1 else ''}",
                "query": query,
            }
        )
    return rows


def _build_research_history_run_entries(topic: Topic) -> list[dict]:
    runs = list(topic.source_discovery_runs.order_by("-created_at", "-id"))
    entries: list[dict] = []
    for run in runs:
        diagnostics = dict(run.diagnostics or {})
        provider_errors = _build_research_history_provider_errors(diagnostics)
        compact_metrics = _build_research_history_compact_metrics(run, diagnostics)
        stage_diagnostics = _build_research_history_stage_diagnostics(run, diagnostics)
        entries.append(
            {
                "run": run,
                "title": _format_research_history_status_title(run.status, run=run),
                "subtitle": _build_research_history_status_subtitle(run),
                "completed_label": _format_research_history_timestamp(run),
                "compact_metrics": compact_metrics,
                "stage_diagnostics": stage_diagnostics,
                "strategy_rows": _build_research_history_strategy_rows(run, diagnostics),
                "technical_rows": _build_research_history_detail_rows(run, diagnostics, provider_errors),
                "query_rows": _build_discovery_query_rows(diagnostics),
                "quality_feedback": _build_research_history_quality_feedback(diagnostics),
                "cycle_info": _build_research_history_cycle_info(diagnostics),
                "provider_errors": provider_errors,
                "warning_title": _build_research_history_warning_title(run, provider_errors),
                "warning_body": _build_research_history_warning_body(run, provider_errors),
            }
        )
    return entries


def _build_seen_source_history_section(
    topic: Topic,
    *,
    status_filter: str,
    search_query: str,
    page_number: str,
) -> dict:
    valid_filters = {
        "": "",
        "kept": SourceDiscoveryHistory.STATUS_KEPT,
        "shown": SourceDiscoveryHistory.STATUS_SHOWN,
        "removed": SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
        "rejected": SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
        "seen": SourceDiscoveryHistory.STATUS_SEEN,
    }
    applied_filter = status_filter if status_filter in valid_filters else ""

    queryset = topic.source_discovery_history.order_by("-last_seen_at", "-id")
    if valid_filters[applied_filter]:
        queryset = queryset.filter(status=valid_filters[applied_filter])
    if search_query:
        queryset = queryset.filter(
            Q(title__icontains=search_query)
            | Q(domain__icontains=search_query)
            | Q(url__icontains=search_query)
            | Q(normalized_url__icontains=search_query)
        )

    paginator = Paginator(queryset, 25)
    page_obj = paginator.get_page(page_number)
    entries = _build_seen_source_history_entries(list(page_obj.object_list))
    filters = _build_seen_source_history_filters(topic, applied_filter, search_query)
    pagination = _build_seen_source_history_pagination(page_obj, applied_filter, search_query)
    return {
        "entries": entries,
        "filters": filters,
        "search_query": search_query,
        "active_filter": applied_filter,
        "pagination": pagination,
    }


def _build_seen_source_history_entries(history_rows: list[SourceDiscoveryHistory]) -> list[dict]:
    entries: list[dict] = []
    for row in history_rows:
        display_url = row.url or row.normalized_url
        details: list[dict[str, str]] = []
        freshness_label = _format_source_history_freshness_label(row.freshness_status)
        if freshness_label and freshness_label not in {"Unknown date", "Unknown"}:
            details.append(
                {"label": "Freshness", "value": freshness_label, "kind": "text"}
            )
        if row.detected_publication_year:
            details.append({"label": "Publication year", "value": str(row.detected_publication_year), "kind": "text"})
        if str(row.quality_rejection_reason or "").strip():
            details.append({"label": "Quality note", "value": row.quality_rejection_reason.strip(), "kind": "text"})
        if details:
            details.insert(
                0,
                {"label": "First seen", "value": _format_history_timestamp(row.first_seen_at), "kind": "text"},
            )
        entries.append(
            {
                "title": row.title or row.normalized_url or row.url,
                "url": display_url,
                "domain": row.domain or "unknown",
                "status_label": _format_source_history_status_label(row.status),
                "outcome_label": _format_source_history_outcome_label(row.last_run_outcome),
                "seen_count": str(row.seen_count or 0),
                "last_seen": _format_history_timestamp(row.last_seen_at),
                "details": details,
                "has_details": bool(details),
            }
        )
    return entries


def _build_query_performance_section(topic: Topic) -> dict:
    entries: list[dict] = []
    for run in topic.source_discovery_runs.order_by("-completed_at", "-created_at", "-id"):
        diagnostics = run.diagnostics if isinstance(run.diagnostics, dict) else {}
        query_rows = diagnostics.get("query_performance") or _build_legacy_query_performance_rows(run, diagnostics)
        for item in query_rows:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "").strip()
            if not query:
                continue
            entries.append(
                {
                    "query": query,
                    "provider": str(item.get("provider") or run.provider_name or "").strip() or "unknown",
                    "purpose": _build_query_performance_purpose(item),
                    "returned_count": _format_query_metric_value(item.get("returned_count")),
                    "accepted_count": _format_query_metric_value(item.get("accepted_count")),
                    "visible_count": _format_query_metric_value(item.get("visible_new_suggestions_count")),
                    "rejected_count": _format_query_metric_value(item.get("rejected_count")),
                    "duplicate_count": _format_query_metric_value(item.get("duplicate_count")),
                    "status_label": _format_query_performance_status_label(str(item.get("status") or "").strip()),
                    "last_used": _format_research_history_timestamp(run),
                }
            )
    return {"entries": entries}


def _build_source_quality_feedback_section(topic: Topic) -> dict:
    aggregate = {
        "quality_rejected_count": 0,
        "known_or_duplicate_count": 0,
        "shown_count": 0,
        "dominant_rejection_reasons": Counter(),
        "weak_domains": Counter(),
        "weak_material_types": Counter(),
        "preferred_material_types_found": Counter(),
        "planner_quality_guidance": [],
    }
    latest_main_quality_issue = ""

    for run in topic.source_discovery_runs.order_by("-completed_at", "-created_at", "-id")[:5]:
        diagnostics = run.diagnostics if isinstance(run.diagnostics, dict) else {}
        feedback = diagnostics.get("source_quality_feedback")
        if not isinstance(feedback, dict):
            continue
        aggregate["quality_rejected_count"] += int(feedback.get("quality_rejected_count") or 0)
        aggregate["known_or_duplicate_count"] += int(feedback.get("known_or_duplicate_count") or 0)
        aggregate["shown_count"] += int(feedback.get("shown_count") or 0)
        if not latest_main_quality_issue:
            latest_main_quality_issue = str(feedback.get("main_quality_issue") or "").strip()
        for item in feedback.get("dominant_rejection_reasons") or []:
            if isinstance(item, dict):
                reason = str(item.get("reason") or "").strip()
                if reason:
                    aggregate["dominant_rejection_reasons"][reason] += int(item.get("count") or 0)
        for item in feedback.get("weak_domains") or []:
            if isinstance(item, dict):
                domain = str(item.get("domain") or "").strip()
                reason = str(item.get("reason") or "").strip()
                if domain:
                    aggregate["weak_domains"][(domain, reason)] += int(item.get("count") or 0)
        for item in feedback.get("weak_material_types") or []:
            if isinstance(item, dict):
                material_type = str(item.get("material_type") or "").strip()
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if material_type:
                    aggregate["weak_material_types"][(material_type, label)] += int(item.get("count") or 0)
        for item in feedback.get("preferred_material_types_found") or []:
            if isinstance(item, dict):
                material_type = str(item.get("material_type") or "").strip()
                label = str(item.get("label") or item.get("material_type") or "").strip()
                if material_type:
                    aggregate["preferred_material_types_found"][(material_type, label)] += int(item.get("count") or 0)
        for item in feedback.get("planner_quality_guidance") or []:
            cleaned = str(item or "").strip()
            if cleaned and cleaned not in aggregate["planner_quality_guidance"]:
                aggregate["planner_quality_guidance"].append(cleaned)

    has_feedback = bool(
        aggregate["quality_rejected_count"]
        or aggregate["known_or_duplicate_count"]
        or aggregate["shown_count"]
        or aggregate["dominant_rejection_reasons"]
        or aggregate["weak_domains"]
        or aggregate["weak_material_types"]
        or aggregate["preferred_material_types_found"]
        or aggregate["planner_quality_guidance"]
    )
    return {
        "has_feedback": has_feedback,
        "main_quality_issue": latest_main_quality_issue or "No strong quality pattern detected yet.",
        "quality_rejected_count": str(aggregate["quality_rejected_count"]),
        "known_or_duplicate_count": str(aggregate["known_or_duplicate_count"]),
        "shown_count": str(aggregate["shown_count"]),
        "dominant_rejection_reasons": [
            {"reason": reason, "count": str(count)}
            for reason, count in aggregate["dominant_rejection_reasons"].most_common(3)
        ],
        "weak_domains": [
            {"domain": domain, "reason": reason, "count": str(count)}
            for (domain, reason), count in aggregate["weak_domains"].most_common(3)
        ],
        "weak_material_types": [
            {"material_type": material_type, "label": label, "count": str(count)}
            for (material_type, label), count in aggregate["weak_material_types"].most_common(4)
        ],
        "preferred_material_types_found": [
            {"material_type": material_type, "label": label, "count": str(count)}
            for (material_type, label), count in aggregate["preferred_material_types_found"].most_common(4)
        ],
        "planner_quality_guidance": aggregate["planner_quality_guidance"][:4],
    }


def _build_search_surface_memory_section(topic: Topic) -> dict:
    latest_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    if latest_run is None or not isinstance(latest_run.diagnostics, dict):
        return {"has_memory": False}
    history_summary = latest_run.diagnostics.get("query_history_summary")
    if not isinstance(history_summary, dict):
        return {"has_memory": False}
    memory = history_summary.get("search_surface_memory")
    if not isinstance(memory, dict):
        return {"has_memory": False}

    surfaces = [
        item
        for item in memory.get("surfaces") or []
        if isinstance(item, dict) and str(item.get("surface_key") or "").strip()
    ]
    if not surfaces and not any(memory.get(key) for key in ("avoided_surfaces", "preferred_surfaces", "underexplored_surfaces")):
        return {"has_memory": False}

    return {
        "has_memory": True,
        "recent_run_count": int(memory.get("recent_run_count") or 0),
        "avoided_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("avoided_surfaces") or []
            if str(item or "").strip()
        ],
        "preferred_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("preferred_surfaces") or []
            if str(item or "").strip()
        ],
        "underexplored_surfaces": [
            _humanize_surface_key_label(item)
            for item in memory.get("underexplored_surfaces") or []
            if str(item or "").strip()
        ],
        "surfaces": [
            {
                "label": _humanize_surface_key_label(item.get("surface_key")),
                "status": _humanize_surface_status_label(item.get("status")),
                "reason": str(item.get("reason") or "").strip(),
            }
            for item in surfaces[:4]
        ],
    }


def _build_legacy_query_performance_rows(run: SourceDiscoveryRun, diagnostics: dict) -> list[dict]:
    rows: list[dict] = []
    for item in diagnostics.get("per_query_result_counts", []) or []:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query") or "").strip()
        if not query:
            continue
        rows.append(
            {
                "query": query,
                "provider": str(item.get("provider_name") or run.provider_name or "").strip(),
                "angle": str(item.get("angle") or "").strip(),
                "purpose": str(item.get("purpose") or item.get("query_reason") or "").strip(),
                "returned_count": int(item.get("result_count") or 0),
                "accepted_count": None,
                "visible_new_suggestions_count": None,
                "rejected_count": None,
                "duplicate_count": int(item.get("duplicate_url_count") or 0),
                "status": "partial_error" if str(item.get("error") or "").strip() else "no_visible_results",
            }
        )
    return rows


def _build_query_performance_purpose(item: dict) -> str:
    angle = str(item.get("angle") or "").strip()
    purpose = str(item.get("purpose") or "").strip()
    if angle and purpose:
        return f"{angle} — {purpose}"
    if angle:
        return angle
    if purpose:
        return purpose
    return "—"


def _format_query_metric_value(value) -> str:
    if value is None:
        return "—"
    return str(int(value))


def _format_query_performance_status_label(status: str) -> str:
    mapping = {
        "useful": "Useful",
        "weak": "Weak",
        "duplicate_heavy": "Duplicate-heavy",
        "no_visible_results": "No visible results",
        "partial_error": "Partial/error",
    }
    return mapping.get(str(status or "").strip(), "No visible results")


def _build_seen_source_history_filters(topic: Topic, active_filter: str, search_query: str) -> list[dict[str, str | bool]]:
    filter_specs = [
        {"key": "", "label": "All"},
        {"key": "kept", "label": "Kept"},
        {"key": "shown", "label": "Shown"},
        {"key": "rejected", "label": "Rejected"},
        {"key": "seen", "label": "Seen only"},
    ]
    filters: list[dict[str, str | bool]] = []
    for spec in filter_specs:
        params = {}
        if spec["key"]:
            params["status"] = spec["key"]
        if search_query:
            params["q"] = search_query
        url = reverse("topic-research-history", args=[topic.id])
        if params:
            url = f"{url}?{urlencode(params)}"
        url = f"{url}#seen-sources"
        filters.append(
            {
                "label": spec["label"],
                "url": url,
                "is_active": spec["key"] == active_filter,
            }
        )
    return filters


def _build_seen_source_history_pagination(page_obj, active_filter: str, search_query: str) -> dict:
    def build_url(page_number: int) -> str:
        params = {"page": page_number}
        if active_filter:
            params["status"] = active_filter
        if search_query:
            params["q"] = search_query
        return f"?{urlencode(params)}#seen-sources"

    return {
        "has_previous": page_obj.has_previous(),
        "has_next": page_obj.has_next(),
        "previous_url": build_url(page_obj.previous_page_number()) if page_obj.has_previous() else "",
        "next_url": build_url(page_obj.next_page_number()) if page_obj.has_next() else "",
        "page_number": page_obj.number,
        "total_pages": page_obj.paginator.num_pages,
        "showing_count": len(page_obj.object_list),
        "total_count": page_obj.paginator.count,
    }


def _build_research_history_compact_metrics(run: SourceDiscoveryRun, diagnostics: dict) -> str:
    status = str(run.status or "").strip().lower()
    status_label = "partial run" if status == SourceDiscoveryRun.STATUS_PARTIAL_FAILED else (
        "failed run" if status == SourceDiscoveryRun.STATUS_FAILED else (
            "blocked run" if status == SourceDiscoveryRun.STATUS_BLOCKED else "completed run"
        )
    )
    return " · ".join(
        [
            f"{int(run.provider_result_count or 0)} URLs returned",
            f"{int(run.new_suggestions_count or 0)} visible new suggestions",
            status_label,
        ]
    )


def _build_research_history_stage_diagnostics(run: SourceDiscoveryRun, diagnostics: dict) -> dict:
    duplicate_count = int(diagnostics.get("duplicate_url_count") or 0) + int(run.already_known_count or 0)
    return {
        "intro": "These counts describe different pipeline checks and may overlap; they are not an additive breakdown.",
        "rows": [
            {"label": "Passed filtering", "value": str(int(run.accepted_count or 0))},
            {"label": "Rejected by filters", "value": str(int(run.rejected_count or 0))},
            {"label": "Already known or duplicate", "value": str(duplicate_count)},
        ],
    }


def _build_research_history_strategy_rows(run: SourceDiscoveryRun, diagnostics: dict) -> list[dict[str, str]]:
    query_angle_key = str(diagnostics.get("selected_query_angle_key") or "").strip()
    query_angle_suffix = str(diagnostics.get("selected_query_angle_suffix") or "").strip()
    rows = [
        {"label": "Recency", "value": _format_recency_label(int(run.search_recency_months or 1))},
        {"label": "Queries used", "value": str(run.query_count)},
    ]
    if query_angle_key:
        rows.insert(
            0,
            {"label": "Search angle", "value": _format_query_angle_label(query_angle_key, query_angle_suffix)},
        )
    return rows


def _build_research_history_detail_rows(
    run: SourceDiscoveryRun,
    diagnostics: dict,
    provider_errors: list[str],
) -> list[dict[str, str]]:
    query_angle_key = str(diagnostics.get("selected_query_angle_key") or "").strip()
    query_angle_suffix = str(diagnostics.get("selected_query_angle_suffix") or "").strip()
    query_angle_reason = str(diagnostics.get("selected_query_angle_reason") or "").strip()
    previous_run_count = int(diagnostics.get("previous_discovery_run_count") or 0)

    rows = [
        {"label": "Raw status", "value": str(run.status or "unknown")},
        {"label": "Provider", "value": run.provider_name or "unknown"},
    ]
    if query_angle_reason:
        rows.append({"label": "Angle reason", "value": query_angle_reason})
    rows.extend(
        [
            {"label": "Previous discovery runs", "value": str(previous_run_count)},
            {"label": "Recency", "value": _format_recency_label(int(run.search_recency_months or 1))},
            {"label": "Provider filter", "value": run.search_time_filter or str(diagnostics.get("provider_tbs") or "").strip() or "none"},
        ]
    )
    if provider_errors:
        rows.append({"label": "Technical reason", "value": provider_errors[0]})
    return rows


def _build_research_history_quality_feedback(diagnostics: dict) -> dict:
    feedback = diagnostics.get("source_quality_feedback")
    if not isinstance(feedback, dict):
        return {"has_feedback": False}
    return {
        "has_feedback": True,
        "main_quality_issue": str(feedback.get("main_quality_issue") or "").strip(),
        "rows": [
            {"label": "Quality rejected", "value": str(int(feedback.get("quality_rejected_count") or 0))},
            {"label": "Known / duplicate", "value": str(int(feedback.get("known_or_duplicate_count") or 0))},
            {"label": "Shown", "value": str(int(feedback.get("shown_count") or 0))},
        ],
        "weak_material_types": [
            {
                "label": str(item.get("label") or item.get("material_type") or "").strip(),
                "count": str(int(item.get("count") or 0)),
            }
            for item in feedback.get("weak_material_types") or []
            if isinstance(item, dict)
        ],
        "preferred_material_types_found": [
            {
                "label": str(item.get("label") or item.get("material_type") or "").strip(),
                "count": str(int(item.get("count") or 0)),
            }
            for item in feedback.get("preferred_material_types_found") or []
            if isinstance(item, dict)
        ],
        "planner_quality_guidance": [
            str(item).strip()
            for item in feedback.get("planner_quality_guidance") or []
            if str(item).strip()
        ][:4],
    }


def _build_research_history_cycle_info(diagnostics: dict) -> dict:
    cycle = diagnostics.get("discovery_cycle")
    if not isinstance(cycle, dict):
        return {"has_cycle": False}
    cycle_diagnosis = cycle.get("cycle_diagnosis") if isinstance(cycle.get("cycle_diagnosis"), dict) else {}
    repair_plan = cycle.get("repair_plan") if isinstance(cycle.get("repair_plan"), dict) else {}
    current_round_item = {}
    current_round_index = int(cycle.get("round_index") or 0)
    for item in cycle.get("rounds") or []:
        if isinstance(item, dict) and int(item.get("round_index") or 0) == current_round_index:
            current_round_item = item
            break
    return {
        "has_cycle": True,
        "summary": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip()),
        "rows": [
            {"label": "Cycle round", "value": f"{int(cycle.get('round_index') or 0)} of {int(cycle.get('round_count') or 0)}"},
            {"label": "Cycle target", "value": str(int(cycle.get("target_visible_new_suggestions") or 0))},
            {"label": "Max immediate rounds", "value": str(int(cycle.get("max_immediate_rounds") or 0))},
            {"label": "Rounds run", "value": str(int(cycle.get("round_count") or 0))},
            {
                "label": "Accumulated visible suggestions",
                "value": str(int(cycle.get("accumulated_visible_suggestions") or 0)),
            },
            {"label": "Cycle decision", "value": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip())},
        ],
        "decision": format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip()),
        "diagnosis": _build_research_history_cycle_diagnosis(cycle_diagnosis),
        "repair": _build_research_history_cycle_repair(repair_plan),
        "repair_used": _build_research_history_repair_usage(current_round_item),
    }


def _build_current_research_state(topic: Topic) -> dict:
    history_qs = topic.source_discovery_history.all()
    status_counts = Counter(history_qs.values_list("status", flat=True))
    current_discovered_sources = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED)
    last_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    last_cycle = {}
    if last_run is not None and isinstance(last_run.diagnostics, dict):
        last_cycle = last_run.diagnostics.get("discovery_cycle") or {}
    return {
        "cards": [
            {"label": "Kept", "value": str(current_discovered_sources.filter(is_pinned=True).count())},
            {"label": "Shown now", "value": str(current_discovered_sources.filter(is_pinned=False).count())},
            {"label": "Rejected", "value": str(int(status_counts.get(SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY, 0)))},
            {"label": "Seen only", "value": str(int(status_counts.get(SourceDiscoveryHistory.STATUS_SEEN, 0)))},
        ],
        "last_run_status": _format_research_history_status_title(last_run.status, run=last_run) if last_run else "No discovery runs yet",
        "last_run_subtitle": _build_current_research_feedback_note(last_run) if last_run else "Run Find sources to start source discovery for this topic.",
        "last_cycle_summary": _build_current_research_cycle_summary(last_cycle),
    }


def _build_current_research_cycle_summary(cycle: dict) -> str:
    if not isinstance(cycle, dict) or not cycle:
        return ""
    target = int(cycle.get("target_visible_new_suggestions") or 0)
    visible = int(cycle.get("accumulated_visible_suggestions") or 0)
    decision = format_discovery_cycle_decision_label(str(cycle.get("decision") or "").strip())
    cycle_diagnosis = cycle.get("cycle_diagnosis") if isinstance(cycle.get("cycle_diagnosis"), dict) else {}
    diagnosis_note = ""
    if cycle_diagnosis:
        primary_cause = str(cycle_diagnosis.get("primary_cause") or "").strip()
        if primary_cause not in {"", "target_reached", "provider_unavailable"}:
            diagnosis_note = f" — {format_discovery_cycle_diagnosis_label(primary_cause).lower()}."
    if target > 0:
        return f"Last discovery cycle: {decision.lower()} ({visible} of {target} visible suggestions).{diagnosis_note}"
    return f"Last discovery cycle: {decision.lower()}.{diagnosis_note}"


def _build_current_research_feedback_note(run: SourceDiscoveryRun) -> str:
    diagnostics = dict(getattr(run, "diagnostics", {}) or {})
    cycle = diagnostics.get("discovery_cycle")
    if not isinstance(cycle, dict) or not cycle:
        return _build_research_history_status_subtitle(run)

    decision = str(cycle.get("decision") or "").strip()
    visible = int(cycle.get("accumulated_visible_suggestions") or 0)
    rounds_run = int(cycle.get("rounds_run") or cycle.get("round_count") or 0)
    if decision == "provider_unavailable":
        return _build_research_history_status_subtitle(run)
    if decision == "target_reached" and visible > 0 and rounds_run > 0:
        return (
            f"Target reached: {visible} new source suggestion{'s' if visible != 1 else ''} "
            f"after {rounds_run} search round{'s' if rounds_run != 1 else ''}."
        )
    if visible > 0 and rounds_run > 0:
        return (
            f"{visible} new source suggestion{'s' if visible != 1 else ''} were found "
            f"after {rounds_run} search round{'s' if rounds_run != 1 else ''}."
        )
    return _build_research_history_status_subtitle(run)


def _build_full_research_history_copy_report(
    *,
    topic: Topic,
    current_research_state: dict,
    query_performance_entries: list[dict],
    source_quality_feedback: dict,
    search_surface_memory: dict,
    history_runs: list[dict],
    seen_source_history: list[dict],
) -> str:
    lines: list[str] = []
    active_source_count = topic.sources.filter(is_active=True).count()
    active_discovered_count = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED, is_active=True).count()
    active_manual_count = topic.sources.filter(is_active=True).exclude(origin=TopicSourceOrigin.DISCOVERED).count()
    current_discovered_sources = topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED)
    latest_run = topic.source_discovery_runs.order_by("-created_at", "-id").first()
    latest_history_summary = {}
    if latest_run is not None and isinstance(latest_run.diagnostics, dict):
        latest_history_summary = latest_run.diagnostics.get("query_history_summary") or {}

    def add_section(title: str) -> None:
        if lines:
            lines.append("")
        lines.append(title)

    add_section("Topic")
    lines.append(f"- id: {topic.id}")
    lines.append(f"- name: {topic.name}")

    add_section("Current research state")
    lines.append(f"- source mode: {topic.source_mode}")
    for metric in current_research_state.get("cards", []):
        lines.append(f"- {metric.get('label')}: {metric.get('value')}")
    lines.append(f"- active selected sources: {active_source_count}")
    lines.append(f"- active selected research sources: {active_discovered_count}")
    lines.append(f"- active selected my sources: {active_manual_count}")
    lines.append(f"- current kept discovered sources: {current_discovered_sources.filter(is_pinned=True).count()}")
    lines.append(f"- current shown discovered sources: {current_discovered_sources.filter(is_pinned=False).count()}")
    lines.append(f"- last discovery run status: {current_research_state.get('last_run_status')}")
    lines.append(f"- last discovery run note: {current_research_state.get('last_run_subtitle')}")

    add_section("Query performance")
    if query_performance_entries:
        for item in query_performance_entries:
            lines.append(f"- query: {item.get('query')}")
            lines.append(f"  provider: {item.get('provider')}")
            lines.append(f"  purpose / angle: {item.get('purpose')}")
            lines.append(f"  returned: {item.get('returned_count')}")
            lines.append(f"  accepted: {item.get('accepted_count')}")
            lines.append(f"  visible: {item.get('visible_count')}")
            lines.append(f"  rejected: {item.get('rejected_count')}")
            lines.append(f"  known / duplicates: {item.get('duplicate_count')}")
            lines.append(f"  status: {item.get('status_label')}")
            lines.append(f"  last used: {item.get('last_used')}")
    else:
        lines.append("- No query performance data yet.")

    add_section("Source quality feedback")
    if source_quality_feedback.get("has_feedback"):
        lines.append(f"- quality rejected count: {source_quality_feedback.get('quality_rejected_count')}")
        lines.append(f"- known or duplicate count: {source_quality_feedback.get('known_or_duplicate_count')}")
        lines.append(f"- shown count: {source_quality_feedback.get('shown_count')}")
        lines.append(f"- main quality issue: {source_quality_feedback.get('main_quality_issue')}")
        if source_quality_feedback.get("dominant_rejection_reasons"):
            lines.append("- dominant rejection reasons:")
            for item in source_quality_feedback.get("dominant_rejection_reasons", []):
                lines.append(f"  - {item.get('reason')}: {item.get('count')}")
        if source_quality_feedback.get("weak_domains"):
            lines.append("- weak domains:")
            for item in source_quality_feedback.get("weak_domains", []):
                lines.append(f"  - {item.get('domain')}: {item.get('count')} ({item.get('reason')})")
        if source_quality_feedback.get("weak_material_types"):
            lines.append("- weak material types:")
            for item in source_quality_feedback.get("weak_material_types", []):
                lines.append(f"  - {item.get('label')}: {item.get('count')}")
        if source_quality_feedback.get("preferred_material_types_found"):
            lines.append("- preferred material types found:")
            for item in source_quality_feedback.get("preferred_material_types_found", []):
                lines.append(f"  - {item.get('label')}: {item.get('count')}")
        if source_quality_feedback.get("planner_quality_guidance"):
            lines.append("- planner quality guidance:")
            for item in source_quality_feedback.get("planner_quality_guidance", []):
                lines.append(f"  - {item}")
    else:
        lines.append("- No source quality feedback yet.")

    add_section("Search surface memory")
    if search_surface_memory.get("has_memory"):
        lines.append(f"- recent run count: {search_surface_memory.get('recent_run_count')}")
        if search_surface_memory.get("avoided_surfaces"):
            lines.append(f"- avoided surfaces: {', '.join(str(item) for item in search_surface_memory.get('avoided_surfaces') or [])}")
        if search_surface_memory.get("preferred_surfaces"):
            lines.append(f"- preferred surfaces: {', '.join(str(item) for item in search_surface_memory.get('preferred_surfaces') or [])}")
        if search_surface_memory.get("underexplored_surfaces"):
            lines.append(f"- underexplored surfaces: {', '.join(str(item) for item in search_surface_memory.get('underexplored_surfaces') or [])}")
        if search_surface_memory.get("surfaces"):
            lines.append("- surfaces:")
            for item in search_surface_memory.get("surfaces", []):
                lines.append(f"  - key: {item.get('label')}")
                lines.append(f"    status: {item.get('status')}")
                lines.append(f"    reason: {item.get('reason')}")
    else:
        lines.append("- No search surface memory yet.")

    add_section("Discovery runs")
    if history_runs:
        for item in history_runs:
            run = item.get("run")
            lines.append(f"- run id: {getattr(run, 'id', '')}")
            lines.append(f"  timestamp: {item.get('completed_label') or '—'}")
            lines.append(f"  status: {item.get('title')}")
            if item.get("subtitle"):
                lines.append(f"  note: {item.get('subtitle')}")
            if item.get("compact_metrics"):
                lines.append(f"  compact metrics: {item.get('compact_metrics')}")
            if item.get("warning_title") or item.get("warning_body"):
                lines.append(f"  provider warning: {item.get('warning_body') or item.get('warning_title')}")
            stage_diagnostics = item.get("stage_diagnostics") or {}
            if stage_diagnostics.get("rows"):
                lines.append("  stage diagnostics:")
                for metric in stage_diagnostics.get("rows", []):
                    lines.append(f"    - {metric.get('label')}: {metric.get('value')}")
            if item.get("strategy_rows"):
                lines.append("  search strategy:")
                for detail in item.get("strategy_rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
            if item.get("query_rows"):
                lines.append("  query rows:")
                for query in item.get("query_rows", []):
                    lines.append(f"    - {query.get('label')}: {query.get('value')}")
                    if query.get("query"):
                        lines.append(f"      query: {query.get('query')}")
            if item.get("technical_rows"):
                lines.append("  technical details:")
                for detail in item.get("technical_rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
            quality_feedback = item.get("quality_feedback") or {}
            if quality_feedback.get("has_feedback"):
                lines.append("  quality diagnostics:")
                for detail in quality_feedback.get("rows", []):
                    lines.append(f"    - {detail.get('label')}: {detail.get('value')}")
                if quality_feedback.get("main_quality_issue"):
                    lines.append(f"    - Main issue: {quality_feedback.get('main_quality_issue')}")
                for weak_item in quality_feedback.get("weak_material_types", []):
                    lines.append(f"    - Weak material type: {weak_item.get('label')} ({weak_item.get('count')})")
                for preferred_item in quality_feedback.get("preferred_material_types_found", []):
                    lines.append(f"    - Preferred material type: {preferred_item.get('label')} ({preferred_item.get('count')})")
    else:
        lines.append("- No discovery runs yet.")

    add_section("Seen sources")
    if seen_source_history:
        for item in seen_source_history:
            lines.append(f"- title: {item.get('title')}")
            lines.append(f"  url: {item.get('url') or '—'}")
            lines.append(f"  domain: {item.get('domain')}")
            lines.append(f"  status: {item.get('status_label')}")
            lines.append(f"  last outcome: {item.get('outcome_label')}")
            lines.append(f"  seen count: {item.get('seen_count')}")
            lines.append(f"  last seen: {item.get('last_seen')}")
            for detail in item.get("details", []):
                lines.append(f"  {str(detail.get('label') or '').lower()}: {detail.get('value')}")
    else:
        lines.append("- No seen sources yet.")

    add_section("Planner history guidance")
    if latest_history_summary:
        lines.append(f"- history available: {latest_history_summary.get('history_available')}")
        lines.append(f"- recent run count: {latest_history_summary.get('recent_run_count')}")
        lines.append(f"- malformed run count: {latest_history_summary.get('malformed_run_count')}")
        lines.append(f"- total query rows: {latest_history_summary.get('total_query_rows')}")
        useful_angles = latest_history_summary.get("useful_angles") or []
        weak_angles = latest_history_summary.get("weak_angles") or []
        quality_guidance = _dedupe_guidance_strings(latest_history_summary.get("quality_guidance") or [])
        planning_guidance = _dedupe_guidance_strings(
            latest_history_summary.get("planning_guidance") or [],
            skip_existing=quality_guidance,
        )
        if useful_angles:
            lines.append("- useful angles:")
            for item in useful_angles:
                lines.append(f"  - {item.get('angle')}: {item.get('count')}")
        if weak_angles:
            lines.append("- weak angles:")
            for item in weak_angles:
                lines.append(f"  - {item.get('angle')}: {item.get('count')}")
        if quality_guidance:
            lines.append("- quality guidance used for next run:")
            for item in quality_guidance:
                lines.append(f"  - {item}")
        if planning_guidance:
            lines.append("- planning guidance:")
            for item in planning_guidance:
                lines.append(f"  - {item}")
    else:
        lines.append("- No compact planner history guidance available.")

    add_section("Discovery cycle")
    latest_cycle = latest_run.diagnostics.get("discovery_cycle") if latest_run and isinstance(latest_run.diagnostics, dict) else {}
    if isinstance(latest_cycle, dict) and latest_cycle:
        lines.append(f"- target visible suggestions: {latest_cycle.get('target_visible_new_suggestions')}")
        lines.append(f"- max immediate rounds: {latest_cycle.get('max_immediate_rounds')}")
        lines.append(f"- rounds run: {latest_cycle.get('round_count')}")
        lines.append(f"- accumulated visible suggestions: {latest_cycle.get('accumulated_visible_suggestions')}")
        lines.append(f"- decision: {latest_cycle.get('decision')}")
        if str(latest_cycle.get("decision") or "").strip() not in {"", "target_reached"}:
            lines.append(f"- stop reason: {latest_cycle.get('decision')}")
        repair_plan = latest_cycle.get("repair_plan") if isinstance(latest_cycle.get("repair_plan"), dict) else {}
        if repair_plan:
            lines.append("- Strategy repair")
            lines.append(f"  - strategy: {repair_plan.get('strategy')}")
            lines.append(f"  - reason: {repair_plan.get('reason')}")
            constraints = repair_plan.get("constraints") if isinstance(repair_plan.get("constraints"), dict) else {}
            if constraints:
                lines.append("  - constraints:")
                for key in (
                    "avoid_repeating_queries",
                    "avoid_verbatim_failed_queries",
                    "avoid_duplicate_repaired_queries",
                    "avoid_near_duplicate_repaired_queries",
                    "avoid_long_natural_language_queries",
                    "prefer_compact_search_grade_queries",
                    "require_semantic_distance_from_failed_query",
                    "require_query_surface_diversity",
                ):
                    lines.append(f"    - {key}: {constraints.get(key)}")
                if constraints.get("prefer_material_types"):
                    lines.append(f"    - prefer_material_types: {', '.join(str(item) for item in constraints.get('prefer_material_types') or [])}")
                if constraints.get("avoid_material_types"):
                    lines.append(f"    - avoid_material_types: {', '.join(str(item) for item in constraints.get('avoid_material_types') or [])}")
                if constraints.get("avoid_domains"):
                    lines.append(f"    - avoid_domains: {', '.join(str(item) for item in constraints.get('avoid_domains') or [])}")
            lines.append("  - query repair plan:")
            for item in repair_plan.get("query_repair_plan") or []:
                if not isinstance(item, dict):
                    continue
                lines.append(f"    - old: {item.get('old_query')}")
                lines.append(f"      action: {item.get('action')}")
                lines.append(f"      semantic shift: {item.get('semantic_shift_type')}")
                lines.append(f"      new: {item.get('new_query')}")
                lines.append(f"      repair reason: {item.get('repair_reason')}")
                lines.append(f"      angle: {item.get('angle')}")
                lines.append(f"      material type: {item.get('material_type')}")
                if item.get("surface_key"):
                    lines.append(f"      surface key: {item.get('surface_key')}")
                if item.get("diversity_reason"):
                    lines.append(f"      diversity reason: {item.get('diversity_reason')}")
        latest_round_index = int(latest_cycle.get("round_index") or 0)
        latest_round_item = {}
        for round_item in latest_cycle.get("rounds") or []:
            if isinstance(round_item, dict) and int(round_item.get("round_index") or 0) == latest_round_index:
                latest_round_item = round_item
                break
        latest_repair_usage = latest_round_item.get("repair_plan_usage") if isinstance(latest_round_item.get("repair_plan_usage"), dict) else {}
        if latest_round_item.get("used_repair_plan"):
            lines.append("- Repair plan used")
            lines.append(f"  - source round: {latest_repair_usage.get('repair_plan_source_round')}")
            lines.append(f"  - queries used: {latest_repair_usage.get('queries_used_count') or len(latest_repair_usage.get('repair_queries_used') or [])}")
            lines.append(f"  - strategy: {latest_repair_usage.get('strategy')}")
            for item in latest_repair_usage.get("repair_queries_used") or []:
                if not isinstance(item, dict):
                    continue
                lines.append(f"  - query: {item.get('query')}")
                lines.append(f"    old query: {item.get('old_query')}")
                lines.append(f"    action: {item.get('action')}")
                lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                lines.append(f"    material type: {item.get('material_type')}")
        cycle_diagnosis = latest_cycle.get("cycle_diagnosis") if isinstance(latest_cycle.get("cycle_diagnosis"), dict) else {}
        if cycle_diagnosis:
            lines.append("- Search diagnosis")
            lines.append(f"  - primary cause: {cycle_diagnosis.get('primary_cause')}")
            secondary = cycle_diagnosis.get("secondary_causes") or []
            if secondary:
                lines.append(f"  - secondary causes: {', '.join(str(item) for item in secondary)}")
            lines.append(f"  - severity: {cycle_diagnosis.get('severity')}")
            lines.append(f"  - explanation: {cycle_diagnosis.get('explanation')}")
            lines.append(f"  - recommended next action: {cycle_diagnosis.get('recommended_next_action')}")
        for round_item in latest_cycle.get("rounds") or []:
            if not isinstance(round_item, dict):
                continue
            lines.append("")
            lines.append(f"Round {round_item.get('round_index')}")
            lines.append(f"- run id: {round_item.get('run_id')}")
            lines.append(f"- visible new suggestions: {round_item.get('visible_new_suggestions')}")
            lines.append(f"- accepted count: {round_item.get('accepted_count')}")
            lines.append(f"- quality rejected: {round_item.get('quality_rejected_count')}")
            lines.append(f"- known / duplicate: {round_item.get('known_or_duplicate_count')}")
            lines.append(f"- provider errors: {round_item.get('provider_error_count')}")
            lines.append(f"- returned count: {round_item.get('returned_count')}")
            lines.append(f"- reason summary: {round_item.get('reason_summary')}")
            if round_item.get("used_repair_plan"):
                round_usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
                lines.append("- repair plan used:")
                lines.append(f"  - source round: {round_usage.get('repair_plan_source_round')}")
                lines.append(f"  - queries used: {round_usage.get('queries_used_count') or len(round_usage.get('repair_queries_used') or [])}")
                lines.append(f"  - strategy: {round_usage.get('strategy')}")
                for item in round_usage.get("repair_queries_used") or []:
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"  - query: {item.get('query')}")
                    lines.append(f"    old query: {item.get('old_query')}")
                    lines.append(f"    action: {item.get('action')}")
                    lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                    lines.append(f"    material type: {item.get('material_type')}")
            round_repair = round_item.get("repair_plan_for_next_round") if isinstance(round_item.get("repair_plan_for_next_round"), dict) else {}
            if round_repair:
                lines.append("- strategy repair:")
                lines.append(f"  - strategy: {round_repair.get('strategy')}")
                lines.append(f"  - reason: {round_repair.get('reason')}")
                for item in round_repair.get("query_repair_plan") or []:
                    if not isinstance(item, dict):
                        continue
                    lines.append(f"  - old: {item.get('old_query')}")
                    lines.append(f"    action: {item.get('action')}")
                    lines.append(f"    semantic shift: {item.get('semantic_shift_type')}")
                    lines.append(f"    new: {item.get('new_query')}")
            round_diagnosis = round_item.get("diagnosis") if isinstance(round_item.get("diagnosis"), dict) else {}
            if round_diagnosis:
                lines.append("- diagnosis:")
                lines.append(f"  - primary cause: {round_diagnosis.get('primary_cause')}")
                secondary = round_diagnosis.get("secondary_causes") or []
                if secondary:
                    lines.append(f"  - secondary causes: {', '.join(str(item) for item in secondary)}")
                lines.append(f"  - severity: {round_diagnosis.get('severity')}")
                lines.append(f"  - explanation: {round_diagnosis.get('explanation')}")
                lines.append(f"  - recommended next action: {round_diagnosis.get('recommended_next_action')}")
    else:
        lines.append("- No discovery cycle diagnostics available yet.")

    return "\n".join(str(line) for line in lines if line is not None)


def _dedupe_guidance_strings(items, *, skip_existing=None) -> list[str]:
    normalized_items = items if isinstance(items, (list, tuple)) else []
    deduped: list[str] = []
    seen: set[str] = {
        re.sub(r"\s+", " ", str(item or "").strip()).casefold()
        for item in (skip_existing or [])
        if re.sub(r"\s+", " ", str(item or "").strip())
    }
    for item in normalized_items:
        cleaned = re.sub(r"\s+", " ", str(item or "").strip())
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(cleaned)
    return deduped


def _humanize_surface_key_label(surface_key: str) -> str:
    value = str(surface_key or "").strip()
    if not value:
        return ""
    labels = {
        "etf_flows_report": "ETF flows",
        "etf_flow_data_market_report": "ETF flow data",
        "institutional_demand_report": "institutional demand",
        "institutional_flows_report": "institutional flows",
        "funding_open_interest_report": "funding rates / open interest",
        "funding_rates_analysis": "funding rates",
        "open_interest_futures_positioning": "open interest",
        "derivatives_positioning_market_structure": "derivatives positioning",
        "market_structure_report": "market structure",
        "market_structure_research_paper": "market structure research paper",
        "research_paper": "research paper",
        "on_chain_exchange_reserves_analysis": "on-chain exchange reserves",
        "on_chain_weekly_report": "on-chain weekly report",
        "on_chain_analysis": "on-chain analysis",
        "analyst_report": "analyst report",
        "volatility_market_structure_report": "volatility market structure",
        "volatility_drawdown_risk_analysis": "volatility drawdown risk",
    }
    return labels.get(value, value.replace("_", " "))


def _humanize_surface_status_label(status: str) -> str:
    value = str(status or "").strip().replace("_", " ")
    if not value:
        return ""
    return value.capitalize()


def _format_research_history_timestamp(run: SourceDiscoveryRun) -> str:
    timestamp = run.completed_at or run.started_at
    return _format_history_timestamp(timestamp)


def _format_history_timestamp(timestamp) -> str:
    if timestamp is None:
        return ""
    return timezone.localtime(timestamp).strftime("%Y-%m-%d %H:%M")


def _build_research_history_provider_errors(diagnostics: dict) -> list[str]:
    errors: list[str] = []
    for item in diagnostics.get("provider_errors", []) or []:
        if not isinstance(item, dict):
            continue
        message = str(item.get("message") or "").strip()
        if message:
            errors.append(message)
    return errors


def _format_query_angle_label(query_angle_key: str, query_angle_suffix: str) -> str:
    if query_angle_key == "base":
        return "base"
    if query_angle_suffix:
        return query_angle_suffix
    return query_angle_key.replace("_", " ")


def _format_query_intent_label(intent: str) -> str:
    if not intent:
        return ""
    return intent.replace("_", " ")


def _format_recency_label(recency_months: int) -> str:
    if recency_months == 1:
        return "last 1 month"
    return f"last {recency_months} months"


def _format_source_history_status_label(status: str) -> str:
    mapping = {
        SourceDiscoveryHistory.STATUS_SEEN: "Seen only",
        SourceDiscoveryHistory.STATUS_SHOWN: "Shown as suggestion",
        SourceDiscoveryHistory.STATUS_KEPT: "Kept",
        SourceDiscoveryHistory.STATUS_REMOVED_BY_USER: "Removed by user",
        SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY: "Rejected by quality",
    }
    return mapping.get(str(status or "").strip(), str(status or "").strip() or "Unknown")


def _format_source_history_outcome_label(outcome: str) -> str:
    mapping = {
        SourceDiscoveryHistory.OUTCOME_NEW_SHOWN: "New suggestion",
        SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN: "Already known",
        SourceDiscoveryHistory.OUTCOME_DUPLICATE_URL: "Duplicate URL",
        SourceDiscoveryHistory.OUTCOME_DUPLICATE_DOMAIN: "Duplicate domain",
        SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED: "Previously removed",
        SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED: "Previously rejected",
        SourceDiscoveryHistory.OUTCOME_QUALITY_REJECTED: "Quality rejected",
        SourceDiscoveryHistory.OUTCOME_STALE_REJECTED: "Stale rejected",
        SourceDiscoveryHistory.OUTCOME_COMMERCIAL_REJECTED: "Commercial rejected",
        SourceDiscoveryHistory.OUTCOME_NONE: "None recorded",
    }
    return mapping.get(str(outcome or "").strip(), str(outcome or "").strip() or "None recorded")


def _format_source_history_freshness_label(freshness_status: str) -> str:
    mapping = {
        "fresh": "Fresh",
        "acceptable": "Acceptable",
        "unknown": "Unknown date",
        "stale": "Stale",
        "very_stale": "Very stale",
    }
    return mapping.get(str(freshness_status or "").strip().lower(), str(freshness_status or "").strip() or "Unknown")


def _format_research_history_status_title(status: str, *, run: SourceDiscoveryRun) -> str:
    normalized = str(status or "").strip().lower()
    if normalized == SourceDiscoveryRun.STATUS_COMPLETED:
        if int(run.new_suggestions_count or 0) == 0:
            return "No new suggestions found"
        return "Discovery completed"
    if normalized == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Discovery partially completed"
    if normalized == SourceDiscoveryRun.STATUS_FAILED:
        return "Discovery failed"
    if normalized == SourceDiscoveryRun.STATUS_BLOCKED:
        return "Discovery did not run"
    if normalized == SourceDiscoveryRun.STATUS_STARTED:
        return "Discovery started"
    return "Discovery update"


def _build_research_history_status_subtitle(run: SourceDiscoveryRun) -> str:
    status = str(run.status or "").strip().lower()
    if status == SourceDiscoveryRun.STATUS_COMPLETED:
        if int(run.new_suggestions_count or 0) > 0:
            count = int(run.new_suggestions_count or 0)
            return f"{count} new source suggestion{'s' if count != 1 else ''} were added." if count != 1 else "1 new source suggestion was added."
        if int(run.provider_result_count or 0) == 0:
            return "No new suggestions were found."
        return "The search ran, but all usable results were already known, rejected, or duplicates."
    if status == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some provider queries returned results, but at least one query failed."
    if status == SourceDiscoveryRun.STATUS_FAILED:
        return "Provider results could not be loaded."
    if status == SourceDiscoveryRun.STATUS_BLOCKED:
        return "Research provider was unavailable, so discovery did not run."
    if status == SourceDiscoveryRun.STATUS_STARTED:
        return "Source discovery started for this topic."
    return ""


def _build_research_history_cycle_diagnosis(cycle_diagnosis: dict) -> dict:
    if not isinstance(cycle_diagnosis, dict) or not cycle_diagnosis:
        return {"has_diagnosis": False}
    secondary = [
        format_discovery_cycle_diagnosis_label(str(item).strip())
        for item in cycle_diagnosis.get("secondary_causes") or []
        if str(item).strip()
    ]
    return {
        "has_diagnosis": True,
        "primary_cause": format_discovery_cycle_diagnosis_label(str(cycle_diagnosis.get("primary_cause") or "").strip()),
        "secondary_causes": secondary,
        "severity": str(cycle_diagnosis.get("severity") or "").strip() or "unknown",
        "explanation": str(cycle_diagnosis.get("explanation") or "").strip(),
        "recommended_next_action": format_discovery_cycle_next_action_label(
            str(cycle_diagnosis.get("recommended_next_action") or "").strip()
        ),
    }


def _build_research_history_cycle_repair(repair_plan: dict) -> dict:
    if not isinstance(repair_plan, dict) or not repair_plan:
        return {"has_repair": False}
    items = [
        item
        for item in repair_plan.get("query_repair_plan") or []
        if isinstance(item, dict)
    ]
    changed_count = sum(1 for item in items if str(item.get("action") or "").strip() == "replace_query")
    recovered_failed_area_count = sum(
        1
        for item in items
        if str(item.get("repair_reason") or "").strip().casefold().find("failed search area") >= 0
    )
    strategy = str(repair_plan.get("strategy") or "").strip()
    return {
        "has_repair": True,
        "strategy": strategy.replace("_", " ").strip().title() if strategy else "Repair plan",
        "reason": str(repair_plan.get("reason") or "").strip(),
        "rows": [
            {"label": "Changed queries", "value": str(changed_count)},
            {"label": "Kept queries", "value": str(max(len(items) - changed_count, 0))},
            {"label": "Recovered failed search areas", "value": str(recovered_failed_area_count)},
        ],
        "items": items[:3],
    }


def _build_research_history_repair_usage(round_item: dict) -> dict:
    if not isinstance(round_item, dict) or not round_item.get("used_repair_plan"):
        return {"has_used_repair": False}
    usage = round_item.get("repair_plan_usage") if isinstance(round_item.get("repair_plan_usage"), dict) else {}
    return {
        "has_used_repair": True,
        "source_round": int(usage.get("repair_plan_source_round") or 0),
        "queries_used_count": int(usage.get("queries_used_count") or len(usage.get("repair_queries_used") or [])),
        "strategy": str(usage.get("strategy") or "").replace("_", " ").strip().title() or "Repair plan",
    }


def _build_research_history_warning_title(run: SourceDiscoveryRun, provider_errors: list[str]) -> str:
    if not provider_errors:
        return ""
    if str(run.status or "").strip().lower() == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some provider queries failed."
    return "Provider results could not be fully loaded."


def _build_research_history_warning_body(run: SourceDiscoveryRun, provider_errors: list[str]) -> str:
    if not provider_errors:
        return ""
    if str(run.status or "").strip().lower() == SourceDiscoveryRun.STATUS_PARTIAL_FAILED:
        return "Some searches could not be completed. Other searches still returned results."
    return "Some searches could not be completed."


def _request_has_discovery_context(request: HttpRequest) -> bool:
    return bool(
        str(request.GET.get(DISCOVERY_CONTEXT_PARAM) or request.POST.get(DISCOVERY_CONTEXT_PARAM) or "").strip()
    )


def _request_wants_all_new_suggestions(request: HttpRequest) -> bool:
    return bool(
        str(
            request.GET.get(SHOW_ALL_NEW_SUGGESTIONS_PARAM)
            or request.POST.get(SHOW_ALL_NEW_SUGGESTIONS_PARAM)
            or ""
        ).strip()
    )


def _build_topic_workspace_url(
    topic_id: int,
    *,
    discovery_context_active: bool = False,
    show_all_new_suggestions: bool = False,
) -> str:
    base_url = reverse("topic-workspace", kwargs={"topic_id": topic_id})
    params: dict[str, str] = {}
    if discovery_context_active:
        params[DISCOVERY_CONTEXT_PARAM] = "1"
    if show_all_new_suggestions:
        params[SHOW_ALL_NEW_SUGGESTIONS_PARAM] = "1"
    if not params:
        return base_url
    return f"{base_url}?{urlencode(params)}"


def _build_source_review_summary(
    topic: Topic | None,
    candidate_records: list[dict],
) -> dict:
    if topic is None:
        return {
            "mode": TopicSourceMode.HYBRID,
            "mode_label": _get_user_facing_source_mode_label(TopicSourceMode.HYBRID),
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
    mode_label = _get_user_facing_source_mode_label(mode)

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


def _build_persisted_new_source_candidates(topic: Topic | None) -> list[dict]:
    if topic is None:
        return []

    groups = split_topic_sources(topic.sources.order_by("id"))
    candidate_records: list[dict] = []
    for source in groups.new_research_sources:
        candidate_records.append(
            {
                "title": _build_safe_saved_source_display_title(source.name, source.url),
                "url": source.url,
                "display_url": _build_compact_source_url(source.url),
                "normalized_url": source.normalized_url,
                "source_type": source.source_type or "unknown",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "persisted_source_id": source.id,
                "is_pinned": False,
                "selected": source.is_active,
                "description": _build_topic_source_description(source),
                "has_recent_article_count": False,
                "recent_article_count": None,
            }
        )
    return candidate_records


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
        for source in _build_run_eligibility(topic)["selected_sources"]
        if str(source.url).strip()
    ]


def _build_selected_source_count(topic: Topic | None) -> int:
    if topic is None:
        return 0
    return _build_run_eligibility(topic)["selected_source_count"]


def _get_mode_active_sources(topic: Topic) -> list[TopicSource]:
    active_sources = list(topic.sources.filter(is_active=True).order_by("id"))
    mode = str(topic.source_mode or TopicSourceMode.HYBRID)
    if mode == TopicSourceMode.CURATED_ONLY:
        return [source for source in active_sources if source.origin != TopicSourceOrigin.DISCOVERED]
    if mode == TopicSourceMode.DISCOVERY_ONLY:
        return [source for source in active_sources if source.origin == TopicSourceOrigin.DISCOVERED]
    return active_sources


def _build_run_eligibility(topic: Topic | None) -> dict:
    if topic is None:
        return {
            "is_eligible": False,
            "message": "Please select at least one source to run a new digest.",
            "short_message": "Needs sources",
            "selected_source_count": 0,
            "selected_sources": [],
            "active_my_source_count": 0,
            "active_research_source_count": 0,
            "source_mode": TopicSourceMode.HYBRID,
        }

    active_sources = list(topic.sources.filter(is_active=True).order_by("id"))
    active_my_sources = [source for source in active_sources if source.origin != TopicSourceOrigin.DISCOVERED]
    active_research_sources = [source for source in active_sources if source.origin == TopicSourceOrigin.DISCOVERED]
    mode = str(topic.source_mode or TopicSourceMode.HYBRID)

    if mode == TopicSourceMode.CURATED_ONLY:
        selected_sources = active_my_sources
        is_eligible = bool(selected_sources)
        message = (
            _build_selected_source_count_message(len(selected_sources))
            if is_eligible
            else "Select at least one my source before running this digest."
        )
        short_message = "" if is_eligible else "Needs a my source"
    elif mode == TopicSourceMode.DISCOVERY_ONLY:
        selected_sources = active_research_sources
        is_eligible = bool(selected_sources)
        message = (
            _build_selected_source_count_message(len(selected_sources))
            if is_eligible
            else "Find or keep at least one research source before running this digest."
        )
        short_message = "" if is_eligible else "Needs a research source"
    else:
        selected_sources = [*active_my_sources, *active_research_sources]
        has_my_sources = bool(active_my_sources)
        has_research_sources = bool(active_research_sources)
        is_eligible = has_my_sources and has_research_sources
        if is_eligible:
            message = _build_selected_source_count_message(len(selected_sources))
            short_message = ""
        elif not has_my_sources and not has_research_sources:
            message = "Please select at least one my source and one research source."
            short_message = "Needs sources"
        elif not has_my_sources:
            message = "Select at least one my source before running this digest."
            short_message = "Needs a my source"
        else:
            message = "Find or keep at least one research source before running this digest."
            short_message = "Needs a research source"

    return {
        "is_eligible": is_eligible,
        "message": message,
        "short_message": short_message,
        "selected_source_count": len(selected_sources),
        "selected_sources": selected_sources,
        "active_my_source_count": len(active_my_sources),
        "active_research_source_count": len(active_research_sources),
        "source_mode": mode,
    }


def _build_selected_source_count_message(selected_source_count: int) -> str:
    if selected_source_count <= 0:
        return "Please select at least one source to run a new digest."
    if selected_source_count == 1:
        return "1 selected source will be used in the next digest run."
    return f"{selected_source_count} selected sources will be used in the next digest run."


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


def _build_pinned_research_source_inventory(topic: Topic | None) -> list[dict]:
    if topic is None:
        return []

    groups = split_topic_sources(topic.sources.order_by("id"))
    inventory: list[dict] = []
    for source in groups.pinned_research_sources:
        inventory.append(
            {
                "id": source.id,
                "name": _build_safe_saved_source_display_title(source.name, source.url),
                "url": source.url,
                "display_url": _build_compact_source_url(source.url),
                "source_type": source.source_type or "unknown",
                "origin": source.origin,
                "origin_label": source.get_origin_display(),
                "is_active": source.is_active,
                "validation_status": source.validation_status,
                "validation_status_label": source.get_validation_status_display(),
                "last_validation_error": source.last_validation_error,
                "is_pinned": True,
            }
        )
    return inventory


def _topic_has_research_discovery_results(topic: Topic | None) -> bool:
    if topic is None:
        return False
    return topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED).exists()


def _can_find_research_sources(topic: Topic | None, *, provider_blocked: bool = False) -> bool:
    return bool(_build_topic_focus_terms(topic)) and not provider_blocked

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


def _build_compact_domain(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    domain = str(parsed.netloc or "").strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain or _build_compact_source_url(url)


def _build_used_article_history(topic: Topic) -> list[dict]:
    history = (
        UsedArticle.objects.filter(topic=topic)
        .select_related("first_used_in_run", "last_used_in_run")
        .order_by("-last_used_at", "-id")
    )
    return [
        {
            "id": article.id,
            "title": article.title,
            "article_url": article.article_url,
            "source_display": _build_compact_domain(article.source_url or article.article_url),
            "use_count": article.use_count,
            "first_used_in_run_id": article.first_used_in_run_id,
            "last_used_in_run_id": article.last_used_in_run_id or article.digest_run_id,
            "first_used_at": article.first_used_at,
            "last_used_at": article.last_used_at or article.used_at,
        }
        for article in history
    ]


def _upsert_and_build_source_candidates(
    topic: Topic,
    candidate_records: list[dict],
    *,
    prune_missing_discovered: bool = True,
) -> list[dict]:
    existing_sources = list(topic.sources.all())
    existing_by_normalized = {source.normalized_url: source for source in existing_sources}
    existing_unpinned_discovered_normalized_urls = {
        str(source.normalized_url or "").strip()
        for source in existing_sources
        if source.origin == TopicSourceOrigin.DISCOVERED
        and not source.is_pinned
        and not source.is_active
        and str(source.normalized_url or "").strip()
    }
    existing_active_unpinned_discovered_sources = [
        source
        for source in existing_sources
        if source.origin == TopicSourceOrigin.DISCOVERED and not source.is_pinned and source.is_active
    ]
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

    if prune_missing_discovered:
        pruned_normalized_urls = existing_unpinned_discovered_normalized_urls.difference(seen_discovered_normalized_urls)
        mark_removed_discovered_sources_as_seen(topic, pruned_normalized_urls)
        topic.sources.filter(
            origin=TopicSourceOrigin.DISCOVERED,
            is_pinned=False,
            is_active=False,
        ).exclude(normalized_url__in=seen_discovered_normalized_urls).delete()

    preserved_active_candidates: list[dict] = []
    for source in existing_active_unpinned_discovered_sources:
        normalized_url = str(source.normalized_url or "").strip()
        if not normalized_url or normalized_url in seen_discovered_normalized_urls:
            continue
        preserved_active_candidates.append(
            {
                "title": _build_safe_saved_source_display_title(source.name, source.url),
                "url": source.url,
                "display_url": _build_compact_source_url(source.url),
                "normalized_url": normalized_url,
                "source_type": source.source_type or "unknown",
                "candidate_origin": TopicSourceOrigin.DISCOVERED,
                "persisted_source_id": source.id,
                "is_pinned": False,
                "selected": True,
                "description": _build_topic_source_description(source),
                "has_recent_article_count": False,
                "recent_article_count": None,
            }
        )

    return [*preserved_active_candidates, *prepared_candidates]


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
    active_sources = _get_mode_active_sources(topic)
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

