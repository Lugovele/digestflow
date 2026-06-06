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
from services.sources.discovery_constants import (
    DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS,
    DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS,
    DISCOVERY_DECISION_MAX_ROUNDS_REACHED,
    DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR,
    DISCOVERY_DECISION_PARTIAL_TARGET_NOT_REACHED,
    DISCOVERY_DECISION_PROVIDER_UNAVAILABLE,
    DISCOVERY_DECISION_TARGET_REACHED,
)
from services.sources.discovery_repair import (
    _build_discovery_repair_plan,
    _build_next_round_repair_override,
    _build_round_repair_plan,
    _extract_repair_query_rows,
)
from services.sources.research_history_presenter import (
    _build_current_research_state,
    _build_full_research_history_copy_report,
    _build_query_performance_section,
    _build_research_history_run_entries,
    _build_search_surface_memory_section,
    _build_seen_source_history_entries,
    _build_seen_source_history_filters,
    _build_seen_source_history_pagination,
    _build_source_quality_feedback_section,
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
INSUFFICIENT_QUALITY_GENERIC_FALLBACK = "Not enough high-quality articles were available for a publish-ready post."
VISIBLE_NEW_SOURCE_LIMIT = 12
DISCOVERY_CONTEXT_PARAM = "discovery_context"
SHOW_ALL_NEW_SUGGESTIONS_PARAM = "show_all_suggestions"
ONBOARDING_COMPLETED_SESSION_KEY = "onboarding_completed"
AUTOMATIC_CREATE_POST_SOURCE_TARGET = 6


@require_GET
def app_entry_view(_request: HttpRequest) -> HttpResponse:
    return redirect("onboarding")


@require_GET
def onboarding_view(request: HttpRequest) -> HttpResponse:
    return render(
        request,
        "digestflow/onboarding.html",
        {
            "start_page_url": reverse("topic-list"),
            "complete_onboarding_url": reverse("complete-onboarding"),
        },
    )


@require_POST
def complete_onboarding_view(request: HttpRequest) -> HttpResponse:
    request.session[ONBOARDING_COMPLETED_SESSION_KEY] = True
    return redirect("topic-list")


@require_GET
def legacy_topics_redirect_view(_request: HttpRequest) -> HttpResponse:
    return redirect("topic-list")


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
def topic_list_view(request: HttpRequest, topic_id: int | None = None) -> HttpResponse:
    editing_topic = None
    if topic_id is not None:
        editing_topic = get_object_or_404(Topic, pk=topic_id, user=_get_or_create_ui_user())
    return render(
        request,
        "digestflow/topic_list.html",
        _build_topic_list_context(editing_topic=editing_topic),
    )


@require_GET
def idea_history_view(request: HttpRequest) -> HttpResponse:
    user = _get_or_create_ui_user()
    history_topics = _build_history_topics_for_user(user)
    return render(
        request,
        "digestflow/idea_history.html",
        {
            "history_topics": history_topics,
            "start_page_url": reverse("topic-list"),
        },
    )


@require_GET
def topic_setup_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    return _render_topic_setup(request, topic)


@require_POST
def continue_topic_setup_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    _mark_topic_committed(topic)
    run = _create_ui_digest_run(topic, source="setup_auto")
    return redirect("post-result", run_id=run.id)


@require_GET
def topic_workspace_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    _mark_topic_committed(topic)
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
    entry_step = str(request.POST.get("entry_step") or "").strip()
    topic_name = form.cleaned_data["topic_name"]
    source_url = str(form.cleaned_data.get("source_url") or "").strip()
    source_mode = form.cleaned_data.get("source_mode") or TopicSourceMode.HYBRID
    was_existing_topic = False
    if not topic_id:
        user = _get_or_create_ui_user()
        was_existing_topic = Topic.objects.filter(user=user, name=topic_name).exists()
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
    created_from_workspace_start = entry_step == "workspace-start" and not topic_id and not was_existing_topic
    updated_from_workspace_start = entry_step == "workspace-start" and bool(topic_id)
    if created_from_workspace_start and not discovery_requested:
        return redirect("topic-setup", topic_id=topic.id)
    if updated_from_workspace_start and not discovery_requested:
        return redirect("topic-setup", topic_id=topic.id)
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
    _mark_topic_committed(topic)
    run = _create_ui_digest_run(topic, source="web_ui_form")

    _start_topic_run(run, topic, default_source="web_ui_form")
    return redirect("run-detail", run_id=run.id)


@require_GET
def post_result_view(request: HttpRequest, run_id: int) -> HttpResponse:
    run = get_object_or_404(DigestRun.objects.select_related("topic"), pk=run_id)
    topic = run.topic
    digest = getattr(run, "digest", None)
    content_package = getattr(digest, "content_package", None) if digest else None
    provenance = _build_post_result_provenance(run, digest, content_package)

    opening_options = _normalize_post_result_options(
        content_package.hook_variants if content_package else [],
        fallback=[content_package.primary_hook()] if content_package and content_package.primary_hook() else [],
    )
    closing_options = _normalize_post_result_options(
        content_package.cta_variants if content_package else [],
        fallback=[content_package.primary_cta()] if content_package and content_package.primary_cta() else [],
    )
    final_post_text = str(getattr(content_package, "post_text", "") or "").strip()
    hashtags_text = str(content_package.hashtags_text() if content_package else "").strip()
    research_items = _build_post_result_research_items(digest)
    state = _resolve_post_result_state(run, content_package, provenance)
    stage = _build_post_result_stage(run)

    opening_default = opening_options[0] if opening_options else ""
    closing_default = closing_options[0] if closing_options else ""
    assembled_post_default = _assemble_post_result_text(
        opening_text=opening_default,
        body_text=final_post_text,
        closing_text=closing_default,
        hashtags_text=hashtags_text,
    )

    return render(
        request,
        "digestflow/post_result.html",
        {
            "topic": topic,
            "post_result_state": state,
            "post_result_heading": _build_post_result_heading(state),
            "post_result_subcopy": _build_post_result_subcopy(state),
            "post_result_detail": _build_post_result_detail(run, state),
            "stage_line_items": _build_post_result_stage_line_items(run),
            "current_stage_title": stage["title"],
            "current_stage_copy": stage["copy"],
            "start_generation_url": reverse("start-post-result", args=[run.id]),
            "retry_post_url": reverse("retry-post-result", args=[run.id]),
            "back_to_direction_url": reverse("topic-setup", args=[topic.id]),
            "review_sources_url": reverse("topic-workspace", args=[topic.id]),
            "opening_options": opening_options,
            "closing_options": closing_options,
            "selected_opening": opening_default,
            "selected_closing": closing_default,
            "final_post_text": final_post_text,
            "hashtags_text": hashtags_text,
            "assembled_post_default": assembled_post_default,
            "research_items": research_items,
            "has_research_items": bool(research_items),
            "has_opening_options": bool(opening_options),
            "has_closing_options": bool(closing_options),
            "can_auto_start": run.status == DigestRun.STATUS_PENDING,
            "show_copy_button": state == "ready" and bool(final_post_text),
        },
    )


@require_POST
def start_post_result_view(request: HttpRequest, run_id: int) -> JsonResponse:
    run = get_object_or_404(DigestRun.objects.select_related("topic"), pk=run_id)
    topic = run.topic

    if run.status == DigestRun.STATUS_PENDING:
        discovery_outcome = _run_automatic_create_post_discovery(topic)
        if discovery_outcome["usable_source_count"] < AUTOMATIC_CREATE_POST_SOURCE_TARGET:
            run.input_snapshot = {
                **(run.input_snapshot if isinstance(run.input_snapshot, dict) else {}),
                "needs_sources": True,
                "automatic_source_discovery_attempted": discovery_outcome["attempted"],
                "usable_source_count": discovery_outcome["usable_source_count"],
                "usable_source_target": AUTOMATIC_CREATE_POST_SOURCE_TARGET,
            }
            run.save(update_fields=["input_snapshot", "updated_at"])
            return JsonResponse(
                {
                    "status": run.status,
                    "redirect_url": reverse("post-result", args=[run.id]),
                    "needs_sources": True,
                }
            )

        claimed = DigestRun.objects.filter(pk=run.id, status=DigestRun.STATUS_PENDING).update(
            status=DigestRun.STATUS_COLLECTING,
            started_at=timezone.now(),
            updated_at=timezone.now(),
        )
        if claimed:
            run.refresh_from_db()
            run.input_snapshot = {
                **(run.input_snapshot if isinstance(run.input_snapshot, dict) else {}),
                "needs_sources": False,
                "automatic_source_discovery_attempted": discovery_outcome["attempted"],
                "usable_source_count": discovery_outcome["usable_source_count"],
                "usable_source_target": AUTOMATIC_CREATE_POST_SOURCE_TARGET,
            }
            run.save(update_fields=["input_snapshot", "updated_at"])
            _start_topic_run(run, topic, default_source="setup_auto", allow_demo_fallback=False)
            run.refresh_from_db()

    return JsonResponse(
        {
            "status": run.status,
            "redirect_url": reverse("post-result", args=[run.id]),
        }
    )


@require_POST
def retry_post_result_view(request: HttpRequest, run_id: int) -> HttpResponse:
    existing_run = get_object_or_404(DigestRun.objects.select_related("topic"), pk=run_id)
    topic = existing_run.topic
    _mark_topic_committed(topic)
    if _count_usable_real_sources(topic) < AUTOMATIC_CREATE_POST_SOURCE_TARGET:
        existing_run.input_snapshot = {
            **(existing_run.input_snapshot if isinstance(existing_run.input_snapshot, dict) else {}),
            "needs_sources": True,
            "usable_source_count": _count_usable_real_sources(topic),
            "usable_source_target": AUTOMATIC_CREATE_POST_SOURCE_TARGET,
        }
        existing_run.save(update_fields=["input_snapshot", "updated_at"])
        return redirect("post-result", run_id=existing_run.id)
    retry_run = _create_ui_digest_run(topic, source="post_result_retry")
    return redirect("post-result", run_id=retry_run.id)


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
    return_target = str(request.POST.get("return_to") or "").strip().lower()
    focus_terms = _parse_focus_terms(request.POST)
    raw_focus_candidate = " ".join(str(request.POST.get("focus_candidate") or "").strip().split())
    if return_target == "setup" and "focus_candidate" in request.POST and not raw_focus_candidate and not focus_terms:
        return _render_topic_setup(
            request,
            topic,
            focus_feedback={
                "level": "error",
                "message": "Enter a focus point.",
            },
            status=400,
        )
    validation_error = validate_new_focus_terms(_build_topic_focus_terms(topic), focus_terms)
    if validation_error:
        feedback_message = validation_error.message
        if return_target == "setup" and validation_error.message == FOCUS_VALIDATION_MESSAGE:
            feedback_message = "Enter a focus point."
        if return_target == "setup":
            return _render_topic_setup(
                request,
                topic,
                focus_feedback={
                    "level": "error",
                    "message": feedback_message,
                },
                focus_input_value=validation_error.term,
                status=400,
            )
        return _render_topic_source_review(
            request,
            topic,
            focus_feedback={
                "level": "error",
                "message": feedback_message,
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
    if return_target == "setup":
        return _render_topic_setup(request, topic)
    return _render_topic_source_review(request, topic)


@require_POST
def run_pipeline_view(request: HttpRequest, topic_id: int) -> HttpResponse:
    topic = get_object_or_404(Topic, pk=topic_id)
    if topic.sources.exists():
        run_eligibility = _build_run_eligibility(topic)
        if not run_eligibility["is_eligible"]:
            return redirect("topic-workspace", topic_id=topic.id)
    _mark_topic_committed(topic)
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
    _mark_topic_committed(topic)
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
        context["source_selection_error"] = "Select at least one source before generating the post."
        return render(request, "digestflow/topic_list.html", context, status=400)

    run = _create_ui_digest_run(
        topic,
        source="selected_sources_web_ui",
        selected_source_urls=[str(candidate.get("url") or "").strip() for candidate in selected_candidates if str(candidate.get("url") or "").strip()],
    )
    _start_selected_source_run(run, topic, selected_candidates, default_source="selected_sources_web_ui")
    return redirect("run-detail", run_id=run.id)


def _resolve_post_result_state(
    run: DigestRun,
    content_package,
    provenance: dict[str, object] | None = None,
) -> str:
    input_snapshot = run.input_snapshot if isinstance(run.input_snapshot, dict) else {}
    if input_snapshot.get("needs_sources"):
        return "needs_sources"
    if run.status == DigestRun.STATUS_COMPLETED and content_package and str(getattr(content_package, "post_text", "") or "").strip():
        if provenance and not provenance.get("is_safe", True):
            logger.info(
                "[DigestRun %s] post result downgraded to source recovery due to provenance: %s",
                run.id,
                ", ".join(provenance.get("reasons", [])),
            )
            return "needs_sources"
        return "ready"
    if run.status in {
        DigestRun.STATUS_FAILED,
        DigestRun.STATUS_INSUFFICIENT_QUALITY,
        DigestRun.STATUS_PARTIAL_FAILED,
    }:
        return "failed"
    return "loading"


def _build_post_result_heading(state: str) -> str:
    if state == "ready":
        return "Your post is ready"
    if state == "needs_sources":
        return "We need real sources first"
    if state == "failed":
        return "We couldn't create the post"
    return "Creating your post"


def _build_post_result_subcopy(state: str) -> str:
    if state == "ready":
        return "Pick the opening and closing you like. The preview updates automatically."
    if state == "needs_sources":
        return "PostFlow could not find enough reliable sources automatically."
    if state == "failed":
        return "Something interrupted the generation process."
    return "PostFlow is preparing your final version."


def _build_post_result_detail(run: DigestRun, state: str) -> str:
    if state == "needs_sources":
        return "Review sources first, then create the post again."
    if state != "failed":
        return ""
    if run.status == DigestRun.STATUS_INSUFFICIENT_QUALITY:
        return "There wasn't enough strong material to create the post this time."
    return ""


def _build_post_result_stage(run: DigestRun) -> dict[str, str]:
    if run.status in {DigestRun.STATUS_PENDING, DigestRun.STATUS_COLLECTING}:
        return {
            "title": "Researching sources",
            "copy": "PostFlow is collecting useful material for your post.",
        }
    if run.status == DigestRun.STATUS_PROCESSING:
        return {
            "title": "Selecting insights",
            "copy": "PostFlow is picking the most useful ideas for your post.",
        }
    if run.status in {DigestRun.STATUS_GENERATING_DIGEST, DigestRun.STATUS_PACKAGING}:
        return {
            "title": "Writing your post",
            "copy": "PostFlow is turning selected insights into a ready-to-use post.",
        }
    if run.status == DigestRun.STATUS_COMPLETED:
        return {
            "title": "Your post is ready",
            "copy": "Your final post is ready to review and copy.",
        }
    return {
        "title": "Checking your post",
        "copy": "PostFlow is preparing the next step.",
    }


def _build_post_result_stage_line_items(run: DigestRun) -> list[dict[str, str]]:
    current_step = "research"
    if run.status == DigestRun.STATUS_PROCESSING:
        current_step = "select"
    elif run.status in {DigestRun.STATUS_GENERATING_DIGEST, DigestRun.STATUS_PACKAGING}:
        current_step = "write"
    elif run.status == DigestRun.STATUS_COMPLETED:
        current_step = "ready"

    ordered_steps = [
        ("research", "Research"),
        ("select", "Select"),
        ("write", "Write"),
        ("ready", "Ready"),
    ]
    current_index = next((index for index, (key, _) in enumerate(ordered_steps) if key == current_step), 0)
    items: list[dict[str, str]] = []
    for index, (key, label) in enumerate(ordered_steps):
        state = "upcoming"
        if index < current_index:
            state = "complete"
        elif index == current_index:
            state = "active"
        items.append({"key": key, "label": label, "state": state})
    return items


def _normalize_post_result_options(values, *, fallback: list[str] | None = None) -> list[str]:
    normalized: list[str] = []
    for value in list(values or []) + list(fallback or []):
        text = str(value or "").strip()
        if not text or text in normalized:
            continue
        normalized.append(text)
    return normalized


def _assemble_post_result_text(*, opening_text: str, body_text: str, closing_text: str, hashtags_text: str) -> str:
    lines = [part for part in [opening_text.strip(), body_text.strip(), closing_text.strip(), hashtags_text.strip()] if part]
    return "\n\n".join(lines)


def _build_post_result_research_items(digest) -> list[dict[str, object]]:
    if digest is None:
        return []

    items: list[dict[str, object]] = []
    for article in digest.get_articles():
        key_points = article.get("key_points") if isinstance(article.get("key_points"), list) else []
        items.append(
            {
                "title": str(article.get("title") or "").strip() or "Research source",
                "summary": str(article.get("summary") or "").strip(),
                "url": str(article.get("url") or "").strip(),
                "key_points": [str(point).strip() for point in key_points if str(point).strip()],
            }
        )
    return items


def _build_post_result_provenance(run: DigestRun, digest, content_package) -> dict[str, object]:
    input_snapshot = run.input_snapshot if isinstance(run.input_snapshot, dict) else {}
    metrics = run.metrics if isinstance(run.metrics, dict) else {}
    digest_stage = metrics.get("digest_stage") if isinstance(metrics.get("digest_stage"), dict) else {}
    packaging_stage = metrics.get("packaging_stage") if isinstance(metrics.get("packaging_stage"), dict) else {}

    reasons: list[str] = []
    if input_snapshot.get("used_demo_source"):
        reasons.append("demo_source")
    if digest_stage.get("is_mock"):
        reasons.append("digest_mock")
    if packaging_stage.get("is_mock"):
        reasons.append("packaging_mock")
    if str(packaging_stage.get("fallback_reason") or "").strip():
        reasons.append("packaging_fallback")
    if _content_package_uses_safe_fallback(digest, content_package):
        reasons.append("safe_fallback_package")

    return {
        "is_safe": not reasons,
        "reasons": reasons,
    }


def _content_package_uses_safe_fallback(digest, content_package) -> bool:
    if digest is None or content_package is None:
        return False
    expected_text = f"{digest.title}\n\nNo post draft articles were available."
    return str(getattr(content_package, "post_text", "") or "").strip() == expected_text.strip()


def _run_automatic_create_post_discovery(topic: Topic) -> dict[str, int | bool]:
    usable_source_count = _count_usable_real_sources(topic)
    if usable_source_count >= AUTOMATIC_CREATE_POST_SOURCE_TARGET:
        return {
            "attempted": False,
            "usable_source_count": usable_source_count,
        }

    provider_resolution = resolve_configured_search_provider(topic)
    provider_diagnostics = provider_resolution.diagnostics
    provider_status = str(provider_diagnostics.get("search_provider_status") or "").strip().lower()
    provider_name = str(provider_diagnostics.get("search_provider_name") or "").strip().lower()

    attempted = False
    if (
        provider_status == "ready"
        and provider_name == "serpapi"
        and _can_find_research_sources(topic)
    ):
        attempted = True
        _run_provider_discovery_cycle(
            topic=topic,
            provider_name=provider_name,
            provider_diagnostics=dict(provider_diagnostics),
        )

    return {
        "attempted": attempted,
        "usable_source_count": _count_usable_real_sources(topic),
    }


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
        "display_result_message": _get_display_result_message(run),
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


def _get_display_result_message(run: DigestRun) -> str:
    result_message = str(getattr(run, "result_message", "") or "").strip()
    if not result_message:
        return ""

    legacy_map = {
        "Digest generated successfully.": result_messages.COMPLETED,
        "Not enough high-quality articles for a full digest.": result_messages.INSUFFICIENT_QUALITY,
        "Source processed, but no usable articles were found.": result_messages.SOURCE_NO_USABLE_ARTICLES,
        "Digest run failed before completion.": result_messages.FAILED,
        "Digest generated, but content packaging failed.": result_messages.PARTIAL_FAILED,
    }
    return legacy_map.get(result_message, result_message)


def _get_insufficient_quality_message(run: DigestRun) -> str:
    result_message = _get_display_result_message(run)
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
            ("Selected for post draft", _display_metric_value(ranking_stage.get("selected_for_prompt"))),
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
        return "Post draft generation skipped because too few articles passed quality validation."
    if ranking_stage.get("selected_for_prompt"):
        return "Post draft generation proceeded with the selected articles."
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
        f"Post idea: {run.topic.name}",
        f"Status: {run.status}",
        (
            "Result: insufficient quality"
            if is_insufficient_quality
            else f"Result: {_get_display_result_message(run) or '-'}"
        ),
        (
            "Error: see diagnostics"
            if is_insufficient_quality and display_error_message
            else f"Error: {display_error_message or '-'}"
        ),
        "",
        "Post draft",
        f"Title: {digest_payload.get('title') or '-'}",
        f"Post draft articles: {len(digest_payload.get('articles') or [])}",
    ]

    digest_articles = digest_payload.get("articles") or []
    if digest_articles:
        lines.append("Post draft article summaries:")
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
        lines.append("Post draft article summaries: none")

    lines.extend(
        [
            "",
            "Publish-ready post",
            f"Validation status: {validation_status}",
            f"Publish-ready post: {getattr(content_package, 'post_text', '') or '-'}",
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
            f"Post draft provider: {digest_provider or '-'}",
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
    editing_topic: Topic | None = None,
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
    history_topics = _build_history_topics_for_user(user, topics=topics)
    run_eligibility = _build_run_eligibility(discovered_topic)
    research_provider_state = _build_research_provider_state(discovered_topic)
    hidden_new_source_candidate_count = max(0, len(total_new_source_candidates) - len(visible_new_source_candidates))
    has_research_discovery_results = _topic_has_research_discovery_results(discovered_topic)
    topic_form = form
    if topic_form is None:
        if editing_topic is not None:
            topic_form = TopicInputForm(
                initial={
                    "topic_name": editing_topic.name,
                    "source_mode": editing_topic.source_mode or TopicSourceMode.HYBRID,
                }
            )
        else:
            topic_form = TopicInputForm()
    return {
        "topics": topics,
        "history_topics": history_topics,
        "recent_history_topics": history_topics[:3],
        "idea_history_url": reverse("idea-history"),
        "recent_runs": recent_runs,
        "topic_form": topic_form,
        "editing_topic": editing_topic,
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
        "ready_post_history": (
            _build_ready_post_history_for_topics([discovered_topic.id]).get(discovered_topic.id, [])
            if discovered_topic is not None
            else []
        ),
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
        body = "PostFlow can still use your sources, but automatic research is turned off."
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


def _build_ready_post_history_for_topics(topic_ids: list[int]) -> dict[int, list[dict]]:
    normalized_topic_ids = [int(topic_id) for topic_id in topic_ids if topic_id]
    if not normalized_topic_ids:
        return {}

    ready_post_history_by_topic_id: dict[int, list[dict]] = {topic_id: [] for topic_id in normalized_topic_ids}
    candidate_runs = (
        DigestRun.objects.filter(topic_id__in=normalized_topic_ids, status=DigestRun.STATUS_COMPLETED)
        .select_related("topic", "digest__content_package")
        .order_by("-created_at")
    )

    for run in candidate_runs:
        digest = getattr(run, "digest", None)
        content_package = getattr(digest, "content_package", None) if digest else None
        provenance = _build_post_result_provenance(run, digest, content_package)
        if _resolve_post_result_state(run, content_package, provenance) != "ready":
            continue

        post_text = str(getattr(content_package, "post_text", "") or "").strip()
        if not post_text:
            continue

        created_at = (
            getattr(content_package, "created_at", None)
            or getattr(digest, "generated_at", None)
            or run.finished_at
            or run.created_at
        )
        ready_post_history_by_topic_id.setdefault(run.topic_id, []).append(
            {
                "run_id": run.id,
                "created_display": _format_recent_run_time(created_at) if created_at else "",
                "preview": _build_ready_post_preview(post_text),
                "status_label": "Ready post",
                "open_url": reverse("post-result", args=[run.id]),
            }
        )

    return ready_post_history_by_topic_id


def _build_ready_post_preview(post_text: str, *, limit: int = 180) -> str:
    normalized_text = " ".join(str(post_text or "").split())
    if len(normalized_text) <= limit:
        return normalized_text
    truncated = normalized_text[:limit].rsplit(" ", 1)[0].strip()
    if not truncated:
        truncated = normalized_text[:limit].strip()
    return f"{truncated}..."


def _build_ready_post_count_label(count: int) -> str:
    if count <= 0:
        return "No ready posts"
    if count == 1:
        return "1 ready post"
    return f"{count} ready posts"


def _build_history_topics_for_user(user, *, topics: list[Topic] | None = None) -> list[dict]:
    topic_list = topics
    if topic_list is None:
        topic_list = list(
            Topic.objects.filter(user=user)
            .order_by("display_order", "name")
            .prefetch_related("sources")
        )
        for topic in topic_list:
            topic.source_count = sum(1 for source in topic.sources.all() if source.origin != TopicSourceOrigin.DISCOVERED)
            topic.research_source_count = sum(1 for source in topic.sources.all() if source.origin == TopicSourceOrigin.DISCOVERED)
            topic.active_source_count = sum(
                1 for source in topic.sources.all() if source.is_active and source.origin != TopicSourceOrigin.DISCOVERED
            )
            topic.run_eligibility = _build_run_eligibility(topic)
            topic.legacy_source_display = _build_legacy_source_display(topic)

    ready_post_history_by_topic_id = _build_ready_post_history_for_topics([topic.id for topic in topic_list])
    history_runs = list(
        DigestRun.objects.filter(topic__user=user)
        .select_related("topic")
        .order_by("-created_at")
    )
    history_runs_by_topic_id: dict[int, list[DigestRun]] = {}
    for run in history_runs:
        run.display_time = _format_recent_run_time(run.created_at)
        history_runs_by_topic_id.setdefault(run.topic_id, []).append(run)

    history_topics = []
    for topic in topic_list:
        topic_runs = history_runs_by_topic_id.get(topic.id, [])
        if not _topic_is_committed_for_history(topic, topic_runs):
            continue
        latest_run = topic_runs[0] if topic_runs else None
        latest_activity_at = latest_run.created_at if latest_run is not None else topic.updated_at
        latest_activity_display = _format_recent_run_time(latest_activity_at)
        history_topics.append(
            {
                "topic": topic,
                "runs": topic_runs[:6],
                "run_count": len(topic_runs),
                "latest_run": latest_run,
                "latest_run_display_time": latest_run.display_time if latest_run else "",
                "latest_activity_at": latest_activity_at,
                "latest_activity_display": latest_activity_display,
                "status": _build_idea_history_status(latest_run),
                "ready_post_count": len(ready_post_history_by_topic_id.get(topic.id, [])),
                "ready_post_count_label": _build_ready_post_count_label(
                    len(ready_post_history_by_topic_id.get(topic.id, []))
                ),
            }
        )

    history_topics.sort(
        key=lambda item: (
            item["latest_activity_at"],
            item["topic"].id,
        ),
        reverse=True,
    )
    return history_topics


def _topic_is_committed_for_history(topic: Topic, topic_runs: list[DigestRun]) -> bool:
    if topic.committed_at is not None:
        return True
    if topic_runs:
        return True
    if str(topic.source_url or "").strip():
        return True
    prefetched_sources = getattr(topic, "_prefetched_objects_cache", {}).get("sources")
    if prefetched_sources is not None:
        return bool(prefetched_sources)
    return topic.sources.exists()


def _mark_topic_committed(topic: Topic) -> None:
    if topic.committed_at is not None:
        return
    topic.committed_at = timezone.now()
    topic.save(update_fields=["committed_at", "updated_at"])


def _build_idea_history_status(latest_run: DigestRun | None) -> dict:
    if latest_run is None:
        return {"label": "Idea", "tone": "neutral"}

    status = str(latest_run.status or "").strip().lower()
    if status in {DigestRun.STATUS_PENDING}:
        return {"label": "Waiting", "tone": "neutral"}
    if status in {DigestRun.STATUS_COLLECTING, DigestRun.STATUS_PROCESSING}:
        return {"label": "Searching", "tone": "info"}
    if status in {DigestRun.STATUS_GENERATING_DIGEST, DigestRun.STATUS_PACKAGING}:
        return {"label": "Generating", "tone": "info"}
    if status == DigestRun.STATUS_COMPLETED:
        return {"label": "Post ready", "tone": "success"}
    if status in {DigestRun.STATUS_INSUFFICIENT_QUALITY, DigestRun.STATUS_PARTIAL_FAILED, DigestRun.STATUS_FAILED}:
        return {"label": "Needs attention", "tone": "warning"}
    return {"label": "Idea", "tone": "neutral"}


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


def _render_topic_setup(
    request: HttpRequest,
    topic: Topic,
    *,
    status: int = 200,
    focus_feedback: dict | None = None,
    focus_input_value: str = "",
) -> HttpResponse:
    focus_terms = _build_topic_focus_terms(topic)
    ready_post_history = _build_ready_post_history_for_topics([topic.id]).get(topic.id, [])
    return render(
        request,
        "digestflow/topic_setup.html",
        {
            "topic": topic,
            "focus_terms": focus_terms,
            "focus_chip_terms": _build_topic_setup_focus_chip_terms(focus_terms),
            "focus_feedback": focus_feedback,
            "focus_input_value": focus_input_value,
            "review_sources_url": reverse("topic-workspace", args=[topic.id]),
            "continue_setup_url": reverse("continue-topic-setup", args=[topic.id]),
            "update_focus_url": reverse("update-topic-focus", args=[topic.id]),
            "ready_post_history": ready_post_history,
        },
        status=status,
    )


def _build_topic_focus_terms(topic: Topic | None) -> list[str]:
    if topic is None:
        return []
    raw_terms = topic.keywords if isinstance(topic.keywords, list) else []
    return clean_focus_terms(raw_terms)


def _build_topic_setup_focus_chip_terms(focus_terms: list[str]) -> list[dict[str, str]]:
    chip_terms: list[dict[str, str]] = []
    for term in focus_terms:
        chip_terms.append(
            {
                "value": term,
                "label": _build_topic_setup_focus_chip_label(term),
            }
        )
    return chip_terms


def _build_topic_setup_focus_chip_label(term: str) -> str:
    normalized = " ".join(str(term or "").strip().split())
    if not normalized:
        return ""

    exact_display_labels = {
        "hands-on activities for early childhood education": "hands-on activities",
        "language development resources for toddlers": "language development",
        "music and movement activities for young children": "music and movement",
    }
    exact_match = exact_display_labels.get(normalized.casefold())
    if exact_match:
        return exact_match

    if len(normalized.split()) <= 4:
        return normalized

    shortened = re.split(r"\s+(?:for|with|about|around|to)\s+", normalized, maxsplit=1, flags=re.IGNORECASE)[0]
    if shortened:
        normalized = shortened

    words = normalized.split()
    trailing_generic_terms = {
        "activities",
        "activity",
        "resources",
        "resource",
        "tools",
        "tool",
        "ideas",
        "idea",
        "strategies",
        "strategy",
        "examples",
        "example",
    }
    if len(words) > 2 and words[-1].casefold() in trailing_generic_terms:
        words = words[:-1]

    if len(words) > 4:
        words = words[:4]
    return " ".join(words)


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
                    decision=DISCOVERY_DECISION_PROVIDER_UNAVAILABLE,
                    rounds=[],
                ),
            },
        )
        return _build_persisted_new_source_candidates(topic), {
            "title": "Source search is temporarily unavailable",
            "body": (
                "PostFlow could not connect to the search provider. Existing suggestions were kept."
                if _count_existing_new_suggestions(topic) > 0
                else "PostFlow could not connect to the search provider. Please try again later."
            ),
            "provider_name": provider_name or str(provider_diagnostics.get("search_provider_name") or "").strip(),
            "execution_status": DISCOVERY_DECISION_PROVIDER_UNAVAILABLE,
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
    decision = DISCOVERY_DECISION_PARTIAL_TARGET_NOT_REACHED
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
            decision = DISCOVERY_DECISION_TARGET_REACHED
            break
        if round_result["provider_unavailable"]:
            decision = DISCOVERY_DECISION_PROVIDER_UNAVAILABLE
            break
        if round_index >= DISCOVERY_CYCLE_MAX_IMMEDIATE_ROUNDS:
            decision = DISCOVERY_DECISION_MAX_ROUNDS_REACHED
            break
        next_round_query_plan, next_round_repair_usage, continuation_decision = _build_next_round_repair_override(
            topic=topic,
            round_summary=round_results[-1],
            prior_rounds=round_results,
            query_limit=int(getattr(round_result.get("discovery_run"), "query_count", 0) or 0),
        )
        if next_round_query_plan is None:
            decision = continuation_decision or DISCOVERY_DECISION_PARTIAL_NO_USABLE_REPAIR
            break

    final_candidate_records = _finalize_discovery_cycle_candidate_records(
        topic=topic,
        accumulated_new_candidates=accumulated_new_candidates,
        prune_missing_discovered=(decision != DISCOVERY_DECISION_PROVIDER_UNAVAILABLE),
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

    if decision == DISCOVERY_DECISION_PROVIDER_UNAVAILABLE:
        title = "Source search is temporarily unavailable"
        if existing_new_suggestion_count > 0:
            body = "PostFlow could not connect to the search provider. Existing suggestions were kept."
        else:
            body = "PostFlow could not connect to the search provider. Please try again later."
        execution_status = DISCOVERY_DECISION_PROVIDER_UNAVAILABLE
    elif decision == DISCOVERY_DECISION_TARGET_REACHED:
        title = "Source discovery completed"
        body = (
            f"Target reached: {accumulated_visible_suggestions} new source suggestion"
            f"{'s' if accumulated_visible_suggestions != 1 else ''} "
            f"after {round_count} search round{'s' if round_count != 1 else ''}."
        )
        if provider_error_count > 0:
            body = f"{body} Some provider queries failed."
        execution_status = "completed"
    elif provider_error_count > 0 and accumulated_visible_suggestions > 0:
        title = "Source discovery partially completed"
        body = (
            f"Some searches could not be completed. {accumulated_visible_suggestions} new source suggestion"
            f"{'s' if accumulated_visible_suggestions != 1 else ''} "
            f"{'are' if accumulated_visible_suggestions != 1 else 'is'} still available"
            f"{f' after {round_count} search rounds' if round_count > 1 else ''}."
        )
        execution_status = "failed"
    else:
        title = "Source discovery partially completed"
        if accumulated_visible_suggestions > 0:
            body = (
                f"Found {accumulated_visible_suggestions} new source suggestion"
                f"{'s' if accumulated_visible_suggestions != 1 else ''} after {round_count} search rounds. "
                f"PostFlow could not reach the {DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS}-source target "
                f"with the current search strategy."
            )
        else:
            body = (
                f"PostFlow could not reach the {DISCOVERY_CYCLE_TARGET_VISIBLE_NEW_SUGGESTIONS}-source target "
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


def _count_usable_real_sources(topic: Topic) -> int:
    usable_sources = _get_mode_active_sources(topic)
    normalized_urls = {
        str(source.normalized_url or "").strip()
        for source in usable_sources
        if str(source.normalized_url or "").strip()
    }
    if str(topic.source_url or "").strip():
        normalized_legacy_source = classify_source_url(str(topic.source_url or "").strip()).normalized_url
        if normalized_legacy_source and normalized_legacy_source not in normalized_urls:
            normalized_urls.add(normalized_legacy_source)
    return len(normalized_urls)


def _topic_has_real_generation_inputs(topic: Topic) -> bool:
    return _count_usable_real_sources(topic) > 0


def _build_run_eligibility(topic: Topic | None) -> dict:
    if topic is None:
        return {
            "is_eligible": False,
            "message": "Please select at least one source to generate a new post.",
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
            else "Select at least one source for this post before generating it."
        )
        short_message = "" if is_eligible else "Needs a my source"
    elif mode == TopicSourceMode.DISCOVERY_ONLY:
        selected_sources = active_research_sources
        is_eligible = bool(selected_sources)
        message = (
            _build_selected_source_count_message(len(selected_sources))
            if is_eligible
            else "Find or keep at least one research source before generating this post."
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
            message = "Please select at least one source for this post and one research source."
            short_message = "Needs sources"
        elif not has_my_sources:
            message = "Select at least one source for this post before generating it."
            short_message = "Needs a my source"
        else:
            message = "Find or keep at least one research source before generating this post."
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
        return "Please select at least one source to generate a new post."
    if selected_source_count == 1:
        return "1 selected source will be used in the next post run."
    return f"{selected_source_count} selected sources will be used in the next post run."


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
            or "Source added and saved for this post idea. It will be used when generating the post."
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


def _start_topic_run(run: DigestRun, topic: Topic, default_source: str, *, allow_demo_fallback: bool = True) -> None:
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

    if not allow_demo_fallback:
        _mark_run_failed_for_empty_selected_sources(run, [])
        return

    raw_items = get_demo_articles_for_topic(topic.name)
    run.input_snapshot = {
        **run.input_snapshot,
        "source": default_source,
        "source_url": "",
        "raw_items_count": len(raw_items),
        "used_demo_source": True,
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


