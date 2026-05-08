import json
import logging
from urllib.parse import urlparse

from django.contrib.auth import get_user_model
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_GET, require_POST

from apps.digests import result_messages
from apps.digests.models import DigestRun
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.sources import get_demo_articles_for_topic
from services.sources.rss_adapter import fetch_rss_articles

from .forms import TOPIC_NAME_REQUIRED_MESSAGE, TopicInputForm

logger = logging.getLogger(__name__)
INSUFFICIENT_QUALITY_ERROR_FALLBACK = "Insufficient-quality diagnostics are available in metrics."
INSUFFICIENT_QUALITY_GENERIC_FALLBACK = "Not enough high-quality articles were available for a full digest."


@require_GET
def topic_list_view(request: HttpRequest) -> HttpResponse:
    return render(request, "digestflow/topic_list.html", _build_topic_list_context())


@require_POST
def create_topic_and_run_view(request: HttpRequest) -> HttpResponse:
    form = TopicInputForm(request.POST)
    if not form.is_valid():
        context = _build_topic_list_context(form=form)
        context["topic_form_error"] = _get_topic_form_error(form)
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
