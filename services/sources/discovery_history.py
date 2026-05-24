from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from urllib.parse import urlparse

from django.utils import timezone

from apps.digests.models import SourceDiscoveryHistory, SourceDiscoveryRun
from apps.topics.models import Topic, TopicSource, TopicSourceOrigin
from services.sources.candidates import SourceCandidateStatus
from services.sources.detector import classify_source_url


@dataclass(frozen=True)
class RecordedDiscoveryHistory:
    discovery_run: SourceDiscoveryRun
    history_by_normalized_url: dict[str, SourceDiscoveryHistory]


_STATUS_STRENGTH = {
    SourceDiscoveryHistory.STATUS_SEEN: 0,
    SourceDiscoveryHistory.STATUS_SHOWN: 1,
    SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY: 2,
    SourceDiscoveryHistory.STATUS_REMOVED_BY_USER: 3,
    SourceDiscoveryHistory.STATUS_KEPT: 4,
}


def build_topic_known_url_set(topic: Topic) -> set[str]:
    known_urls = {
        str(value or "").strip()
        for value in topic.sources.values_list("normalized_url", flat=True)
        if str(value or "").strip()
    }
    known_urls.update(
        str(value or "").strip()
        for value in topic.source_discovery_history.values_list("normalized_url", flat=True)
        if str(value or "").strip()
    )
    return known_urls


def record_source_discovery_run_started(
    *,
    topic: Topic,
    provider_name: str,
    diagnostics: dict,
) -> SourceDiscoveryRun:
    return SourceDiscoveryRun.objects.create(
        user=topic.user,
        topic=topic,
        provider_name=str(provider_name or "").strip(),
        status=SourceDiscoveryRun.STATUS_STARTED,
        search_recency_months=int(diagnostics.get("search_recency_months") or 1),
        search_time_filter=str(
            diagnostics.get("search_time_filter") or diagnostics.get("provider_tbs") or ""
        ).strip(),
        diagnostics=dict(diagnostics or {}),
    )


def finalize_source_discovery_run(
    discovery_run: SourceDiscoveryRun,
    *,
    status: str,
    diagnostics: dict,
    known_url_count: int = 0,
    accepted_count: int = 0,
    rejected_count: int = 0,
    new_suggestions_count: int = 0,
    already_known_count: int = 0,
) -> SourceDiscoveryRun:
    discovery_run.provider_name = str(
        diagnostics.get("provider_name") or discovery_run.provider_name or ""
    ).strip()
    discovery_run.status = status
    discovery_run.completed_at = timezone.now()
    discovery_run.search_recency_months = int(
        diagnostics.get("search_recency_months") or discovery_run.search_recency_months or 1
    )
    discovery_run.search_time_filter = str(
        diagnostics.get("search_time_filter")
        or diagnostics.get("provider_tbs")
        or discovery_run.search_time_filter
        or ""
    ).strip()
    discovery_run.query_count = int(diagnostics.get("query_count") or 0)
    discovery_run.provider_result_count = int(diagnostics.get("raw_result_count") or 0)
    discovery_run.known_url_count = int(known_url_count or 0)
    discovery_run.accepted_count = int(accepted_count or 0)
    discovery_run.rejected_count = int(rejected_count or 0)
    discovery_run.new_suggestions_count = int(new_suggestions_count or 0)
    discovery_run.already_known_count = int(already_known_count or 0)
    discovery_run.diagnostics = dict(diagnostics or {})
    discovery_run.save(
        update_fields=[
            "provider_name",
            "status",
            "completed_at",
            "search_recency_months",
            "search_time_filter",
            "query_count",
            "provider_result_count",
            "known_url_count",
            "accepted_count",
            "rejected_count",
            "new_suggestions_count",
            "already_known_count",
            "diagnostics",
            "updated_at",
        ]
    )
    return discovery_run


def record_source_discovery_history(
    *,
    topic: Topic,
    discovery_run: SourceDiscoveryRun,
    source_research_result,
    shown_candidates: list[dict],
    known_normalized_urls: set[str],
) -> RecordedDiscoveryHistory:
    shown_by_normalized = {
        str(candidate.get("normalized_url") or "").strip(): candidate
        for candidate in shown_candidates
        if str(candidate.get("normalized_url") or "").strip()
    }
    now = timezone.now()

    evaluated_by_normalized = {
        candidate.normalized_url: candidate
        for candidate in source_research_result.evaluated_candidates
        if str(candidate.normalized_url or "").strip()
    }
    existing_history_by_normalized = {
        item.normalized_url: item
        for item in topic.source_discovery_history.filter(
            normalized_url__in=list(evaluated_by_normalized.keys())
        )
    }

    for normalized_url, candidate in evaluated_by_normalized.items():
        existing_history = existing_history_by_normalized.get(normalized_url)
        was_known = normalized_url in known_normalized_urls
        proposed_status = _proposed_history_status(candidate, shown=normalized_url in shown_by_normalized)
        next_status = _merge_history_status(
            existing_history.status if existing_history is not None else "",
            proposed_status,
        )
        last_run_outcome = _build_last_run_outcome(
            candidate=candidate,
            shown=normalized_url in shown_by_normalized,
            was_known=was_known,
            existing_status=existing_history.status if existing_history is not None else "",
        )
        shown_candidate = shown_by_normalized.get(normalized_url)
        source_url = str(candidate.url or "").strip()
        published_date = _parse_detected_publication_date(
            str(candidate.diagnostics.get("detected_publication_date") or "").strip()
        )
        source_domain = _extract_domain(normalized_url or source_url)
        provider_name = str(candidate.diagnostics.get("provider_name") or "").strip()
        query_text = str(candidate.diagnostics.get("query") or "").strip()
        topic_source_id = shown_candidate.get("persisted_source_id") if shown_candidate is not None else None

        if existing_history is None:
            history_item = SourceDiscoveryHistory.objects.create(
                user=topic.user,
                topic=topic,
                discovery_run=discovery_run,
                topic_source_id=topic_source_id,
                normalized_url=normalized_url,
                url=source_url,
                title=str(candidate.title or "").strip(),
                snippet=str(candidate.snippet or "").strip(),
                domain=source_domain,
                provider_name=provider_name,
                query_text=query_text,
                status=next_status,
                last_run_outcome=last_run_outcome,
                source_content_type=str(candidate.diagnostics.get("source_content_type") or "").strip(),
                quality_score=float(candidate.diagnostics.get("quality_score") or 0.0),
                substance_score=float(candidate.diagnostics.get("substance_score") or 0.0),
                commercial_intent_score=float(candidate.diagnostics.get("commercial_intent_score") or 0.0),
                quality_rejection_reason=str(candidate.diagnostics.get("quality_rejection_reason") or "").strip(),
                freshness_status=str(candidate.diagnostics.get("freshness_status") or "").strip(),
                detected_publication_date=published_date,
                detected_publication_year=_safe_year(candidate.diagnostics.get("detected_publication_year")),
                first_seen_at=now,
                last_seen_at=now,
                seen_count=1,
                created_topic_source=bool(topic_source_id),
                diagnostics=dict(candidate.diagnostics or {}),
            )
            existing_history_by_normalized[normalized_url] = history_item
            continue

        existing_history.discovery_run = discovery_run
        if topic_source_id:
            existing_history.topic_source_id = topic_source_id
            existing_history.created_topic_source = True
        existing_history.url = source_url or existing_history.url
        existing_history.title = str(candidate.title or existing_history.title or "").strip()
        existing_history.snippet = str(candidate.snippet or existing_history.snippet or "").strip()
        existing_history.domain = source_domain or existing_history.domain
        existing_history.provider_name = provider_name or existing_history.provider_name
        existing_history.query_text = query_text or existing_history.query_text
        existing_history.status = next_status
        existing_history.last_run_outcome = last_run_outcome
        existing_history.source_content_type = str(
            candidate.diagnostics.get("source_content_type") or existing_history.source_content_type or ""
        ).strip()
        existing_history.quality_score = float(candidate.diagnostics.get("quality_score") or 0.0)
        existing_history.substance_score = float(candidate.diagnostics.get("substance_score") or 0.0)
        existing_history.commercial_intent_score = float(
            candidate.diagnostics.get("commercial_intent_score") or 0.0
        )
        existing_history.quality_rejection_reason = str(
            candidate.diagnostics.get("quality_rejection_reason") or ""
        ).strip()
        existing_history.freshness_status = str(candidate.diagnostics.get("freshness_status") or "").strip()
        existing_history.detected_publication_date = published_date
        existing_history.detected_publication_year = _safe_year(candidate.diagnostics.get("detected_publication_year"))
        existing_history.last_seen_at = now
        existing_history.seen_count += 1
        existing_history.diagnostics = dict(candidate.diagnostics or {})
        existing_history.save(
            update_fields=[
                "discovery_run",
                "topic_source",
                "url",
                "title",
                "snippet",
                "domain",
                "provider_name",
                "query_text",
                "status",
                "last_run_outcome",
                "source_content_type",
                "quality_score",
                "substance_score",
                "commercial_intent_score",
                "quality_rejection_reason",
                "freshness_status",
                "detected_publication_date",
                "detected_publication_year",
                "last_seen_at",
                "seen_count",
                "created_topic_source",
                "diagnostics",
                "updated_at",
            ]
        )

    return RecordedDiscoveryHistory(
        discovery_run=discovery_run,
        history_by_normalized_url=existing_history_by_normalized,
    )


def update_history_for_kept_source(source: TopicSource) -> None:
    if source.origin != TopicSourceOrigin.DISCOVERED:
        return
    history_item = _get_history_for_source(source)
    if history_item is None:
        return
    history_item.status = SourceDiscoveryHistory.STATUS_KEPT
    history_item.topic_source = source
    history_item.created_topic_source = True
    history_item.save(update_fields=["status", "topic_source", "created_topic_source", "updated_at"])


def update_history_for_removed_source(source: TopicSource) -> None:
    if source.origin != TopicSourceOrigin.DISCOVERED:
        return
    history_item = _get_history_for_source(source)
    if history_item is None:
        return
    history_item.status = SourceDiscoveryHistory.STATUS_REMOVED_BY_USER
    history_item.topic_source = None
    history_item.save(update_fields=["status", "topic_source", "updated_at"])


def _get_history_for_source(source: TopicSource) -> SourceDiscoveryHistory | None:
    normalized_url = str(source.normalized_url or "").strip()
    if not normalized_url:
        normalized_url = classify_source_url(str(source.url or "").strip()).normalized_url
    return SourceDiscoveryHistory.objects.filter(topic=source.topic, normalized_url=normalized_url).first()


def _proposed_history_status(candidate, *, shown: bool) -> str:
    if shown:
        return SourceDiscoveryHistory.STATUS_SHOWN
    if _is_quality_rejected(candidate):
        return SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY
    return SourceDiscoveryHistory.STATUS_SEEN


def _merge_history_status(existing_status: str, proposed_status: str) -> str:
    existing_value = str(existing_status or "").strip()
    if not existing_value:
        return proposed_status
    if _STATUS_STRENGTH.get(existing_value, -1) >= _STATUS_STRENGTH.get(proposed_status, -1):
        return existing_value
    return proposed_status


def _build_last_run_outcome(*, candidate, shown: bool, was_known: bool, existing_status: str) -> str:
    if shown:
        if was_known:
            if existing_status == SourceDiscoveryHistory.STATUS_REMOVED_BY_USER:
                return SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED
            if existing_status == SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY:
                return SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED
            return SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN
        return SourceDiscoveryHistory.OUTCOME_NEW_SHOWN

    if candidate.diagnostics.get("duplicate_url"):
        return SourceDiscoveryHistory.OUTCOME_DUPLICATE_URL
    if candidate.diagnostics.get("duplicate_hostname"):
        return SourceDiscoveryHistory.OUTCOME_DUPLICATE_DOMAIN
    if _is_stale_rejected(candidate):
        return SourceDiscoveryHistory.OUTCOME_STALE_REJECTED
    if _is_commercial_rejected(candidate):
        return SourceDiscoveryHistory.OUTCOME_COMMERCIAL_REJECTED
    if _is_quality_rejected(candidate):
        return SourceDiscoveryHistory.OUTCOME_QUALITY_REJECTED
    if was_known:
        if existing_status == SourceDiscoveryHistory.STATUS_REMOVED_BY_USER:
            return SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED
        if existing_status == SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY:
            return SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REJECTED
        return SourceDiscoveryHistory.OUTCOME_ALREADY_KNOWN
    return SourceDiscoveryHistory.OUTCOME_NONE


def _is_quality_rejected(candidate) -> bool:
    return candidate.status in {
        SourceCandidateStatus.REJECTED,
        SourceCandidateStatus.LOW_RELEVANCE,
        SourceCandidateStatus.WEAK_CONTENT,
    } or bool(
        str(candidate.diagnostics.get("quality_rejection_reason") or "").strip()
        or str(candidate.diagnostics.get("freshness_rejection_reason") or "").strip()
    )


def _is_stale_rejected(candidate) -> bool:
    return str(candidate.diagnostics.get("freshness_status") or "").strip() in {"stale", "very_stale"}


def _is_commercial_rejected(candidate) -> bool:
    reason = str(candidate.diagnostics.get("quality_rejection_reason") or "").strip().lower()
    return any(
        phrase in reason
        for phrase in (
            "commercial service-page",
            "product/demo/pricing intent",
            "promotional language",
        )
    )


def _extract_domain(url: str) -> str:
    return str(urlparse(str(url or "").strip()).netloc or "").strip().lower()


def _parse_detected_publication_date(value: str) -> date | None:
    raw_value = str(value or "").strip()
    if not raw_value:
        return None
    try:
        return date.fromisoformat(raw_value)
    except ValueError:
        return None


def _safe_year(value) -> int | None:
    try:
        year_value = int(value)
    except (TypeError, ValueError):
        return None
    if year_value <= 0:
        return None
    return year_value
