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

_PASSIVE_STATUS_STRENGTH = {
    SourceDiscoveryHistory.STATUS_SEEN: 0,
    SourceDiscoveryHistory.STATUS_SHOWN: 1,
    SourceDiscoveryHistory.STATUS_KEPT: 2,
    SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY: 3,
    SourceDiscoveryHistory.STATUS_REMOVED_BY_USER: 4,
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


def build_topic_history_by_normalized_url(topic: Topic) -> dict[str, SourceDiscoveryHistory]:
    return {
        item.normalized_url: item
        for item in topic.source_discovery_history.all()
        if str(item.normalized_url or "").strip()
    }


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
    history_item = _get_or_create_history_for_source(
        source,
        default_status=SourceDiscoveryHistory.STATUS_KEPT,
        default_last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
    )
    history_item.status = SourceDiscoveryHistory.STATUS_KEPT
    history_item.topic_source = source
    history_item.created_topic_source = True
    history_item.save(update_fields=["status", "topic_source", "created_topic_source", "updated_at"])


def update_history_for_removed_source(source: TopicSource) -> None:
    if source.origin != TopicSourceOrigin.DISCOVERED:
        return
    history_item = _get_or_create_history_for_source(
        source,
        default_status=SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
        default_last_run_outcome=SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED,
    )
    history_item.status = SourceDiscoveryHistory.STATUS_REMOVED_BY_USER
    history_item.last_run_outcome = SourceDiscoveryHistory.OUTCOME_PREVIOUSLY_REMOVED
    history_item.topic_source = None
    history_item.save(update_fields=["status", "last_run_outcome", "topic_source", "updated_at"])


def update_history_for_unpinned_source(source: TopicSource) -> None:
    if source.origin != TopicSourceOrigin.DISCOVERED:
        return
    history_item = _get_or_create_history_for_source(
        source,
        default_status=SourceDiscoveryHistory.STATUS_SHOWN,
        default_last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
    )
    if history_item.status in {
        SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
        SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
    }:
        return
    history_item.status = SourceDiscoveryHistory.STATUS_SHOWN
    history_item.topic_source = source
    history_item.created_topic_source = True
    history_item.save(update_fields=["status", "topic_source", "created_topic_source", "updated_at"])


def mark_removed_discovered_sources_as_seen(topic: Topic, normalized_urls: set[str]) -> None:
    normalized_values = {str(value or "").strip() for value in normalized_urls if str(value or "").strip()}
    if not normalized_values:
        return
    history_rows = list(
        topic.source_discovery_history.filter(normalized_url__in=list(normalized_values))
    )
    for history_item in history_rows:
        update_fields: list[str] = []
        if history_item.status == SourceDiscoveryHistory.STATUS_SHOWN:
            history_item.status = SourceDiscoveryHistory.STATUS_SEEN
            update_fields.append("status")
        if history_item.topic_source_id is not None:
            history_item.topic_source = None
            update_fields.append("topic_source")
        if update_fields:
            update_fields.append("updated_at")
            history_item.save(update_fields=update_fields)


def sync_topic_discovered_sources_into_history(topic: Topic) -> None:
    for source in topic.sources.filter(origin=TopicSourceOrigin.DISCOVERED):
        default_status = (
            SourceDiscoveryHistory.STATUS_KEPT if source.is_pinned else SourceDiscoveryHistory.STATUS_SHOWN
        )
        history_item = _get_or_create_history_for_source(
            source,
            default_status=default_status,
            default_last_run_outcome=SourceDiscoveryHistory.OUTCOME_NEW_SHOWN,
        )
        next_status = _merge_history_status_for_sync(history_item.status, default_status)
        update_fields: list[str] = []
        if history_item.status != next_status:
            history_item.status = next_status
            update_fields.append("status")
        if history_item.topic_source_id != source.id:
            history_item.topic_source = source
            update_fields.append("topic_source")
        if not history_item.created_topic_source:
            history_item.created_topic_source = True
            update_fields.append("created_topic_source")
        if update_fields:
            update_fields.append("updated_at")
            history_item.save(update_fields=update_fields)


def _get_history_for_source(source: TopicSource) -> SourceDiscoveryHistory | None:
    canonical_normalized_url = _get_canonical_normalized_url_for_source(source)
    matching_rows = _find_matching_history_rows_for_source(source)
    if not matching_rows:
        return None

    primary_row = _select_primary_history_row(matching_rows)
    duplicate_rows = [row for row in matching_rows if row.pk != primary_row.pk]
    if duplicate_rows:
        primary_row = _merge_duplicate_history_rows(
            primary_row,
            duplicate_rows,
            canonical_normalized_url=canonical_normalized_url,
            topic_source=source,
        )
    elif canonical_normalized_url and primary_row.normalized_url != canonical_normalized_url:
        primary_row.normalized_url = canonical_normalized_url
        primary_row.save(update_fields=["normalized_url", "updated_at"])
    return primary_row


def _get_or_create_history_for_source(
    source: TopicSource,
    *,
    default_status: str,
    default_last_run_outcome: str,
) -> SourceDiscoveryHistory:
    history_item = _get_history_for_source(source)
    if history_item is not None:
        return history_item
    normalized_url = _get_canonical_normalized_url_for_source(source)
    source_domain = _extract_domain(normalized_url or str(source.url or "").strip())
    now = timezone.now()
    return SourceDiscoveryHistory.objects.create(
        user=source.topic.user,
        topic=source.topic,
        topic_source=source if source.is_active else None,
        normalized_url=normalized_url,
        url=str(source.url or "").strip(),
        title=str(source.name or "").strip(),
        snippet="",
        domain=source_domain,
        provider_name="",
        query_text="",
        status=default_status,
        last_run_outcome=default_last_run_outcome,
        first_seen_at=now,
        last_seen_at=now,
        seen_count=1,
        created_topic_source=True,
        diagnostics={},
    )


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


def _merge_history_status_for_sync(existing_status: str, proposed_status: str) -> str:
    existing_value = str(existing_status or "").strip()
    if existing_value in {
        SourceDiscoveryHistory.STATUS_REMOVED_BY_USER,
        SourceDiscoveryHistory.STATUS_REJECTED_BY_QUALITY,
    }:
        return existing_value
    return _merge_history_status(existing_value, proposed_status)


def _find_matching_history_rows_for_source(source: TopicSource) -> list[SourceDiscoveryHistory]:
    source_keys = _build_source_identity_keys(source.url, source.normalized_url)
    if not source_keys:
        return []
    return [
        item
        for item in source.topic.source_discovery_history.all().order_by("-last_seen_at", "-id")
        if _build_source_identity_keys(item.url, item.normalized_url).intersection(source_keys)
    ]


def _select_primary_history_row(
    history_rows: list[SourceDiscoveryHistory],
) -> SourceDiscoveryHistory:
    def sort_key(item: SourceDiscoveryHistory) -> tuple[int, int, str, int]:
        return (
            _PASSIVE_STATUS_STRENGTH.get(str(item.status or "").strip(), -1),
            1 if item.topic_source_id else 0,
            item.last_seen_at.isoformat() if item.last_seen_at else "",
            item.pk or 0,
        )

    return max(history_rows, key=sort_key)


def _merge_duplicate_history_rows(
    primary_row: SourceDiscoveryHistory,
    duplicate_rows: list[SourceDiscoveryHistory],
    *,
    canonical_normalized_url: str,
    topic_source: TopicSource,
) -> SourceDiscoveryHistory:
    merged_rows = [primary_row, *duplicate_rows]
    latest_row = max(
        merged_rows,
        key=lambda item: (item.last_seen_at.isoformat() if item.last_seen_at else "", item.pk or 0),
    )

    primary_row.normalized_url = canonical_normalized_url or primary_row.normalized_url
    primary_row.url = str(primary_row.url or latest_row.url or topic_source.url or "").strip()
    primary_row.title = str(primary_row.title or latest_row.title or topic_source.name or "").strip()
    primary_row.snippet = str(primary_row.snippet or latest_row.snippet or "").strip()
    primary_row.domain = str(primary_row.domain or latest_row.domain or _extract_domain(primary_row.url)).strip()
    primary_row.provider_name = str(latest_row.provider_name or primary_row.provider_name or "").strip()
    primary_row.query_text = str(latest_row.query_text or primary_row.query_text or "").strip()
    primary_row.status = _select_primary_history_status(merged_rows)
    primary_row.last_run_outcome = str(latest_row.last_run_outcome or primary_row.last_run_outcome or "").strip()
    primary_row.source_content_type = str(
        latest_row.source_content_type or primary_row.source_content_type or ""
    ).strip()
    primary_row.quality_score = max(float(primary_row.quality_score or 0.0), float(latest_row.quality_score or 0.0))
    primary_row.substance_score = max(
        float(primary_row.substance_score or 0.0),
        float(latest_row.substance_score or 0.0),
    )
    primary_row.commercial_intent_score = max(
        float(primary_row.commercial_intent_score or 0.0),
        float(latest_row.commercial_intent_score or 0.0),
    )
    primary_row.quality_rejection_reason = str(
        latest_row.quality_rejection_reason or primary_row.quality_rejection_reason or ""
    ).strip()
    primary_row.freshness_status = str(latest_row.freshness_status or primary_row.freshness_status or "").strip()
    primary_row.detected_publication_date = (
        latest_row.detected_publication_date or primary_row.detected_publication_date
    )
    primary_row.detected_publication_year = latest_row.detected_publication_year or primary_row.detected_publication_year
    primary_row.first_seen_at = min(
        [item.first_seen_at for item in merged_rows if item.first_seen_at],
        default=primary_row.first_seen_at,
    )
    primary_row.last_seen_at = max(
        [item.last_seen_at for item in merged_rows if item.last_seen_at],
        default=primary_row.last_seen_at,
    )
    primary_row.seen_count = sum(max(int(item.seen_count or 0), 0) for item in merged_rows) or 1
    if primary_row.topic_source_id != topic_source.id:
        primary_row.topic_source = topic_source
    primary_row.created_topic_source = any(item.created_topic_source for item in merged_rows) or bool(topic_source.id)
    primary_row.discovery_run = latest_row.discovery_run or primary_row.discovery_run
    primary_row.diagnostics = dict(latest_row.diagnostics or primary_row.diagnostics or {})
    primary_row.save(
        update_fields=[
            "normalized_url",
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
            "first_seen_at",
            "last_seen_at",
            "seen_count",
            "topic_source",
            "created_topic_source",
            "discovery_run",
            "diagnostics",
            "updated_at",
        ]
    )
    if duplicate_rows:
        SourceDiscoveryHistory.objects.filter(pk__in=[item.pk for item in duplicate_rows]).delete()
    return primary_row


def _select_primary_history_status(history_rows: list[SourceDiscoveryHistory]) -> str:
    strongest_status = SourceDiscoveryHistory.STATUS_SEEN
    for item in history_rows:
        candidate_status = str(item.status or "").strip()
        if _PASSIVE_STATUS_STRENGTH.get(candidate_status, -1) > _PASSIVE_STATUS_STRENGTH.get(strongest_status, -1):
            strongest_status = candidate_status
    return strongest_status


def _get_canonical_normalized_url_for_source(source: TopicSource) -> str:
    for value in (source.normalized_url, source.url):
        normalized_value = _canonicalize_source_identity_value(value)
        if normalized_value:
            return normalized_value
    return ""


def _build_source_identity_keys(*values: str) -> set[str]:
    identity_keys: set[str] = set()
    for value in values:
        raw_value = str(value or "").strip()
        if not raw_value:
            continue
        identity_keys.add(raw_value)
        canonical_value = _canonicalize_source_identity_value(raw_value)
        if canonical_value:
            identity_keys.add(canonical_value)
    return identity_keys


def _canonicalize_source_identity_value(value: str) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""
    try:
        return str(classify_source_url(raw_value).normalized_url or "").strip()
    except Exception:
        return raw_value


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
