"""Deterministic source-candidate evaluation primitives for future research flows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Iterable, Mapping, Sequence
from urllib.parse import urlparse
import re

from services.sources.detector import classify_source_url
from services.sources.source_quality import assess_source_quality


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "this",
    "to",
    "with",
}


class SourceCandidateStatus(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    NEEDS_REVIEW = "needs_review"
    INVALID_URL = "invalid_url"
    DUPLICATE = "duplicate"
    UNREACHABLE = "unreachable"
    WEAK_CONTENT = "weak_content"
    LOW_RELEVANCE = "low_relevance"


@dataclass(frozen=True)
class SourceCandidateInput:
    url: str
    title: str = ""
    snippet: str = ""
    origin_reason: str = ""
    source_type_guess: str = ""
    fetch_status: int | None = None
    fetch_failure_reason: str = ""
    readable_text_length: int = 0
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluatedSourceCandidate:
    url: str
    normalized_url: str
    title: str
    snippet: str
    hostname: str
    candidate_type: str
    origin_reason: str
    score: float
    status: SourceCandidateStatus
    rejection_reasons: tuple[str, ...]
    diagnostics: dict[str, Any]

    @property
    def accepted(self) -> bool:
        return self.status == SourceCandidateStatus.ACCEPTED


def evaluate_source_candidate(
    candidate: SourceCandidateInput,
    *,
    topic: str = "",
    focus_terms: Sequence[str] = (),
    existing_normalized_urls: Iterable[str] = (),
    existing_hostnames: Iterable[str] = (),
    seen_normalized_urls: Iterable[str] = (),
    seen_hostnames: Iterable[str] = (),
) -> EvaluatedSourceCandidate:
    normalized_source = classify_source_url(candidate.url)
    normalized_url = normalized_source.normalized_url
    hostname = _extract_hostname(normalized_url, fallback=candidate.url)
    title = str(candidate.title or "").strip()
    snippet = str(candidate.snippet or "").strip()
    origin_reason = str(candidate.origin_reason or normalized_source.detection_reason).strip()
    candidate_type = str(candidate.source_type_guess or normalized_source.source_type).strip()

    rejection_reasons: list[str] = []
    diagnostics = dict(candidate.diagnostics or {})

    diagnostics.update(
        {
            "normalized_url": normalized_url,
            "hostname": hostname,
            "candidate_type": candidate_type,
            "platform": normalized_source.platform,
            "origin_reason": origin_reason,
            "fetch_status": candidate.fetch_status,
            "fetch_failure_reason": str(candidate.fetch_failure_reason or "").strip(),
            "readable_text_length": int(candidate.readable_text_length or 0),
        }
    )

    invalid_url = not _is_valid_candidate_url(normalized_url)
    diagnostics["invalid_url"] = invalid_url
    if invalid_url:
        rejection_reasons.append("invalid url")

    existing_url_set = {str(value or "").strip() for value in existing_normalized_urls if str(value or "").strip()}
    existing_host_set = {str(value or "").strip().lower() for value in existing_hostnames if str(value or "").strip()}
    seen_url_set = {str(value or "").strip() for value in seen_normalized_urls if str(value or "").strip()}
    seen_host_set = {str(value or "").strip().lower() for value in seen_hostnames if str(value or "").strip()}

    duplicate_url = normalized_url in existing_url_set or normalized_url in seen_url_set
    duplicate_hostname = hostname in existing_host_set or hostname in seen_host_set

    diagnostics["duplicate_url"] = duplicate_url
    diagnostics["duplicate_hostname"] = duplicate_hostname

    if duplicate_url:
        rejection_reasons.append("duplicate normalized url")

    fetch_failure_reason = str(candidate.fetch_failure_reason or "").strip()
    is_unreachable = bool(fetch_failure_reason) or candidate.fetch_status in {401, 403, 404, 408, 429, 500, 502, 503, 504}
    diagnostics["is_unreachable"] = is_unreachable
    if is_unreachable:
        rejection_reasons.append(fetch_failure_reason or f"http {candidate.fetch_status}")

    readable_text_length = int(candidate.readable_text_length or 0)
    weak_content = readable_text_length > 0 and readable_text_length < 120
    diagnostics["weak_content"] = weak_content
    if weak_content:
        rejection_reasons.append(f"weak content ({readable_text_length} chars)")

    topic_terms = _build_topic_terms(topic, focus_terms)
    matched_terms = _match_terms(topic_terms, title=title, snippet=snippet, normalized_url=normalized_url)
    diagnostics["topic_terms"] = sorted(topic_terms)
    diagnostics["matched_terms"] = matched_terms
    diagnostics["match_count"] = len(matched_terms)

    quality_assessment = assess_source_quality(
        title=title,
        url=normalized_url or candidate.url,
        snippet=snippet,
        provider_published_at=str(
            (diagnostics.get("raw_result_diagnostics") or {}).get("provider_published_at") or ""
        ).strip(),
    )
    diagnostics.update(
        {
            "source_content_type": quality_assessment.source_content_type,
            "quality_score": quality_assessment.quality_score,
            "commercial_intent_score": quality_assessment.commercial_intent_score,
            "substance_score": quality_assessment.substance_score,
            "freshness_status": quality_assessment.freshness_status,
            "detected_publication_date": quality_assessment.detected_publication_date,
            "detected_publication_year": quality_assessment.detected_publication_year,
            "freshness_score": quality_assessment.freshness_score,
            "freshness_rejection_reason": quality_assessment.freshness_rejection_reason,
            "freshness_signals": list(quality_assessment.freshness_signals),
            "quality_tags": list(quality_assessment.quality_tags),
            "quality_rejection_reason": quality_assessment.rejection_reason,
            "quality_accepted_reason": quality_assessment.accepted_reason,
            "quality_accepted": quality_assessment.accepted,
        }
    )

    score_breakdown = {
        "relevance": len(matched_terms) * 22,
        "title_bonus": 8 if _contains_any(title, matched_terms) else 0,
        "snippet_bonus": 6 if _contains_any(snippet, matched_terms) else 0,
        "content_bonus": min(readable_text_length, 600) / 20 if readable_text_length else 0,
        "source_type_bonus": _source_type_bonus(candidate_type),
        "source_quality_bonus": quality_assessment.quality_score * 3,
        "duplicate_hostname_penalty": -12 if duplicate_hostname and not duplicate_url else 0,
        "weak_content_penalty": -18 if weak_content else 0,
        "unreachable_penalty": -40 if is_unreachable else 0,
    }
    score = round(sum(score_breakdown.values()), 2)
    diagnostics["score_breakdown"] = score_breakdown

    status = _resolve_candidate_status(
        invalid_url=invalid_url,
        duplicate_url=duplicate_url,
        unreachable=is_unreachable,
        weak_content=weak_content,
        duplicate_hostname=duplicate_hostname,
        score=score,
        match_count=len(matched_terms),
        quality_accepted=quality_assessment.accepted,
        quality_rejection_reason=quality_assessment.rejection_reason,
    )

    if status == SourceCandidateStatus.LOW_RELEVANCE:
        rejection_reasons.append("low relevance")
    elif status == SourceCandidateStatus.REJECTED and quality_assessment.rejection_reason:
        rejection_reasons.append(quality_assessment.rejection_reason)
    elif status == SourceCandidateStatus.INVALID_URL and "invalid url" not in rejection_reasons:
        rejection_reasons.append("invalid url")
    elif status == SourceCandidateStatus.NEEDS_REVIEW and duplicate_hostname:
        rejection_reasons.append("duplicate hostname")
    elif status == SourceCandidateStatus.REJECTED and not rejection_reasons:
        rejection_reasons.append("candidate was rejected")

    diagnostics["status"] = status.value
    diagnostics["rejection_reasons"] = rejection_reasons[:]

    return EvaluatedSourceCandidate(
        url=normalized_source.original_url,
        normalized_url=normalized_url,
        title=title,
        snippet=snippet,
        hostname=hostname,
        candidate_type=candidate_type,
        origin_reason=origin_reason,
        score=score,
        status=status,
        rejection_reasons=tuple(rejection_reasons),
        diagnostics=diagnostics,
    )


def evaluate_source_candidates(
    candidates: Sequence[SourceCandidateInput],
    *,
    topic: str = "",
    focus_terms: Sequence[str] = (),
    existing_normalized_urls: Iterable[str] = (),
    existing_hostnames: Iterable[str] = (),
) -> list[EvaluatedSourceCandidate]:
    evaluations: list[EvaluatedSourceCandidate] = []
    seen_normalized_urls: set[str] = set()
    seen_hostnames: set[str] = set()

    for candidate in candidates:
        evaluation = evaluate_source_candidate(
            candidate,
            topic=topic,
            focus_terms=focus_terms,
            existing_normalized_urls=existing_normalized_urls,
            existing_hostnames=existing_hostnames,
            seen_normalized_urls=seen_normalized_urls,
            seen_hostnames=seen_hostnames,
        )
        evaluations.append(evaluation)
        seen_normalized_urls.add(evaluation.normalized_url)
        seen_hostnames.add(evaluation.hostname)

    return sort_evaluated_candidates(evaluations)


def sort_evaluated_candidates(
    candidates: Sequence[EvaluatedSourceCandidate],
) -> list[EvaluatedSourceCandidate]:
    status_priority = {
        SourceCandidateStatus.ACCEPTED: 0,
        SourceCandidateStatus.NEEDS_REVIEW: 1,
        SourceCandidateStatus.DUPLICATE: 2,
        SourceCandidateStatus.INVALID_URL: 3,
        SourceCandidateStatus.WEAK_CONTENT: 4,
        SourceCandidateStatus.UNREACHABLE: 5,
        SourceCandidateStatus.LOW_RELEVANCE: 6,
        SourceCandidateStatus.REJECTED: 7,
    }
    return sorted(
        candidates,
        key=lambda candidate: (
            status_priority.get(candidate.status, 99),
            -candidate.score,
            candidate.hostname,
            candidate.normalized_url,
        ),
    )


def _resolve_candidate_status(
    *,
    invalid_url: bool,
    duplicate_url: bool,
    unreachable: bool,
    weak_content: bool,
    duplicate_hostname: bool,
    score: float,
    match_count: int,
    quality_accepted: bool,
    quality_rejection_reason: str,
) -> SourceCandidateStatus:
    if invalid_url:
        return SourceCandidateStatus.INVALID_URL
    if duplicate_url:
        return SourceCandidateStatus.DUPLICATE
    if unreachable:
        return SourceCandidateStatus.UNREACHABLE
    if weak_content:
        return SourceCandidateStatus.WEAK_CONTENT
    if quality_rejection_reason:
        return SourceCandidateStatus.REJECTED
    if match_count == 0 or score < 20:
        return SourceCandidateStatus.LOW_RELEVANCE
    if duplicate_hostname:
        return SourceCandidateStatus.NEEDS_REVIEW
    if not quality_accepted:
        return SourceCandidateStatus.LOW_RELEVANCE
    return SourceCandidateStatus.ACCEPTED


def _build_topic_terms(topic: str, focus_terms: Sequence[str]) -> set[str]:
    terms: set[str] = set()
    for phrase in [str(topic or "").strip(), *(str(term or "").strip() for term in (focus_terms or ()))]:
        if not phrase:
            continue
        normalized_phrase = _normalize_text(phrase)
        if normalized_phrase:
            terms.add(normalized_phrase)
        for token in re.findall(r"[a-z0-9]+", normalized_phrase):
            if len(token) >= 3 and token not in _STOP_WORDS:
                terms.add(token)
    return terms


def _match_terms(topic_terms: set[str], *, title: str, snippet: str, normalized_url: str) -> list[str]:
    corpus = " ".join(
        part
        for part in (
            _normalize_text(title),
            _normalize_text(snippet),
            _normalize_text(normalized_url.replace("-", " ").replace("/", " ")),
        )
        if part
    )
    matches = [term for term in sorted(topic_terms) if term and term in corpus]
    return matches


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    normalized = _normalize_text(text)
    return any(term in normalized for term in terms)


def _normalize_text(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return cleaned


def _source_type_bonus(candidate_type: str) -> int:
    bonuses = {
        "rss_feed": 12,
        "blog_index": 8,
        "publication": 6,
        "generic_html": 4,
        "devto_tag": 6,
        "devto_author": 6,
        "devto_article": 5,
    }
    return bonuses.get(str(candidate_type or "").strip(), 3)


def _extract_hostname(normalized_url: str, *, fallback: str) -> str:
    parsed = urlparse(normalized_url)
    hostname = str(parsed.netloc or "").strip().lower()
    if hostname:
        return hostname
    return str(urlparse(str(fallback or "")).netloc or fallback).strip().lower()


def _is_valid_candidate_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
