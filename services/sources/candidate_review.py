"""Pure adapters that prepare evaluated source candidates for review UIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
from urllib.parse import urlparse

from services.sources.candidates import (
    EvaluatedSourceCandidate,
    SourceCandidateStatus,
    sort_evaluated_candidates,
)


@dataclass(frozen=True)
class CandidateReviewItem:
    url: str
    normalized_url: str
    hostname: str
    label: str
    source_type: str
    status: SourceCandidateStatus
    score: float
    rejection_reasons: tuple[str, ...]
    diagnostics: dict
    is_selectable: bool
    can_be_persisted: bool
    default_selected: bool


def build_candidate_review_item(candidate: EvaluatedSourceCandidate) -> CandidateReviewItem:
    label = _build_candidate_review_label(candidate)
    is_selectable = candidate.status in {
        SourceCandidateStatus.ACCEPTED,
        SourceCandidateStatus.NEEDS_REVIEW,
    }
    can_be_persisted = is_selectable
    default_selected = candidate.status == SourceCandidateStatus.ACCEPTED

    return CandidateReviewItem(
        url=candidate.url,
        normalized_url=candidate.normalized_url,
        hostname=_normalize_display_hostname(candidate.hostname),
        label=label,
        source_type=candidate.candidate_type,
        status=candidate.status,
        score=candidate.score,
        rejection_reasons=tuple(candidate.rejection_reasons),
        diagnostics=dict(candidate.diagnostics or {}),
        is_selectable=is_selectable,
        can_be_persisted=can_be_persisted,
        default_selected=default_selected,
    )


def build_candidate_review_items(
    candidates: Sequence[EvaluatedSourceCandidate],
) -> list[CandidateReviewItem]:
    ordered_candidates = sort_evaluated_candidates(candidates)
    return [build_candidate_review_item(candidate) for candidate in ordered_candidates]


def _build_candidate_review_label(candidate: EvaluatedSourceCandidate) -> str:
    title = str(candidate.title or "").strip()
    if title:
        return title
    return _fallback_label_from_url(candidate.normalized_url or candidate.url)


def _fallback_label_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    hostname = _normalize_display_hostname(parsed.netloc or parsed.path or url)
    path = str(parsed.path or "").strip("/")
    if path:
        return f"{hostname}/{path}"
    return hostname or str(url or "").strip()


def _normalize_display_hostname(hostname: str) -> str:
    normalized = str(hostname or "").strip().lower()
    if normalized.startswith("www."):
        return normalized[4:]
    return normalized
