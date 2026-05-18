"""Pure helpers that prepare research review items for later UI or persistence steps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

from apps.topics.models import TopicSourceOrigin
from services.sources.candidate_review import CandidateReviewItem
from services.sources.candidates import SourceCandidateStatus
from services.sources.research_orchestrator import SourceResearchResult


@dataclass(frozen=True)
class ResearchReviewContext:
    review_items: tuple[CandidateReviewItem, ...]
    persistable_items: tuple[CandidateReviewItem, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def total_review_item_count(self) -> int:
        return int(self.diagnostics.get("total_review_item_count", len(self.review_items)))

    @property
    def selectable_review_item_count(self) -> int:
        return int(
            self.diagnostics.get(
                "selectable_review_item_count",
                sum(1 for item in self.review_items if item.is_selectable),
            )
        )

    @property
    def accepted_count(self) -> int:
        return int(
            self.diagnostics.get(
                "accepted_count",
                sum(1 for item in self.review_items if item.status == SourceCandidateStatus.ACCEPTED),
            )
        )

    @property
    def needs_review_count(self) -> int:
        return int(
            self.diagnostics.get(
                "needs_review_count",
                sum(1 for item in self.review_items if item.status == SourceCandidateStatus.NEEDS_REVIEW),
            )
        )

    @property
    def rejected_count(self) -> int:
        return int(
            self.diagnostics.get(
                "rejected_count",
                len(self.review_items) - self.accepted_count - self.needs_review_count,
            )
        )

    @property
    def persistable_count(self) -> int:
        return int(self.diagnostics.get("persistable_count", len(self.persistable_items)))


def build_research_review_context(source_research_result: SourceResearchResult) -> ResearchReviewContext:
    review_items = tuple(source_research_result.review_items)
    persistable_items = tuple(get_persistable_research_candidates(review_items))

    accepted_count = sum(1 for item in review_items if item.status == SourceCandidateStatus.ACCEPTED)
    needs_review_count = sum(1 for item in review_items if item.status == SourceCandidateStatus.NEEDS_REVIEW)
    rejected_count = len(review_items) - accepted_count - needs_review_count
    selectable_review_item_count = sum(1 for item in review_items if item.is_selectable)

    diagnostics = {
        "total_review_item_count": len(review_items),
        "selectable_review_item_count": selectable_review_item_count,
        "accepted_count": accepted_count,
        "needs_review_count": needs_review_count,
        "rejected_count": rejected_count,
        "persistable_count": len(persistable_items),
        "provider_name": source_research_result.diagnostics.get("provider_name", ""),
        "topic_domain": source_research_result.diagnostics.get("topic_domain", ""),
    }

    return ResearchReviewContext(
        review_items=review_items,
        persistable_items=persistable_items,
        diagnostics=diagnostics,
    )


def get_persistable_research_candidates(review_items: Sequence[CandidateReviewItem]) -> list[CandidateReviewItem]:
    return [item for item in review_items if item.can_be_persisted]


def build_topic_source_payloads_from_review_items(
    review_items: Sequence[CandidateReviewItem],
) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for item in review_items:
        if not item.can_be_persisted:
            continue
        payloads.append(
            {
                "url": item.url,
                "title": item.label,
                "source_type": item.source_type,
                "origin": TopicSourceOrigin.DISCOVERED,
                "diagnostics": dict(item.diagnostics or {}),
            }
        )
    return payloads
