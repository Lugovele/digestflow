"""Persistence boundary for accepted LinkedIn PostFlow publication output.

This module is deliberately narrow: it converts an already accepted final-post
attempt outcome into the existing ContentPackage model used by the result page.
It does not adjudicate, repair, score, generate, or call providers.
"""
from __future__ import annotations

from apps.digests.models import Digest
from apps.packaging.models import ContentPackage
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    FinalPostAttemptOutcome,
)
from services.packaging.linkedin_post_publication_assembler import (
    LinkedInPublicationPackage,
    accepted_post_from_payload,
    assemble_linkedin_publication_package,
)


def persist_accepted_linkedin_publication_from_outcome(
    *,
    digest: Digest,
    attempt_outcome: FinalPostAttemptOutcome,
    deterministic_gate_passed: bool,
    semantic_grounding_passed: bool,
    quality_passed: bool,
) -> ContentPackage | None:
    """Persist a ContentPackage only for an already accepted PostFlow outcome.

    If an accepted outcome reaches this boundary, the clean PostFlow result is
    authoritative. Existing package rows are removed before assembly so an
    invalid accepted payload cannot leave stale legacy content visible as the
    current accepted post.
    """

    if not isinstance(attempt_outcome, FinalPostAttemptOutcome):
        raise TypeError("attempt_outcome must be a FinalPostAttemptOutcome.")
    if attempt_outcome.outcome != OUTCOME_ACCEPTED:
        return None
    if attempt_outcome.accepted_result is None:
        raise ValueError("accepted outcome must include accepted_result.")
    accepted_payload = attempt_outcome.accepted_result.accepted_payload
    if accepted_payload is None:
        raise ValueError("accepted_result must include accepted_payload.")

    ContentPackage.objects.filter(digest=digest).delete()

    accepted_post = accepted_post_from_payload(accepted_payload)
    publication_package = assemble_linkedin_publication_package(
        accepted_post=accepted_post,
        deterministic_gate_passed=deterministic_gate_passed,
        semantic_grounding_passed=semantic_grounding_passed,
        quality_passed=quality_passed,
        final_accepted=True,
    )
    if publication_package.validation_report.get("status") != "valid":
        raise ValueError("accepted publication package must be valid.")
    return persist_linkedin_publication_package(
        digest=digest,
        publication_package=publication_package,
    )


def persist_linkedin_publication_package(
    *,
    digest: Digest,
    publication_package: LinkedInPublicationPackage,
) -> ContentPackage:
    """Persist a typed publication package into the existing result-page model."""

    if not isinstance(publication_package, LinkedInPublicationPackage):
        raise TypeError("publication_package must be a LinkedInPublicationPackage.")

    content_package, _created = ContentPackage.objects.update_or_create(
        digest=digest,
        defaults={
            "post_text": publication_package.post_text,
            "hook_variants": list(publication_package.hook_variants),
            "cta_variants": list(publication_package.cta_variants),
            "hashtags": list(publication_package.hashtags),
            "carousel_outline": [
                dict(item) for item in publication_package.carousel_outline
            ],
            "validation_report": dict(publication_package.validation_report),
        },
    )
    return content_package
