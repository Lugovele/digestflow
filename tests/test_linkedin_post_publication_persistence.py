from __future__ import annotations

import inspect

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.digests.models import Digest, DigestRun
from apps.packaging.models import ContentPackage
from apps.topics.models import Topic
from services.packaging.linkedin_post_attempt_outcome import (
    OUTCOME_ACCEPTED,
    OUTCOME_NOT_READY,
    OUTCOME_REPAIR_REQUIRED,
    FinalPostAttemptOutcome,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_ACCEPT,
    ACTION_NOT_READY,
    ACTION_REPAIR_EDITORIAL,
    STATUS_ACCEPTED,
    STATUS_NOT_READY,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
    FinalPostFlowResult,
)
from services.packaging.linkedin_post_publication_assembler import (
    accepted_post_from_payload,
    assemble_linkedin_publication_package,
)
from services.packaging import linkedin_post_publication_persistence
from services.packaging.linkedin_post_publication_persistence import (
    persist_accepted_linkedin_publication_from_outcome,
    persist_linkedin_publication_package,
)


class LinkedInPostPublicationPersistenceTests(TestCase):
    def setUp(self) -> None:
        user = get_user_model().objects.create_user(
            username="postflow-publication",
            password="unused",
        )
        self.topic = Topic.objects.create(user=user, name="Bitcoin market structure")
        self.run = DigestRun.objects.create(
            topic=self.topic,
            status=DigestRun.STATUS_COMPLETED,
        )
        self.digest = Digest.objects.create(
            run=self.run,
            title="Bitcoin digest",
            payload={"version": 1, "title": "Bitcoin digest", "articles": []},
            quality_score=0.9,
        )

    def test_accepted_outcome_persists_content_package_from_publication_package(self) -> None:
        post_text = "Accepted post with a concrete market point."
        outcome = _accepted_outcome(post_text=post_text)

        content_package = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=outcome,
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        self.assertEqual(ContentPackage.objects.count(), 1)
        self.assertEqual(content_package.digest, self.digest)
        self.assertEqual(content_package.post_text, post_text)
        self.assertEqual(content_package.hook_variants, [])
        self.assertEqual(content_package.cta_variants, [])
        self.assertEqual(content_package.hashtags, [])
        self.assertEqual(content_package.carousel_outline, [])
        self.assertEqual(content_package.validation_report["status"], "valid")
        self.assertTrue(content_package.is_publish_ready())

    def test_non_accepted_outcome_does_not_persist_content_package(self) -> None:
        outcome = _not_ready_outcome()

        result = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=outcome,
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        self.assertIsNone(result)
        self.assertEqual(ContentPackage.objects.count(), 0)

    def test_repair_required_outcome_does_not_persist_content_package(self) -> None:
        outcome = _repair_required_outcome()

        result = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=outcome,
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=False,
        )

        self.assertIsNone(result)
        self.assertEqual(ContentPackage.objects.count(), 0)

    def test_accepted_outcome_with_failed_clean_flow_proof_does_not_persist(self) -> None:
        outcome = _accepted_outcome(post_text="Accepted shape with failed proof.")

        with self.assertRaisesRegex(
            ValueError,
            "accepted publication package must be valid",
        ):
            persist_accepted_linkedin_publication_from_outcome(
                digest=self.digest,
                attempt_outcome=outcome,
                deterministic_gate_passed=True,
                semantic_grounding_passed=False,
                quality_passed=True,
            )

        self.assertEqual(ContentPackage.objects.count(), 0)
    def test_persistence_rejects_raw_text_instead_of_typed_publication_package(self) -> None:
        with self.assertRaisesRegex(
            TypeError,
            "publication_package must be a LinkedInPublicationPackage",
        ):
            persist_linkedin_publication_package(
                digest=self.digest,
                publication_package="raw post text",  # type: ignore[arg-type]
            )

    def test_publication_failure_clears_stale_legacy_content(self) -> None:
        ContentPackage.objects.create(
            digest=self.digest,
            post_text="Stale legacy post that must not be shown.",
            validation_report={"status": "valid"},
        )
        outcome = _accepted_outcome(payload={"post_text": ""})

        with self.assertRaises(ValueError):
            persist_accepted_linkedin_publication_from_outcome(
                digest=self.digest,
                attempt_outcome=outcome,
                deterministic_gate_passed=True,
                semantic_grounding_passed=True,
                quality_passed=True,
            )

        self.assertFalse(ContentPackage.objects.filter(digest=self.digest).exists())

    def test_retry_same_accepted_outcome_leaves_one_visible_package(self) -> None:
        outcome = _accepted_outcome(post_text="Same accepted post.")

        first = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=outcome,
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )
        second = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=outcome,
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        self.assertEqual(ContentPackage.objects.filter(digest=self.digest).count(), 1)
        self.assertEqual(first.post_text, second.post_text)
        self.assertEqual(second.post_text, "Same accepted post.")

    def test_newer_accepted_outcome_replaces_existing_package(self) -> None:
        persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=_accepted_outcome(post_text="Earlier accepted post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        updated = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=_accepted_outcome(post_text="Newer accepted post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        self.assertEqual(ContentPackage.objects.filter(digest=self.digest).count(), 1)
        self.assertEqual(updated.post_text, "Newer accepted post.")
        self.assertEqual(self.digest.content_package.post_text, "Newer accepted post.")

    def test_result_page_copy_payload_uses_exact_accepted_text(self) -> None:
        post_text = "Exact accepted post.\n\nSecond paragraph stays intact."
        persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=_accepted_outcome(post_text=post_text),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        response = self.client.get(reverse("post-result", args=[self.run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["post_result_state"], "ready")
        self.assertTrue(response.context["show_copy_button"])
        self.assertEqual(response.context["final_post_text"], post_text)
        self.assertEqual(response.context["assembled_post_default"], post_text)

    def test_persisted_package_uses_existing_publication_assembler_report(self) -> None:
        payload = {"post_text": "Accepted post."}
        expected_package = assemble_linkedin_publication_package(
            accepted_post=accepted_post_from_payload(payload),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
            final_accepted=True,
        )

        content_package = persist_accepted_linkedin_publication_from_outcome(
            digest=self.digest,
            attempt_outcome=_accepted_outcome(payload=payload),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
        )

        self.assertEqual(
            content_package.validation_report,
            expected_package.validation_report,
        )

    def test_publication_persistence_does_not_import_ai_or_generation_paths(self) -> None:
        source = inspect.getsource(linkedin_post_publication_persistence)

        forbidden_terms = (
            "build_ai_client",
            "OpenAI",
            "Gemini",
            "Anthropic",
            "generate_content_package_for_digest",
            "execute_candidate_writer_prompt",
            "quality_evaluator",
            "repair_writer",
            "genericization_guard",
            "linkedin_api",
            "oauth",
        )
        for term in forbidden_terms:
            self.assertNotIn(term, source)


def _accepted_outcome(
    *,
    post_text: str = "Accepted post.",
    payload: dict | None = None,
) -> FinalPostAttemptOutcome:
    payload = payload if payload is not None else {"post_text": post_text}
    decision = _decision(ACTION_ACCEPT, reason="accepted")
    history = FinalPostAttemptHistory(attempts=[])
    attempt = _attempt(payload=payload, decision=decision)
    accepted_result = FinalPostFlowResult(
        status=STATUS_ACCEPTED,
        accepted_payload=payload,
        attempt_history=history.add_attempt(attempt),
        final_decision=decision,
        reason=decision.reason,
    )
    return FinalPostAttemptOutcome(
        outcome=OUTCOME_ACCEPTED,
        attempt=attempt,
        attempt_history=accepted_result.attempt_history,
        decision=decision,
        accepted_result=accepted_result,
        repair_required=False,
        repair_plan=None,
        terminal_result=accepted_result,
        reason=decision.reason,
    )


def _not_ready_outcome() -> FinalPostAttemptOutcome:
    decision = _decision(ACTION_NOT_READY, reason="not ready")
    history = FinalPostAttemptHistory(attempts=[])
    attempt = _attempt(payload={"post_text": "Not accepted."}, decision=decision)
    terminal_result = FinalPostFlowResult(
        status=STATUS_NOT_READY,
        accepted_payload=None,
        attempt_history=history.add_attempt(attempt),
        final_decision=decision,
        reason=decision.reason,
    )
    return FinalPostAttemptOutcome(
        outcome=OUTCOME_NOT_READY,
        attempt=attempt,
        attempt_history=terminal_result.attempt_history,
        decision=decision,
        accepted_result=None,
        repair_required=False,
        repair_plan=None,
        terminal_result=terminal_result,
        reason=decision.reason,
    )


def _repair_required_outcome() -> FinalPostAttemptOutcome:
    decision = _decision(
        ACTION_REPAIR_EDITORIAL,
        reason="quality repair required",
        repair_type="editorial",
    )
    history = FinalPostAttemptHistory(attempts=[])
    attempt = _attempt(payload={"post_text": "Needs repair."}, decision=decision)
    return FinalPostAttemptOutcome(
        outcome=OUTCOME_REPAIR_REQUIRED,
        attempt=attempt,
        attempt_history=history.add_attempt(attempt),
        decision=decision,
        accepted_result=None,
        repair_required=True,
        repair_plan={"repair_type": "editorial"},
        terminal_result=None,
        reason=decision.reason,
    )


def _decision(
    action: str,
    *,
    reason: str,
    repair_type: str | None = None,
) -> FinalPostDecision:
    return FinalPostDecision(
        action=action,
        reason=reason,
        repair_type=repair_type,
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _attempt(*, payload: dict, decision: FinalPostDecision) -> FinalPostAttempt:
    return FinalPostAttempt(
        attempt_index=0,
        payload=payload,
        validation_passed=True,
        validation_error="",
        diagnostics=None,
        quality_review={"pass": True},
        repair_plan=None,
        decision=decision,
        provider=None,
        model=None,
        prompt_name=None,
        prompt_version=None,
        token_usage=None,
        cost_metadata=None,
        created_at=None,
        parent_attempt_index=None,
    )
