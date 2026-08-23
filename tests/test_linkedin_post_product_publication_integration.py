import json
import re
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.digests.models import Digest, DigestRun
from apps.packaging.models import ContentPackage
from apps.topics.models import Topic
from services.pipeline.run_pipeline import run_digest_pipeline
from services.packaging.linkedin_post_product_publication import FinalPostProviderClients


ACCEPTED_POST_TEXT = (
    "I would not treat faster intake as proof that the review burden disappeared.\n\n"
    "The useful signal is narrower: teams got faster at the front of the workflow, "
    "but they still needed human checks where labels, handoffs, and unsupported "
    "details could break the work.\n\n"
    "That distinction matters because speed at the first step can hide fragility "
    "later.\n\n"
    "Where does your workflow still need a human check?"
)


RAW_ITEMS = [
    {
        "title": "Remote workplace expectations need inclusion",
        "url": "https://example.com/source-1",
        "source_name": "Example Source",
        "snippet": (
            "A content team cut prep from 6 hours to 2.5 hours, but editors still "
            "checked every claim before publishing. The workflow got faster at the "
            "start, yet review stayed manual and unsupported details still had to "
            "be removed before publication."
        ),
    },
    {
        "title": "Hybrid communication rules need inclusive design",
        "url": "https://example.com/source-2",
        "source_name": "Example Source",
        "snippet": (
            "A support team got triage 28% faster with structured forms, but bad "
            "labels still broke routing. Intake accelerated while handoffs still "
            "needed operators to correct tickets sent to the wrong team."
        ),
    },
    {
        "title": "Remote work habits affect team trust",
        "url": "https://example.com/source-3",
        "source_name": "Example Source",
        "snippet": (
            "An ops team cut review time by 35% after redesigning the workflow "
            "before adding automation. They changed the handoff, clarified "
            "validation, and stopped repeating the same review loop."
        ),
    },
]


@override_settings(OPENAI_API_KEY="sk-test-key", GEMINI_API_KEY="gemini-test-key")
class LinkedInPostProductPublicationIntegrationTests(TestCase):
    def test_run_digest_pipeline_clean_accepted_path_persists_content_package(self):
        run = self._make_run("clean-accepted-user")
        clients = _passing_provider_clients()

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ):
            result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=clients,
            )

        run.refresh_from_db()
        content_package = run.digest.content_package

        self.assertEqual(result.id, run.id)
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(content_package.post_text, ACCEPTED_POST_TEXT)
        self.assertEqual(ContentPackage.objects.filter(digest=run.digest).count(), 1)
        self.assertEqual(run.metrics["packaging_stage"]["status"], "completed")
        self.assertEqual(run.metrics["packaging_stage"]["provider"], "clean_postflow")
        self.assertFalse(run.metrics["packaging_stage"]["is_mock"])
        self.assertEqual(
            run.metrics["packaging_stage"]["final_post_flow"]["outcome"],
            "accepted",
        )
        self.assertEqual(len(clients.candidate_writer_client.calls), 1)
        self.assertEqual(len(clients.semantic_grounding_client.calls), 1)
        self.assertEqual(len(clients.quality_evaluator_client.calls), 1)

    def test_post_result_view_renders_and_copies_exact_accepted_text(self):
        run = self._make_run("clean-result-view-user")
        clients = _passing_provider_clients()

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ):
            run_digest_pipeline(run.id, RAW_ITEMS, final_post_provider_clients=clients)

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["post_result_state"], "ready")
        self.assertTrue(response.context["show_copy_button"])
        self.assertEqual(response.context["final_post_text"], ACCEPTED_POST_TEXT)
        self.assertEqual(response.context["assembled_post_default"], ACCEPTED_POST_TEXT)
        self.assertContains(response, ACCEPTED_POST_TEXT)
        self.assertContains(response, "Copy full post")

    def test_legacy_generator_cannot_overwrite_clean_accepted_post(self):
        run = self._make_run("clean-authority-user")
        clients = _passing_provider_clients()

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ), patch(
            "services.packaging.generator.generate_content_package_for_digest",
            side_effect=AssertionError("legacy generator must not run"),
        ) as legacy_generator:
            run_digest_pipeline(run.id, RAW_ITEMS, final_post_provider_clients=clients)

        run.refresh_from_db()
        legacy_generator.assert_not_called()
        self.assertEqual(run.digest.content_package.post_text, ACCEPTED_POST_TEXT)

    def test_clean_non_accepted_outcome_does_not_leave_legacy_post_visible_as_success(self):
        run = self._make_run("clean-non-accepted-user")
        clients = _provider_clients(quality_response=_quality_review_payload(passed=False))

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ):
            result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=clients,
            )

        run.refresh_from_db()
        self.assertEqual(result.status, DigestRun.STATUS_PARTIAL_FAILED)
        self.assertFalse(hasattr(run.digest, "content_package"))
        self.assertEqual(ContentPackage.objects.count(), 0)
        self.assertIn("Clean final-post flow did not accept", run.error_message)

    def test_clean_execution_failure_does_not_leave_stale_content_package_visible(self):
        run = self._make_run("clean-stale-user")
        digest = self._create_digest(run)
        ContentPackage.objects.create(
            digest=digest,
            post_text="Stale legacy text",
            validation_report={"status": "valid"},
        )
        clients = _provider_clients(candidate_response=RuntimeError("provider down"))

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            return_value=(digest, _digest_debug()),
        ):
            result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=clients,
            )

        run.refresh_from_db()
        self.assertEqual(result.status, DigestRun.STATUS_PARTIAL_FAILED)
        self.assertFalse(ContentPackage.objects.filter(digest=digest).exists())
        self.assertFalse(hasattr(run.digest, "content_package"))

    def test_retry_clean_acceptance_does_not_create_duplicate_content_package_rows(self):
        run = self._make_run("clean-retry-user")
        digest = self._create_digest(run)

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            return_value=(digest, _digest_debug()),
        ), patch(
            "services.pipeline.run_pipeline.get_used_article_urls_for_topic",
            return_value=set(),
        ):
            first_result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=_passing_provider_clients(),
            )
            second_result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=_passing_provider_clients(),
            )

        self.assertEqual(first_result.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(second_result.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(ContentPackage.objects.filter(digest=digest).count(), 1)
        self.assertEqual(digest.content_package.post_text, ACCEPTED_POST_TEXT)

    def test_no_model_provider_call_happens_after_acceptance(self):
        run = self._make_run("clean-no-extra-ai-user")
        clients = _passing_provider_clients()

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ):
            run_digest_pipeline(run.id, RAW_ITEMS, final_post_provider_clients=clients)

        self.assertEqual(len(clients.candidate_writer_client.calls), 1)
        self.assertEqual(len(clients.semantic_grounding_client.calls), 1)
        self.assertEqual(len(clients.quality_evaluator_client.calls), 1)

    def test_repair_required_outcome_cannot_reach_publication_persistence(self):
        run = self._make_run("clean-no-repair-publication-user")
        clients = _provider_clients(quality_response=_quality_review_payload(passed=False))

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ), patch(
            "services.packaging.linkedin_post_product_publication.persist_accepted_linkedin_publication_from_outcome"
        ) as persist:
            result = run_digest_pipeline(
                run.id,
                RAW_ITEMS,
                final_post_provider_clients=clients,
            )

        self.assertEqual(result.status, DigestRun.STATUS_PARTIAL_FAILED)
        persist.assert_not_called()
        self.assertFalse(ContentPackage.objects.exists())

    def test_pipeline_completion_state_requires_publication_availability(self):
        run = self._make_run("clean-completion-state-user")
        clients = _provider_clients(quality_response=_quality_review_payload(passed=False))

        with patch(
            "services.pipeline.run_pipeline.generate_digest_for_run",
            side_effect=self._generate_digest,
        ):
            run_digest_pipeline(run.id, RAW_ITEMS, final_post_provider_clients=clients)

        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_PARTIAL_FAILED)
        self.assertEqual(run.metrics["packaging_stage"]["status"], "failed")
        self.assertFalse(hasattr(run.digest, "content_package"))

    def _make_run(self, username: str) -> DigestRun:
        user = get_user_model().objects.create_user(
            username=username,
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Remote work inclusion",
            keywords=["remote", "workplace", "inclusion"],
            excluded_keywords=[],
            default_quality_threshold=0.2,
        )
        return DigestRun.objects.create(
            topic=topic,
            input_snapshot={"mode": "raw_items", "source": "integration_test"},
        )

    def _generate_digest(self, run: DigestRun, _selected_items: list[dict]):
        return self._create_digest(run), _digest_debug()

    def _create_digest(self, run: DigestRun) -> Digest:
        digest, _created = Digest.objects.update_or_create(
            run=run,
            defaults={
                "title": "Remote work inclusion digest",
                "payload": {
                    "title": "Remote work inclusion digest",
                    "articles": [
                        {
                            "url": "https://example.com/source-1",
                            "title": "Remote workplace expectations need inclusion",
                            "summary": (
                                "A content team cut prep from 6 hours to 2.5 hours, "
                                "but editors still checked every claim before publishing."
                            ),
                            "key_points": [
                                "The workflow got faster at intake while review stayed manual.",
                            ],
                            "source_name": "Example Source",
                        },
                        {
                            "url": "https://example.com/source-2",
                            "title": "Hybrid communication rules need inclusive design",
                            "summary": (
                                "A support team got triage 28% faster with structured "
                                "forms, but bad labels still broke routing."
                            ),
                            "key_points": [
                                "Operators still corrected tickets that landed with the wrong team.",
                            ],
                            "source_name": "Example Source",
                        },
                        {
                            "url": "https://example.com/source-3",
                            "title": "Remote work habits affect team trust",
                            "summary": (
                                "An ops team cut review time by 35% after redesigning "
                                "handoffs before adding automation."
                            ),
                            "key_points": [
                                "Clear validation stopped repeated back-and-forth review loops.",
                            ],
                            "source_name": "Example Source",
                        },
                    ],
                },
                "quality_score": 0.8,
            },
        )
        return digest


class FakeTextClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        if isinstance(self.response, BaseException):
            raise self.response
        if callable(self.response):
            raw_text = self.response(kwargs)
        else:
            raw_text = self.response
        return SimpleNamespace(
            text=raw_text,
            raw={"id": "fake-response"},
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            provider_response_metadata=None,
        )


def _passing_provider_clients() -> FinalPostProviderClients:
    return _provider_clients()


def _provider_clients(
    *,
    candidate_response=None,
    semantic_response=None,
    quality_response=None,
) -> FinalPostProviderClients:
    return FinalPostProviderClients(
        candidate_writer_client=FakeTextClient(
            candidate_response
            if candidate_response is not None
            else json.dumps({"post_text": ACCEPTED_POST_TEXT})
        ),
        semantic_grounding_client=FakeTextClient(
            semantic_response if semantic_response is not None else _semantic_response
        ),
        quality_evaluator_client=FakeTextClient(
            json.dumps(
                quality_response
                if quality_response is not None
                else _quality_review_payload(passed=True)
            )
        ),
    )


def _semantic_response(kwargs: dict) -> str:
    selected_evidence_ids = _evidence_ids_from_prompt(kwargs.get("prompt", ""))
    supported_ids = selected_evidence_ids[:1] or ["a0-summary"]
    return json.dumps(
        {
            "pass": True,
            "claims": [
                {
                    "claim_id": "c1",
                    "field_name": "post_text",
                    "value_index": None,
                    "claim_text": ACCEPTED_POST_TEXT.split("\n", 1)[0],
                    "claim_type": "author_interpretation",
                    "support_status": "supported",
                    "severity": "info",
                    "supported_evidence_ids": supported_ids,
                    "required_qualifications": [],
                    "missing_qualifications": [],
                    "rationale": "Grounded in selected evidence.",
                    "repair_hint": "",
                }
            ],
            "failed_claim_ids": [],
            "automatic_fail_reason": "",
            "requires_human_review": False,
            "human_review_reason": "",
            "repairable": False,
            "repair_instructions": [],
        }
    )


def _evidence_ids_from_prompt(prompt: str) -> list[str]:
    evidence_ids = []
    for match in re.finditer(r'"evidence_id"\s*:\s*"([^"]+)"', str(prompt or "")):
        evidence_id = match.group(1)
        if evidence_id not in evidence_ids:
            evidence_ids.append(evidence_id)
    return evidence_ids


def _quality_review_payload(*, passed: bool) -> dict:
    scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 4,
        "practical_value": 4,
        "cta": 4 if passed else 2,
    }
    return {
        "scores": scores,
        "total_score": sum(scores.values()),
        "pass": passed,
        "failed_criteria": [] if passed else ["cta"],
        "automatic_fail_reason": "",
        "notes": ["Evaluator note."],
        "criterion_rationales": {
            criterion: {
                "score": score,
                "max_score": 5,
                "rationale": f"{criterion} rationale tied to the candidate text.",
                "post_text_evidence": f"{criterion} evidence from post_text.",
                "failure_reason": "" if passed or criterion != "cta" else "CTA is weak.",
            }
            for criterion, score in scores.items()
        },
    }


def _digest_debug() -> dict:
    return {
        "provider": "test",
        "is_mock": False,
        "tokens": None,
        "estimated_cost_usd": None,
    }
