import json
import re
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.digests.models import DigestRun
from apps.packaging.models import ContentPackage
from apps.topics.models import Topic, TopicSource, TopicSourceMode, TopicSourceOrigin
from services.ai.digest_smoke_test import DigestGenerationPayload


ACCEPTED_BROWSER_POST_TEXT = (
    "POSTFLOW_BROWSER_E2E_SENTINEL\n\n"
    "I would not treat faster intake as proof that review disappeared.\n\n"
    "The useful signal is narrower: teams got faster at the front of the workflow, "
    "but they still needed human checks where labels, handoffs, and unsupported "
    "details could break the work.\n\n"
    "That distinction matters because speed at the first step can hide fragility later."
)


class PostFlowBrowserPublicationIntegrationTests(TestCase):
    @override_settings(
        OPENAI_API_KEY="sk-test-key",
        GEMINI_API_KEY="gemini-test-key",
        ANTHROPIC_API_KEY="anthropic-test-key",
    )
    def test_browser_generate_post_reaches_clean_publication_result(self) -> None:
        user = get_user_model().objects.create_user(
            username="browser-publication-user",
            password="not-used-in-test",
        )
        topic = Topic.objects.create(
            user=user,
            name="Remote work inclusion",
            source_mode=TopicSourceMode.HYBRID,
            keywords=["remote", "workflow", "review"],
            excluded_keywords=[],
            default_quality_threshold=0.2,
        )
        manual_source = TopicSource.objects.create(
            topic=topic,
            name="Manual operations source",
            url="https://example.com/manual-feed.xml",
            normalized_url="https://example.com/manual-feed.xml",
            source_type="rss",
            origin=TopicSourceOrigin.MANUAL,
            is_active=True,
        )
        research_source = TopicSource.objects.create(
            topic=topic,
            name="Research operations source",
            url="https://example.com/research-feed.xml",
            normalized_url="https://example.com/research-feed.xml",
            source_type="rss",
            origin=TopicSourceOrigin.DISCOVERED,
            is_active=True,
        )
        candidate_client = FakeTextClient(json.dumps({"post_text": ACCEPTED_BROWSER_POST_TEXT}))
        semantic_client = FakeTextClient(_semantic_grounding_response)
        quality_client = FakeTextClient(json.dumps(_quality_review_payload()))

        with patch(
            "apps.digests.views.fetch_rss_articles",
            side_effect=[_manual_source_items(), _research_source_items()],
        ) as fetch_rss_articles, patch(
            "services.digests.generator.generate_digest_payload",
            return_value=_digest_generation_payload(),
        ) as generate_digest_payload, patch(
            "services.packaging.linkedin_post_candidate_writer_execution.build_ai_client",
            return_value=candidate_client,
        ) as build_candidate_client, patch(
            "services.packaging.linkedin_post_semantic_grounding_execution.build_ai_client",
            return_value=semantic_client,
        ) as build_semantic_client, patch(
            "services.packaging.linkedin_post_quality_evaluator_execution.build_ai_client",
            return_value=quality_client,
        ) as build_quality_client, patch(
            "services.packaging.generator.generate_content_package_for_digest",
            side_effect=AssertionError("legacy generator must not run"),
        ) as legacy_generator:
            response = self.client.post(reverse("run-pipeline", args=[topic.id]))

        self.assertEqual(response.status_code, 302)
        run = DigestRun.objects.select_related("topic").get(topic=topic)
        self.assertRedirects(response, reverse("run-detail", args=[run.id]), fetch_redirect_response=False)

        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_COMPLETED)
        self.assertEqual(run.metrics["packaging_stage"]["status"], "completed")
        self.assertEqual(run.metrics["packaging_stage"]["provider"], "clean_postflow")
        self.assertEqual(run.metrics["packaging_stage"]["final_post_flow"]["outcome"], "accepted")
        self.assertEqual(ContentPackage.objects.filter(digest=run.digest).count(), 1)
        self.assertEqual(ContentPackage.objects.count(), 1)
        content_package = run.digest.content_package
        self.assertEqual(content_package.post_text, ACCEPTED_BROWSER_POST_TEXT)
        self.assertIn("POSTFLOW_BROWSER_E2E_SENTINEL", content_package.post_text)
        self.assertNotIn("legacy", content_package.post_text.lower())

        result_response = self.client.get(reverse("post-result", args=[run.id]))
        result_html = result_response.content.decode("utf-8")
        self.assertEqual(result_response.status_code, 200)
        self.assertEqual(result_response.context["post_result_state"], "ready")
        self.assertEqual(result_response.context["final_post_text"], ACCEPTED_BROWSER_POST_TEXT)
        self.assertEqual(result_response.context["assembled_post_default"], ACCEPTED_BROWSER_POST_TEXT)
        self.assertContains(result_response, ACCEPTED_BROWSER_POST_TEXT)
        self.assertContains(result_response, "POSTFLOW_BROWSER_E2E_SENTINEL")
        self.assertContains(result_response, "Copy full post")
        self.assertIn('data-copy-full-post', result_html)
        self.assertIn('data-preview-body', result_html)

        fetch_rss_articles.assert_any_call(manual_source.normalized_url)
        fetch_rss_articles.assert_any_call(research_source.normalized_url)
        self.assertEqual(fetch_rss_articles.call_count, 2)
        generate_digest_payload.assert_called_once()
        build_candidate_client.assert_called_once()
        build_semantic_client.assert_called_once()
        build_quality_client.assert_called_once()
        self.assertEqual(len(candidate_client.calls), 1)
        self.assertEqual(len(semantic_client.calls), 1)
        self.assertEqual(len(quality_client.calls), 1)
        legacy_generator.assert_not_called()


class FakeTextClient:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def generate_text(self, **kwargs):
        self.calls.append(kwargs)
        raw_text = self.response(kwargs) if callable(self.response) else self.response
        return SimpleNamespace(
            text=raw_text,
            raw={"id": "fake-browser-e2e-response"},
            usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7},
            provider_response_metadata=None,
        )


def _manual_source_items() -> list[dict]:
    return [
        {
            "title": "Remote workplace expectations need inclusion",
            "url": "https://example.com/articles/manual-1",
            "source_name": "Manual operations source",
            "snippet": (
                "A content team cut prep from 6 hours to 2.5 hours, but editors still "
                "checked every claim before publishing. The workflow got faster at the "
                "start, yet review stayed manual and unsupported details still had to be "
                "removed before publication."
            ),
        },
        {
            "title": "Hybrid communication rules need inclusive design",
            "url": "https://example.com/articles/manual-2",
            "source_name": "Manual operations source",
            "snippet": (
                "A support team got triage 28% faster with structured forms, but bad "
                "labels still broke routing. Intake accelerated while handoffs still "
                "needed operators to correct tickets sent to the wrong team."
            ),
        },
    ]


def _research_source_items() -> list[dict]:
    return [
        {
            "title": "Remote work habits affect team trust",
            "url": "https://example.com/articles/research-1",
            "source_name": "Research operations source",
            "snippet": (
                "An ops team cut review time by 35% after redesigning the workflow before "
                "adding automation. They changed the handoff, clarified validation, and "
                "stopped repeating the same review loop."
            ),
        }
    ]


def _digest_generation_payload() -> DigestGenerationPayload:
    payload = {
        "title": "Remote work inclusion digest",
        "articles": [
            {
                "url": "https://example.com/articles/manual-1",
                "title": "Remote workplace expectations need inclusion",
                "summary": (
                    "A content team cut prep from 6 hours to 2.5 hours, but editors "
                    "still checked every claim before publishing."
                ),
                "key_points": [
                    "The workflow got faster at intake while review stayed manual.",
                ],
                "source_name": "Manual operations source",
                "confidence": 0.8,
            },
            {
                "url": "https://example.com/articles/manual-2",
                "title": "Hybrid communication rules need inclusive design",
                "summary": (
                    "A support team got triage 28% faster with structured forms, but "
                    "bad labels still broke routing."
                ),
                "key_points": [
                    "Operators still corrected tickets that landed with the wrong team.",
                ],
                "source_name": "Manual operations source",
                "confidence": 0.8,
            },
            {
                "url": "https://example.com/articles/research-1",
                "title": "Remote work habits affect team trust",
                "summary": (
                    "An ops team cut review time by 35% after redesigning handoffs "
                    "before adding automation."
                ),
                "key_points": [
                    "Clear validation stopped repeated back-and-forth review loops.",
                ],
                "source_name": "Research operations source",
                "confidence": 0.8,
            },
        ],
    }
    return DigestGenerationPayload(
        prompt="fake digest prompt",
        response_text=json.dumps(payload),
        payload=payload,
        is_mock=False,
        provider="fake-test-provider",
        fallback_reason="",
        tokens={"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
        estimated_cost_usd=None,
        articles=payload["articles"],
    )


def _semantic_grounding_response(kwargs: dict) -> str:
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
                    "claim_text": "The workflow got faster at the front of the workflow.",
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


def _quality_review_payload() -> dict:
    scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 4,
        "human_voice": 4,
        "practical_value": 4,
        "cta": 4,
    }
    return {
        "scores": scores,
        "total_score": sum(scores.values()),
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
        "notes": ["Evaluator note."],
        "criterion_rationales": {
            criterion: {
                "score": score,
                "max_score": 5,
                "rationale": f"{criterion} rationale tied to the candidate text.",
                "post_text_evidence": f"{criterion} evidence from post_text.",
                "failure_reason": "",
            }
            for criterion, score in scores.items()
        },
    }
