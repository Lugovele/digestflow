from __future__ import annotations

import copy
import json
from types import SimpleNamespace

from django.test import SimpleTestCase

from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterRawResponse,
)
from services.packaging.linkedin_post_candidate_writer_length_repair import (
    FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_VALIDATION,
    assess_candidate_writer_length_repair_eligibility,
    execute_candidate_writer_length_repair,
    parse_candidate_writer_length_repair_response,
    render_candidate_writer_length_repair_prompt_input,
)
from services.packaging.linkedin_post_candidate_writer_output_adapter import (
    CandidateWriterOutputAdaptationError,
    adapt_candidate_writer_payload,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)


class CandidateWriterLengthRepairTests(SimpleTestCase):
    def test_eligible_only_for_sole_post_text_above_max_length(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301)
        error = _adaptation_error(candidate)

        eligibility = assess_candidate_writer_length_repair_eligibility(
            parsed_candidate=candidate,
            adaptation_error=error,
        )

        self.assertTrue(eligibility.eligible)
        self.assertEqual(eligibility.original_post_text_length, 1301)
        self.assertEqual(eligibility.maximum_allowed, FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS)

    def test_ineligible_for_multiple_invalid_fields(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301, hook_variants=["too few"])
        error = _adaptation_error(candidate)

        eligibility = assess_candidate_writer_length_repair_eligibility(
            parsed_candidate=candidate,
            adaptation_error=error,
        )

        self.assertFalse(eligibility.eligible)
        self.assertIn("invalid fields are not exactly post_text", eligibility.reason)

    def test_ineligible_for_wrong_type_post_text(self) -> None:
        candidate = _candidate_payload(post_text=["not", "text"])
        error = _adaptation_error(candidate)

        eligibility = assess_candidate_writer_length_repair_eligibility(
            parsed_candidate=candidate,
            adaptation_error=error,
        )

        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "post_text is not a string")

    def test_ineligible_for_missing_fields(self) -> None:
        candidate = {"post_text": "A" * 1301}
        error = _adaptation_error(candidate)

        eligibility = assess_candidate_writer_length_repair_eligibility(
            parsed_candidate=candidate,
            adaptation_error=error,
        )

        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "parsed candidate is missing required fields")

    def test_ineligible_for_unexpected_fields(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301)
        candidate["debug"] = "not allowed"
        error = _adaptation_error(candidate)

        eligibility = assess_candidate_writer_length_repair_eligibility(
            parsed_candidate=candidate,
            adaptation_error=error,
        )

        self.assertFalse(eligibility.eligible)
        self.assertEqual(eligibility.reason, "parsed candidate has unexpected fields")

    def test_repair_response_accepts_exactly_one_post_text_key(self) -> None:
        provider_reply = _provider_reply(
            json.dumps({"post_text": "Short repaired post."})
        )

        self.assertEqual(
            parse_candidate_writer_length_repair_response(provider_reply),
            "Short repaired post.",
        )

    def test_repair_response_rejects_extra_keys(self) -> None:
        provider_reply = _provider_reply(
            json.dumps({"post_text": "Short repaired post.", "hashtags": []})
        )

        with self.assertRaisesMessage(
            ValueError,
            "candidate writer length repair response must contain only post_text",
        ):
            parse_candidate_writer_length_repair_response(provider_reply)

    def test_repair_response_rejects_plain_text(self) -> None:
        provider_reply = _provider_reply("Short repaired post.")

        with self.assertRaisesMessage(
            ValueError,
            "candidate writer length repair response must be JSON",
        ):
            parse_candidate_writer_length_repair_response(provider_reply)

    def test_repair_response_rejects_malformed_json(self) -> None:
        provider_reply = _provider_reply('{"post_text": ')

        with self.assertRaisesMessage(
            ValueError,
            "candidate writer length repair response is malformed JSON",
        ):
            parse_candidate_writer_length_repair_response(provider_reply)

    def test_repair_response_rejects_empty_post_text(self) -> None:
        provider_reply = _provider_reply(json.dumps({"post_text": "   "}))

        with self.assertRaisesMessage(
            ValueError,
            "candidate writer length repair post_text must be a non-empty string",
        ):
            parse_candidate_writer_length_repair_response(provider_reply)

    def test_repair_response_rejects_still_over_maximum(self) -> None:
        provider_reply = _provider_reply(json.dumps({"post_text": "B" * 1301}))

        with self.assertRaisesMessage(
            ValueError,
            "candidate writer length repair post_text remains above maximum",
        ):
            parse_candidate_writer_length_repair_response(provider_reply)

    def test_successful_repair_replaces_only_post_text(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301)
        before = copy.deepcopy(candidate)
        client = FakeLengthRepairClient(
            _provider_response(json.dumps({"post_text": "Short repaired post."}))
        )

        result = execute_candidate_writer_length_repair(
            parsed_candidate=candidate,
            adaptation_error=_adaptation_error(candidate),
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_text="Repair prompt.",
            client=client,
        )

        self.assertIsNone(result.failure_code)
        self.assertTrue(result.repair_executed)
        self.assertEqual(client.call_count, 1)
        self.assertEqual(client.prompts[0].count("Repair prompt."), 1)
        repaired = result.repaired_candidate
        self.assertIsNotNone(repaired)
        self.assertEqual(repaired["post_text"], "Short repaired post.")
        for field_name in (
            "hook_variants",
            "cta_variants",
            "hashtags",
            "quality_checks",
            "carousel_outline",
        ):
            self.assertEqual(repaired[field_name], before[field_name])
        self.assertEqual(candidate, before)

    def test_still_invalid_repair_returns_validation_failure(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301)
        client = FakeLengthRepairClient(
            _provider_response(json.dumps({"post_text": "B" * 1301}))
        )

        result = execute_candidate_writer_length_repair(
            parsed_candidate=candidate,
            adaptation_error=_adaptation_error(candidate),
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_text="Repair prompt.",
            client=client,
        )

        self.assertEqual(
            result.failure_code,
            FAILURE_CANDIDATE_WRITER_LENGTH_REPAIR_VALIDATION,
        )

    def test_render_includes_length_context_and_selected_evidence(self) -> None:
        candidate = _candidate_payload(post_text="A" * 1301)

        render = render_candidate_writer_length_repair_prompt_input(
            parsed_candidate=candidate,
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
        )

        self.assertIn("CURRENT_POST_TEXT_JSON", render.input_text)
        self.assertIn("LENGTH_CONSTRAINTS_JSON", render.input_text)
        self.assertIn("AUTHORIAL_VOICE_DIRECTIVE_JSON", render.input_text)
        self.assertIn("ev-1", render.variables["selected_evidence_json"])
        self.assertIn("1301", render.variables["length_constraints_json"])


class FakeLengthRepairClient:
    def __init__(self, response: SimpleNamespace) -> None:
        self.response = response
        self.call_count = 0
        self.prompts: list[str] = []

    def generate_text(self, **kwargs) -> SimpleNamespace:
        self.call_count += 1
        self.prompts.append(kwargs["prompt"])
        self.provider = kwargs.get("provider")
        self.max_output_tokens = kwargs["max_output_tokens"]
        self.json_mode = kwargs["json_mode"]
        self.thinking_mode = kwargs["thinking_mode"]
        return self.response


def _adaptation_error(candidate: dict) -> CandidateWriterOutputAdaptationError:
    try:
        adapt_candidate_writer_payload(candidate)
    except CandidateWriterOutputAdaptationError as exc:
        return exc
    raise AssertionError("candidate unexpectedly adapted")


def _provider_reply(raw_text: str) -> CandidateWriterRawResponse:
    return CandidateWriterRawResponse(
        raw_text=raw_text,
        provider="openai",
        model="gpt-4.1-2025-04-14",
    )


def _provider_response(raw_text: str) -> SimpleNamespace:
    return SimpleNamespace(
        text=raw_text,
        raw={"id": "resp-1"},
        usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        provider_response_metadata={},
    )


def _candidate_payload(*, post_text: object, hook_variants: list[str] | None = None) -> dict:
    return {
        "post_text": post_text,
        "hook_variants": hook_variants
        or [
            "A practical remote work policy starts here.",
            "Remote policy is not just a document.",
            "Hybrid work needs clearer operating habits.",
        ],
        "cta_variants": [
            "What would you clarify first in a remote policy?",
            "Where does your team still need shared expectations?",
            "What makes hybrid work sustainable in your organization?",
        ],
        "hashtags": ["#remotework", "#futureofwork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }


def _post_brief() -> dict:
    return {
        "core_point": "Clear remote work policies need human operating habits.",
        "evidence_to_use": [
            {
                "evidence_id": "ev-1",
                "evidence_text": "Remote work policies need clear expectations.",
                "role_in_post": "proof",
            }
        ],
    }


def _angle_decision() -> dict:
    return {
        "controlling_angle": "Remote policies need clarity and inclusion.",
        "authorial_voice_directive": {
            "authorial_observation": "Policy clarity is not the same as support.",
            "rejected_reading": "Do not treat documentation as the whole answer.",
            "why_distinction_matters": "Remote work needs both clarity and care.",
            "personal_presence_requirement": "explicit_author_owned_statement_required",
            "first_person_policy": "allowed_not_required",
            "forbidden_author_claims": ["personal experience"],
        },
    }
