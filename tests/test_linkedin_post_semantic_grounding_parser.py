from __future__ import annotations

import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    ERROR_EMPTY_RAW_RESPONSE,
    ERROR_MALFORMED_FENCE,
    ERROR_MALFORMED_JSON,
    ERROR_NORMALIZATION_FAILED,
    SemanticGroundingResponseParseError,
    parse_and_normalize_semantic_grounding_response,
    parse_semantic_grounding_raw_response,
)


class LinkedInPostSemanticGroundingParserTests(SimpleTestCase):
    def test_parses_json_object(self) -> None:
        payload = {"pass": True, "claims": [], "failed_claim_ids": []}

        parsed = parse_semantic_grounding_raw_response(_raw(json.dumps(payload)))

        self.assertEqual(parsed, payload)

    def test_parses_fenced_json_object(self) -> None:
        payload = _review_payload()
        raw = _raw("```json\n" + json.dumps(payload) + "\n```")

        result = parse_and_normalize_semantic_grounding_response(
            raw,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)

    def test_parses_generic_fenced_json_object(self) -> None:
        payload = _review_payload()
        raw = _raw("```\n" + json.dumps(payload) + "\n```")

        result = parse_and_normalize_semantic_grounding_response(
            raw,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)

    def test_surrounding_whitespace_is_allowed(self) -> None:
        payload = {"pass": True, "claims": [], "failed_claim_ids": []}

        parsed = parse_semantic_grounding_raw_response(
            _raw("  \n" + json.dumps(payload) + "\n  ")
        )

        self.assertEqual(parsed, payload)

    def test_prose_before_or_after_json_is_rejected(self) -> None:
        for raw_text in (
            "Here is JSON " + json.dumps(_review_payload()),
            json.dumps(_review_payload()) + " trailing prose",
        ):
            with self.subTest(raw_text=raw_text):
                with self.assertRaises(SemanticGroundingResponseParseError) as error:
                    parse_semantic_grounding_raw_response(_raw(raw_text))

                self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_multiple_json_objects_are_rejected(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw('{"pass":true}{"pass":false}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_incomplete_json_is_rejected(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw('{"pass":'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_unclosed_fence_is_rejected_as_malformed_fence(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw('```json\n{"pass":true}'))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_FENCE)

    def test_empty_response_is_distinct(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw(" "))

        self.assertEqual(error.exception.code, ERROR_EMPTY_RAW_RESPONSE)

    def test_malformed_json_is_distinct(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw("{not-json"))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)
        self.assertEqual(error.exception.line, 1)
        self.assertEqual(error.exception.column, 2)
        self.assertEqual(error.exception.position, 1)

    def test_normalization_failure_is_distinct(self) -> None:
        payload = _review_payload()
        payload["claims"][0]["supported_evidence_ids"] = ["unselected"]

        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_and_normalize_semantic_grounding_response(
                _raw(json.dumps(payload)),
                selected_evidence_ids=("a0-summary",),
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)
        self.assertEqual(
            error.exception.normalization_error,
            "semantic grounding claim c1 references unselected evidence ID: unselected",
        )

    def test_normalization_failure_redacts_sensitive_cause_details(self) -> None:
        sensitive_messages = (
            "x-api-key sk-test leaked in error",
            "authorization token leaked in error",
            "password credential leaked in error",
        )

        for message in sensitive_messages:
            with self.subTest(message=message):
                with patch(
                    "services.packaging.linkedin_post_semantic_grounding_parser.normalize_semantic_grounding_review_result",
                    side_effect=ValueError(message),
                ):
                    with self.assertRaises(SemanticGroundingResponseParseError) as error:
                        parse_and_normalize_semantic_grounding_response(
                            _raw(json.dumps(_review_payload())),
                            selected_evidence_ids=("a0-summary",),
                        )

                self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)
                self.assertEqual(
                    error.exception.normalization_error,
                    "redacted safe normalization error",
                )

    def test_legacy_string_repair_instruction_is_normalization_failure(self) -> None:
        payload = _review_payload(passed=False)
        payload["claims"][0]["support_status"] = "unsupported"
        payload["claims"][0]["severity"] = "major"
        payload["failed_claim_ids"] = ["c1"]
        payload["automatic_fail_reason"] = "unsupported claim"
        payload["repair_instructions"] = ["Remove unsupported claim."]

        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_and_normalize_semantic_grounding_response(
                _raw(json.dumps(payload)),
                selected_evidence_ids=("a0-summary",),
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)

    def test_invalid_consistency_combinations_are_normalization_failures(self) -> None:
        invalid_payloads = [
            _review_payload(repairable=True),
            _review_payload(passed=False),
            _review_payload(
                passed=False,
                automatic_fail_reason="automatic failure",
                human_review_reason="stale reason",
            ),
            _review_payload(
                passed=False,
                claims=[
                    {
                        **_claim_payload(),
                        "support_status": "unsupported",
                        "severity": "major",
                    }
                ],
                failed_claim_ids=[],
                automatic_fail_reason="unsupported claim",
                repairable=False,
            ),
        ]

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                with self.assertRaises(SemanticGroundingResponseParseError) as error:
                    parse_and_normalize_semantic_grounding_response(
                        _raw(json.dumps(payload)),
                        selected_evidence_ids=("a0-summary",),
                    )

                self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)
                self.assertTrue(error.exception.normalization_error)


def _raw(raw_text: str) -> SemanticGroundingRawResponse:
    return SemanticGroundingRawResponse(
        raw_text=raw_text,
        provider="openai",
        model="grounding-model",
    )


def _review_payload(*, passed: bool = True, **overrides) -> dict:
    payload = {
        "pass": passed,
        "claims": [_claim_payload()],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": False,
        "repair_instructions": [],
    }
    payload.update(overrides)
    return payload


def _claim_payload() -> dict:
    return {
        "claim_id": "c1",
        "field_name": "post_text",
        "value_index": None,
        "claim_text": "Bitcoin adoption has security and volatility constraints.",
        "claim_type": "attributed_source_claim",
        "support_status": "supported",
        "severity": "info",
        "supported_evidence_ids": ["a0-summary"],
        "required_qualifications": [],
        "missing_qualifications": [],
        "rationale": "Directly supported.",
        "repair_hint": "",
    }
