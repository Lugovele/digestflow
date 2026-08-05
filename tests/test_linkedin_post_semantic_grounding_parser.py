from __future__ import annotations

import json

from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)
from services.packaging.linkedin_post_semantic_grounding_parser import (
    ERROR_EMPTY_RAW_RESPONSE,
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

    def test_empty_response_is_distinct(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw(" "))

        self.assertEqual(error.exception.code, ERROR_EMPTY_RAW_RESPONSE)

    def test_malformed_json_is_distinct(self) -> None:
        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_semantic_grounding_raw_response(_raw("{not-json"))

        self.assertEqual(error.exception.code, ERROR_MALFORMED_JSON)

    def test_normalization_failure_is_distinct(self) -> None:
        payload = _review_payload()
        payload["claims"][0]["supported_evidence_ids"] = ["unselected"]

        with self.assertRaises(SemanticGroundingResponseParseError) as error:
            parse_and_normalize_semantic_grounding_response(
                _raw(json.dumps(payload)),
                selected_evidence_ids=("a0-summary",),
            )

        self.assertEqual(error.exception.code, ERROR_NORMALIZATION_FAILED)

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


def _raw(raw_text: str) -> SemanticGroundingRawResponse:
    return SemanticGroundingRawResponse(
        raw_text=raw_text,
        provider="openai",
        model="grounding-model",
    )


def _review_payload(*, passed: bool = True) -> dict:
    return {
        "pass": passed,
        "claims": [
            {
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
        ],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": True,
        "repair_instructions": [],
    }
