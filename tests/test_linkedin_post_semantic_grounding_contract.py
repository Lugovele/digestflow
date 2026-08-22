from __future__ import annotations

import copy
import json

from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_contract import (
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_PASS,
    SEMANTIC_GROUNDING_STATUS_SEVERITY_MATRIX,
    SUPPORT_STATUS_CONTRADICTED,
    SUPPORT_STATUS_CAUSAL_OVERREACH,
    SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION,
    SUPPORT_STATUS_NOT_CLAIM,
    SUPPORT_STATUS_NOT_EVALUABLE,
    SUPPORT_STATUS_PARTIALLY_SUPPORTED,
    SUPPORT_STATUS_SUPPORTED,
    SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION,
    SUPPORT_STATUS_UNSUPPORTED,
    FinalPostSemanticGroundingState,
    SemanticGroundingReviewResult,
    build_semantic_grounding_prompt_rules,
    is_blocking_claim_state,
    is_valid_status_severity_pair,
    normalize_semantic_grounding_review_result,
)


class LinkedInPostSemanticGroundingContractTests(SimpleTestCase):
    def test_status_severity_matrix_lists_every_support_status(self) -> None:
        self.assertEqual(
            set(SEMANTIC_GROUNDING_STATUS_SEVERITY_MATRIX),
            {
                SUPPORT_STATUS_NOT_CLAIM,
                SUPPORT_STATUS_SUPPORTED,
                SUPPORT_STATUS_SUPPORTED_WITH_REQUIRED_QUALIFICATION,
                SUPPORT_STATUS_PARTIALLY_SUPPORTED,
                SUPPORT_STATUS_MISSING_REQUIRED_QUALIFICATION,
                SUPPORT_STATUS_CAUSAL_OVERREACH,
                SUPPORT_STATUS_UNSUPPORTED,
                SUPPORT_STATUS_CONTRADICTED,
                SUPPORT_STATUS_NOT_EVALUABLE,
            },
        )

    def test_status_severity_pair_helpers_follow_matrix(self) -> None:
        self.assertTrue(
            is_valid_status_severity_pair(SUPPORT_STATUS_PARTIALLY_SUPPORTED, "minor")
        )
        self.assertTrue(
            is_valid_status_severity_pair(SUPPORT_STATUS_PARTIALLY_SUPPORTED, "major")
        )
        self.assertFalse(
            is_valid_status_severity_pair(SUPPORT_STATUS_SUPPORTED, "major")
        )
        self.assertTrue(
            is_blocking_claim_state(SUPPORT_STATUS_PARTIALLY_SUPPORTED, "major")
        )
        self.assertFalse(
            is_blocking_claim_state(SUPPORT_STATUS_PARTIALLY_SUPPORTED, "minor")
        )

    def test_prompt_rules_payload_is_json_safe_and_defensive(self) -> None:
        rules = build_semantic_grounding_prompt_rules()
        rules["status_severity_matrix"][SUPPORT_STATUS_SUPPORTED].append("major")

        json.dumps(build_semantic_grounding_prompt_rules(), allow_nan=False)
        self.assertNotIn(
            "major",
            build_semantic_grounding_prompt_rules()["status_severity_matrix"][
                SUPPORT_STATUS_SUPPORTED
            ],
        )

    def test_directly_supported_claim_passes(self) -> None:
        result = normalize_semantic_grounding_review_result(
            _review_payload(),
            selected_evidence_ids=("a0-summary",),
        )

        self.assertIsInstance(result, SemanticGroundingReviewResult)
        self.assertTrue(result.passed)
        self.assertEqual(result.blocking_claim_ids, ())
        self.assertEqual(result.claim_reviews[0].support_status, SUPPORT_STATUS_SUPPORTED)

    def test_qualified_forecast_remains_qualified(self) -> None:
        payload = _review_payload(
            claims=[
                _claim(
                    claim_type="forecast_or_projection",
                    support_status="supported_with_required_qualification",
                    required_qualifications=["projected"],
                )
            ],
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.claim_reviews[0].required_qualifications, ("projected",))

    def test_partially_supported_minor_is_advisory_and_can_pass(self) -> None:
        result = normalize_semantic_grounding_review_result(
            _review_payload(
                claims=[
                    _claim(
                        support_status=SUPPORT_STATUS_PARTIALLY_SUPPORTED,
                        severity="minor",
                        repair_hint="Optional wording can be tightened.",
                    )
                ],
            ),
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.blocking_claim_ids, ())

    def test_partially_supported_major_is_blocking(self) -> None:
        result = normalize_semantic_grounding_review_result(
            _review_payload(
                passed=False,
                claims=[
                    _claim(
                        support_status=SUPPORT_STATUS_PARTIALLY_SUPPORTED,
                        severity="major",
                        repair_hint="Narrow the claim to the supported portion.",
                    )
                ],
                failed_claim_ids=["c1"],
                automatic_fail_reason="material overstatement",
                repairable=True,
                repair_instructions=["c1: Narrow the claim to the supported portion."],
            ),
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ("c1",))

    def test_invalid_support_status_severity_pair_fails(self) -> None:
        payload = _review_payload(
            claims=[_claim(support_status=SUPPORT_STATUS_SUPPORTED, severity="major")]
        )

        with self.assertRaisesRegex(ValueError, "support_status/severity pair"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_missing_likelihood_qualification_fails(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status="missing_required_qualification",
                    severity="major",
                    required_qualifications=["likely"],
                    missing_qualifications=["likely"],
                    repair_hint="Restore likely attribution.",
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="missing required qualification",
            repairable=True,
            repair_instructions=["c1: Restore likely attribution."],
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ("c1",))
        self.assertEqual(result.claim_reviews[0].missing_qualifications, ("likely",))

    def test_unselected_evidence_cannot_support_claim(self) -> None:
        payload = _review_payload(
            claims=[_claim(evidence_ids=["a9-unselected"])],
        )

        with self.assertRaisesRegex(ValueError, "unselected evidence ID"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_automatic_fail_overrides_pass(self) -> None:
        payload = _review_payload(automatic_fail_reason="unsupported claim")

        with self.assertRaisesRegex(ValueError, "automatic_fail_reason"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_human_review_cannot_be_clean_grounding_pass(self) -> None:
        payload = _review_payload(
            requires_human_review=True,
            human_review_reason="Needs human factuality review.",
        )

        with self.assertRaisesRegex(ValueError, "human review is required"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_false_without_blocking_reason_or_human_review_fails(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status="supported_with_required_qualification",
                    severity="info",
                    required_qualifications=["likely"],
                ),
                _claim(
                    claim_id="c2",
                    support_status="not_claim",
                    severity="info",
                ),
            ],
            failed_claim_ids=[],
            automatic_fail_reason="",
            requires_human_review=False,
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "pass=false requires"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_true_with_blocking_claim_fails(self) -> None:
        payload = _review_payload(
            passed=True,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=["c1"],
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "pass cannot be true"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_true_with_repairable_flag_fails(self) -> None:
        payload = _review_payload(repairable=True)

        with self.assertRaisesRegex(ValueError, "repairable"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_true_with_repair_instructions_fails(self) -> None:
        payload = _review_payload(
            repairable=False,
            repair_instructions=["Repair text despite pass."],
        )

        with self.assertRaisesRegex(ValueError, "repair_instructions"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_failed_grounding_requires_repair_instructions(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported claim",
            repairable=True,
            repair_instructions=[],
        )

        with self.assertRaisesRegex(ValueError, "repair_instructions"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_automatic_fail_reason_requires_blocking_claims(self) -> None:
        payload = _review_payload(
            passed=False,
            automatic_fail_reason="generic automatic fail without a claim",
        )

        with self.assertRaisesRegex(ValueError, "automatic_fail_reason requires"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_blocking_claim_requires_repair_hint(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                    repair_hint="",
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported claim",
            repairable=True,
            repair_instructions=["c1: Remove unsupported claim."],
        )

        with self.assertRaisesRegex(ValueError, "repair_hint"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_instructions_must_be_claim_addressed(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                    repair_hint="Remove unsupported claim.",
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported claim",
            repairable=True,
            repair_instructions=["Remove unsupported claim."],
        )

        with self.assertRaisesRegex(ValueError, "start with claim_id"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_instructions_must_match_each_failed_claim(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                    repair_hint="Remove unsupported claim.",
                ),
                _claim(
                    claim_id="c2",
                    support_status=SUPPORT_STATUS_CAUSAL_OVERREACH,
                    severity="major",
                    repair_hint="Remove unsupported causality.",
                ),
            ],
            failed_claim_ids=["c1", "c2"],
            automatic_fail_reason="unsupported claims",
            repairable=True,
            repair_instructions=["c1: Remove unsupported claim."],
        )

        with self.assertRaisesRegex(ValueError, "address each blocking claim"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_failed_grounding_requires_blocking_claims(
        self,
    ) -> None:
        payload = _review_payload(
            passed=False,
            claims=[_claim()],
            failed_claim_ids=[],
            automatic_fail_reason="",
            repairable=True,
            repair_instructions=["c1: Repair unsupported claim."],
        )

        with self.assertRaisesRegex(ValueError, "pass=false requires blocking"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_failed_grounding_with_human_review_fails(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported claim",
            requires_human_review=True,
            human_review_reason="Needs human factuality review.",
            repairable=True,
            repair_instructions=["c1: Repair unsupported claim."],
        )

        with self.assertRaisesRegex(ValueError, "human review"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_human_review_requires_reason(self) -> None:
        payload = _review_payload(
            passed=False,
            requires_human_review=True,
            human_review_reason="",
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "human_review_reason"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_bitcoin_causal_drift_is_not_a_clean_grounding_pass(self) -> None:
        payload = {
            "pass": False,
            "claims": [
                _claim(
                    claim_id="c1",
                    claim_text="optimism builds",
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                    rationale="Selected evidence does not state optimism is building.",
                    repair_hint="Remove the unsupported optimism bridge claim.",
                ),
                _claim(
                    claim_id="c2",
                    claim_text="some stability emerges",
                    support_status="missing_required_qualification",
                    severity="major",
                    required_qualifications=["likely", "may", "risk remains"],
                    missing_qualifications=["likely", "may", "risk remains"],
                    rationale="Likely bottoming and risk remaining cannot become stability.",
                    repair_hint="Keep K33's likelihood and risk language.",
                ),
                _claim(
                    claim_id="c3",
                    claim_text="caution influencing the pace of recovery",
                    claim_type="causal_claim",
                    support_status=SUPPORT_STATUS_CAUSAL_OVERREACH,
                    severity="major",
                    rationale="Evidence says pessimistic positioning may limit deeper downside, not cause recovery pace.",
                    repair_hint="Remove recovery causality.",
                ),
                _claim(
                    claim_id="c4",
                    claim_text="caution influencing the path of future growth",
                    claim_type="causal_claim",
                    support_status=SUPPORT_STATUS_CAUSAL_OVERREACH,
                    severity="major",
                    rationale="Evidence does not connect trader caution to future growth.",
                    repair_hint="Remove future growth causality.",
                ),
            ],
            "failed_claim_ids": ["c1", "c2", "c3", "c4"],
            "automatic_fail_reason": "unsupported strengthened Bitcoin claims",
            "requires_human_review": False,
            "human_review_reason": "",
            "repairable": True,
            "repair_instructions": [
                "c1: Remove the unsupported optimism bridge claim.",
                "c2: Keep K33's likelihood and risk language.",
                "c3: Remove recovery causality.",
                "c4: Remove future growth causality.",
            ],
        }

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary", "a1-kp0", "a2-summary"),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ("c1", "c2", "c3", "c4"))
        self.assertIn("unsupported strengthened Bitcoin", result.automatic_fail_reason)

    def test_state_to_dict_is_json_safe_and_defensive(self) -> None:
        result = normalize_semantic_grounding_review_result(
            _review_payload(),
            selected_evidence_ids=("a0-summary",),
        )
        state = FinalPostSemanticGroundingState(
            status=GROUNDING_STATUS_PASS,
            grounding_review=result,
            metadata={"raw": {"id": "resp"}},
        )
        serialized = state.to_dict()
        serialized["metadata"]["raw"]["id"] = "changed"

        json.dumps(state.to_dict(), allow_nan=False, sort_keys=True)
        self.assertEqual(state.metadata, {"raw": {"id": "resp"}})

    def test_failed_claim_ids_must_match_claims(self) -> None:
        payload = _review_payload(
            passed=False,
            failed_claim_ids=["missing"],
            automatic_fail_reason="unsupported",
        )

        with self.assertRaisesRegex(ValueError, "unknown claim"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_failed_claim_ids_must_reference_blocking_claims(self) -> None:
        payload = _review_payload(
            passed=False,
            failed_claim_ids=["c1"],
            automatic_fail_reason="unsupported",
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "must reference blocking claims"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_failed_claim_ids_must_include_each_derived_blocking_claim(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=[],
            automatic_fail_reason="unsupported claim",
            requires_human_review=False,
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "exactly match blocking claims"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_valid_human_review_grounding_result_normalizes(self) -> None:
        payload = _review_payload(
            passed=False,
            failed_claim_ids=[],
            automatic_fail_reason="",
            requires_human_review=True,
            human_review_reason="Needs human factuality review.",
            repairable=False,
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertTrue(result.requires_human_review)
        self.assertEqual(result.human_review_reason, "Needs human factuality review.")


def _review_payload(**overrides) -> dict:
    if "passed" in overrides:
        overrides["pass"] = overrides.pop("passed")
    payload = {
        "pass": True,
        "claims": [_claim()],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": False,
        "repair_instructions": [],
    }
    payload.update(overrides)
    return payload


def _claim(**overrides) -> dict:
    claim = {
        "claim_id": "c1",
        "field_name": "post_text",
        "value_index": None,
        "claim_text": "Bitcoin adoption is constrained by security concerns and volatility.",
        "claim_type": "attributed_source_claim",
        "support_status": SUPPORT_STATUS_SUPPORTED,
        "severity": "info",
        "supported_evidence_ids": ["a0-summary"],
        "required_qualifications": [],
        "missing_qualifications": [],
        "rationale": "Selected evidence states this directly.",
        "repair_hint": "",
    }
    claim.update(copy.deepcopy(overrides))
    if "evidence_ids" in claim:
        claim["supported_evidence_ids"] = claim.pop("evidence_ids")
    return claim
