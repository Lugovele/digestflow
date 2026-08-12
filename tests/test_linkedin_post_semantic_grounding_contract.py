from __future__ import annotations

import copy
import json

from django.test import SimpleTestCase

from services.packaging.linkedin_post_semantic_grounding_contract import (
    CLAIM_TYPE_AUTHOR_INTERPRETATION,
    CLAIM_TYPE_CAUSAL_CLAIM,
    CLAIM_TYPE_CTA_OR_RHETORICAL,
    CLAIM_TYPE_METRIC_OR_DATE,
    GROUNDING_STATUS_FAIL,
    GROUNDING_STATUS_PASS,
    SEMANTIC_GROUNDING_SEVERITIES,
    SEMANTIC_GROUNDING_SUPPORT_STATUSES,
    SUPPORT_STATUS_CAUSAL_OVERREACH,
    SUPPORT_STATUS_NOT_CLAIM,
    SUPPORT_STATUS_PARTIALLY_SUPPORTED,
    SUPPORT_STATUS_SUPPORTED,
    SUPPORT_STATUS_UNSUPPORTED,
    SEVERITY_BLOCKING,
    SEVERITY_MAJOR,
    SEVERITY_MINOR,
    FinalPostSemanticGroundingState,
    SemanticGroundingRepairInstruction,
    SemanticGroundingReviewResult,
    build_semantic_grounding_prompt_rules,
    normalize_semantic_grounding_review_result,
)


class LinkedInPostSemanticGroundingContractTests(SimpleTestCase):
    def test_prompt_rules_are_json_serializable_and_stable(self) -> None:
        rules = build_semantic_grounding_prompt_rules()

        first = json.dumps(rules, sort_keys=True)
        second = json.dumps(build_semantic_grounding_prompt_rules(), sort_keys=True)

        self.assertEqual(first, second)
        self.assertTrue(rules["authoritative"])
        self.assertTrue(rules["selected_evidence_only"])

    def test_prompt_rules_represent_every_status_severity_combination_once(self) -> None:
        rules = build_semantic_grounding_prompt_rules()
        combinations = rules["claim_decision_combinations"]

        observed = [
            (item["support_status"], item["severity"])
            for item in combinations
        ]

        self.assertEqual(len(observed), len(set(observed)))
        self.assertEqual(
            set(observed),
            {
                (support_status, severity)
                for support_status in SEMANTIC_GROUNDING_SUPPORT_STATUSES
                for severity in SEMANTIC_GROUNDING_SEVERITIES
            },
        )

    def test_prompt_rule_decisions_match_normalized_blocking_semantics(self) -> None:
        rules = build_semantic_grounding_prompt_rules()

        for combination in rules["claim_decision_combinations"]:
            with self.subTest(combination=combination):
                support_status = combination["support_status"]
                severity = combination["severity"]
                payload = _review_payload(
                    passed=not combination["blocking"],
                    claims=[
                        _claim(
                            support_status=support_status,
                            severity=severity,
                        )
                    ],
                    failed_claim_ids=(["c1"] if combination["blocking"] else []),
                    automatic_fail_reason=(
                        "blocking claim" if combination["blocking"] else ""
                    ),
                    repairable=combination["blocking"],
                    repair_instructions=(
                        [_repair_instruction("c1", "Repair blocking claim.")]
                        if combination["blocking"]
                        else []
                    ),
                )
                result = normalize_semantic_grounding_review_result(
                    payload,
                    selected_evidence_ids=("a0-summary",),
                )

                self.assertEqual(bool(result.blocking_claim_ids), combination["blocking"])
                self.assertEqual(
                    combination["decision"],
                    "blocking" if result.blocking_claim_ids else "advisory",
                )
                self.assertEqual(
                    combination["requires_failure"],
                    bool(result.blocking_claim_ids),
                )

    def test_prompt_rules_keep_known_advisory_and_blocking_combinations(self) -> None:
        rules = build_semantic_grounding_prompt_rules()
        combinations = {
            (item["support_status"], item["severity"]): item["blocking"]
            for item in rules["claim_decision_combinations"]
        }

        self.assertFalse(
            combinations[(SUPPORT_STATUS_PARTIALLY_SUPPORTED, SEVERITY_MINOR)]
        )
        self.assertTrue(combinations[(SUPPORT_STATUS_UNSUPPORTED, SEVERITY_MAJOR)])
        self.assertTrue(combinations[(SUPPORT_STATUS_CAUSAL_OVERREACH, SEVERITY_BLOCKING)])

    def test_prompt_rules_include_contract_consistency_guidance(self) -> None:
        rules = build_semantic_grounding_prompt_rules()

        self.assertTrue(
            rules["failure_rules"]["failed_claim_ids_must_match_blocking_claim_ids"]
        )
        self.assertTrue(rules["repairability_rules"]["repairable_requires_blocking_claims"])
        self.assertTrue(
            rules["repairability_rules"][
                "repair_instructions_must_reference_blocking_claims"
            ]
        )
        self.assertTrue(rules["human_review_rules"]["human_review_requires_pass_false"])
        self.assertTrue(
            rules["automatic_fail_rules"]["automatic_fail_reason_requires_pass_false"]
        )

    def test_prompt_rules_preserve_adoption_interest_causal_fidelity_guard(self) -> None:
        rules = build_semantic_grounding_prompt_rules()

        self.assertIn(
            "Do not turn adoption interest into mainstream inevitability.",
            rules["assessment_guidance"]["causal_fidelity"],
        )

    def test_prompt_rules_allow_source_bounded_authorial_synthesis_without_verbatim_wording(
        self,
    ) -> None:
        rules = build_semantic_grounding_prompt_rules()

        synthesis_rules = rules["assessment_guidance"][
            "source_bounded_authorial_synthesis"
        ]

        self.assertIn(
            "Do not mark source-bounded synthesis unsupported solely because the exact wording is absent from evidence.",
            synthesis_rules,
        )
        self.assertIn(
            "Authorial synthesis still fails when it invents facts, metrics, examples, actors, dates, causal mechanisms, or stronger conditions not present in selected evidence.",
            synthesis_rules,
        )

    def test_prompt_rules_require_mixed_claim_splitting(self) -> None:
        rules = build_semantic_grounding_prompt_rules()

        splitting_rules = rules["assessment_guidance"]["mixed_claim_splitting"]

        self.assertIn(
            "Split rhetorical or authorial framing from factual, causal, comparative, predictive, or prescriptive assertions.",
            splitting_rules,
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

    def test_mixed_authorial_sentence_can_split_framing_from_supported_claim(
        self,
    ) -> None:
        payload = _review_payload(
            claims=[
                _claim(
                    claim_id="c1",
                    claim_text="That distinction matters.",
                    claim_type=CLAIM_TYPE_CTA_OR_RHETORICAL,
                    support_status=SUPPORT_STATUS_NOT_CLAIM,
                    severity="info",
                    evidence_ids=[],
                    rationale="Authorial framing, not an evidence-bearing claim.",
                ),
                _claim(
                    claim_id="c2",
                    claim_text="The adoption number shows exposure.",
                    claim_type=CLAIM_TYPE_METRIC_OR_DATE,
                    support_status=SUPPORT_STATUS_SUPPORTED,
                    severity="info",
                    evidence_ids=["a0-summary"],
                    rationale="Selected evidence supports the adoption/exposure fact.",
                ),
            ],
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.blocking_claim_ids, ())
        self.assertEqual(result.claim_reviews[0].support_status, SUPPORT_STATUS_NOT_CLAIM)
        self.assertEqual(result.claim_reviews[1].support_status, SUPPORT_STATUS_SUPPORTED)

    def test_source_bounded_synthesis_can_pass_while_invented_causality_blocks(
        self,
    ) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    claim_id="c1",
                    claim_text="The stronger point is that adoption and impact are not the same thing.",
                    claim_type=CLAIM_TYPE_AUTHOR_INTERPRETATION,
                    support_status=SUPPORT_STATUS_SUPPORTED,
                    severity="info",
                    evidence_ids=["a0-summary"],
                    rationale="This is bounded by the selected adoption evidence.",
                ),
                _claim(
                    claim_id="c2",
                    claim_text="Frameworks turn exposure into student learning outcomes.",
                    claim_type=CLAIM_TYPE_CAUSAL_CLAIM,
                    support_status=SUPPORT_STATUS_CAUSAL_OVERREACH,
                    severity=SEVERITY_MAJOR,
                    evidence_ids=["a0-summary"],
                    rationale="The selected evidence does not support that causal mechanism.",
                    repair_hint="Remove the invented causal mechanism.",
                ),
            ],
            failed_claim_ids=["c2"],
            automatic_fail_reason="invented causal mechanism",
            repairable=True,
            repair_instructions=[
                _repair_instruction("c2", "Remove the invented causal mechanism.")
            ],
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ("c2",))
        self.assertEqual(result.claim_reviews[0].support_status, SUPPORT_STATUS_SUPPORTED)
        self.assertEqual(result.claim_reviews[1].support_status, SUPPORT_STATUS_CAUSAL_OVERREACH)

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

    def test_missing_likelihood_qualification_fails(self) -> None:
        payload = _review_payload(
            passed=False,
            claims=[
                _claim(
                    support_status="missing_required_qualification",
                    severity="major",
                    required_qualifications=["likely"],
                    missing_qualifications=["likely"],
                )
            ],
            failed_claim_ids=["c1"],
            automatic_fail_reason="missing required qualification",
            repairable=True,
            repair_instructions=[_repair_instruction("c1", "Restore likely attribution.")],
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ("c1",))
        self.assertEqual(result.claim_reviews[0].missing_qualifications, ("likely",))
        self.assertEqual(
            result.repair_instructions,
            (
                SemanticGroundingRepairInstruction(
                    claim_id="c1",
                    instruction="Restore likely attribution.",
                ),
            ),
        )

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

    def test_pass_with_blocking_claim_is_rejected(self) -> None:
        payload = _review_payload(
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=["c1"],
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "pass cannot be true with failed claims"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_with_failed_claim_ids_is_rejected(self) -> None:
        payload = _review_payload(failed_claim_ids=["c1"])

        with self.assertRaisesRegex(ValueError, "failed_claim_ids must exactly match"):
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

    def test_pass_with_repairable_true_is_rejected(self) -> None:
        payload = _review_payload(repairable=True)

        with self.assertRaisesRegex(ValueError, "pass cannot be true when repairable"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_pass_with_repair_instructions_is_rejected(self) -> None:
        payload = _review_payload(
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                )
            ],
            failed_claim_ids=["c1"],
            repairable=True,
            repair_instructions=[_repair_instruction("c1", "Remove unsupported claim.")],
        )

        with self.assertRaisesRegex(ValueError, "pass cannot be true with failed claims"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_bare_fail_with_no_signal_is_rejected(self) -> None:
        payload = _review_payload(passed=False, repairable=False)

        with self.assertRaisesRegex(ValueError, "fail requires a failure signal"):
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

    def test_repairable_failed_grounding_requires_instructions_even_without_failed_claim_ids(
        self,
    ) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[],
        )

        with self.assertRaisesRegex(ValueError, "repair_instructions"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_failed_claim_ids_missing_blocker_are_rejected(self) -> None:
        payload = _blocking_repair_payload(
            failed_claim_ids=[],
            repairable=False,
            repair_instructions=[],
        )

        with self.assertRaisesRegex(ValueError, "failed_claim_ids must exactly match"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_failed_claim_ids_with_extra_advisory_claim_are_rejected(self) -> None:
        payload = _blocking_repair_payload(
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                ),
                _claim(claim_id="c2", support_status=SUPPORT_STATUS_SUPPORTED),
            ],
            failed_claim_ids=["c1", "c2"],
        )

        with self.assertRaisesRegex(ValueError, "failed_claim_ids must exactly match"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_duplicate_failed_claim_ids_are_rejected(self) -> None:
        payload = _blocking_repair_payload(failed_claim_ids=["c1", "c1"])

        with self.assertRaisesRegex(ValueError, "duplicate failed_claim_id"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_non_canonical_failed_claim_id_order_is_rejected(self) -> None:
        payload = _blocking_repair_payload(
            claims=[
                _claim(
                    claim_id="c1",
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                ),
                _claim(
                    claim_id="c2",
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                ),
            ],
            failed_claim_ids=["c2", "c1"],
            repair_instructions=[
                _repair_instruction("c1", "Remove first unsupported claim."),
                _repair_instruction("c2", "Remove second unsupported claim."),
            ],
        )

        with self.assertRaisesRegex(ValueError, "failed_claim_ids must exactly match"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_human_review_reason_when_review_false_is_rejected(self) -> None:
        payload = _review_payload(
            passed=False,
            automatic_fail_reason="manual context required",
            requires_human_review=False,
            human_review_reason="Needs human review.",
            repairable=False,
        )

        with self.assertRaisesRegex(ValueError, "human_review_reason must be empty"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repairable_without_blockers_is_rejected(self) -> None:
        payload = _review_payload(
            passed=False,
            automatic_fail_reason="automatic failure",
            repairable=True,
        )

        with self.assertRaisesRegex(ValueError, "repairable requires blocking claims"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repair_instructions_with_repairable_false_are_rejected(self) -> None:
        payload = _blocking_repair_payload(repairable=False)

        with self.assertRaisesRegex(ValueError, "repair_instructions require repairable"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_advisory_only_result_may_pass(self) -> None:
        payload = _review_payload(
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="minor",
                )
            ]
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertTrue(result.passed)
        self.assertEqual(result.blocking_claim_ids, ())

    def test_automatic_fail_only_result_is_valid_when_not_repairable(self) -> None:
        payload = _review_payload(
            passed=False,
            automatic_fail_reason="non-claim automatic failure",
            repairable=False,
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.blocking_claim_ids, ())
        self.assertFalse(result.repairable)

    def test_human_review_only_result_is_valid_when_reason_exists(self) -> None:
        payload = _review_payload(
            passed=False,
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
        self.assertEqual(result.blocking_claim_ids, ())

    def test_valid_structured_repair_instruction_to_dict(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[_repair_instruction("c1", "Remove unsupported claim.")]
        )

        result = normalize_semantic_grounding_review_result(
            payload,
            selected_evidence_ids=("a0-summary",),
        )

        self.assertEqual(
            result.to_dict()["repair_instructions"],
            [{"claim_id": "c1", "instruction": "Remove unsupported claim."}],
        )

    def test_repair_instruction_empty_claim_id_is_rejected(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[_repair_instruction("", "Remove unsupported claim.")]
        )

        with self.assertRaisesRegex(ValueError, "claim_id"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repair_instruction_empty_instruction_is_rejected(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[_repair_instruction("c1", " ")]
        )

        with self.assertRaisesRegex(ValueError, "instruction"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repair_instruction_unknown_claim_is_rejected(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[
                _repair_instruction("missing", "Remove unsupported claim.")
            ]
        )

        with self.assertRaisesRegex(ValueError, "unknown claim"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_repair_instruction_non_blocking_claim_is_rejected(self) -> None:
        payload = _blocking_repair_payload(
            claims=[
                _claim(
                    support_status=SUPPORT_STATUS_UNSUPPORTED,
                    severity="major",
                ),
                _claim(claim_id="c2", support_status=SUPPORT_STATUS_SUPPORTED),
            ],
            repair_instructions=[_repair_instruction("c2", "Revise advisory claim.")],
        )

        with self.assertRaisesRegex(ValueError, "non-blocking claim"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_duplicate_repair_instruction_claim_references_are_rejected(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=[
                _repair_instruction("c1", "Remove unsupported claim."),
                _repair_instruction("c1", "Qualify unsupported claim."),
            ]
        )

        with self.assertRaisesRegex(ValueError, "duplicate repair instruction claim_id"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_legacy_string_repair_instruction_payload_fails(self) -> None:
        payload = _blocking_repair_payload(
            repair_instructions=["Remove unsupported claim."]
        )

        with self.assertRaisesRegex(ValueError, "repair_instructions entries must be objects"):
            normalize_semantic_grounding_review_result(
                payload,
                selected_evidence_ids=("a0-summary",),
            )

    def test_automatic_fail_only_result_cannot_emit_repair_instructions(self) -> None:
        payload = _review_payload(
            passed=False,
            failed_claim_ids=[],
            automatic_fail_reason="non-claim failure",
            repairable=False,
            repair_instructions=[_repair_instruction("c1", "Repair automatic failure.")],
        )

        with self.assertRaisesRegex(ValueError, "non-blocking claim"):
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
                _repair_instruction(
                    "c1",
                    "Remove unsupported optimism bridge claim.",
                ),
                _repair_instruction(
                    "c2",
                    "Keep K33's likelihood and risk language.",
                ),
                _repair_instruction("c3", "Remove recovery causality."),
                _repair_instruction("c4", "Remove future growth causality."),
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


def _blocking_repair_payload(**overrides) -> dict:
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
        repair_instructions=[_repair_instruction("c1", "Remove unsupported claim.")],
    )
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


def _repair_instruction(claim_id: str, instruction: str) -> dict:
    return {
        "claim_id": claim_id,
        "instruction": instruction,
    }
