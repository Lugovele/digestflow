from __future__ import annotations

import copy
from dataclasses import dataclass
import inspect
import json
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_input_builders
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_editorial_boundary import (
    PostEditorialInput,
    PostGenerationMetadata,
    PromptMetadata,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
)
from services.packaging.linkedin_post_flow_input_builders import (
    CandidateWriterInput,
    build_candidate_writer_input,
    build_post_editorial_input,
)


@dataclass(frozen=True)
class EvidenceUseStub:
    evidence_id: str
    evidence_text: str
    role_in_post: str


@dataclass(frozen=True)
class PostBriefStub:
    core_point: str
    evidence_to_use: list[EvidenceUseStub]


@dataclass(frozen=True)
class AngleDecisionStub:
    controlling_angle: str
    supporting_evidence_ids: list[str]
    authorial_voice_directive: dict[str, object] | None = None


class LinkedInPostFlowInputBuildersTests(SimpleTestCase):
    def test_build_candidate_writer_input_returns_contract(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertIsInstance(candidate_input, CandidateWriterInput)

    def test_selected_evidence_is_derived_from_post_brief_evidence_to_use(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertEqual(len(candidate_input.selected_evidence), 2)
        self.assertEqual(
            candidate_input.selected_evidence,
            (
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Remote teams need clear operating agreements.",
                    "role_in_post": "opening support",
                },
                {
                    "evidence_id": "a1-kp0",
                    "evidence_text": "Isolation can rise when remote work is unmanaged.",
                    "role_in_post": "practical tension",
                },
            ),
        )

    def test_selected_evidence_id_order_is_preserved(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertEqual(
            [item["evidence_id"] for item in candidate_input.selected_evidence],
            ["a0-summary", "a1-kp0"],
        )

    def test_selected_evidence_comes_from_post_brief_when_angle_decision_disagrees(self) -> None:
        angle_decision = AngleDecisionStub(
            controlling_angle="Make remote work explicit.",
            supporting_evidence_ids=["a-extra", "a1-kp0"],
            authorial_voice_directive=_angle_decision().authorial_voice_directive,
        )

        candidate_input = build_candidate_writer_input(_post_brief(), angle_decision)

        self.assertEqual(
            [item["evidence_id"] for item in candidate_input.selected_evidence],
            ["a0-summary", "a1-kp0"],
        )
        self.assertNotIn(
            "a-extra",
            [item["evidence_id"] for item in candidate_input.selected_evidence],
        )

    def test_evidence_text_is_preserved(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertEqual(
            [item["evidence_text"] for item in candidate_input.selected_evidence],
            [
                "Remote teams need clear operating agreements.",
                "Isolation can rise when remote work is unmanaged.",
            ],
        )

    def test_role_in_post_is_preserved(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertEqual(
            [item["role_in_post"] for item in candidate_input.selected_evidence],
            ["opening support", "practical tension"],
        )

    def test_post_brief_and_angle_decision_are_preserved(self) -> None:
        post_brief = _post_brief()
        angle_decision = _angle_decision()

        candidate_input = build_candidate_writer_input(post_brief, angle_decision)

        self.assertIs(candidate_input.post_brief, post_brief)
        self.assertIs(candidate_input.angle_decision, angle_decision)

    def test_candidate_writer_input_preserves_authorial_voice_directive(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        self.assertEqual(
            candidate_input.to_dict()["angle_decision"]["authorial_voice_directive"][
                "authorial_observation"
            ],
            "What stands out is that remote policies and isolation are separate signals.",
        )
        self.assertEqual(
            candidate_input.to_dict()["angle_decision"]["authorial_voice_directive"][
                "personal_presence_requirement"
            ],
            "explicit_author_owned_statement_required",
        )

    def test_prompt_metadata_is_included_when_provided(self) -> None:
        prompt_metadata = PromptMetadata(
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        )

        candidate_input = build_candidate_writer_input(
            _post_brief(),
            _angle_decision(),
            prompt_metadata=prompt_metadata,
        )

        self.assertEqual(candidate_input.prompt_metadata, prompt_metadata)
        self.assertEqual(
            candidate_input.to_dict()["prompt_metadata"],
            prompt_metadata.to_dict(),
        )

    def test_builder_accepts_dict_style_inputs(self) -> None:
        post_brief = {
            "core_point": "Use selected evidence.",
            "evidence_to_use": [
                {
                    "evidence_id": "a2-summary",
                    "evidence_text": "Hybrid work needs explicit coordination.",
                    "role_in_post": "practical point",
                }
            ],
        }
        angle_decision = {
            "controlling_angle": "Make hybrid work explicit.",
            "supporting_evidence_ids": ["a2-summary"],
            "authorial_voice_directive": {
                "authorial_observation": "What stands out is the coordination gap.",
                "rejected_reading": "Reject treating flexibility as self-managing.",
                "why_distinction_matters": "The distinction matters for hybrid teams.",
                "personal_presence_requirement": "explicit_author_owned_statement_required",
                "first_person_policy": "allowed_not_required",
                "forbidden_author_claims": ["personal experience"],
            },
        }

        candidate_input = build_candidate_writer_input(post_brief, angle_decision)

        self.assertIs(candidate_input.post_brief, post_brief)
        self.assertIs(candidate_input.angle_decision, angle_decision)
        self.assertEqual(
            candidate_input.selected_evidence,
            (
                {
                    "evidence_id": "a2-summary",
                    "evidence_text": "Hybrid work needs explicit coordination.",
                    "role_in_post": "practical point",
                },
            ),
        )

    def test_builder_rejects_missing_authorial_voice_directive(self) -> None:
        with self.assertRaisesRegex(ValueError, "authorial_voice_directive"):
            build_candidate_writer_input(
                _post_brief(),
                {
                    "controlling_angle": "Make remote work explicit.",
                    "supporting_evidence_ids": ["a0-summary", "a1-kp0"],
                },
            )

    def test_builder_rejects_missing_personal_presence_policy(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive.pop("personal_presence_requirement")

        with self.assertRaisesRegex(ValueError, "personal_presence_requirement"):
            build_candidate_writer_input(
                _post_brief(),
                AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
            )

    def test_builder_rejects_unknown_personal_presence_policy(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive["personal_presence_requirement"] = "invent_persona"

        with self.assertRaisesRegex(ValueError, "personal-presence policy"):
            build_candidate_writer_input(
                _post_brief(),
                AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
            )

    def test_builder_rejects_blank_authorial_observation(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive["authorial_observation"] = "   "

        with self.assertRaisesRegex(ValueError, "authorial_observation"):
            build_candidate_writer_input(
                _post_brief(),
                AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
            )

    def test_builder_rejects_invalid_first_person_policy(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive["first_person_policy"] = "required"

        with self.assertRaisesRegex(ValueError, "first_person_policy"):
            build_candidate_writer_input(
                _post_brief(),
                AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
            )

    def test_builder_rejects_empty_forbidden_author_claims(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive["forbidden_author_claims"] = []

        with self.assertRaisesRegex(ValueError, "forbidden_author_claims"):
            build_candidate_writer_input(
                _post_brief(),
                AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
            )

    def test_to_dict_is_json_serializable(self) -> None:
        candidate_input = build_candidate_writer_input(_post_brief(), _angle_decision())

        serialized = json.dumps(candidate_input.to_dict(), sort_keys=True)

        self.assertIn("Remote teams need clear operating agreements.", serialized)
        self.assertIn("Make remote work explicit.", serialized)

    def test_builder_does_not_mutate_selected_evidence_items(self) -> None:
        post_brief = _post_brief()

        candidate_input = build_candidate_writer_input(post_brief, _angle_decision())
        candidate_input.selected_evidence[0]["evidence_text"] = "Changed snapshot text."

        self.assertEqual(
            post_brief.evidence_to_use[0].evidence_text,
            "Remote teams need clear operating agreements.",
        )

    def test_builder_does_not_import_api_prompt_execution_or_runtime_wiring(self) -> None:
        source = inspect.getsource(linkedin_post_flow_input_builders)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)

    def test_builder_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_flow_input_builders).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)

    def test_build_post_editorial_input_returns_contract_for_passing_gate(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(),
            gate_output=_passing_gate_output(),
        )

        self.assertIsInstance(editorial_input, PostEditorialInput)
        self.assertTrue(editorial_input.final_payload_validation_passed)
        self.assertEqual(editorial_input.final_payload_validation_error, "")

    def test_post_editorial_input_rejects_incomplete_authorial_voice_directive(self) -> None:
        directive = dict(_angle_decision().authorial_voice_directive)
        directive["why_distinction_matters"] = ""

        with self.assertRaisesRegex(ValueError, "why_distinction_matters"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=AngleDecisionStub(
                    controlling_angle="Make remote work explicit.",
                    supporting_evidence_ids=["a0-summary", "a1-kp0"],
                    authorial_voice_directive=directive,
                ),
                candidate_output=_candidate_output(),
                gate_output=_passing_gate_output(),
            )

    def test_post_editorial_input_rejects_validation_failure(self) -> None:
        gate_output = _passing_gate_output(validation_passed=False)

        with self.assertRaisesRegex(ValueError, "validation_passed must be True"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(),
                gate_output=gate_output,
            )

    def test_post_editorial_input_rejects_non_ready_diagnostics(self) -> None:
        payload = _valid_payload(post_text="This leaks a0-summary.")
        gate_output = _passing_gate_output(
            payload=payload,
            diagnostics=diagnose_final_post_payload(
                payload,
                selected_evidence_ids=["a0-summary", "a1-kp0"],
                schema_validation_passed=True,
            ),
        )

        with self.assertRaisesRegex(ValueError, "system_linkedin_ready must be True"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(payload=payload),
                gate_output=gate_output,
            )

    def test_post_editorial_input_rejects_schema_validation_failure(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _valid_payload(),
            selected_evidence_ids=["a0-summary", "a1-kp0"],
            schema_validation_passed=False,
            schema_validation_error="schema failed",
        )
        gate_output = _passing_gate_output(diagnostics=diagnostics)

        with self.assertRaisesRegex(ValueError, "schema validation must pass"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(),
                gate_output=gate_output,
            )

    def test_post_editorial_input_rejects_evidence_id_leak(self) -> None:
        payload = _valid_payload(post_text="This leaks a0-summary.")

        with self.assertRaisesRegex(ValueError, "system_linkedin_ready must be True"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(payload=payload),
                gate_output=_passing_gate_output(
                    payload=payload,
                    diagnostics=diagnose_final_post_payload(
                        payload,
                        selected_evidence_ids=["a0-summary", "a1-kp0"],
                        schema_validation_passed=True,
                    ),
                ),
            )

    def test_post_editorial_input_rejects_scaffold_phrase_leak(self) -> None:
        payload = _valid_payload(
            hook_variants=[
                "Separate signals before overclaiming.",
                "Second hook",
                "Third hook",
            ],
        )

        with self.assertRaisesRegex(ValueError, "system_linkedin_ready must be True"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(payload=payload),
                gate_output=_passing_gate_output(
                    payload=payload,
                    diagnostics=diagnose_final_post_payload(
                        payload,
                        selected_evidence_ids=["a0-summary", "a1-kp0"],
                        schema_validation_passed=True,
                    ),
                ),
            )

    def test_post_editorial_input_rejects_source_summary_phrase_leak(self) -> None:
        payload = _valid_payload(
            cta_variants=[
                "According to one source, this matters.",
                "Second CTA",
                "Third CTA",
            ],
        )

        with self.assertRaisesRegex(ValueError, "system_linkedin_ready must be True"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(payload=payload),
                gate_output=_passing_gate_output(
                    payload=payload,
                    diagnostics=diagnose_final_post_payload(
                        payload,
                        selected_evidence_ids=["a0-summary", "a1-kp0"],
                        schema_validation_passed=True,
                    ),
                ),
            )

    def test_post_editorial_candidate_payload_comes_from_gate_output(self) -> None:
        payload = _valid_payload(post_text="Payload from gate.")

        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(payload=payload),
            gate_output=_passing_gate_output(payload=payload),
        )

        self.assertEqual(editorial_input.candidate_payload["post_text"], "Payload from gate.")
        self.assertIsNot(editorial_input.candidate_payload, payload)

    def test_post_editorial_rejects_candidate_and_gate_payload_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload must match"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(payload=_valid_payload(post_text="A")),
                gate_output=_passing_gate_output(payload=_valid_payload(post_text="B")),
            )

    def test_post_editorial_selected_evidence_comes_from_post_brief(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=AngleDecisionStub(
                controlling_angle="Make remote work explicit.",
                supporting_evidence_ids=["a-extra", "a1-kp0"],
                authorial_voice_directive=_angle_decision().authorial_voice_directive,
            ),
            candidate_output=_candidate_output(),
            gate_output=_passing_gate_output(),
        )

        self.assertEqual(
            [item["evidence_id"] for item in editorial_input.selected_evidence],
            ["a0-summary", "a1-kp0"],
        )

    def test_post_editorial_evidence_order_values_text_and_role_are_preserved(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(
                evidence_to_use=[
                    EvidenceUseStub("a1-kp0", "Second evidence.", "practical point"),
                    EvidenceUseStub("a0-summary", "First evidence.", "proof"),
                ],
            ),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(),
            gate_output=_passing_gate_output(
                selected_evidence_ids=("a1-kp0", "a0-summary"),
            ),
        )

        self.assertEqual(
            editorial_input.selected_evidence,
            [
                {
                    "evidence_id": "a1-kp0",
                    "evidence_text": "Second evidence.",
                    "role_in_post": "practical point",
                },
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "First evidence.",
                    "role_in_post": "proof",
                },
            ],
        )

    def test_post_editorial_rejects_evidence_id_mismatch_with_gate_output(self) -> None:
        with self.assertRaisesRegex(ValueError, "evidence IDs must match"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(),
                gate_output=_passing_gate_output(selected_evidence_ids=("a0-summary",)),
            )

    def test_post_editorial_rejects_missing_empty_malformed_or_duplicate_evidence(self) -> None:
        malformed_post_briefs = [
            {"core_point": "No evidence."},
            {"evidence_to_use": []},
            {"evidence_to_use": [{"evidence_id": "", "evidence_text": "Text", "role_in_post": "proof"}]},
            {"evidence_to_use": [{"evidence_id": "a0-summary", "role_in_post": "proof"}]},
            {
                "evidence_to_use": [
                    {"evidence_id": "a0-summary", "evidence_text": "One", "role_in_post": "proof"},
                    {"evidence_id": "a0-summary", "evidence_text": "Two", "role_in_post": "proof"},
                ]
            },
        ]

        for post_brief in malformed_post_briefs:
            with self.subTest(post_brief=post_brief):
                with self.assertRaises(ValueError):
                    build_post_editorial_input(
                        post_brief=post_brief,
                        angle_decision=_angle_decision(),
                        candidate_output=_candidate_output(),
                        gate_output=_passing_gate_output(),
                    )

    def test_post_editorial_preserves_angle_decision_and_post_brief_snapshots(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(),
            gate_output=_passing_gate_output(),
        )

        self.assertEqual(editorial_input.post_brief["core_point"], "Use selected evidence.")
        self.assertEqual(
            editorial_input.angle_decision["controlling_angle"],
            "Make remote work explicit.",
        )

    def test_post_editorial_generation_and_prompt_metadata_are_copied(self) -> None:
        metadata = PostGenerationMetadata(
            provider="openai",
            model="gpt-4.1-2025-04-14",
            run_id="run-1",
            created_at="2026-07-05T10:00:00Z",
            token_usage={"input_tokens": 101},
            cost_metadata={"estimated_usd": "0.02"},
        )
        prompt_metadata = PromptMetadata(
            prompt_name="final_post_candidate_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        )

        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(
                token_usage={"input_tokens": 101},
                cost_metadata={"estimated_usd": "0.02"},
            ),
            gate_output=_passing_gate_output(),
            generation_metadata=metadata,
            prompt_metadata=prompt_metadata,
        )

        self.assertEqual(editorial_input.generation_metadata.run_id, "run-1")
        self.assertEqual(editorial_input.generation_metadata.created_at, "2026-07-05T10:00:00Z")
        self.assertEqual(editorial_input.generation_metadata.token_usage, {"input_tokens": 101})
        self.assertEqual(editorial_input.generation_metadata.cost_metadata, {"estimated_usd": "0.02"})
        self.assertEqual(editorial_input.prompt_metadata, prompt_metadata)

    def test_post_editorial_rejects_conflicting_token_usage(self) -> None:
        candidate_output = _candidate_output(token_usage={"input_tokens": 101})
        metadata = _generation_metadata(token_usage={"input_tokens": 100})
        before = copy.deepcopy((candidate_output, metadata))

        with patch(
            "services.packaging.linkedin_post_flow_input_builders.PostEditorialInput"
        ) as editorial_input_class:
            with self.assertRaisesRegex(
                ValueError,
                "CandidateWriterOutput token_usage conflicts "
                "with PostGenerationMetadata.token_usage.",
            ):
                build_post_editorial_input(
                    post_brief=_post_brief(),
                    angle_decision=_angle_decision(),
                    candidate_output=candidate_output,
                    gate_output=_passing_gate_output(),
                    generation_metadata=metadata,
                )

        editorial_input_class.assert_not_called()
        self.assertEqual((candidate_output, metadata), before)

    def test_post_editorial_accepts_equal_token_usage(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(
                token_usage={"input_tokens": 101, "nested": {"cached": 10}},
            ),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(
                token_usage={"input_tokens": 101, "nested": {"cached": 10}},
            ),
        )

        self.assertEqual(
            editorial_input.generation_metadata.token_usage,
            {"input_tokens": 101, "nested": {"cached": 10}},
        )

    def test_post_editorial_preserves_candidate_only_token_usage(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(token_usage={"input_tokens": 101}),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(token_usage=None),
        )

        self.assertEqual(
            editorial_input.generation_metadata.token_usage,
            {"input_tokens": 101},
        )

    def test_post_editorial_preserves_generation_metadata_only_token_usage(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(token_usage=None),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(token_usage={"input_tokens": 100}),
        )

        self.assertEqual(
            editorial_input.generation_metadata.token_usage,
            {"input_tokens": 100},
        )

    def test_post_editorial_rejects_conflicting_cost_metadata(self) -> None:
        candidate_output = _candidate_output(cost_metadata={"estimated_usd": "0.02"})
        metadata = _generation_metadata(cost_metadata={"estimated_usd": "0.01"})
        before = copy.deepcopy((candidate_output, metadata))

        with patch(
            "services.packaging.linkedin_post_flow_input_builders.PostEditorialInput"
        ) as editorial_input_class:
            with self.assertRaisesRegex(
                ValueError,
                "CandidateWriterOutput cost_metadata conflicts "
                "with PostGenerationMetadata.cost_metadata.",
            ):
                build_post_editorial_input(
                    post_brief=_post_brief(),
                    angle_decision=_angle_decision(),
                    candidate_output=candidate_output,
                    gate_output=_passing_gate_output(),
                    generation_metadata=metadata,
                )

        editorial_input_class.assert_not_called()
        self.assertEqual((candidate_output, metadata), before)

    def test_post_editorial_accepts_equal_cost_metadata(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(
                cost_metadata={"estimated_usd": "0.02", "nested": {"currency": "USD"}},
            ),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(
                cost_metadata={"estimated_usd": "0.02", "nested": {"currency": "USD"}},
            ),
        )

        self.assertEqual(
            editorial_input.generation_metadata.cost_metadata,
            {"estimated_usd": "0.02", "nested": {"currency": "USD"}},
        )

    def test_post_editorial_preserves_candidate_only_cost_metadata(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(cost_metadata={"estimated_usd": "0.02"}),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(cost_metadata=None),
        )

        self.assertEqual(
            editorial_input.generation_metadata.cost_metadata,
            {"estimated_usd": "0.02"},
        )

    def test_post_editorial_preserves_generation_metadata_only_cost_metadata(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(cost_metadata=None),
            gate_output=_passing_gate_output(),
            generation_metadata=_generation_metadata(cost_metadata={"estimated_usd": "0.01"}),
        )

        self.assertEqual(
            editorial_input.generation_metadata.cost_metadata,
            {"estimated_usd": "0.01"},
        )

    def test_post_editorial_metadata_snapshots_are_defensively_copied(self) -> None:
        token_usage = {"input_tokens": 101, "nested": {"cached": 10}}
        cost_metadata = {"estimated_usd": "0.02", "nested": {"currency": "USD"}}
        candidate_output = _candidate_output(
            token_usage=token_usage,
            cost_metadata=cost_metadata,
        )

        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=candidate_output,
            gate_output=_passing_gate_output(),
        )
        token_usage["nested"]["cached"] = 99
        cost_metadata["nested"]["currency"] = "EUR"

        self.assertEqual(
            editorial_input.generation_metadata.token_usage,
            {"input_tokens": 101, "nested": {"cached": 10}},
        )
        self.assertEqual(
            editorial_input.generation_metadata.cost_metadata,
            {"estimated_usd": "0.02", "nested": {"currency": "USD"}},
        )

    def test_post_editorial_rejects_missing_required_metadata(self) -> None:
        with self.assertRaisesRegex(ValueError, "CandidateWriterOutput.provider"):
            build_post_editorial_input(
                post_brief=_post_brief(),
                angle_decision=_angle_decision(),
                candidate_output=_candidate_output(provider=None),
                gate_output=_passing_gate_output(),
            )

    def test_post_editorial_optional_metadata_remains_none(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(token_usage=None, cost_metadata=None),
            gate_output=_passing_gate_output(),
        )

        self.assertIsNone(editorial_input.generation_metadata.run_id)
        self.assertIsNone(editorial_input.generation_metadata.created_at)
        self.assertIsNone(editorial_input.generation_metadata.token_usage)
        self.assertIsNone(editorial_input.generation_metadata.cost_metadata)
        self.assertIsNone(editorial_input.prompt_metadata.prompt_path)

    def test_post_editorial_input_contracts_are_not_mutated(self) -> None:
        post_brief = _post_brief()
        angle_decision = _angle_decision()
        candidate_output = _candidate_output()
        gate_output = _passing_gate_output()
        before = copy.deepcopy((post_brief, angle_decision, candidate_output, gate_output))

        build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=angle_decision,
            candidate_output=candidate_output,
            gate_output=gate_output,
        )

        self.assertEqual((post_brief, angle_decision, candidate_output, gate_output), before)

    def test_post_editorial_nested_snapshots_are_independent_in_both_directions(self) -> None:
        post_brief = {
            "core_point": "Use selected evidence.",
            "evidence_to_use": [
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Original evidence.",
                    "role_in_post": "proof",
                }
            ],
        }
        payload = _valid_payload()
        candidate_output = _candidate_output(payload=payload)
        gate_output = _passing_gate_output(
            payload=payload,
            selected_evidence_ids=("a0-summary",),
        )

        editorial_input = build_post_editorial_input(
            post_brief=post_brief,
            angle_decision=_angle_decision(),
            candidate_output=candidate_output,
            gate_output=gate_output,
        )
        post_brief["evidence_to_use"][0]["evidence_text"] = "Changed source."
        payload["post_text"] = "Changed source payload."

        self.assertEqual(
            editorial_input.candidate_payload["post_text"],
            "Clear final post text for LinkedIn.",
        )
        self.assertEqual(
            editorial_input.selected_evidence[0]["evidence_text"],
            "Original evidence.",
        )

        editorial_input.candidate_payload["post_text"] = "Changed output."
        editorial_input.selected_evidence[0]["evidence_text"] = "Changed output evidence."

        self.assertEqual(gate_output.payload["post_text"], "Changed source payload.")
        self.assertEqual(
            post_brief["evidence_to_use"][0]["evidence_text"],
            "Changed source.",
        )

    def test_post_editorial_to_dict_is_json_serializable(self) -> None:
        editorial_input = build_post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
            candidate_output=_candidate_output(),
            gate_output=_passing_gate_output(),
        )

        serialized = json.dumps(editorial_input.to_dict(), sort_keys=True)

        self.assertIn("Clear final post text for LinkedIn.", serialized)
        self.assertIn("system_linkedin_ready", serialized)

    def test_post_editorial_builder_does_not_execute_gate_decision_attempt_or_repair(self) -> None:
        source = inspect.getsource(linkedin_post_flow_input_builders)

        self.assertNotIn("run_final_post_deterministic_gate", source)
        self.assertNotIn("FinalPostDecisionController", source)
        self.assertNotIn("FinalPostAttempt", source)
        self.assertNotIn("TargetedRepairPlan", source)
        self.assertNotIn("RepairAgent", source)


def _post_brief(
    *,
    evidence_to_use: list[EvidenceUseStub] | None = None,
) -> PostBriefStub:
    return PostBriefStub(
        core_point="Use selected evidence.",
        evidence_to_use=evidence_to_use
        or [
            EvidenceUseStub(
                evidence_id="a0-summary",
                evidence_text="Remote teams need clear operating agreements.",
                role_in_post="opening support",
            ),
            EvidenceUseStub(
                evidence_id="a1-kp0",
                evidence_text="Isolation can rise when remote work is unmanaged.",
                role_in_post="practical tension",
            ),
        ],
    )


def _angle_decision() -> AngleDecisionStub:
    return AngleDecisionStub(
        controlling_angle="Make remote work explicit.",
        supporting_evidence_ids=["a0-summary", "a1-kp0"],
        authorial_voice_directive={
            "authorial_observation": (
                "What stands out is that remote policies and isolation are separate signals."
            ),
            "rejected_reading": (
                "Do not treat policy documentation as proof that isolation has been solved."
            ),
            "why_distinction_matters": (
                "The distinction matters because remote work needs both operating rules and support."
            ),
            "personal_presence_requirement": "explicit_author_owned_statement_required",
            "first_person_policy": "allowed_not_required",
            "forbidden_author_claims": [
                "personal experience",
                "professional authority",
            ],
        },
    )


def _candidate_output(
    *,
    payload: dict | None = None,
    provider: str | None = "openai",
    model: str | None = "gpt-4.1-2025-04-14",
    prompt_name: str | None = "final_post_candidate_from_brief",
    prompt_version: str | None = "1.0",
    token_usage: dict | None = None,
    cost_metadata: dict | None = None,
) -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=payload or _valid_payload(),
        raw_output=None,
        provider=provider,
        model=model,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        token_usage=token_usage,
        cost_metadata=cost_metadata,
    )


def _generation_metadata(
    *,
    provider: str = "openai",
    model: str = "gpt-4.1-2025-04-14",
    run_id: str | None = None,
    created_at: str | None = None,
    token_usage: dict | None = None,
    cost_metadata: dict | None = None,
) -> PostGenerationMetadata:
    return PostGenerationMetadata(
        provider=provider,
        model=model,
        run_id=run_id,
        created_at=created_at,
        token_usage=token_usage,
        cost_metadata=cost_metadata,
    )


def _passing_gate_output(
    *,
    payload: dict | None = None,
    validation_passed: bool = True,
    validation_error: str = "",
    diagnostics=None,
    selected_evidence_ids: tuple[str, ...] = ("a0-summary", "a1-kp0"),
) -> DeterministicGateOutput:
    resolved_payload = payload or _valid_payload()
    return DeterministicGateOutput(
        payload=resolved_payload,
        validation_passed=validation_passed,
        validation_error=validation_error,
        diagnostics=diagnostics
        or diagnose_final_post_payload(
            resolved_payload,
            selected_evidence_ids=list(selected_evidence_ids),
            schema_validation_passed=validation_passed,
            schema_validation_error=validation_error,
        ),
        selected_evidence_ids=selected_evidence_ids,
    )


def _valid_payload(**overrides) -> dict:
    payload = {
        "post_text": "Clear final post text for LinkedIn.",
        "hook_variants": ["First hook", "Second hook", "Third hook"],
        "cta_variants": ["First CTA", "Second CTA", "Third CTA"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }
    payload.update(overrides)
    return payload
