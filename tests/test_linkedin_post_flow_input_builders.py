from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_input_builders
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_input_builders import (
    CandidateWriterInput,
    build_candidate_writer_input,
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


def _post_brief() -> PostBriefStub:
    return PostBriefStub(
        core_point="Use selected evidence.",
        evidence_to_use=[
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
    )
