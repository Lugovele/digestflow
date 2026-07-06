from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_prompt_renderers
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_input_builders import (
    build_candidate_writer_input,
)
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
    render_candidate_writer_prompt_input,
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


class LinkedInPostPromptRenderersTests(SimpleTestCase):
    def test_render_candidate_writer_prompt_input_returns_contract(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertIsInstance(render, CandidateWriterPromptRender)

    def test_variables_include_expected_json_sections(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertEqual(
            set(render.variables),
            {
                "post_brief_json",
                "angle_decision_json",
                "selected_evidence_json",
                "candidate_writer_input_json",
            },
        )

    def test_rendered_selected_evidence_preserves_id_order(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a0-summary", "a1-kp0"],
        )

    def test_rendered_selected_evidence_preserves_evidence_text(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_text"] for item in selected_evidence],
            [
                "Remote teams need clear operating agreements.",
                "Isolation can rise when remote work is unmanaged.",
            ],
        )

    def test_rendered_selected_evidence_preserves_role_in_post(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["role_in_post"] for item in selected_evidence],
            ["opening support", "practical tension"],
        )

    def test_rendered_output_includes_serialized_post_brief(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        post_brief = json.loads(render.variables["post_brief_json"])

        self.assertEqual(post_brief["core_point"], "Use selected evidence.")
        self.assertEqual(post_brief["evidence_to_use"][0]["evidence_id"], "a0-summary")
        self.assertIn("POST_BRIEF_JSON", render.input_text)

    def test_rendered_output_includes_serialized_angle_decision(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        angle_decision = json.loads(render.variables["angle_decision_json"])

        self.assertEqual(
            angle_decision["controlling_angle"],
            "Make remote work explicit.",
        )
        self.assertEqual(
            angle_decision["supporting_evidence_ids"],
            ["a0-summary", "a1-kp0"],
        )
        self.assertIn("ANGLE_DECISION_JSON", render.input_text)

    def test_prompt_metadata_is_copied_when_provided(self) -> None:
        prompt_metadata = _prompt_metadata()
        render = render_candidate_writer_prompt_input(
            _candidate_input(prompt_metadata=prompt_metadata)
        )

        self.assertEqual(render.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(render.prompt_version, "1.0")
        self.assertEqual(render.prompt_path, "prompts/linkedin/final_post_from_brief.txt")

    def test_prompt_metadata_fields_are_none_when_not_provided(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input(prompt_metadata=None))

        self.assertIsNone(render.prompt_name)
        self.assertIsNone(render.prompt_version)
        self.assertIsNone(render.prompt_path)

    def test_input_text_is_deterministic_for_same_input(self) -> None:
        first_render = render_candidate_writer_prompt_input(_candidate_input())
        second_render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertEqual(first_render.variables, second_render.variables)
        self.assertEqual(first_render.input_text, second_render.input_text)

    def test_to_dict_output_is_json_serializable(self) -> None:
        render = render_candidate_writer_prompt_input(
            _candidate_input(prompt_metadata=_prompt_metadata())
        )

        serialized = json.dumps(render.to_dict(), sort_keys=True)

        self.assertIn("candidate_writer_input_json", serialized)
        self.assertIn("final_post_candidate_from_brief", serialized)

    def test_renderer_does_not_mutate_candidate_writer_input(self) -> None:
        candidate_input = _candidate_input()
        before = candidate_input.to_dict()

        render_candidate_writer_prompt_input(candidate_input)

        self.assertEqual(candidate_input.to_dict(), before)

    def test_renderer_does_not_import_api_prompt_execution_registry_or_runtime_wiring(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_renderers)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("linkedin_post_prompt_registry", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)

    def test_renderer_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_renderers).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)


def _candidate_input(prompt_metadata=None):
    return build_candidate_writer_input(
        _post_brief(),
        _angle_decision(),
        prompt_metadata=prompt_metadata,
    )


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


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
    )
