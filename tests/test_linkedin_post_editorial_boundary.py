from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_editorial_boundary
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_editorial_boundary import (
    PostEditorialInput,
    PostGenerationMetadata,
    PromptMetadata,
)


@dataclass(frozen=True)
class BriefStub:
    core_point: str
    evidence_to_use: list[dict]


@dataclass(frozen=True)
class AngleStub:
    controlling_angle: str
    supporting_evidence_ids: list[str]


class LinkedInPostEditorialBoundaryTests(SimpleTestCase):
    def test_post_generation_metadata_to_dict(self) -> None:
        metadata = PostGenerationMetadata(
            provider="openai",
            model="gpt-4.1-2025-04-14",
            run_id="run-1",
            created_at="2026-07-05T10:00:00Z",
            token_usage={"input_tokens": 100, "output_tokens": 50},
            cost_metadata={"estimated_usd": "0.01"},
        )

        self.assertEqual(
            metadata.to_dict(),
            {
                "provider": "openai",
                "model": "gpt-4.1-2025-04-14",
                "run_id": "run-1",
                "created_at": "2026-07-05T10:00:00Z",
                "token_usage": {"input_tokens": 100, "output_tokens": 50},
                "cost_metadata": {"estimated_usd": "0.01"},
            },
        )

    def test_prompt_metadata_to_dict(self) -> None:
        metadata = PromptMetadata(
            prompt_name="final_post_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        )

        self.assertEqual(
            metadata.to_dict(),
            {
                "prompt_name": "final_post_from_brief",
                "prompt_version": "1.0",
                "prompt_path": "prompts/linkedin/final_post_from_brief.txt",
            },
        )

    def test_post_editorial_input_to_dict_includes_boundary_fields(self) -> None:
        boundary = _boundary_input()

        boundary_dict = boundary.to_dict()

        self.assertEqual(boundary_dict["candidate_payload"]["post_text"], "Final post text.")
        self.assertEqual(boundary_dict["post_brief"]["core_point"], "Use selected evidence.")
        self.assertEqual(boundary_dict["angle_decision"]["controlling_angle"], "Clear angle")
        self.assertEqual(boundary_dict["selected_evidence"][0]["evidence_id"], "a0-summary")
        self.assertTrue(boundary_dict["final_payload_validation_passed"])
        self.assertEqual(boundary_dict["final_payload_validation_error"], "")
        self.assertIn("system_linkedin_ready", boundary_dict["diagnostics"])
        self.assertEqual(boundary_dict["repair_reasons"], [])
        self.assertEqual(boundary_dict["generation_metadata"]["provider"], "openai")
        self.assertEqual(boundary_dict["prompt_metadata"]["prompt_name"], "final_post_from_brief")

    def test_post_editorial_input_preserves_diagnostics_repair_reasons(self) -> None:
        diagnostics = diagnose_final_post_payload(
            _candidate_payload(quality_checks={"linkedin_ready": True}),
            selected_evidence_ids=[],
            schema_validation_passed=True,
        )
        boundary = _boundary_input(
            diagnostics=diagnostics,
            repair_reasons=diagnostics.repair_reasons,
        )

        boundary_dict = boundary.to_dict()

        self.assertEqual(boundary_dict["repair_reasons"], ["missing_quality_checks"])
        self.assertEqual(
            boundary_dict["diagnostics"]["repair_reasons"],
            ["missing_quality_checks"],
        )

    def test_post_editorial_input_to_dict_is_json_serializable(self) -> None:
        boundary_dict = _boundary_input().to_dict()

        serialized = json.dumps(boundary_dict, sort_keys=True)

        self.assertIn("Final post text.", serialized)

    def test_boundary_module_does_not_call_api_or_execute_prompts(self) -> None:
        source = inspect.getsource(linkedin_post_editorial_boundary)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)


def _boundary_input(**overrides) -> PostEditorialInput:
    diagnostics = overrides.pop(
        "diagnostics",
        diagnose_final_post_payload(
            _candidate_payload(),
            selected_evidence_ids=["a0-summary"],
            schema_validation_passed=True,
        ),
    )
    values = {
        "candidate_payload": _candidate_payload(),
        "post_brief": BriefStub(
            core_point="Use selected evidence.",
            evidence_to_use=[
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Remote work requires clearer policy.",
                }
            ],
        ),
        "angle_decision": AngleStub(
            controlling_angle="Clear angle",
            supporting_evidence_ids=["a0-summary"],
        ),
        "selected_evidence": [
            {
                "evidence_id": "a0-summary",
                "evidence_text": "Remote work requires clearer policy.",
            }
        ],
        "final_payload_validation_passed": True,
        "final_payload_validation_error": "",
        "diagnostics": diagnostics,
        "repair_reasons": list(diagnostics.repair_reasons),
        "generation_metadata": PostGenerationMetadata(
            provider="openai",
            model="gpt-4.1-2025-04-14",
            run_id="run-1",
            created_at="2026-07-05T10:00:00Z",
            token_usage=None,
            cost_metadata=None,
        ),
        "prompt_metadata": PromptMetadata(
            prompt_name="final_post_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
    }
    values.update(overrides)
    return PostEditorialInput(**values)


def _candidate_payload(**overrides) -> dict:
    payload = {
        "post_text": "Final post text.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
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
