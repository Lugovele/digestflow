from __future__ import annotations

import inspect
from pathlib import Path

from django.test import SimpleTestCase

from services.packaging import linkedin_post_prompt_registry
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_access import ROLE_CANDIDATE_WRITER
from services.packaging.linkedin_post_flow_access import get_access_contract
from services.packaging.linkedin_post_prompt_registry import (
    FINAL_POST_PROMPT_REGISTRY,
    MODEL_ROLE_CANDIDATE_WRITER_PRIMARY,
    PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
    PROMPT_STATUS_BASELINE,
    get_prompt_contract,
    list_prompt_contracts,
    prompt_contract_to_prompt_metadata,
)


class LinkedInPostPromptRegistryTests(SimpleTestCase):
    def test_registry_contains_baseline_candidate_writer_prompt(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(contract.prompt_path, "prompts/linkedin/final_post_from_brief.txt")
        self.assertEqual(contract.agent_role, ROLE_CANDIDATE_WRITER)
        self.assertEqual(contract.status, PROMPT_STATUS_BASELINE)
        self.assertEqual(list_prompt_contracts(), FINAL_POST_PROMPT_REGISTRY)
        self.assertEqual(len(list_prompt_contracts()), 1)

    def test_registered_prompt_path_exists(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertTrue(Path(contract.prompt_path).exists())

    def test_prompt_role_exists_in_access_contract(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        access_contract = get_access_contract(contract.agent_role)

        self.assertEqual(access_contract.agent_role, ROLE_CANDIDATE_WRITER)

    def test_registry_access_mode_matches_candidate_writer_access_contract(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)
        access_contract = get_access_contract(ROLE_CANDIDATE_WRITER)

        self.assertEqual(contract.access_mode, access_contract.access_mode)

    def test_candidate_writer_prompt_contract_matches_access_contract_terms(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)
        access_contract = get_access_contract(ROLE_CANDIDATE_WRITER)

        self.assertEqual(contract.agent_role, ROLE_CANDIDATE_WRITER)
        self.assertEqual(contract.access_mode, access_contract.access_mode)
        self.assertIn("PostBrief", access_contract.allowed_inputs)
        self.assertIn("AngleDecision", access_contract.allowed_inputs)
        self.assertIn("selected_evidence", access_contract.allowed_inputs)
        self.assertIn("PostBrief", contract.input_contract)
        self.assertIn("AngleDecision", contract.input_contract)
        self.assertIn("selected evidence", contract.input_contract)
        self.assertIn("FinalPostPayload", access_contract.allowed_outputs)
        self.assertEqual(contract.output_contract, "FinalPostPayload")

    def test_prompt_input_contract_is_post_brief_angle_decision_and_selected_evidence(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(
            contract.input_contract,
            "PostBrief + AngleDecision + selected evidence",
        )

    def test_prompt_output_contract_is_final_post_payload(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.output_contract, "FinalPostPayload")

    def test_prompt_status_is_baseline(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.status, "baseline")

    def test_prompt_metadata_conversion_returns_prompt_metadata(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        metadata = prompt_contract_to_prompt_metadata(contract)

        self.assertIsInstance(metadata, PromptMetadata)
        self.assertEqual(metadata.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(metadata.prompt_version, "1.0")
        self.assertEqual(metadata.prompt_path, "prompts/linkedin/final_post_from_brief.txt")

    def test_registry_uses_model_role_name_not_concrete_provider_or_model(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.model_role, MODEL_ROLE_CANDIDATE_WRITER_PRIMARY)
        self.assertNotIn("/", contract.model_role)
        self.assertNotIn("openai", contract.model_role)
        self.assertNotIn("gpt-", contract.model_role)
        self.assertNotIn("gemini", contract.model_role)
        self.assertNotIn("claude", contract.model_role)

    def test_registry_does_not_call_api_execute_prompts_or_touch_runtime_generation(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_registry)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)

    def test_unknown_prompt_name_raises_key_error(self) -> None:
        with self.assertRaises(KeyError):
            get_prompt_contract("missing_prompt")
