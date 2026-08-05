from __future__ import annotations

import inspect
from pathlib import Path

from django.test import SimpleTestCase

from services.packaging import linkedin_post_prompt_registry
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_flow_access import ROLE_CANDIDATE_WRITER
from services.packaging.linkedin_post_flow_access import ROLE_QUALITY_EVALUATOR
from services.packaging.linkedin_post_flow_access import get_access_contract
from services.packaging.linkedin_post_prompt_registry import (
    FINAL_POST_PROMPT_REGISTRY,
    MODEL_ROLE_CANDIDATE_WRITER_PRIMARY,
    MODEL_ROLE_QUALITY_EVALUATOR_PRIMARY,
    PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF,
    PROMPT_FINAL_POST_QUALITY_EVALUATOR,
    PROMPT_STATUS_BASELINE,
    PROMPT_STATUS_EXPERIMENTAL,
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
        self.assertEqual(len(list_prompt_contracts()), 2)

    def test_registry_contains_experimental_quality_evaluator_prompt(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)

        self.assertEqual(contract.prompt_name, "final_post_quality_evaluator")
        self.assertEqual(
            contract.prompt_path,
            "prompts/linkedin/final_post_quality_evaluator.txt",
        )
        self.assertEqual(contract.prompt_version, "1.0")
        self.assertEqual(contract.agent_role, ROLE_QUALITY_EVALUATOR)
        self.assertEqual(contract.status, PROMPT_STATUS_EXPERIMENTAL)

    def test_registered_prompt_path_exists(self) -> None:
        for contract in list_prompt_contracts():
            with self.subTest(prompt_name=contract.prompt_name):
                self.assertTrue(Path(contract.prompt_path).exists())

    def test_prompt_role_exists_in_access_contract(self) -> None:
        for contract in list_prompt_contracts():
            with self.subTest(prompt_name=contract.prompt_name):
                access_contract = get_access_contract(contract.agent_role)

                self.assertEqual(access_contract.agent_role, contract.agent_role)

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
        self.assertIn("AngleDecision.authorial_voice_directive", contract.input_contract)
        self.assertIn("personal_presence_instruction", contract.input_contract)
        self.assertIn("selected evidence", contract.input_contract)
        self.assertIn("FinalPostPayload", access_contract.allowed_outputs)
        self.assertEqual(contract.output_contract, "FinalPostPayload")

    def test_quality_evaluator_prompt_contract_matches_access_contract_terms(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)
        access_contract = get_access_contract(ROLE_QUALITY_EVALUATOR)

        self.assertEqual(contract.agent_role, ROLE_QUALITY_EVALUATOR)
        self.assertEqual(contract.access_mode, access_contract.access_mode)
        self.assertIn("PostEditorialInput", access_contract.allowed_inputs)
        self.assertIn("PostEditorialInput", contract.input_contract)
        self.assertIn("AngleDecision.authorial_voice_directive", contract.input_contract)
        self.assertIn("QualityReviewResult", access_contract.allowed_outputs)
        self.assertEqual(contract.output_contract, "QualityReviewResult")

    def test_prompt_input_contract_is_post_brief_angle_decision_and_selected_evidence(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(
            contract.input_contract,
            "PostBrief + AngleDecision + AngleDecision.authorial_voice_directive "
            "+ personal_presence_instruction + selected evidence",
        )

    def test_prompt_output_contract_is_final_post_payload(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.output_contract, "FinalPostPayload")

    def test_quality_evaluator_prompt_contract_is_post_editorial_input_to_quality_review(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)

        self.assertEqual(
            contract.input_contract,
            "PostEditorialInput + AngleDecision.authorial_voice_directive",
        )
        self.assertEqual(contract.output_contract, "QualityReviewResult")

    def test_prompt_status_is_baseline(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        self.assertEqual(contract.status, "baseline")

    def test_quality_evaluator_prompt_status_is_experimental(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_QUALITY_EVALUATOR)

        self.assertEqual(contract.status, "experimental")

    def test_prompt_metadata_conversion_returns_prompt_metadata(self) -> None:
        contract = get_prompt_contract(PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF)

        metadata = prompt_contract_to_prompt_metadata(contract)

        self.assertIsInstance(metadata, PromptMetadata)
        self.assertEqual(metadata.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(metadata.prompt_version, "1.0")
        self.assertEqual(metadata.prompt_path, "prompts/linkedin/final_post_from_brief.txt")

    def test_registry_uses_model_role_name_not_concrete_provider_or_model(self) -> None:
        expected_model_roles = {
            PROMPT_FINAL_POST_CANDIDATE_FROM_BRIEF: MODEL_ROLE_CANDIDATE_WRITER_PRIMARY,
            PROMPT_FINAL_POST_QUALITY_EVALUATOR: MODEL_ROLE_QUALITY_EVALUATOR_PRIMARY,
        }

        for contract in list_prompt_contracts():
            with self.subTest(prompt_name=contract.prompt_name):
                self.assertEqual(
                    contract.model_role,
                    expected_model_roles[contract.prompt_name],
                )
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
