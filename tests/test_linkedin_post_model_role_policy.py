from __future__ import annotations

import json
from types import MappingProxyType

from django.test import SimpleTestCase

from apps.ai.client import THINKING_MODE_DISABLED, THINKING_MODE_PROVIDER_DEFAULT
from services.packaging.linkedin_post_model_role_policy import (
    FINAL_POST_MODEL_ROLES,
    FINAL_POST_ROLE_CANDIDATE_WRITER,
    FINAL_POST_ROLE_PROVIDER_MODEL_POLICY,
    FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_CODE,
    FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX,
    FINAL_POST_ROLE_QUALITY_EVALUATOR,
    FINAL_POST_ROLE_REPAIR_WRITER,
    FINAL_POST_ROLE_SEMANTIC_GROUNDING,
    SAFE_POLICY_VALUE_MAX_LENGTH,
    get_allowed_final_post_provider_models,
    get_final_post_role_provider_model_policy_failure,
    get_final_post_role_thinking_mode,
    normalize_final_post_model_role,
    validate_final_post_role_provider_model,
)


class FinalPostModelRolePolicyTests(SimpleTestCase):
    def test_policy_matrix_contains_expected_roles(self) -> None:
        self.assertEqual(
            FINAL_POST_MODEL_ROLES,
            (
                "candidate_writer",
                "semantic_grounding",
                "quality_evaluator",
                "repair_writer",
            ),
        )

    def test_policy_matrix_is_immutable(self) -> None:
        self.assertIsInstance(FINAL_POST_ROLE_PROVIDER_MODEL_POLICY, MappingProxyType)
        self.assertIsInstance(
            FINAL_POST_ROLE_PROVIDER_MODEL_POLICY[FINAL_POST_ROLE_CANDIDATE_WRITER],
            MappingProxyType,
        )

    def test_candidate_writer_allows_openai_gemini_and_anthropic(self) -> None:
        cases = (
            ("openai", "gpt-4.1-2025-04-14"),
            ("gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5"),
        )

        for provider, model in cases:
            with self.subTest(provider=provider):
                self.assertIsNone(
                    get_final_post_role_provider_model_policy_failure(
                        role=FINAL_POST_ROLE_CANDIDATE_WRITER,
                        provider=provider,
                        model=model,
                    )
                )

    def test_semantic_grounding_allows_openai_gemini_and_anthropic(self) -> None:
        cases = (
            ("openai", "gpt-4.1-2025-04-14"),
            ("gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5"),
        )

        for provider, model in cases:
            with self.subTest(provider=provider):
                self.assertIsNone(
                    get_final_post_role_provider_model_policy_failure(
                        role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
                        provider=provider,
                        model=model,
                    )
                )

    def test_quality_and_repair_roles_are_openai_only(self) -> None:
        rejected_cases = (
            ("gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5"),
        )

        for role in (FINAL_POST_ROLE_QUALITY_EVALUATOR, FINAL_POST_ROLE_REPAIR_WRITER):
            with self.subTest(role=role, provider="openai"):
                self.assertIsNone(
                    get_final_post_role_provider_model_policy_failure(
                        role=role,
                        provider="openai",
                        model="gpt-4.1-2025-04-14",
                    )
                )
            for provider, model in rejected_cases:
                with self.subTest(role=role, provider=provider):
                    failure = get_final_post_role_provider_model_policy_failure(
                        role=role,
                        provider=provider,
                        model=model,
                    )

                    self.assertIsNotNone(failure)
                    self.assertEqual(
                        failure.code,
                        FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_CODE,
                    )

    def test_anthropic_model_is_rejected_for_wrong_providers(self) -> None:
        for provider in ("openai", "gemini"):
            with self.subTest(provider=provider):
                failure = get_final_post_role_provider_model_policy_failure(
                    role=FINAL_POST_ROLE_CANDIDATE_WRITER,
                    provider=provider,
                    model="claude-sonnet-5",
                )

                self.assertIsNotNone(failure)

    def test_non_anthropic_models_are_rejected_for_anthropic_provider(self) -> None:
        for model in ("gpt-4.1-2025-04-14", "gemini-3.6-flash"):
            with self.subTest(model=model):
                failure = get_final_post_role_provider_model_policy_failure(
                    role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
                    provider="anthropic",
                    model=model,
                )

                self.assertIsNotNone(failure)

    def test_unsupported_claude_models_are_rejected_for_postflow_roles(self) -> None:
        failure = get_final_post_role_provider_model_policy_failure(
            role=FINAL_POST_ROLE_CANDIDATE_WRITER,
            provider="anthropic",
            model="claude-3-5-sonnet",
        )

        self.assertIsNotNone(failure)
        self.assertIn(
            FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX,
            failure.message,
        )

    def test_obsolete_gemini_models_are_rejected_for_postflow_roles(self) -> None:
        for obsolete_model in (
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-3.1-pro-preview",
        ):
            with self.subTest(model=obsolete_model):
                failure = get_final_post_role_provider_model_policy_failure(
                    role=FINAL_POST_ROLE_CANDIDATE_WRITER,
                    provider="gemini",
                    model=obsolete_model,
                )

                self.assertIsNotNone(failure)
                self.assertIn(
                    FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX,
                    failure.message,
                )

    def test_failure_contains_only_safe_role_provider_model_fields(self) -> None:
        failure = get_final_post_role_provider_model_policy_failure(
            role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
            provider="gemini",
            model="gemini-3.6-flash",
        )

        self.assertEqual(
            failure.to_dict(),
            {
                "code": FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_CODE,
                "role": "quality_evaluator",
                "provider": "gemini",
                "model": "gemini-3.6-flash",
                "message": str(failure),
            },
        )
        self.assertNotIn("key", json.dumps(failure.to_dict()).lower())
        self.assertNotIn("prompt", json.dumps(failure.to_dict()).lower())

    def test_failure_values_are_single_line_and_length_limited(self) -> None:
        failure = get_final_post_role_provider_model_policy_failure(
            role="QUALITY_EVALUATOR\r\nignored",
            provider="gemini\nignored",
            model="x" * 200,
        )

        self.assertIsNotNone(failure)
        failure_dict = failure.to_dict()
        for field in ("role", "provider", "model", "message"):
            with self.subTest(field=field):
                self.assertNotIn("\r", failure_dict[field])
                self.assertNotIn("\n", failure_dict[field])
        self.assertLessEqual(len(failure_dict["role"]), SAFE_POLICY_VALUE_MAX_LENGTH)
        self.assertLessEqual(
            len(failure_dict["provider"]),
            SAFE_POLICY_VALUE_MAX_LENGTH,
        )
        self.assertLessEqual(len(failure_dict["model"]), SAFE_POLICY_VALUE_MAX_LENGTH)
        self.assertNotIn("x" * 200, failure_dict["message"])

    def test_validate_raises_value_error_for_unsupported_combination(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX,
        ):
            validate_final_post_role_provider_model(
                role=FINAL_POST_ROLE_REPAIR_WRITER,
                provider="gemini",
                model="gemini-3.6-flash",
            )

    def test_normalize_role_strips_and_lowercases(self) -> None:
        self.assertEqual(
            normalize_final_post_model_role(" Quality_Evaluator "),
            "quality_evaluator",
        )

    def test_allowed_provider_models_returns_safe_copy(self) -> None:
        allowed = get_allowed_final_post_provider_models(
            FINAL_POST_ROLE_CANDIDATE_WRITER
        )
        allowed["gemini"] = ("mutated",)

        self.assertEqual(
            get_allowed_final_post_provider_models(FINAL_POST_ROLE_CANDIDATE_WRITER)[
                "gemini"
            ],
            ("gemini-3.6-flash",),
        )

    def test_candidate_writer_disables_anthropic_sonnet_5_thinking(self) -> None:
        self.assertEqual(
            get_final_post_role_thinking_mode(
                role=FINAL_POST_ROLE_CANDIDATE_WRITER,
                provider="anthropic",
                model="claude-sonnet-5",
            ),
            THINKING_MODE_DISABLED,
        )

    def test_semantic_grounding_preserves_anthropic_provider_default_thinking(
        self,
    ) -> None:
        self.assertEqual(
            get_final_post_role_thinking_mode(
                role=FINAL_POST_ROLE_SEMANTIC_GROUNDING,
                provider="anthropic",
                model="claude-sonnet-5",
            ),
            THINKING_MODE_PROVIDER_DEFAULT,
        )

    def test_quality_and_repair_roles_preserve_provider_default_thinking(self) -> None:
        for role in (FINAL_POST_ROLE_QUALITY_EVALUATOR, FINAL_POST_ROLE_REPAIR_WRITER):
            with self.subTest(role=role):
                self.assertEqual(
                    get_final_post_role_thinking_mode(
                        role=role,
                        provider="openai",
                        model="gpt-4.1-2025-04-14",
                    ),
                    THINKING_MODE_PROVIDER_DEFAULT,
                )
