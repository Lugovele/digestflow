from __future__ import annotations

import json
from types import MappingProxyType

from django.test import SimpleTestCase

from apps.ai.client import (
    AI_THINKING_MODE_DISABLED,
    AI_THINKING_MODE_PROVIDER_DEFAULT,
)
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

    def test_quality_evaluator_accepts_only_openai_gpt_4_1(self) -> None:
        self.assertIsNone(
            get_final_post_role_provider_model_policy_failure(
                role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )
        )
        for provider, model in (
            ("gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5"),
        ):
            with self.subTest(provider=provider):
                self.assertIsNotNone(
                    get_final_post_role_provider_model_policy_failure(
                        role=FINAL_POST_ROLE_QUALITY_EVALUATOR,
                        provider=provider,
                        model=model,
                    )
                )

    def test_repair_writer_accepts_only_openai_gpt_4_1(self) -> None:
        self.assertIsNone(
            get_final_post_role_provider_model_policy_failure(
                role=FINAL_POST_ROLE_REPAIR_WRITER,
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )
        )
        for provider, model in (
            ("gemini", "gemini-3.6-flash"),
            ("anthropic", "claude-sonnet-5"),
        ):
            with self.subTest(provider=provider):
                self.assertIsNotNone(
                    get_final_post_role_provider_model_policy_failure(
                        role=FINAL_POST_ROLE_REPAIR_WRITER,
                        provider=provider,
                        model=model,
                    )
                )

    def test_cross_provider_model_mismatch_is_rejected(self) -> None:
        rejected_cases = (
            (FINAL_POST_ROLE_CANDIDATE_WRITER, "openai", "gemini-3.6-flash"),
            (FINAL_POST_ROLE_CANDIDATE_WRITER, "gemini", "gpt-4.1-2025-04-14"),
            (FINAL_POST_ROLE_SEMANTIC_GROUNDING, "anthropic", "unsupported-model"),
        )

        for role, provider, model in rejected_cases:
            with self.subTest(role=role, provider=provider, model=model):
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

    def test_unknown_role_and_provider_are_rejected(self) -> None:
        for role, provider in (("unknown", "openai"), (FINAL_POST_ROLE_CANDIDATE_WRITER, "unknown")):
            with self.subTest(role=role, provider=provider):
                failure = get_final_post_role_provider_model_policy_failure(
                    role=role,
                    provider=provider,
                    model="gpt-4.1-2025-04-14",
                )

                self.assertIsNotNone(failure)
                self.assertIn(FINAL_POST_ROLE_PROVIDER_MODEL_POLICY_FAILURE_PREFIX, failure.message)

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
        serialized = json.dumps(failure.to_dict()).lower()
        self.assertNotIn("key", serialized)
        self.assertNotIn("authorization", serialized)
        self.assertNotIn("raw provider", serialized)

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
        self.assertLessEqual(len(failure_dict["provider"]), SAFE_POLICY_VALUE_MAX_LENGTH)
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
            AI_THINKING_MODE_DISABLED,
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
            AI_THINKING_MODE_PROVIDER_DEFAULT,
        )

    def test_other_valid_role_provider_model_pairs_use_provider_default_thinking(
        self,
    ) -> None:
        cases = (
            (FINAL_POST_ROLE_CANDIDATE_WRITER, "openai", "gpt-4.1-2025-04-14"),
            (FINAL_POST_ROLE_CANDIDATE_WRITER, "gemini", "gemini-3.6-flash"),
            (FINAL_POST_ROLE_SEMANTIC_GROUNDING, "openai", "gpt-4.1-2025-04-14"),
            (FINAL_POST_ROLE_SEMANTIC_GROUNDING, "gemini", "gemini-3.6-flash"),
            (FINAL_POST_ROLE_QUALITY_EVALUATOR, "openai", "gpt-4.1-2025-04-14"),
            (FINAL_POST_ROLE_REPAIR_WRITER, "openai", "gpt-4.1-2025-04-14"),
        )

        for role, provider, model in cases:
            with self.subTest(role=role, provider=provider):
                self.assertEqual(
                    get_final_post_role_thinking_mode(
                        role=role,
                        provider=provider,
                        model=model,
                    ),
                    AI_THINKING_MODE_PROVIDER_DEFAULT,
                )
