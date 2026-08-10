from __future__ import annotations

import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_publication_assembler
from services.packaging.linkedin_post_publication_assembler import (
    AcceptedPost,
    accepted_post_from_payload,
    assemble_linkedin_publication_package,
)


class LinkedInPostPublicationAssemblerTests(SimpleTestCase):
    def test_accepted_post_preserves_core_post_text(self) -> None:
        accepted_post = accepted_post_from_payload({"post_text": "Accepted core post."})

        self.assertEqual(accepted_post, AcceptedPost(post_text="Accepted core post."))
        self.assertEqual(accepted_post.to_dict(), {"post_text": "Accepted core post."})

    def test_accepted_post_rejects_packaging_fields(self) -> None:
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            accepted_post_from_payload(
                {
                    "post_text": "Accepted core post.",
                    "hook_variants": ["Packaging does not belong to core."],
                }
            )

    def test_assembler_copies_post_text_without_fabricating_optional_packaging(self) -> None:
        package = assemble_linkedin_publication_package(
            AcceptedPost(post_text="Accepted text-only post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
            final_accepted=True,
        )

        self.assertEqual(package.post_text, "Accepted text-only post.")
        self.assertEqual(package.hook_variants, ())
        self.assertEqual(package.cta_variants, ())
        self.assertEqual(package.hashtags, ())
        self.assertEqual(package.carousel_outline, ())

    def test_quality_checks_are_derived_from_validation_owners(self) -> None:
        package = assemble_linkedin_publication_package(
            AcceptedPost(post_text="Accepted core post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
            final_accepted=True,
        )

        self.assertEqual(
            package.quality_checks,
            {
                "linkedin_ready": True,
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
            },
        )
        self.assertEqual(
            package.validation_report["derived_from"],
            {
                "deterministic_gate": True,
                "semantic_grounding": True,
                "quality_evaluator": True,
                "adjudication": True,
            },
        )

    def test_grounding_and_quality_failures_are_reflected_in_derived_checks(self) -> None:
        package = assemble_linkedin_publication_package(
            AcceptedPost(post_text="Accepted core post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=False,
            quality_passed=False,
            final_accepted=True,
        )

        self.assertFalse(package.quality_checks["linkedin_ready"])
        self.assertFalse(package.quality_checks["uses_only_provided_facts"])
        self.assertFalse(package.quality_checks["has_clear_point_of_view"])
        self.assertEqual(package.validation_report["status"], "invalid")

    def test_publication_package_can_include_explicit_post_acceptance_metadata(self) -> None:
        package = assemble_linkedin_publication_package(
            AcceptedPost(post_text="Accepted core post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
            final_accepted=True,
            hook_variants=("Optional hook",),
            cta_variants=("Optional CTA",),
            hashtags=("#Optional",),
            carousel_outline=({"title": "Optional derivative"},),
        )

        package_dict = package.to_dict()

        self.assertEqual(package_dict["hook_variants"], ["Optional hook"])
        self.assertEqual(package_dict["cta_variants"], ["Optional CTA"])
        self.assertEqual(package_dict["hashtags"], ["#Optional"])
        self.assertEqual(
            package_dict["carousel_outline"],
            [{"title": "Optional derivative"}],
        )

    def test_assembler_requires_final_acceptance(self) -> None:
        with self.assertRaisesRegex(ValueError, "after acceptance"):
            assemble_linkedin_publication_package(
                AcceptedPost(post_text="Not accepted yet."),
                deterministic_gate_passed=True,
                semantic_grounding_passed=True,
                quality_passed=True,
                final_accepted=False,
            )

    def test_to_dict_is_json_serializable(self) -> None:
        package = assemble_linkedin_publication_package(
            AcceptedPost(post_text="Accepted core post."),
            deterministic_gate_passed=True,
            semantic_grounding_passed=True,
            quality_passed=True,
            final_accepted=True,
        )

        json.dumps(package.to_dict(), sort_keys=True)

    def test_module_does_not_import_runtime_providers_or_generation_layers(self) -> None:
        source = inspect.getsource(linkedin_post_publication_assembler)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("generate_content_package_for_digest", source)
        self.assertNotIn("ContentPackage.objects", source)
