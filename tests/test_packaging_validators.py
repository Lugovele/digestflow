from django.test import SimpleTestCase

from services.packaging.validators import (
    ContentPackageValidationError,
    validate_content_package_payload,
)


class ContentPackageValidatorTests(SimpleTestCase):
    def test_valid_payload_passes_validation(self):
        payload = self._build_payload()

        validate_content_package_payload(payload)

    def test_empty_post_text_fails_validation(self):
        payload = self._build_payload(post_text="   ")

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_too_long_post_text_fails_validation(self):
        payload = self._build_payload(post_text="x" * 1301)

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_hook_variants_with_less_than_three_items_fails_validation(self):
        payload = self._build_payload(hook_variants=["Only one", "Only two"])

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_cta_variants_with_less_than_three_items_fails_validation(self):
        payload = self._build_payload(cta_variants=["Only one", "Only two"])

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_empty_hashtags_list_fails_validation(self):
        payload = self._build_payload(hashtags=[])

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_missing_quality_checks_fails_validation(self):
        payload = self._build_payload()
        payload.pop("quality_checks")

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def test_non_boolean_quality_checks_field_fails_validation(self):
        payload = self._build_payload(
            quality_checks={
                "uses_only_provided_facts": "yes",
                "has_clear_point_of_view": True,
                "linkedin_ready": True,
            }
        )

        with self.assertRaises(ContentPackageValidationError):
            validate_content_package_payload(payload)

    def _build_payload(self, **overrides):
        payload = {
            "post_text": "A practical LinkedIn post grounded in the provided digest.",
            "hook_variants": ["Hook one", "Hook two", "Hook three"],
            "cta_variants": ["CTA one", "CTA two", "CTA three"],
            "hashtags": ["#AI", "#LinkedIn"],
            "carousel_outline": [
                {
                    "slide": 1,
                    "title": "Slide title",
                    "bullets": ["Point one", "Point two"],
                }
            ],
            "quality_checks": {
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
                "linkedin_ready": True,
            },
        }
        payload.update(overrides)
        return payload
