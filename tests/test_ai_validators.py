from django.test import SimpleTestCase

from services.ai.validators import DigestPayloadValidationError, validate_digest_payload


class DigestValidatorTests(SimpleTestCase):
    def test_valid_digest_payload_passes_validation(self):
        payload = self._build_payload()

        validate_digest_payload(payload)

    def test_empty_title_fails_validation(self):
        payload = self._build_payload(title="   ")

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_empty_summary_fails_validation(self):
        payload = self._build_payload(summary="   ")

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_key_points_with_less_than_three_items_fails_validation(self):
        payload = self._build_payload(key_points=["Point one", "Point two"])

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_sources_with_less_than_three_items_fails_validation(self):
        payload = self._build_payload(
            sources=["https://example.com/1", "https://example.com/2"]
        )

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_key_points_not_a_list_fails_validation(self):
        payload = self._build_payload(key_points="not-a-list")

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_sources_not_a_list_fails_validation(self):
        payload = self._build_payload(sources="not-a-list")

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_extra_fields_do_not_break_validation(self):
        payload = self._build_payload(extra_field="allowed")

        validate_digest_payload(payload)

    def _build_payload(self, **overrides):
        payload = {
            "title": "Digest title",
            "summary": "A short factual summary based on the provided snippets.",
            "key_points": ["Point one", "Point two", "Point three"],
            "sources": [
                "https://example.com/1",
                "https://example.com/2",
                "https://example.com/3",
            ],
        }
        payload.update(overrides)
        return payload
