from django.test import SimpleTestCase

from services.ai.validators import DigestPayloadValidationError, validate_digest_payload
from services.ai.digest_smoke_test import (
    clean_llm_json,
    ensure_list,
    safe_json_loads,
    validate_article_analysis,
)


class DigestValidatorTests(SimpleTestCase):
    def test_valid_digest_payload_passes_validation(self):
        payload = self._build_payload()

        validate_digest_payload(payload)

    def test_missing_version_is_treated_as_version_one(self):
        payload = self._build_payload()
        payload.pop("version")

        validate_digest_payload(payload)

    def test_empty_title_fails_validation(self):
        payload = self._build_payload(title="   ")

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_empty_articles_fail_validation(self):
        payload = self._build_payload(articles=[])

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_article_with_empty_summary_fails_validation(self):
        payload = self._build_payload(
            articles=[
                {
                    "url": "https://example.com/1",
                    "title": "Example title",
                    "summary": "   ",
                    "key_points": ["Point one"],
                    "content_type": "news",
                    "confidence": 0.8,
                }
            ]
        )

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_article_with_non_list_key_points_fails_validation(self):
        payload = self._build_payload(
            articles=[
                {
                    "url": "https://example.com/1",
                    "title": "Example title",
                    "summary": "Summary",
                    "key_points": "not-a-list",
                    "content_type": "news",
                    "confidence": 0.8,
                }
            ]
        )

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_article_with_empty_url_fails_validation(self):
        payload = self._build_payload(
            articles=[
                {
                    "url": " ",
                    "title": "Example title",
                    "summary": "Summary",
                    "key_points": ["Point one"],
                    "content_type": "news",
                    "confidence": 0.8,
                }
            ]
        )

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_article_with_empty_title_fails_validation(self):
        payload = self._build_payload(
            articles=[
                {
                    "url": "https://example.com/1",
                    "title": " ",
                    "summary": "Summary",
                    "key_points": ["Point one"],
                    "content_type": "news",
                    "confidence": 0.8,
                }
            ]
        )

        with self.assertRaises(DigestPayloadValidationError):
            validate_digest_payload(payload)

    def test_extra_fields_do_not_break_validation(self):
        payload = self._build_payload(extra_field="allowed")

        validate_digest_payload(payload)

    def test_clean_llm_json_normalizes_quotes_and_trailing_commas(self):
        raw_text = 'Intro {“summary”: “Done”, “key_points”: [“One”,], “content_type”: “news”,}'

        cleaned = clean_llm_json(raw_text)

        self.assertIn('"summary": "Done"', cleaned)
        self.assertNotIn("“", cleaned)
        self.assertNotIn(",]", cleaned)
        self.assertNotIn(",}", cleaned)

    def test_safe_json_loads_recovers_malformed_json(self):
        raw_text = '{“summary”: “Done”, “key_points”: “One point”, “confidence”: “0.8”,}'

        payload = safe_json_loads(raw_text)

        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["summary"], "Done")

    def test_ensure_list_wraps_string_and_flattens_nested_values(self):
        self.assertEqual(ensure_list("One point"), ["One point"])
        self.assertEqual(
            ensure_list([{"value": "Nested point"}, "Second point"]),
            ["Nested point", "Second point"],
        )

    def test_validate_article_analysis_normalizes_corrupted_fields(self):
        normalized = validate_article_analysis(
            {
                "summary": "Short article summary",
                "key_points": ["news", {"value": "Concrete point"}, "Second point"],
                "content_type": "",
                "confidence": "0.7",
            }
        )

        self.assertEqual(normalized["content_type"], "news")
        self.assertEqual(normalized["confidence"], 0.7)
        self.assertEqual(normalized["key_points"], ["Concrete point", "Second point"])

    def test_validate_article_analysis_removes_embedded_json_and_summary_duplicates(self):
        normalized = validate_article_analysis(
            {
                "summary": '{"summary": "Broken", "key_points": ["One"]}',
                "key_points": [
                    "The article argues that workflow redesign matters most.",
                    "Specific supporting point",
                ],
                "content_type": "tutorial",
                "confidence": 0.9,
            }
        )

        self.assertEqual(normalized["summary"], "Failed to extract")
        self.assertEqual(
            normalized["key_points"],
            [
                "The article argues that workflow redesign matters most.",
                "Specific supporting point",
            ],
        )

    def test_validate_article_analysis_deduplicates_key_points_from_summary(self):
        normalized = validate_article_analysis(
            {
                "summary": "The article argues that workflow redesign matters most before adding AI.",
                "key_points": [
                    "The article argues that workflow redesign matters most before adding AI.",
                    "Teams saw faster review cycles after redesign.",
                ],
                "content_type": "opinion",
                "confidence": 0.8,
            }
        )

        self.assertEqual(
            normalized["key_points"],
            ["Teams saw faster review cycles after redesign."],
        )

    def _build_payload(self, **overrides):
        payload = {
            "version": 1,
            "title": "Digest title",
            "articles": [
                {
                    "url": "https://example.com/1",
                    "title": "First article title",
                    "summary": "A concise summary with a conclusion.",
                    "key_points": ["Point one", "Point two"],
                    "content_type": "news",
                    "confidence": 0.8,
                },
                {
                    "url": "https://example.com/2",
                    "title": "Second article title",
                    "summary": "Another concise summary with a takeaway.",
                    "key_points": ["Point three"],
                    "content_type": "opinion",
                    "confidence": 0.7,
                },
            ],
        }
        payload.update(overrides)
        return payload
