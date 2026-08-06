from __future__ import annotations

import json

from django.test import SimpleTestCase

from services.packaging.linkedin_post_candidate_writer_structural_diagnostics import (
    ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
    ADAPTER_ERROR_UNKNOWN,
    FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
    FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
    MAX_DIAGNOSTIC_FIELD_NAME_LENGTH,
    MAX_DIAGNOSTIC_LIST_ITEMS,
    PARSER_ERROR_MALFORMED_JSON,
    CandidateWriterStructuralDiagnostics,
    build_adapter_structural_diagnostics,
    build_parser_structural_diagnostics,
    structural_diagnostics_from_dict,
)


class CandidateWriterStructuralDiagnosticsTests(SimpleTestCase):
    def test_valid_diagnostics_construct_and_serialize(self) -> None:
        diagnostics = CandidateWriterStructuralDiagnostics(
            failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
            parser_error_code=PARSER_ERROR_MALFORMED_JSON,
            adapter_error_code=None,
            top_level_json_type=None,
            received_top_level_keys=(),
            missing_required_fields=(),
            unexpected_fields=(),
            invalid_field_names=(),
            candidate_text_length=12,
            diagnostics_truncated=False,
            redacted_key_count=0,
        )

        serialized = diagnostics.to_dict()

        self.assertEqual(serialized["parser_error_code"], PARSER_ERROR_MALFORMED_JSON)
        json.dumps(serialized, sort_keys=True)

    def test_invalid_failure_stage_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidateWriterStructuralDiagnostics(
                failure_stage="not_a_stage",
                parser_error_code=PARSER_ERROR_MALFORMED_JSON,
                adapter_error_code=None,
                top_level_json_type=None,
                received_top_level_keys=(),
                missing_required_fields=(),
                unexpected_fields=(),
                invalid_field_names=(),
                candidate_text_length=1,
                diagnostics_truncated=False,
                redacted_key_count=0,
            )

    def test_invalid_parser_code_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_parser_structural_diagnostics(
                parser_error_code="new_unapproved_code",
                candidate_text="{}",
            )

    def test_invalid_adapter_code_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidateWriterStructuralDiagnostics(
                failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
                parser_error_code=None,
                adapter_error_code="new_unapproved_code",
                top_level_json_type="dict",
                received_top_level_keys=(),
                missing_required_fields=(),
                unexpected_fields=(),
                invalid_field_names=(),
                candidate_text_length=None,
                diagnostics_truncated=False,
                redacted_key_count=0,
            )

    def test_negative_text_length_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CandidateWriterStructuralDiagnostics(
                failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
                parser_error_code=PARSER_ERROR_MALFORMED_JSON,
                adapter_error_code=None,
                top_level_json_type=None,
                received_top_level_keys=(),
                missing_required_fields=(),
                unexpected_fields=(),
                invalid_field_names=(),
                candidate_text_length=-1,
                diagnostics_truncated=False,
                redacted_key_count=0,
            )

    def test_keys_are_sorted_deduplicated_and_bounded(self) -> None:
        parsed = {f"field_{index:02d}": index for index in range(25)}
        parsed["field_03"] = "duplicate"

        diagnostics = build_adapter_structural_diagnostics(
            adapter_error_code=ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
            parsed_candidate=parsed,
            required_fields=("post_text",),
            allowed_fields=("post_text",),
        )

        self.assertEqual(
            diagnostics.received_top_level_keys,
            tuple(sorted(diagnostics.received_top_level_keys)),
        )
        self.assertEqual(
            len(diagnostics.received_top_level_keys),
            MAX_DIAGNOSTIC_LIST_ITEMS,
        )
        self.assertTrue(diagnostics.diagnostics_truncated)
        self.assertGreaterEqual(diagnostics.redacted_key_count, 5)

    def test_overlong_names_are_omitted_and_counted(self) -> None:
        overlong = "x" * (MAX_DIAGNOSTIC_FIELD_NAME_LENGTH + 1)

        diagnostics = build_adapter_structural_diagnostics(
            adapter_error_code=ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
            parsed_candidate={"post_text": "ok", overlong: "secret"},
            required_fields=("hook_variants",),
            allowed_fields=("post_text", "hook_variants"),
        )

        self.assertNotIn(overlong, diagnostics.received_top_level_keys)
        self.assertTrue(diagnostics.diagnostics_truncated)
        self.assertGreaterEqual(diagnostics.redacted_key_count, 1)

    def test_serialization_contains_only_approved_fields(self) -> None:
        diagnostics = build_parser_structural_diagnostics(
            parser_error_code=PARSER_ERROR_MALFORMED_JSON,
            candidate_text='{"post_text":"secret post text","api_key":"secret"}',
        )

        serialized = diagnostics.to_dict()
        text = json.dumps(serialized, sort_keys=True)

        self.assertEqual(
            set(serialized),
            {
                "schema_version",
                "failure_stage",
                "parser_error_code",
                "adapter_error_code",
                "top_level_json_type",
                "received_top_level_keys",
                "missing_required_fields",
                "unexpected_fields",
                "invalid_field_names",
                "candidate_text_length",
                "diagnostics_truncated",
                "redacted_key_count",
            },
        )
        self.assertNotIn("secret post text", text)
        self.assertNotIn("api_key", text)

    def test_reconstruction_resanitizes_untrusted_field_names(self) -> None:
        diagnostics = structural_diagnostics_from_dict(
            {
                "failure_stage": FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
                "parser_error_code": None,
                "adapter_error_code": ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
                "top_level_json_type": "dict",
                "received_top_level_keys": [
                    "post_text",
                    "api_key",
                    "x" * (MAX_DIAGNOSTIC_FIELD_NAME_LENGTH + 1),
                    123,
                ],
                "missing_required_fields": ["hook_variants"],
                "unexpected_fields": ["provider_payload", "debug_extra"],
                "invalid_field_names": ["post_text"],
                "candidate_text_length": None,
                "diagnostics_truncated": False,
                "redacted_key_count": "not an int",
            }
        )

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertEqual(diagnostics.received_top_level_keys, ("post_text",))
        self.assertEqual(diagnostics.unexpected_fields, ("debug_extra",))
        self.assertTrue(diagnostics.diagnostics_truncated)
        self.assertGreaterEqual(diagnostics.redacted_key_count, 3)
        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)
        self.assertNotIn("api_key", serialized)
        self.assertNotIn("provider_payload", serialized)

    def test_reconstruction_drops_unsafe_top_level_type_name(self) -> None:
        diagnostics = structural_diagnostics_from_dict(
            {
                "failure_stage": FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
                "parser_error_code": None,
                "adapter_error_code": ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
                "top_level_json_type": "prompt: secret raw response text",
                "received_top_level_keys": ["post_text"],
                "missing_required_fields": ["hook_variants"],
                "unexpected_fields": [],
                "invalid_field_names": [],
                "candidate_text_length": None,
                "diagnostics_truncated": False,
                "redacted_key_count": 0,
            }
        )

        self.assertIsNotNone(diagnostics)
        assert diagnostics is not None
        self.assertIsNone(diagnostics.top_level_json_type)
        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)
        self.assertNotIn("secret raw response text", serialized)

    def test_direct_construction_drops_unsafe_top_level_type_name(self) -> None:
        diagnostics = CandidateWriterStructuralDiagnostics(
            failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
            parser_error_code=None,
            adapter_error_code=ADAPTER_ERROR_MISSING_REQUIRED_FIELDS,
            top_level_json_type="prompt: secret raw response text",
            received_top_level_keys=(),
            missing_required_fields=("post_text",),
            unexpected_fields=(),
            invalid_field_names=(),
            candidate_text_length=None,
            diagnostics_truncated=False,
            redacted_key_count=0,
        )

        self.assertIsNone(diagnostics.top_level_json_type)
        serialized = json.dumps(diagnostics.to_dict(), sort_keys=True)
        self.assertNotIn("secret raw response text", serialized)

    def test_stage_code_ownership_is_enforced(self) -> None:
        with self.assertRaises(ValueError):
            CandidateWriterStructuralDiagnostics(
                failure_stage=FAILURE_STAGE_CANDIDATE_WRITER_PARSE,
                parser_error_code=PARSER_ERROR_MALFORMED_JSON,
                adapter_error_code=ADAPTER_ERROR_UNKNOWN,
                top_level_json_type=None,
                received_top_level_keys=(),
                missing_required_fields=(),
                unexpected_fields=(),
                invalid_field_names=(),
                candidate_text_length=1,
                diagnostics_truncated=False,
                redacted_key_count=0,
            )

        self.assertIsNone(
            structural_diagnostics_from_dict(
                {
                    "failure_stage": FAILURE_STAGE_CANDIDATE_WRITER_ADAPTATION,
                    "parser_error_code": PARSER_ERROR_MALFORMED_JSON,
                    "adapter_error_code": ADAPTER_ERROR_UNKNOWN,
                    "top_level_json_type": "dict",
                    "received_top_level_keys": [],
                    "missing_required_fields": [],
                    "unexpected_fields": [],
                    "invalid_field_names": [],
                    "candidate_text_length": None,
                    "diagnostics_truncated": False,
                    "redacted_key_count": 0,
                }
            )
        )
