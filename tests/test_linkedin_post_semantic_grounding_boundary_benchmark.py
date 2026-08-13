from __future__ import annotations

import ast
from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile

from django.test import SimpleTestCase

from services.packaging import linkedin_post_semantic_grounding_boundary_benchmark as boundary
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)


class LinkedInPostSemanticGroundingBoundaryBenchmarkTests(SimpleTestCase):
    def test_corpus_inventory_is_balanced_and_paired(self) -> None:
        cases = boundary.load_semantic_grounding_boundary_cases()

        self.assertEqual(len(cases), 16)
        self.assertEqual(
            sum(1 for case in cases if case.expected_label == boundary.EXPECTED_VALID),
            8,
        )
        self.assertEqual(
            sum(1 for case in cases if case.expected_label == boundary.EXPECTED_INVALID),
            8,
        )
        self.assertEqual(len({case.category for case in cases}), 8)
        self.assertEqual(len({case.pair_id for case in cases}), 8)

    def test_fixture_metadata_is_contractual(self) -> None:
        fixture = Path(
            "tests/fixtures/linkedin_post_semantic_grounding_boundary/"
            "boundary_cases_v1.json"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8-sig"))

        self.assertEqual(
            payload["schema_version"],
            boundary.BOUNDARY_BENCHMARK_SCHEMA_VERSION,
        )
        self.assertEqual(
            payload["corpus_id"],
            boundary.BOUNDARY_BENCHMARK_CORPUS_ID,
        )
        self.assertEqual(
            payload["provenance"],
            boundary.BOUNDARY_BENCHMARK_PROVENANCE,
        )

    def test_pair_invariants_allow_only_candidate_boundary_delta(self) -> None:
        cases = boundary.load_semantic_grounding_boundary_cases()

        for pair_id in sorted({case.pair_id for case in cases}):
            pair = [case for case in cases if case.pair_id == pair_id]
            self.assertEqual(len(pair), 2)
            first, second = pair
            self.assertEqual(first.evidence, second.evidence)
            self.assertEqual(first.post_brief, second.post_brief)
            self.assertEqual(first.angle_decision, second.angle_decision)
            self.assertEqual(first.semantic_topic, second.semantic_topic)
            self.assertEqual(first.rhetorical_intent, second.rhetorical_intent)
            self.assertNotEqual(first.candidate_post, second.candidate_post)

    def test_must_inspect_fragments_are_present_in_candidate_post(self) -> None:
        for case in boundary.load_semantic_grounding_boundary_cases():
            normalized_post = _normalize(case.candidate_post)
            for fragment in case.must_inspect_fragments:
                self.assertIn(_normalize(fragment), normalized_post)

    def test_default_plans_have_approved_provider_specific_budgets(self) -> None:
        plans = boundary.default_semantic_grounding_boundary_plans()

        self.assertEqual([plan.plan_id for plan in plans], ["gpt_grounding", "gemini_grounding"])
        self.assertEqual(plans[0].provider, "openai")
        self.assertEqual(plans[0].model, "gpt-4.1-2025-04-14")
        self.assertEqual(plans[0].max_output_tokens, 2400)
        self.assertEqual(plans[1].provider, "gemini")
        self.assertEqual(plans[1].model, "gemini-3.6-flash")
        self.assertEqual(plans[1].max_output_tokens, 4800)

    def test_prompt_render_uses_candidate_post_as_post_text_only(self) -> None:
        case = boundary.load_semantic_grounding_boundary_cases()[0]

        render = boundary.build_boundary_semantic_grounding_prompt_render(case)

        candidate_payload = json.loads(render.variables["candidate_payload_json"])
        self.assertEqual(candidate_payload, {"post_text": case.candidate_post})
        selected_evidence = json.loads(render.variables["selected_evidence_json"])
        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            list(case.selected_evidence_ids),
        )

    def test_dry_run_writes_artifacts_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = boundary.run_semantic_grounding_boundary_benchmark(
                boundary.SemanticGroundingBoundaryRequest(
                    cases=boundary.load_semantic_grounding_boundary_cases(),
                    plans=boundary.default_semantic_grounding_boundary_plans(),
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
            )

            self.assertEqual(result.status, boundary.STATUS_DRY_RUN)
            self.assertEqual(result.provider_call_count, 0)
            self.assertEqual(result.planned_provider_call_count, 32)
            artifacts = result.artifacts.to_dict()
            for key in (
                "manifest_json",
                "runs_jsonl",
                "summary_csv",
                "boundary_metrics_json",
                "boundary_comparison_md",
                "report_md",
            ):
                self.assertTrue(Path(artifacts[key]).exists(), key)

            forbidden_fragments = (
                "prompt_text",
                "provider_payload",
                "provider_reply",
                "raw_response",
                "api_key",
                "secret",
            )
            for artifact_path in (
                artifacts["manifest_json"],
                artifacts["runs_jsonl"],
                artifacts["boundary_metrics_json"],
            ):
                artifact_text = Path(artifact_path).read_text(encoding="utf-8")
                for fragment in forbidden_fragments:
                    self.assertNotIn(fragment, artifact_text)

    def test_valid_strong_voice_non_pass_counts_as_false_positive(self) -> None:
        record = _record(
            expected_label=boundary.EXPECTED_VALID,
            grounding_pass=False,
            blocking_claim_ids=[],
            human_review_required=True,
        )

        self.assertEqual(
            boundary.classify_boundary_run(record),
            boundary.OUTCOME_FALSE_POSITIVE,
        )

    def test_invalid_overreach_pass_counts_as_false_negative(self) -> None:
        record = _record(
            expected_label=boundary.EXPECTED_INVALID,
            grounding_pass=True,
            blocking_claim_ids=[],
        )

        self.assertEqual(
            boundary.classify_boundary_run(record),
            boundary.OUTCOME_FALSE_NEGATIVE,
        )

    def test_infrastructure_failures_are_counted_outside_false_positive_negative(self) -> None:
        metrics = boundary.compute_boundary_metrics(
            (
                _record(
                    expected_label=boundary.EXPECTED_VALID,
                    execution_success=False,
                    parse_success=False,
                    normalization_success=False,
                    failure_stage="execution",
                ),
            )
        )["plans"]["gpt_grounding"]

        self.assertEqual(metrics["infrastructure_failure_count"], 1)
        self.assertEqual(metrics["provider_failure_count"], 1)
        self.assertEqual(metrics["evaluated_cases_total"], 0)
        self.assertEqual(metrics["false_positive_count"], 0)
        self.assertEqual(metrics["false_negative_count"], 0)
        self.assertEqual(metrics["decision_band"], boundary.DECISION_INCOMPLETE)

    def test_voice_sensitive_metric_tracks_valid_false_positives(self) -> None:
        metrics = boundary.compute_boundary_metrics(
            (
                _record(
                    expected_label=boundary.EXPECTED_VALID,
                    voice_sensitive=True,
                    grounding_pass=True,
                    blocking_claim_ids=[],
                ),
                _record(
                    expected_label=boundary.EXPECTED_VALID,
                    voice_sensitive=True,
                    grounding_pass=False,
                    blocking_claim_ids=["c1"],
                ),
            )
        )["plans"]["gpt_grounding"]

        self.assertEqual(metrics["voice_sensitive_valid_total"], 2)
        self.assertEqual(metrics["voice_sensitive_preserved"], 1)
        self.assertEqual(metrics["voice_sensitive_false_positive_rate"], 0.5)

    def test_fragment_coverage_metric_uses_normalized_claim_text(self) -> None:
        metrics = boundary.compute_boundary_metrics(
            (
                _record(
                    expected_label=boundary.EXPECTED_VALID,
                    must_inspect_fragment_coverage=[
                        {"fragment": "Growth signals don't erase", "covered": True},
                        {"fragment": "risk phase is over", "covered": False},
                    ],
                ),
            )
        )["plans"]["gpt_grounding"]

        self.assertEqual(metrics["required_fragments_total"], 2)
        self.assertEqual(metrics["required_fragments_covered"], 1)
        self.assertEqual(metrics["required_fragment_coverage_rate"], 0.5)

    def test_weighted_score_math_uses_approved_weights(self) -> None:
        metrics = boundary.compute_boundary_metrics(
            (
                _record(
                    expected_label=boundary.EXPECTED_VALID,
                    grounding_pass=False,
                    blocking_claim_ids=["c1"],
                ),
                _record(
                    expected_label=boundary.EXPECTED_INVALID,
                    grounding_pass=True,
                    blocking_claim_ids=[],
                ),
            ),
            false_positive_weight=1.0,
            false_negative_weight=1.25,
        )["plans"]["gpt_grounding"]

        self.assertEqual(metrics["false_positive_count"], 1)
        self.assertEqual(metrics["false_negative_count"], 1)
        self.assertEqual(metrics["weighted_error_score"], 1.125)

    def test_live_path_with_fake_executor_records_detection_and_preservation(self) -> None:
        cases = boundary.load_semantic_grounding_boundary_cases()
        expected_by_case_id = {
            case.case_id: case.expected_label
            for case in cases
        }

        def fake_executor(request):
            if (
                expected_by_case_id[request.execution_metadata["case_id"]]
                == boundary.EXPECTED_INVALID
            ):
                return _raw(_review_payload(passed=False, blocking=True))
            return _raw(_review_payload())

        with tempfile.TemporaryDirectory() as tempdir:
            result = boundary.run_semantic_grounding_boundary_benchmark(
                boundary.SemanticGroundingBoundaryRequest(
                    cases=cases,
                    plans=(boundary.default_semantic_grounding_boundary_plans()[0],),
                    allow_api=True,
                    output_root=Path(tempdir),
                ),
                now_factory=_fixed_now,
                semantic_grounding_executor=fake_executor,
            )

        self.assertEqual(result.status, boundary.STATUS_COMPLETED)
        self.assertEqual(result.provider_call_count, 16)
        outcomes_by_case_id = {
            record["case_id"]: record["boundary_outcome"]
            for record in result.run_records
        }
        self.assertEqual(
            outcomes_by_case_id["authorial_synthesis_valid"],
            boundary.OUTCOME_VALID_PRESERVED,
        )
        self.assertEqual(
            outcomes_by_case_id["authorial_synthesis_invalid"],
            boundary.OUTCOME_INVALID_BLOCKED,
        )

    def test_module_imports_no_runtime_writer_quality_repair_provider_boundaries(self) -> None:
        source = Path(boundary.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported_modules = {
            node.module
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module
        }
        imported_symbols = {
            alias.name
            for node in tree.body
            if isinstance(node, ast.ImportFrom)
            for alias in node.names
        }

        self.assertNotIn("services.packaging.generator", imported_modules)
        self.assertNotIn("apps.packaging.models", imported_modules)
        self.assertNotIn(
            "services.packaging.linkedin_post_candidate_writer_execution",
            imported_modules,
        )
        self.assertNotIn(
            "services.packaging.linkedin_post_quality_evaluator_execution",
            imported_modules,
        )
        self.assertNotIn(
            "services.packaging.linkedin_post_repair_writer_execution",
            imported_modules,
        )
        self.assertNotIn(
            "services.packaging.linkedin_post_semantic_grounding_execution",
            imported_modules,
        )
        self.assertNotIn("execute_semantic_grounding_prompt", imported_symbols)
        self.assertNotIn("build_ai_client", source)
        self.assertNotIn("Content" + "Package", source)


def _record(**overrides) -> dict:
    record = {
        "plan_id": "gpt_grounding",
        "execution_status": boundary.RUN_STATUS_PASS,
        "execution_success": True,
        "parse_success": True,
        "normalization_success": True,
        "expected_label": boundary.EXPECTED_VALID,
        "grounding_pass": True,
        "blocking_claim_ids": [],
        "voice_sensitive": False,
        "failure_stage": None,
        "must_inspect_fragment_coverage": [],
    }
    record.update(overrides)
    return record


def _review_payload(*, passed: bool = True, blocking: bool = False) -> dict:
    claim = {
        "claim_id": "c1",
        "field_name": "post_text",
        "value_index": None,
        "claim_text": "Growth signals prove the risk phase is over."
        if blocking
        else "Growth signals do not erase risk signals.",
        "claim_type": "author_interpretation",
        "support_status": "unsupported" if blocking else "supported",
        "severity": "major" if blocking else "info",
        "supported_evidence_ids": [],
        "required_qualifications": [],
        "missing_qualifications": [],
        "rationale": "Fake benchmark response.",
        "repair_hint": "Remove unsupported conclusion." if blocking else "",
    }
    return {
        "pass": passed,
        "claims": [claim],
        "failed_claim_ids": ["c1"] if blocking else [],
        "automatic_fail_reason": "unsupported conclusion" if blocking else "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": blocking,
        "repair_instructions": [
            {
                "claim_id": "c1",
                "instruction": "Remove unsupported conclusion.",
            }
        ]
        if blocking
        else [],
    }


def _raw(payload: dict) -> SemanticGroundingRawResponse:
    return SemanticGroundingRawResponse(
        raw_text=json.dumps(payload),
        provider="openai",
        model="gpt-4.1-2025-04-14",
    )


def _fixed_now() -> datetime:
    return datetime(2026, 8, 13, 12, 0, 0, tzinfo=UTC)


def _normalize(value: str) -> str:
    return " ".join(boundary.WORD_RE.findall(value.lower()))
