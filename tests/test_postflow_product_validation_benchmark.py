from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from django.test import SimpleTestCase

from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)
from services.packaging.linkedin_post_repair_writer_execution import (
    REPAIR_WRITER_CANDIDATE_POST_RESPONSE_SCHEMA,
    REPAIR_WRITER_EXECUTION_PATH_NATIVE_GEMINI,
    REPAIR_WRITER_NATIVE_GEMINI_THINKING_BUDGET,
    RepairWriterRawResponse,
)
from services.packaging.linkedin_post_semantic_grounding_execution import (
    SemanticGroundingRawResponse,
)
from services.packaging import postflow_product_validation_corpus as corpus
from services.packaging import postflow_product_validation_live_execution as live_execution
from services.packaging import postflow_product_validation_metrics as metrics
from services.packaging import postflow_product_validation_runner as runner


class PostFlowProductValidationBenchmarkTests(SimpleTestCase):
    def test_manifest_loads_exactly_33_cases_and_frozen_anchors(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()

        self.assertEqual(len(manifest.cases), 33)
        self.assertEqual(manifest.target_case_count, 33)
        self.assertEqual(
            tuple(manifest.frozen_anchor_case_ids),
            corpus.FROZEN_ANCHOR_CASE_IDS,
        )
        self.assertTrue(
            set(corpus.FROZEN_ANCHOR_CASE_IDS).issubset(
                {case.case_id for case in manifest.cases}
            )
        )

    def test_manifest_includes_expected_case_family_mix(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        families = [case.family for case in manifest.cases]

        self.assertEqual(families.count(corpus.CASE_FAMILY_SOURCE_ARTICLE), 5)
        self.assertEqual(families.count(corpus.CASE_FAMILY_CANDIDATE_QE), 12)
        self.assertEqual(families.count(corpus.CASE_FAMILY_SEMANTIC_BOUNDARY), 16)

    def test_manifest_references_existing_fixtures_by_hash(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()

        for case in manifest.cases:
            self.assertTrue(case.fixture_path.exists(), case.case_id)
            self.assertEqual(corpus.sha256_file(case.fixture_path), case.sha256)

    def test_boundary_cases_resolve_to_single_nested_case(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            case for case in manifest.cases
            if case.case_id == "boundary:authorial_synthesis_valid"
        )

        payload = corpus.resolve_product_validation_case(case)

        self.assertEqual(payload["case_id"], "authorial_synthesis_valid")
        self.assertIn("candidate_post", payload)

    def test_runner_writes_provider_free_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(
                    output_root=Path(tempdir),
                    export_human_review=True,
                ),
                now_factory=_fixed_now,
            )

            self.assertEqual(result.status, runner.STATUS_DRY_RUN)
            self.assertEqual(result.exit_code, 0)
            self.assertEqual(result.provider_call_count, 0)
            self.assertEqual(result.run_count, 33)
            self.assertEqual(result.corpus_case_count, 33)
            artifacts = result.artifacts.to_dict()
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), path)

    def test_runner_live_mode_requires_explicit_allow_api(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(
                    output_root=Path(tempdir),
                    allow_api=False,
                    dry_run=False,
                ),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.status, runner.STATUS_CONFIG_ERROR)
        self.assertIn("live execution requires --allow-api", result.safe_failure_message)

    def test_live_mode_routes_each_case_family_through_live_executor(self) -> None:
        calls: list[tuple[str, str]] = []

        def fake_live_case(**kwargs):
            case = kwargs["case"]
            calls.append((case.case_id, case.family))
            planned = live_execution.planned_provider_calls_for_case_family(case.family)
            if case.family == corpus.CASE_FAMILY_SOURCE_ARTICLE:
                outcome = live_execution.LIVE_SOURCE_READY
                actual = {key: 0 for key in planned}
            elif case.family == corpus.CASE_FAMILY_SEMANTIC_BOUNDARY:
                outcome = live_execution.LIVE_BOUNDARY_VALID_PRESERVED
                actual = {key: 0 for key in planned}
                actual["semantic_grounding_provider_api_calls"] = 1
            else:
                outcome = live_execution.LIVE_ACCEPTED_FIRST_ATTEMPT
                actual = {key: 0 for key in planned}
                actual["semantic_grounding_provider_api_calls"] = 1
                actual["quality_evaluator_provider_api_calls"] = 1
            return {
                "live_execution_status": live_execution.LIVE_EXECUTION_COMPLETED,
                "live_outcome": outcome,
                "live_failure_category": "product_behavior",
                "live_failure_code": None,
                "live_failure_stage": None,
                "live_stage_outcomes": {},
                "provider_invocation_counts": actual,
                "total_provider_calls": sum(actual.values()),
                "publication_packaging_invocations": 0,
            }

        with tempfile.TemporaryDirectory() as tempdir:
            with patch.object(runner, "execute_product_validation_live_case", fake_live_case):
                result = runner.run_product_validation_benchmark(
                    runner.ProductValidationRequest(
                        output_root=Path(tempdir),
                        allow_api=True,
                        dry_run=False,
                    ),
                    now_factory=_fixed_now,
                )

        self.assertEqual(result.status, runner.STATUS_COMPLETED)
        self.assertEqual(len(calls), 33)
        self.assertEqual(
            [family for _, family in calls].count(corpus.CASE_FAMILY_SOURCE_ARTICLE),
            5,
        )
        self.assertEqual(
            [family for _, family in calls].count(corpus.CASE_FAMILY_SEMANTIC_BOUNDARY),
            16,
        )
        self.assertEqual(
            [family for _, family in calls].count(corpus.CASE_FAMILY_CANDIDATE_QE),
            12,
        )
        self.assertEqual(result.provider_call_count, 40)

    def test_dry_run_plans_single_repair_continuation_for_product_cases(self) -> None:
        planned = live_execution.planned_provider_calls_for_case_family(
            corpus.CASE_FAMILY_CANDIDATE_QE
        )

        self.assertEqual(planned["candidate_writer_provider_api_calls"], 0)
        self.assertEqual(planned["semantic_grounding_provider_api_calls"], 2)
        self.assertEqual(planned["quality_evaluator_provider_api_calls"], 2)
        self.assertEqual(planned["repair_writer_provider_api_calls"], 1)
        self.assertEqual(planned["publication_packaging_invocations"], 0)

    def test_historical_labels_are_separate_from_live_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

        record = result.run_records[0]
        self.assertIn("historical_expected_outcome", record)
        self.assertIn("historical_classification", record)
        self.assertIn("historical_notes", record)
        self.assertIn("live_execution_status", record)
        self.assertIn("live_outcome", record)
        self.assertEqual(record["live_execution_status"], runner.STATUS_DRY_RUN)
        self.assertIsNone(record["live_outcome"])

    def test_runtime_input_fingerprint_excludes_historical_labels(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        runtime_input = live_execution.runtime_input_for_case(case, payload)

        forbidden = {
            "expected_outcome",
            "historical_outcome",
            "historical_expected_outcome",
            "historical_classification",
            "risk_type",
            "notes",
        }
        self.assertTrue(forbidden.isdisjoint(runtime_input))
        self.assertEqual(
            set(runtime_input),
            {"candidate_payload", "post_brief", "angle_decision", "selected_evidence"},
        )

    def test_live_product_case_uses_fixed_candidate_and_fake_downstream_executors(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        calls = {"grounding": 0, "quality": 0}

        def fake_grounding(_request):
            calls["grounding"] += 1
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_semantic_grounding_pass_payload(payload)),
                provider="gemini",
                model="gemini-3.6-flash",
            )

        def fake_quality(_request):
            calls["quality"] += 1
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(_quality_review_pass_payload()),
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        record = live_execution.execute_product_validation_live_case(
            case=case,
            fixture_payload=payload,
            experiment_id="fake-live-product",
            started_at=_fixed_now().isoformat(),
            completed_at=_fixed_now().isoformat(),
            executors=live_execution.ProductValidationLiveExecutors(
                semantic_grounding_executor=fake_grounding,
                quality_evaluator_executor=fake_quality,
            ),
        )

        self.assertEqual(record["live_outcome"], live_execution.LIVE_ACCEPTED_FIRST_ATTEMPT)
        self.assertEqual(record["provider_invocation_counts"]["candidate_writer_provider_api_calls"], 0)
        self.assertEqual(record["provider_invocation_counts"]["semantic_grounding_provider_api_calls"], 1)
        self.assertEqual(record["provider_invocation_counts"]["quality_evaluator_provider_api_calls"], 1)
        self.assertEqual(record["total_provider_calls"], 2)
        self.assertEqual(calls, {"grounding": 1, "quality": 1})

    def test_repair_required_product_case_continues_through_repair_once(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        calls = {"grounding": 0, "quality": 0, "repair": 0}
        quality_payloads = [
            _quality_review_fail_payload(),
            _quality_review_pass_payload(),
        ]

        def fake_grounding(_request):
            calls["grounding"] += 1
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_semantic_grounding_pass_payload(payload)),
                provider="gemini",
                model="gemini-3.6-flash",
            )

        def fake_quality(_request):
            calls["quality"] += 1
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(quality_payloads.pop(0)),
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        def fake_repair(_request):
            calls["repair"] += 1
            self.assertEqual(_request.provider, "gemini")
            self.assertEqual(_request.model, "gemini-3.6-flash")
            self.assertEqual(
                _request.execution_path,
                REPAIR_WRITER_EXECUTION_PATH_NATIVE_GEMINI,
            )
            self.assertEqual(
                _request.thinking_budget,
                REPAIR_WRITER_NATIVE_GEMINI_THINKING_BUDGET,
            )
            self.assertEqual(
                _request.response_schema,
                REPAIR_WRITER_CANDIDATE_POST_RESPONSE_SCHEMA,
            )
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": "Repaired human post."}),
                provider="gemini",
                model="gemini-3.6-flash",
                prompt_metadata=_repair_prompt_metadata(),
            )

        record = live_execution.execute_product_validation_live_case(
            case=case,
            fixture_payload=payload,
            experiment_id="fake-live-product-repair-needed",
            started_at=_fixed_now().isoformat(),
            completed_at=_fixed_now().isoformat(),
            executors=live_execution.ProductValidationLiveExecutors(
                semantic_grounding_executor=fake_grounding,
                quality_evaluator_executor=fake_quality,
                repair_writer_executor=fake_repair,
            ),
        )

        self.assertEqual(record["live_outcome"], live_execution.LIVE_REPAIR_ACCEPTED)
        self.assertEqual(record["provider_invocation_counts"]["candidate_writer_provider_api_calls"], 0)
        self.assertEqual(record["provider_invocation_counts"]["semantic_grounding_provider_api_calls"], 2)
        self.assertEqual(record["provider_invocation_counts"]["quality_evaluator_provider_api_calls"], 2)
        self.assertEqual(record["provider_invocation_counts"]["repair_writer_provider_api_calls"], 1)
        self.assertEqual(record["total_provider_calls"], 5)
        self.assertEqual(calls, {"grounding": 2, "quality": 2, "repair": 1})
        self.assertEqual(record["repaired_candidate"]["post_text"], "Repaired human post.")
        self.assertEqual(record["final_candidate"]["post_text"], "Repaired human post.")
        self.assertTrue(record["target_repair_success"])
        self.assertEqual(
            record["repair_target_enforcement"]["initiating_failed_criterion"],
            "author_point_of_view",
        )
        self.assertIn("repair", record["live_stage_outcomes"])
        self.assertTrue(record["live_stage_outcomes"]["repair"]["repair_executed"])

    def test_repair_target_comes_from_live_quality_review_not_historical_labels(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        payload["expected_outcome"] = "historical_label_must_not_drive_repair"
        human_voice_failure = _quality_review_pass_payload()
        human_voice_failure["pass"] = False
        human_voice_failure["scores"]["human_voice"] = 3
        human_voice_failure["total_score"] = 36
        human_voice_failure["failed_criteria"] = ["human_voice"]
        human_voice_failure["notes"] = ["Needs a more human voice."]
        human_voice_failure["criterion_rationales"]["human_voice"]["score"] = 3
        human_voice_failure["criterion_rationales"]["human_voice"]["failure_reason"] = "Needs repair."
        quality_payloads = [
            human_voice_failure,
            _quality_review_pass_payload(),
        ]

        def fake_grounding(_request):
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_semantic_grounding_pass_payload(payload)),
                provider="gemini",
                model="gemini-3.6-flash",
            )

        def fake_quality(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(quality_payloads.pop(0)),
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        def fake_repair(_request):
            self.assertIn('"failed_criterion": "human_voice"', _request.rendered_prompt_input.input_text)
            self.assertNotIn("historical_label_must_not_drive_repair", _request.rendered_prompt_input.input_text)
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": "Repaired human voice post."}),
                provider="gemini",
                model="gemini-3.6-flash",
                prompt_metadata=_repair_prompt_metadata(),
            )

        record = live_execution.execute_product_validation_live_case(
            case=case,
            fixture_payload=payload,
            experiment_id="fake-live-product-live-target",
            started_at=_fixed_now().isoformat(),
            completed_at=_fixed_now().isoformat(),
            executors=live_execution.ProductValidationLiveExecutors(
                semantic_grounding_executor=fake_grounding,
                quality_evaluator_executor=fake_quality,
                repair_writer_executor=fake_repair,
            ),
        )

        self.assertEqual(
            record["repair_target_enforcement"]["initiating_failed_criterion"],
            "human_voice",
        )

    def test_post_repair_quality_rejection_counts_as_repair_attempt(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        cta_failure = _quality_review_pass_payload()
        cta_failure["pass"] = False
        cta_failure["scores"]["cta"] = 3
        cta_failure["total_score"] = 37
        cta_failure["failed_criteria"] = ["cta"]
        cta_failure["notes"] = ["CTA needs repair."]
        cta_failure["criterion_rationales"]["cta"]["score"] = 3
        cta_failure["criterion_rationales"]["cta"]["failure_reason"] = "Needs repair."
        unrelated_failure = _quality_review_pass_payload()
        unrelated_failure["pass"] = False
        unrelated_failure["scores"]["human_voice"] = 3
        unrelated_failure["total_score"] = 36
        unrelated_failure["failed_criteria"] = ["human_voice"]
        unrelated_failure["notes"] = ["Still needs a more human voice."]
        unrelated_failure["criterion_rationales"]["human_voice"]["score"] = 3
        unrelated_failure["criterion_rationales"]["human_voice"]["failure_reason"] = "Needs repair."
        quality_payloads = [cta_failure, unrelated_failure]

        def fake_grounding(_request):
            return SemanticGroundingRawResponse(
                raw_text=json.dumps(_semantic_grounding_pass_payload(payload)),
                provider="gemini",
                model="gemini-3.6-flash",
            )

        def fake_quality(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(quality_payloads.pop(0)),
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        def fake_repair(_request):
            return RepairWriterRawResponse(
                raw_text=json.dumps({"post_text": "Repaired but still weak post."}),
                provider="gemini",
                model="gemini-3.6-flash",
                prompt_metadata=_repair_prompt_metadata(),
            )

        record = live_execution.execute_product_validation_live_case(
            case=case,
            fixture_payload=payload,
            experiment_id="fake-live-product-repair-still-rejected",
            started_at=_fixed_now().isoformat(),
            completed_at=_fixed_now().isoformat(),
            executors=live_execution.ProductValidationLiveExecutors(
                semantic_grounding_executor=fake_grounding,
                quality_evaluator_executor=fake_quality,
                repair_writer_executor=fake_repair,
            ),
        )
        record["family"] = corpus.CASE_FAMILY_CANDIDATE_QE
        computed = metrics.compute_product_validation_metrics((record,))
        product_metrics = computed["product_metrics"]

        self.assertEqual(record["live_outcome"], live_execution.LIVE_QE_REJECTED)
        self.assertEqual(record["live_failure_category"], corpus.FAILURE_CATEGORY_PRODUCT)
        self.assertTrue(record["live_stage_outcomes"]["repair"]["repair_executed"])
        self.assertEqual(product_metrics["repair_attempt_count"], 1)
        self.assertEqual(product_metrics["repair_attempt_rate"], 1.0)
        self.assertEqual(product_metrics["final_not_ready_count"], 1)
        self.assertEqual(product_metrics["repair_accepted_count"], 0)
        self.assertEqual(product_metrics["repair_accepted_rate"], 0.0)
        self.assertEqual(product_metrics["TARGET_FIXED_CLEAN_count"], 0)

    def test_target_not_fixed_metrics_use_live_repair_target(self) -> None:
        record = {
            "family": corpus.CASE_FAMILY_CANDIDATE_QE,
            "live_execution_status": live_execution.LIVE_EXECUTION_COMPLETED,
            "live_outcome": live_execution.LIVE_REPAIR_TARGET_NOT_FIXED,
            "live_failure_category": corpus.FAILURE_CATEGORY_PRODUCT,
            "live_stage_outcomes": {"repair": {"repair_executed": True}},
            "provider_invocation_counts": {
                "candidate_writer_provider_api_calls": 0,
                "semantic_grounding_provider_api_calls": 2,
                "quality_evaluator_provider_api_calls": 2,
                "repair_writer_provider_api_calls": 1,
                "publication_packaging_invocations": 0,
            },
            "product_case_distribution": {
                "length": 900,
                "repair_target": "historical_target_must_not_drive_signature",
            },
            "repair_target_enforcement": {
                "initiating_failed_criterion": "human_voice",
                "repair_target_fixed": False,
            },
            "quality_review_summary": {"failed_criteria": ["human_voice"]},
        }

        record["family"] = corpus.CASE_FAMILY_CANDIDATE_QE
        computed = metrics.compute_product_validation_metrics((record,))
        product_metrics = computed["product_metrics"]
        signatures = computed["repeated_failure_signatures"]["signature_counts"]

        self.assertEqual(product_metrics["repair_attempt_count"], 1)
        self.assertEqual(product_metrics["target_not_fixed_count"], 1)
        self.assertEqual(product_metrics["target_not_fixed_rate"], 1.0)
        self.assertIn("repair_target_not_fixed:human_voice", signatures)
        self.assertNotIn(
            "repair_target_not_fixed:historical_target_must_not_drive_signature",
            signatures,
        )
    def test_metrics_separate_product_and_infrastructure_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(
                    output_root=Path(tempdir),
                    export_human_review=True,
                ),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.metrics["provider_call_count"], 0)
        self.assertIn("product_metrics", result.metrics)
        self.assertIn("boundary_metrics", result.metrics)
        self.assertIn("source_metrics", result.metrics)
        self.assertIn("critical_invariants", result.metrics)
        self.assertIn("repeated_failure_signatures", result.metrics)
        self.assertEqual(result.metrics["product_metrics"]["product_case_count"], 12)
        self.assertEqual(result.metrics["product_metrics"]["live_product_case_count"], 0)
        self.assertEqual(result.metrics["product_metrics"]["live_product_not_measured_count"], 12)
        self.assertEqual(result.metrics["product_metrics"]["product_evaluable_case_count"], 0)
        self.assertIsNone(result.metrics["product_metrics"]["first_attempt_acceptance_rate"])
        self.assertIsNone(result.metrics["product_metrics"]["evaluable_final_acceptance_rate"])
        self.assertEqual(result.metrics["boundary_metrics"]["boundary_case_count"], 16)
        self.assertEqual(result.metrics["source_metrics"]["source_case_count"], 5)
        self.assertEqual(result.metrics["source_metrics"]["source_valid"], 5)
        self.assertEqual(result.metrics["source_metrics"]["source_invalid"], 0)
        self.assertGreater(result.metrics["product_failure_case_count"], 0)
        self.assertEqual(result.metrics["infrastructure_failure_case_count"], 0)
        self.assertEqual(result.metrics["human_review_case_count"], 10)

    def test_product_cases_include_structured_human_ground_truth(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        product_cases = [
            case for case in manifest.cases
            if case.family == corpus.CASE_FAMILY_CANDIDATE_QE
        ]

        self.assertEqual(len(product_cases), 12)
        for case in product_cases:
            truth = case.human_ground_truth
            self.assertIsInstance(truth, dict, case.case_id)
            self.assertIn(truth["final_disposition"], {"ACCEPT", "REPAIR", "NOT_READY"})
            self.assertIn(truth["review_status"], {"REVIEWED", "NEEDS_SECOND_REVIEW"})
            self.assertIn(truth["confidence"], {"HIGH", "MEDIUM", "LOW"})
            self.assertTrue(case.independence_group)

    def test_truncated_product_artifacts_are_not_labeled_accepted_or_repairable(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        product_cases = [
            case for case in manifest.cases
            if case.family == corpus.CASE_FAMILY_CANDIDATE_QE
        ]

        truncated_cases = []
        for case in product_cases:
            payload = corpus.resolve_product_validation_case(case)
            post_text = payload["candidate_payload"]["post_text"]
            if (
                payload.get("artifact_completeness_status") == "TRUNCATED_SOURCE_ARTIFACT"
                or len(post_text) == 500 and not post_text.rstrip().endswith((".", "?", "!"))
            ):
                truncated_cases.append(case.case_id)
                truth = case.human_ground_truth
                self.assertEqual(payload["artifact_completeness_status"], "TRUNCATED_SOURCE_ARTIFACT")
                self.assertEqual(case.failure_category, corpus.FAILURE_CATEGORY_PRODUCT)
                self.assertEqual(case.risk_type, "truncated_artifact_product_review")
                self.assertEqual(truth["final_disposition"], "NOT_READY", case.case_id)
                self.assertEqual(truth["length_validity"], "FAIL", case.case_id)
                self.assertEqual(truth["repair_target"], "", case.case_id)
                self.assertEqual(truth["repair_locality"], "none", case.case_id)
                self.assertIn("truncated", truth["primary_reason"].lower())

        self.assertEqual(len(truncated_cases), 4)

    def test_runtime_input_fingerprint_excludes_human_ground_truth(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
            and item.human_ground_truth
        )
        payload = corpus.resolve_product_validation_case(case)
        original_truth = case.human_ground_truth
        runtime_input = live_execution.runtime_input_for_case(case, payload)

        forbidden = {
            "human_ground_truth",
            "human_final_disposition",
            "historical_expected_outcome",
            "independence_group",
            "review_status",
            "confidence",
        }
        self.assertTrue(forbidden.isdisjoint(runtime_input))
        self.assertEqual(case.human_ground_truth, original_truth)

    def test_human_ground_truth_label_does_not_change_runtime_input(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        changed_case = corpus.ProductValidationCorpusCase(
            **{
                **case.to_dict(),
                "human_ground_truth": {
                    **case.human_ground_truth,
                    "final_disposition": "NOT_READY",
                    "primary_reason": "Changed label must remain benchmark-only.",
                },
            }
        )

        self.assertEqual(
            live_execution.runtime_input_for_case(case, payload),
            live_execution.runtime_input_for_case(changed_case, payload),
        )

    def test_changing_human_ground_truth_does_not_change_live_role_requests(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        payload = corpus.resolve_product_validation_case(case)
        changed_case = corpus.ProductValidationCorpusCase(
            **{
                **case.to_dict(),
                "human_ground_truth": {
                    **case.human_ground_truth,
                    "final_disposition": "NOT_READY",
                    "primary_reason": "Changed human label must not alter live runtime.",
                    "genericization": "MATERIAL",
                },
            }
        )

        def run_with(target_case):
            captured: dict[str, list[str]] = {"grounding": [], "quality": [], "repair": []}
            quality_payloads = [_quality_review_fail_payload(), _quality_review_pass_payload()]

            def fake_grounding(request):
                captured["grounding"].append(request.rendered_prompt_input.input_text)
                return SemanticGroundingRawResponse(
                    raw_text=json.dumps(_semantic_grounding_pass_payload(payload)),
                    provider="gemini",
                    model="gemini-3.6-flash",
                )

            def fake_quality(request):
                captured["quality"].append(request.rendered_prompt_input.input_text)
                return QualityEvaluatorRawResponse(
                    raw_text=json.dumps(quality_payloads.pop(0)),
                    provider="openai",
                    model="gpt-4.1-2025-04-14",
                )

            def fake_repair(request):
                captured["repair"].append(request.rendered_prompt_input.input_text)
                return RepairWriterRawResponse(
                    raw_text=json.dumps({"post_text": "Repaired human post."}),
                    provider="gemini",
                    model="gemini-3.6-flash",
                    prompt_metadata=_repair_prompt_metadata(),
                )

            record = live_execution.execute_product_validation_live_case(
                case=target_case,
                fixture_payload=payload,
                experiment_id="human-truth-runtime-isolation",
                started_at=_fixed_now().isoformat(),
                completed_at=_fixed_now().isoformat(),
                executors=live_execution.ProductValidationLiveExecutors(
                    semantic_grounding_executor=fake_grounding,
                    quality_evaluator_executor=fake_quality,
                    repair_writer_executor=fake_repair,
                ),
            )
            return captured, record

        original_requests, original_record = run_with(case)
        changed_requests, changed_record = run_with(changed_case)

        self.assertEqual(original_requests, changed_requests)
        for record in (original_record, changed_record):
            self.assertEqual(record["live_outcome"], live_execution.LIVE_REPAIR_ACCEPTED)
            self.assertNotIn("human_ground_truth", json.dumps(record, sort_keys=True))
        self.assertEqual(original_record["live_stage_outcomes"], changed_record["live_stage_outcomes"])
        self.assertEqual(original_record["provider_invocation_counts"], changed_record["provider_invocation_counts"])

    def test_record_uses_explicit_historical_expected_outcome_when_present(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        case = next(
            item for item in manifest.cases
            if item.family == corpus.CASE_FAMILY_CANDIDATE_QE
        )
        changed_case_payload = case.to_dict()
        changed_case_payload["fixture_path"] = Path(changed_case_payload["fixture_path"])
        changed_case_payload["historical_expected_outcome"] = "historical_inconclusive"
        changed_case = corpus.ProductValidationCorpusCase(**changed_case_payload)

        with patch.object(
            runner,
            "load_product_validation_corpus_manifest",
            return_value=corpus.ProductValidationCorpusManifest(
                corpus_id=manifest.corpus_id,
                schema_version=manifest.schema_version,
                target_case_count=1,
                frozen_anchor_case_ids=(),
                cases=(changed_case,),
                frozen_model_role_configuration=runner.FROZEN_ROLE_CONFIGURATION,
                provenance=manifest.provenance,
            ),
        ), tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

        self.assertEqual(
            result.run_records[0]["historical_expected_outcome"],
            "historical_inconclusive",
        )
        self.assertEqual(result.run_records[0]["historical_outcome"], "historical_inconclusive")

    def test_dry_run_duplicate_candidate_hashes_are_reported_from_fixture_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

        distribution = result.metrics["product_case_distribution"]

        self.assertEqual(distribution["duplicate_candidate_hash_count"], 0)
        self.assertEqual(distribution["raw_product_case_count"], 12)
        self.assertTrue(distribution["rates_should_be_read_with_independence_groups"])
        self.assertIn("candidate_text_hash", result.run_records[5]["product_case_distribution"])

    def test_human_ground_truth_metrics_include_confusion_and_genericization_counts(self) -> None:
        record = {
            "family": corpus.CASE_FAMILY_CANDIDATE_QE,
            "live_execution_status": live_execution.LIVE_EXECUTION_COMPLETED,
            "live_outcome": live_execution.LIVE_ACCEPTED_FIRST_ATTEMPT,
            "live_failure_category": corpus.FAILURE_CATEGORY_PRODUCT_BEHAVIOR,
            "human_ground_truth": {
                "final_disposition": "NOT_READY",
                "genericization": "MATERIAL",
                "grounding_fidelity": "PASS",
            },
            "product_case_distribution": {"length": 700, "length_band": "medium"},
            "provider_invocation_counts": {},
        }

        computed = metrics.compute_product_validation_metrics((record,))
        human_metrics = computed["human_ground_truth_metrics"]

        self.assertEqual(human_metrics["human_ground_truth_case_count"], 1)
        self.assertEqual(human_metrics["runtime_accept_vs_human_not_ready"], 1)
        self.assertEqual(human_metrics["unsafe_accept_count"], 1)
        self.assertEqual(human_metrics["human_material_genericization_count"], 1)
        self.assertEqual(
            human_metrics["runtime_accepted_material_genericization_count"],
            1,
        )

    def test_repeated_failure_signatures_count_independent_groups(self) -> None:
        records = (
            {
                "family": corpus.CASE_FAMILY_CANDIDATE_QE,
                "case_id": "a",
                "independence_group": "same-source",
                "live_outcome": live_execution.LIVE_REPAIR_TARGET_NOT_FIXED,
                "live_stage_outcomes": {"repair": {"repair_executed": True}},
                "repair_target_enforcement": {"initiating_failed_criterion": "cta"},
                "product_case_distribution": {"length": 700},
                "provider_invocation_counts": {},
            },
            {
                "family": corpus.CASE_FAMILY_CANDIDATE_QE,
                "case_id": "b",
                "independence_group": "same-source",
                "live_outcome": live_execution.LIVE_REPAIR_TARGET_NOT_FIXED,
                "live_stage_outcomes": {"repair": {"repair_executed": True}},
                "repair_target_enforcement": {"initiating_failed_criterion": "cta"},
                "product_case_distribution": {"length": 700},
                "provider_invocation_counts": {},
            },
        )

        signatures = metrics.compute_product_validation_metrics(records)[
            "repeated_failure_signatures"
        ]

        self.assertEqual(signatures["signature_counts"]["repair_target_not_fixed:cta"], 2)
        self.assertEqual(
            signatures["independent_group_counts"]["repair_target_not_fixed:cta"],
            1,
        )
        self.assertEqual(signatures["candidate_systemic_issues"], {})

    def test_configuration_fingerprint_contains_frozen_roles_and_hashes(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        fingerprint = runner.build_configuration_fingerprint(manifest, now=_fixed_now())

        roles = fingerprint["frozen_model_role_configuration"]
        self.assertEqual(roles["candidate_writer"]["provider"], "anthropic")
        self.assertEqual(roles["semantic_grounding"]["provider"], "gemini")
        self.assertEqual(roles["quality_evaluator"]["provider"], "openai")
        self.assertEqual(roles["repair_writer"]["provider"], "gemini")
        self.assertEqual(roles["repair_writer"]["execution_path"], "native_gemini")
        self.assertEqual(roles["repair_writer"]["thinking_budget"], 256)
        self.assertEqual(roles["repair_writer"]["max_output_tokens"], 2800)
        self.assertEqual(fingerprint["provider_call_count"], 0)
        self.assertFalse(fingerprint["live_mode"])
        self.assertIn("benchmark_runner_version", fingerprint)
        self.assertIn("manifest_hash", fingerprint)
        self.assertTrue(fingerprint["fixture_hashes"])
        self.assertIn("missing_contract_file_paths", fingerprint)
        self.assertEqual(fingerprint["missing_contract_file_paths"], [])
        self.assertIn(
            "services/packaging/linkedin_post_repair_writer_execution.py",
            fingerprint["contract_file_hashes"],
        )

    def test_human_review_exports_expected_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(
                    output_root=Path(tempdir),
                    export_human_review=True,
                ),
                now_factory=_fixed_now,
            )

            human_csv = Path(result.artifacts.human_review_csv)
            header = human_csv.read_text(encoding="utf-8").splitlines()[0]

        for column in (
            "case_id",
            "family",
            "topic",
            "direction",
            "historical_expected_outcome",
            "live_outcome",
            "original_candidate",
            "final_candidate",
            "repair_target",
            "qe_scores",
            "failed_criteria",
            "human_review_priority",
            "human_final_disposition",
            "human_primary_reason",
            "human_repair_target",
            "human_genericization",
            "review_confidence",
            "review_status",
            "would_publish_yes_no",
            "reviewer_label",
            "backlog_action",
        ):
            self.assertIn(column, header)

    def test_product_corpus_is_reported_as_minimum_met_for_live_product_rates(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

            manifest_payload = json.loads(
                Path(result.artifacts.corpus_manifest_json).read_text(encoding="utf-8")
            )
            report_text = Path(result.artifacts.report_md).read_text(encoding="utf-8")

        adequacy = manifest_payload["product_corpus_adequacy"]
        self.assertEqual(adequacy["status"], "PRODUCT_CORPUS_MINIMUM_MET")
        self.assertEqual(adequacy["current_product_case_count"], 12)
        self.assertEqual(adequacy["gap_to_minimum"], 0)
        product_metrics = result.metrics["product_metrics"]
        self.assertEqual(
            product_metrics["product_corpus_adequacy_status"],
            "PRODUCT_CORPUS_MINIMUM_MET",
        )
        self.assertTrue(product_metrics["product_rates_are_exploratory"])
        self.assertIn("## Product Corpus Adequacy", report_text)
        self.assertIn("PRODUCT_CORPUS_MINIMUM_MET", report_text)
        self.assertLess(
            report_text.index("## Live Outcome Taxonomy"),
            report_text.index("## Historical Outcome Taxonomy"),
        )

    def test_human_review_exports_are_omitted_when_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(
                    output_root=Path(tempdir),
                    export_human_review=False,
                ),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.artifacts.human_review_csv, "")
        self.assertEqual(result.artifacts.human_review_md, "")
        output_dir = Path(result.artifacts.output_dir)
        self.assertFalse((output_dir / "human_review.csv").exists())
        self.assertFalse((output_dir / "human_review.md").exists())

    def test_manifest_frozen_role_configuration_must_match_runner_policy(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        bad_manifest = corpus.ProductValidationCorpusManifest(
            corpus_id=manifest.corpus_id,
            schema_version=manifest.schema_version,
            target_case_count=manifest.target_case_count,
            frozen_anchor_case_ids=manifest.frozen_anchor_case_ids,
            cases=manifest.cases,
            frozen_model_role_configuration={"candidate_writer": {"provider": "openai"}},
            provenance=manifest.provenance,
        )

        with self.assertRaises(corpus.ProductValidationCorpusError):
            runner._validate_frozen_role_configuration(bad_manifest)

    def test_runner_rejects_overwriting_existing_artifact_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            output_dir = Path(tempdir) / runner.DEFAULT_EXPERIMENT_ID
            output_dir.mkdir(parents=True)
            (output_dir / "existing.txt").write_text("preserve", encoding="utf-8")

            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

        self.assertEqual(result.status, runner.STATUS_CONFIG_ERROR)
        self.assertEqual(result.safe_failure_code, "product_validation_corpus_invalid")
        self.assertIn("output directory already contains artifacts", result.safe_failure_message)

    def test_artifacts_do_not_contain_provider_secrets_or_raw_prompt_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            result = runner.run_product_validation_benchmark(
                runner.ProductValidationRequest(output_root=Path(tempdir)),
                now_factory=_fixed_now,
            )

            forbidden = (
                "api_key",
                "secret",
                "credential",
                "prompt_text",
                "raw_provider_response",
            )
            for key, artifact_path in result.artifacts.to_dict().items():
                if key == "output_dir" or not artifact_path:
                    continue
                text = Path(artifact_path).read_text(encoding="utf-8")
                for fragment in forbidden:
                    self.assertNotIn(fragment, text)

    def test_missing_fixture_fails_before_final_artifacts(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        bad_case = corpus.ProductValidationCorpusCase(
            **{
                **manifest.cases[0].to_dict(),
                "fixture_path": Path("tests/fixtures/does-not-exist.json"),
            }
        )
        bad_manifest = corpus.ProductValidationCorpusManifest(
            corpus_id=manifest.corpus_id,
            schema_version=manifest.schema_version,
            target_case_count=manifest.target_case_count,
            frozen_anchor_case_ids=manifest.frozen_anchor_case_ids,
            cases=(bad_case, *manifest.cases[1:]),
            frozen_model_role_configuration=manifest.frozen_model_role_configuration,
            provenance=manifest.provenance,
        )

        with self.assertRaises(corpus.ProductValidationCorpusError):
            corpus.resolve_product_validation_case(bad_manifest.cases[0])

    def test_benchmark_labels_do_not_enter_prompt_renderer_or_runtime_modules(self) -> None:
        for path in (
            Path("services/packaging/linkedin_post_prompt_renderers.py"),
            Path("services/packaging/linkedin_post_final_post_attempt_execution.py"),
            Path("services/packaging/linkedin_post_repair_writer_execution.py"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("known_accepted_projection", text)
            self.assertNotIn("boundary_invalid_blocked", text)

    def test_product_validation_modules_do_not_import_provider_execution(self) -> None:
        for path in (
            Path("services/packaging/postflow_product_validation_corpus.py"),
            Path("services/packaging/postflow_product_validation_runner.py"),
            Path("services/packaging/postflow_product_validation_metrics.py"),
            Path("services/packaging/postflow_product_validation_reports.py"),
        ):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("execute_candidate_writer_prompt", text)
            self.assertNotIn("execute_quality_evaluator_prompt", text)
            self.assertNotIn("execute_repair_writer_prompt", text)
            self.assertNotIn("OpenAIClient", text)
            self.assertNotIn("Gemini", text)
            self.assertNotIn("Anthropic", text)

    def test_live_execution_module_is_benchmark_only(self) -> None:
        text = Path(
            "services/packaging/postflow_product_validation_live_execution.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("generator.py", text)
        self.assertNotIn("ContentPackage", text)
        self.assertNotIn("django.db", text)

    def test_content_package_payload_contract_is_not_modified_by_scope(self) -> None:
        paths = [path.as_posix() for path in Path(".").glob("**/*") if path.is_file()]
        self.assertIn("services/packaging/postflow_product_validation_runner.py", paths)
        self.assertTrue(Path("services/packaging/linkedin_post_final_post_payload_contract.py").exists())


def _repair_prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_repair_writer",
        prompt_version="1.0",
        prompt_path=None,
    )


def _fixed_now() -> datetime:
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)


def _semantic_grounding_pass_payload(fixture_payload: dict) -> dict:
    evidence_id = fixture_payload["selected_evidence"][0]["evidence_id"]
    return {
        "pass": True,
        "claims": [
            {
                "claim_id": "c1",
                "field_name": "post_text",
                "value_index": None,
                "claim_text": "The post stays within selected evidence.",
                "claim_type": "attributed_source_claim",
                "support_status": "supported",
                "severity": "info",
                "supported_evidence_ids": [evidence_id],
                "required_qualifications": [],
                "missing_qualifications": [],
                "rationale": "Supported by selected evidence.",
                "repair_hint": "",
            }
        ],
        "failed_claim_ids": [],
        "automatic_fail_reason": "",
        "requires_human_review": False,
        "human_review_reason": "",
        "repairable": False,
        "repair_instructions": [],
    }


def _quality_review_pass_payload() -> dict:
    scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 4,
        "pattern_interrupt": 4,
        "evidence": 4,
        "author_point_of_view": 5,
        "human_voice": 5,
        "practical_value": 4,
        "cta": 4,
    }
    return {
        "scores": scores,
        "total_score": 38,
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
        "notes": ["Ready."],
        "criterion_rationales": {
            criterion: {
                "score": score,
                "max_score": 5,
                "rationale": f"{criterion} passes for this benchmark fixture.",
                "post_text_evidence": "A phrase from the candidate post.",
                "failure_reason": "",
            }
            for criterion, score in scores.items()
        },
    }


def _quality_review_fail_payload(
    *,
    failed_criteria: list[str] | None = None,
) -> dict:
    scores = {
        "hook": 4,
        "controlling_angle": 4,
        "reader_problem": 3,
        "pattern_interrupt": 3,
        "evidence": 4,
        "author_point_of_view": 2,
        "human_voice": 3,
        "practical_value": 3,
        "cta": 4,
    }
    failed = ["author_point_of_view", "human_voice"] if failed_criteria is None else failed_criteria
    return {
        "scores": scores,
        "total_score": 30,
        "pass": False,
        "failed_criteria": failed,
        "automatic_fail_reason": "",
        "notes": ["Needs a stronger author-owned point of view."],
        "criterion_rationales": {
            criterion: {
                "score": score,
                "max_score": 5,
                "rationale": f"{criterion} score recorded for this benchmark fixture.",
                "post_text_evidence": "A phrase from the candidate post.",
                "failure_reason": (
                    "Needs repair." if criterion in set(failed) else ""
                ),
            }
            for criterion, score in scores.items()
        },
    }
