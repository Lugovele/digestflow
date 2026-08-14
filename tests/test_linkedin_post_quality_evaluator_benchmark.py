from __future__ import annotations

import ast
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from django.test import SimpleTestCase

from services.packaging.linkedin_post_attempt_adjudication import QUALITY_EVALUATION_READY
from services.packaging.linkedin_post_quality_evaluator_benchmark import (
    BENCHMARK_STATUS_COMPLETED,
    BENCHMARK_STATUS_DRY_RUN,
    DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
    GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
    PLAN_GEMINI_QUALITY,
    PLAN_GPT_QUALITY,
    QualityEvaluatorBenchmarkPlan,
    QualityEvaluatorBenchmarkRequest,
    build_quality_evaluator_benchmark_prompt_render,
    default_quality_evaluator_benchmark_cases,
    default_quality_evaluator_benchmark_plans,
    quality_evaluator_benchmark_max_output_tokens_for_provider,
    run_quality_evaluator_benchmark,
)
from services.packaging.linkedin_post_quality_evaluator_execution import (
    QualityEvaluatorRawResponse,
)
from services.packaging.linkedin_post_quality_rubric_contract import QUALITY_CRITERIA


class QualityEvaluatorBenchmarkTests(SimpleTestCase):
    def test_default_cases_load_canonical_frozen_inputs(self) -> None:
        cases = default_quality_evaluator_benchmark_cases()

        self.assertEqual(
            [case.case_id for case in cases],
            ["topic_200_digest_134", "topic_140_digest_126"],
        )
        self.assertTrue(all(case.canonical_candidate_valid for case in cases))
        self.assertEqual(
            [item["evidence_id"] for item in cases[0].selected_evidence],
            cases[0].angle_decision["supporting_evidence_ids"],
        )
        self.assertEqual(cases[0].selected_evidence, tuple(cases[0].post_brief["evidence_to_use"]))

    def test_default_plans_compare_gpt_and_gemini_quality_evaluator(self) -> None:
        plans = default_quality_evaluator_benchmark_plans()

        self.assertEqual([plan.plan_id for plan in plans], [PLAN_GPT_QUALITY, PLAN_GEMINI_QUALITY])
        self.assertEqual(plans[0].provider, "openai")
        self.assertEqual(plans[0].model, "gpt-4.1-2025-04-14")
        self.assertEqual(plans[0].max_output_tokens, DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS)
        self.assertEqual(plans[1].provider, "gemini")
        self.assertEqual(plans[1].model, "gemini-3.6-flash")
        self.assertEqual(plans[1].max_output_tokens, GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS)

    def test_provider_budget_helper_uses_quality_evaluator_budgets(self) -> None:
        self.assertEqual(
            quality_evaluator_benchmark_max_output_tokens_for_provider("gemini"),
            GEMINI_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(
            quality_evaluator_benchmark_max_output_tokens_for_provider("openai"),
            DEFAULT_QUALITY_EVALUATOR_MAX_OUTPUT_TOKENS,
        )

    def test_dry_run_expands_cases_and_plans_without_provider_calls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            request = QualityEvaluatorBenchmarkRequest(
                experiment_id="quality-evaluator-dry-test",
                cases=default_quality_evaluator_benchmark_cases(),
                plans=default_quality_evaluator_benchmark_plans(),
                output_root=Path(temp_dir),
            )

            result = run_quality_evaluator_benchmark(request, now_factory=_fixed_now)

        self.assertEqual(result.status, BENCHMARK_STATUS_DRY_RUN)
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.run_count, 4)
        self.assertEqual(result.provider_call_count, 0)
        self.assertTrue(all(record["provider_invocation_counts"]["quality_evaluator"] == 0 for record in result.run_records))
        self.assertTrue(all(record["provider_invocation_counts"]["candidate_writer"] == 0 for record in result.run_records))
        self.assertTrue(all(record["provider_invocation_counts"]["semantic_grounding"] == 0 for record in result.run_records))
        self.assertTrue(all(record["provider_invocation_counts"]["repair_writer"] == 0 for record in result.run_records))
        self.assertTrue(all(record["provider_invocation_counts"]["publication_packaging"] == 0 for record in result.run_records))

    def test_input_hashes_are_identical_across_plans_for_same_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-hash-test",
                    cases=(default_quality_evaluator_benchmark_cases()[0],),
                    plans=default_quality_evaluator_benchmark_plans(),
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
            )

        first, second = result.run_records
        self.assertNotEqual(first["provider"], second["provider"])
        self.assertEqual(first["quality_input_summary"], second["quality_input_summary"])
        self.assertEqual(first["prompt_metadata"], second["prompt_metadata"])

    def test_live_fake_success_flows_through_parser_normalizer_and_adjudication(self) -> None:
        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(_quality_review_payload()),
                provider="openai",
                model="gpt-4.1-2025-04-14",
                usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-live-success",
                    cases=(default_quality_evaluator_benchmark_cases()[0],),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )

        record = result.run_records[0]
        self.assertEqual(result.status, BENCHMARK_STATUS_COMPLETED)
        self.assertEqual(result.provider_call_count, 1)
        self.assertTrue(record["execution_success"])
        self.assertTrue(record["parse_success"])
        self.assertTrue(record["normalization_success"])
        self.assertEqual(record["quality_evaluation_status"], QUALITY_EVALUATION_READY)
        self.assertEqual(record["quality_review"]["total_score"], 36)
        self.assertEqual(record["quality_review"]["pass"], True)
        self.assertEqual(record["adjudication_projection"]["decision"]["action"], "accept")
        self.assertEqual(record["adjudication_projection"]["outcome_status"], "accepted")
        self.assertTrue(record["adjudication_projection"]["accepted"])
        self.assertTrue(record["adjudication_projection"]["terminal"])
        self.assertFalse(record["adjudication_projection"]["repair_required"])

    def test_live_fake_empty_response_records_execution_failure(self) -> None:
        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text="",
                provider="openai",
                model="gpt-4.1-2025-04-14",
                execution_error="empty provider response",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-empty",
                    cases=(default_quality_evaluator_benchmark_cases()[0],),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "execution")
        self.assertEqual(record["failure_code"], "quality_evaluator_empty_response")
        self.assertFalse(record["parse_success"])
        self.assertFalse(record["normalization_success"])

    def test_live_fake_parse_failure_records_parse_failure(self) -> None:
        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text="not json",
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-parse",
                    cases=(default_quality_evaluator_benchmark_cases()[0],),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "parse")
        self.assertEqual(record["failure_code"], "quality_evaluator_parse_failure")
        self.assertFalse(record["parse_success"])

    def test_live_fake_normalization_failure_records_normalization_failure(self) -> None:
        payload = _quality_review_payload()
        payload["total_score"] = 45

        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(payload),
                provider="openai",
                model="gpt-4.1-2025-04-14",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-normalization",
                    cases=(default_quality_evaluator_benchmark_cases()[0],),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )

        record = result.run_records[0]
        self.assertEqual(record["failure_stage"], "normalization")
        self.assertEqual(record["failure_code"], "quality_evaluator_normalization_failure")
        self.assertTrue(record["parse_success"])
        self.assertFalse(record["normalization_success"])

    def test_dry_run_artifacts_are_sanitized_and_exclude_full_inputs(self) -> None:
        case = default_quality_evaluator_benchmark_cases()[0]
        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-artifacts",
                    cases=(case,),
                    plans=default_quality_evaluator_benchmark_plans(),
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
            )
            output_dir = Path(result.artifacts.output_dir)
            artifact_text = "\n".join(path.read_text(encoding="utf-8") for path in output_dir.iterdir())

        self.assertNotIn(case.candidate_payload["post_text"], artifact_text)
        for item in case.selected_evidence:
            self.assertNotIn(item["evidence_text"], artifact_text)
        self.assertNotIn("raw_provider_response", artifact_text)
        self.assertNotIn("prompt_text", artifact_text)
        self.assertNotIn("sk-", artifact_text.lower())


    def test_live_artifacts_summarize_evaluator_free_text_without_storing_quotes(self) -> None:
        case = default_quality_evaluator_benchmark_cases()[0]
        dangerous_text = case.candidate_payload["post_text"]
        dangerous_evidence = case.selected_evidence[0]["evidence_text"]

        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(
                    _quality_review_payload(dangerous_text, dangerous_evidence)
                ),
                provider="openai",
                model="gpt-4.1-2025-04-14",
                usage={"total_tokens": 30},
                raw_provider_response={"id": "resp_fake"},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-live-artifacts",
                    cases=(case,),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )
            output_dir = Path(result.artifacts.output_dir)
            artifact_text = "\n".join(
                path.read_text(encoding="utf-8") for path in output_dir.iterdir()
            )

        self.assertEqual(result.provider_call_count, 1)
        self.assertNotIn(dangerous_text, artifact_text)
        self.assertNotIn(dangerous_evidence, artifact_text)
        self.assertIn("rationale_summary", artifact_text)
        self.assertIn("post_text_evidence_summary", artifact_text)
        self.assertNotIn("raw_provider_response", artifact_text)
        self.assertNotIn("prompt_text", artifact_text)

    def test_failed_live_artifacts_sanitize_adjudication_reason(self) -> None:
        case = default_quality_evaluator_benchmark_cases()[0]
        dangerous_text = case.candidate_payload["post_text"]
        dangerous_evidence = case.selected_evidence[0]["evidence_text"]
        payload = _quality_review_payload(dangerous_text, dangerous_evidence)
        payload["scores"]["hook"] = 2
        payload["total_score"] = 34
        payload["pass"] = False
        payload["failed_criteria"] = ["hook"]
        payload["automatic_fail_reason"] = f"{dangerous_text} {dangerous_evidence}"
        payload["criterion_rationales"]["hook"]["score"] = 2
        payload["criterion_rationales"]["hook"]["failure_reason"] = dangerous_text

        def fake_executor(_request):
            return QualityEvaluatorRawResponse(
                raw_text=json.dumps(payload),
                provider="openai",
                model="gpt-4.1-2025-04-14",
                usage={"total_tokens": 30},
                raw_provider_response={"id": "resp_fake"},
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            result = run_quality_evaluator_benchmark(
                QualityEvaluatorBenchmarkRequest(
                    experiment_id="quality-evaluator-failed-artifacts",
                    cases=(case,),
                    plans=(default_quality_evaluator_benchmark_plans()[0],),
                    allow_api=True,
                    output_root=Path(temp_dir),
                ),
                now_factory=_fixed_now,
                quality_evaluator_executor=fake_executor,
            )
            output_dir = Path(result.artifacts.output_dir)
            artifact_text = "\n".join(
                path.read_text(encoding="utf-8") for path in output_dir.iterdir()
            )

        result_text = json.dumps(result.to_dict(), ensure_ascii=False)
        self.assertEqual(result.provider_call_count, 1)
        self.assertNotIn(dangerous_text, artifact_text)
        self.assertNotIn(dangerous_evidence, artifact_text)
        self.assertNotIn(dangerous_text, result_text)
        self.assertNotIn(dangerous_evidence, result_text)
        self.assertIn("reason_summary", artifact_text)
        self.assertNotIn("reason\":", artifact_text)

    def test_render_contains_quality_evaluator_inputs_without_plan_specific_drift(self) -> None:
        case = default_quality_evaluator_benchmark_cases()[0]
        render = build_quality_evaluator_benchmark_prompt_render(case)

        self.assertEqual(
            sorted(render.variables),
            [
                "angle_decision_json",
                "authorial_voice_directive_json",
                "candidate_payload_json",
                "post_brief_json",
                "quality_rubric_json",
                "selected_evidence_json",
            ],
        )
        candidate_payload = json.loads(render.variables["candidate_payload_json"])
        self.assertEqual(set(candidate_payload), {"post_text"})

    def test_benchmark_module_does_not_import_other_role_execution_or_runtime_modules(self) -> None:
        source = Path("services/packaging/linkedin_post_quality_evaluator_benchmark.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

        forbidden_fragments = (
            "generator",
            "publication",
            "candidate_writer_execution",
            "semantic_grounding_execution",
            "repair_writer_execution",
            "final_post_smoke_runner",
        )
        for module_name in imported:
            with self.subTest(module=module_name):
                self.assertFalse(any(fragment in module_name for fragment in forbidden_fragments))


def _quality_review_payload(rationale_text: str | None = None, evidence_text: str | None = None) -> dict:
    scores = {criterion: 4 for criterion in QUALITY_CRITERIA}
    return {
        "scores": scores,
        "total_score": 36,
        "pass": True,
        "failed_criteria": [],
        "automatic_fail_reason": "",
        "notes": [rationale_text or "Readable candidate-specific review."],
        "requires_human_review": False,
        "human_review_reason": "",
        "criterion_rationales": {
            criterion: {
                "score": 4,
                "max_score": 5,
                "rationale": rationale_text or f"{criterion} is strong enough for this frozen post.",
                "post_text_evidence": evidence_text or "The post contains a relevant textual signal.",
                "failure_reason": "",
            }
            for criterion in QUALITY_CRITERIA
        },
    }


def _fixed_now() -> datetime:
    return datetime(2026, 8, 13, 10, 0, tzinfo=UTC)
