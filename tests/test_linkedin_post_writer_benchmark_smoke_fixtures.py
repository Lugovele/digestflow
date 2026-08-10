from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import tempfile

from django.test import SimpleTestCase

from services.packaging.linkedin_post_final_post_smoke_runner import (
    EXIT_OK,
    SMOKE_STATUS_CONFIG_ERROR,
    SMOKE_STATUS_DRY_RUN,
    FinalPostSmokeRunRequest,
    run_final_post_smoke,
)
from services.packaging.linkedin_post_model_experiment_harness import (
    EXPERIMENT_STATUS_COMPLETED,
    FinalPostExperimentCase,
    FinalPostModelExperimentRequest,
    default_final_post_experiment_plans,
    run_linkedin_final_post_model_experiment,
)
from services.packaging.linkedin_post_pipeline import (
    PipelineInput,
    SelectedArticle,
    build_angle_decision_from_contextual_evidence_pack,
    build_article_evidence_pack_from_pipeline_input,
    build_contextual_evidence_pack_from_article_evidence_pack,
    build_post_brief_from_angle_decision,
    validate_linkedin_post_stage_relationships,
    validate_pipeline_input,
)


RAW_FIXTURE_DIR = Path("tests/fixtures/linkedin_post_cases")
CANONICAL_FIXTURE_DIR = Path("tests/fixtures/linkedin_post_smoke/writer_benchmarks")
BENCHMARK_CASE_IDS = (
    "topic_200_digest_134",
    "topic_214_digest_128",
    "topic_140_digest_126",
)
FORBIDDEN_RUNTIME_FIELDS = {
    "articles",
    "content_package",
    "api_key",
    "openai_api_key",
    "client",
    "headers",
}


class LinkedInPostWriterBenchmarkSmokeFixtureTests(SimpleTestCase):
    def test_raw_benchmark_cases_are_rejected_as_runtime_fixtures(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                result = run_final_post_smoke(
                    FinalPostSmokeRunRequest(input_path=RAW_FIXTURE_DIR / f"{case_id}.json")
                )

                self.assertEqual(result.status, SMOKE_STATUS_CONFIG_ERROR)
                self.assertIn("forbidden runtime fields", result.safe_failure_message)
                self.assertEqual(result.invocation_counts["candidate_writer"], 0)
                self.assertEqual(result.invocation_counts["semantic_grounding"], 0)
                self.assertEqual(result.invocation_counts["quality_evaluator"], 0)

    def test_canonical_benchmark_fixtures_match_deterministic_staged_conversion(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                raw_fixture = _load_json(RAW_FIXTURE_DIR / f"{case_id}.json")
                canonical_fixture = _load_json(CANONICAL_FIXTURE_DIR / f"{case_id}.json")
                expected = _canonical_payload_from_raw_fixture(case_id, raw_fixture)

                self.assertEqual(canonical_fixture, expected)

    def test_canonical_benchmark_fixtures_contain_no_forbidden_runtime_fields(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                canonical_fixture = _load_json(CANONICAL_FIXTURE_DIR / f"{case_id}.json")

                self.assertFalse(FORBIDDEN_RUNTIME_FIELDS.intersection(canonical_fixture))

    def test_canonical_benchmark_fixtures_preserve_selected_evidence_identity(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                raw_fixture = _load_json(RAW_FIXTURE_DIR / f"{case_id}.json")
                canonical_fixture = _load_json(CANONICAL_FIXTURE_DIR / f"{case_id}.json")
                expected = _canonical_payload_from_raw_fixture(case_id, raw_fixture)

                canonical_evidence = canonical_fixture["post_brief"]["evidence_to_use"]
                expected_evidence = expected["post_brief"]["evidence_to_use"]
                self.assertEqual(
                    [item["evidence_id"] for item in canonical_evidence],
                    canonical_fixture["angle_decision"]["supporting_evidence_ids"],
                )
                self.assertEqual(
                    [item["evidence_id"] for item in canonical_evidence],
                    [item["evidence_id"] for item in expected_evidence],
                )
                self.assertEqual(
                    [item["evidence_text"] for item in canonical_evidence],
                    [item["evidence_text"] for item in expected_evidence],
                )

    def test_canonical_benchmark_fixtures_preserve_editorial_identity_fields(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                fixture = _load_json(CANONICAL_FIXTURE_DIR / f"{case_id}.json")
                angle_decision = fixture["angle_decision"]
                post_brief = fixture["post_brief"]
                directive = angle_decision["authorial_voice_directive"]

                for value in (
                    angle_decision["controlling_angle"],
                    angle_decision["reader_problem"],
                    angle_decision["author_position"],
                    directive["authorial_observation"],
                    directive["rejected_reading"],
                    directive["why_distinction_matters"],
                    post_brief["opening_direction"],
                    post_brief["core_point"],
                    post_brief["practical_point"],
                ):
                    self.assertIsInstance(value, str)
                    self.assertTrue(value.strip())

    def test_canonical_benchmark_fixtures_dry_run_through_smoke_without_provider_calls(self) -> None:
        for case_id in BENCHMARK_CASE_IDS:
            with self.subTest(case_id=case_id):
                result = run_final_post_smoke(
                    FinalPostSmokeRunRequest(
                        input_path=CANONICAL_FIXTURE_DIR / f"{case_id}.json"
                    )
                )

                self.assertEqual(result.status, SMOKE_STATUS_DRY_RUN)
                self.assertEqual(result.exit_code, EXIT_OK)
                self.assertEqual(
                    result.sanitized_result["selected_evidence_ids"],
                    [
                        item["evidence_id"]
                        for item in _load_json(
                            CANONICAL_FIXTURE_DIR / f"{case_id}.json"
                        )["post_brief"]["evidence_to_use"]
                    ],
                )
                self.assertEqual(
                    result.invocation_counts,
                    {
                        "candidate_writer": 0,
                        "semantic_grounding": 0,
                        "quality_evaluator": 0,
                        "repair_writer": 0,
                    },
                )

    def test_writer_benchmark_experiment_dry_run_generates_six_artifact_records(self) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            cases = tuple(
                FinalPostExperimentCase(
                    case_id=case_id,
                    input_path=CANONICAL_FIXTURE_DIR / f"{case_id}.json",
                )
                for case_id in BENCHMARK_CASE_IDS
            )
            result = run_linkedin_final_post_model_experiment(
                FinalPostModelExperimentRequest(
                    experiment_id="writer_benchmark_dry_contract",
                    cases=cases,
                    plans=default_final_post_experiment_plans(),
                    runs_per_plan=1,
                    allow_api=False,
                    output_root=Path(tempdir),
                )
            )

            self.assertEqual(result.status, EXPERIMENT_STATUS_COMPLETED)
            self.assertEqual(result.exit_code, EXIT_OK)
            self.assertEqual(result.run_count, 6)
            artifacts = result.artifacts
            self.assertTrue(Path(artifacts.runs_jsonl).exists())
            self.assertTrue(Path(artifacts.summary_csv).exists())
            self.assertTrue(Path(artifacts.report_md).exists())
            self.assertTrue(Path(artifacts.writer_comparison_md).exists())
            self.assertTrue(Path(artifacts.manifest_json).exists())

            run_records = [
                json.loads(line)
                for line in Path(artifacts.runs_jsonl).read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(run_records), 6)
            self.assertEqual(
                {
                    (record["case_id"], record["plan_id"])
                    for record in run_records
                },
                {
                    (case_id, plan.plan_id)
                    for case_id in BENCHMARK_CASE_IDS
                    for plan in default_final_post_experiment_plans()
                },
            )
            self.assertTrue(all(record["status"] == SMOKE_STATUS_DRY_RUN for record in run_records))
            self.assertTrue(
                all(
                    sum(record["provider_invocation_counts"].values()) == 0
                    for record in run_records
                )
            )
            manifest = _load_json(Path(artifacts.manifest_json))
            self.assertIn("git_commit", manifest)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_payload_from_raw_fixture(case_id: str, raw_fixture: dict) -> dict:
    chain = _staged_chain_from_raw_fixture(raw_fixture)
    return _json_safe(
        {
            "case_id": raw_fixture["case_id"],
            "source_fixture": str(RAW_FIXTURE_DIR / f"{case_id}.json").replace("\\", "/"),
            "source_type": raw_fixture["source_type"],
            "topic": raw_fixture["topic"],
            "digest": raw_fixture["digest"],
            "benchmark_expected_behavior": raw_fixture["expected_behavior"],
            "post_brief": asdict(chain["post_brief"]),
            "angle_decision": asdict(chain["angle_decision"]),
        }
    )


def _staged_chain_from_raw_fixture(raw_fixture: dict) -> dict:
    pipeline_input = _pipeline_input_from_raw_fixture(raw_fixture)
    article_evidence_pack = build_article_evidence_pack_from_pipeline_input(pipeline_input)
    contextual_evidence_pack = build_contextual_evidence_pack_from_article_evidence_pack(
        article_evidence_pack
    )
    angle_decision = build_angle_decision_from_contextual_evidence_pack(
        contextual_evidence_pack
    )
    post_brief = build_post_brief_from_angle_decision(
        contextual_evidence_pack,
        angle_decision,
    )
    validate_linkedin_post_stage_relationships(
        pipeline_input,
        article_evidence_pack,
        contextual_evidence_pack,
        angle_decision,
        post_brief,
    )
    return {"angle_decision": angle_decision, "post_brief": post_brief}


def _pipeline_input_from_raw_fixture(raw_fixture: dict) -> PipelineInput:
    articles = [
        SelectedArticle(
            source_index=article["source_index"],
            title=article["title"],
            url=article["url"],
            summary=article["summary"],
            key_points=article.get("key_points", []),
            source_name=str(article.get("source_name") or "").strip(),
            published_at=str(article.get("published_at") or "").strip(),
            content_type=str(article.get("content_type") or "").strip(),
            confidence=article.get("confidence"),
        )
        for article in raw_fixture["articles"]
    ]
    pipeline_input = PipelineInput(
        digest_id=raw_fixture["digest"]["id"],
        topic_name=raw_fixture["topic"]["name"],
        digest_title=raw_fixture["digest"]["title"],
        articles=articles,
        author_profile={},
    )
    validate_pipeline_input(pipeline_input)
    return pipeline_input


def _json_safe(value):
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))
