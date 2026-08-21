from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path
import tempfile

from django.test import SimpleTestCase

from services.packaging import postflow_product_validation_corpus as corpus
from services.packaging import postflow_product_validation_runner as runner


class PostFlowProductValidationBenchmarkTests(SimpleTestCase):
    def test_manifest_loads_exactly_27_cases_and_frozen_anchors(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()

        self.assertEqual(len(manifest.cases), 27)
        self.assertEqual(manifest.target_case_count, 27)
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
        self.assertEqual(families.count(corpus.CASE_FAMILY_CANDIDATE_QE), 6)
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
            self.assertEqual(result.run_count, 27)
            self.assertEqual(result.corpus_case_count, 27)
            artifacts = result.artifacts.to_dict()
            for path in artifacts.values():
                self.assertTrue(Path(path).exists(), path)

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
        self.assertGreater(result.metrics["product_failure_case_count"], 0)
        self.assertEqual(result.metrics["infrastructure_failure_case_count"], 0)
        self.assertEqual(result.metrics["human_review_case_count"], 4)

    def test_configuration_fingerprint_contains_frozen_roles_and_hashes(self) -> None:
        manifest = corpus.load_product_validation_corpus_manifest()
        fingerprint = runner.build_configuration_fingerprint(manifest, now=_fixed_now())

        roles = fingerprint["frozen_model_role_configuration"]
        self.assertEqual(roles["candidate_writer"]["provider"], "anthropic")
        self.assertEqual(roles["semantic_grounding"]["provider"], "gemini")
        self.assertEqual(roles["quality_evaluator"]["provider"], "openai")
        self.assertEqual(roles["repair_writer"]["provider"], "gemini")
        self.assertEqual(fingerprint["provider_call_count"], 0)
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
            "expected_outcome",
            "reviewer_label",
            "backlog_action",
        ):
            self.assertIn(column, header)

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
                "provider_payload",
                "provider_reply",
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
            self.assertNotIn("execute_semantic_grounding_prompt", text)
            self.assertNotIn("execute_quality_evaluator_prompt", text)
            self.assertNotIn("execute_repair_writer_prompt", text)
            self.assertNotIn("OpenAIClient", text)
            self.assertNotIn("Gemini", text)
            self.assertNotIn("Anthropic", text)

    def test_content_package_payload_contract_is_not_modified_by_scope(self) -> None:
        paths = [path.as_posix() for path in Path(".").glob("**/*") if path.is_file()]
        self.assertIn("services/packaging/postflow_product_validation_runner.py", paths)
        self.assertTrue(Path("services/packaging/linkedin_post_final_post_payload_contract.py").exists())


def _fixed_now() -> datetime:
    return datetime(2026, 8, 21, 12, 0, tzinfo=UTC)
