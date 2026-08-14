from __future__ import annotations

import io
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import SimpleTestCase


class BenchmarkLinkedInRepairWriterCommandTests(SimpleTestCase):
    def test_dry_run_command_outputs_zero_provider_calls(self) -> None:
        with TemporaryDirectory() as tempdir:
            stdout = io.StringIO()

            call_command(
                "benchmark_linkedin_repair_writer",
                "--dry-run",
                "--output-root",
                tempdir,
                stdout=stdout,
            )

        output = stdout.getvalue()
        self.assertIn("=== PostFlow REPAIR WRITER BENCHMARK ===", output)
        self.assertIn("status: dry_run", output)
        self.assertIn("provider_calls: 0", output)
        self.assertIn("candidate_writer_invocations: 0", output)
        self.assertIn("publication_packaging_invocations: 0", output)

    def test_command_rejects_dry_run_with_allow_api(self) -> None:
        with TemporaryDirectory() as tempdir:
            with self.assertRaisesRegex(CommandError, "--dry-run and --allow-api"):
                call_command(
                    "benchmark_linkedin_repair_writer",
                    "--dry-run",
                    "--allow-api",
                    "--output-root",
                    tempdir,
                )

    def test_command_accepts_explicit_fixture_and_plan_without_provider_calls(self) -> None:
        fixture = (
            "tests/fixtures/linkedin_post_repair_writer_benchmark/"
            "topic_214_digest_128__claude_v3.json"
        )
        with TemporaryDirectory() as tempdir:
            stdout = io.StringIO()

            call_command(
                "benchmark_linkedin_repair_writer",
                "--dry-run",
                "--case",
                fixture,
                "--plan",
                "claude_repair=anthropic,claude-sonnet-5,1800",
                "--output-root",
                tempdir,
                stdout=stdout,
            )

        output = stdout.getvalue()
        self.assertIn("run_count: 1", output)
        self.assertIn("claude_repair", output)

    def test_command_dry_run_does_not_call_repair_writer_execution(self) -> None:
        with TemporaryDirectory() as tempdir:
            with patch(
                "services.packaging.linkedin_post_repair_writer_benchmark.execute_repair_writer_prompt",
                side_effect=AssertionError("provider execution should not run"),
            ):
                call_command(
                    "benchmark_linkedin_repair_writer",
                    "--dry-run",
                    "--output-root",
                    tempdir,
                    stdout=io.StringIO(),
                )

    def test_command_writes_artifacts_under_requested_output_root(self) -> None:
        with TemporaryDirectory() as tempdir:
            call_command(
                "benchmark_linkedin_repair_writer",
                "--dry-run",
                "--output-root",
                tempdir,
                stdout=io.StringIO(),
            )

            output_dir = Path(tempdir) / "repair-writer-gpt-vs-claude-vs-gemini-v1"

            self.assertTrue((output_dir / "manifest.json").exists())
            self.assertTrue((output_dir / "runs.jsonl").exists())
            self.assertTrue((output_dir / "summary.csv").exists())
