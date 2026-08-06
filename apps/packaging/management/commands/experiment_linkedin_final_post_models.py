from __future__ import annotations

import json
from pathlib import Path
import sys

from django.core.management.base import BaseCommand, CommandError

from services.packaging.linkedin_post_model_experiment_harness import (
    SMOKE_MODE_STANDALONE,
    FinalPostExperimentCase,
    FinalPostExperimentPlan,
    FinalPostExperimentRoleModel,
    FinalPostModelExperimentRequest,
    default_final_post_experiment_plans,
    run_linkedin_final_post_model_experiment,
)


class Command(BaseCommand):
    help = "Run a local PostFlow final-post model experiment. Dry-run by default."
    requires_system_checks: list[str] = []

    def add_arguments(self, parser):
        parser.add_argument("--experiment-id", required=True)
        parser.add_argument(
            "--case",
            action="append",
            default=[],
            help="Case mapping in the form case_id=path. May be repeated.",
        )
        parser.add_argument(
            "--plan",
            action="append",
            default=[],
            help=(
                "Optional plan in the form "
                "plan_id=mode,candidate_provider,candidate_model,"
                "grounding_provider,grounding_model,evaluator_provider,evaluator_model"
                "[,repair_provider,repair_model]. May be repeated."
            ),
        )
        parser.add_argument("--runs-per-plan", type=int, default=1)
        parser.add_argument("--output-root")
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Keep provider calls disabled. This is the default.",
        )
        parser.add_argument(
            "--allow-api",
            action="store_true",
            help="Explicitly allow provider calls for this manual experiment.",
        )

    def handle(self, *args, **options):
        if options["dry_run"] and options["allow_api"]:
            raise CommandError("--dry-run and --allow-api cannot be used together")
        cases = tuple(_parse_case(value) for value in options["case"])
        if not cases:
            raise CommandError("at least one --case is required")
        plans = tuple(_parse_plan(value) for value in options["plan"])
        if not plans:
            plans = default_final_post_experiment_plans()

        request = FinalPostModelExperimentRequest(
            experiment_id=options["experiment_id"],
            cases=cases,
            plans=plans,
            runs_per_plan=options["runs_per_plan"],
            allow_api=bool(options["allow_api"]),
            output_root=Path(options["output_root"]) if options.get("output_root") else None,
        )
        result = run_linkedin_final_post_model_experiment(request)
        self.stdout.write("=== POSTFLOW FINAL POST MODEL EXPERIMENT ===")
        self.stdout.write(f"experiment_id: {result.experiment_id}")
        self.stdout.write(f"status: {result.status}")
        self.stdout.write(f"exit_code: {result.exit_code}")
        self.stdout.write(f"run_count: {result.run_count}")
        self.stdout.write(f"dry_run: {not request.allow_api}")
        self.stdout.write(f"allow_api: {request.allow_api}")
        self.stdout.write(f"production_runtime_wiring: {False}")
        if result.safe_failure_code:
            self.stdout.write(f"safe_failure_code: {result.safe_failure_code}")
        if result.safe_failure_message:
            self.stdout.write(f"safe_failure_message: {result.safe_failure_message}")
        if result.artifacts:
            self.stdout.write("")
            self.stdout.write("=== ARTIFACTS ===")
            self.stdout.write(
                json.dumps(
                    result.artifacts.to_dict(),
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
            )
        self.stdout.write("")
        self.stdout.write("=== SUMMARY ===")
        accepted_count = sum(1 for record in result.run_records if record.get("accepted") is True)
        self.stdout.write(f"accepted_runs: {accepted_count}")
        self.stdout.write(f"failed_runs: {result.run_count - accepted_count}")
        if result.exit_code:
            sys.exit(result.exit_code)


def _parse_case(value: str) -> FinalPostExperimentCase:
    if "=" not in value:
        raise CommandError("--case must use case_id=path")
    case_id, path = value.split("=", 1)
    if not case_id.strip() or not path.strip():
        raise CommandError("--case must include both case_id and path")
    return FinalPostExperimentCase(case_id=case_id.strip(), input_path=Path(path.strip()))


def _parse_plan(value: str) -> FinalPostExperimentPlan:
    if "=" not in value:
        raise CommandError("--plan must use plan_id=comma-separated-fields")
    plan_id, raw_fields = value.split("=", 1)
    fields = [field.strip() for field in raw_fields.split(",")]
    if len(fields) not in (7, 9):
        raise CommandError(
            "--plan must include 7 fields, or 9 fields when repair_writer is present"
        )
    mode = fields[0] or SMOKE_MODE_STANDALONE
    repair_writer = None
    if len(fields) == 9:
        repair_writer = FinalPostExperimentRoleModel(fields[7], fields[8])
    return FinalPostExperimentPlan(
        plan_id=plan_id.strip(),
        mode=mode,
        candidate_writer=FinalPostExperimentRoleModel(fields[1], fields[2]),
        semantic_grounding=FinalPostExperimentRoleModel(fields[3], fields[4]),
        quality_evaluator=FinalPostExperimentRoleModel(fields[5], fields[6]),
        repair_writer=repair_writer,
    )
