from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_handoffs
from services.packaging.linkedin_final_post_diagnostics import diagnose_final_post_payload
from services.packaging.linkedin_post_editorial_boundary import (
    PostEditorialInput,
    PostGenerationMetadata,
    PromptMetadata,
)
from services.packaging.linkedin_post_flow_contracts import (
    ACTION_REPAIR_EDITORIAL,
    FinalPostAttempt,
    FinalPostAttemptHistory,
    FinalPostDecision,
)
from services.packaging.linkedin_post_flow_handoffs import (
    CandidateWriterOutput,
    DeterministicGateOutput,
    QualityReviewResult,
    RepairAgentInput,
    RepairAgentOutput,
    RepairPlannerInput,
    TargetedRepairPlan,
)


@dataclass(frozen=True)
class BriefStub:
    core_point: str
    evidence_to_use: list[dict]


@dataclass(frozen=True)
class AngleStub:
    controlling_angle: str
    supporting_evidence_ids: list[str]


class LinkedInPostFlowHandoffsTests(SimpleTestCase):
    def test_quality_review_result_to_dict_includes_human_voice_score(self) -> None:
        review = _quality_review()

        review_dict = review.to_dict()

        self.assertEqual(review_dict["scores"]["human_voice"], 3)
        self.assertEqual(review_dict["total_score"], 31)
        self.assertFalse(review_dict["pass"])
        self.assertNotIn("passed", review_dict)
        self.assertEqual(review_dict["failed_criteria"], ["human_voice"])
        self.assertEqual(review_dict["automatic_fail_reason"], "")
        self.assertEqual(review_dict["notes"], ["Human voice is too generic."])

    def test_quality_review_result_uses_canonical_nine_criterion_score_keys(self) -> None:
        review_dict = _quality_review().to_dict()

        self.assertEqual(
            set(review_dict["scores"]),
            {
                "hook",
                "controlling_angle",
                "reader_problem",
                "pattern_interrupt",
                "evidence",
                "author_point_of_view",
                "human_voice",
                "practical_value",
                "cta",
            },
        )

    def test_targeted_repair_plan_to_dict_serializes_preserve_and_avoid(self) -> None:
        repair_plan = _repair_plan()

        repair_plan_dict = repair_plan.to_dict()

        self.assertEqual(repair_plan_dict["failed_criterion"], "human_voice")
        self.assertEqual(repair_plan_dict["preserve"], ["selected evidence", "controlling angle"])
        self.assertEqual(repair_plan_dict["avoid"], ["new facts", "revised post text"])
        self.assertEqual(repair_plan_dict["repair_type"], "editorial")

    def test_candidate_writer_output_to_dict_serializes_payload_and_metadata(self) -> None:
        output = _candidate_writer_output()

        output_dict = output.to_dict()

        self.assertEqual(output_dict["payload"]["post_text"], "Final post text.")
        self.assertEqual(output_dict["raw_output"], '{"post_text": "Final post text."}')
        self.assertEqual(output_dict["provider"], "openai")
        self.assertEqual(output_dict["model"], "gpt-4.1-2025-04-14")
        self.assertEqual(output_dict["prompt_name"], "final_post_from_brief")
        self.assertEqual(output_dict["token_usage"]["input_tokens"], 100)
        self.assertEqual(output_dict["cost_metadata"]["estimated_usd"], "0.01")

    def test_deterministic_gate_output_serializes_diagnostics_through_to_dict(self) -> None:
        diagnostics = _diagnostics()
        output = DeterministicGateOutput(
            payload=_payload(),
            validation_passed=True,
            validation_error="",
            diagnostics=diagnostics,
            selected_evidence_ids=("a0-summary", "a1-kp0"),
        )

        output_dict = output.to_dict()

        self.assertEqual(output_dict["diagnostics"], diagnostics.to_dict())
        self.assertEqual(output_dict["selected_evidence_ids"], ["a0-summary", "a1-kp0"])
        self.assertTrue(output_dict["validation_passed"])

    def test_repair_planner_input_to_dict_includes_required_context(self) -> None:
        boundary = _post_editorial_input()
        decision = _decision()
        history = _attempt_history()
        planner_input = RepairPlannerInput(
            post_editorial_input=boundary,
            decision=decision,
            quality_review=_quality_review(),
            attempt_history=history,
        )

        input_dict = planner_input.to_dict()

        self.assertEqual(input_dict["post_editorial_input"], boundary.to_dict())
        self.assertEqual(input_dict["decision"], decision.to_dict())
        self.assertEqual(input_dict["quality_review"], _quality_review().to_dict())
        self.assertEqual(input_dict["attempt_history"], history.to_dict())

    def test_repair_agent_input_to_dict_requires_targeted_repair_plan_and_attempt_metadata(self) -> None:
        repair_input = RepairAgentInput(
            post_editorial_input=_post_editorial_input(),
            repair_plan=_repair_plan(),
            attempt_index=1,
            max_attempts=3,
        )

        input_dict = repair_input.to_dict()

        self.assertEqual(input_dict["repair_plan"], _repair_plan().to_dict())
        self.assertEqual(input_dict["repair_plan"]["repair_type"], "editorial")
        self.assertEqual(input_dict["attempt_index"], 1)
        self.assertEqual(input_dict["max_attempts"], 3)
        self.assertIn("post_editorial_input", input_dict)

    def test_repair_plan_does_not_include_revised_post_text_or_acceptance_decision(self) -> None:
        repair_plan_dict = _repair_plan().to_dict()

        self.assertNotIn("payload", repair_plan_dict)
        self.assertNotIn("post_text", repair_plan_dict)
        self.assertNotIn("accepted_payload", repair_plan_dict)
        self.assertNotIn("decision", repair_plan_dict)

    def test_repair_agent_output_to_dict_serializes_revised_payload_and_parent_attempt(self) -> None:
        output = RepairAgentOutput(
            payload=_payload(post_text="Revised final post text."),
            raw_output='{"post_text": "Revised final post text."}',
            provider="openai",
            model="gpt-4.1-2025-04-14",
            prompt_name="final_post_repair",
            prompt_version="1.0",
            token_usage={"input_tokens": 120, "output_tokens": 60},
            cost_metadata={"estimated_usd": "0.02"},
            parent_attempt_index=0,
        )

        output_dict = output.to_dict()

        self.assertEqual(output_dict["payload"]["post_text"], "Revised final post text.")
        self.assertEqual(output_dict["parent_attempt_index"], 0)
        self.assertEqual(output_dict["prompt_name"], "final_post_repair")

    def test_handoff_objects_are_json_serializable(self) -> None:
        objects = [
            _candidate_writer_output().to_dict(),
            DeterministicGateOutput(
                payload=_payload(),
                validation_passed=True,
                validation_error="",
                diagnostics=_diagnostics(),
                selected_evidence_ids=("a0-summary",),
            ).to_dict(),
            RepairPlannerInput(
                post_editorial_input=_post_editorial_input(),
                decision=_decision(),
                quality_review=_quality_review(),
                attempt_history=_attempt_history(),
            ).to_dict(),
            RepairAgentInput(
                post_editorial_input=_post_editorial_input(),
                repair_plan=_repair_plan(repair_type="mechanical"),
                attempt_index=1,
                max_attempts=3,
            ).to_dict(),
            RepairAgentOutput(
                payload=_payload(post_text="Revised final post text."),
                raw_output=None,
                provider=None,
                model=None,
                prompt_name=None,
                prompt_version=None,
                token_usage=None,
                cost_metadata=None,
                parent_attempt_index=0,
            ).to_dict(),
        ]

        serialized = json.dumps(objects, sort_keys=True)

        self.assertIn("Final post text.", serialized)
        self.assertIn("selected_evidence_ids", serialized)

    def test_candidate_output_represents_new_payload_not_mutation(self) -> None:
        source = inspect.getsource(CandidateWriterOutput)

        self.assertIn("not a mutation of an existing payload", source)
        self.assertIn("FinalPostDeterministicGate", source)

    def test_repair_output_represents_new_payload_not_mutation(self) -> None:
        source = inspect.getsource(RepairAgentOutput)

        self.assertIn("not an in-place mutation", source)
        self.assertIn("FinalPostDeterministicGate", source)
        self.assertIn("does not include an acceptance", source)

    def test_post_editorial_input_is_documented_as_current_perspective_boundary(self) -> None:
        source = inspect.getsource(linkedin_post_flow_handoffs)

        self.assertIn("PostEditorialInput is the current perspective boundary", source)
        self.assertIn("author_take.core_opinion", source)
        self.assertIn("separate architecture step", source)

    def test_payload_dict_snapshot_limitation_is_documented(self) -> None:
        source = inspect.getsource(linkedin_post_flow_handoffs)

        self.assertIn("Payload snapshots are represented as dictionaries", source)
        self.assertIn("Stricter", source)

    def test_handoff_module_does_not_call_api_execute_prompts_or_orchestrate(self) -> None:
        source = inspect.getsource(linkedin_post_flow_handoffs)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("orchestrate", source.lower())


def _candidate_writer_output() -> CandidateWriterOutput:
    return CandidateWriterOutput(
        payload=_payload(),
        raw_output='{"post_text": "Final post text."}',
        provider="openai",
        model="gpt-4.1-2025-04-14",
        prompt_name="final_post_from_brief",
        prompt_version="1.0",
        token_usage={"input_tokens": 100, "output_tokens": 50},
        cost_metadata={"estimated_usd": "0.01"},
    )


def _post_editorial_input() -> PostEditorialInput:
    diagnostics = _diagnostics()
    return PostEditorialInput(
        candidate_payload=_payload(),
        post_brief=BriefStub(
            core_point="Use selected evidence.",
            evidence_to_use=[
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Remote work requires clearer policy.",
                }
            ],
        ),
        angle_decision=AngleStub(
            controlling_angle="Clear angle",
            supporting_evidence_ids=["a0-summary"],
        ),
        selected_evidence=[
            {
                "evidence_id": "a0-summary",
                "evidence_text": "Remote work requires clearer policy.",
            }
        ],
        final_payload_validation_passed=True,
        final_payload_validation_error="",
        diagnostics=diagnostics,
        repair_reasons=list(diagnostics.repair_reasons),
        generation_metadata=PostGenerationMetadata(
            provider="openai",
            model="gpt-4.1-2025-04-14",
            run_id="run-1",
            created_at="2026-07-05T10:00:00Z",
            token_usage=None,
            cost_metadata=None,
        ),
        prompt_metadata=PromptMetadata(
            prompt_name="final_post_from_brief",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
    )


def _attempt_history() -> FinalPostAttemptHistory:
    return FinalPostAttemptHistory(
        attempts=[
            FinalPostAttempt(
                attempt_index=0,
                payload=_payload(),
                validation_passed=True,
                validation_error="",
                diagnostics=_diagnostics(),
                quality_review=_quality_review(),
                repair_plan=None,
                decision=_decision(),
                provider="openai",
                model="gpt-4.1-2025-04-14",
                prompt_name="final_post_from_brief",
                prompt_version="1.0",
                token_usage=None,
                cost_metadata=None,
                created_at="2026-07-05T10:00:00Z",
                parent_attempt_index=None,
            )
        ]
    )


def _decision() -> FinalPostDecision:
    return FinalPostDecision(
        action=ACTION_REPAIR_EDITORIAL,
        reason="human voice below minimum",
        repair_type="editorial",
        target_model_provider=None,
        target_model_name=None,
        needs_human_review=False,
    )


def _diagnostics():
    return diagnose_final_post_payload(
        _payload(),
        selected_evidence_ids=["a0-summary"],
        schema_validation_passed=True,
    )


def _quality_review() -> QualityReviewResult:
    return QualityReviewResult(
        scores={
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 3,
            "pattern_interrupt": 3,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 3,
            "practical_value": 3,
            "cta": 3,
        },
        total_score=31,
        passed=False,
        failed_criteria=("human_voice",),
        automatic_fail_reason="",
        notes=("Human voice is too generic.",),
    )


def _repair_plan(*, repair_type: str | None = "editorial") -> TargetedRepairPlan:
    return TargetedRepairPlan(
        failed_criterion="human_voice",
        repair_scope="human-facing text",
        repair_instruction="Make the post sound more like a person with a clear point.",
        preserve=("selected evidence", "controlling angle"),
        avoid=("new facts", "revised post text"),
        repair_type=repair_type,
    )


def _payload(**overrides) -> dict:
    payload = {
        "post_text": "Final post text.",
        "hook_variants": ["Hook one", "Hook two", "Hook three"],
        "cta_variants": ["CTA one", "CTA two", "CTA three"],
        "hashtags": ["#FutureOfWork"],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
        "carousel_outline": [],
    }
    payload.update(overrides)
    return payload
