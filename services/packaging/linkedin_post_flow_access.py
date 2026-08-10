"""Spec-only access contracts for final LinkedIn post flow agents.

The orchestrator coordinates the final-post flow, but it does not expand payload
permissions and does not allow writer or repair outputs to bypass the
Deterministic Gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


ACCESS_READ_ONLY = "read_only"
ACCESS_CREATE_CANDIDATE = "create_candidate"
ACCESS_WRITE_REVISED_PAYLOAD = "write_revised_payload"
ACCESS_EVALUATE_ONLY = "evaluate_only"
ACCESS_DECISION_ONLY = "decision_only"
ACCESS_PLAN_ONLY = "plan_only"
ACCESS_RECORD_ONLY = "record_only"
ACCESS_ORCHESTRATE_ONLY = "orchestrate_only"

ROLE_CANDIDATE_WRITER = "final_post_candidate_writer"
ROLE_DETERMINISTIC_GATE = "final_post_deterministic_gate"
ROLE_QUALITY_EVALUATOR = "final_post_quality_evaluator"
ROLE_DECISION_CONTROLLER = "final_post_decision_controller"
ROLE_REPAIR_PLANNER = "final_post_repair_planner"
ROLE_REPAIR_AGENT = "final_post_repair_agent"
ROLE_FACTUALITY_REVIEWER = "final_post_factuality_reviewer"
ROLE_ATTEMPT_HISTORY = "final_post_attempt_history"
ROLE_ORCHESTRATOR = "final_post_orchestrator"
ROLE_MODEL_EXPERIMENT_RUNNER = "final_post_model_experiment_runner"

FINAL_POST_PAYLOAD_FIELDS = (
    "post_text",
    "hook_variants",
    "cta_variants",
    "hashtags",
    "quality_checks",
    "carousel_outline",
)

FINAL_POST_REQUIRED_SEQUENCE = (
    ROLE_CANDIDATE_WRITER,
    ROLE_DETERMINISTIC_GATE,
    ROLE_QUALITY_EVALUATOR,
    ROLE_DECISION_CONTROLLER,
)
FINAL_POST_REQUIRED_REPAIR_SEQUENCE = (
    ROLE_REPAIR_PLANNER,
    ROLE_REPAIR_AGENT,
    ROLE_DETERMINISTIC_GATE,
    ROLE_QUALITY_EVALUATOR,
    ROLE_DECISION_CONTROLLER,
)


@dataclass(frozen=True)
class FinalPostAgentAccessContract:
    agent_role: str
    access_mode: str
    allowed_inputs: tuple[str, ...]
    allowed_outputs: tuple[str, ...]
    allowed_payload_actions: tuple[str, ...]
    forbidden_payload_fields: tuple[str, ...]
    may_create_final_post_payload: bool
    may_create_revised_payload: bool
    may_mutate_existing_payload: bool
    may_modify_selected_evidence: bool
    may_modify_post_brief: bool
    may_modify_angle_decision: bool
    next_allowed_handoffs: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_role": self.agent_role,
            "access_mode": self.access_mode,
            "allowed_inputs": list(self.allowed_inputs),
            "allowed_outputs": list(self.allowed_outputs),
            "allowed_payload_actions": list(self.allowed_payload_actions),
            "forbidden_payload_fields": list(self.forbidden_payload_fields),
            "may_create_final_post_payload": self.may_create_final_post_payload,
            "may_create_revised_payload": self.may_create_revised_payload,
            "may_mutate_existing_payload": self.may_mutate_existing_payload,
            "may_modify_selected_evidence": self.may_modify_selected_evidence,
            "may_modify_post_brief": self.may_modify_post_brief,
            "may_modify_angle_decision": self.may_modify_angle_decision,
            "next_allowed_handoffs": list(self.next_allowed_handoffs),
        }


FINAL_POST_AGENT_ACCESS_CONTRACTS = {
    ROLE_CANDIDATE_WRITER: FinalPostAgentAccessContract(
        agent_role=ROLE_CANDIDATE_WRITER,
        access_mode=ACCESS_CREATE_CANDIDATE,
        allowed_inputs=("PostBrief", "AngleDecision", "selected_evidence", "PromptMetadata"),
        allowed_outputs=("CandidatePost", "PostGenerationMetadata"),
        allowed_payload_actions=("create_first_candidate_post",),
        forbidden_payload_fields=(),
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_DETERMINISTIC_GATE,),
    ),
    ROLE_DETERMINISTIC_GATE: FinalPostAgentAccessContract(
        agent_role=ROLE_DETERMINISTIC_GATE,
        access_mode=ACCESS_EVALUATE_ONLY,
        allowed_inputs=("CandidatePost", "FinalPostPayload", "selected_evidence_ids"),
        allowed_outputs=("validation_result", "FinalPostDiagnostics"),
        allowed_payload_actions=("read_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_QUALITY_EVALUATOR, ROLE_DECISION_CONTROLLER),
    ),
    ROLE_QUALITY_EVALUATOR: FinalPostAgentAccessContract(
        agent_role=ROLE_QUALITY_EVALUATOR,
        access_mode=ACCESS_EVALUATE_ONLY,
        allowed_inputs=("PostEditorialInput",),
        allowed_outputs=("QualityReviewResult",),
        allowed_payload_actions=("read_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_DECISION_CONTROLLER,),
    ),
    ROLE_DECISION_CONTROLLER: FinalPostAgentAccessContract(
        agent_role=ROLE_DECISION_CONTROLLER,
        access_mode=ACCESS_DECISION_ONLY,
        allowed_inputs=("FinalPostAttempt", "FinalPostAttemptHistory", "FinalPostDecisionPolicy"),
        allowed_outputs=("FinalPostDecision",),
        allowed_payload_actions=("read_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(
            ROLE_REPAIR_PLANNER,
            ROLE_ATTEMPT_HISTORY,
            ROLE_MODEL_EXPERIMENT_RUNNER,
        ),
    ),
    ROLE_REPAIR_PLANNER: FinalPostAgentAccessContract(
        agent_role=ROLE_REPAIR_PLANNER,
        access_mode=ACCESS_PLAN_ONLY,
        allowed_inputs=("PostEditorialInput", "QualityReviewResult", "FinalPostDecision"),
        allowed_outputs=("TargetedRepairPlan",),
        allowed_payload_actions=("read_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_REPAIR_AGENT,),
    ),
    ROLE_REPAIR_AGENT: FinalPostAgentAccessContract(
        agent_role=ROLE_REPAIR_AGENT,
        access_mode=ACCESS_WRITE_REVISED_PAYLOAD,
        allowed_inputs=(
            "FinalPostPayload",
            "PostBrief",
            "AngleDecision",
            "selected_evidence",
            "FinalPostDiagnostics",
            "QualityReviewResult",
            "TargetedRepairPlan",
        ),
        allowed_outputs=("FinalPostPayload",),
        allowed_payload_actions=("create_revised_payload",),
        forbidden_payload_fields=(),
        may_create_final_post_payload=False,
        may_create_revised_payload=True,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_DETERMINISTIC_GATE,),
    ),
    ROLE_FACTUALITY_REVIEWER: FinalPostAgentAccessContract(
        agent_role=ROLE_FACTUALITY_REVIEWER,
        access_mode=ACCESS_EVALUATE_ONLY,
        allowed_inputs=("PostEditorialInput", "QualityReviewResult"),
        allowed_outputs=("factuality_review_result",),
        allowed_payload_actions=("read_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_DECISION_CONTROLLER,),
    ),
    ROLE_ATTEMPT_HISTORY: FinalPostAgentAccessContract(
        agent_role=ROLE_ATTEMPT_HISTORY,
        access_mode=ACCESS_RECORD_ONLY,
        allowed_inputs=("FinalPostAttempt",),
        allowed_outputs=("FinalPostAttemptHistory",),
        allowed_payload_actions=("record_payload_snapshot",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_ORCHESTRATOR,),
    ),
    ROLE_ORCHESTRATOR: FinalPostAgentAccessContract(
        agent_role=ROLE_ORCHESTRATOR,
        access_mode=ACCESS_ORCHESTRATE_ONLY,
        allowed_inputs=("flow_state",),
        allowed_outputs=("FinalPostFlowResult",),
        allowed_payload_actions=("pass_payload",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_CANDIDATE_WRITER,),
    ),
    ROLE_MODEL_EXPERIMENT_RUNNER: FinalPostAgentAccessContract(
        agent_role=ROLE_MODEL_EXPERIMENT_RUNNER,
        access_mode=ACCESS_ORCHESTRATE_ONLY,
        allowed_inputs=("PostEditorialInput", "model_experiment_config"),
        allowed_outputs=("model_comparison_result",),
        allowed_payload_actions=("compare_payloads",),
        forbidden_payload_fields=FINAL_POST_PAYLOAD_FIELDS,
        may_create_final_post_payload=False,
        may_create_revised_payload=False,
        may_mutate_existing_payload=False,
        may_modify_selected_evidence=False,
        may_modify_post_brief=False,
        may_modify_angle_decision=False,
        next_allowed_handoffs=(ROLE_ATTEMPT_HISTORY, ROLE_DECISION_CONTROLLER),
    ),
}

# Normal successful candidates follow Gate -> Quality Evaluator -> Decision Controller.
# Gate -> Decision Controller exists only as the deterministic failure / mechanical
# repair route; it must not be interpreted as permission for valid payloads to skip
# editorial quality evaluation.
ORCHESTRATED_HANDOFFS = tuple(
    zip(FINAL_POST_REQUIRED_SEQUENCE, FINAL_POST_REQUIRED_SEQUENCE[1:])
) + tuple(
    zip(FINAL_POST_REQUIRED_REPAIR_SEQUENCE, FINAL_POST_REQUIRED_REPAIR_SEQUENCE[1:])
) + (
    (ROLE_DETERMINISTIC_GATE, ROLE_DECISION_CONTROLLER),
    (ROLE_DECISION_CONTROLLER, ROLE_REPAIR_PLANNER),
    (ROLE_DECISION_CONTROLLER, ROLE_ATTEMPT_HISTORY),
    (ROLE_DECISION_CONTROLLER, ROLE_MODEL_EXPERIMENT_RUNNER),
    (ROLE_FACTUALITY_REVIEWER, ROLE_DECISION_CONTROLLER),
    (ROLE_ORCHESTRATOR, ROLE_CANDIDATE_WRITER),
)


def get_access_contract(agent_role: str) -> FinalPostAgentAccessContract:
    return FINAL_POST_AGENT_ACCESS_CONTRACTS[agent_role]


def roles_that_can_create_payload() -> tuple[str, ...]:
    return tuple(
        role
        for role, contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.items()
        if contract.may_create_final_post_payload
    )


def roles_that_can_create_revised_payload() -> tuple[str, ...]:
    return tuple(
        role
        for role, contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.items()
        if contract.may_create_revised_payload
    )


def can_handoff_to(from_role: str, to_role: str) -> bool:
    return to_role in get_access_contract(from_role).next_allowed_handoffs


def is_allowed_handoff(from_role: str, to_role: str) -> bool:
    return (from_role, to_role) in ORCHESTRATED_HANDOFFS


def required_final_post_sequence() -> tuple[str, ...]:
    return FINAL_POST_REQUIRED_SEQUENCE


def required_repair_sequence() -> tuple[str, ...]:
    return FINAL_POST_REQUIRED_REPAIR_SEQUENCE


def orchestrated_handoff_sequence() -> tuple[tuple[str, str], ...]:
    return ORCHESTRATED_HANDOFFS
