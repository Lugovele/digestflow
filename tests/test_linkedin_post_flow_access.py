from __future__ import annotations

import inspect

from django.test import SimpleTestCase

from services.packaging import linkedin_post_flow_access
from services.packaging.linkedin_post_flow_access import (
    FINAL_POST_AGENT_ACCESS_CONTRACTS,
    FINAL_POST_PAYLOAD_FIELDS,
    ROLE_ATTEMPT_HISTORY,
    ROLE_CANDIDATE_WRITER,
    ROLE_DECISION_CONTROLLER,
    ROLE_DETERMINISTIC_GATE,
    ROLE_FACTUALITY_REVIEWER,
    ROLE_MODEL_EXPERIMENT_RUNNER,
    ROLE_ORCHESTRATOR,
    ROLE_QUALITY_EVALUATOR,
    ROLE_REPAIR_AGENT,
    ROLE_REPAIR_PLANNER,
    can_handoff_to,
    get_access_contract,
    is_allowed_handoff,
    orchestrated_handoff_sequence,
    required_final_post_sequence,
    required_repair_sequence,
    roles_that_can_create_payload,
    roles_that_can_create_revised_payload,
)


class LinkedInPostFlowAccessTests(SimpleTestCase):
    def test_no_role_creates_first_final_post_payload_in_core_path(self) -> None:
        self.assertEqual(roles_that_can_create_payload(), ())

    def test_only_repair_agent_can_create_revised_candidate_post(self) -> None:
        self.assertEqual(roles_that_can_create_revised_payload(), (ROLE_REPAIR_AGENT,))

    def test_no_role_may_mutate_existing_payload_in_place(self) -> None:
        for contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.values():
            self.assertFalse(contract.may_mutate_existing_payload, contract.agent_role)

    def test_no_role_may_modify_selected_evidence(self) -> None:
        for contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.values():
            self.assertFalse(contract.may_modify_selected_evidence, contract.agent_role)

    def test_no_role_may_modify_post_brief(self) -> None:
        for contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.values():
            self.assertFalse(contract.may_modify_post_brief, contract.agent_role)

    def test_no_role_may_modify_angle_decision(self) -> None:
        for contract in FINAL_POST_AGENT_ACCESS_CONTRACTS.values():
            self.assertFalse(contract.may_modify_angle_decision, contract.agent_role)

    def test_non_writer_roles_cannot_write_payload_fields(self) -> None:
        read_only_roles = (
            ROLE_DETERMINISTIC_GATE,
            ROLE_QUALITY_EVALUATOR,
            ROLE_DECISION_CONTROLLER,
            ROLE_REPAIR_PLANNER,
            ROLE_FACTUALITY_REVIEWER,
            ROLE_ORCHESTRATOR,
            ROLE_MODEL_EXPERIMENT_RUNNER,
        )

        for role in read_only_roles:
            contract = get_access_contract(role)
            self.assertEqual(contract.forbidden_payload_fields, FINAL_POST_PAYLOAD_FIELDS)
            self.assertFalse(contract.may_create_final_post_payload)
            self.assertFalse(contract.may_create_revised_payload)

    def test_repair_agent_must_handoff_to_deterministic_gate(self) -> None:
        self.assertTrue(can_handoff_to(ROLE_REPAIR_AGENT, ROLE_DETERMINISTIC_GATE))
        self.assertTrue(is_allowed_handoff(ROLE_REPAIR_AGENT, ROLE_DETERMINISTIC_GATE))

    def test_repair_agent_access_contract_uses_candidate_post_boundary(self) -> None:
        contract = get_access_contract(ROLE_REPAIR_AGENT)

        self.assertIn("CandidatePost", contract.allowed_inputs)
        self.assertNotIn("FinalPostPayload", contract.allowed_inputs)
        self.assertEqual(contract.allowed_outputs, ("CandidatePost",))

    def test_candidate_writer_must_handoff_to_deterministic_gate(self) -> None:
        self.assertTrue(can_handoff_to(ROLE_CANDIDATE_WRITER, ROLE_DETERMINISTIC_GATE))
        self.assertTrue(is_allowed_handoff(ROLE_CANDIDATE_WRITER, ROLE_DETERMINISTIC_GATE))

    def test_required_final_post_sequence_is_explicit(self) -> None:
        self.assertEqual(
            required_final_post_sequence(),
            (
                ROLE_CANDIDATE_WRITER,
                ROLE_DETERMINISTIC_GATE,
                ROLE_QUALITY_EVALUATOR,
                ROLE_DECISION_CONTROLLER,
            ),
        )

    def test_required_repair_sequence_is_explicit(self) -> None:
        self.assertEqual(
            required_repair_sequence(),
            (
                ROLE_REPAIR_PLANNER,
                ROLE_REPAIR_AGENT,
                ROLE_DETERMINISTIC_GATE,
                ROLE_QUALITY_EVALUATOR,
                ROLE_DECISION_CONTROLLER,
            ),
        )

    def test_required_sequence_handoffs_are_allowed(self) -> None:
        expected_handoffs = (
            (ROLE_DETERMINISTIC_GATE, ROLE_QUALITY_EVALUATOR),
            (ROLE_QUALITY_EVALUATOR, ROLE_DECISION_CONTROLLER),
            (ROLE_DECISION_CONTROLLER, ROLE_REPAIR_PLANNER),
            (ROLE_REPAIR_PLANNER, ROLE_REPAIR_AGENT),
        )

        for from_role, to_role in expected_handoffs:
            self.assertTrue(can_handoff_to(from_role, to_role), (from_role, to_role))
            self.assertTrue(is_allowed_handoff(from_role, to_role), (from_role, to_role))

    def test_gate_to_decision_controller_is_fail_fast_repair_route(self) -> None:
        self.assertTrue(can_handoff_to(ROLE_DETERMINISTIC_GATE, ROLE_DECISION_CONTROLLER))
        self.assertTrue(is_allowed_handoff(ROLE_DETERMINISTIC_GATE, ROLE_DECISION_CONTROLLER))
        self.assertTrue(can_handoff_to(ROLE_DETERMINISTIC_GATE, ROLE_QUALITY_EVALUATOR))
        self.assertTrue(is_allowed_handoff(ROLE_DETERMINISTIC_GATE, ROLE_QUALITY_EVALUATOR))

    def test_quality_evaluator_cannot_handoff_directly_to_repair_agent(self) -> None:
        self.assertFalse(can_handoff_to(ROLE_QUALITY_EVALUATOR, ROLE_REPAIR_AGENT))
        self.assertFalse(is_allowed_handoff(ROLE_QUALITY_EVALUATOR, ROLE_REPAIR_AGENT))
        self.assertTrue(can_handoff_to(ROLE_QUALITY_EVALUATOR, ROLE_DECISION_CONTROLLER))

    def test_candidate_writer_cannot_bypass_deterministic_gate(self) -> None:
        forbidden_targets = (
            ROLE_QUALITY_EVALUATOR,
            ROLE_DECISION_CONTROLLER,
            ROLE_ATTEMPT_HISTORY,
        )

        for target in forbidden_targets:
            self.assertFalse(can_handoff_to(ROLE_CANDIDATE_WRITER, target), target)
            self.assertFalse(is_allowed_handoff(ROLE_CANDIDATE_WRITER, target), target)

    def test_repair_agent_cannot_bypass_deterministic_gate(self) -> None:
        forbidden_targets = (
            ROLE_QUALITY_EVALUATOR,
            ROLE_DECISION_CONTROLLER,
            ROLE_ATTEMPT_HISTORY,
        )

        for target in forbidden_targets:
            self.assertFalse(can_handoff_to(ROLE_REPAIR_AGENT, target), target)
            self.assertFalse(is_allowed_handoff(ROLE_REPAIR_AGENT, target), target)

    def test_orchestrator_is_not_wildcard_bypass_permission(self) -> None:
        forbidden_targets = (
            ROLE_QUALITY_EVALUATOR,
            ROLE_DECISION_CONTROLLER,
            ROLE_REPAIR_AGENT,
            ROLE_ATTEMPT_HISTORY,
        )

        for target in forbidden_targets:
            self.assertFalse(can_handoff_to(ROLE_ORCHESTRATOR, target), target)
            self.assertFalse(is_allowed_handoff(ROLE_ORCHESTRATOR, target), target)

    def test_decision_controller_is_only_route_to_attempt_history(self) -> None:
        self.assertTrue(can_handoff_to(ROLE_DECISION_CONTROLLER, ROLE_ATTEMPT_HISTORY))
        self.assertTrue(is_allowed_handoff(ROLE_DECISION_CONTROLLER, ROLE_ATTEMPT_HISTORY))

        forbidden_sources = (
            ROLE_CANDIDATE_WRITER,
            ROLE_DETERMINISTIC_GATE,
            ROLE_QUALITY_EVALUATOR,
            ROLE_REPAIR_AGENT,
            ROLE_REPAIR_PLANNER,
            ROLE_ORCHESTRATOR,
        )
        for source in forbidden_sources:
            self.assertFalse(can_handoff_to(source, ROLE_ATTEMPT_HISTORY), source)
            self.assertFalse(is_allowed_handoff(source, ROLE_ATTEMPT_HISTORY), source)

    def test_orchestrated_handoff_sequence_is_declarative(self) -> None:
        self.assertIn(
            (ROLE_CANDIDATE_WRITER, ROLE_DETERMINISTIC_GATE),
            orchestrated_handoff_sequence(),
        )
        self.assertIn(
            (ROLE_REPAIR_AGENT, ROLE_DETERMINISTIC_GATE),
            orchestrated_handoff_sequence(),
        )
        self.assertNotIn(
            (ROLE_ORCHESTRATOR, ROLE_QUALITY_EVALUATOR),
            orchestrated_handoff_sequence(),
        )

    def test_contract_serialization_to_dict(self) -> None:
        contract_dict = get_access_contract(ROLE_REPAIR_AGENT).to_dict()

        self.assertEqual(contract_dict["agent_role"], ROLE_REPAIR_AGENT)
        self.assertEqual(contract_dict["access_mode"], "write_revised_payload")
        self.assertTrue(contract_dict["may_create_revised_payload"])
        self.assertEqual(contract_dict["next_allowed_handoffs"], [ROLE_DETERMINISTIC_GATE])

    def test_access_module_does_not_call_api_or_execute_prompts(self) -> None:
        source = inspect.getsource(linkedin_post_flow_access)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
