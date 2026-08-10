from __future__ import annotations

from dataclasses import dataclass
import inspect
import json

from django.test import SimpleTestCase

from services.packaging import linkedin_post_prompt_renderers
from services.packaging.linkedin_final_post_diagnostics import FinalPostDiagnostics
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_editorial_boundary import PostEditorialInput
from services.packaging.linkedin_post_editorial_boundary import PostGenerationMetadata
from services.packaging.linkedin_post_candidate_post_contract import (
    build_candidate_post_constraints,
)
from services.packaging.linkedin_post_final_post_payload_contract import (
    FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
)
from services.packaging.linkedin_post_flow_input_builders import (
    CandidateWriterInput,
    build_candidate_writer_input,
)
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
    CANDIDATE_POST_PROMPT_FIELDS,
    QualityEvaluatorPromptRender,
    RepairWriterPromptRender,
    SELECTED_EVIDENCE_PROMPT_FIELDS,
    SemanticGroundingPromptRender,
    render_candidate_writer_prompt_input,
    render_quality_evaluator_prompt_input,
    render_repair_writer_prompt_input,
    render_semantic_grounding_prompt_input,
)
from services.packaging.linkedin_post_quality_rubric_contract import (
    get_quality_evaluator_rubric_payload,
)
from services.packaging.linkedin_post_semantic_grounding_contract import (
    build_semantic_grounding_prompt_rules,
)


QUALITY_EVALUATOR_VARIABLES = (
    "candidate_payload_json",
    "post_brief_json",
    "angle_decision_json",
    "authorial_voice_directive_json",
    "selected_evidence_json",
    "quality_rubric_json",
)

SEMANTIC_GROUNDING_VARIABLES = (
    "candidate_payload_json",
    "post_brief_json",
    "angle_decision_json",
    "selected_evidence_json",
    "semantic_grounding_rules_json",
)

REPAIR_WRITER_VARIABLES = (
    "original_candidate_payload_json",
    "post_brief_json",
    "angle_decision_json",
    "selected_evidence_json",
    "deterministic_findings_json",
    "quality_findings_json",
    "repair_instruction_json",
    "repair_attempt_json",
)


@dataclass(frozen=True)
class EvidenceUseStub:
    evidence_id: str
    evidence_text: str
    role_in_post: str


@dataclass(frozen=True)
class PostBriefStub:
    core_point: str
    evidence_to_use: list[EvidenceUseStub]


@dataclass(frozen=True)
class AngleDecisionStub:
    controlling_angle: str
    supporting_evidence_ids: list[str]
    authorial_voice_directive: dict[str, object]


class LinkedInPostPromptRenderersTests(SimpleTestCase):
    def test_render_candidate_writer_prompt_input_returns_contract(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertIsInstance(render, CandidateWriterPromptRender)

    def test_variables_include_expected_json_sections(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertEqual(
            set(render.variables),
            {
                "post_brief_json",
                "angle_decision_json",
                "authorial_voice_directive_json",
                "personal_presence_instruction",
                "selected_evidence_json",
                "candidate_writer_input_json",
                "candidate_post_constraints_json",
            },
        )

    def test_candidate_writer_variables_include_canonical_payload_constraints(
        self,
    ) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertEqual(
            json.loads(render.variables["candidate_post_constraints_json"]),
            build_candidate_post_constraints(),
        )
        candidate_post_constraints = json.loads(
            render.variables["candidate_post_constraints_json"]
        )
        self.assertEqual(
            candidate_post_constraints["post_text"]["max_chars"],
            FINAL_POST_PAYLOAD_POST_TEXT_MAX_CHARS,
        )
        self.assertEqual(
            candidate_post_constraints["post_text"]["prompt_target_min_chars"],
            1100,
        )
        self.assertEqual(
            candidate_post_constraints["post_text"]["prompt_target_max_chars"],
            1200,
        )
        self.assertIn("CANDIDATE_POST_CONSTRAINTS_JSON", render.input_text)

    def test_rendered_selected_evidence_preserves_id_order(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a0-summary", "a1-kp0"],
        )

    def test_rendered_selected_evidence_preserves_evidence_text(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_text"] for item in selected_evidence],
            [
                "Remote teams need clear operating agreements.",
                "Isolation can rise when remote work is unmanaged.",
            ],
        )

    def test_rendered_selected_evidence_preserves_role_in_post(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["role_in_post"] for item in selected_evidence],
            ["opening support", "practical tension"],
        )

    def test_rendered_output_includes_serialized_post_brief(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        post_brief = json.loads(render.variables["post_brief_json"])

        self.assertEqual(post_brief["core_point"], "Use selected evidence.")
        self.assertEqual(post_brief["evidence_to_use"][0]["evidence_id"], "a0-summary")
        self.assertIn("POST_BRIEF_JSON", render.input_text)

    def test_rendered_output_includes_serialized_angle_decision(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        angle_decision = json.loads(render.variables["angle_decision_json"])

        self.assertEqual(
            angle_decision["controlling_angle"],
            "Make remote work explicit.",
        )
        self.assertEqual(
            angle_decision["supporting_evidence_ids"],
            ["a0-summary", "a1-kp0"],
        )
        self.assertIn("ANGLE_DECISION_JSON", render.input_text)

    def test_candidate_writer_render_includes_authorial_voice_directive(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        directive = json.loads(render.variables["authorial_voice_directive_json"])

        self.assertEqual(directive, _authorial_voice_directive())
        self.assertIn("AUTHORIAL_VOICE_DIRECTIVE_JSON", render.input_text)

    def test_candidate_writer_render_includes_personal_presence_instruction(
        self,
    ) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertIn("personal_presence_instruction", render.variables)
        self.assertIn(
            "Include exactly one naturally integrated author-owned interpretive statement",
            render.variables["personal_presence_instruction"],
        )
        self.assertIn(
            "An impersonal editorial judgment is not sufficient",
            render.variables["personal_presence_instruction"],
        )
        self.assertIn("Use first person only when natural", render.variables["personal_presence_instruction"])
        self.assertIn("PERSONAL_PRESENCE_INSTRUCTION", render.input_text)

    def test_candidate_writer_render_rejects_missing_personal_presence_requirement(
        self,
    ) -> None:
        directive = dict(_authorial_voice_directive())
        directive.pop("personal_presence_requirement")
        candidate_input = _malformed_candidate_input_for_renderer(
            {
                "controlling_angle": "Make remote work explicit.",
                "supporting_evidence_ids": ["a0-summary", "a1-kp0"],
                "authorial_voice_directive": directive,
            }
        )

        with self.assertRaisesRegex(TypeError, "personal_presence_requirement"):
            render_candidate_writer_prompt_input(candidate_input)

    def test_candidate_writer_render_rejects_unknown_personal_presence_requirement(
        self,
    ) -> None:
        directive = {
            **_authorial_voice_directive(),
            "personal_presence_requirement": "invented_policy",
        }
        candidate_input = _malformed_candidate_input_for_renderer(
            {
                "controlling_angle": "Make remote work explicit.",
                "supporting_evidence_ids": ["a0-summary", "a1-kp0"],
                "authorial_voice_directive": directive,
            }
        )

        with self.assertRaisesRegex(ValueError, "personal_presence_requirement"):
            render_candidate_writer_prompt_input(candidate_input)

    def test_candidate_writer_render_rejects_missing_authorial_voice_directive(
        self,
    ) -> None:
        candidate_input = _malformed_candidate_input_for_renderer(
            {
                "controlling_angle": "Make remote work explicit.",
                "supporting_evidence_ids": ["a0-summary", "a1-kp0"],
            }
        )

        with self.assertRaisesRegex(TypeError, "authorial_voice_directive"):
            render_candidate_writer_prompt_input(candidate_input)

    def test_prompt_metadata_is_copied_when_provided(self) -> None:
        prompt_metadata = _prompt_metadata()
        render = render_candidate_writer_prompt_input(
            _candidate_input(prompt_metadata=prompt_metadata)
        )

        self.assertEqual(render.prompt_name, "final_post_candidate_from_brief")
        self.assertEqual(render.prompt_version, "1.0")
        self.assertEqual(render.prompt_path, "prompts/linkedin/final_post_from_brief.txt")

    def test_prompt_metadata_fields_are_none_when_not_provided(self) -> None:
        render = render_candidate_writer_prompt_input(_candidate_input(prompt_metadata=None))

        self.assertIsNone(render.prompt_name)
        self.assertIsNone(render.prompt_version)
        self.assertIsNone(render.prompt_path)

    def test_input_text_is_deterministic_for_same_input(self) -> None:
        first_render = render_candidate_writer_prompt_input(_candidate_input())
        second_render = render_candidate_writer_prompt_input(_candidate_input())

        self.assertEqual(first_render.variables, second_render.variables)
        self.assertEqual(first_render.input_text, second_render.input_text)

    def test_to_dict_output_is_json_serializable(self) -> None:
        render = render_candidate_writer_prompt_input(
            _candidate_input(prompt_metadata=_prompt_metadata())
        )

        serialized = json.dumps(render.to_dict(), sort_keys=True)

        self.assertIn("candidate_writer_input_json", serialized)
        self.assertIn("final_post_candidate_from_brief", serialized)

    def test_renderer_does_not_mutate_candidate_writer_input(self) -> None:
        candidate_input = _candidate_input()
        before = candidate_input.to_dict()

        render_candidate_writer_prompt_input(candidate_input)

        self.assertEqual(candidate_input.to_dict(), before)

    def test_renderer_does_not_import_api_prompt_execution_registry_or_runtime_wiring(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_renderers)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("generate_text", source)
        self.assertNotIn("call_command", source)
        self.assertNotIn("linkedin_post_prompt_registry", source)
        self.assertNotIn("services.packaging.generator", source)
        self.assertNotIn("generate_content_package_for_digest", source)

    def test_renderer_does_not_reference_raw_articles(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_renderers).lower()

        self.assertNotIn("raw_article", source)
        self.assertNotIn("articles", source)

    def test_render_quality_evaluator_prompt_input_returns_contract(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertIsInstance(render, QualityEvaluatorPromptRender)

    def test_quality_evaluator_variables_are_exact_and_ordered(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertEqual(tuple(render.variables), QUALITY_EVALUATOR_VARIABLES)

    def test_quality_evaluator_receives_canonical_authorial_voice_directive(
        self,
    ) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertEqual(
            json.loads(render.variables["authorial_voice_directive_json"]),
            _authorial_voice_directive(),
        )
        self.assertIn("AUTHORIAL_VOICE_DIRECTIVE_JSON", render.input_text)
        self.assertNotIn(
            "authorial_voice_directive",
            render.variables["angle_decision_json"],
        )

    def test_quality_evaluator_authorial_voice_directive_json_filters_extra_fields(
        self,
    ) -> None:
        directive = {
            **_authorial_voice_directive(),
            "debug": "directive-debug-sentinel",
            "provider_metadata": "directive-provider-sentinel",
            "runtime_control": "directive-runtime-sentinel",
        }
        editorial_input = _post_editorial_input(
            angle_decision={
                **_post_editorial_input().angle_decision,
                "authorial_voice_directive": directive,
            }
        )

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )
        rendered_directive = json.loads(
            render.variables["authorial_voice_directive_json"]
        )
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        self.assertEqual(rendered_directive, _authorial_voice_directive())
        for forbidden in (
            "debug",
            "provider_metadata",
            "runtime_control",
            "directive-debug-sentinel",
            "directive-provider-sentinel",
            "directive-runtime-sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_variable_values_are_strings(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        for value in render.variables.values():
            self.assertIsInstance(value, str)

    def test_quality_evaluator_variables_use_stable_json_format(self) -> None:
        editorial_input = _post_editorial_input()
        rubric = get_quality_evaluator_rubric_payload()

        render = render_quality_evaluator_prompt_input(editorial_input, rubric)

        self.assertEqual(
            render.variables["candidate_payload_json"],
            _stable_json(_canonical_candidate_payload(editorial_input.candidate_payload)),
        )
        self.assertEqual(
            render.variables["post_brief_json"],
            _stable_json(editorial_input.post_brief),
        )
        self.assertEqual(
            render.variables["angle_decision_json"],
            _stable_json(
                {
                    field_name: editorial_input.angle_decision[field_name]
                    for field_name in linkedin_post_prompt_renderers.QUALITY_ANGLE_DECISION_PROMPT_FIELDS
                    if field_name in editorial_input.angle_decision
                }
            ),
        )
        self.assertEqual(
            render.variables["selected_evidence_json"],
            _stable_json(_prompt_selected_evidence(editorial_input.selected_evidence)),
        )
        self.assertEqual(
            render.variables["quality_rubric_json"],
            _stable_json(rubric.to_prompt_dict()),
        )

    def test_quality_evaluator_candidate_payload_json_contains_only_candidate_payload(self) -> None:
        editorial_input = _post_editorial_input()

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        self.assertEqual(
            json.loads(render.variables["candidate_payload_json"]),
            _canonical_candidate_payload(editorial_input.candidate_payload),
        )
        self.assertIn("post_text", json.loads(render.variables["candidate_payload_json"]))

    def test_quality_evaluator_candidate_payload_json_uses_core_candidate_post_fields(self) -> None:
        self.assertEqual(CANDIDATE_POST_PROMPT_FIELDS, ("post_text",))

    def test_render_semantic_grounding_prompt_input_returns_contract(self) -> None:
        render = render_semantic_grounding_prompt_input(_post_editorial_input())

        self.assertIsInstance(render, SemanticGroundingPromptRender)

    def test_semantic_grounding_variables_are_exact_and_ordered(self) -> None:
        render = render_semantic_grounding_prompt_input(_post_editorial_input())

        self.assertEqual(tuple(render.variables), SEMANTIC_GROUNDING_VARIABLES)

    def test_semantic_grounding_rules_variable_matches_contract_helper(self) -> None:
        render = render_semantic_grounding_prompt_input(_post_editorial_input())

        self.assertEqual(
            json.loads(render.variables["semantic_grounding_rules_json"]),
            build_semantic_grounding_prompt_rules(),
        )
        self.assertIn("SEMANTIC_GROUNDING_RULES_JSON", render.input_text)

    def test_semantic_grounding_preserves_selected_evidence_order_and_text(self) -> None:
        editorial_input = _post_editorial_input(
            selected_evidence=(
                {
                    "evidence_id": "a2-summary",
                    "evidence_text": "K33 says Bitcoin likely bottomed at $60K; risk remains.",
                    "role_in_post": "qualification",
                },
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Security concerns and volatility limit broader adoption.",
                    "role_in_post": "constraint",
                },
            )
        )

        render = render_semantic_grounding_prompt_input(editorial_input)
        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a2-summary", "a0-summary"],
        )
        self.assertEqual(
            selected_evidence[0]["evidence_text"],
            "K33 says Bitcoin likely bottomed at $60K; risk remains.",
        )
        self.assertEqual(selected_evidence[0]["role_in_post"], "qualification")

    def test_semantic_grounding_prompt_filters_runtime_metadata(self) -> None:
        editorial_input = _post_editorial_input(
            candidate_payload={
                **_candidate_payload(),
                "provider": "provider-sentinel",
                "model": "model-sentinel",
                "token_usage": {"total_tokens": 12},
                "raw_provider_response": {"id": "raw-sentinel"},
                "raw_articles": ["raw-article-sentinel"],
            }
        )

        render = render_semantic_grounding_prompt_input(editorial_input)
        rendered = "\n".join([*render.variables.values(), render.input_text])

        self.assertNotIn("provider-sentinel", rendered)
        self.assertNotIn("model-sentinel", rendered)
        self.assertNotIn("raw-sentinel", rendered)
        self.assertNotIn("raw-article-sentinel", rendered)
        self.assertIn("selected evidence", rendered)

    def test_semantic_grounding_candidate_payload_json_is_core_post_only(self) -> None:
        editorial_input = _post_editorial_input(candidate_payload=_candidate_payload())

        render = render_semantic_grounding_prompt_input(editorial_input)
        rendered_payload = json.loads(render.variables["candidate_payload_json"])
        rendered_text = render.variables["candidate_payload_json"]

        self.assertEqual(rendered_payload, {"post_text": _candidate_payload()["post_text"]})
        for forbidden in (
            "hook_variants",
            "cta_variants",
            "hashtags",
            "quality_checks",
            "carousel_outline",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_payload)
                self.assertNotIn(forbidden, rendered_text)

    def test_semantic_grounding_prompt_filters_brief_and_angle_to_selected_context(
        self,
    ) -> None:
        editorial_input = _post_editorial_input(
            post_brief={
                **_post_editorial_input().post_brief,
                "raw_articles": ["brief-raw-article-sentinel"],
                "debug": {"trace": "brief-debug-sentinel"},
                "runtime_metadata": "brief-runtime-sentinel",
                "evidence_to_use": [
                    {
                        "evidence_id": "a0-summary",
                        "evidence_text": "Remote teams need clear operating agreements.",
                        "role_in_post": "opening support",
                        "raw_article": "selected-raw-article-sentinel",
                        "debug": "selected-debug-sentinel",
                    },
                    {
                        "evidence_id": "a1-kp0",
                        "evidence_text": "Isolation can rise when remote work is unmanaged.",
                        "role_in_post": "practical tension",
                    },
                    {
                        "evidence_id": "unselected-ev",
                        "evidence_text": "Unselected evidence sentinel.",
                        "role_in_post": "must_not_enter_prompt",
                    },
                ],
            },
            angle_decision={
                **_post_editorial_input().angle_decision,
                "supporting_evidence_ids": ["a0-summary", "unselected-ev", "a1-kp0"],
                "raw_articles": ["angle-raw-article-sentinel"],
                "debug": {"trace": "angle-debug-sentinel"},
                "provider_metadata": "angle-provider-sentinel",
                "unselected_evidence": "angle-unselected-evidence-sentinel",
            },
        )

        render = render_semantic_grounding_prompt_input(editorial_input)
        post_brief = json.loads(render.variables["post_brief_json"])
        angle_decision = json.loads(render.variables["angle_decision_json"])
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        self.assertEqual(
            [item["evidence_id"] for item in post_brief["evidence_to_use"]],
            ["a0-summary", "a1-kp0"],
        )
        self.assertEqual(
            angle_decision["supporting_evidence_ids"],
            ["a0-summary", "a1-kp0"],
        )
        for item in post_brief["evidence_to_use"]:
            self.assertEqual(set(item), set(SELECTED_EVIDENCE_PROMPT_FIELDS))
        for forbidden in (
            "brief-raw-article-sentinel",
            "brief-debug-sentinel",
            "brief-runtime-sentinel",
            "selected-raw-article-sentinel",
            "selected-debug-sentinel",
            "unselected-ev",
            "Unselected evidence sentinel.",
            "must_not_enter_prompt",
            "angle-raw-article-sentinel",
            "angle-debug-sentinel",
            "angle-provider-sentinel",
            "angle-unselected-evidence-sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_semantic_grounding_does_not_receive_authorial_voice_directive(
        self,
    ) -> None:
        render = render_semantic_grounding_prompt_input(_post_editorial_input())

        self.assertNotIn("authorial_voice_directive_json", render.variables)
        self.assertNotIn("AUTHORIAL_VOICE_DIRECTIVE_JSON", render.input_text)
        self.assertNotIn(
            "authorial_voice_directive",
            render.variables["angle_decision_json"],
        )

    def test_quality_evaluator_candidate_payload_json_filters_extra_fields(self) -> None:
        extra_candidate_fields = {
            "diagnostics": "diagnostics-sentinel",
            "repair_reasons": ["repair-sentinel"],
            "validation_error": "validation-error-extra-sentinel",
            "provider": "provider-sentinel",
            "model": "model-sentinel",
            "token_usage": {"input_tokens": 100},
            "cost_metadata": {"estimated_usd": "0.01"},
            "run_id": "run-id-extra-sentinel",
            "created_at": "created-at-extra-sentinel",
            "attempt_history": "attempt-history-sentinel",
            "decision": "decision-sentinel",
            "debug": "debug-sentinel",
            "runtime": "runtime-sentinel",
            "raw_articles": ["raw-article-extra-sentinel"],
            "source_articles": ["source-article-extra-sentinel"],
            "internal_notes": "internal-notes-sentinel",
        }
        editorial_input = _post_editorial_input(
            candidate_payload={
                **_candidate_payload(),
                **extra_candidate_fields,
            }
        )

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )
        candidate_payload = json.loads(render.variables["candidate_payload_json"])
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        self.assertEqual(set(candidate_payload), set(CANDIDATE_POST_PROMPT_FIELDS))
        for forbidden in (
            *extra_candidate_fields,
            "diagnostics-sentinel",
            "repair-sentinel",
            "validation-error-extra-sentinel",
            "provider-sentinel",
            "model-sentinel",
            "input_tokens",
            "estimated_usd",
            "run-id-extra-sentinel",
            "created-at-extra-sentinel",
            "attempt-history-sentinel",
            "decision-sentinel",
            "debug-sentinel",
            "runtime-sentinel",
            "raw-article-extra-sentinel",
            "source-article-extra-sentinel",
            "internal-notes-sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_rejects_missing_candidate_post_text(self) -> None:
        editorial_input = _post_editorial_input(candidate_payload={"hook_variants": []})

        with self.assertRaisesRegex(TypeError, "candidate_payload.post_text"):
            render_quality_evaluator_prompt_input(
                editorial_input,
                get_quality_evaluator_rubric_payload(),
            )

    def test_semantic_grounding_rejects_missing_candidate_post_text(self) -> None:
        editorial_input = _post_editorial_input(candidate_payload={"hook_variants": []})

        with self.assertRaisesRegex(TypeError, "candidate_payload.post_text"):
            render_semantic_grounding_prompt_input(editorial_input)
    def test_quality_evaluator_candidate_payload_json_excludes_packaging_fields(self) -> None:
        editorial_input = _post_editorial_input(candidate_payload=_candidate_payload())

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        rendered_payload = json.loads(render.variables["candidate_payload_json"])
        rendered_text = render.variables["candidate_payload_json"]

        self.assertEqual(rendered_payload, {"post_text": _candidate_payload()["post_text"]})
        for forbidden in (
            "hook_variants",
            "cta_variants",
            "hashtags",
            "quality_checks",
            "carousel_outline",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_payload)
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_serializes_complete_post_brief_snapshot(self) -> None:
        editorial_input = _post_editorial_input()

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        self.assertEqual(
            json.loads(render.variables["post_brief_json"]),
            editorial_input.post_brief,
        )
        self.assertIn(
            "evidence_to_use",
            json.loads(render.variables["post_brief_json"]),
        )

    def test_quality_evaluator_serializes_quality_angle_decision_snapshot(self) -> None:
        editorial_input = _post_editorial_input()

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        angle_decision_json = json.loads(render.variables["angle_decision_json"])
        self.assertEqual(
            angle_decision_json,
            {
                field_name: editorial_input.angle_decision[field_name]
                for field_name in linkedin_post_prompt_renderers.QUALITY_ANGLE_DECISION_PROMPT_FIELDS
                if field_name in editorial_input.angle_decision
            },
        )
        self.assertEqual(
            angle_decision_json["controlling_angle"],
            "Make remote work explicit.",
        )

    def test_quality_evaluator_serializes_object_brief_and_angle_inputs(self) -> None:
        editorial_input = _post_editorial_input(
            post_brief=_post_brief(),
            angle_decision=_angle_decision(),
        )

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        post_brief = json.loads(render.variables["post_brief_json"])
        angle_decision = json.loads(render.variables["angle_decision_json"])

        self.assertEqual(post_brief["core_point"], "Use selected evidence.")
        self.assertEqual(post_brief["evidence_to_use"][0]["evidence_id"], "a0-summary")
        self.assertEqual(
            angle_decision["controlling_angle"],
            "Make remote work explicit.",
        )
        self.assertEqual(
            angle_decision["supporting_evidence_ids"],
            ["a0-summary", "a1-kp0"],
        )

    def test_quality_evaluator_selected_evidence_order_text_and_role_are_preserved(self) -> None:
        editorial_input = _post_editorial_input()

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )
        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a0-summary", "a1-kp0"],
        )
        self.assertEqual(
            [item["evidence_text"] for item in selected_evidence],
            [
                "Remote teams need clear operating agreements.",
                "Isolation can rise when remote work is unmanaged. café",
            ],
        )
        self.assertEqual(
            [item["role_in_post"] for item in selected_evidence],
            ["opening support", "practical tension"],
        )

    def test_quality_evaluator_selected_evidence_json_filters_extra_fields(self) -> None:
        first_extra_fields = {
            "source_url": "https://example.invalid/source",
            "article_id": "article-id-sentinel",
            "article": {"title": "raw article title sentinel"},
            "raw_article": "raw article body sentinel",
            "source_summary": "source summary sentinel",
            "source_title": "source title sentinel",
            "source_index": 0,
            "provider": "evidence-provider-sentinel",
            "score": 99,
            "debug": "evidence-debug-sentinel",
            "internal_notes": "evidence-internal-notes-sentinel",
        }
        second_extra_fields = {
            "source_url": "https://example.invalid/second",
            "article_id": "second-article-id-sentinel",
            "article": {"title": "second raw article title sentinel"},
            "raw_article": "second raw article body sentinel",
            "source_summary": "second source summary sentinel",
            "source_title": "second source sentinel",
            "source_index": 1,
            "provider": "second-evidence-provider-sentinel",
            "score": 88,
            "debug": "second-evidence-debug-sentinel",
            "internal_notes": "second-evidence-internal-notes-sentinel",
        }
        editorial_input = _post_editorial_input(
            selected_evidence=[
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Remote teams need clear operating agreements.",
                    "role_in_post": "opening support",
                    **first_extra_fields,
                },
                {
                    "evidence_id": "a1-kp0",
                    "evidence_text": "Isolation can rise when remote work is unmanaged.",
                    "role_in_post": "practical tension",
                    **second_extra_fields,
                },
            ]
        )

        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )
        selected_evidence = json.loads(render.variables["selected_evidence_json"])
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        for item in selected_evidence:
            self.assertEqual(set(item), set(SELECTED_EVIDENCE_PROMPT_FIELDS))
        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a0-summary", "a1-kp0"],
        )
        self.assertEqual(
            [item["evidence_text"] for item in selected_evidence],
            [
                "Remote teams need clear operating agreements.",
                "Isolation can rise when remote work is unmanaged.",
            ],
        )
        self.assertEqual(
            [item["role_in_post"] for item in selected_evidence],
            ["opening support", "practical tension"],
        )
        for extra_field in (*first_extra_fields, *second_extra_fields):
            with self.subTest(extra_field=extra_field):
                self.assertNotIn(extra_field, render.variables["selected_evidence_json"])
        for forbidden in (
            "https://example.invalid/source",
            "article-id-sentinel",
            "raw article title sentinel",
            "raw article body sentinel",
            "source summary sentinel",
            "source title sentinel",
            "evidence-provider-sentinel",
            "evidence-debug-sentinel",
            "evidence-internal-notes-sentinel",
            "https://example.invalid/second",
            "second-article-id-sentinel",
            "second raw article title sentinel",
            "second raw article body sentinel",
            "second source summary sentinel",
            "second source sentinel",
            "second-evidence-provider-sentinel",
            "second-evidence-debug-sentinel",
            "second-evidence-internal-notes-sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_rubric_json_uses_prompt_dict_not_to_dict(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        rubric_json = json.loads(render.variables["quality_rubric_json"])

        self.assertIn("rubric_version", rubric_json)
        self.assertNotIn("source_document", rubric_json)

    def test_quality_evaluator_unicode_is_preserved(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertIn("café", render.variables["selected_evidence_json"])
        self.assertIn("человечность", render.variables["candidate_payload_json"])

    def test_quality_evaluator_candidate_payload_json_is_core_post_only(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertEqual(
            json.loads(render.variables["candidate_payload_json"]),
            {"post_text": _post_editorial_input().candidate_payload["post_text"]},
        )

    def test_quality_evaluator_input_text_has_exact_section_order(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )
        headers = [
            "## CANDIDATE_PAYLOAD_JSON",
            "## POST_BRIEF_JSON",
            "## ANGLE_DECISION_JSON",
            "## AUTHORIAL_VOICE_DIRECTIVE_JSON",
            "## SELECTED_EVIDENCE_JSON",
            "## QUALITY_RUBRIC_JSON",
        ]

        self.assertEqual(
            [line for line in render.input_text.splitlines() if line.startswith("## ")],
            headers,
        )
        for header in headers:
            self.assertEqual(render.input_text.count(header), 1)
        self.assertNotIn("```", render.input_text)

    def test_quality_evaluator_every_serialized_variable_appears_once_in_input_text(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        for value in render.variables.values():
            self.assertEqual(render.input_text.count(value), 1)

    def test_quality_evaluator_does_not_serialize_full_post_editorial_input(self) -> None:
        rendered_text = _rendered_quality_evaluator_text()

        for forbidden in (
            "final_payload_validation_passed",
            "final_payload_validation_error",
            "diagnostics",
            "repair_reasons",
            "generation_metadata",
            "prompt_metadata",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_excludes_audit_runtime_fields(self) -> None:
        rendered_text = _rendered_quality_evaluator_text()

        for forbidden in (
            "schema_validation_passed",
            "system_linkedin_ready",
            "repair-reason-sentinel",
            "candidate-writer-prompt",
            "provider-sentinel",
            "model-sentinel",
            "input_tokens",
            "estimated_usd",
            "run-sentinel",
            "2026-07-05T10:00:00Z",
            "source_document",
            "raw article sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_quality_evaluator_prompt_metadata_is_explicit(self) -> None:
        prompt_metadata = PromptMetadata(
            prompt_name="final_post_quality_evaluator",
            prompt_version="1.0",
            prompt_path="prompts/linkedin/final_post_quality_evaluator.txt",
        )

        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
            prompt_metadata=prompt_metadata,
        )

        self.assertEqual(render.prompt_name, "final_post_quality_evaluator")
        self.assertEqual(render.prompt_version, "1.0")
        self.assertEqual(
            render.prompt_path,
            "prompts/linkedin/final_post_quality_evaluator.txt",
        )

    def test_quality_evaluator_prompt_metadata_is_not_copied_from_editorial_input(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        self.assertIsNone(render.prompt_name)
        self.assertIsNone(render.prompt_version)
        self.assertIsNone(render.prompt_path)
        self.assertNotIn("candidate-writer-prompt", json.dumps(render.to_dict()))

    def test_quality_evaluator_to_dict_output_is_json_serializable(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
            prompt_metadata=PromptMetadata(
                prompt_name="final_post_quality_evaluator",
                prompt_version="1.0",
                prompt_path="prompts/linkedin/final_post_quality_evaluator.txt",
            ),
        )

        serialized = json.dumps(render.to_dict(), ensure_ascii=False, sort_keys=True)

        self.assertIn("final_post_quality_evaluator", serialized)

    def test_quality_evaluator_renderer_does_not_mutate_inputs(self) -> None:
        editorial_input = _post_editorial_input()
        rubric = get_quality_evaluator_rubric_payload()
        before_editorial = editorial_input.to_dict()
        before_rubric = rubric.to_dict()

        render_quality_evaluator_prompt_input(editorial_input, rubric)

        self.assertEqual(editorial_input.to_dict(), before_editorial)
        self.assertEqual(rubric.to_dict(), before_rubric)

    def test_quality_evaluator_render_output_is_independent_from_source_structures(self) -> None:
        editorial_input = _post_editorial_input()
        render = render_quality_evaluator_prompt_input(
            editorial_input,
            get_quality_evaluator_rubric_payload(),
        )

        mutation_sentinel = "MUTATED_AFTER_RENDER_SENTINEL"
        editorial_input.candidate_payload["post_text"] = mutation_sentinel
        editorial_input.post_brief["core_point"] = mutation_sentinel
        editorial_input.angle_decision["controlling_angle"] = mutation_sentinel
        editorial_input.selected_evidence[0]["evidence_text"] = mutation_sentinel

        self.assertIn("Remote work needs explicit systems", render.input_text)
        self.assertIn("Make remote work explicit.", render.input_text)
        self.assertIn("Remote teams need clear operating agreements.", render.input_text)
        self.assertNotIn(mutation_sentinel, render.input_text)

    def test_quality_evaluator_to_dict_variables_are_defensively_copied(self) -> None:
        render = render_quality_evaluator_prompt_input(
            _post_editorial_input(),
            get_quality_evaluator_rubric_payload(),
        )

        render_dict = render.to_dict()
        render_dict["variables"]["candidate_payload_json"] = "changed"

        self.assertNotEqual(render.variables["candidate_payload_json"], "changed")

    def test_quality_evaluator_renderer_does_not_import_execution_registry_or_runtime_wiring(self) -> None:
        source = inspect.getsource(linkedin_post_prompt_renderers)

        for forbidden in (
            "OpenAIClient",
            "generate_text",
            "call_command",
            "linkedin_post_prompt_registry",
            "normalize_quality_review_result",
            "FinalPostDecisionController",
            "TargetedRepairPlan",
            "FinalPostAttempt",
            "ContentPackage",
            "services.packaging.generator",
            "generate_content_package_for_digest",
            "django",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, source)

    def test_render_repair_writer_prompt_input_returns_contract(self) -> None:
        render = _repair_writer_render()

        self.assertIsInstance(render, RepairWriterPromptRender)
        self.assertEqual(tuple(render.variables), REPAIR_WRITER_VARIABLES)

    def test_repair_writer_render_preserves_selected_evidence_boundary(self) -> None:
        editorial_input = _post_editorial_input()
        render = _repair_writer_render()

        selected_evidence = json.loads(render.variables["selected_evidence_json"])

        self.assertEqual(
            [item["evidence_id"] for item in selected_evidence],
            ["a0-summary", "a1-kp0"],
        )
        self.assertEqual(
            [item["evidence_text"] for item in selected_evidence],
            [item["evidence_text"] for item in editorial_input.selected_evidence],
        )
        self.assertEqual(
            [item["role_in_post"] for item in selected_evidence],
            ["opening support", "practical tension"],
        )

    def test_repair_writer_render_filters_candidate_payload_runtime_fields(self) -> None:
        render = _repair_writer_render(
            original_candidate_payload={
                **_candidate_payload(),
                "hook_variants": ["legacy hook sentinel"],
                "cta_variants": ["legacy cta sentinel"],
                "hashtags": ["#LegacySentinel"],
                "quality_checks": {"linkedin_ready": True},
                "carousel_outline": ["legacy carousel sentinel"],
                "provider": "provider-sentinel",
                "model": "model-sentinel",
                "raw_provider_response": "raw-provider-sentinel",
                "token_usage": {"input_tokens": 100},
                "debug": "debug-sentinel",
            }
        )
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        self.assertEqual(
            set(json.loads(render.variables["original_candidate_payload_json"])),
            set(CANDIDATE_POST_PROMPT_FIELDS),
        )
        for forbidden in (
            "provider-sentinel",
            "model-sentinel",
            "raw-provider-sentinel",
            "input_tokens",
            "debug-sentinel",
            "legacy hook sentinel",
            "legacy cta sentinel",
            "#LegacySentinel",
            "legacy carousel sentinel",
            "linkedin_ready",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_repair_writer_render_filters_brief_and_angle_context(self) -> None:
        render = _repair_writer_render(
            post_brief={
                **_post_editorial_input().post_brief,
                "raw_articles": ["brief-raw-article-sentinel"],
                "debug": {"trace": "brief-debug-sentinel"},
                "runtime_metadata": "brief-runtime-sentinel",
                "evidence_to_use": [
                    {
                        "evidence_id": "a0-summary",
                        "evidence_text": "Remote teams need clear operating agreements.",
                        "role_in_post": "opening support",
                        "raw_article": "selected-raw-article-sentinel",
                        "debug": "selected-debug-sentinel",
                    },
                    {
                        "evidence_id": "a1-kp0",
                        "evidence_text": "Isolation can rise when remote work is unmanaged.",
                        "role_in_post": "practical tension",
                    },
                    {
                        "evidence_id": "unselected-ev",
                        "evidence_text": "Unselected evidence sentinel.",
                        "role_in_post": "must_not_enter_prompt",
                    },
                ],
            },
            angle_decision={
                **_post_editorial_input().angle_decision,
                "supporting_evidence_ids": ["a0-summary", "unselected-ev", "a1-kp0"],
                "raw_articles": ["angle-raw-article-sentinel"],
                "debug": {"trace": "angle-debug-sentinel"},
                "provider_metadata": "angle-provider-sentinel",
                "unselected_evidence": "angle-unselected-evidence-sentinel",
            },
        )
        post_brief = json.loads(render.variables["post_brief_json"])
        angle_decision = json.loads(render.variables["angle_decision_json"])
        rendered_text = "\n".join([*render.variables.values(), render.input_text])

        self.assertEqual(
            [item["evidence_id"] for item in post_brief["evidence_to_use"]],
            ["a0-summary", "a1-kp0"],
        )
        self.assertEqual(
            angle_decision["supporting_evidence_ids"],
            ["a0-summary", "a1-kp0"],
        )
        for item in post_brief["evidence_to_use"]:
            self.assertEqual(set(item), set(SELECTED_EVIDENCE_PROMPT_FIELDS))
        for forbidden in (
            "brief-raw-article-sentinel",
            "brief-debug-sentinel",
            "brief-runtime-sentinel",
            "selected-raw-article-sentinel",
            "selected-debug-sentinel",
            "unselected-ev",
            "Unselected evidence sentinel.",
            "must_not_enter_prompt",
            "angle-raw-article-sentinel",
            "angle-debug-sentinel",
            "angle-provider-sentinel",
            "angle-unselected-evidence-sentinel",
        ):
            with self.subTest(forbidden=forbidden):
                self.assertNotIn(forbidden, rendered_text)

    def test_repair_writer_render_includes_only_structured_repair_facts(self) -> None:
        render = _repair_writer_render()
        repair_findings = json.loads(render.variables["quality_findings_json"])
        repair_instruction = json.loads(render.variables["repair_instruction_json"])

        self.assertEqual(repair_findings["failed_criteria"], ["human_voice"])
        self.assertEqual(repair_instruction["repair_type"], "editorial")
        self.assertIn("QUALITY_FINDINGS_JSON", render.input_text)
        self.assertIn("REPAIR_INSTRUCTION_JSON", render.input_text)

    def test_repair_writer_render_prompt_metadata_is_explicit(self) -> None:
        prompt_metadata = PromptMetadata(
            prompt_name="final_post_repair_writer",
            prompt_version="1.0",
            prompt_path=None,
        )

        render = _repair_writer_render(prompt_metadata=prompt_metadata)

        self.assertEqual(render.prompt_name, "final_post_repair_writer")
        self.assertEqual(render.prompt_version, "1.0")
        self.assertIsNone(render.prompt_path)



def _malformed_candidate_input_for_renderer(angle_decision):
    return CandidateWriterInput(
        post_brief=_post_brief(),
        angle_decision=angle_decision,
        selected_evidence=(
            {
                "evidence_id": "a0-summary",
                "evidence_text": "Remote teams need clear operating agreements.",
                "role_in_post": "opening support",
            },
            {
                "evidence_id": "a1-kp0",
                "evidence_text": "Isolation can rise when remote work is unmanaged.",
                "role_in_post": "practical tension",
            },
        ),
        prompt_metadata=None,
    )

def _candidate_input(prompt_metadata=None):
    return build_candidate_writer_input(
        _post_brief(),
        _angle_decision(),
        prompt_metadata=prompt_metadata,
    )


def _post_brief() -> PostBriefStub:
    return PostBriefStub(
        core_point="Use selected evidence.",
        evidence_to_use=[
            EvidenceUseStub(
                evidence_id="a0-summary",
                evidence_text="Remote teams need clear operating agreements.",
                role_in_post="opening support",
            ),
            EvidenceUseStub(
                evidence_id="a1-kp0",
                evidence_text="Isolation can rise when remote work is unmanaged.",
                role_in_post="practical tension",
            ),
        ],
    )


def _angle_decision() -> AngleDecisionStub:
    return AngleDecisionStub(
        controlling_angle="Make remote work explicit.",
        supporting_evidence_ids=["a0-summary", "a1-kp0"],
        authorial_voice_directive=_authorial_voice_directive(),
    )


def _authorial_voice_directive() -> dict[str, object]:
    return {
        "authorial_observation": (
            "The author notices that remote policies and isolation are separate signals."
        ),
        "rejected_reading": (
            "Reject treating policy documentation as proof that isolation has been solved."
        ),
        "why_distinction_matters": (
            "The distinction matters because remote work needs operating rules and support."
        ),
        "personal_presence_requirement": "explicit_author_owned_statement_required",
        "first_person_policy": "allowed_not_required",
        "forbidden_author_claims": [
            "personal experience",
            "professional authority",
            "direct market exposure",
            "client or customer stories",
            "invented emotional reaction",
            "biographical claims",
        ],
    }


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
    )


def _post_editorial_input(
    *,
    candidate_payload=None,
    post_brief=None,
    angle_decision=None,
    selected_evidence=None,
) -> PostEditorialInput:
    return PostEditorialInput(
        candidate_payload=candidate_payload or {
            "post_text": "Remote work needs explicit systems. человечность",
            "hook_variants": ["Remote work fails quietly."],
            "cta_variants": ["What system changed remote work for your team?"],
            "hashtags": ["#RemoteWork"],
            "carousel_outline": [],
            "quality_checks": {
                "linkedin_ready": True,
                "uses_only_provided_facts": True,
                "has_clear_point_of_view": True,
            },
        },
        post_brief=post_brief or {
            "core_point": "Use selected evidence.",
            "evidence_to_use": [
                {
                    "evidence_id": "a0-summary",
                    "evidence_text": "Remote teams need clear operating agreements.",
                    "role_in_post": "opening support",
                },
                {
                    "evidence_id": "a1-kp0",
                    "evidence_text": "Isolation can rise when remote work is unmanaged. café",
                    "role_in_post": "practical tension",
                },
            ],
        },
        angle_decision=angle_decision or {
            "controlling_angle": "Make remote work explicit.",
            "author_position": "Remote policy is an operating system.",
            "reader_problem": "Hybrid teams often rely on implicit norms.",
            "main_tension": "Flexibility can create isolation.",
            "supporting_evidence_ids": ["a0-summary", "a1-kp0"],
            "authorial_voice_directive": _authorial_voice_directive(),
        },
        selected_evidence=selected_evidence or [
            {
                "evidence_id": "a0-summary",
                "evidence_text": "Remote teams need clear operating agreements.",
                "role_in_post": "opening support",
            },
            {
                "evidence_id": "a1-kp0",
                "evidence_text": "Isolation can rise when remote work is unmanaged. café",
                "role_in_post": "practical tension",
            },
        ],
        final_payload_validation_passed=True,
        final_payload_validation_error="validation-error-sentinel",
        diagnostics=FinalPostDiagnostics(
            schema_validation_passed=True,
            schema_validation_error="schema-error-sentinel",
            missing_quality_check_keys=[],
            non_boolean_quality_check_keys=[],
            evidence_id_leaks=[],
            scaffold_phrase_leaks=[],
            source_summary_phrase_leaks=[],
            model_claimed_linkedin_ready=True,
            deterministic_checks_passed=True,
            system_linkedin_ready=True,
            repair_reasons=["repair-reason-sentinel"],
        ),
        repair_reasons=["repair-reason-sentinel"],
        generation_metadata=PostGenerationMetadata(
            provider="provider-sentinel",
            model="model-sentinel",
            run_id="run-sentinel",
            created_at="2026-07-05T10:00:00Z",
            token_usage={"input_tokens": 100},
            cost_metadata={"estimated_usd": "0.01"},
        ),
        prompt_metadata=PromptMetadata(
            prompt_name="candidate-writer-prompt",
            prompt_version="candidate-version",
            prompt_path="prompts/linkedin/final_post_from_brief.txt",
        ),
    )


def _candidate_payload() -> dict:
    return {
        "post_text": "Remote work needs explicit systems.",
        "hook_variants": ["Remote work fails quietly."],
        "cta_variants": ["What system changed remote work for your team?"],
        "hashtags": ["#RemoteWork"],
        "carousel_outline": [],
        "quality_checks": {
            "linkedin_ready": True,
            "uses_only_provided_facts": True,
            "has_clear_point_of_view": True,
        },
    }


def _selected_evidence() -> list[dict]:
    return [
        {
            "evidence_id": "a0-summary",
            "evidence_text": "Remote teams need clear operating agreements.",
            "role_in_post": "opening support",
        },
        {
            "evidence_id": "a1-kp0",
            "evidence_text": "Isolation can rise when remote work is unmanaged. cafГ©",
            "role_in_post": "practical tension",
        },
    ]


def _canonical_candidate_payload(candidate_payload: dict) -> dict:
    return {
        field_name: candidate_payload[field_name]
        for field_name in CANDIDATE_POST_PROMPT_FIELDS
        if field_name in candidate_payload
    }


def _prompt_selected_evidence(selected_evidence: list[dict]) -> list[dict]:
    return [
        {
            field_name: item[field_name]
            for field_name in SELECTED_EVIDENCE_PROMPT_FIELDS
            if field_name in item
        }
        for item in selected_evidence
    ]


def _rendered_quality_evaluator_text() -> str:
    render = render_quality_evaluator_prompt_input(
        _post_editorial_input(),
        get_quality_evaluator_rubric_payload(),
    )
    return "\n".join([*render.variables.values(), render.input_text])


def _repair_writer_render(
    *,
    original_candidate_payload=None,
    post_brief=None,
    angle_decision=None,
    selected_evidence=None,
    prompt_metadata=None,
) -> RepairWriterPromptRender:
    return render_repair_writer_prompt_input(
        original_candidate_payload=original_candidate_payload or _candidate_payload(),
        post_brief=post_brief or _post_editorial_input().post_brief,
        angle_decision=angle_decision or _post_editorial_input().angle_decision,
        selected_evidence=selected_evidence or _post_editorial_input().selected_evidence,
        deterministic_findings={
            "validation_passed": True,
            "repair_reasons": [],
        },
        quality_findings={
            "scores": {
                "human_voice": 3,
            },
            "failed_criteria": ["human_voice"],
        },
        repair_instruction={
            "repair_type": "editorial",
            "repair_instruction": "Make the post sound less generic.",
        },
        attempt_index=1,
        max_attempts=2,
        prompt_metadata=prompt_metadata,
    )


def _stable_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
