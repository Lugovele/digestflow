from __future__ import annotations

import ast
import copy
import inspect
import json
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase
from django.test import override_settings

from services.packaging import linkedin_post_candidate_writer_execution
from services.packaging.linkedin_post_candidate_writer_execution import (
    CandidateWriterExecutionRequest,
    CandidateWriterRawResponse,
    build_candidate_writer_execution_request,
    execute_candidate_writer_prompt,
)
from services.packaging.linkedin_post_editorial_boundary import PromptMetadata
from services.packaging.linkedin_post_prompt_renderers import (
    CandidateWriterPromptRender,
)


class CandidateWriterExecutionTests(SimpleTestCase):
    def test_execution_request_construction_uses_postflow_post_settings(self) -> None:
        with override_settings(
            POSTFLOW_POST_PROVIDER="openai",
            POSTFLOW_POST_MODEL="candidate-model",
        ):
            request = build_candidate_writer_execution_request(
                _render(),
                prompt_text="Candidate writer prompt.",
                execution_metadata={"attempt": 1},
            )

        self.assertIsInstance(request, CandidateWriterExecutionRequest)
        self.assertEqual(request.provider, "openai")
        self.assertEqual(request.model, "candidate-model")
        self.assertEqual(request.execution_metadata, {"attempt": 1})

    def test_execution_request_to_dict_defensively_copies_metadata(self) -> None:
        request = _request(execution_metadata={"attempt": {"index": 1}})

        serialized = request.to_dict()
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(request.execution_metadata, {"attempt": {"index": 1}})

    def test_raw_response_to_dict_preserves_raw_text_and_metadata(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text='{"post_text": "Draft"}',
            provider="openai",
            model="candidate-model",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 12},
            raw_provider_response={"id": "resp_1"},
            execution_metadata={"attempt": {"index": 1}},
        )

        serialized = response.to_dict()
        serialized["usage"]["total_tokens"] = 99
        serialized["raw_provider_response"]["id"] = "changed"
        serialized["execution_metadata"]["attempt"]["index"] = 2

        self.assertEqual(serialized["raw_text"], '{"post_text": "Draft"}')
        self.assertEqual(response.usage, {"total_tokens": 12})
        self.assertEqual(response.raw_provider_response, {"id": "resp_1"})
        self.assertEqual(response.execution_metadata, {"attempt": {"index": 1}})

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_successful_execution_calls_provider_once_and_returns_raw_response(
        self,
        mock_openai_client,
    ) -> None:
        request = _request()
        provider_response = SimpleNamespace(
            text='{"post_text": "Candidate post"}',
            raw={"id": "resp_123", "status": "completed"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        )
        mock_openai_client.return_value.generate_text.return_value = provider_response

        raw_response = execute_candidate_writer_prompt(request)

        mock_openai_client.assert_called_once_with(model="candidate-model")
        mock_openai_client.return_value.generate_text.assert_called_once_with(
            prompt=f"{request.prompt_text}\n\n{request.rendered_prompt_input.input_text}",
            max_output_tokens=request.max_output_tokens,
            json_mode=False,
        )
        self.assertEqual(raw_response.raw_text, provider_response.text)
        self.assertEqual(raw_response.provider, "openai")
        self.assertEqual(raw_response.model, "candidate-model")
        self.assertEqual(raw_response.usage, provider_response.usage)
        self.assertEqual(raw_response.raw_provider_response, provider_response.raw)
        self.assertIsNone(raw_response.execution_error)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_execution_forwards_prompt_text_and_input_text_unchanged(
        self,
        mock_openai_client,
    ) -> None:
        unicode_word = (
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447"
            "\u043d\u043e\u0441\u0442\u044c"
        )
        render = _render(
            input_text=f"## POST_BRIEF_JSON\nUnicode: {unicode_word}\n```json\n{{}}\n```"
        )
        prompt_text = "Resolved candidate prompt text.\nDo not change this."
        request = _request(render=render, prompt_text=prompt_text)
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_candidate_writer_prompt(request)

        called_prompt = mock_openai_client.return_value.generate_text.call_args.kwargs[
            "prompt"
        ]
        self.assertEqual(called_prompt, f"{prompt_text}\n\n{render.input_text}")

    @override_settings(OPENAI_API_KEY="test-key", OPENAI_TIMEOUT_SECONDS=30)
    @patch("apps.ai.client.OpenAI")
    def test_provider_failure_is_not_retried_by_json_mode_fallback(
        self,
        mock_openai,
    ) -> None:
        request = _request()
        create_calls = []

        def create_response(**kwargs):
            create_calls.append(kwargs)
            if len(create_calls) == 1:
                raise RuntimeError("json mode unavailable")
            return SimpleNamespace(
                output_text="fallback success",
                model_dump=lambda: {"id": "retried"},
                usage=SimpleNamespace(
                    input_tokens=1,
                    output_tokens=1,
                    total_tokens=2,
                ),
            )

        mock_openai.return_value.responses.create.side_effect = create_response

        raw_response = execute_candidate_writer_prompt(request)

        self.assertEqual(len(create_calls), 1)
        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("text", create_calls[0])

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_execution_does_not_mutate_render_request_or_metadata(
        self,
        mock_openai_client,
    ) -> None:
        render = _render()
        request = _request(render=render, execution_metadata={"attempt": {"index": 1}})
        render_before = copy.deepcopy(render)
        request_before = copy.deepcopy(request)
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text="Raw output",
            raw={},
            usage={},
        )

        execute_candidate_writer_prompt(request)

        self.assertEqual(render, render_before)
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_empty_response_returns_execution_error_without_parsing(
        self,
        mock_openai_client,
    ) -> None:
        empty_values = (None, "", "   ", "\r\n")

        for empty_value in empty_values:
            with self.subTest(raw_text=empty_value):
                mock_openai_client.return_value.generate_text.return_value = (
                    SimpleNamespace(
                        text=empty_value,
                        raw={"id": "empty"},
                        usage={"total_tokens": 3},
                    )
                )

                raw_response = execute_candidate_writer_prompt(_request())

                self.assertEqual(raw_response.raw_text, str(empty_value or ""))
                self.assertEqual(
                    raw_response.execution_error,
                    "empty provider response",
                )
                self.assertEqual(raw_response.raw_provider_response, {"id": "empty"})

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_provider_failure_returns_sanitized_execution_error(
        self,
        mock_openai_client,
    ) -> None:
        request = _request()
        request_before = copy.deepcopy(request)
        mock_openai_client.return_value.generate_text.side_effect = RuntimeError(
            "secret provider details"
        )

        raw_response = execute_candidate_writer_prompt(request)

        self.assertEqual(raw_response.raw_text, "")
        self.assertEqual(raw_response.execution_error, "provider invocation failed")
        self.assertNotIn("secret provider details", raw_response.to_dict().values())
        self.assertEqual(request, request_before)

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_unsupported_provider_returns_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(provider="gemini", model="candidate-model")
        )

        self.assertEqual(
            raw_response.execution_error,
            "unsupported candidate writer provider: gemini",
        )
        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_missing_provider_returns_distinct_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(provider="", model="candidate-model")
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer provider",
        )
        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_invalid_max_output_tokens_returns_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        invalid_values = (True, False, "1200", None, 0, -1)

        for invalid_value in invalid_values:
            with self.subTest(max_output_tokens=invalid_value):
                raw_response = execute_candidate_writer_prompt(
                    _request(max_output_tokens=invalid_value)
                )

                self.assertEqual(
                    raw_response.execution_error,
                    "invalid candidate writer max_output_tokens: "
                    "must be a positive integer",
                )
                self.assertEqual(raw_response.raw_text, "")

        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_missing_prompt_content_returns_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(_request(prompt_text=" "))

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer prompt text",
        )
        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_blank_rendered_input_text_returns_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(
            _request(render=_render(input_text=" \n\t "))
        )

        self.assertEqual(
            raw_response.execution_error,
            "missing candidate writer rendered input text",
        )
        self.assertEqual(raw_response.raw_text, "")
        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_missing_model_returns_execution_error_without_provider_call(
        self,
        mock_openai_client,
    ) -> None:
        raw_response = execute_candidate_writer_prompt(_request(model=""))

        self.assertEqual(raw_response.execution_error, "missing candidate writer model")
        mock_openai_client.assert_not_called()

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_raw_markdown_fenced_json_is_preserved_unparsed(
        self,
        mock_openai_client,
    ) -> None:
        raw_text = '```json\n{"post_text": "Candidate"}\n```'
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_candidate_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)
        self.assertNotIn("payload", raw_response.to_dict())

    @patch("services.packaging.linkedin_post_candidate_writer_execution.OpenAIClient")
    def test_unicode_raw_response_is_preserved(
        self,
        mock_openai_client,
    ) -> None:
        raw_text = (
            '{"post_text": "'
            "\u0447\u0435\u043b\u043e\u0432\u0435\u0447\u043d\u043e\u0441\u0442\u044c "
            "\u0442\u0435\u043a\u0441\u0442\u0430"
            '"}'
        )
        mock_openai_client.return_value.generate_text.return_value = SimpleNamespace(
            text=raw_text,
            raw={},
            usage={},
        )

        raw_response = execute_candidate_writer_prompt(_request())

        self.assertEqual(raw_response.raw_text, raw_text)

    def test_raw_response_to_dict_omits_unavailable_optional_metadata(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text="Raw",
            provider="openai",
            model="candidate-model",
        )

        serialized = response.to_dict()

        self.assertEqual(set(serialized), {"raw_text", "provider", "model"})

    def test_response_to_dict_is_json_serializable(self) -> None:
        response = CandidateWriterRawResponse(
            raw_text='{"post_text": "Candidate"}',
            provider="openai",
            model="candidate-model",
            prompt_metadata=_prompt_metadata(),
            usage={"total_tokens": 10},
            raw_provider_response={"id": "resp"},
            execution_metadata={"attempt": 1},
        )

        serialized = json.dumps(response.to_dict(), sort_keys=True)

        self.assertIn("candidate-model", serialized)

    def test_execution_module_has_no_parser_gate_decision_repair_or_runtime_dependencies(
        self,
    ) -> None:
        tree = ast.parse(inspect.getsource(linkedin_post_candidate_writer_execution))
        imported_modules = _imported_modules(tree)
        imported_symbols = _imported_symbols(tree)

        forbidden_modules = {
            "apps.packaging.models",
            "django.db",
            "services.packaging.generator",
            "services.packaging.linkedin_post_flow_handoffs",
            "services.packaging.linkedin_post_pipeline",
            "services.packaging.linkedin_post_quality_evaluator",
        }
        forbidden_symbols = {
            "CandidateWriterOutput",
            "ContentPackage",
            "FinalPostDecisionController",
            "FinalPostPayload",
            "QualityEvaluator",
            "RepairAgent",
            "TargetedRepairPlan",
            "generate_content_package_for_digest",
            "run_final_post_deterministic_gate",
            "validate_final_post_payload",
        }

        self.assertTrue(forbidden_modules.isdisjoint(imported_modules))
        self.assertTrue(forbidden_symbols.isdisjoint(imported_symbols))


def _request(
    *,
    render: CandidateWriterPromptRender | None = None,
    prompt_text: str = "Candidate writer prompt.",
    provider: str = "openai",
    model: str = "candidate-model",
    max_output_tokens: object = 1200,
    execution_metadata: dict | None = None,
) -> CandidateWriterExecutionRequest:
    return CandidateWriterExecutionRequest(
        rendered_prompt_input=render or _render(),
        prompt_text=prompt_text,
        provider=provider,
        model=model,
        max_output_tokens=max_output_tokens,
        execution_metadata=execution_metadata,
    )


def _render(
    *,
    input_text: str = "## CANDIDATE_WRITER_INPUT_JSON\n{}",
) -> CandidateWriterPromptRender:
    return CandidateWriterPromptRender(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
        variables={
            "post_brief_json": "{}",
            "angle_decision_json": "{}",
            "selected_evidence_json": "[]",
            "candidate_writer_input_json": "{}",
        },
        input_text=input_text,
    )


def _prompt_metadata() -> PromptMetadata:
    return PromptMetadata(
        prompt_name="final_post_candidate_from_brief",
        prompt_version="1.0",
        prompt_path="prompts/linkedin/final_post_from_brief.txt",
    )


def _imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
    return modules


def _imported_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            symbols.update(
                alias.asname or alias.name.partition(".")[0] for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            symbols.update(alias.asname or alias.name for alias in node.names)
    return symbols
