from __future__ import annotations

import inspect
import json
from pathlib import Path

from django.test import SimpleTestCase

from services.packaging import linkedin_post_genericization_guard
from services.packaging.linkedin_post_genericization_guard import (
    SIGNAL_WEAK_DISTINCTIVE_SENTENCE_SHAPE,
    evaluate_genericization_selection_blocker,
)
from services.packaging.postflow_product_validation_corpus import (
    load_product_validation_corpus_manifest,
)


BLOCKER_CASE_IDS = (
    "topic_140_digest_126",
    "topic_200_digest_134__gpt_v2",
    "candidate_quality:topic_214_digest_128__generic",
)

STRONG_CONTROL_CASE_IDS = (
    "candidate_quality:topic_140_digest_126__strong",
    "candidate_quality:topic_200_digest_134__strong",
)

LOCAL_REPAIR_CONTROL_CASE_IDS = (
    "candidate_quality:topic_214_digest_128__remote_cta_repair_local",
)

REPAIRED_CRYPTO_RECOVERY_LIVE_V5_POST_TEXT = (
    "Crypto's headline numbers--30% of Americans own some, and nearly 17% CAGR "
    "is forecast through 2035--sound conclusive. But treating these adoption "
    "stats or growth projections as a settled story misses what actually shapes "
    "the crypto market. What stands out is the underlying fragility: yes, public "
    "interest and policy moves generate real momentum, but consistent issues "
    "like security concerns, persistent volatility, and traders' caution keep "
    "broader adoption and lasting confidence just out of reach. Before treating "
    "the market story as settled, are you checking whether growth evidence is "
    "being qualified by underlying confidence and risk evidence?"
)


class LinkedInPostGenericizationGuardTests(SimpleTestCase):
    def test_material_generic_blocker_fixtures_are_blocked(self) -> None:
        for case_id in BLOCKER_CASE_IDS:
            with self.subTest(case_id=case_id):
                result = evaluate_genericization_selection_blocker(
                    _candidate_post_text(case_id)
                )

                self.assertTrue(result.blocked)
                self.assertIn(SIGNAL_WEAK_DISTINCTIVE_SENTENCE_SHAPE, result.signals)
                self.assertIn("material generic/template-like prose", result.reason)

    def test_material_generic_blocker_fixture_set_spans_three_independence_groups(self) -> None:
        cases = tuple(_case_by_id(case_id) for case_id in BLOCKER_CASE_IDS)
        independence_groups = {case.independence_group for case in cases}

        self.assertEqual(len(independence_groups), 3)
        for case in cases:
            self.assertEqual(case.human_ground_truth["final_disposition"], "NOT_READY")
            self.assertEqual(case.human_ground_truth["genericization"], "MATERIAL")

    def test_repaired_crypto_recovery_live_v5_text_is_blocked(self) -> None:
        result = evaluate_genericization_selection_blocker(
            REPAIRED_CRYPTO_RECOVERY_LIVE_V5_POST_TEXT
        )

        self.assertTrue(result.blocked)

    def test_strong_control_fixtures_are_not_blocked(self) -> None:
        for case_id in STRONG_CONTROL_CASE_IDS:
            with self.subTest(case_id=case_id):
                result = evaluate_genericization_selection_blocker(
                    _candidate_post_text(case_id)
                )

                self.assertFalse(result.blocked)

    def test_topic_214_local_repair_control_is_not_blocked(self) -> None:
        for case_id in LOCAL_REPAIR_CONTROL_CASE_IDS:
            with self.subTest(case_id=case_id):
                result = evaluate_genericization_selection_blocker(
                    _candidate_post_text(case_id)
                )

                self.assertFalse(result.blocked)

    def test_first_person_native_repair_control_is_not_blocked(self) -> None:
        result = evaluate_genericization_selection_blocker(
            _candidate_post_text(
                "candidate_quality:topic_140_digest_126__native_repair_voice_preserved"
            )
        )

        self.assertFalse(result.blocked)

    def test_single_generic_phrase_is_not_enough_to_block(self) -> None:
        result = evaluate_genericization_selection_blocker(
            "Before treating the story as settled, ask what evidence changed. "
            "The evidence is narrow, so the decision should stay narrow."
        )

        self.assertFalse(result.blocked)

    def test_guard_ignores_human_labels_case_ids_and_hashes(self) -> None:
        post_text = _candidate_post_text("topic_140_digest_126")
        baseline = evaluate_genericization_selection_blocker(post_text)
        decorated_runtime_record = {
            "case_id": "candidate_quality:topic_140_digest_126__strong",
            "human_ground_truth": {"genericization": "NONE"},
            "candidate_text_hash": "not-used",
            "candidate_payload": {"post_text": post_text},
        }

        decorated = evaluate_genericization_selection_blocker(
            decorated_runtime_record["candidate_payload"]["post_text"]
        )

        self.assertEqual(decorated, baseline)

    def test_output_is_json_serializable(self) -> None:
        result = evaluate_genericization_selection_blocker(
            _candidate_post_text("topic_200_digest_134__gpt_v2")
        )

        serialized = json.dumps(result.to_dict(), sort_keys=True)

        self.assertIn("signals", serialized)

    def test_guard_module_has_no_provider_runtime_or_benchmark_label_dependencies(self) -> None:
        source = inspect.getsource(linkedin_post_genericization_guard)

        self.assertNotIn("OpenAIClient", source)
        self.assertNotIn("execute_", source)
        self.assertNotIn("human_ground_truth", source)
        self.assertNotIn("case_id", source)
        self.assertNotIn("fixture", source)
        self.assertNotIn("django.db", source)
        self.assertNotIn("ContentPackage", source)


def _candidate_post_text(case_id: str) -> str:
    case = _case_by_id(case_id)
    payload = json.loads(Path(case.fixture_path).read_text(encoding="utf-8"))
    return payload["candidate_payload"]["post_text"]


def _case_by_id(case_id: str):
    manifest = load_product_validation_corpus_manifest()
    for case in manifest.cases:
        if case.case_id == case_id:
            return case
    raise AssertionError(f"Missing product validation case {case_id}")
