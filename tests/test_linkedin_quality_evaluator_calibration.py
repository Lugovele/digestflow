from __future__ import annotations

from django.test import SimpleTestCase

from services.packaging.linkedin_post_quality_review_contract import (
    CANONICAL_QUALITY_SCORE_KEYS,
    normalize_quality_review_result,
)


WEAK_BITCOIN_CANDIDATES = (
    {
        "name": "process_language_summary",
        "post_text": (
            "Cryptocurrency ownership in the U.S. is significant, with nearly "
            "a third of Americans now owning crypto. Attaching individual "
            "claims to their evidence matters when reading this market."
        ),
        "scores": {
            "hook": 3,
            "controlling_angle": 2,
            "reader_problem": 2,
            "pattern_interrupt": 3,
            "evidence": 2,
            "author_point_of_view": 2,
            "human_voice": 2,
            "practical_value": 2,
            "cta": 1,
        },
        "failed_criteria": [
            "hook",
            "controlling_angle",
            "evidence",
            "author_point_of_view",
            "human_voice",
        ],
        "automatic_fail_reason": (
            "leaks internal process language into reader-facing text"
        ),
    },
    {
        "name": "inventory_style_synthesis",
        "post_text": (
            "It’s easy to frame the crypto market as a single story—usually "
            "centered on surging adoption or dramatic price action. These "
            "signals—adoption trends, security confidence, forward market "
            "forecasts, and pricing—don’t always move in sync."
        ),
        "scores": {
            "hook": 3,
            "controlling_angle": 3,
            "reader_problem": 3,
            "pattern_interrupt": 3,
            "evidence": 3,
            "author_point_of_view": 3,
            "human_voice": 3,
            "practical_value": 2,
            "cta": 1,
        },
        "failed_criteria": [
            "hook",
            "controlling_angle",
            "author_point_of_view",
            "human_voice",
        ],
        "automatic_fail_reason": "has no required CTA in post_text",
    },
    {
        "name": "causal_strengthening_generic_voice",
        "post_text": (
            "Crypto’s story looks like a clear-cut growth narrative—with 30% "
            "of Americans now owning cryptocurrencies and a market projected "
            "to nearly 17% annual growth through the next decade. Growth "
            "forecasts signal potential, but true mainstream acceptance will "
            "hinge on resolving security and market stability challenges."
        ),
        "scores": {
            "hook": 3,
            "controlling_angle": 3,
            "reader_problem": 3,
            "pattern_interrupt": 3,
            "evidence": 2,
            "author_point_of_view": 3,
            "human_voice": 2,
            "practical_value": 2,
            "cta": 1,
        },
        "failed_criteria": [
            "hook",
            "controlling_angle",
            "evidence",
            "author_point_of_view",
            "human_voice",
        ],
        "automatic_fail_reason": "uses unsupported causal strengthening",
    },
)


# These fixtures calibrate the deterministic review contract and parser surface.
# They intentionally do not claim a live provider will assign identical scores.
class LinkedInQualityEvaluatorCalibrationTests(SimpleTestCase):
    def test_historical_weak_bitcoin_outputs_have_expected_offline_fail_reviews(
        self,
    ) -> None:
        for candidate in WEAK_BITCOIN_CANDIDATES:
            with self.subTest(candidate=candidate["name"]):
                review = _review_fixture(
                    candidate["post_text"],
                    scores=candidate["scores"],
                    failed_criteria=candidate["failed_criteria"],
                    automatic_fail_reason=candidate["automatic_fail_reason"],
                )

                normalized = normalize_quality_review_result(review)

                self.assertIs(normalized["pass"], False)
                self.assertLess(normalized["total_score"], 36)
                self.assertEqual(
                    normalized["automatic_fail_reason"],
                    candidate["automatic_fail_reason"],
                )
                self.assertEqual(normalized["scores"]["cta"], 1)
                self.assertIn("human_voice", normalized["failed_criteria"])
                self.assertIn("criterion_rationales", normalized)

    def test_high_total_score_cannot_override_automatic_fail_for_weak_fixture(
        self,
    ) -> None:
        candidate = WEAK_BITCOIN_CANDIDATES[0]
        scores = {criterion: 5 for criterion in CANONICAL_QUALITY_SCORE_KEYS}

        with self.assertRaisesRegex(
            ValueError,
            "pass cannot be true when automatic_fail_reason is set",
        ):
            normalize_quality_review_result(
                _review_fixture(
                    candidate["post_text"],
                    scores=scores,
                    passed=True,
                    failed_criteria=[],
                    automatic_fail_reason=candidate["automatic_fail_reason"],
                )
            )

    def test_missing_cta_fixture_keeps_cta_at_one(self) -> None:
        candidate = WEAK_BITCOIN_CANDIDATES[1]
        normalized = normalize_quality_review_result(
            _review_fixture(
                candidate["post_text"],
                scores=candidate["scores"],
                failed_criteria=candidate["failed_criteria"],
                automatic_fail_reason=candidate["automatic_fail_reason"],
            )
        )

        self.assertEqual(normalized["scores"]["cta"], 1)
        self.assertIn("CTA absent", normalized["criterion_rationales"]["cta"]["failure_reason"])

    def test_strong_grounded_post_with_cta_can_pass_offline_contract(self) -> None:
        post_text = (
            "Crypto adoption is not one signal.\n\n"
            "To me, ownership can rise while trust still lags. That is the useful "
            "read: growth forecasts, security worries, and market sentiment "
            "need to be judged separately before anyone calls the trend "
            "mainstream.\n\n"
            "Which signal would you trust least when reading crypto momentum?"
        )
        scores = {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        }

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertIs(normalized["pass"], True)
        self.assertEqual(normalized["total_score"], 36)
        self.assertEqual(normalized["failed_criteria"], [])
        self.assertEqual(normalized["automatic_fail_reason"], "")

    def test_article_like_judgment_is_capped_below_author_point_of_view_five(
        self,
    ) -> None:
        post_text = (
            "But that story skips a step.\n\n"
            "Treating one as evidence for the other is where the clean narrative "
            "breaks down.\n\n"
            "Which signal would you test before calling the trend durable?"
        )
        scores = {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 4,
            "human_voice": 5,
            "practical_value": 4,
            "cta": 4,
        }

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertLessEqual(normalized["scores"]["author_point_of_view"], 4)
        self.assertEqual(normalized["scores"]["human_voice"], 5)

    def test_explicit_personal_presence_may_receive_author_point_of_view_five(
        self,
    ) -> None:
        post_text = (
            "What bothers me is not the growth data itself. It is how quickly "
            "that data gets treated as proof that the underlying risk questions "
            "have been resolved.\n\n"
            "Which signal would you separate first?"
        )
        scores = {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 5,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        }

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertEqual(normalized["scores"]["author_point_of_view"], 5)
        self.assertIs(normalized["pass"], True)

    def test_crypto_style_one_marker_satisfies_explicit_author_presence(
        self,
    ) -> None:
        post_text = (
            "Crypto adoption is not the same as crypto confidence. "
            "I would not treat these signals as proof that the market's "
            "underlying risk landscape has been resolved. "
            "Growth evidence does not erase risk; it changes which signal "
            "needs to be tested first.\n\n"
            "Which signal would you separate before calling this durable?"
        )
        scores = _passing_scores(author_point_of_view=5, cta=5)

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertEqual(normalized["scores"]["author_point_of_view"], 5)
        self.assertNotIn("author_point_of_view", normalized["failed_criteria"])
        self.assertEqual(normalized["automatic_fail_reason"], "")
        self.assertIs(normalized["pass"], True)

    def test_generic_first_person_remains_weak_author_point_of_view(self) -> None:
        post_text = (
            "I think this is interesting.\n\n"
            "Crypto adoption and risk signals should be considered together.\n\n"
            "What do you think?"
        )
        scores = {
            "hook": 2,
            "controlling_angle": 3,
            "reader_problem": 3,
            "pattern_interrupt": 2,
            "evidence": 3,
            "author_point_of_view": 3,
            "human_voice": 3,
            "practical_value": 2,
            "cta": 3,
        }

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=[
                    "hook",
                    "controlling_angle",
                    "author_point_of_view",
                    "human_voice",
                    "practical_value",
                ],
            )
        )

        self.assertLessEqual(normalized["scores"]["author_point_of_view"], 3)
        self.assertIn("author_point_of_view", normalized["failed_criteria"])

    def test_fabricated_experience_is_penalized(self) -> None:
        post_text = (
            "In my experience, investors always make this mistake.\n\n"
            "They treat ownership growth as proof that risk has disappeared.\n\n"
            "Which signal would you test first?"
        )
        scores = {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 1,
            "author_point_of_view": 2,
            "human_voice": 3,
            "practical_value": 3,
            "cta": 4,
        }

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=[
                    "evidence",
                    "author_point_of_view",
                    "human_voice",
                    "practical_value",
                ],
                automatic_fail_reason="invents personal experience",
            )
        )

        self.assertIs(normalized["pass"], False)
        self.assertEqual(
            normalized["automatic_fail_reason"],
            "invents personal experience",
        )

    def test_one_explicit_ownership_marker_with_ordinary_analysis_can_receive_author_pov_five(
        self,
    ) -> None:
        post_text = (
            "I'd reject the reading that frequent use settles the question. "
            "That distinction matters because usage measures exposure, not learning. "
            "Adoption is not the same as impact.\n\n"
            "Which signal would you test before calling this durable?"
        )
        scores = _passing_scores(author_point_of_view=5)

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertEqual(normalized["scores"]["author_point_of_view"], 5)
        self.assertIs(normalized["pass"], True)
        self.assertEqual(normalized["automatic_fail_reason"], "")

    def test_two_independent_explicit_ownership_markers_remain_below_author_pov_five(
        self,
    ) -> None:
        post_text = (
            "I don't think a solid policy proves the harder work is done. "
            "My reading is that clear rules still need inclusive practice.\n\n"
            "Which assumption would you test first?"
        )
        scores = _passing_scores(author_point_of_view=4)

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=["author_point_of_view"],
            )
        )

        self.assertLessEqual(normalized["scores"]["author_point_of_view"], 4)
        self.assertIs(normalized["pass"], False)
        self.assertEqual(normalized["automatic_fail_reason"], "")
        self.assertIn("author_point_of_view", normalized["failed_criteria"])

    def test_one_explicit_ownership_marker_plus_cta_does_not_create_second_marker(
        self,
    ) -> None:
        post_text = (
            "I would treat this as an integration problem, not an adoption win. "
            "Clear rules are not inclusive design.\n\n"
            "What would you check before calling the policy mature?"
        )
        scores = _passing_scores(author_point_of_view=5, cta=5)

        normalized = normalize_quality_review_result(
            _review_fixture(post_text, scores=scores, passed=True)
        )

        self.assertEqual(normalized["scores"]["author_point_of_view"], 5)
        self.assertEqual(normalized["scores"]["cta"], 5)
        self.assertIs(normalized["pass"], True)

    def test_non_first_person_crypto_analysis_does_not_count_as_multiple_explicit_markers(
        self,
    ) -> None:
        post_text = (
            "Growth evidence does not erase risk. Distinguishing between "
            "momentum and genuine confidence is essential for seeing the real "
            "picture.\n\n"
            "Which signal would you separate first?"
        )
        scores = _passing_scores(author_point_of_view=3)

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=["author_point_of_view"],
            )
        )

        self.assertLessEqual(normalized["scores"]["author_point_of_view"], 3)
        self.assertIs(normalized["pass"], False)
        self.assertEqual(normalized["automatic_fail_reason"], "")
        self.assertIn("author_point_of_view", normalized["failed_criteria"])

    def test_generic_first_person_scaffolding_without_interpretation_cannot_receive_five(
        self,
    ) -> None:
        post_text = (
            "In my view, this is important. I believe leaders should pay "
            "attention.\n\nWhat do you think?"
        )
        scores = _passing_scores(author_point_of_view=3, human_voice=3, practical_value=2)

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=[
                    "author_point_of_view",
                    "human_voice",
                    "practical_value",
                ],
            )
        )

        self.assertLessEqual(normalized["scores"]["author_point_of_view"], 3)
        self.assertIs(normalized["pass"], False)

    def test_unsupported_personal_experience_cannot_receive_author_pov_five(
        self,
    ) -> None:
        post_text = (
            "In my experience advising teams, this always fails the same way. "
            "Clear rules are not inclusive design.\n\n"
            "What would you test first?"
        )
        scores = _passing_scores(evidence=1, author_point_of_view=2, human_voice=3)

        normalized = normalize_quality_review_result(
            _review_fixture(
                post_text,
                scores=scores,
                failed_criteria=[
                    "evidence",
                    "author_point_of_view",
                    "human_voice",
                ],
                automatic_fail_reason="invents personal experience",
            )
        )

        self.assertIs(normalized["pass"], False)
        self.assertEqual(
            normalized["automatic_fail_reason"],
            "invents personal experience",
        )


def _passing_scores(**overrides: int) -> dict[str, int]:
    scores = {criterion: 4 for criterion in CANONICAL_QUALITY_SCORE_KEYS}
    scores["human_voice"] = 5
    scores.update(overrides)
    return scores


def _review_fixture(
    post_text: str,
    *,
    scores: dict[str, int],
    passed: bool = False,
    failed_criteria: list[str] | None = None,
    automatic_fail_reason: str = "",
) -> dict:
    return {
        "scores": scores,
        "total_score": sum(scores.values()),
        "pass": passed,
        "failed_criteria": failed_criteria or [],
        "automatic_fail_reason": automatic_fail_reason,
        "criterion_rationales": {
            criterion: {
                "score": score,
                "max_score": 5,
                "rationale": f"{criterion} score is tied to the fixed candidate.",
                "post_text_evidence": post_text[:80],
                "failure_reason": (
                    "CTA absent from actual post_text."
                    if criterion == "cta" and score == 1
                    else ""
                ),
            }
            for criterion, score in scores.items()
        },
        "notes": ["Fixed offline calibration fixture."],
        "requires_human_review": False,
        "human_review_reason": "",
    }
