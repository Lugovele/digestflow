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

AUTHORIAL_PRESENCE_CALIBRATION_FIXTURES = (
    {
        "name": "article_like_judgment",
        "post_text": (
            "Bitcoin's current market story sounds cleaner than it is.\n\n"
            "ETF inflows and institutional interest point to broader adoption. "
            "But that story skips a step. Confidence, volatility, and policy "
            "risk are still separate questions.\n\n"
            "Treating one as evidence for the other is where the clean narrative "
            "breaks down.\n\n"
            "Which signal would you trust least?"
        ),
        "scores": {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 3,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        },
        "expected_author_point_of_view_max": 3,
        "failed_criteria": ["author_point_of_view"],
        "rationale": (
            "Strong article-like editorial ownership, but no explicit personal "
            "presence assigning the interpretation to the author."
        ),
    },
    {
        "name": "explicit_personal_presence",
        "post_text": (
            "Bitcoin's current market story sounds cleaner than it is.\n\n"
            "What bothers me is not the growth data itself. It is how quickly "
            "that data gets treated as proof that the underlying risk questions "
            "have been resolved.\n\n"
            "Adoption, confidence, volatility, and policy risk are answering "
            "different questions.\n\n"
            "Which signal would you trust least?"
        ),
        "scores": {
            "hook": 4,
            "controlling_angle": 4,
            "reader_problem": 4,
            "pattern_interrupt": 4,
            "evidence": 4,
            "author_point_of_view": 5,
            "human_voice": 4,
            "practical_value": 4,
            "cta": 4,
        },
        "expected_author_point_of_view": 5,
        "rationale": (
            "Explicit author-owned interpretive statement tied to supplied "
            "market signals without invented experience or authority."
        ),
    },
    {
        "name": "generic_first_person",
        "post_text": (
            "I think this is interesting. Bitcoin adoption is rising, but the "
            "market still has risks. Investors should watch the different "
            "signals carefully.\n\n"
            "Which signal would you trust least?"
        ),
        "scores": {
            "hook": 3,
            "controlling_angle": 3,
            "reader_problem": 3,
            "pattern_interrupt": 2,
            "evidence": 3,
            "author_point_of_view": 2,
            "human_voice": 2,
            "practical_value": 2,
            "cta": 4,
        },
        "failed_criteria": [
            "hook",
            "controlling_angle",
            "author_point_of_view",
            "human_voice",
        ],
        "rationale": "Generic first-person marker without substantive judgment.",
    },
    {
        "name": "fabricated_experience",
        "post_text": (
            "In my experience, investors always make this mistake. They see "
            "Bitcoin adoption rising and assume the remaining risk questions "
            "are solved.\n\n"
            "Which signal would you trust least?"
        ),
        "scores": {
            "hook": 3,
            "controlling_angle": 3,
            "reader_problem": 3,
            "pattern_interrupt": 3,
            "evidence": 2,
            "author_point_of_view": 2,
            "human_voice": 2,
            "practical_value": 2,
            "cta": 4,
        },
        "failed_criteria": [
            "hook",
            "controlling_angle",
            "evidence",
            "author_point_of_view",
            "human_voice",
        ],
        "automatic_fail_reason": "invents personal experience",
        "rationale": "Fabricated experience is forbidden unless supported by input.",
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
            "Ownership can rise while trust still lags. That is the useful "
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

    def test_article_like_required_mode_is_capped_at_author_pov_three(self) -> None:
        candidate = AUTHORIAL_PRESENCE_CALIBRATION_FIXTURES[0]

        normalized = normalize_quality_review_result(
            _review_fixture(
                candidate["post_text"],
                scores=candidate["scores"],
                failed_criteria=candidate["failed_criteria"],
            )
        )

        self.assertIs(normalized["pass"], False)
        self.assertLessEqual(
            normalized["scores"]["author_point_of_view"],
            candidate["expected_author_point_of_view_max"],
        )
        self.assertIn(
            "article-like editorial ownership",
            candidate["rationale"],
        )

    def test_explicit_personal_presence_may_qualify_for_author_pov_five(self) -> None:
        candidate = AUTHORIAL_PRESENCE_CALIBRATION_FIXTURES[1]

        normalized = normalize_quality_review_result(
            _review_fixture(
                candidate["post_text"],
                scores=candidate["scores"],
                passed=True,
            )
        )

        self.assertEqual(
            normalized["scores"]["author_point_of_view"],
            candidate["expected_author_point_of_view"],
        )
        self.assertIn(
            "What bothers me is not the growth data itself",
            candidate["post_text"],
        )

    def test_generic_first_person_does_not_qualify_as_strong_author_pov(self) -> None:
        candidate = AUTHORIAL_PRESENCE_CALIBRATION_FIXTURES[2]

        normalized = normalize_quality_review_result(
            _review_fixture(
                candidate["post_text"],
                scores=candidate["scores"],
                failed_criteria=candidate["failed_criteria"],
            )
        )

        self.assertLess(normalized["scores"]["author_point_of_view"], 4)
        self.assertIn("I think this is interesting", candidate["post_text"])

    def test_fabricated_experience_is_forbidden_or_materially_penalized(self) -> None:
        candidate = AUTHORIAL_PRESENCE_CALIBRATION_FIXTURES[3]

        normalized = normalize_quality_review_result(
            _review_fixture(
                candidate["post_text"],
                scores=candidate["scores"],
                failed_criteria=candidate["failed_criteria"],
                automatic_fail_reason=candidate["automatic_fail_reason"],
            )
        )

        self.assertIs(normalized["pass"], False)
        self.assertEqual(
            normalized["automatic_fail_reason"],
            "invents personal experience",
        )
        self.assertLess(normalized["scores"]["author_point_of_view"], 4)


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
