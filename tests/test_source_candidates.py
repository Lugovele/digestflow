from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource
from services.sources.candidates import (
    SourceCandidateInput,
    SourceCandidateStatus,
    evaluate_source_candidate,
    evaluate_source_candidates,
)


class SourceCandidateEvaluationTests(SimpleTestCase):
    def test_candidate_normalizes_url_and_hostname(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://www.Example.com/blog/post/?utm_source=newsletter",
                title="2026 family travel guide: budget checklist and practical planning steps",
                snippet="2026 practical family travel guide with budget planning, checklist, itinerary examples, tradeoffs, and implementation details.",
                readable_text_length=420,
            ),
            topic="Travel planning",
            focus_terms=("family travel",),
        )

        self.assertEqual(candidate.normalized_url, "https://example.com/blog/post")
        self.assertEqual(candidate.hostname, "example.com")
        self.assertEqual(candidate.candidate_type, "blog_index")
        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertTrue(candidate.diagnostics["quality_accepted"])

    def test_candidate_object_exposes_expected_fields(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/safe-sleep",
                title="Safe sleep for babies",
                snippet="Evidence-based safe sleep advice for infants.",
                origin_reason="manual fixture",
                readable_text_length=360,
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep",),
        )

        self.assertEqual(candidate.url, "https://example.com/safe-sleep")
        self.assertEqual(candidate.title, "Safe sleep for babies")
        self.assertEqual(candidate.origin_reason, "manual fixture")
        self.assertGreater(candidate.score, 0)
        self.assertIn("score_breakdown", candidate.diagnostics)
        self.assertIn("matched_terms", candidate.diagnostics)

    def test_weak_candidate_is_rejected_with_explicit_reason(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/baby-sleep-tips",
                title="Baby sleep tips",
                snippet="A short note.",
                readable_text_length=64,
            ),
            topic="Baby sleeping",
            focus_terms=("sleep tips",),
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.WEAK_CONTENT)
        self.assertIn("weak content (64 chars)", candidate.rejection_reasons)

    def test_unreachable_candidate_is_rejected_with_explicit_reason(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://missing.example/article",
                title="Pregnancy stretches",
                fetch_failure_reason="temporary failure in name resolution",
            ),
            topic="Physical exercises for pregnant women",
            focus_terms=("prenatal",),
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.UNREACHABLE)
        self.assertIn("temporary failure in name resolution", candidate.rejection_reasons)
        self.assertTrue(candidate.diagnostics["is_unreachable"])

    def test_invalid_url_candidate_is_classified_explicitly(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="not-a-valid-url",
                title="Broken input",
                snippet="No usable source here.",
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep",),
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.INVALID_URL)
        self.assertIn("invalid url", candidate.rejection_reasons)
        self.assertTrue(candidate.diagnostics["invalid_url"])

    def test_duplicate_url_is_detected(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/path?utm_source=mail",
                title="Travel planning basics",
                snippet="Practical travel checklist.",
                readable_text_length=240,
            ),
            topic="Travel planning",
            existing_normalized_urls={"https://example.com/path"},
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.DUPLICATE)
        self.assertIn("duplicate normalized url", candidate.rejection_reasons)
        self.assertTrue(candidate.diagnostics["duplicate_url"])

    def test_duplicate_hostname_is_downranked_for_review(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/another-sleep-guide",
                title="Bedtime routine for babies",
                snippet="Bedtime routine and night waking guidance.",
                readable_text_length=300,
            ),
            topic="Baby sleeping",
            focus_terms=("bedtime routine",),
            existing_hostnames={"example.com"},
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.NEEDS_REVIEW)
        self.assertIn("duplicate hostname", candidate.rejection_reasons)
        self.assertTrue(candidate.diagnostics["duplicate_hostname"])

    def test_relevant_candidate_is_accepted(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/prenatal-low-impact-exercises",
                title="2026 low-impact prenatal exercises for the third trimester: safety guidelines and methodology",
                snippet="Recent evidence-based guidance with safety guidelines, methodology, limitations, and practical low-impact exercise examples for late pregnancy.",
                readable_text_length=520,
            ),
            topic="Physical exercises for pregnant women",
            focus_terms=("third trimester", "low impact"),
        )

        self.assertEqual(candidate.status, SourceCandidateStatus.ACCEPTED)
        self.assertGreater(candidate.score, 40)
        self.assertIn("low impact", candidate.diagnostics["matched_terms"])
        self.assertTrue(candidate.diagnostics["quality_accepted"])

    def test_accepted_candidates_are_sorted_by_score(self) -> None:
        candidates = evaluate_source_candidates(
            [
                SourceCandidateInput(
                    url="https://example.com/broad-overview",
                    title="2026 family travel planning guide with concrete budget steps",
                    snippet="Recent practical guide with concrete budget steps, packing checklist, booking tradeoffs, and itinerary examples for family travel.",
                    readable_text_length=360,
                ),
                SourceCandidateInput(
                    url="https://example.org/family-travel-checklist",
                    title="2026 family travel budget planning report: checklist, methodology, case study, and tradeoffs",
                    snippet="Recent family travel report with budget methodology, checklist steps, case study examples, limitations, and tradeoffs.",
                    readable_text_length=520,
                ),
            ],
            topic="Travel planning",
            focus_terms=("family travel", "budget travel"),
        )

        accepted = [candidate for candidate in candidates if candidate.status == SourceCandidateStatus.ACCEPTED]
        self.assertEqual(len(accepted), 2)
        self.assertGreaterEqual(accepted[0].score, accepted[1].score)
        self.assertEqual(accepted[0].normalized_url, "https://example.org/family-travel-checklist")

    def test_diagnostics_include_future_debug_context(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/teen-education-tools",
                title="Education tools for teenagers",
                snippet="Study habits, online learning tools, and teacher guidance.",
                readable_text_length=410,
            ),
            topic="Education for teenagers",
            focus_terms=("study habits", "online learning"),
        )

        self.assertEqual(candidate.diagnostics["status"], "accepted")
        self.assertEqual(candidate.diagnostics["hostname"], "example.com")
        self.assertIn("score_breakdown", candidate.diagnostics)
        self.assertIn("topic_terms", candidate.diagnostics)
        self.assertIn("matched_terms", candidate.diagnostics)


class SourceCandidateEvaluationPersistenceTests(TestCase):
    def test_evaluation_does_not_create_topic_sources_implicitly(self) -> None:
        user = get_user_model().objects.create_user(username="candidate-eval-user", password="pw")
        Topic.objects.create(user=user, name="Candidate evaluation topic")

        self.assertEqual(TopicSource.objects.count(), 0)

        result = evaluate_source_candidates(
            [
                SourceCandidateInput(
                    url="https://example.com/infant-sleep-safety",
                    title="Infant sleep safety",
                    snippet="Safe sleep guidance for babies.",
                    readable_text_length=330,
                )
            ],
            topic="Baby sleeping",
            focus_terms=("safe sleep",),
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(TopicSource.objects.count(), 0)
