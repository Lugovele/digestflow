from django.contrib.auth import get_user_model
from django.test import SimpleTestCase, TestCase

from apps.topics.models import Topic, TopicSource
from services.sources.candidate_review import (
    build_candidate_review_item,
    build_candidate_review_items,
)
from services.sources.candidates import (
    SourceCandidateInput,
    SourceCandidateStatus,
    evaluate_source_candidate,
)


class SourceCandidateReviewAdapterTests(SimpleTestCase):
    def test_accepted_candidate_becomes_selectable_review_item(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://www.hopkinsmedicine.org/health/wellness-and-prevention/infant-safe-sleep",
                title="Infant Safe Sleep: Reducing SIDS Risk",
                snippet="Pediatric safe sleep guidance for babies and SIDS prevention.",
                readable_text_length=540,
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep", "SIDS"),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.status, SourceCandidateStatus.ACCEPTED)
        self.assertTrue(item.is_selectable)
        self.assertTrue(item.can_be_persisted)
        self.assertTrue(item.default_selected)

    def test_rejected_candidate_becomes_non_selectable_review_item(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/",
                title="Home",
                snippet="Welcome.",
                readable_text_length=24,
            ),
            topic="Travel planning",
            focus_terms=("family travel",),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.status, SourceCandidateStatus.WEAK_CONTENT)
        self.assertFalse(item.is_selectable)
        self.assertFalse(item.can_be_persisted)
        self.assertFalse(item.default_selected)

    def test_invalid_url_candidate_becomes_non_selectable_review_item(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="not-a-valid-url",
                title="Broken input",
                snippet="No usable source here.",
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep",),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.status, SourceCandidateStatus.INVALID_URL)
        self.assertIn("invalid url", item.rejection_reasons)
        self.assertTrue(item.diagnostics["invalid_url"])
        self.assertFalse(item.is_selectable)
        self.assertFalse(item.can_be_persisted)

    def test_duplicate_or_needs_review_candidate_is_represented_explicitly(self) -> None:
        first = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://health.example.org/infant-safe-sleep",
                title="2026 infant safe sleep guide: evidence, methodology, and bedtime routine recommendations",
                snippet="Recent safe sleep guidance for infants with evidence, bedtime routine methodology, implementation examples, and limitations.",
                readable_text_length=420,
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep", "bedtime routine"),
        )
        second = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://health.example.org/night-waking-methodology",
                title="2026 infant night waking guide: evidence, bedtime routine methodology, and limitations",
                snippet="Recent evidence-based infant sleep guide with bedtime routine methodology, night waking examples, and limitations for parents.",
                readable_text_length=390,
            ),
            topic="Baby sleeping",
            focus_terms=("night waking", "bedtime routine"),
            seen_hostnames={first.hostname},
        )

        item = build_candidate_review_item(second)

        self.assertEqual(item.status, SourceCandidateStatus.NEEDS_REVIEW)
        self.assertTrue(item.is_selectable)
        self.assertIn("duplicate hostname", item.rejection_reasons)
        self.assertTrue(item.diagnostics["quality_accepted"])

    def test_rejection_reasons_are_preserved(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://missing.example/article",
                title="Infant sleep article",
                snippet="This source could not be reached.",
                fetch_failure_reason="temporary failure in name resolution",
            ),
            topic="Baby sleeping",
            focus_terms=("infant sleep",),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(
            item.rejection_reasons,
            ("temporary failure in name resolution",),
        )

    def test_diagnostics_payload_is_preserved(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.org/family-travel-checklist",
                title="Family travel checklist",
                snippet="A practical family travel guide with budget tips.",
                readable_text_length=520,
                diagnostics={"provider": "fixture"},
            ),
            topic="Travel planning",
            focus_terms=("family travel", "budget travel"),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.diagnostics["provider"], "fixture")
        self.assertIn("score_breakdown", item.diagnostics)
        self.assertIn("matched_terms", item.diagnostics)

    def test_label_and_hostname_are_normalized_for_display(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://www.spinningbabies.com/pregnancy-birth/daily-activities/",
                title="",
                snippet="Daily activities for pregnancy comfort.",
                readable_text_length=280,
            ),
            topic="Physical exercises for pregnant women",
            focus_terms=("pregnancy",),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.hostname, "spinningbabies.com")
        self.assertEqual(item.label, "spinningbabies.com/pregnancy-birth/daily-activities")

    def test_list_adapter_returns_stable_score_aware_ordering(self) -> None:
        weak = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/",
                title="Home",
                snippet="Welcome.",
                readable_text_length=24,
            ),
            topic="Travel planning",
            focus_terms=("family travel",),
        )
        strong = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.net/travel-budget-guide",
                title="2026 family travel budget guide: checklist, tradeoffs, and implementation details",
                snippet="Recent family travel planning checklist with budget tradeoffs, implementation details, and concrete examples.",
                readable_text_length=520,
            ),
            topic="Travel planning",
            focus_terms=("family travel", "budget travel"),
        )
        unreachable = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://missing.example/article",
                title="Travel article",
                fetch_failure_reason="temporary failure in name resolution",
            ),
            topic="Travel planning",
            focus_terms=("family travel",),
        )
        low_relevance = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.edu/biology-lab-notes",
                title="2026 biology lab case study: methodology, evidence, and limitations",
                snippet="Recent classroom case study with methodology, evidence, comparison data, and limitations about cell structure experiments.",
                readable_text_length=260,
            ),
            topic="Travel planning",
            focus_terms=("family travel",),
        )

        items = build_candidate_review_items([weak, unreachable, strong, low_relevance])

        self.assertEqual(items[0].normalized_url, strong.normalized_url)
        self.assertEqual(items[0].status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(items[-1].status, SourceCandidateStatus.LOW_RELEVANCE)

    def test_adapter_does_not_require_http_or_template_context(self) -> None:
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.org/teen-education-tools",
                title="Education tools for teenagers",
                snippet="Study habits, online learning tools, and teacher guidance.",
                readable_text_length=410,
            ),
            topic="Education for teenagers",
            focus_terms=("study habits", "online learning"),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.label, "Education tools for teenagers")
        self.assertIsInstance(item.diagnostics, dict)


class SourceCandidateReviewAdapterPersistenceTests(TestCase):
    def test_adapter_does_not_create_topic_sources(self) -> None:
        user = get_user_model().objects.create_user(username="candidate-review-user", password="pw")
        Topic.objects.create(user=user, name="Candidate review topic")

        before = TopicSource.objects.count()
        candidate = evaluate_source_candidate(
            SourceCandidateInput(
                url="https://example.com/infant-sleep-safety",
                title="Infant sleep safety",
                snippet="Safe sleep guidance for babies.",
                readable_text_length=330,
            ),
            topic="Baby sleeping",
            focus_terms=("safe sleep",),
        )

        item = build_candidate_review_item(candidate)

        self.assertEqual(item.status, SourceCandidateStatus.ACCEPTED)
        self.assertEqual(TopicSource.objects.count(), before)
