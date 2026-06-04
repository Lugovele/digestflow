import re

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.digests.models import Digest, DigestRun, UsedArticle
from apps.packaging.models import ContentPackage
from apps.topics.models import Topic


class RunDetailViewTests(TestCase):
    def test_run_detail_marks_only_actual_digest_inputs_as_selected(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-selection")
        topic = Topic.objects.create(
            user=user,
            name="AI agents",
            keywords=["AI agents"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "status": "completed",
                    "ranked_articles_count": 4,
                    "articles_above_quality_threshold": 4,
                    "selected_for_prompt": 3,
                    "quality_threshold": 0.2,
                    "ranking_scores": [
                        {
                            "title": "Architect A Personalized Multi-Agent System with Long-Term Memory",
                            "url": "https://example.com/architect",
                            "source_name": "DEV Community",
                            "score": 11.85,
                            "diversity_penalty": 0.0,
                            "diversity_adjusted_score": 11.85,
                            "similarity_reasons": [],
                            "quality_score": 1.0,
                            "primary_article_type": "tutorial",
                            "quality_reasons": ["strong relevance to topic"],
                            "rejection_reasons": [],
                        },
                        {
                            "title": "Building Capabilities for a Multi-Agent System with Google ADK, MCP, and Cloud Run",
                            "url": "https://example.com/capabilities",
                            "source_name": "DEV Community",
                            "score": 11.85,
                            "quality_score": 1.0,
                            "primary_article_type": "tutorial",
                            "quality_reasons": ["strong relevance to topic"],
                            "rejection_reasons": [],
                        },
                        {
                            "title": "Deterministic Guardrails for Non-Deterministic Agents",
                            "url": "https://example.com/guardrails",
                            "source_name": "DEV Community",
                            "score": 10.37,
                            "quality_score": 1.0,
                            "primary_article_type": "deep_technical",
                            "quality_reasons": ["strong relevance to topic"],
                            "rejection_reasons": [],
                        },
                        {
                            "title": "I Built My Mom an AI Recipe Helper for Mother's Day",
                            "url": "https://example.com/recipe-helper",
                            "source_name": "DEV Community",
                            "score": 9.9,
                            "diversity_penalty": 0.44,
                            "diversity_adjusted_score": 9.46,
                            "similarity_reasons": [
                                "moderate supporting-tag overlap with an already selected article",
                                "same source or publication family as selected article",
                            ],
                            "quality_score": 1.0,
                            "primary_article_type": "tutorial",
                            "quality_reasons": ["good technical/practical article"],
                            "rejection_reasons": [],
                        },
                    ],
                },
                "digest_stage": {"status": "completed", "articles_count": 3},
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI agents",
            payload={
                "title": "Digest for AI agents",
                "articles": [
                    {
                        "url": "https://example.com/architect",
                        "title": "Architect A Personalized Multi-Agent System with Long-Term Memory",
                        "summary": "Architect summary",
                    },
                    {
                        "url": "https://example.com/capabilities",
                        "title": "Building Capabilities for a Multi-Agent System with Google ADK, MCP, and Cloud Run",
                        "summary": "Capabilities summary",
                    },
                    {
                        "url": "https://example.com/guardrails",
                        "title": "Deterministic Guardrails for Non-Deterministic Agents",
                        "summary": "Guardrails summary",
                    },
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))
        response_text = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        ranked_articles = response.context["ranked_articles"]
        selected_by_title = {
            article["title"]: article["is_selected_for_digest"]
            for article in ranked_articles
        }
        self.assertTrue(
            selected_by_title["Architect A Personalized Multi-Agent System with Long-Term Memory"]
        )
        self.assertTrue(
            selected_by_title["Building Capabilities for a Multi-Agent System with Google ADK, MCP, and Cloud Run"]
        )
        self.assertTrue(
            selected_by_title["Deterministic Guardrails for Non-Deterministic Agents"]
        )
        self.assertFalse(
            selected_by_title["I Built My Mom an AI Recipe Helper for Mother's Day"]
        )
        self.assertRegex(
            response_text,
            re.compile(
                r"I Built My Mom an AI Recipe Helper for Mother's Day.*?metric-pill metric-pill--warning\">rejected<",
                re.DOTALL,
            ),
        )
        self.assertRegex(
            response_text,
            re.compile(
                r"Architect A Personalized Multi-Agent System with Long-Term Memory.*?metric-pill metric-pill--quality\">selected<",
                re.DOTALL,
            ),
        )

    def test_run_detail_exposes_copy_diagnostics_payload_with_collapsed_data(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-copy")
        topic = Topic.objects.create(
            user=user,
            name="AI agents",
            keywords=["AI agents"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "source_stage": {
                    "status": "completed",
                    "source_url": "https://dev.to/t/ai",
                    "detected_source_type": "devto_tag",
                    "detection_reason": "matched dev.to topic pattern",
                    "raw_items_count": 10,
                    "article_links_extracted": 10,
                    "article_contents_fetched": 10,
                    "articles_count": 10,
                    "articles_after_cleaning": 10,
                    "full_article_count": 10,
                    "articles_after_dedupe": 10,
                    "duplicate_urls_removed": 0,
                    "duplicate_titles_removed": 0,
                    "saved_articles_count": 10,
                },
                "ranking_stage": {
                    "status": "completed",
                    "ranked_articles_count": 2,
                    "articles_above_quality_threshold": 2,
                    "quality_threshold": 0.2,
                    "selected_for_prompt": 1,
                    "average_quality_score": 0.9,
                    "max_quality_score": 1.0,
                    "min_actual_quality_score": 0.8,
                    "rejected_low_quality_count": 0,
                    "ranking_scores": [
                        {
                            "title": "Architect A Personalized Multi-Agent System with Long-Term Memory",
                            "url": "https://example.com/architect",
                            "source_name": "DEV Community",
                            "score": 11.85,
                            "diversity_penalty": 0.0,
                            "diversity_adjusted_score": 11.85,
                            "similarity_reasons": [],
                            "quality_score": 1.0,
                            "primary_article_type": "tutorial",
                            "dominant_tags": ["multi_agent", "memory"],
                            "supporting_tags": ["cloud", "mcp"],
                            "weak_tags": ["security"],
                            "quality_reasons": ["strong relevance to topic"],
                            "topic_relevance_reason": "matched strong agent-system signals in the title",
                            "topic_specificity_reason": "matched multiple strong topic-specific agent signals",
                            "heading_diagnostics": {
                                "detected_headings": ["Long-Term Memory"],
                                "normalized_headings": ["long term memory"],
                                "heading_count": 1,
                                "raw_html_heading_count": 2,
                                "extracted_heading_count": 1,
                                "heading_extraction_strategy": "markdown_headings",
                                "sample_detected_headings": ["Long-Term Memory"],
                                "heading_source": "inferred",
                                "matched_heading_tags": {
                                    "memory": {
                                        "matches": ["Long-Term Memory"],
                                        "normalized_matches": ["long term memory"],
                                    }
                                },
                            },
                            "weighted_secondary_tags": {
                                "multi_agent": {
                                    "strength": 2.0,
                                    "reason": "matched title intent and repeated body signals",
                                    "signals": ["multi-agent"],
                                    "title_matches": ["multi-agent"],
                                    "intro_matches": [],
                                    "heading_matches": [],
                                    "body_match_count": 7,
                                    "editorial_weight": 3.1,
                                    "body_weight_component": 1.1,
                                    "body_saturation_applied": True,
                                    "heading_weight_component": 0.0,
                                    "centrality_reason": "tag appears in the title and is reinforced through the article body",
                                }
                            },
                            "rejection_reasons": [],
                        },
                        {
                            "title": "I Built My Mom an AI Recipe Helper for Mother's Day",
                            "url": "https://example.com/recipe-helper",
                            "source_name": "DEV Community",
                            "score": 9.9,
                            "diversity_penalty": 0.44,
                            "diversity_adjusted_score": 9.46,
                            "similarity_reasons": [
                                "moderate supporting-tag overlap with an already selected article",
                                "same source or publication family as selected article",
                            ],
                            "quality_score": 1.0,
                            "primary_article_type": "tutorial",
                            "dominant_tags": [],
                            "supporting_tags": [],
                            "weak_tags": ["workflow"],
                            "quality_reasons": ["good technical/practical article"],
                            "topic_relevance_reason": "matched only broad AI-adjacent signals",
                            "topic_specificity_reason": "matched broad AI signals without strong agent specificity",
                            "heading_diagnostics": {
                                "detected_headings": [],
                                "normalized_headings": [],
                                "heading_count": 0,
                                "raw_html_heading_count": 0,
                                "extracted_heading_count": 0,
                                "heading_extraction_strategy": "none",
                                "sample_detected_headings": [],
                                "heading_source": "none",
                                "matched_heading_tags": {},
                            },
                            "weighted_secondary_tags": {},
                            "rejection_reasons": [],
                        },
                    ],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
                "packaging_stage": {"status": "completed"},
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI agents",
            payload={
                "title": "Digest for AI agents",
                "articles": [
                    {
                        "url": "https://example.com/architect",
                        "title": "Architect A Personalized Multi-Agent System with Long-Term Memory",
                        "summary": "Architect summary",
                        "key_points": ["Memory orchestration", "Long-term context"],
                    }
                ],
            },
        )
        ContentPackage.objects.create(
            digest=run.digest,
            post_text="Generated editorial post text.",
            hook_variants=["Hook one", "Hook two"],
            cta_variants=["CTA one", "CTA two"],
            hashtags=["#aiagents", "#mcp"],
            validation_report={
                "status": "valid",
                "post_text_length": 29,
                "cta_variants_count": 2,
                "hashtags_count": 2,
                "quality_checks": {"length_ok": True},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-testid="run-detail-copy-button"', html=False)
        self.assertContains(response, 'data-testid="run-detail-toggle-details-button"', html=False)
        self.assertContains(response, "Expand all")
        self.assertContains(response, 'id="copy-diagnostics-payload"', html=False)
        copy_payload = response.context["copy_diagnostics_text"]
        self.assertIn("Run ID: {}".format(run.id), copy_payload)
        self.assertIn("Post idea: AI agents", copy_payload)
        self.assertIn("Architect A Personalized Multi-Agent System with Long-Term Memory", copy_payload)
        self.assertIn("I Built My Mom an AI Recipe Helper for Mother's Day", copy_payload)
        self.assertIn("Status: selected", copy_payload)
        self.assertIn("Status: rejected", copy_payload)
        self.assertIn("CTA options:", copy_payload)
        self.assertIn("CTA one", copy_payload)
        self.assertIn("Hashtags:", copy_payload)
        self.assertIn("#aiagents", copy_payload)
        self.assertIn("Raw metrics JSON", copy_payload)
        self.assertContains(response, "Show diversity diagnostics")
        self.assertContains(response, "Diversity penalty")
        self.assertContains(response, "Diversity-adjusted score")
        self.assertContains(response, "moderate supporting-tag overlap with an already selected article")

    def test_run_detail_exposes_topic_level_used_article_history(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles")
        topic = Topic.objects.create(
            user=user,
            name="Used article diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        previous_run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Previous digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "selected_for_prompt": 1,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        Digest.objects.create(
            run=previous_run,
            title="Previous digest",
            payload={"title": "Previous digest", "articles": []},
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "selected_for_prompt": 2,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 2},
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for Used article diagnostics",
            payload={"title": "Digest for Used article diagnostics", "articles": []},
        )
        UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=previous_run,
            normalized_url="https://example.com/previous-article",
            article_url="https://example.com/previous-article",
            title="Used article from previous run",
            source_url="https://example.com/feed",
        )
        UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )
        UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-2",
            article_url="https://example.com/article-2",
            title="Used article two",
            source_url="https://example.com/feed",
        )
        other_topic = Topic.objects.create(
            user=user,
            name="Other topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        other_run = DigestRun.objects.create(
            topic=other_topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Other digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "selected_for_prompt": 1,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        UsedArticle.objects.create(
            user=user,
            topic=other_topic,
            digest_run=other_run,
            normalized_url="https://example.com/other-topic-article",
            article_url="https://example.com/other-topic-article",
            title="Other topic article",
            source_url="https://example.com/other-feed",
        )
        topic.name = "Renamed topic diagnostics"
        topic.save(update_fields=["name", "updated_at"])

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["used_article_count"], 3)
        self.assertEqual(len(response.context["used_articles"]), 3)
        self.assertContains(response, 'data-testid="used-article-history-region"', html=False)
        self.assertContains(response, "Used article history for this post idea (3)")
        self.assertContains(
            response,
            "Articles already used in successful posts for this post idea. Future repeat filtering will use this post idea history.",
        )
        self.assertContains(response, "Renamed topic diagnostics")
        self.assertContains(response, "Used article from previous run")
        self.assertContains(response, "Used article one")
        self.assertContains(response, "Used article two")
        self.assertContains(response, "example.com")
        self.assertContains(
            response,
            'href="https://example.com/article-1"',
            html=False,
        )
        self.assertContains(response, "Run {}".format(previous_run.id))
        self.assertContains(response, "Run {}".format(run.id))
        self.assertContains(response, "Used once")
        self.assertNotContains(response, "Other topic article")
        self.assertNotContains(response, "https://example.com/other-topic-article")
        self.assertNotContains(response, "<th>Source</th>", html=False)
        self.assertNotContains(response, "<th>Link</th>", html=False)
        self.assertNotContains(response, "<th>Open</th>", html=False)
        self.assertNotContains(response, "Open article")
        self.assertTrue(all("normalized_url" not in article for article in response.context["used_articles"]))
        self.assertTrue(all("source_url" not in article for article in response.context["used_articles"]))
        self.assertTrue(all(article["use_count"] == 1 for article in response.context["used_articles"]))

    def test_run_detail_exposes_repeat_usage_in_used_article_history(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-repeat")
        topic = Topic.objects.create(
            user=user,
            name="Used article repeat diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        first_run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="First digest generated successfully.",
            metrics={"ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []}, "digest_stage": {"status": "completed", "articles_count": 1}},
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Second digest generated successfully.",
            metrics={"ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []}, "digest_stage": {"status": "completed", "articles_count": 1}},
        )
        UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            first_used_in_run=first_run,
            last_used_in_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Repeated article",
            source_url="https://example.com/feed",
            use_count=2,
            first_used_at=first_run.created_at,
            last_used_at=run.created_at,
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Used article history for this post idea (1)")
        self.assertContains(response, "Repeated article")
        self.assertContains(response, "Used 2 times")
        self.assertContains(response, "First")
        self.assertContains(response, "Last")
        self.assertContains(response, "Run {}".format(first_run.id))
        self.assertContains(response, "Run {}".format(run.id))

    def test_run_detail_keeps_used_article_history_collapsed_and_read_only_by_default(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-collapsed")
        topic = Topic.objects.create(
            user=user,
            name="Collapsed used article diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "selected_for_prompt": 1,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'data-testid="used-article-history-region"', html=False)
        self.assertNotContains(
            response,
            '<details id="used-article-history" class="detail-block used-article-history" open>',
            html=False,
        )
        self.assertContains(
            response,
            'href="https://example.com/article-1"',
            html=False,
        )
        self.assertNotContains(response, "Edit used articles")
        self.assertNotContains(response, "Delete used article")

    def test_run_detail_used_article_history_renders_delete_action(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-delete")
        topic = Topic.objects.create(
            user=user,
            name="Delete used article diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        used_article = UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "Delete")
        self.assertContains(response, 'id="used-article-history"', html=False)
        self.assertContains(response, 'data-testid="used-article-delete-button"', html=False)
        self.assertContains(
            response,
            reverse("delete-used-article", args=[run.id, used_article.id]),
        )

    def test_delete_used_article_removes_only_targeted_row_and_redirects(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-delete-post")
        topic = Topic.objects.create(
            user=user,
            name="Delete post topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 2, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 2},
            },
        )
        target_article = UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )
        other_article = UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-2",
            article_url="https://example.com/article-2",
            title="Used article two",
            source_url="https://example.com/feed",
        )

        response = self.client.post(
            reverse("delete-used-article", args=[run.id, target_article.id]),
        )

        self.assertEqual(
            response["Location"],
            "{}#used-article-history".format(reverse("run-detail", args=[run.id])),
        )
        self.assertFalse(UsedArticle.objects.filter(pk=target_article.id).exists())
        self.assertTrue(UsedArticle.objects.filter(pk=other_article.id).exists())

        detail_response = self.client.get(reverse("run-detail", args=[run.id]))
        self.assertEqual(detail_response.context["used_article_count"], 1)
        self.assertContains(detail_response, "Used article history for this post idea (1)")
        self.assertNotContains(detail_response, "Used article one")
        self.assertContains(detail_response, "Used article two")

    def test_delete_used_article_last_row_shows_empty_state(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-empty")
        topic = Topic.objects.create(
            user=user,
            name="Delete last used article topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        used_article = UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )

        self.client.post(reverse("delete-used-article", args=[run.id, used_article.id]))
        detail_response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(detail_response.context["used_article_count"], 0)
        self.assertContains(detail_response, "Used article history for this post idea (0)")
        self.assertContains(detail_response, "No used article history yet for this post idea.")

    def test_delete_used_article_invalid_target_does_not_delete_unrelated_rows(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-used-articles-invalid")
        topic = Topic.objects.create(
            user=user,
            name="Delete invalid used article topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        kept_article = UsedArticle.objects.create(
            user=user,
            topic=topic,
            digest_run=run,
            normalized_url="https://example.com/article-1",
            article_url="https://example.com/article-1",
            title="Used article one",
            source_url="https://example.com/feed",
        )
        other_topic = Topic.objects.create(
            user=user,
            name="Other delete topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        other_run = DigestRun.objects.create(
            topic=other_topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        other_article = UsedArticle.objects.create(
            user=user,
            topic=other_topic,
            digest_run=other_run,
            normalized_url="https://example.com/article-2",
            article_url="https://example.com/article-2",
            title="Other topic article",
            source_url="https://example.com/feed",
        )
        foreign_user = get_user_model().objects.create_user(username="foreign-used-article-user")
        foreign_topic = Topic.objects.create(
            user=foreign_user,
            name="Foreign topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        foreign_run = DigestRun.objects.create(
            topic=foreign_topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {"selected_for_prompt": 1, "ranking_scores": []},
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        foreign_article = UsedArticle.objects.create(
            user=foreign_user,
            topic=foreign_topic,
            digest_run=foreign_run,
            normalized_url="https://example.com/article-3",
            article_url="https://example.com/article-3",
            title="Foreign topic article",
            source_url="https://example.com/feed",
        )

        response = self.client.post(
            reverse("delete-used-article", args=[run.id, other_article.id]),
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(UsedArticle.objects.filter(pk=kept_article.id).exists())
        self.assertTrue(UsedArticle.objects.filter(pk=other_article.id).exists())
        self.assertTrue(UsedArticle.objects.filter(pk=foreign_article.id).exists())

        foreign_response = self.client.post(
            reverse("delete-used-article", args=[run.id, foreign_article.id]),
        )
        self.assertEqual(foreign_response.status_code, 404)
        self.assertTrue(UsedArticle.objects.filter(pk=kept_article.id).exists())
        self.assertTrue(UsedArticle.objects.filter(pk=foreign_article.id).exists())

    def test_run_detail_renders_only_per_article_digest_view(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user")
        topic = Topic.objects.create(
            user=user,
            name="AI workflows",
            keywords=["AI workflows"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.8,
                    "min_actual_quality_score": 0.6,
                    "average_quality_score": 0.7,
                    "articles_above_quality_threshold": 2,
                    "selected_for_prompt": 2,
                    "ranking_scores": [
                        {
                            "title": "First article title",
                            "url": "https://example.com/article-1",
                            "source_name": "Example Source",
                            "content_tier": "full_article",
                            "content_length": 850,
                            "primary_article_type": "tutorial",
                            "secondary_article_tags": ["multi_agent", "testing", "memory"],
                            "weighted_secondary_tags": {
                                "multi_agent": {
                                    "strength": 2.0,
                                    "reason": "matched title intent and repeated body signals",
                                    "signals": ["multi-agent"],
                                    "title_matches": ["multi-agent"],
                                    "intro_matches": ["multi-agent"],
                                    "heading_matches": [],
                                    "body_match_count": 3,
                                    "editorial_weight": 4.5,
                                    "body_weight_component": 0.6,
                                    "body_saturation_applied": False,
                                    "heading_weight_component": 0.0,
                                    "centrality_reason": "tag appears in the title and is reinforced through the article body",
                                },
                                "testing": {"strength": 2.0, "reason": "matched repeatedly in title and body", "signals": ["testing"]},
                                "memory": {"strength": 2.0, "reason": "matched repeatedly in title and body", "signals": ["memory"]},
                            },
                            "dominant_tags": ["multi_agent", "testing", "memory"],
                            "supporting_tags": [],
                            "weak_tags": [],
                            "article_type": "tutorial",
                            "article_type_reason": "matched instructional or implementation-oriented language",
                            "article_type_score_modifier": 0.5,
                            "dominant_theme_reason": "instructional, deployment, or testing framing dominates the article structure and title intent",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["instructional signals: guide, testing"],
                                "tag_signals": ["multi_agent: matched multi-agent phrasing (multi-agent)", "testing: matched testing wording (testing)"],
                            },
                            "heading_diagnostics": {
                                "detected_headings": ["Architecture", "Long-Term Memory"],
                                "normalized_headings": ["architecture", "long term memory"],
                                "heading_count": 2,
                                "raw_html_heading_count": 3,
                                "extracted_heading_count": 2,
                                "heading_extraction_strategy": "markdown_headings",
                                "sample_detected_headings": ["Architecture", "Long-Term Memory"],
                                "heading_source": "inferred",
                                "matched_heading_tags": {
                                    "memory": {
                                        "matches": ["Long-Term Memory"],
                                        "normalized_matches": ["long term memory"],
                                    }
                                },
                            },
                            "score": 7,
                            "quality_score": 0.8,
                            "final_quality_score": 0.8,
                            "topic_relevance_score": 4.0,
                            "topic_relevance_reason": "matched strong agent-system signals in the title",
                            "relevance_signals": ["multi-agent"],
                            "weak_relevance_signals": ["cloud"],
                            "missing_relevance_signals": [],
                            "topic_specificity_score": 1.5,
                            "topic_specificity_reason": "matched clear topic-specific agent signals",
                            "specificity_signals": ["multi-agent", "memory"],
                            "generic_topic_signals": [],
                            "evidence_score": 2.0,
                            "practical_value_score": 1.0,
                            "novelty_score": 1.0,
                            "quality_reasons": ["strong relevance to topic"],
                            "rejection_reasons": [],
                            "diagnostic_warnings": [],
                        },
                        {
                            "title": "Second article title",
                            "url": "https://example.com/article-2",
                            "source_name": "Example Source",
                            "content_tier": "full_article",
                            "content_length": 620,
                            "primary_article_type": "architecture_security",
                            "secondary_article_tags": ["ai_agents", "security", "auth", "oauth"],
                            "weighted_secondary_tags": {
                                "ai_agents": {"strength": 2.0, "reason": "matched strongly in title or repeated body signals", "signals": ["ai agents"]},
                                "security": {
                                    "strength": 2.0,
                                    "reason": "matched title intent and repeated body signals",
                                    "signals": ["authorization", "security"],
                                    "title_matches": ["authorization"],
                                    "intro_matches": ["authorization"],
                                    "heading_matches": [],
                                    "body_match_count": 4,
                                    "editorial_weight": 4.5,
                                    "body_weight_component": 0.85,
                                    "body_saturation_applied": False,
                                    "heading_weight_component": 0.0,
                                    "centrality_reason": "tag appears in the title and is reinforced through the article body",
                                },
                                "auth": {"strength": 2.0, "reason": "matched repeatedly in title and body", "signals": ["authorize", "authorization"]},
                                "oauth": {"strength": 2.0, "reason": "matched strongly in title or repeated body signals", "signals": ["oauth", "token exchange"]},
                            },
                            "dominant_tags": ["ai_agents", "security", "auth", "oauth"],
                            "supporting_tags": [],
                            "weak_tags": [],
                            "article_type": "architecture_security",
                            "article_type_reason": "matched authorization, security, or access-control language as the main editorial focus",
                            "article_type_score_modifier": 0.5,
                            "dominant_theme_reason": "security and authorization concerns are central in the title and/or introduction",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["security signals: token exchange, authorization, oauth"],
                                "tag_signals": ["oauth: matched OAuth/token-exchange wording (oauth, token exchange)"],
                            },
                            "heading_diagnostics": {
                                "detected_headings": [],
                                "normalized_headings": [],
                                "heading_count": 0,
                                "raw_html_heading_count": 0,
                                "extracted_heading_count": 0,
                                "heading_extraction_strategy": "none",
                                "sample_detected_headings": [],
                                "heading_source": "none",
                                "matched_heading_tags": {},
                            },
                            "score": 6,
                            "quality_score": 0.6,
                            "final_quality_score": 0.6,
                            "topic_relevance_score": 3.0,
                            "topic_relevance_reason": "matched the topic phrase directly in the title",
                            "relevance_signals": ["ai agents"],
                            "weak_relevance_signals": [],
                            "missing_relevance_signals": [],
                            "topic_specificity_score": 2.0,
                            "topic_specificity_reason": "matched multiple strong topic-specific agent signals",
                            "specificity_signals": ["ai agents", "token exchange", "oauth"],
                            "generic_topic_signals": [],
                            "evidence_score": 1.5,
                            "practical_value_score": 1.0,
                            "novelty_score": 0.5,
                            "quality_reasons": ["good technical/practical article"],
                            "rejection_reasons": [],
                            "diagnostic_warnings": [],
                        },
                    ],
                },
                "digest_stage": {
                    "status": "completed",
                    "articles_count": 2,
                }
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI workflows",
            payload={
                "title": "Digest for AI workflows",
                "articles": [
                    {
                        "url": "https://example.com/article-1",
                        "title": "First article title",
                        "summary": "Article one summary",
                        "key_points": ["Point one", "Point two"],
                        "content_type": "news",
                        "confidence": 0.8,
                    },
                    {
                        "url": "https://example.com/article-2",
                        "title": "Second article title",
                        "summary": "Article two summary",
                        "key_points": ["Point three", "Point four"],
                        "content_type": "opinion",
                        "confidence": 0.6,
                    },
                ],
            },
        )
        ContentPackage.objects.create(
            digest=run.digest,
            post_text="Generated editorial post text.",
            hook_variants=[
                "Hook one",
                "Hook two",
                "Hook three",
            ],
            cta_variants=[
                "CTA one",
                "CTA two",
                "CTA three",
            ],
            hashtags=["#aiagents", "#mcp", "#cloudrun"],
            carousel_outline=[
                {"title": "Why agent memory matters"},
                {"headline": "Deployment patterns"},
            ],
            validation_report={
                "status": "valid",
                "post_text_length": 29,
                "hook_variants_count": 3,
                "cta_variants_count": 3,
                "hashtags_count": 3,
                "carousel_outline_count": 2,
                "quality_checks": {
                    "length_ok": True,
                    "has_hashtags": True,
                },
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(run.digest.has_articles())
        self.assertTrue(response.context["has_digest_articles"])
        self.assertEqual(
            response.context["digest_payload"]["title"],
            "Digest for AI workflows",
        )
        self.assertEqual(len(response.context["digest_payload"]["articles"]), 2)
        self.assertContains(response, 'href="https://example.com/article-1"', html=False)
        self.assertContains(response, "First article title")
        self.assertContains(response, "Article one summary")
        self.assertContains(response, "Publish-ready post generated successfully.")
        self.assertContains(response, "Publish-ready post")
        self.assertContains(response, "Validation status")
        self.assertContains(response, "valid")
        self.assertContains(response, "Publish-ready post")
        self.assertContains(response, "Generated editorial post text.")
        self.assertContains(response, "Primary hook")
        self.assertContains(response, "Hook one")
        self.assertContains(response, "Hooks")
        self.assertContains(response, "Hook two")
        self.assertContains(response, "CTA options")
        self.assertContains(response, "CTA one")
        self.assertContains(response, "#aiagents")
        self.assertContains(response, "#mcp")
        self.assertNotContains(response, "<summary>Carousel outline</summary>", html=False)
        self.assertContains(response, "Why agent memory matters")
        self.assertContains(response, "Validation report and quality checks")
        self.assertContains(response, "Post length:")
        self.assertContains(response, "Length ok:")
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, "Post generation result")
        self.assertContains(response, "Post idea:</strong> AI workflows", html=False)
        self.assertContains(response, "Status:</strong> completed", html=False)
        self.assertContains(response, "Pipeline diagnostics")
        self.assertContains(response, "Source Stage")
        self.assertContains(response, "Ranking Stage")
        self.assertContains(response, "Ranking summary")
        self.assertContains(response, "Pipeline decision:")
        self.assertContains(response, "topic_relevance_score: 4,0")
        self.assertContains(response, "topic_relevance_reason: matched strong agent-system signals in the title")
        self.assertContains(response, "relevance_signals: multi-agent")
        self.assertContains(response, "weak_relevance_signals: cloud")
        self.assertContains(response, "content_tier: full_article")
        self.assertContains(response, "primary_article_type: tutorial")
        self.assertContains(response, "secondary_article_tags: multi_agent, testing, memory")
        self.assertContains(response, "dominant_tags: multi_agent, testing, memory")
        self.assertContains(response, "weighted_secondary_tags:")
        self.assertContains(response, "editorial_tag_weighting:")
        self.assertContains(response, "title matches: multi-agent")
        self.assertContains(response, "editorial weight: 4,5")
        self.assertContains(response, "body weight component: 0,6")
        self.assertContains(response, "body saturation applied: no")
        self.assertContains(response, "heading weight component: 0,0")
        self.assertContains(response, "centrality reason: tag appears in the title and is reinforced through the article body")
        self.assertContains(response, "heading_diagnostics:")
        self.assertContains(response, "detected_headings: Architecture; Long-Term Memory")
        self.assertContains(response, "normalized_headings: architecture; long term memory")
        self.assertContains(response, "heading_source: inferred")
        self.assertContains(response, "matched_heading_tags:")
        self.assertContains(response, "memory: Long-Term Memory")
        self.assertContains(response, "heading_source: none")
        first_heading_display = response.context["ranked_articles"][0]["article_card"]["heading_display"]
        self.assertEqual(first_heading_display["heading_extraction_strategy"], "markdown_headings")
        self.assertEqual(first_heading_display["raw_html_heading_count"], 3)
        self.assertEqual(first_heading_display["extracted_heading_count"], 2)
        self.assertEqual(first_heading_display["sample_detected_headings"], ["Architecture", "Long-Term Memory"])
        self.assertContains(response, "article_type: tutorial")
        self.assertContains(response, "article_type_score_modifier: 0,5")
        self.assertContains(response, "dominant_theme_reason: instructional, deployment, or testing framing dominates the article structure and title intent")
        self.assertContains(response, "classification_signal_summary:")
        self.assertContains(response, "topic_specificity_score: 1,5")
        self.assertContains(response, "topic_specificity_reason: matched clear topic-specific agent signals")
        self.assertContains(response, "specificity_signals: multi-agent, memory")
        self.assertContains(response, "Selected articles")
        self.assertContains(response, "All ranked articles")
        self.assertNotContains(response, "Legacy compatibility view")
        self.assertNotContains(response, "Sources:")

    def test_run_detail_reads_articles_from_digest_helpers(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-v1")
        topic = Topic.objects.create(
            user=user,
            name="AI operations",
            keywords=["AI operations"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 1,
                    "ranking_scores": [
                        {
                            "title": "Linked article title",
                            "url": "https://example.com/article-v1",
                            "source_name": "Example Source",
                            "score": 5,
                            "quality_score": 0.5,
                            "quality_reasons": ["strong relevance to topic"],
                        }
                    ],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        digest = Digest.objects.create(
            run=run,
            title="Digest for AI operations",
            payload={
                "title": "Digest for AI operations",
                "articles": [
                    {
                        "url": "https://example.com/article-v1",
                        "title": "Linked article title",
                        "summary": "Article summary",
                        "key_points": ["Point one"],
                        "content_type": "news",
                        "confidence": 0.5,
                    }
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(
            response.context["digest_payload"]["articles"][0]["url"],
            digest.get_articles()[0]["url"],
        )
        self.assertEqual(
            response.context["digest_payload"]["articles"][0]["summary"],
            digest.get_articles()[0]["summary"],
        )
        self.assertEqual(response.context["digest_payload"]["articles"][0]["domain"], "example.com")
        self.assertEqual(response.context["digest_payload"]["articles"][0]["title"], "Linked article title")
        self.assertEqual(response.context["digest_payload"]["articles"][0]["link_label"], "Linked article title")
        self.assertEqual(response.context["ranked_articles"][0]["title"], "Linked article title")

    def test_run_detail_shows_fallback_when_article_url_is_missing(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-empty-url")
        topic = Topic.objects.create(
            user=user,
            name="AI delivery",
            keywords=["AI delivery"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 1,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "completed", "articles_count": 1},
            },
        )
        Digest.objects.create(
            run=run,
            title="Digest for AI delivery",
            payload={
                "title": "Digest for AI delivery",
                "articles": [
                    {
                        "url": "",
                        "title": "",
                        "summary": "Article without source url",
                        "key_points": ["Point one"],
                        "content_type": "news",
                        "confidence": 0.4,
                    }
                ],
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "No source available")

    def test_run_detail_shows_insufficient_quality_message_without_digest_or_package(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-insufficient")
        topic = Topic.objects.create(
            user=user,
            name="Broad AI source",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            result_message="Not enough high-quality articles for a full digest.",
            error_message=(
                "Недостаточно качественных статей для полноценного дайджеста. "
                "Источник обработан, но найденные материалы слишком слабые или разрозненные."
            ),
            metrics={
                "ranking_stage": {
                    "status": "insufficient_quality",
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.3,
                    "min_actual_quality_score": 0.0,
                    "average_quality_score": 0.08,
                    "articles_above_quality_threshold": 1,
                    "selected_for_prompt": 0,
                    "rejected_low_quality_count": 3,
                    "ranking_scores": [
                        {
                            "title": "Weak article one",
                            "url": "https://example.com/weak-1",
                            "source_name": "Example Blog",
                            "content_tier": "full_article",
                            "content_length": 780,
                            "primary_article_type": "community_update",
                            "secondary_article_tags": ["event"],
                            "weighted_secondary_tags": {
                                "event": {"strength": 2.0, "reason": "matched strongly in title or repeated body signals", "signals": ["winners", "challenge"]},
                            },
                            "dominant_tags": ["event"],
                            "supporting_tags": [],
                            "weak_tags": [],
                            "article_type": "community_update",
                            "article_type_reason": "matched community/event wording in the title",
                            "article_type_score_modifier": -0.5,
                            "dominant_theme_reason": "community or event framing dominates the title and editorial purpose",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["title event signals: winners, challenge"],
                                "tag_signals": ["event: matched event/community title wording (winners, challenge)"],
                            },
                            "score": 3,
                            "quality_score": 0.3,
                            "final_quality_score": 0.3,
                            "topic_relevance_score": 1.0,
                            "topic_relevance_reason": "matched only broad AI-adjacent signals",
                            "relevance_signals": [],
                            "weak_relevance_signals": ["ai"],
                            "missing_relevance_signals": ["ai agents", "multi-agent", "agent orchestration"],
                            "topic_specificity_score": 0.0,
                            "topic_specificity_reason": "no strong topic-specific signals were detected",
                            "specificity_signals": [],
                            "generic_topic_signals": ["ai"],
                            "evidence_score": 1.0,
                            "practical_value_score": 0.5,
                            "novelty_score": 0.5,
                            "quality_reasons": [
                                "too narrow for selected topic",
                                "low practical value",
                            ],
                            "rejection_reasons": ["low relevance", "low practical value"],
                            "diagnostic_warnings": [],
                        },
                        {
                            "title": "Weak article two",
                            "url": "https://example.com/weak-2",
                            "source_name": "Example Blog",
                            "content_tier": "rich_summary",
                            "content_length": 160,
                            "primary_article_type": "lightweight_post",
                            "secondary_article_tags": [],
                            "weighted_secondary_tags": {
                                "workflow": {"strength": 0.5, "reason": "single incidental mention", "signals": ["workflow"]},
                            },
                            "dominant_tags": [],
                            "supporting_tags": [],
                            "weak_tags": ["workflow"],
                            "article_type": "lightweight_post",
                            "article_type_reason": "summary-level blog item without stronger structural signals",
                            "article_type_score_modifier": -0.5,
                            "dominant_theme_reason": "summary-style content does not show a stronger editorial format",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["summary-level blog content"],
                                "tag_signals": [],
                            },
                            "score": 2,
                            "quality_score": 0.2,
                            "final_quality_score": 0.2,
                            "topic_relevance_score": 0.0,
                            "topic_relevance_reason": "did not match meaningful topic signals",
                            "relevance_signals": [],
                            "weak_relevance_signals": ["workflow"],
                            "missing_relevance_signals": ["ai agents", "multi-agent", "agent orchestration"],
                            "topic_specificity_score": 0.0,
                            "topic_specificity_reason": "matched broad AI signals without strong agent specificity",
                            "specificity_signals": [],
                            "generic_topic_signals": ["workflow"],
                            "evidence_score": 0.5,
                            "practical_value_score": 0.5,
                            "novelty_score": 1.0,
                            "quality_reasons": [
                                "weak relevance to topic",
                                "insufficient evidence/detail",
                            ],
                            "rejection_reasons": ["low relevance", "insufficient detail"],
                            "diagnostic_warnings": [],
                        },
                    ],
                    "top_rejected_articles": [
                        {
                            "title": "Weak article one",
                            "url": "https://example.com/weak-1",
                            "content_tier": "full_article",
                            "content_length": 780,
                            "primary_article_type": "community_update",
                            "secondary_article_tags": ["event"],
                            "weighted_secondary_tags": {
                                "event": {"strength": 2.0, "reason": "matched strongly in title or repeated body signals", "signals": ["winners", "challenge"]},
                            },
                            "dominant_tags": ["event"],
                            "supporting_tags": [],
                            "weak_tags": [],
                            "article_type": "community_update",
                            "article_type_reason": "matched community/event wording in the title",
                            "article_type_score_modifier": -0.5,
                            "dominant_theme_reason": "community or event framing dominates the title and editorial purpose",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["title event signals: winners, challenge"],
                                "tag_signals": ["event: matched event/community title wording (winners, challenge)"],
                            },
                            "quality_score": 0.3,
                            "final_quality_score": 0.3,
                            "topic_relevance_score": 1.0,
                            "topic_relevance_reason": "matched only broad AI-adjacent signals",
                            "relevance_signals": [],
                            "weak_relevance_signals": ["ai"],
                            "missing_relevance_signals": ["ai agents", "multi-agent", "agent orchestration"],
                            "topic_specificity_score": 0.0,
                            "topic_specificity_reason": "no strong topic-specific signals were detected",
                            "specificity_signals": [],
                            "generic_topic_signals": ["ai"],
                            "evidence_score": 1.0,
                            "practical_value_score": 0.5,
                            "novelty_score": 0.5,
                            "quality_reasons": [
                                "too narrow for selected topic",
                                "low practical value",
                            ],
                            "rejection_reasons": ["low relevance", "low practical value"],
                            "diagnostic_warnings": [],
                        },
                        {
                            "title": "Weak article two",
                            "url": "https://example.com/weak-2",
                            "content_tier": "rich_summary",
                            "content_length": 160,
                            "primary_article_type": "lightweight_post",
                            "secondary_article_tags": [],
                            "weighted_secondary_tags": {
                                "workflow": {"strength": 0.5, "reason": "single incidental mention", "signals": ["workflow"]},
                            },
                            "dominant_tags": [],
                            "supporting_tags": [],
                            "weak_tags": ["workflow"],
                            "article_type": "lightweight_post",
                            "article_type_reason": "summary-level blog item without stronger structural signals",
                            "article_type_score_modifier": -0.5,
                            "dominant_theme_reason": "summary-style content does not show a stronger editorial format",
                            "primary_type_override_reason": None,
                            "classification_signal_summary": {
                                "primary_signals": ["summary-level blog content"],
                                "tag_signals": [],
                            },
                            "quality_score": 0.2,
                            "final_quality_score": 0.2,
                            "topic_relevance_score": 0.0,
                            "topic_relevance_reason": "did not match meaningful topic signals",
                            "relevance_signals": [],
                            "weak_relevance_signals": ["workflow"],
                            "missing_relevance_signals": ["ai agents", "multi-agent", "agent orchestration"],
                            "topic_specificity_score": 0.0,
                            "topic_specificity_reason": "matched broad AI signals without strong agent specificity",
                            "specificity_signals": [],
                            "generic_topic_signals": ["workflow"],
                            "evidence_score": 0.5,
                            "practical_value_score": 0.5,
                            "novelty_score": 1.0,
                            "quality_reasons": [
                                "weak relevance to topic",
                                "insufficient evidence/detail",
                            ],
                            "rejection_reasons": ["low relevance", "insufficient detail"],
                            "diagnostic_warnings": [],
                        },
                    ],
                    "insufficient_quality_message": (
                        "Недостаточно качественных статей для полноценного дайджеста. "
                        "Источник обработан, но найденные материалы слишком слабые или разрозненные."
                    ),
                },
                "digest_stage": {"status": "skipped", "reason": "insufficient_quality"},
                "packaging_stage": {"status": "skipped", "reason": "insufficient_quality"},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))
        response_text = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context["is_insufficient_quality"])
        self.assertEqual(
            response.context["insufficient_quality_message"],
            "Not enough high-quality articles for a publish-ready post.",
        )
        self.assertEqual(
            response.context["display_error_message"],
            "Insufficient-quality diagnostics are available in metrics.",
        )
        self.assertContains(response, "Not enough high-quality articles for a publish-ready post.")
        self.assertContains(response, "Insufficient-quality diagnostics are available in metrics.")
        self.assertEqual(
            response_text.count("Not enough high-quality articles for a publish-ready post."),
            1,
        )
        self.assertLess(
            response_text.index("Not enough high-quality articles for a publish-ready post."),
            response_text.index("Insufficient-quality diagnostics are available in metrics."),
        )
        visible_without_pre = re.sub(r"<pre.*?</pre>", "", response_text, flags=re.DOTALL)
        self.assertNotIn(run.error_message, visible_without_pre)
        self.assertContains(response, "Pipeline diagnostics")
        self.assertContains(response, "Source Stage")
        self.assertContains(response, "Ranking Stage")
        self.assertContains(response, "Pipeline decision:")
        self.assertContains(
            response,
            "Post draft generation skipped because too few articles passed quality validation.",
        )
        self.assertContains(response, "Primary article type:</strong> community_update", html=False)
        self.assertContains(response, "Secondary tags:</strong> event", html=False)
        self.assertContains(response, "Dominant tags:</strong> event", html=False)
        self.assertContains(response, "weak_tags: workflow")
        self.assertContains(response, "Weighted tag strengths:</strong>", html=False)
        self.assertContains(response, "Article type:</strong> community_update", html=False)
        self.assertContains(response, "Article type score modifier:</strong> -0,5", html=False)
        self.assertContains(response, "Dominant theme reason:</strong> community or event framing dominates the title and editorial purpose", html=False)
        self.assertContains(response, "topic_relevance_score: 0,0")
        self.assertContains(response, "Topic relevance score:</strong> 1,0", html=False)
        self.assertContains(response, "Topic specificity score:</strong> 0,0", html=False)
        self.assertContains(response, "Topic specificity reason:</strong> no strong topic-specific signals were detected", html=False)
        self.assertContains(response, "Evidence score:</strong> 1,0", html=False)
        self.assertContains(response, "Rejected because:")
        self.assertContains(response, "low relevance")
        self.assertContains(response, "All ranked articles")
        self.assertContains(response, "Weak article one")
        self.assertContains(response, "https://example.com/weak-1")
        self.assertContains(
            response,
            "No post draft was generated because the selected articles did not meet the required quality level.",
        )
        self.assertContains(
            response,
            "No publish-ready post was generated because too few articles passed the quality threshold.",
        )
        self.assertContains(response, "Articles above threshold:</strong> 1", html=False)

    def test_run_detail_uses_english_insufficient_quality_fallback_for_legacy_records(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-insufficient-legacy")
        topic = Topic.objects.create(
            user=user,
            name="Legacy insufficient quality",
            keywords=["AI"],
            excluded_keywords=[],
        )
        legacy_error_message = (
            "Недостаточно качественных статей для полноценного дайджеста. "
            "Источник обработан, но найденные материалы слишком слабые или разрозненные."
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            result_message="",
            error_message=legacy_error_message,
            metrics={
                "ranking_stage": {
                    "status": "insufficient_quality",
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 0,
                    "selected_for_prompt": 0,
                    "ranking_scores": [],
                    "insufficient_quality_message": legacy_error_message,
                },
                "digest_stage": {"status": "skipped", "reason": "insufficient_quality"},
                "packaging_stage": {"status": "skipped", "reason": "insufficient_quality"},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))
        response_text = response.content.decode("utf-8")
        visible_without_pre = re.sub(r"<pre.*?</pre>", "", response_text, flags=re.DOTALL)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.context["insufficient_quality_message"],
            "Not enough high-quality articles for a publish-ready post.",
        )
        self.assertEqual(
            response.context["display_error_message"],
            "Insufficient-quality diagnostics are available in metrics.",
        )
        self.assertContains(response, "Not enough high-quality articles for a publish-ready post.")
        self.assertContains(response, "Insufficient-quality diagnostics are available in metrics.")
        self.assertNotIn(legacy_error_message, visible_without_pre)
        self.assertIn(legacy_error_message, response_text)

    def test_run_detail_shows_zero_values_as_zero(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-zero")
        topic = Topic.objects.create(
            user=user,
            name="Zero diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            metrics={
                "ranking_stage": {
                    "quality_threshold": 0.4,
                    "max_quality_score": 0.2,
                    "min_actual_quality_score": 0.0,
                    "average_quality_score": 0.08,
                    "articles_above_quality_threshold": 0,
                    "selected_for_prompt": 0,
                    "rejected_low_quality_count": 3,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "skipped", "tokens": {"total": 0}},
                "packaging_stage": {"status": "skipped", "tokens": {"total": 0}, "estimated_cost_usd": 0},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "Articles above threshold:</strong> 0", html=False)
        self.assertContains(response, "Selected for prompt:</strong> 0", html=False)
        self.assertContains(response, "Total tokens:</strong> 0", html=False)
        self.assertContains(response, "Total estimated cost:</strong> 0", html=False)

    def test_run_detail_renders_cleaning_rejections_in_source_stage(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-cleaning-rejections")
        topic = Topic.objects.create(
            user=user,
            name="Cleaning diagnostics",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_INSUFFICIENT_QUALITY,
            result_message="Not enough high-quality articles for a full digest.",
            metrics={
                "source_stage": {
                    "status": "completed",
                    "articles_count": 3,
                    "articles_after_cleaning": 1,
                    "removed_during_cleaning": 2,
                    "full_article_count": 0,
                    "rich_summary_count": 1,
                    "weak_snippet_count": 1,
                    "missing_content_count": 1,
                    "cleaning_rejections": [
                        {
                            "title": "Missing content article",
                            "url": "https://example.com/no-content",
                            "source_name": "Example Source",
                            "reason": "missing extracted content",
                            "content_tier": "missing_content",
                            "final_content_source": "direct_content",
                            "content_length": 0,
                            "content_preview": "",
                            "extraction_method": "fallback_text",
                            "extraction_warning": "extracted content is very short",
                            "extraction_candidates": [
                                {
                                    "selector": "article_tag",
                                    "found": False,
                                    "text_length": 0,
                                    "text_preview": "",
                                    "rejection_reason": "not found",
                                },
                                {
                                    "selector": "main_tag",
                                    "found": True,
                                    "text_length": 92,
                                    "text_preview": "OpenAI News Products Research Safety API Login Pricing",
                                    "rejection_reason": "too short",
                                },
                            ],
                        },
                        {
                            "title": "Tiny article",
                            "url": "https://example.com/tiny",
                            "source_name": "Example Source",
                            "reason": "content too short",
                            "content_tier": "weak_snippet",
                            "final_content_source": "html_article_body",
                            "content_length": 10,
                            "content_preview": "Too short.",
                            "extraction_method": "article_tag",
                            "extraction_warning": "extracted content is very short",
                            "extraction_candidates": [],
                        },
                    ],
                },
                "ranking_stage": {
                    "status": "insufficient_quality",
                    "quality_threshold": 0.4,
                    "articles_above_quality_threshold": 0,
                    "selected_for_prompt": 0,
                    "ranking_scores": [],
                },
                "digest_stage": {"status": "skipped", "reason": "insufficient_quality"},
                "packaging_stage": {"status": "skipped", "reason": "insufficient_quality"},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "Rejected during cleaning:")
        self.assertContains(response, "Content quality tiers:")
        self.assertContains(response, "Full articles:</strong> 0", html=False)
        self.assertContains(response, "Summary-only items:</strong> 1", html=False)
        self.assertContains(response, "Short snippets:</strong> 1", html=False)
        self.assertContains(response, "No extracted content:</strong> 1", html=False)
        self.assertNotContains(response, "Rich summaries")
        self.assertNotContains(response, "Weak snippets")
        self.assertNotContains(response, "Missing content:</strong> 1", html=False)
        self.assertContains(response, "Missing content article")
        self.assertContains(response, 'href="https://example.com/no-content"', html=False)
        self.assertContains(response, "Content tier:</strong> missing_content", html=False)
        self.assertContains(response, "Final content source:</strong> direct_content", html=False)
        self.assertContains(response, "missing extracted content")
        self.assertContains(response, "Extraction method:</strong> fallback_text", html=False)
        self.assertContains(response, "Extraction warning:</strong> extracted content is very short", html=False)
        self.assertContains(response, "Extraction candidates:")
        self.assertContains(response, "Selector:</strong> article_tag", html=False)
        self.assertContains(response, "Found:</strong> no", html=False)
        self.assertContains(response, "Rejection reason:</strong> not found", html=False)
        self.assertContains(response, "Selector:</strong> main_tag", html=False)
        self.assertContains(response, "Text length:</strong> 92", html=False)
        self.assertContains(response, "Rejection reason:</strong> too short", html=False)
        self.assertContains(response, "Extracted content length:</strong> 0", html=False)
        self.assertContains(response, "No extracted content available.")
        self.assertContains(response, "Tiny article")
        self.assertContains(response, "Content tier:</strong> weak_snippet", html=False)
        self.assertContains(response, "Final content source:</strong> html_article_body", html=False)
        self.assertContains(response, "content too short")
        self.assertContains(response, "Extraction method:</strong> article_tag", html=False)
        self.assertContains(response, "Extracted content length:</strong> 10", html=False)
        self.assertContains(response, "Too short.")

    def test_run_detail_hides_zero_value_source_stage_labels_from_visible_ui(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-zero-source-stage")
        topic = Topic.objects.create(
            user=user,
            name="Zero source stage labels",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            result_message="Digest generated successfully.",
            metrics={
                "source_stage": {
                    "status": "completed",
                    "raw_items_count": 10,
                    "article_links_extracted": 10,
                    "article_contents_fetched": 10,
                    "content_unavailable_count": 0,
                    "articles_count": 10,
                    "articles_after_cleaning": 10,
                    "removed_during_cleaning": 0,
                    "full_article_count": 10,
                    "rich_summary_count": 0,
                    "weak_snippet_count": 0,
                    "missing_content_count": 0,
                    "articles_after_dedupe": 10,
                    "duplicate_urls_removed": 0,
                    "duplicate_titles_removed": 0,
                    "saved_articles_count": 10,
                },
                "ranking_stage": {
                    "status": "completed",
                    "ranked_articles_count": 10,
                    "selected_for_prompt": 3,
                    "articles_above_quality_threshold": 3,
                    "quality_threshold": 0.4,
                    "average_quality_score": 0.7,
                    "max_quality_score": 0.9,
                    "min_actual_quality_score": 0.5,
                    "rejected_low_quality_count": 7,
                    "ranking_scores": [],
                },
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "Full articles:</strong> 10", html=False)
        self.assertNotContains(response, "Summary-only items")
        self.assertNotContains(response, "Short snippets")
        self.assertNotContains(response, "No extracted content")
        self.assertNotContains(response, "Content unavailable:</strong> 0", html=False)
        self.assertNotContains(response, "Removed during cleaning:</strong> 0", html=False)

    def test_run_detail_renders_raw_metrics_inside_collapsible_section(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-raw-metrics")
        topic = Topic.objects.create(
            user=user,
            name="Raw metrics topic",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_FAILED,
            metrics={
                "source_stage": {"status": "failed", "raw_items_count": 0},
                "ranking_stage": {"status": "skipped"},
            },
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertContains(response, "Raw metrics JSON")
        self.assertContains(response, "source_stage", html=False)

    def test_run_detail_hides_result_message_block_when_empty(self) -> None:
        user = get_user_model().objects.create_user(username="detail-user-no-result-message")
        topic = Topic.objects.create(
            user=user,
            name="No result message",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_FAILED,
            result_message="",
            error_message="Technical failure",
            metrics={},
        )

        response = self.client.get(reverse("run-detail", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, "<strong>Result:</strong>", html=False)
        self.assertContains(response, "Technical failure")
