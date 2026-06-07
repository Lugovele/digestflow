import re
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from apps.digests.models import Digest, DigestRun
from apps.packaging.models import ContentPackage
from apps.topics.models import Topic


class PostResultViewTests(TestCase):
    def _create_user(self):
        return get_user_model().objects.create_user(username="post-result-user")

    def test_loading_post_result_page_renders_user_facing_copy(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Creator workflow",
            source_url="https://example.com/feed.xml",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_PENDING)

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Creating your post")
        self.assertContains(response, "Researching sources and writing your post")
        self.assertContains(response, "This can take a minute. The post will appear here automatically.")
        self.assertContains(response, "&larr; Edit direction", html=False)
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, reverse("topic-setup", args=[topic.id]))
        self.assertContains(response, reverse("topic-list"))
        self.assertContains(
            response,
            f'<form method="post" action="{reverse("start-post-result", args=[run.id])}" data-post-start-form class="sr-only">',
            html=False,
        )
        self.assertContains(response, reverse("start-post-result", args=[run.id]))
        self.assertContains(response, "Post idea:")
        self.assertContains(response, "Creator workflow")
        self.assertContains(response, 'data-testid="post-result-loading-card"', html=False)
        self.assertContains(response, "post-result-loading-spinner")
        self.assertNotContains(response, "Preparing your post")
        self.assertNotContains(response, "Current status")
        self.assertNotContains(response, "Starting now")
        self.assertNotContains(response, "Research → Select → Write → Ready")
        self.assertNotContains(response, "Research → Sources → Writing → Ready")
        self.assertNotContains(response, "digest")
        self.assertNotContains(response, "pipeline")
        self.assertNotContains(response, "draft")

    def test_completed_post_result_page_shows_final_post_before_controls(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Editorial workflow",
            source_url="https://example.com/feed.xml",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_COMPLETED)
        digest = Digest.objects.create(
            run=run,
            title="Editorial workflow",
            payload={
                "title": "Editorial workflow",
                "articles": [
                    {
                        "url": "https://example.com/one",
                        "title": "Research one",
                        "summary": "First summary",
                        "key_points": ["Point one"],
                    }
                ],
            },
        )
        ContentPackage.objects.create(
            digest=digest,
            post_text="Here is the core post body.",
            hook_variants=["Opening one", "Opening two"],
            cta_variants=["Closing one", "Closing two"],
            hashtags=["#AI", "#Workflows"],
            validation_report={"status": "valid"},
        )

        response = self.client.get(reverse("post-result", args=[run.id]))
        html = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Your post is ready")
        self.assertContains(response, "Final post")
        self.assertContains(response, "Choose opening and closing")
        self.assertContains(response, "Opening")
        self.assertContains(response, "Closing")
        self.assertContains(response, "&larr; Edit direction", html=False)
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, reverse("topic-setup", args=[topic.id]))
        self.assertContains(response, reverse("topic-list"))
        self.assertContains(response, "Copy full post")
        self.assertContains(response, "Copy includes the selected opening, post body, closing, and hashtags.")
        self.assertContains(response, "Opening one")
        self.assertContains(response, "Opening two")
        self.assertContains(response, "Closing one")
        self.assertContains(response, "Closing two")
        self.assertContains(response, "Here is the core post body.")
        self.assertContains(response, "#AI #Workflows")
        self.assertContains(response, "View research behind this post")
        self.assertLess(html.index("Final post"), html.index("Choose opening and closing"))
        self.assertIn("Opening one", html)
        self.assertIn("Here is the core post body.", html)
        self.assertIn("Closing one", html)
        self.assertIn("#AI #Workflows", html)

    def test_completed_post_result_page_does_not_repeat_hashtags_when_body_already_ends_with_them(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Hashtag display",
            source_url="https://example.com/feed.xml",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_COMPLETED)
        digest = Digest.objects.create(
            run=run,
            title="Hashtag display",
            payload={
                "title": "Hashtag display",
                "articles": [
                    {
                        "url": "https://example.com/one",
                        "title": "Research one",
                        "summary": "First summary",
                        "key_points": ["Point one"],
                    }
                ],
            },
        )
        ContentPackage.objects.create(
            digest=digest,
            post_text="Here is the core post body.\n\n#PersonalBranding #Authority #Storytelling",
            hook_variants=["Opening one", "Opening two"],
            cta_variants=["Closing one", "Closing two"],
            hashtags=["#PersonalBranding", "#Authority", "#Storytelling"],
            validation_report={"status": "valid"},
        )

        response = self.client.get(reverse("post-result", args=[run.id]))
        html = response.content.decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "#PersonalBranding #Authority #Storytelling")
        self.assertEqual(html.count("#PersonalBranding #Authority #Storytelling"), 1)

    def test_completed_mock_package_renders_source_recovery_instead_of_ready_state(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Untrusted workflow",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_COMPLETED,
            input_snapshot={"used_demo_source": True},
            metrics={"packaging_stage": {"is_mock": True, "fallback_reason": "placeholder-key"}},
        )
        digest = Digest.objects.create(
            run=run,
            title="Untrusted workflow",
            payload={"title": "Untrusted workflow", "articles": []},
        )
        ContentPackage.objects.create(
            digest=digest,
            post_text="Generic final body",
            hook_variants=["Opening one"],
            cta_variants=["Closing one"],
            hashtags=["#AI"],
            validation_report={"status": "valid"},
        )

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "We need real sources first")
        self.assertContains(response, "PostFlow could not find enough reliable sources automatically.")
        self.assertContains(response, "Review sources first, then create the post again.")
        self.assertContains(response, "Review sources first")
        self.assertContains(response, "&larr; Edit direction", html=False)
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, "Back to direction")
        self.assertNotContains(response, "Your post is ready")
        self.assertNotContains(response, "Final post")
        visible_text = re.sub(r"<script.*?</script>", "", response.content.decode("utf-8"), flags=re.DOTALL)
        visible_text = re.sub(r"<style.*?</style>", "", visible_text, flags=re.DOTALL)
        visible_text = re.sub(r"<[^>]+>", " ", visible_text)
        visible_text = re.sub(r"\s+", " ", visible_text).lower()
        self.assertNotIn("mock", visible_text)
        self.assertNotIn("demo", visible_text)
        self.assertNotIn("fallback", visible_text)
        self.assertNotIn("digest", visible_text)
        self.assertNotIn("pipeline", visible_text)
        self.assertNotIn("run", visible_text)
        self.assertNotIn("provider", visible_text)
        self.assertNotIn("tokens", visible_text)

    def test_safe_fallback_package_renders_source_recovery_instead_of_ready_state(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Fallback workflow",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_COMPLETED)
        digest = Digest.objects.create(
            run=run,
            title="Fallback workflow",
            payload={"title": "Fallback workflow", "articles": []},
        )
        ContentPackage.objects.create(
            digest=digest,
            post_text="Fallback workflow\n\nNo post draft articles were available.",
            hook_variants=["No article pattern was available."],
            cta_variants=["What data would help here?"],
            hashtags=["#AI", "#Workflows"],
            validation_report={"status": "valid"},
        )

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "We need real sources first")
        self.assertNotContains(response, "Your post is ready")

    def test_pending_post_result_without_completed_discovery_attempt_renders_loading(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Missing source inputs",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_PENDING)

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Creating your post")
        self.assertContains(response, "Researching sources and writing your post")
        self.assertContains(response, "This can take a minute. The post will appear here automatically.")
        self.assertContains(
            response,
            f'<form method="post" action="{reverse("start-post-result", args=[run.id])}" data-post-start-form class="sr-only">',
            html=False,
        )
        self.assertNotContains(response, "Preparing your post")
        self.assertNotContains(response, "Current status")
        self.assertNotContains(response, "Starting now")
        self.assertNotContains(response, "We need real sources first")
        self.assertNotContains(response, "Your post is ready")
        self.assertNotContains(response, "Final post")

    def test_post_result_with_failed_automatic_discovery_shows_needs_sources_state(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Discovery failed inputs",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_PENDING,
            input_snapshot={
                "needs_sources": True,
                "automatic_source_discovery_attempted": True,
                "usable_source_count": 2,
                "usable_source_target": 6,
            },
        )

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "We need real sources first")
        self.assertContains(response, "PostFlow could not find enough reliable sources automatically.")
        self.assertContains(response, "Review sources first")
        self.assertContains(response, "&larr; Edit direction", html=False)
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, "Back to direction")
        self.assertNotContains(
            response,
            f'<form method="post" action="{reverse("start-post-result", args=[run.id])}" data-post-start-form class="sr-only">',
            html=False,
        )
        self.assertNotContains(
            response,
            f'<form method="post" action="{reverse("start-post-result", args=[run.id])}" data-post-start-form class="sr-only">',
            html=False,
        )
        self.assertNotContains(response, "Researching sources and writing your post")
        self.assertNotContains(response, "This can take a minute. The post will appear here automatically.")
        self.assertNotContains(response, "Your post is ready")
        self.assertNotContains(response, "Final post")
        self.assertNotContains(response, "None")

    @patch("apps.digests.views._start_topic_run")
    @patch("apps.digests.views._run_automatic_create_post_discovery")
    def test_start_post_after_successful_automatic_discovery_allows_generation(
        self,
        mock_discovery,
        mock_start_topic_run,
    ) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Successful auto discovery",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_PENDING)
        mock_discovery.return_value = {"attempted": True, "usable_source_count": 6}

        response = self.client.post(reverse("start-post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        mock_discovery.assert_called_once_with(topic)
        mock_start_topic_run.assert_called_once_with(
            run,
            topic,
            default_source="setup_auto",
            allow_demo_fallback=False,
        )
        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_COLLECTING)
        self.assertFalse(run.input_snapshot.get("needs_sources"))
        self.assertEqual(run.input_snapshot.get("usable_source_count"), 6)
        self.assertTrue(run.input_snapshot.get("automatic_source_discovery_attempted"))
        self.assertFalse(hasattr(run, "digest"))
        self.assertFalse(ContentPackage.objects.exists())

    @patch("apps.digests.views._start_topic_run")
    @patch("apps.digests.views._run_automatic_create_post_discovery")
    def test_start_post_with_failed_automatic_discovery_stays_in_needs_sources_state(
        self,
        mock_discovery,
        mock_start_topic_run,
    ) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Failed auto discovery",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_PENDING)
        mock_discovery.return_value = {"attempted": True, "usable_source_count": 3}

        response = self.client.post(reverse("start-post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        mock_discovery.assert_called_once_with(topic)
        mock_start_topic_run.assert_not_called()
        run.refresh_from_db()
        self.assertEqual(run.status, DigestRun.STATUS_PENDING)
        self.assertTrue(run.input_snapshot.get("needs_sources"))
        self.assertEqual(run.input_snapshot.get("usable_source_count"), 3)
        self.assertEqual(run.input_snapshot.get("usable_source_target"), 6)
        self.assertTrue(run.input_snapshot.get("automatic_source_discovery_attempted"))
        self.assertFalse(hasattr(run, "digest"))
        self.assertFalse(ContentPackage.objects.exists())

        page_response = self.client.get(reverse("post-result", args=[run.id]))
        self.assertContains(page_response, "We need real sources first")
        self.assertNotContains(page_response, "Your post is ready")

    def test_retry_without_real_sources_stays_user_facing(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Retry missing sources",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(topic=topic, status=DigestRun.STATUS_FAILED)

        response = self.client.post(reverse("retry-post-result", args=[run.id]))

        self.assertRedirects(response, reverse("post-result", args=[run.id]), fetch_redirect_response=False)
        self.assertEqual(DigestRun.objects.filter(topic=topic).count(), 1)
        self.assertEqual(Topic.objects.count(), 1)

        page_response = self.client.get(reverse("post-result", args=[run.id]))
        self.assertContains(page_response, "We need real sources first")
        self.assertContains(page_response, "Review sources first")

    def test_failed_post_result_page_renders_recovery_actions(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Recovery flow",
            source_url="https://example.com/feed.xml",
            keywords=["AI"],
            excluded_keywords=[],
        )
        run = DigestRun.objects.create(
            topic=topic,
            status=DigestRun.STATUS_FAILED,
            error_message="Technical timeout",
        )

        response = self.client.get(reverse("post-result", args=[run.id]))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "We couldn&#x27;t create the post", html=False)
        self.assertContains(response, "Something interrupted the generation process.")
        self.assertContains(response, "Try again or adjust the direction")
        self.assertContains(response, "Try again")
        self.assertContains(response, "&larr; Edit direction", html=False)
        self.assertContains(response, "Back to workspace")
        self.assertContains(response, "Back to direction")
        self.assertNotContains(response, "diagnostics")
        self.assertNotContains(response, "provider")
        self.assertNotContains(response, "tokens")

    def test_create_post_from_setup_enters_post_result_route_without_duplicate_topic(self) -> None:
        topic = Topic.objects.create(
            user=self._create_user(),
            name="Stable topic",
            source_url="https://example.com/feed.xml",
            keywords=["AI"],
            excluded_keywords=[],
            focus_initialized=True,
        )

        response = self.client.post(reverse("continue-topic-setup", args=[topic.id]))

        self.assertEqual(Topic.objects.count(), 1)
        run = DigestRun.objects.get(topic=topic)
        self.assertRedirects(response, reverse("post-result", args=[run.id]), fetch_redirect_response=False)
