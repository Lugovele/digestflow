from django.test import SimpleTestCase

from services.sources.detector import classify_source_url, detect_source_type


class SourceDetectorTests(SimpleTestCase):
    def test_detects_devto_tag_source(self):
        source = classify_source_url("https://dev.to/t/ai")

        self.assertEqual(source.source_type, "devto_tag")
        self.assertEqual(source.normalized_url, "https://dev.to/api/articles?tag=ai")
        self.assertEqual(source.detection_reason, "matched dev.to topic pattern")
        self.assertEqual(source.metadata["tag"], "ai")
        self.assertEqual(detect_source_type("https://dev.to/t/ai"), "devto_tag")

    def test_detects_rss_feed_source(self):
        source = classify_source_url("https://openai.com/news/rss.xml")

        self.assertEqual(source.source_type, "rss_feed")
        self.assertEqual(source.detection_reason, "matched RSS/XML URL pattern")

    def test_detects_blog_index_source(self):
        source = classify_source_url("https://huggingface.co/blog")

        self.assertEqual(source.source_type, "blog_index")
        self.assertEqual(source.detection_reason, "matched blog or news index path")

    def test_detects_publication_homepage_source(self):
        source = classify_source_url("https://stratechery.com")

        self.assertEqual(source.source_type, "publication")
        self.assertEqual(source.detection_reason, "matched publication homepage pattern")

    def test_detects_generic_html_source(self):
        source = classify_source_url("https://example.com/random-page")

        self.assertEqual(source.source_type, "generic_html")
        self.assertEqual(source.detection_reason, "defaulted to generic HTML page")

    def test_detects_devto_article_source(self):
        source = classify_source_url("https://dev.to/alice/my-article")

        self.assertEqual(source.source_type, "devto_article")
        self.assertEqual(source.detection_reason, "matched dev.to article path")
