"""Demo source provider for the first local MVP pipeline."""
from __future__ import annotations


def get_demo_articles_for_topic(topic_name: str) -> list[dict]:
    """Return a stable list of demo articles for the given topic name.

    The returned item format is the contract for the early source stage:
    title, url, source, snippet.
    """
    topic_label = topic_name.strip() or "Topic"

    return [
        {
            "title": f"{topic_label}: market signal one",
            "url": "https://example.com/article-1",
            "source": "Example News",
            "snippet": "A practical update relevant to the topic.",
        },
        {
            "title": f"{topic_label}: market signal one",
            "url": "https://example.com/article-1",
            "source": "Example News",
            "snippet": "Duplicate item to test deduplication.",
        },
        {
            "title": f"{topic_label}: product release analysis",
            "url": "https://example.com/article-2",
            "source": "Example Blog",
            "snippet": "Another relevant source item for digest generation.",
        },
        {
            "title": f"{topic_label}: operator takeaway",
            "url": "https://example.com/article-3",
            "source": "Example Research",
            "snippet": "Operational implications and strategic context.",
        },
    ]
