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
            "snippet": (
                f"A product team in {topic_label} reduced manual research time by 35% after "
                "switching from weekly summaries to daily AI-assisted briefings. The team also "
                "cut the time needed to prepare stakeholder updates from two hours to forty minutes."
            ),
        },
        {
            "title": f"{topic_label}: market signal one",
            "url": "https://example.com/article-1",
            "source": "Example News",
            "snippet": (
                f"The same {topic_label} case study reported a 35% drop in manual research time "
                "and faster stakeholder reporting after introducing AI-assisted briefings. The team "
                "said the briefing draft came together sooner, but the review step still required "
                "someone to check the claims one by one before the update was safe to share."
            ),
        },
        {
            "title": f"{topic_label}: product release analysis",
            "url": "https://example.com/article-2",
            "source": "Example Blog",
            "snippet": (
                f"A new {topic_label} workflow release added source-level citation blocks and "
                "structured output templates. Early users said the change made review easier and "
                "reduced editing passes before publishing. Teams said the package looked cleaner, "
                "but they still had to compare claims against original notes before anything went "
                "live, and that extra review work still sat with the same operators at the end."
            ),
        },
        {
            "title": f"{topic_label}: operator takeaway",
            "url": "https://example.com/article-3",
            "source": "Example Research",
            "snippet": (
                f"Operations leads testing {topic_label} workflows found that teams using a fixed "
                "pipeline with explicit validation caught output issues earlier. The report noted "
                "fewer last-minute corrections when digest generation and packaging were kept separate. "
                "It also described how review cycles stayed shorter when the handoff between drafting "
                "and validation was clear, because teams stopped rediscovering the same problems at the end."
            ),
        },
    ]
