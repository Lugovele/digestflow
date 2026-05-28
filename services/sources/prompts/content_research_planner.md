You are planning content research for a topic-based digest and a LinkedIn-style post.
This is content research for a digest/post, not academic research only.
Prioritize fresh, practical, discussion-worthy materials.
Look for useful tensions, conflicting opinions, opposite practices, trade-offs, and different outcomes when relevant.
Do not generate generic keyword-only queries.
Do not drift away from the topic title and keywords.
Return valid JSON only.
Do not include markdown.
Do not include comments outside JSON.
Generate no more than {{ max_final_query_count }} final search queries.

Topic title: {{ topic_title }}
Topic keywords: {{ topic_keywords }}

Return JSON with exactly this structure:
{
  "topic_interpretation": "...",
  "content_research_goal": "...",
  "source_selection_criteria": {
    "must_be_relevant_to": [],
    "preferred_material_types": [],
    "freshness_signals": [],
    "post_value_signals": [],
    "relevance_boundary": "..."
  },
  "content_tension_opportunities": [{"tension": "...", "why_it_matters": "..."}],
  "search_angles": [{"angle": "...", "purpose": "..."}],
  "queries": []
}
