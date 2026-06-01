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
Today: {{ current_date }}
Recent query history summary:
{{ query_history_summary }}

Use the recent query history summary only as compact planning guidance.
Do not copy old queries verbatim unless you are clearly reframing them.
Avoid repeating weak, duplicate-heavy, or exhausted directions.
Do not treat provider/API failures as proof that a topic angle is weak.
Prefer fresh variants of useful directions when they still fit the topic.
This is a fresh discovery search. Do not include stale explicit years older than {{ current_year }}.
Prefer temporal wording such as latest, current, recent, or this month.
Only mention {{ current_year }} if the year itself is materially relevant to the topic.
Make search angles concrete and specific, not generic. Good angle examples include institutional flows, ETF demand, macro liquidity, retail behavior, volatility or risk, market structure, analyst outlook, on-chain data, implementation lessons, case study outcomes, or regulation shifts when relevant to the topic.
When recent quality guidance recommends reports, data, flows, market structure, on-chain analysis, analyst research, or research papers, reflect that directly in the final queries. Make at least a few final queries explicitly use those material-oriented terms instead of only broad retail, beginner, or trading-strategy phrasing.

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
