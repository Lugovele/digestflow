## Research Layer Follow-ups

- Research layer follow-up: research-cycle orchestration is currently concentrated in `apps/digests/views.py`. This is acceptable for MVP closeout, but a later refactor should move orchestration helpers into research-layer services to reduce view-layer complexity. This is a maintainability follow-up, not a blocking product issue.

## Future test split

`tests/test_topic_rss_source.py` still contains a large share of Research / Source Discovery coverage. This is acceptable for the current closeout, but a future maintainability pass should split research tests by responsibility.

Suggested future split:

- `tests/test_discovery_cycle.py` - discovery cycle rounds, stop conditions, target reached / not reached, provider unavailable, empty/no-evidence stops
- `tests/test_discovery_repair.py` - repair planning, repair application, repair query deduplication, repair surface diversity, repair plan usage
- `tests/test_search_surface_memory.py` - recent surface classification, exhausted/useful/underexplored surfaces, provider-error-only not exhausted, first-round steering
- `tests/test_research_history_view.py` - Research History rendering, Copy full history, current research state, cycle-total feedback

This is a follow-up, not a blocker for Research / Source Discovery layer closeout.

The split should be done gradually, after behavior is stable and covered by the current tests.

Do not combine this future split with feature work.
