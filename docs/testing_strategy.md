# Testing Strategy

Use the narrowest test target that matches the layer you are changing.

## Source candidate and research foundation work

During development of candidate or research foundation code, run:

```powershell
.\scripts\test_source_candidates.ps1
```

This should cover pure service-layer candidate evaluation without pulling in broader source workflow regressions.

## Before committing source-layer work

Run:

```powershell
.\scripts\test_sources.ps1
```

This combines:

- `tests.test_source_candidates`
- `tests.test_topic_rss_source`

Use it for source-layer changes that should still preserve the current `TopicSource` persistence and source review behavior.

## Before push or closing an architectural layer

Run:

```powershell
.\scripts\test_all.ps1
```

This runs the full Django suite.

## Module boundaries

Do not put future research-layer tests into `tests.test_topic_rss_source` unless they are true integration or regression tests for the existing `TopicSource` or source review behavior.

Prefer separate modules such as:

- `tests.test_source_candidate_review`
- `tests.test_source_research_queries`
- `tests.test_source_research_discovery`
