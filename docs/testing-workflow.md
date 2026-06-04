# Testing Workflow

## Goal

DigestFlow should use a repeatable testing workflow that keeps the development loop fast during normal work and reserves the full Django suite for the right moments.

Related:
- `docs/frontend-style-guide.md` - frontend implementation quality rules for Django templates, CSS, JavaScript, and UI tests.

The intent is simple:

- use the narrowest test that proves the change
- only expand scope when the previous level passes
- avoid running the full suite during every normal development iteration

## Level 1 - Single Regression Test

Use this for the exact behavior currently being changed.

This is the fastest development loop and should be the default starting point.

Example:

```powershell
.\.venv\Scripts\python.exe manage.py test tests.test_topic_rss_source.TopicRssSourceTests.test_specific_behavior
```

Via the helper script:

```powershell
.\scripts\test.ps1 single tests.test_topic_rss_source.TopicRssSourceTests.test_specific_behavior
```

## Level 2 - Related Module Tests

Use this when working inside one module or one tightly scoped subsystem.

Examples:

```powershell
.\.venv\Scripts\python.exe manage.py test tests.test_rss_adapter
.\.venv\Scripts\python.exe manage.py test tests.test_topic_rss_source
```

Typical use:

- source ingestion and validation work
- one adapter
- one view or one UI test module
- one ranking module

## Level 3 - Feature Area Tests

Use this when the change spans a larger feature area such as sources, ranking, pipeline, or UI behavior that crosses multiple related modules.

These should be run through the reusable PowerShell wrapper:

```powershell
.\scripts\test.ps1 sources
.\scripts\test.ps1 ranking
.\scripts\test.ps1 pipeline
```

Current mappings:

- `sources`
  - `tests.test_rss_adapter`
  - `tests.test_topic_rss_source`
- `ranking`
  - `tests.test_ranker`
- `pipeline`
  - `tests.test_pipeline_happy_path`
  - `tests.test_pipeline_failures`

Pipeline is intentionally conservative for now. TODO: decide whether additional digest-generation or packaging modules should join the `pipeline` group once the team wants a broader feature-area command.

## Level 4 - Full Suite

Use the full suite only:

- before finalizing a task
- before commit or push
- after broad refactors
- after cross-system changes

Do not run the full suite during every normal development iteration.

Commands:

```powershell
.\.venv\Scripts\python.exe manage.py test
.\scripts\test.ps1 full
```

## Example Workflows

### Saved-source ingestion work

```powershell
.\scripts\test.ps1 sources
```

### Single regression

```powershell
.\scripts\test.ps1 single tests.test_topic_rss_source.TopicRssSourceTests.test_name_here
```

### Ranking work

```powershell
.\scripts\test.ps1 ranking
```

### Final verification

```powershell
.\scripts\test.ps1 full
```

## Fast Iteration Scopes

### saved-sources

Use for saved-source form and view behavior.

This is the right loop for:

- saved-source add flow
- saved-source duplicate handling
- source persistence in the saved inventory
- toggle/remove behavior
- saved-source ordering

Command:

```powershell
.\scripts\test.ps1 saved-sources
```

### source-ingestion

Use for fetch, extraction, source detection, and acceptance or rejection behavior.

This is the right loop for:

- article URL inspection
- source classification
- extraction strategy changes
- readability and acceptance logic
- saved-source acceptance tests that depend on ingestion behavior

Command:

```powershell
.\scripts\test.ps1 source-ingestion
```

### ui

Use for validation rendering, form state behavior, and hidden diagnostics display.

This is the right loop for:

- inline validation feedback
- input retention after failure
- hidden diagnostics rendering
- workspace section visibility
- template-level state behavior

Command:

```powershell
.\scripts\test.ps1 ui
```

### live-diagnostics

Use only for manual or diagnostic investigation of real external URLs.

This is not a stable regression suite. Do not treat live external fetch behavior as part of a normal fast test loop.

Command:

```powershell
.\scripts\test.ps1 live-diagnostics
.\scripts\test.ps1 live-diagnostics https://example.com/article
```

### Source Work Escalation Path

For source ingestion and debugging:

1. single regression test
2. source-ingestion
3. saved-sources or ui if form behavior changed
4. sources
5. full only at task completion if needed

## Development Guidance

- Start at Level 1 whenever a single failing behavior is known.
- Move to Level 2 when one module is stable and you want subsystem confidence.
- Move to Level 3 when the change spans multiple related modules in one feature area.
- Move to Level 4 only once the task is substantively complete.

When live manual verification tells you more than a full suite would, do the manual verification first and then choose the smallest matching automated test level.

## Semantic Output Testing

For semantic systems, avoid brittle exact-string expectations unless the wording itself is the feature.

Prefer tests that validate:

- topic grounding
- category coverage
- intent preservation
- diversity of outputs
- constraint compliance

Good semantic tests should tolerate:

- paraphrasing
- richer terminology
- more specific phrasing
- broader but still relevant wording

Exact wording should only be asserted when wording itself is critical, such as fixed UI copy, contract text, or canonical machine-readable output.

## Future Pytest or Marker Migration

If the suite becomes larger or slower, future markers should distinguish slower or less stable scopes:

- `slow`
- `live`
- `integration`
- `semantic`

Until then:

- do not introduce live external URL checks into normal fast suites
- if tests become slow, isolate them behind a slower command or a future marker
- do not migrate the current Django unittest suite to pytest in the middle of routine feature work

## CI Policy

Intended future policy:

- local iteration uses scoped script commands
- CI should run the full suite on push or pull request
- CI can later split fast suites and full-suite jobs if the project needs shorter feedback loops

This document does not create CI configuration yet.
