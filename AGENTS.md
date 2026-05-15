# DigestFlow Assistant Notes

## Testing Policy

- Start with the narrowest relevant test.
- Do not run the full suite after every small edit.
- Escalate test scope only when the previous level passes.
- Use feature-area test commands during normal development.
- Run the full suite only once at the end of a completed task unless the change is explicitly cross-system.
- If live or manual verification is more relevant than a full suite, do the manual check first.
- Do not default to `sources` if `source-ingestion` or `saved-sources` is enough.
- Do not default to the full suite during iteration.
- For live URL issues, first identify whether the issue is fetch, redirect, source detection, extraction, acceptance or rejection, or UI/form handling.
- Choose the narrowest test alias that matches that layer.

## Preferred Test Escalation

1. Level 1: a single regression test for the exact behavior being changed
2. Level 2: related module tests for the subsystem being touched
3. Level 3: feature-area tests through `.\scripts\test.ps1`
4. Level 4: the full Django test suite only at the end of the task

## Source Debugging Bias

For source-related work, use this escalation path:

1. `single`
2. `source-ingestion`
3. `saved-sources` or `ui` if the form behavior changed
4. `sources`
5. `full` only at task completion when needed

See [docs/testing-workflow.md](C:\Users\Елена\Documents\DigestFlow\docs\testing-workflow.md) for command examples and feature-area mappings.
