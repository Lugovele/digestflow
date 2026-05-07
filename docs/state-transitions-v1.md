# DigestFlow State Transitions v1

## Purpose

This document defines the allowed state transitions for DigestFlow runs.

It builds on:

- [docs/run-states-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/run-states-v1.md)
- [docs/ux-flows-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/ux-flows-v1.md)

This is a planning document only.

It does not change code, data models, or UI yet.

The goal is to make future state handling explicit before implementation expands further.

---

## Transition Principles

The state transition model should:

- reflect actual product meaning, not only technical execution
- distinguish happy-path progress from warning and failure outcomes
- allow retry where it makes sense
- require user input where the next step depends on user intent
- prevent impossible or misleading jumps

---

## Main Happy Path

The canonical successful run path is:

`draft -> ready_to_run -> running -> collecting_sources -> ranking_articles -> generating_digest -> generating_content_package -> completed`

This is the default path when:

- topic is usable
- source strategy is valid
- enough strong articles are found
- digest generation succeeds
- packaging succeeds

---

## Transition Definitions

Each transition below includes:

- `from_state`
- `to_state`
- `trigger`
- `user-facing message`
- `retry_possible`
- `user_input_required`

---

### 1. `draft -> ready_to_run`

- **from_state:** `draft`
- **to_state:** `ready_to_run`
- **trigger:** the user has provided enough valid configuration to execute a run
- **user-facing message:** Run setup is complete. Ready to start.
- **retry_possible:** not applicable
- **user_input_required:** yes

---

### 2. `ready_to_run -> running`

- **from_state:** `ready_to_run`
- **to_state:** `running`
- **trigger:** the user explicitly starts the run
- **user-facing message:** Run started.
- **retry_possible:** not applicable
- **user_input_required:** yes

---

### 3. `running -> collecting_sources`

- **from_state:** `running`
- **to_state:** `collecting_sources`
- **trigger:** pipeline execution begins source collection
- **user-facing message:** Collecting sources.
- **retry_possible:** not applicable
- **user_input_required:** no

---

### 4. `collecting_sources -> ranking_articles`

- **from_state:** `collecting_sources`
- **to_state:** `ranking_articles`
- **trigger:** usable source candidates were collected and passed initial validation
- **user-facing message:** Ranking articles.
- **retry_possible:** not applicable
- **user_input_required:** no

---

### 5. `ranking_articles -> generating_digest`

- **from_state:** `ranking_articles`
- **to_state:** `generating_digest`
- **trigger:** enough articles passed the quality threshold
- **user-facing message:** Generating digest.
- **retry_possible:** not applicable
- **user_input_required:** no

---

### 6. `generating_digest -> generating_content_package`

- **from_state:** `generating_digest`
- **to_state:** `generating_content_package`
- **trigger:** digest generation completed successfully and packaging is enabled
- **user-facing message:** Generating content package.
- **retry_possible:** not applicable
- **user_input_required:** no

---

### 7. `generating_content_package -> completed`

- **from_state:** `generating_content_package`
- **to_state:** `completed`
- **trigger:** content package generation completed successfully without important warnings
- **user-facing message:** Digest and content package are ready.
- **retry_possible:** not applicable
- **user_input_required:** no

---

## Warning Paths

These transitions represent runs that should stop or finish with a meaningful warning rather than pretending to be a clean success.

---

### 8. `collecting_sources -> insufficient_sources`

- **from_state:** `collecting_sources`
- **to_state:** `insufficient_sources`
- **trigger:** too few usable source candidates were collected to continue meaningfully
- **user-facing message:** Not enough usable sources were found to continue.
- **retry_possible:** yes
- **user_input_required:** usually yes

---

### 9. `ranking_articles -> low_quality_sources`

- **from_state:** `ranking_articles`
- **to_state:** `low_quality_sources`
- **trigger:** source collection succeeded, but too few articles passed the editorial quality threshold
- **user-facing message:** Sources were processed, but the article set was too weak for a reliable digest.
- **retry_possible:** yes
- **user_input_required:** usually yes

---

### 10. `generating_content_package -> completed_with_warnings`

- **from_state:** `generating_content_package`
- **to_state:** `completed_with_warnings`
- **trigger:** packaging finished, but the system detected meaningful warnings that should stay visible
- **user-facing message:** Digest is ready, but there are warnings you should review.
- **retry_possible:** yes
- **user_input_required:** no

---

## Refinement Paths

These transitions represent cases where the system should pause for clarification rather than continue with weak assumptions.

---

### 11. `ready_to_run -> needs_topic_refinement`

- **from_state:** `ready_to_run`
- **to_state:** `needs_topic_refinement`
- **trigger:** topic validation or pre-run heuristics detect that the topic is too broad or ambiguous
- **user-facing message:** This topic needs refinement before we can run it confidently.
- **retry_possible:** yes
- **user_input_required:** yes

---

### 12. `collecting_sources -> needs_topic_refinement`

- **from_state:** `collecting_sources`
- **to_state:** `needs_topic_refinement`
- **trigger:** source collection reveals that the topic is too broad for the current strategy and is producing noisy or mismatched results
- **user-facing message:** The topic appears too broad for the current source strategy. Refinement is recommended.
- **retry_possible:** yes
- **user_input_required:** yes

---

### 13. `low_quality_sources -> needs_topic_refinement`

- **from_state:** `low_quality_sources`
- **to_state:** `needs_topic_refinement`
- **trigger:** quality diagnostics suggest that the weakest point is topic breadth or ambiguity rather than source failure alone
- **user-facing message:** The source worked, but the topic likely needs refinement before the next run.
- **retry_possible:** yes
- **user_input_required:** yes

---

## Failure Paths

These transitions represent technical or stage-specific failures where the pipeline could not continue.

---

### 14. `collecting_sources -> source_collection_failed`

- **from_state:** `collecting_sources`
- **to_state:** `source_collection_failed`
- **trigger:** source normalization, retrieval, parsing, or extraction failed in a way that prevented usable collection
- **user-facing message:** Source collection failed. Please check the source setup or try again.
- **retry_possible:** yes
- **user_input_required:** sometimes

---

### 15. `generating_digest -> digest_generation_failed`

- **from_state:** `generating_digest`
- **to_state:** `digest_generation_failed`
- **trigger:** digest generation failed because of model failure, validation failure, or unrecoverable payload issues
- **user-facing message:** Digest generation failed. You can retry or inspect the selected articles.
- **retry_possible:** yes
- **user_input_required:** not always

---

### 16. `generating_content_package -> packaging_failed`

- **from_state:** `generating_content_package`
- **to_state:** `packaging_failed`
- **trigger:** packaging failed after a digest was successfully generated
- **user-facing message:** The digest is available, but content packaging failed.
- **retry_possible:** yes
- **user_input_required:** not always

---

## Transition Summary Table

| From | To | Trigger category | Retry possible | User input required |
|---|---|---|---|---|
| `draft` | `ready_to_run` | configuration complete | no | yes |
| `ready_to_run` | `running` | user starts run | no | yes |
| `running` | `collecting_sources` | pipeline execution begins | no | no |
| `collecting_sources` | `ranking_articles` | usable candidates collected | no | no |
| `ranking_articles` | `generating_digest` | enough strong articles found | no | no |
| `generating_digest` | `generating_content_package` | digest succeeded | no | no |
| `generating_content_package` | `completed` | packaging succeeded cleanly | no | no |
| `collecting_sources` | `insufficient_sources` | too few sources | yes | yes |
| `ranking_articles` | `low_quality_sources` | too few strong articles | yes | yes |
| `generating_content_package` | `completed_with_warnings` | usable output with warnings | yes | no |
| `ready_to_run` | `needs_topic_refinement` | broad or ambiguous topic | yes | yes |
| `collecting_sources` | `needs_topic_refinement` | noisy collection reveals topic issue | yes | yes |
| `low_quality_sources` | `needs_topic_refinement` | weak result suggests topic problem | yes | yes |
| `collecting_sources` | `source_collection_failed` | technical collection failure | yes | sometimes |
| `generating_digest` | `digest_generation_failed` | digest stage failure | yes | not always |
| `generating_content_package` | `packaging_failed` | packaging stage failure | yes | not always |

---

## Invalid Transitions

The following transitions should not be allowed because they collapse meaning, skip required work, or hide user decisions.

### `draft -> completed`

Why invalid:

- the run never actually happened
- success cannot be claimed without execution

### `draft -> generating_digest`

Why invalid:

- source collection and ranking would be skipped
- this breaks the entire editorial pipeline model

### `ready_to_run -> completed`

Why invalid:

- a ready state means configuration is sufficient, not that the run finished

### `collecting_sources -> completed`

Why invalid:

- source collection alone is not a product outcome
- ranking and digest generation would be bypassed

### `ranking_articles -> completed`

Why invalid:

- ranking is not the final user outcome
- digest and optional packaging still matter

### `low_quality_sources -> completed`

Why invalid:

- this would hide the exact problem the system is supposed to surface

### `source_collection_failed -> completed_with_warnings`

Why invalid:

- a hard source failure is not a warning-level success

### `digest_generation_failed -> completed`

Why invalid:

- no digest exists, so completion would be misleading

### `packaging_failed -> completed`

Why invalid:

- if packaging is part of the configured run, a failed packaging stage should stay visible
- future UX may choose whether digest-only success rolls up differently, but not silently

### `needs_topic_refinement -> running`

Why invalid:

- refinement requires explicit user resolution
- the system should not proceed as if the ambiguity were resolved automatically

### `insufficient_sources -> generating_digest`

Why invalid:

- article generation cannot proceed when source collection is insufficient

### `completed -> running`

Why invalid:

- a completed run should not resume as the same run
- the correct behavior is to create a rerun or duplicate

---

## Implementation Notes

This section is intentionally high-level.

### 1. State transitions likely need explicit validation

Eventually the system should validate allowed transitions rather than allowing arbitrary status assignment.

### 2. Some states may belong to setup rather than execution

`draft`, `ready_to_run`, and `needs_topic_refinement` may eventually live partly in a pre-run workflow instead of the same execution lifecycle as `collecting_sources` and `generating_digest`.

### 3. Warning states and terminal states should remain distinct

`completed_with_warnings` is not the same as a failure state, and `low_quality_sources` is not the same as a source failure.

### 4. Retry behavior should be modeled intentionally

“Retry” may mean different things:

- rerun with identical configuration
- rerun after fixing sources
- rerun after topic refinement
- rerun with a new quality threshold

These should not all be treated as the same product action forever.

### 5. Parent/child run structures may affect transitions later

If one broad topic becomes multiple digest streams, the product may need:

- parent request transitions
- child run transitions
- roll-up logic for mixed outcomes

### 6. Current implementation should not be force-fit too early

The current code already has useful partial states, but this document is a target architecture map. The immediate goal is not to rename everything at once. The goal is to guide a future cleanup safely.

---

## Notes

- This document does not require immediate implementation.
- It should be used as a planning artifact before state-machine logic or UI workflow changes are introduced.
- The next useful follow-up document would be a transition ownership map:
  - which transitions are system-driven
  - which transitions require explicit user choice
  - which transitions should create a new run instead of mutating an old one

