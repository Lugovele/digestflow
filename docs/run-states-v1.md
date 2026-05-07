# DigestFlow Run States v1

## Purpose

This document defines the first clear map of internal and user-facing run states for DigestFlow.

It is a planning document only. It does not require immediate code changes, UI redesign, or backend refactoring.

The goal is to make run outcomes more honest and more useful:

- a run should not look "successful" only because the pipeline technically finished
- topic clarity, source quality, and digest generation quality should be represented separately
- the user should understand what happened and what to do next

---

## Current Problem

Today, a state like `completed` is too broad.

A run may technically finish while still producing weak, misleading, or low-value output. Examples:

- the topic was too broad, but the system still proceeded
- sources worked, but article quality was weak
- digest generation technically succeeded, but the result should have been blocked earlier
- packaging succeeded, but only because the system pushed weak material through the pipeline

This creates three product problems:

1. source quality is underrepresented
2. topic clarity is underrepresented
3. digest/package generation outcome is overrepresented

As DigestFlow evolves into an editorial system, these layers need to be visible separately.

---

## Design Principles

The run state model should:

- distinguish preparation from execution
- distinguish source problems from ranking problems
- distinguish editorial insufficiency from technical failure
- make next actions obvious
- avoid treating low-signal output as a normal success

---

## Proposed Run States

### 1. `draft`

#### Internal meaning

The run configuration exists, but is incomplete or not yet ready for execution.

#### User-facing label

Draft

#### Trigger condition

- user has started setting up a topic, source strategy, audience, or threshold
- required inputs are still missing
- run has not been confirmed

#### What the user can do next

- refine topic
- add sources
- select audience
- adjust threshold
- save and continue later
- move to ready state

#### Should the run count as successful

No

---

### 2. `ready_to_run`

#### Internal meaning

The run has enough validated input to start execution.

#### User-facing label

Ready to run

#### Trigger condition

- topic or refined topic is accepted
- source strategy is available
- validation has passed enough checks to execute

#### What the user can do next

- start run
- revise topic
- revise sources
- change source mode or strictness

#### Should the run count as successful

No

---

### 3. `running`

#### Internal meaning

The run is active, but the system is not exposing the current sub-stage yet.

#### User-facing label

Running

#### Trigger condition

- user started the run
- the pipeline is executing
- detailed stage may be unknown or not surfaced yet

#### What the user can do next

- wait
- monitor progress
- optionally cancel in a future implementation

#### Should the run count as successful

No

---

### 4. `collecting_sources`

#### Internal meaning

The system is gathering source candidates and validating them.

#### User-facing label

Collecting sources

#### Trigger condition

- run has started
- source collection is in progress

#### What the user can do next

- wait
- inspect source configuration
- in future, intervene on failing sources

#### Should the run count as successful

No

---

### 5. `ranking_articles`

#### Internal meaning

Source items were collected and cleaned, and the system is ranking them for editorial usefulness.

#### User-facing label

Ranking articles

#### Trigger condition

- source collection produced usable article candidates
- ranking and quality filtering are in progress

#### What the user can do next

- wait
- in future, review live diagnostics or threshold suggestions

#### Should the run count as successful

No

---

### 6. `generating_digest`

#### Internal meaning

Enough articles passed quality checks, and the system is generating the digest.

#### User-facing label

Generating digest

#### Trigger condition

- ranking produced enough articles above the threshold
- digest stage has started

#### What the user can do next

- wait
- in future, inspect selected article set

#### Should the run count as successful

No

---

### 7. `generating_content_package`

#### Internal meaning

Digest generation succeeded and packaging is being generated from the digest articles.

#### User-facing label

Generating content package

#### Trigger condition

- digest payload exists
- packaging stage has started

#### What the user can do next

- wait
- in future, choose packaging mode or skip packaging

#### Should the run count as successful

No

---

### 8. `completed`

#### Internal meaning

The run completed without meaningful warnings. Source collection, ranking, digest generation, and optional packaging all produced acceptable results.

#### User-facing label

Completed

#### Trigger condition

- the configured flow finished successfully
- no major editorial warning remains unresolved
- output is considered usable

#### What the user can do next

- read digest
- review selected articles
- use packaged content
- duplicate or rerun with changes

#### Should the run count as successful

Yes

---

### 9. `completed_with_warnings`

#### Internal meaning

The run completed and produced usable output, but there were meaningful warnings that the user should understand.

#### User-facing label

Completed with warnings

#### Trigger condition

Examples:

- digest succeeded, but source mix was weak or uneven
- hybrid source mode partly failed, but enough strong articles remained
- packaging succeeded, but some content signals were thin

#### What the user can do next

- use the result
- inspect warnings
- improve sources
- rerun with a narrower topic or different threshold

#### Should the run count as successful

Yes, but with caution

---

### 10. `insufficient_sources`

#### Internal meaning

The system could not collect enough usable source candidates to continue meaningfully.

#### User-facing label

Not enough sources

#### Trigger condition

- source list is empty
- too few sources are reachable
- too few source items are extracted after validation

#### What the user can do next

- add sources
- switch source mode
- broaden source strategy
- retry later

#### Should the run count as successful

No

---

### 11. `low_quality_sources`

#### Internal meaning

Sources worked, but too few articles passed the editorial quality threshold.

#### User-facing label

Source quality too low

#### Trigger condition

- source collection succeeded
- ranking succeeded
- article set was too weak, fragmented, or low-signal to produce a digest safely

#### What the user can do next

- narrow topic
- lower quality threshold
- add or replace sources
- change source mode

#### Should the run count as successful

No

---

### 12. `needs_topic_refinement`

#### Internal meaning

The system believes the topic is too broad or too ambiguous for a reliable run without clarification.

#### User-facing label

Needs topic refinement

#### Trigger condition

- topic broadness or ambiguity is detected before execution
- refinement flow should happen before source collection

#### What the user can do next

- choose one direction
- choose multiple directions
- keep the topic broad and continue anyway
- revise the topic manually

#### Should the run count as successful

No

---

### 13. `source_collection_failed`

#### Internal meaning

Source collection failed for technical or structural reasons.

#### User-facing label

Source collection failed

#### Trigger condition

Examples:

- source unreachable
- parsing failed
- normalization failed
- supported source returned no usable content due to extraction failure

#### What the user can do next

- retry
- change source
- change source mode
- inspect source validation details

#### Should the run count as successful

No

---

### 14. `digest_generation_failed`

#### Internal meaning

The system had enough articles to proceed, but digest generation failed.

#### User-facing label

Digest generation failed

#### Trigger condition

- digest stage raised an error
- digest payload could not be validated
- LLM response failed in a way that could not be recovered safely

#### What the user can do next

- retry
- inspect selected articles
- adjust topic or threshold
- report issue if failure persists

#### Should the run count as successful

No

---

### 15. `packaging_failed`

#### Internal meaning

The digest exists, but content packaging failed.

#### User-facing label

Content packaging failed

#### Trigger condition

- digest generation completed
- packaging stage failed or produced invalid output

#### What the user can do next

- use digest without packaging
- retry packaging
- change audience or packaging mode in a future flow

#### Should the run count as successful

Partially. The digest succeeded, but the full run did not.

---

## How This Connects to `ux-flows-v1.md`

This state model is the operational companion to [docs/ux-flows-v1.md](C:/Users/Елена/Documents/DigestFlow/docs/ux-flows-v1.md).

The flows document explains the intended user journeys. This document explains how those journeys should map to explicit system states.

High-level alignment:

- **Simple Successful Digest Flow**
  - likely states:
    - `ready_to_run`
    - `running`
    - `collecting_sources`
    - `ranking_articles`
    - `generating_digest`
    - `generating_content_package`
    - `completed`

- **Broad Topic Refinement Flow**
  - likely state:
    - `needs_topic_refinement`

- **Multiple Directions / Multiple Digest Streams Flow**
  - likely pattern:
    - one parent planning state
    - then one run state sequence per stream

- **Insufficient Quality Flow**
  - likely state:
    - `low_quality_sources`

- **User-Curated Sources Flow**
  - likely states depend on outcome:
    - `ready_to_run`
    - `collecting_sources`
    - `insufficient_sources`
    - `source_collection_failed`
    - `completed_with_warnings`

- **Source Mismatch Warning Flow**
  - this may remain a warning layer rather than a terminal state
  - but it may lead to:
    - `needs_topic_refinement`
    - `completed_with_warnings`
    - `low_quality_sources`

- **Audience Selection Flow**
  - probably modifies ranking and packaging behavior
  - does not necessarily introduce a new run state by itself

- **Quality Threshold / Strictness Flow**
  - changes the probability of:
    - `completed`
    - `completed_with_warnings`
    - `low_quality_sources`

This means run states should not replace UX flows. They should support them and make them observable.

---

## Backend Implications, but No Implementation Yet

This section is intentionally architectural only.

### 1. Run state model likely needs more than one layer

One field may not be enough forever.

In the future, the system may need to distinguish:

- lifecycle state
- outcome state
- warning state

For example:

- lifecycle:
  - running
  - generating_digest
- outcome:
  - completed
  - low_quality_sources
- warnings:
  - source mismatch
  - partial source failure

### 2. Topic refinement may need to exist before run creation

If `needs_topic_refinement` becomes a real state, the product may need a distinction between:

- a topic setup session
- an actual pipeline run

Otherwise the run model may get overloaded with pre-run UX states.

### 3. Partial success should be explicit

`packaging_failed` and `completed_with_warnings` suggest that:

- not every non-complete result is a hard failure
- not every technically successful run is a clean success

The backend should eventually treat these as first-class outcomes.

### 4. Source quality and source failure should stay distinct

The system should keep separating:

- `insufficient_sources`
- `low_quality_sources`
- `source_collection_failed`

These look similar from far away, but they imply different user actions.

### 5. Metrics and state should remain related but separate

Metrics explain why a state happened.
State explains what happened at the product level.

The state should not be inferred only by raw metrics on the UI side forever.

### 6. Parent/child stream structure may be needed

If one broad topic becomes multiple digest streams, the backend may eventually need:

- parent topic request
- child stream runs

This affects how `completed`, `low_quality_sources`, and `packaging_failed` roll up at a group level.

### 7. Existing current-state mapping

Today’s implementation already suggests partial analogues:

- `insufficient_quality` is close to `low_quality_sources`
- `failed` currently covers multiple distinct failure meanings
- `partial_failed` is close to one future version of `packaging_failed`

Future cleanup should normalize these meanings rather than expanding ambiguity.

---

## Notes

- This document does not require immediate implementation.
- It is intended to guide future pipeline and UX decisions.
- The next useful planning step is mapping these states onto entities and transitions:
  - which states belong to topic setup
  - which states belong to a run
  - which states are terminal
  - which states are warnings rather than outcomes

