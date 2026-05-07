# DigestFlow UX Flows v1

## Context

DigestFlow is evolving from a simple AI digest generator into an AI-powered editorial system.

The core product value is no longer just summarization. The product should help users move from broad or ambiguous topics toward usable digest streams through:

- topic refinement
- source strategy
- quality filtering
- audience-aware content packaging

This document describes the first product/UX architecture for that evolution. It is intentionally documentation-first and does not assume immediate full implementation.

---

## 1. Simple Successful Digest Flow

### Purpose

Allow a user with a reasonably specific topic to generate a useful digest with minimal friction.

### Trigger

The user enters a topic that is specific enough to support immediate source collection and ranking without refinement.

Examples:

- AI agents for support workflows
- developer productivity tools
- AI research tooling
- workflow automation for operations teams

### User steps

1. User enters a topic.
2. User optionally accepts the default source mode and default quality threshold.
3. User starts digest generation.
4. User reviews ranked articles, digest output, and optional packaged content.

### System behavior

1. Interpret the topic.
2. Select a default source strategy.
3. Collect source items.
4. Clean, deduplicate, and rank articles.
5. Keep only articles above the quality threshold.
6. If enough articles pass, generate a digest.
7. If packaging is enabled, generate a content package from the digest articles.

### UI states

- Topic input state
- Pipeline running state
- Source collection state
- Ranking complete state
- Digest ready state
- Content package ready state
- Diagnostics available state

### Backend implications

- Topic interpretation must exist before collection.
- A default source strategy must be chosen automatically when the topic is specific enough.
- Ranking must remain threshold-aware.
- Digest generation must operate on selected articles only.
- Packaging must remain grounded in the selected article set.

### Success criteria

- User gets a digest without extra clarification.
- At least the minimum number of articles pass the threshold.
- The digest reflects the selected articles rather than broad topic hype.
- Optional packaged output stays grounded in the digest article set.

### Open questions

- What exact heuristics define “specific enough” for immediate execution?
- Should source strategy be visible before the run starts, or only after?

---

## 2. Broad Topic Refinement Flow

### Purpose

Help the user turn a broad topic into a usable editorial direction before the pipeline runs.

### Trigger

The user enters a broad topic that is likely to produce weak, fragmented, or overly generic results.

Examples:

- AI
- crypto
- marketing
- education
- automation

### User steps

1. User enters a broad topic.
2. System detects that the topic is broad.
3. System suggests several topic directions.
4. User chooses one or more directions, or keeps the topic broad.

### System behavior

1. Detect topic broadness before source collection.
2. Generate a set of candidate directions.
3. Present directions as selectable refinements.
4. If the user chooses one direction, proceed with that refined topic.
5. If the user chooses several directions, move toward a split-stream flow.
6. If the user keeps the topic broad, continue with a warning that results may be low-signal.

### UI states

- Broad-topic detected state
- Suggested directions state
- Single-direction selection state
- Multi-direction selection state
- Continue-broad warning state

### Backend implications

- Topic interpretation becomes a first-class step before source collection.
- The system needs a representation of:
  - raw topic
  - refined topic directions
- A broad topic may need a different source strategy than a refined topic.

### Success criteria

- Users are not pushed straight into a low-quality run for broad topics.
- Suggested directions feel relevant and actionable.
- The system can proceed cleanly from a chosen direction.

### Open questions

- Should direction suggestions be rule-based at first or LLM-assisted?
- How many directions should be shown by default?
- Should the user be allowed to rename a suggested direction before running it?

---

## 3. Multiple Directions / Multiple Digest Streams Flow

### Purpose

Avoid mixing unrelated subtopics into one digest when the user wants coverage across several directions.

### Trigger

The user selects multiple topic directions during the refinement flow.

Example:

- Topic: AI
- Directions:
  - AI engineering
  - AI business

### User steps

1. User chooses multiple directions.
2. System explains that separate digest streams will be created.
3. User confirms the split.
4. User receives one digest stream per direction.

### System behavior

1. Convert each selected direction into its own digest stream.
2. Run collection, ranking, digest generation, and packaging independently per stream.
3. Keep results separate unless the user later asks for a cross-stream overview.

### UI states

- Multi-direction selected state
- Stream split confirmation state
- Parallel stream progress state
- Multi-stream result state

### Backend implications

- The system needs a stream-level entity or equivalent internal grouping.
- Each stream needs:
  - its own topic interpretation
  - source strategy
  - ranking output
  - digest payload
- Shared parent topic and child stream topics may both need to be represented.

### Success criteria

- Unrelated content is not forced into one digest.
- Each stream remains coherent.
- The user can understand which result belongs to which direction.

### Open questions

- Should multiple streams be separate runs or one parent run with child runs?
- Should packaging be optional per stream?

---

## 4. Insufficient Quality Flow

### Purpose

Handle the case where source collection works, but the article set is too weak for a real digest.

### Trigger

Articles are collected and ranked, but fewer than the required number pass the quality threshold.

### User steps

1. User runs the pipeline.
2. System collects and ranks articles.
3. System determines that article quality is insufficient.
4. User sees diagnostics and suggested next actions.

### System behavior

1. Stop before digest generation.
2. Stop before packaging.
3. Preserve ranking diagnostics.
4. Explain that the source worked, but the article set was not strong enough.
5. Suggest next actions:
   - narrow topic
   - lower quality threshold
   - add or change sources
   - switch source mode

### UI states

- Insufficient quality status state
- Diagnostic ranking state
- Suggested next actions state
- Digest skipped state
- Packaging skipped state

### User-facing message examples

- Недостаточно качественных статей для полноценного дайджеста. Источник обработан, но найденные материалы слишком слабые или разрозненные.
- Источник сработал, но статьи не прошли редакторский порог качества.
- Мы нашли материалы по теме, но пока не видим достаточно сильный набор для уверенного дайджеста.

### Backend implications

- Ranking must produce explainable diagnostics, not just scores.
- Pipeline status must represent insufficient quality as a first-class result.
- Digest and packaging stages must be explicitly skipped.
- Ranking output must remain available to the UI even when no digest is created.

### Success criteria

- User understands that the source worked.
- User understands why a digest was not produced.
- User can see which articles were ranked and why they were rejected.
- The system does not generate fake confidence from weak/random material.

### Open questions

- Should the system surface “recommended next action” automatically based on the failure pattern?
- Should the quality threshold be editable directly from the insufficient-quality screen?

---

## 5. User-Curated Sources Flow

### Purpose

Let users guide the source strategy directly when they know which sources matter.

### Trigger

The user wants to control the source set rather than relying only on automatic collection.

### User steps

1. User enters one or more sources:
   - RSS feed
   - blog
   - company news page
   - website
   - newsletter or source URL
2. User selects a source mode:
   - Automatic
   - Custom only
   - Hybrid
3. User starts or confirms the run.

### System behavior

1. Validate each source.
2. Normalize each source where needed.
3. Apply source mode:
   - Automatic: system uses curated or discovered sources only
   - Custom only: system uses only user-provided sources
   - Hybrid: system combines user sources with automatic discovery
4. Continue collection and ranking using the chosen mode.

### UI states

- Source entry state
- Source validation state
- Source mode selection state
- Source accepted state
- Source failed state

### Source validation states

- Valid and usable
- Valid but low-confidence
- Unsupported source pattern
- Reachable but content extraction failed
- Empty or no usable items found

### What happens if a source fails

- The user sees which source failed and why.
- In Hybrid mode, the system may continue with the remaining valid sources and automatic sources.
- In Custom only mode, the run may stop unless at least one valid source remains.

### Backend implications

- Source normalization must be explicit and reusable.
- Source mode becomes part of the run configuration.
- Source validation results need to be stored and surfaced.
- Multiple source origins must be traceable at the article level.

### Success criteria

- User can supply sources in human-facing form.
- Source mode is understandable and predictable.
- Failed sources are visible and actionable.

### Open questions

- Should source validation happen live before run start or during the run?
- How many user sources should be allowed in v1?

---

## 6. Source Mismatch Warning Flow

### Purpose

Warn the user when the chosen source is likely to produce weak results for the chosen topic direction.

### Trigger

The topic direction and the source strategy appear misaligned.

Example:

- Topic direction: AI research
- Source: dev.to/t/ai

### User steps

1. User selects a topic direction and source.
2. System detects a likely mismatch.
3. User chooses one of several paths:
   - continue anyway
   - change source mode
   - add better sources
   - narrow or adjust the topic

### System behavior

1. Compare topic direction with source characteristics.
2. If mismatch risk is high, show a warning before the run.
3. Allow the user to continue, but do not hide the risk.

### UI states

- Source mismatch warning state
- Continue anyway state
- Source adjustment state
- Topic adjustment state

### Backend implications

- Source metadata must be rich enough to estimate topic fit.
- The system needs a mismatch heuristic between:
  - topic direction
  - source type
  - source platform
  - historical quality outcomes

### Success criteria

- Users are warned before low-signal runs.
- The warning is actionable rather than vague.
- Users can still override the warning when needed.

### Open questions

- Should mismatch be based only on rules at first, or should historical pipeline outcomes influence it?
- How strong should the warning be before the system recommends refinement instead of immediate execution?

---

## 7. Audience Selection Flow

### Purpose

Make digest quality and packaging quality depend on the intended audience, not just on the topic.

### Trigger

The user starts a digest flow and either explicitly selects an audience or leaves audience unspecified.

### User steps

1. User optionally chooses an audience:
   - AI engineers
   - founders
   - marketers
   - creators
   - researchers
   - investors
   - automation specialists
2. User proceeds with the digest flow.

### System behavior

1. If audience is not selected, use a default audience assumption.
2. Use audience as a ranking and packaging modifier.
3. Prefer articles that are high-signal for the selected audience.
4. Shape packaged output around the audience’s likely questions and interests.

### UI states

- Audience optional selection state
- Audience selected state
- Default audience applied state

### Default behavior if audience is not selected

- Use a balanced editorial default.
- The default should avoid overly technical or overly executive packaging.

### Backend implications

- Audience becomes part of run configuration.
- Ranking may need audience-aware scoring rules.
- Packaging prompts should accept audience context.
- Topic quality and audience relevance should remain distinct concepts.

### Success criteria

- The same topic can produce different editorial outputs for different audiences.
- The user can understand why an article is relevant for one audience and weak for another.

### Open questions

- Should audience selection happen before topic refinement, after it, or both?
- Should multiple audiences be allowed in one run?

---

## 8. Quality Threshold / Strictness Flow

### Purpose

Give users direct control over how strict article selection should be.

### Trigger

The user wants more exploratory coverage or stronger filtering.

### User steps

1. User adjusts a strictness control.
2. User starts or reruns the digest.

### System behavior

1. Use the selected threshold during ranking.
2. Keep only articles above that threshold.
3. Increase the chance of insufficient-quality results when strictness is high.
4. Allow more exploratory sets when strictness is low.

### UI states

- Strictness slider or numeric input state
- Default threshold state
- Rerun with new threshold state

### Suggested threshold semantics

- 0.2 exploratory: more articles, more noise tolerated
- 0.4 balanced/default
- 0.7 strict: fewer but stronger articles
- 0.8+ very strict: high-signal only

### Recommended default

- 0.4 balanced/default

### How threshold affects insufficient_quality

- Higher threshold increases the chance that the run ends in insufficient quality.
- Lower threshold increases the chance of getting enough articles, but may reduce editorial coherence.

### Backend implications

- Threshold must be stored in run or topic settings.
- Ranking metrics must clearly separate:
  - configured threshold
  - actual article score distribution
- Reruns should preserve or explicitly override the threshold.

### Success criteria

- The user can understand the tradeoff between coverage and quality.
- Threshold changes have visible downstream effects in ranking diagnostics.

### Open questions

- Should threshold be exposed as a raw numeric control, a labeled slider, or both?
- Should the product suggest threshold changes after insufficient-quality runs?

---

## Implementation Notes

- This document does not require immediate full implementation.
- It should guide future UI and pipeline changes.
- The next implementation step should identify which data model and entity changes are required by these flows.
- Avoid visual polish before the functional UX is stable.
- The main goal of this phase is architectural clarity:
  - define where topic refinement happens
  - define when source strategy becomes explicit
  - define how quality outcomes are explained
  - define how multiple digest streams should be represented

