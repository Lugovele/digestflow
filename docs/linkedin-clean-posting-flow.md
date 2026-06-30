# LinkedIn Clean Posting Flow

## Purpose

This document defines the new clean LinkedIn posting architecture for PostFlow.

The purpose is to replace the old monolithic and prompt-heavy posting flow with a staged editorial pipeline. The new flow should make each decision visible, testable, and easier to reason about before the final post is written.

## Starting Point

This rebuild starts from:

`ae8895d refine: isolate topic setup direction flow`

At this point:

- research/source discovery is available;
- the PostFlow workspace exists;
- guided topic setup exists;
- the post result flow exists;
- topic setup is focused on direction/configuration and creating a post;
- the later posting stack is intentionally excluded.

The excluded later stack includes post synthesis, source evidence extraction prompts, author take routing, post brief prompts, repair prompts, editorial review prompts, humanization prompts, and dialogue intent prompts.

## Current Packaging Baseline

The current simple baseline is:

```text
Digest.get_articles()
-> generate_post_from_articles()
-> generate_carousel_from_articles()
-> ContentPackage payload
```

This is currently a monolithic `articles -> prompt -> post payload` flow.

The new architecture should replace the middle while preserving the external input/output contract.

Required output payload remains:

```json
{
  "post_text": "...",
  "hook_variants": ["...", "...", "..."],
  "cta_variants": ["...", "...", "..."],
  "hashtags": ["#AI"],
  "carousel_outline": [],
  "quality_checks": {
    "uses_only_provided_facts": true,
    "has_clear_point_of_view": true,
    "linkedin_ready": true
  }
}
```

Payload rules:

- `post_text` must be non-empty and under 1300 characters;
- at least 3 hooks are required;
- at least 3 CTAs are required;
- at least 1 hashtag is required;
- required quality checks are booleans;
- `carousel_outline` is optional and may be an empty list.

## Problem With The Previous Posting Architecture

The previous architecture became hard to reason about because too many responsibilities were layered together:

- source evidence;
- author take;
- post brief;
- synthesis;
- editorial review;
- repair;
- humanization;
- dialogue intent;
- banned phrase cleanup.

Failure modes:

- late prompts tried to repair earlier unclear decisions;
- source terminology could hijack the main angle;
- repair loops made the system harder to reason about;
- final posts could become article summaries or generic LinkedIn content.

## Core Principle

The final post should not be written directly from raw articles.

Raw articles should first be converted into editorial material.

The final post should be written from a selected angle and post brief.

## Intended Staged Flow

Full target flow:

```text
selected_articles
-> article_evidence_pack
-> contextual_evidence_pack
-> angle_decision
-> post_brief
-> final_post
-> quality_review
-> targeted_repair
```

The first implementation should stop at:

```text
selected_articles
-> article_evidence_pack
-> contextual_evidence_pack
-> angle_decision
-> post_brief
-> final_post
```

Quality review and targeted repair should be added only after the base generation is stable.

## Stage: selected_articles

Purpose:

Provide the selected source material to the posting layer.

Inputs:

- selected/ranked article dictionaries from the existing digest pipeline;
- source metadata already available to packaging;
- article summaries/key points from the current digest payload.

Outputs:

- a stable list of selected articles for the posting pipeline.

Rules:

- do not decide the post angle;
- do not rerun research;
- do not add new sources;
- do not filter for style preferences;
- provide selected source material to the posting layer.

## Stage: article_evidence_pack

Purpose:

Extract usable evidence from articles.

Output example:

```json
{
  "source_id": "...",
  "source_title": "...",
  "evidence": "...",
  "evidence_type": "fact | pattern | example | contrast | warning | practical_point",
  "specificity_level": "high | medium | low",
  "source_limitations": "..."
}
```

Rules:

- do not summarize the whole article;
- do not invent claims;
- prefer concrete facts, examples, numbers, contrasts, and workflow implications;
- generic claims should not become main evidence.

## Stage: contextual_evidence_pack

Purpose:

Add editorial context to evidence so the system knows how each evidence item should and should not be used.

Output example:

```json
{
  "evidence": "...",
  "what_it_says": "...",
  "supports_argument": "...",
  "best_use_in_post": "hook | tension | proof | practical_point | ending | background_only",
  "do_not_use_for": "...",
  "risk_of_misuse": "..."
}
```

Rules:

- evidence supports the author angle;
- evidence must not create a new angle by accident;
- source terminology must not become the controlling angle unless selected in `angle_decision`;
- background evidence should remain background.

## Stage: angle_decision

Purpose:

Choose one controlling angle for the post.

Output example:

```json
{
  "controlling_angle": "...",
  "reader_problem": "...",
  "author_position": "...",
  "main_tension": "...",
  "supporting_evidence_ids": ["..."],
  "angle_to_avoid": ["..."]
}
```

Rules:

- one post = one controlling angle;
- do not combine multiple possible posts;
- do not write an article summary;
- do not write generic AI commentary;
- author position must be visible;
- reader problem must be concrete.

## Stage: post_brief

Purpose:

Convert the selected angle into a compact writing plan.

Output example:

```json
{
  "opening_direction": "...",
  "pattern_interrupt": "...",
  "core_point": "...",
  "evidence_to_use": [
    {
      "evidence": "...",
      "role_in_post": "..."
    }
  ],
  "practical_point": "...",
  "ending_direction": "...",
  "cta_direction": "..."
}
```

Rules:

- the brief is not the post;
- the brief must not introduce a new angle;
- evidence must be assigned a role;
- practical point should be specific.

## Stage: final_post

Purpose:

Write the final LinkedIn post from the brief.

Required payload shape:

```json
{
  "post_text": "...",
  "hook_variants": ["...", "...", "..."],
  "cta_variants": ["...", "...", "..."],
  "hashtags": ["#AI"],
  "carousel_outline": [],
  "quality_checks": {
    "uses_only_provided_facts": true,
    "has_clear_point_of_view": true,
    "linkedin_ready": true
  }
}
```

Rules:

- write from `post_brief`, not directly from articles;
- preserve the controlling angle;
- use evidence only according to its assigned role;
- do not introduce unsupported claims;
- do not add a second main angle;
- do not over-explain;
- do not produce generic LinkedIn content.

## Later Stage: quality_review

This should not be implemented first.

It should later check:

- hook;
- angle;
- reader problem;
- pattern interrupt;
- evidence;
- author point of view;
- practical value;
- CTA;
- unsupported claims;
- links in the body;
- length.

## Later Stage: targeted_repair

This should not be implemented first.

Repair should address only failed criteria.

Examples:

- weak hook -> rewrite hook only;
- unclear angle -> rebuild around `controlling_angle`;
- source term hijacked the post -> demote source term back to evidence;
- generic language -> replace abstractions with concrete details;
- unsupported claim -> remove or ground it.

## Implementation Order

Recommended order:

1. Documentation.
2. Stage data contracts.
3. Deterministic pipeline boundary.
4. LLM-backed `article_evidence_pack`.
5. LLM-backed `contextual_evidence_pack`.
6. LLM-backed `angle_decision`.
7. LLM-backed `post_brief`.
8. LLM-backed `final_post`.
9. Quality review.
10. Targeted repair.

Do not start by adding repair prompts.
