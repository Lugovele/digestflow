# LinkedIn Posting Stage Contracts

## Purpose

This document defines the structured data contracts for the clean LinkedIn posting pipeline.

The goal is to avoid ad hoc prompt outputs and prevent the new architecture from drifting back into prompt sprawl. Each stage should receive a clear object, return a clear object, and make its assumptions visible enough to test before the final post is written.

First implementation flow:

```text
selected_articles
-> article_evidence_pack
-> contextual_evidence_pack
-> angle_decision
-> post_brief
-> final_post
```

Later stages, not for first implementation:

```text
quality_review
-> targeted_repair
```

## Relationship To Existing Docs

Related documents:

- `docs/linkedin-clean-posting-flow.md`
- `docs/linkedin-post-quality-target.md`

The flow document defines stage order and responsibilities.

The quality target defines how final posts are evaluated.

This document defines the objects exchanged between stages.

## Existing Integration Boundary

Current baseline:

```text
Digest.get_articles()
-> generate_post_from_articles()
-> payload
-> ContentPackage
```

Future code should preserve:

- `generate_content_package_for_digest()` as the public integration entrypoint;
- `generator.py` as the persistence, validation, normalization, and final debug metadata boundary;
- final `ContentPackage` payload compatibility.

Staged editorial generation should live outside `generator.py`, likely in a separate module such as:

```text
services/packaging/linkedin_post_pipeline.py
```

Do not use `post_synthesis.py` as the preferred name for the clean rebuild.

## Contract: PipelineInput

Purpose:
Normalize existing digest and article inputs for the posting pipeline.

Required fields:

- `digest_id`
- `topic_name`
- `digest_title`
- `articles`
- `author_profile`

Article dictionaries may currently include:

- `url`
- `title`
- `summary`
- `key_points`
- `content_type`
- `confidence`

Validation rules:

- `articles` must be a list;
- each article must preserve source identity where available;
- `author_profile` must be normalized before prompt generation.

Must not contain:

- generated post text;
- selected angle;
- post brief;
- repair instructions.

Example JSON:

```json
{
  "digest_id": 130,
  "topic_name": "Personal Branding",
  "digest_title": "Personal Branding source review",
  "articles": [
    {
      "url": "https://example.com/article",
      "title": "Why proof matters in personal branding",
      "summary": "The article argues that visible work examples help people understand current capability.",
      "key_points": [
        "Recent examples make expertise easier to evaluate.",
        "Visual polish alone does not show decision quality."
      ],
      "content_type": "article",
      "confidence": 0.82
    }
  ],
  "author_profile": {
    "author_role": "AI Automation Specialist",
    "author_focus": "practical automation and workflow clarity"
  }
}
```

## Contract: SelectedArticle

Purpose:
Represent one article selected for post generation.

Required fields:

- `source_index`
- `title`
- `url`
- `summary`
- `key_points`

Note:
Internal contracts use zero-based `source_index`, matching Python list indexing. UI or debug output may derive human-readable numbering as `source_index + 1`.

Optional fields:

- `source_name`
- `published_at`
- `content_type`
- `confidence`

Validation rules:

- `source_index` must be stable within a run;
- `source_index` is zero-based and must match the article position in the selected article list;
- `title` should be non-empty when available;
- `summary` should be non-empty for evidence extraction;
- `key_points` should be a list.

Must not contain:

- generated post wording;
- angle decision;
- author opinion;
- repair notes.

Example JSON:

```json
{
  "source_index": 0,
  "title": "Why proof matters in personal branding",
  "url": "https://example.com/article",
  "summary": "The article argues that visible work examples help people understand current capability.",
  "key_points": [
    "Recent examples make expertise easier to evaluate.",
    "Visual polish alone does not show decision quality."
  ],
  "source_name": "Example Source",
  "published_at": "2026-06-01",
  "content_type": "article",
  "confidence": 0.82
}
```

## Contract: ArticleEvidence

Purpose:
Represent one usable evidence fragment extracted from one selected article.

Required fields:

- `evidence_id`
- `source_index`
- `source_title`
- `evidence_text`
- `evidence_type`
- `specificity_level`
- `source_limitations`

Allowed `evidence_type` values:

- `fact`
- `pattern`
- `example`
- `contrast`
- `warning`
- `practical_point`

Allowed `specificity_level` values:

- `high`
- `medium`
- `low`

Validation rules:

- `evidence_text` must be source-grounded;
- `evidence_text` should be specific enough to support a post;
- `source_index` must refer to the zero-based index of the source article in the selected article list;
- low-specificity evidence should not become main proof;
- evidence must keep source traceability.

Must not contain:

- invented facts;
- whole-article summaries;
- final post phrasing;
- controlling angle.

Example JSON:

```json
{
  "evidence_id": "e1",
  "source_index": 0,
  "source_title": "Why proof matters in personal branding",
  "evidence_text": "Recent examples of decisions and tradeoffs make expertise easier to evaluate than a polished profile alone.",
  "evidence_type": "contrast",
  "specificity_level": "medium",
  "source_limitations": "The article does not provide a numeric study or named case."
}
```

## Contract: ArticleEvidencePack

Purpose:
Group all extracted evidence from selected articles.

Required fields:

- `items`
- `usable_count`
- `rejected_count`

Optional fields:

- `rejection_reasons`
- `coverage_notes`

Validation rules:

- `items` must be a list of `ArticleEvidence`;
- `usable_count` should match usable items;
- `rejected_count` should make weak source sets visible;
- the pack should expose whether there is enough evidence for post generation.

Must not contain:

- final post text;
- post brief;
- repair plan.

Example JSON:

```json
{
  "items": [
    {
      "evidence_id": "e1",
      "source_index": 0,
      "source_title": "Why proof matters in personal branding",
      "evidence_text": "Recent examples of decisions and tradeoffs make expertise easier to evaluate than a polished profile alone.",
      "evidence_type": "contrast",
      "specificity_level": "medium",
      "source_limitations": "No metric or named case."
    }
  ],
  "usable_count": 1,
  "rejected_count": 2,
  "rejection_reasons": [
    "generic marketing language",
    "unsupported causal claim"
  ],
  "coverage_notes": "Evidence supports a contrast between polished presentation and proof of judgment."
}
```

## Contract: ContextualEvidence

Purpose:
Add editorial context to an evidence fragment.

Required fields:

- `evidence_id`
- `evidence_text`
- `what_it_says`
- `supports_argument`
- `best_use_in_post`
- `do_not_use_for`
- `risk_of_misuse`

Allowed `best_use_in_post` values:

- `hook`
- `tension`
- `proof`
- `practical_point`
- `ending`
- `background_only`

Validation rules:

- must explain how evidence may support an argument;
- must preserve the original evidence meaning;
- must prevent source terminology from hijacking the controlling angle;
- `background_only` evidence must not be promoted into main proof.

Must not contain:

- final post text;
- final angle decision;
- unsupported interpretation;
- invented author experience.

Example JSON:

```json
{
  "evidence_id": "e1",
  "evidence_text": "Recent examples of decisions and tradeoffs make expertise easier to evaluate than a polished profile alone.",
  "what_it_says": "The source contrasts visible proof of work with surface-level presentation.",
  "supports_argument": "A reader's public content should show judgment, not only finished outcomes.",
  "best_use_in_post": "tension",
  "do_not_use_for": "Do not turn this into a broad claim that visuals never matter.",
  "risk_of_misuse": "Could drift into generic personal branding advice."
}
```

## Contract: ContextualEvidencePack

Purpose:
Group all contextualized evidence and expose editorial risks.

Required fields:

- `items`
- `main_candidate_evidence_ids`
- `background_evidence_ids`
- `risks`

Validation rules:

- `items` must be a list of `ContextualEvidence`;
- main candidate IDs must refer to existing evidence;
- risks should identify likely misuse such as source-term drift, generic summary, or unsupported claims.

Must not contain:

- final post text;
- post brief;
- repair plan.

Example JSON:

```json
{
  "items": [
    {
      "evidence_id": "e1",
      "evidence_text": "Recent examples of decisions and tradeoffs make expertise easier to evaluate than a polished profile alone.",
      "what_it_says": "The source contrasts visible proof of work with surface-level presentation.",
      "supports_argument": "A reader's public content should show judgment, not only finished outcomes.",
      "best_use_in_post": "tension",
      "do_not_use_for": "Do not make visual identity the main angle.",
      "risk_of_misuse": "Could drift into generic personal branding advice."
    }
  ],
  "main_candidate_evidence_ids": ["e1"],
  "background_evidence_ids": [],
  "risks": [
    "source-term drift",
    "generic summary"
  ]
}
```

## Contract: AngleDecision

Purpose:
Select one controlling angle for the post.

Required fields:

- `controlling_angle`
- `reader_problem`
- `author_position`
- `main_tension`
- `supporting_evidence_ids`
- `angle_to_avoid`
- `authorial_voice_directive`

The `authorial_voice_directive` object must include:

- `authorial_observation`: what the author notices in the selected evidence;
- `rejected_reading`: the tempting but unsupported reading the writer must avoid;
- `why_distinction_matters`: why the distinction matters for the reader;
- `personal_presence_requirement`: bounded policy, currently
  `explicit_author_owned_statement_required` for this final-post flow;
- `first_person_policy`: currently `allowed_not_required`, meaning first person is permitted but not required;
- `forbidden_author_claims`: author claims the writer must not invent, such as personal experience, professional authority, direct market exposure, client or customer stories, invented emotional reaction, or biographical claims.

Validation rules:

- exactly one controlling angle;
- `reader_problem` must be concrete;
- `author_position` must be visible;
- supporting evidence IDs must refer to contextualized evidence;
- `angle_to_avoid` should prevent likely drift;
- `authorial_voice_directive` must derive from the selected evidence relationship and preserve author perspective without inventing biography, credentials, client work, market exposure, emotion, or personal experience;
- `personal_presence_requirement` must be one of
  `explicit_author_owned_statement_required`, `author_owned_statement_allowed`,
  or `editorial_stance_only`.

Must not contain:

- multiple competing post ideas;
- article summary;
- generic AI commentary;
- final post text.

Example JSON:

```json
{
  "controlling_angle": "A polished profile is weak proof if recent posts do not show decisions, tradeoffs, or lessons.",
  "reader_problem": "The reader is showing finished work but not the thinking behind it.",
  "author_position": "Proof of judgment matters more than surface polish.",
  "main_tension": "Finished outcomes look credible, but they can hide how the person actually works.",
  "supporting_evidence_ids": ["e1"],
  "angle_to_avoid": [
    "Do not make this a broad post about visual identity, authenticity, or personal brand strategy."
  ],
  "authorial_voice_directive": {
    "authorial_observation": "The author notices that polished output and visible judgment should not be treated as the same proof.",
    "rejected_reading": "Reject treating polished output as proof that judgment has been demonstrated.",
    "why_distinction_matters": "The distinction matters because readers need evidence of thinking, not another neutral recap of branding advice.",
    "personal_presence_requirement": "explicit_author_owned_statement_required",
    "first_person_policy": "allowed_not_required",
    "forbidden_author_claims": [
      "personal experience",
      "professional authority",
      "direct market exposure",
      "client or customer stories",
      "invented emotional reaction",
      "biographical claims"
    ]
  }
}
```

## Contract: PostBrief

Purpose:
Convert the selected angle into a compact writing plan.

Required fields:

- `opening_direction`
- `pattern_interrupt`
- `core_point`
- `evidence_to_use`
- `practical_point`
- `ending_direction`
- `cta_direction`

Each `evidence_to_use` item must include:

- `evidence_id`
- `evidence_text`
- `role_in_post`

Validation rules:

- must preserve `controlling_angle`;
- must not introduce a new angle;
- each evidence item must have a role;
- practical point must be specific enough to guide writing;
- brief should be compact and not become a draft.

Must not contain:

- full final post;
- unrelated evidence;
- repair instructions.

Example JSON:

```json
{
  "opening_direction": "Start with the gap between polished outcomes and visible judgment.",
  "pattern_interrupt": "The issue is not whether the profile looks strong. The issue is whether recent content proves how the person thinks.",
  "core_point": "A reader needs evidence of decisions, tradeoffs, and lessons, not another surface-level update.",
  "evidence_to_use": [
    {
      "evidence_id": "e1",
      "evidence_text": "Recent examples of decisions and tradeoffs make expertise easier to evaluate than a polished profile alone.",
      "role_in_post": "Use as the proof point for why visible work process matters."
    }
  ],
  "practical_point": "Ask whether the last ten posts show a decision, tradeoff, mistake, lesson, or shipped artifact.",
  "ending_direction": "End by reframing proof as visible judgment, not presentation polish.",
  "cta_direction": "Ask one open question about what kind of proof the reader looks for in someone's work."
}
```

## Contract: FinalPostPayload

Purpose:
Final payload compatible with current `ContentPackage` validation.

Required fields:

- `post_text`
- `hook_variants`
- `cta_variants`
- `hashtags`
- `quality_checks`

Optional fields:

- `carousel_outline`

Validation rules:

- `post_text` must be non-empty;
- `post_text` must be under 1300 characters;
- `hook_variants` must contain at least 3 non-empty strings;
- `cta_variants` must contain at least 3 non-empty strings;
- `hashtags` must contain at least 1 hashtag;
- `quality_checks.uses_only_provided_facts` must be boolean;
- `quality_checks.has_clear_point_of_view` must be boolean;
- `quality_checks.linkedin_ready` must be boolean;
- `carousel_outline` may be omitted or an empty list.

Rules:

- final post should be written from `PostBrief`, not raw articles;
- should not introduce unsupported claims;
- should not invent personal experience;
- should not include external links in the body;
- should have one CTA in the post text.

Example JSON:

```json
{
  "post_text": "A polished profile is weak proof if recent posts do not show how you think.\n\nFinished work can look impressive and still hide the important part: the decision behind it.\n\nLook at the last ten posts. How many show a tradeoff, mistake, lesson, or shipped artifact?\n\nThe profile helps people recognize you. The content stream helps them understand what they can trust you to solve.\n\nWhat kind of proof do you look for before you trust someone's work?",
  "hook_variants": [
    "A polished profile is weak proof.",
    "Finished work can hide the important part.",
    "Your last ten posts may reveal less than you think."
  ],
  "cta_variants": [
    "What kind of proof do you look for before you trust someone's work?",
    "What makes someone's expertise feel real to you?",
    "Which matters more to you: the result or the decision behind it?"
  ],
  "hashtags": ["#PersonalBranding", "#ThoughtLeadership", "#LinkedInContent"],
  "quality_checks": {
    "uses_only_provided_facts": true,
    "has_clear_point_of_view": true,
    "linkedin_ready": true
  },
  "carousel_outline": []
}
```

## Later Contract: QualityReviewResult

This is a later contract, not part of the first implementation.

Purpose:
Represent post evaluation using `docs/linkedin-post-quality-target.md`.

Required fields:

- `scores`
- `total_score`
- `pass`
- `failed_criteria`
- `automatic_fail_reason`

Rules:

- scores must cover all 9 rubric criteria defined in `docs/linkedin-post-quality-target.md`;
- total score max is 45;
- pass threshold is 36 out of 45;
- required minimums are `hook >= 4`, `controlling_angle >= 4`, `author_point_of_view >= 4`, `human_voice >= 4`, and `evidence >= 3`;
- under `explicit_author_owned_statement_required`, `author_point_of_view = 5`
  requires exactly one qualifying explicit author-owned interpretive statement;
- `human_voice` remains separate and evaluates naturalness, rhythm, clarity,
  non-corporate language, and non-generic prose;
- automatic fail reason should be explicit when present.

Example JSON:

```json
{
  "scores": {
    "hook": 4,
    "controlling_angle": 5,
    "reader_problem": 4,
    "pattern_interrupt": 4,
    "evidence": 3,
    "author_point_of_view": 5,
    "human_voice": 4,
    "practical_value": 4,
    "cta": 4
  },
  "total_score": 37,
  "pass": true,
  "failed_criteria": [],
  "automatic_fail_reason": ""
}
```

## Later Contract: TargetedRepairPlan

This is a later contract, not part of the first implementation.

Purpose:
Represent a targeted repair instruction.

Required fields:

- `failed_criterion`
- `repair_scope`
- `repair_instruction`
- `preserve`
- `avoid`

Rules:

- repair must target the failed criterion;
- repair must preserve the controlling angle unless angle failure is the issue;
- repair must not blindly rewrite the whole post unless needed;
- repair must not add unsupported claims.

Example JSON:

```json
{
  "failed_criterion": "hook",
  "repair_scope": "first_line_only",
  "repair_instruction": "Make the first line sharper and more specific without changing the angle.",
  "preserve": [
    "controlling angle",
    "source-grounded claim",
    "reader diagnostic"
  ],
  "avoid": [
    "generic opening",
    "new facts",
    "new source terminology"
  ]
}
```

## Layer Boundaries

- `generator.py` should not become the editorial pipeline.
- `generator.py` should keep integration, persistence, final validation, normalization, and debug metadata.
- staged editorial logic should live in a separate module.
- carousel should not be mixed into the first posting rewrite unless explicitly scoped.
- quality review and targeted repair should not be added before base generation is stable.

## Implementation Notes

- first code implementation should add deterministic data structures and boundaries before LLM prompts;
- stage outputs should be structured and contract-like, not freeform prose;
- each stage should be testable independently;
- preserve final payload compatibility with `validators.py`;
- do not start by adding repair prompts.
