# LinkedIn Post Quality Target

This document defines the target quality standard for PostFlow-generated LinkedIn posts.

PostFlow is not optimizing for one specific Personal Branding output. Digest 130 is a regression/debug case for the generation algorithm, not the target post itself.

The current staged architecture is useful, but architecture completion does not mean final post quality is complete. Guardrails and validators prevent obvious drift. The final product still needs a LinkedIn-native quality standard that is evaluated separately from whether the pipeline technically succeeds.

## Target Output

A strong PostFlow LinkedIn post should read like a practitioner making a useful point from source-grounded evidence.

It should not read like:

* an article summary;
* a corporate blog excerpt;
* generic marketing advice;
* content strategy fluff;
* a polished but empty thought-leadership paragraph.

The post should have one clear point of view, one focused reader problem, and one useful takeaway.

## LinkedIn-Native Voice

The post should sound like a person with practical judgment, not a brand article.

It should:

* make a specific claim;
* name a concrete problem or tradeoff;
* show why the point matters;
* use plain, direct language;
* avoid broad motivational phrasing.

It should avoid soft abstractions unless they are directly grounded and necessary, especially:

* visibility;
* engagement;
* trust;
* authenticity;
* growth;
* journey;
* narrative.

## Hook

The first 8-12 words should carry the hook.

Accepted hook types:

* personal action with concrete detail;
* recognizable reader pain with an unexpected cause;
* counterintuitive fact or claim.

The opening should not start as broad scene-setting or generic topic definition.

## Pattern Interrupt

The first third of the post should interrupt the expected path.

Useful pattern interrupts include:

* a reversal;
* a mistake the reader is making;
* a surprising diagnosis;
* a concrete observation;
* a visible cost the reader may not have noticed.

The pattern interrupt should move the post away from summary and toward a point of view.

## Specificity

Specificity matters more than polish.

Prefer:

* numbers;
* checks;
* named situations;
* concrete behaviors;
* concrete constraints;
* decisions;
* tradeoffs;
* examples;
* visible reader actions.

Avoid generic abstractions unless the source material makes them necessary and concrete.

## Human Tension

The post should include practical human tension, such as:

* doubt;
* a mistake;
* rethinking;
* an observed failure mode;
* a practical conflict;
* an uncomfortable tradeoff.

It should not resolve into motivational language.

## Series And Topic Continuity

A strong post should feel like part of a coherent thinking track, not a one-off recap.

This does not mean inventing a series if none exists. It means the post should have a focused niche angle that could belong to a broader body of thinking.

## Links

Generated body copy should not include links.

Source material can ground the post, but URLs should not appear in the generated body text.

## CTA

The CTA should be one open question.

Avoid:

* follow me;
* like;
* subscribe;
* salesy prompts;
* multiple CTAs.

CTA questions belong in `cta_variants`. They do not need to be duplicated inside `post_text` when the current schema keeps CTA variants separate.

## PostFlow-Specific Criteria

The final post must preserve the selected angle from staged artifacts.

It must:

* follow the staged editorial direction;
* avoid letting source-owned terms hijack the final angle;
* use source facts as grounding, not as article recap;
* include a concrete reader diagnostic when the topic supports it;
* keep one clear point of view.

If a source term is useful but could hijack the angle, the post should translate the underlying mechanism into plain language.

## Architecture Is Not Quality

The staged generation architecture creates inspectable artifacts and makes failures easier to diagnose.

It does not, by itself, guarantee a 10/10 LinkedIn post.

Current `editorial_review` score is helpful as a safety and evaluation signal, but score 8 is not the final quality target. A post can pass guardrails and still sound too generic. A post can pass editorial review and still need stronger hook, sharper tension, or more specific reader value.

Future quality work should treat the rubric below as the target, not merely provider success, schema validity, or a passing deterministic gate.

## Suggested 10-Point Evaluation Rubric

Use this checklist when reviewing generated posts:

1. Hook strength: the first 8-12 words create immediate attention.
2. Pattern interrupt: the first third breaks the expected path.
3. Specificity: the post uses concrete details, checks, constraints, examples, decisions, or tradeoffs.
4. Human/practitioner tension: the post contains a real mistake, doubt, conflict, or observed failure mode.
5. Diagnostic/actionability: the reader gets a practical check or useful next move.
6. Point of view: the post makes one clear claim instead of balancing generic advice.
7. Source grounding without recap: source facts support the point without becoming article summaries.
8. No corporate-blog tone: the post avoids polished generic business language.
9. Clean structure/readability: paragraphs are easy to scan and the post has momentum.
10. CTA quality: CTA variants are open questions, not salesy engagement bait.

## Regression Note

Digest 130 should remain a regression/debug fixture for the generation algorithm.

It is useful because it exposes drift toward source-owned terms and corporate personal-branding language. It should not become the quality target or the only scenario that defines success.
