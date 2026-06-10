# LinkedIn Prompt Architecture

PostFlow's LinkedIn generation prompts should support a synthesis pipeline, not a single direct "articles -> post" prompt.

The target quality flow is:

```text
selected_articles
-> EvidencePack
-> PostBrief
-> FinalPost
-> ContentPackage.post_text
```

This flow lives inside the post generation/packaging step. It is called after orchestration has selected the usable sources.

## Prompt Responsibilities

`extract_source_evidence_for_post.txt`

* Produces EvidencePack-style raw material.
* Extracts mechanisms, contrasts, specific claims, examples, useful source terms, and terms to avoid.
* Should not write the final post.

`generate_author_take_from_evidence.txt`

* Produces the human editorial position that the post should develop.
* Should be grounded in EvidencePack.
* Should not invent personal experience or author biography.

`generate_post_brief_from_articles.txt`

* Produces the PostBrief.
* Makes the editorial decision before writing.
* Chooses one angle, thesis, target reader, evidence to use, evidence to ignore, hook direction, and ending frame.

`generate_post_from_articles.txt`

* Produces the FinalPost.
* Must follow the PostBrief and author take.
* Should use evidence as support, not as the subject.
* Should not recap articles or mechanically mention research.

`repair_post_quality.txt`

* Optional quality repair step.
* Should preserve the selected angle and source grounding.
* Should not become a second broad generation pass.

`review_post_editorial_quality.txt`

* Optional editorial review step.
* Should evaluate whether the final post is LinkedIn-native, specific, grounded, and aligned with the brief.

## Naming Direction

Current file names do not need to be renamed immediately.

Conceptually, future prompt names should map to:

```text
generate_evidence_pack_from_articles
generate_post_brief_from_evidence
generate_post_from_brief
```

Avoid treating prompt wording as the only quality fix. The quality layer depends on explicit intermediate artifacts: EvidencePack, PostBrief, and FinalPost.
