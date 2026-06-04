# Frontend Style Guide

## Purpose

This document defines frontend implementation standards for PostFlow/DigestFlow.

It should be used when creating, refactoring, or reviewing:
- Django templates
- HTML
- CSS/SCSS
- small JavaScript interactions
- UI and E2E tests

The goal is to keep the UI implementation readable, accessible, testable, and safe for Codex-driven development.

## When to use this guide

Use this guide when:
- creating or editing Django templates
- editing UI CSS/SCSS
- adding or changing small JavaScript interactions
- refactoring UI code
- writing or updating UI/E2E tests
- reviewing Codex-generated UI code
- closing UI-heavy layers or features

## Priority rules for this project

- Use semantic HTML for interactive and structural elements.
- Use real `<button>`, `<nav>`, `<main>`, `<section>`, `<header>`, and `<footer>` elements where appropriate.
- Do not use clickable `<div>` elements for buttons.
- Keep templates readable and consistently formatted.
- Put multiple attributes on separate lines for complex elements.
- Use stable `data-testid` attributes for UI/E2E tests.
- Do not rely on CSS classes as test selectors.
- Keep CSS nesting shallow if SCSS is used.
- Use BEM-style or component-scoped class naming for isolated UI blocks.
- Avoid unnecessary `!important`.
- Extract repeated template fragments only when repetition becomes hard to maintain.
- Preserve existing behavior unless the task explicitly asks to change it.
- Avoid broad cosmetic rewrites.

## HTML and Django templates

### Semantic HTML

Use semantic elements for structure and interaction:
- `<button>` for actions
- `<a>` for navigation
- `<nav>` for navigation groups
- `<main>` for the main page content
- `<section>` for meaningful page sections
- `<header>` and `<footer>` where appropriate

Do not use clickable `<div>` or `<span>` elements for actions.

Prefer:

```html
<button
  type="button"
  data-testid="view-all-ideas-button"
>
  View all ideas
</button>
```

Avoid:

```html
<div
  class="button"
  onclick="..."
>
  View all ideas
</div>
```

### Buttons and links

Use buttons for in-page actions:

- opening a modal
- expanding or collapsing content
- submitting a form
- triggering generation or research actions

Use links for navigation:

- opening history
- moving to another page
- going back to a previous route

### Forms and inputs

Use proper labels for form fields.

Prefer explicit label/input relationships where possible.

Keep form actions and submit buttons clear and user-facing.

### Readable template structure

Keep templates easy to scan.

For complex elements with multiple attributes, place each attribute on a separate line.

Prefer:

```html
<form
  method="post"
  action="{% url 'topic_create' %}"
  data-testid="topic-create-form"
>
```

Avoid:

```html
<form method="post" action="{% url 'topic_create' %}" data-testid="topic-create-form">
```

### Conditional blocks

Keep Django template conditionals readable.

Avoid deeply nested conditional logic in templates when possible.

If template logic becomes hard to read, move the decision into the view context.

### Repeated fragments

Repeated markup can stay inline when it is small and local.

Extract or centralize repeated fragments when:

- the same block appears in multiple templates
- the block is large
- the block has complex conditions
- the repetition makes future changes risky

Do not over-abstract small, one-off markup.

### Accessibility basics

Use:

- semantic elements
- clear button text
- labels for form fields
- meaningful headings
- accessible empty states
- keyboard-friendly controls

Avoid UI controls that only work with mouse interactions.

### Empty states

Empty states should explain:

- what happened
- what the user can do next
- why the page is not broken

Keep empty-state language user-facing and product-aligned.

### Stable test selectors

Use `data-testid` for UI/E2E selectors.

Prefer:

```html
<button
  type="button"
  data-testid="show-less-ideas-button"
>
  Show less
</button>
```

Avoid relying on:

- CSS classes
- exact layout structure
- fragile text that may change during UX copy revisions

## CSS / SCSS

### Scope visual changes

Keep CSS changes scoped to the task.

Do not perform broad visual rewrites unless the task explicitly asks for them.

### Shallow nesting

If SCSS is used, keep nesting shallow.

Avoid nesting deeper than 3 levels.

Prefer readable flat selectors over deeply nested rules.

### Class naming

Use readable, component-scoped class names.

BEM-style naming is preferred for isolated UI blocks when useful:

```css
.idea-card {}

.idea-card__title {}

.idea-card__meta {}

.idea-card--active {}
```

Avoid vague names like:

```css
.box {}

.wrapper {}

.content {}
```

unless the context is very clear.

### Avoid unnecessary `!important`

Do not use `!important` unless there is a specific reason.

If `!important` is unavoidable, add a short reason comment.

Example:

```css
/* reason: overrides third-party inline widget style */
.some-class {
  display: none !important;
}
```

### Avoid duplicated values

When project-level variables, tokens, or shared values exist, reuse them instead of duplicating magic values.

If the project does not yet have shared variables, keep values local and consistent.

### Preserve visual intent

When editing existing UI, preserve the current visual intent unless the task explicitly asks for a design change.

## JavaScript

### Keep JavaScript minimal

Use small JavaScript interactions only where needed.

Avoid introducing a frontend framework pattern unless the project explicitly adopts that framework.

### Keep behavior scoped

JavaScript should be scoped to the relevant page or UI block.

Avoid global behavior unless it is intentionally shared.

### Prefer predictable interactions

UI behavior should be obvious and testable:

- buttons should do one clear thing
- expand/collapse states should be reversible
- navigation should not be hidden inside non-semantic elements

### Avoid inline complexity

Small inline behavior may be acceptable for simple cases.

If logic becomes complex, move it into a named function or separate script.

### Preserve existing behavior

Do not rewrite JavaScript behavior unless the task explicitly asks for it.

## Testing

### Test selectors

Use `data-testid` for UI/E2E selectors.

Do not rely on CSS classes as test selectors.

CSS classes are for styling. `data-testid` is for tests.

### Test behavior

Test user-visible behavior, not implementation details.

Good examples:

- the user sees only 3 recent ideas by default
- clicking `View all` opens the history page
- clicking `Show less` collapses the list
- an empty state appears when there are no ideas

Avoid tests that depend on incidental DOM structure.

### Test names

Use clear behavior-focused test names.

Preferred pattern:

```python
def test_should_show_three_recent_ideas_by_default_when_history_has_more_items():
    ...
```

or, for JS/E2E style tests:

```js
it('should show three recent ideas when history has more items', () => {
  ...
});
```

### Preserve existing behavior

When updating tests, preserve existing behavior unless the task explicitly asks to change it.

Do not update tests just to match accidental regressions.

## Codex usage

Use this instruction block in Codex prompts when touching frontend files:

```text
Follow docs/frontend-style-guide.md for all touched frontend files.
Apply the guide only within the scope of this task.
Do not perform broad cosmetic rewrites.
Preserve existing behavior unless the task explicitly asks to change it.
Use semantic HTML, readable templates, scoped CSS, minimal JavaScript, and data-testid for UI/E2E tests.
```
