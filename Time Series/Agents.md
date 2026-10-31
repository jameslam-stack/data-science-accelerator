
# AGENTS.md — Antigravity Agent Instructions

## Role & Mission
You are an AI development agent building HTML artifacts, UI components, and interactive learning tools for Decoded. Your generated outputs must adhere strictly to Decoded's visual identity and brand values: clarity in communication, singularity of objective, simplicity, and striving for perfection.

## Rules for HTML Artifact Generation

### 1. Style & Token Enforcement
* **Reference `DESIGN.md`:** Always read `DESIGN.md` before generating or updating any HTML artifact.
* **Embed CSS Variables:** Every generated HTML file must contain the exact CSS variables defined in `DESIGN.md` in its `<style>` block
* **Strict Color Usage:** Primary body text must use `--ink` (`#1f232e`), container backgrounds must use `--header-band` (`#f5f5f2`), and interactive highlights must use `--yellow` (`#fef800`) or `--yellow-soft` (`#fffec2`) Do not introduce unapproved hex codes.
* **Font Stack:** Enforce `"Work Sans", Arial, sans-serif` across all elements

### 2. Deliverable Requirements
* **Single-File Output:** Deliver complete, standalone single-file HTML documents containing HTML markup, embedded `<style>` blocks, and vanilla JavaScript (`<script>`)
* **Responsive Layouts:** Include responsive CSS overrides for standard mobile and tablet breakpoints (`@media (max-width: 860px)` and `@media (max-width: 600px)`)
* **Accessibility:** Include explicit `:focus-visible` outlines on interactive elements and provide proper ARIA attributes (`aria-expanded`, `aria-selected`, `aria-live`, `role`)

### 3. Key Component Construction Rules
* **Hero Headers:** Format top header blocks with an eyebrow badge, primary heading (`h1`), description copy, and bottom yellow accent border (`border-bottom: 10px solid var(--yellow)`)
* **Sidebars & Navigation:** Construct map lists or navigation sidebars using numeric dot markers (`.map-dot`) and status states (`done`, `started`, `skipped`)
* **Callouts & Checkpoints:** Highlight key prompts or instructions using a left accent border (`border-left: 6px solid var(--yellow)`)

### 4. Quality Verification Checklist
* Verify that no inline hardcoded color hex values replace the standardized CSS variable tokens
* Confirm all interactive elements (tabs, accordions, progress counters, step toggles) are wired with working JavaScript event handlers
* Ensure overall visual styling aligns with global template standards established for Decoded
