# DESIGN.md — Decoded Brand Design System

## 1. Brand Essence & Core Values
* **Clarity in Communication:** Communication must be clear, concise, and easy to follow across all touchpoints.
* **Singularity of Objective:** Each piece of content or UI artifact must focus on one single, clear objective.
* **Simplicity in Execution:** Keep layouts, interactive workflows, and visual elements simple and intuitive[cite: 1, 2].
* **Striving for Perfection:** Maintain rigorous standards of visual alignment, typography, and code structure[cite: 1, 2].
* **Global Touchpoint Consistency:** Maintain visual consistency across all touchpoints globally, including London, Amsterdam, New York, Sydney, and Hong Kong[cite: 1, 2].

## 2. Color Tokens (CSS Variables)
All generated HTML artifacts must declare and use the following CSS variables inside the `<head>` block:

```css
:root {
  /* Ink & Text Colors */
  --ink: #1f232e;          /* Primary body text, active borders, and primary headings */
  --ink-soft: #2b2c30;     /* Secondary lead text */
  --muted: #666666;        /* Captions, metadata, and inactive labels */

  /* Surface & Background Colors */
  --background: #ffffff;   /* Primary document canvas background */
  --header-band: #f5f5f2;  /* Hero headers and secondary container backgrounds */
  --card: #ffffff;         /* Elevated card component background */
  --quiet: #fafafa;        /* Inactive or completed item backgrounds */

  /* Brand Accent Colors */
  --yellow: #fef800;       /* Signature Decoded accent yellow */
  --yellow-soft: #fffec2;  /* Soft yellow highlight for hovered or selected states */

  /* Border & Divider Colors */
  --line: #d9d9d9;         /* Standard subtle container borders */
  --line-dark: #969696;    /* Darker border lines for input elements and check icons */

  /* Structural Variables */
  --radius: 4px;           /* Standard border radius for cards */
}