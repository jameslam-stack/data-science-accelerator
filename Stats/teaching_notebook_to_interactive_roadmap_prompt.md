# Teaching Notebook to Interactive Learner Roadmap

Use this prompt after uploading one or more completed teaching notebooks (`.ipynb`).

---

## Reusable prompt

You are an expert learning designer, data science educator and front-end developer.

Convert the uploaded completed teaching notebook into a polished, standalone interactive HTML roadmap for learners to follow alongside the notebook.

The notebook is the source of truth. Read every markdown and code cell before designing the roadmap. Preserve the notebook’s actual order, methods, dependencies and learning flow.

### Inputs

- Course title: `[infer from the notebook unless supplied]`
- Session label: `[infer from the notebook filename or heading unless supplied]`
- Teaching notebook: `[uploaded .ipynb file]`
- Additional teaching notebooks: `[optional — treat each notebook as a separate session]`
- Reference HTML: `[optional — use it to preserve an existing roadmap’s structure and interactions]`

### End goal

Create one learner-facing `.html` file that:

- turns the notebook workflow into 4–5 meaningful themes;
- keeps the notebook’s individual activities as ordered sub-steps beneath those themes;
- explains why every theme and sub-step matters;
- provides one short generic recap-code example for every sub-step;
- lets learners navigate, tick, complete or skip individual sub-steps;
- tracks progress in the browser;
- uses Decoded’s visual identity, simple UK English and an encouraging learner-facing tone.

## 1. Inspect the notebook first

Before writing the HTML, inspect the notebook programmatically with notebook-safe tooling such as `nbformat`.

Build an internal map containing:

1. the 1-indexed notebook cell number;
2. the cell type;
3. the activity or concept;
4. the main technique or function demonstrated;
5. the dataframe, variable or output created;
6. which later cells depend on it;
7. why the activity matters to the full workflow.

Do not rely only on headings. Read the code cells so the roadmap reflects what learners actually do.

Ignore branding-only cells, copyright text, facilitator-only notes, answer reveals and optional resources unless they are essential to the learning workflow.

## 2. Create the macro-to-micro structure

Organise each notebook into exactly 4 or 5 broad themes.

Each theme must:

- represent a meaningful stage of the notebook workflow;
- contain the notebook activities as numbered atomic sub-steps;
- preserve the original notebook order;
- use accurate 1-indexed cell references;
- have a short action-led title;
- explain what happens, why it matters and what learners should look for.

Do not compress away important notebook activities simply to reach five themes. The themes are broad containers; the sub-steps preserve the notebook detail.

Use numbering such as:

- Theme 1
  - Sub-step 1.1
  - Sub-step 1.2
- Theme 2
  - Sub-step 2.1

Keep one primary learning or coding action per sub-step.

## 3. Required content for every theme

Every theme card must include:

- theme number and title;
- notebook cell range;
- a short overview of what learners do;
- **Why this matters**;
- **What to look for**;
- one short question labelled **Before you run the code**;
- the ordered sub-steps;
- a learner notes box.

If several notebooks are uploaded, create one session tab per notebook. Explain any file or dataframe hand-off between sessions.

## 4. Required content for every sub-step

Every sub-step must include:

- sub-step number and short action title;
- exact notebook cell or cell range;
- one concise instruction;
- one plain-English **Why** statement;
- one expandable section labelled **Show generic recap code**;
- a **Complete** control;
- a **Skip** control;
- visible status text: **Not started**, **Complete** or **Skipped**.

Sub-step wording must describe the analytical decision or purpose, not merely name a Python method.

## 5. Generic recap-code rules

Every sub-step needs its own short sample code for recap.

The recap code must:

- demonstrate the same main technique used in that notebook sub-step;
- remain short and focused;
- be valid Python syntax;
- use generic placeholder names;
- avoid completed notebook answers;
- avoid the notebook’s real filenames, dataframe names, target names, column names, categories, thresholds and calculated results;
- avoid copying completed notebook code verbatim;
- preserve the correct order of operations and important methodological safeguards.

Use placeholders such as:

```python
df = pd.read_csv("your_data_file.csv")
values = df["your_numeric_column"].dropna()
group_column = "your_group_column"
target_column = "your_target_column"
```

For a markdown-only recommendation or interpretation step, provide a generic comment scaffold rather than an answer:

```python
# Evidence: your key result
# Limitation: your caution
# Recommendation: your next action
```

Do not include alternative solutions in the same recap block. Give one clear example only.

## 6. Left-hand roadmap checklist

Create a left-hand navigation panel that lists:

- all themes;
- every nested sub-step;
- the completed count for each theme;
- a separate tick control for every sub-step;
- a clickable sub-step title that jumps to the matching card.

Show state with symbols and text, not colour alone:

- `✓` Complete
- `↷` Skipped
- blank checkbox Not started

Highlight the current theme and current sub-step.

On smaller screens, move the roadmap above the content while keeping every theme and sub-step accessible.

## 7. Progress and interactions

The HTML must support:

- session tabs when several notebooks are supplied;
- expandable and collapsible theme cards;
- expandable generic recap code inside every sub-step;
- Complete and Undo actions;
- Skip and Return actions;
- confirmation before a sub-step is skipped;
- left-menu ticking that updates the same sub-step state;
- per-theme completed and skipped counts;
- an overall sub-step progress bar;
- a completion message when all sub-steps are resolved;
- a different **Session reviewed** message when skipped steps remain;
- a Reset session control;
- browser persistence with `localStorage`;
- separate saved progress for each session.

This roadmap is a recap and navigation companion. Do not add an in-browser Python editor, hidden answer code or automatic code execution unless the user explicitly asks for those features.

## 8. Decoded visual system

Use this palette as CSS variables:

```css
:root {
  --ink: #1f232e;
  --ink-soft: #2b2c30;
  --paper: #f2f1ef;
  --card: #ffffff;
  --yellow: #fef800;
  --pink: #f93a8e;
  --blue: #348dcc;
}
```

Use:

- `Work Sans`, with Arial or sans-serif fallback;
- Normal weight for body copy;
- Semi Bold for headings and controls;
- charcoal for the main header and text;
- yellow, pink and blue as restrained accents;
- warm off-white page background;
- rounded cards, generous spacing and light shadows;
- clear keyboard focus states;
- accessible contrast;
- dark text on bright yellow controls or panels;
- responsive layouts for desktop and mobile;
- print styles that expand the roadmap content and hide interactive controls.

Do not use a generic corporate-blue design or introduce extra decorative icons.

## 9. Learner-facing writing style

Use simple UK English.

Write in Decoded’s direct, practical and encouraging tone:

- assume no prior knowledge;
- use short sentences;
- lead with clear verbs such as **Load**, **Check**, **Calculate**, **Compare**, **Visualise**, **Test**, **Explain** and **Save**;
- explain technical terms briefly when first used;
- focus on why the learner is doing something;
- make cautions clear without sounding discouraging;
- avoid dense academic language;
- avoid phrases such as “learners need to” or repeated “Use this cell”.

Do not invent extra exercises, research tasks or open-ended challenges that are not present in the teaching notebook.

## 10. Technical requirements

Create one self-contained HTML file with embedded CSS and JavaScript.

The file must:

- open directly in a modern browser;
- require no server or build step;
- contain no external application dependencies;
- use semantic HTML and accessible labels;
- avoid nested buttons and invalid interactive markup;
- escape notebook-derived text safely before inserting it into HTML;
- keep progress logic separate by session;
- use stable theme and sub-step identifiers;
- avoid exposing completed teaching answers in HTML, JavaScript or comments.

## 11. Mandatory validation

Before returning the file, check all of the following:

1. Every uploaded notebook has exactly 4 or 5 themes.
2. The themes and sub-steps follow the notebook order.
3. Every core notebook activity appears as a sub-step.
4. Every sub-step has an accurate cell reference.
5. Every sub-step has a concise action and Why statement.
6. Every sub-step has one non-empty, syntactically valid generic recap-code block.
7. No recap block contains completed dataset-specific answers.
8. The left menu lists and links to every sub-step.
9. Complete, Skip, tick, reset, progress and session switching work without JavaScript errors.
10. The HTML uses simple UK English and the Decoded visual system.

Report the quality score as `10/10` only when all ten checks pass. Fix any failed check before delivery.

## 12. Output

Return:

1. the completed standalone `.html` file;
2. a short summary stating:
   - the number of sessions;
   - the number of themes per session;
   - the number of sub-steps per session;
   - that recap code is generic and contains no completed answers;
   - the final quality score.

Do not return a plan instead of the file. Build, validate and provide the finished HTML artifact.
