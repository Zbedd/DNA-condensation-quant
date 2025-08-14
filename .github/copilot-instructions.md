---
applyTo: "**"
---

# Repository Ground Rules for Copilot

These rules apply to all Copilot Chat/Agent requests in this repo. Use direct, actionable steps; do not add fluff.

## 1) Style & Tone

- **No emojis.**
- Checkboxes are **rare and purposeful** (e.g., a short TODO list ≤ 3 items). Default to bullet points.

## 2) Docs & File Creation

- **Do not create folder-level `README.md` files** unless I explicitly ask.
- If documentation is needed, propose edits to the root `README.md` or an existing doc instead of making new files.

## 3) Failure Philosophy (don't over-sanitize)

- **Fail fast when required inputs/configs are missing. Do not invent defaults that could mask errors.**
- It's OK to throw errors instead of adding "safe" scaffolding.
- Example: If metadata needed to split **control vs. experimental** groups is missing or inconsistent, **stop and raise an error** with:
  - A clear error name (e.g., `ERR_MISSING_GROUPING_METADATA`).
  - Exactly which fields/columns are missing or malformed.
  - A minimal example schema to fix it.
- Avoid *broad `try/catch` that swallows* exceptions; surface actionable messages.

## 4) Push Back (collaborative, not obedient)

- You may **disagree with me**. If my request is likely harmful, brittle, or needlessly complex, say so **briefly** and propose a safer/simpler alternative.
- When challenging, keep it pragmatic: 1–2 sentences + a concrete alternative or a clarifying question.

## 5) Architecture Defaults (modular & extensible)

Design for easy expansion and single sources of truth. Assume a Python-first pipeline unless told otherwise.

- **Single source of truth for pipeline configuration** (e.g., `src/pipeline/config.py` or `pipeline.yaml`).
  - `main.py` and **validation scripts** must **import/read from the same source**—no duplicated parameter definitions.
- **Pipeline stages** live under `src/pipeline/` with a minimal interface (e.g., `run(inputs) -> outputs`).
  - Prefer a **registry/factory** so new stages auto-register and are discoverable by both `main` and validation.
- Validation imports the same stage graph & config used by `main` (no copy-paste).
- If you change interfaces/config keys:
  - Update import sites (main, validation, tests) in one pass.
  - Include a 1–2 line migration note in the PR description you draft.

## 6) When Generating or Editing Code

- Show a **unified diff** for edits where possible and include a short, itemized rationale.
- Keep exceptions explicit and typed when reasonable; avoid silent fallbacks.
- If you add a new parameter or stage, ensure it's read from the central config and automatically visible to validation.

## 7) Decision Rubric (what to do by default)

- If missing config/metadata → **raise explicit error** with fix instructions.
- If asked to add docs → **edit root docs, don't create folder READMEs** unless requested.
- If a design choice affects extensibility → prefer **registry + single-source config**.
- If my request seems risky → **push back once, propose an alternative**, then proceed as directed.

---

### Error Message Template (example)

```
ERR_MISSING_GROUPING_METADATA: required columns not found
Required: group_label, subject_id, timepoint
Found:   subject_id, timepoint
Action:  Add a `group_label` column (values: {control, experimental}) or update the mapping in pipeline config.
```

### Minimal Stage Interface (example, descriptive)

- Location: `src/pipeline/stages/`
- Contract: each stage exposes `name: str` and `run(inputs, config) -> outputs` and registers itself to a `REGISTRY` so `main` and validation can enumerate stages automatically.
