<!--
  Purpose: Quick, repository-specific instructions for AI coding assistants.
  Keep this file short and prescriptive. When updating, preserve any
  existing human-authored guidance and only add discoverable facts.
-->

# Copilot / AI agent instructions (concise)

This repository is a small, Jupyter-based data analysis workspace. The key
files and layout an AI agent should know about:

- `Veri Analizi/Parca_veriSeti.ipynb` — main analysis notebook (canonical
  source of truth for analysis and plots).
- `Veri Analizi/veri_analizi/` — a local Python virtual environment
  (contains `Scripts/`, `Lib/site-packages/` etc.). Do NOT edit or
  commit files inside this directory; treat it as a private venv.

Actionable rules for making edits or suggestions

1. Prefer non-invasive changes first
   - If you can implement your change as a new `.py` script or small
     helper (e.g., `scripts/` or `src/`) rather than modifying the notebook,
     do that. Add a short cell to the notebook that demonstrates using the
     module.
   - Avoid modifying `veri_analizi/` (the venv). If dependencies are missing,
     recommend adding a `requirements.txt` at the repo root instead of
     altering the venv files.

2. How to run locally (Windows PowerShell)
   - Activate the included venv to reproduce the original environment:

```powershell
cd 'Veri Analizi'
.\\veri_analizi\\Scripts\\Activate.ps1
# open the notebook in VS Code or Jupyter after activation
```

3. Path and naming conventions
   - Filenames use Turkish; preserve original names (e.g., `Parca_veriSeti.ipynb`).
   - Use UTF-8 encoding when creating or modifying files.

4. Editing the notebook
   - For small fixes (typos, clarifying comments), edit notebook cells directly.
   - For reproducible code, extract logic into `.py` modules and import them
     from the notebook. Keep the notebook as a readable narrative.

5. Tests and CI
   - No tests or CI config were discovered. If you add tests, place them
     under `tests/` and prefer `pytest`. Mention the test command in your
     commit message (e.g., `python -m pytest tests`).

6. Dependency handling
   - If you add or update dependencies, add or update `requirements.txt` at
     the repository root. Do not modify files under `veri_analizi/Lib/`.

7. What to avoid
   - Do not change the virtual environment files inside
     `Veri Analizi/veri_analizi/`.
   - Avoid large, opinionated refactors of the notebook without an
     explicit human-approved plan.

Representative examples
  - To make a reusable data-loading function, create `src/data_utils.py`
    and demonstrate its usage with a one-cell example in
    `Parca_veriSeti.ipynb`.
  - To capture dependencies after reproducing the environment, recommend
    `pip freeze > requirements.txt` (but do not run that command inside the
    committed venv directory).

When to ask the human
  - If a change requires modifying the virtual environment, adding CI,
    or performing a large-scale refactor of the notebook, ask the repository
    owner for guidance before proceeding.

If you find existing `.github/copilot-instructions.md`, `AGENT.md`, or a
README with overlap, merge conservatively: keep any human notes and add
only the factual, discoverable items above.

---
Please review these notes and tell me if you'd like me to:
- add a `requirements.txt` generated from the current venv,
- extract notebook code into `src/` with a short test, or
- create a small README explaining how to run the notebook.
