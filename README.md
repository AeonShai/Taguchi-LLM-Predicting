# Taguchi + LLM Injection Molding Quality Project

This repository contains the code and selected artifacts for a Taguchi-driven prompt optimization experiment using an LLM for injection molding quality assessment.

What's included (recommended for GitHub):
- `scripts/` — runner, prompt generator, LLM client and analysis helpers.
- `src/` — data processing, features and clustering utilities.
- `reports/Taguchi_LLM_Report.md` — project report (Markdown).
- Selected LLM outputs (lightweight): `outputs/taguchi_runs/predicted_defects_summary.json`, `outputs/taguchi_runs/run_T1.ndjson`, `outputs/taguchi_runs/taguchi_results_summary.json`.

What is intentionally excluded (in `.gitignore`):
- The local Python venv at `Veri Analizi/veri_analizi/` (do NOT commit).
- Large `outputs/` artifacts (most are ignored to keep repo small).
- `.secrets/` (API keys) — keep these locally.

Quickstart (PowerShell)

# create a new repo locally (if not already)
# git init
# create a branch and add files listed in upload_manifest.txt
git add $(Get-Content upload_manifest.txt) 
# commit
git commit -m "Initial upload: scripts, src, report, selected outputs"

# Add a remote and push (example)
# git remote add origin git@github.com:yourorg/yourrepo.git
# git push -u origin main

Notes
- Secrets (GEMINI_API_KEY) should be provided via environment variables or placed in `.secrets/gemini_key.txt` locally and excluded from git.
- To reproduce experiments, install dependencies from `requirements.txt` and follow the instructions in `reports/Taguchi_LLM_Report.md`.
