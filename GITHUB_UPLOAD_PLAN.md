GitHub upload plan — selective file inclusion

Goal: Publish the code and essential artifacts so others can reproduce the Taguchi+LLM pipeline without exposing large datasets, venvs, or secrets.

Files to include (recommended):
- scripts/*.py (core scripts for prompt generation, runner, client, analysis)
- src/*.py
- reports/Taguchi_LLM_Report.md
- README.md
- requirements.txt
- .gitignore
- Selected outputs (small):
  - outputs/taguchi_runs/predicted_defects_summary.json
  - outputs/taguchi_runs/run_T1.ndjson
  - outputs/taguchi_runs/taguchi_results_summary.json

Why these files?
- `scripts/` and `src/` provide reproducible code. The report documents design and results. Selected outputs provide examples of model behaviour without uploading all raw data.

How to push (PowerShell)
1. Review the manifest in `upload_manifest.txt` to confirm the exact files to add.
2. Run:

```powershell
# stage files listed in manifest
git add $(Get-Content upload_manifest.txt)
# commit
git commit -m "Add Taguchi+LLM pipeline, report and selected outputs"
# push to remote (configure remote first)
# git remote add origin <your-repo-url>
# git push -u origin main
```

Security reminder
- Do NOT commit `.secrets/` or any file containing your API key. Use environment variables on CI or local machines.
