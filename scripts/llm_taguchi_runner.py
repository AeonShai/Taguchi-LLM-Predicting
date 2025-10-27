"""Run Taguchi L9 prompt experiments using the repository's llm_client.

This runner reads the L9 matrix from outputs/taguchi_L9_matrix.csv,
samples N examples from the MouldCode=5001 pruned file, builds prompts
with llm_prompts_taguchi.PromptGenerator and calls llm_client.call_llm.

Outputs per-run NDJSON saved to outputs/taguchi_runs/run_<trial>.ndjson
and a summary outputs/taguchi_results_summary.json.

NOTE: This script will make live LLM calls. Set environment variables:
 - GEMINI_ENDPOINT (or ensure llm_client reads key locally)
 - GEMINI_API_KEY or have .secrets/gemini_key.txt available
"""
import os
import csv
import json
import random
from pathlib import Path
from datetime import datetime
import sys

# Ensure project root is on sys.path so 'scripts' package imports work when
# running the script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_prompts_taguchi import PromptGenerator

# llm_client will be imported lazily only when dry_run is False so that a venv
# without HTTP deps (requests) can still run dry-run executions.
llm_client = None

OUTPUT_DIR = ROOT / "outputs" / "taguchi_runs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

L9_CSV = ROOT / "outputs" / "taguchi_L9_matrix.csv"
SAMPLE_CSV = ROOT / "outputs" / "ham_veri_mould_5001_pruned_with_labels.csv"

DEFAULT_SAMPLES_PER_TRIAL = int(os.environ.get("TAGUCHI_SAMPLES_PER_TRIAL", "10"))

# Measurement fields to include in the prompt (from the image supplied by user)
MEASUREMENT_FIELDS = [
    "InjectionStroke",
    "InjectionTime",
    "ActualStrokePosition",
    "MeasuredCycleDuration",
    "cluster3_flag",
    "SliderOutputTimePeriodValue",
    "MoldTemp2",
    "MaxInjectionPressure",
    "SliderInputTimePeriodValue",
    "CoolingTime",
    "OilTemperature",
    "DosingTime",
    "ClosingForceGenerationTimePeriodValue",
    "MoldTemp6",
    "BarrelTemp1",
]


def load_samples(n: int):
    samples = []
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(f"Sample CSV not found: {SAMPLE_CSV}")
    with SAMPLE_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) == 0:
        raise ValueError("No rows in sample CSV")
    chosen = random.sample(rows, min(n, len(rows)))
    out = []
    for r in chosen:
        ts_summary = r.get('timeseries_summary') or 'timeseries not provided'
        setpoints = f"mold_temp={r.get('mold_temperature','?')}C,inj_pressure={r.get('injection_pressure','?')}bar,cycle_time={r.get('MeasuredCycleDuration','?')}s"
        # Collect specific measurement fields into a measurements dict to pass to the prompt
        measurements = {f: r.get(f, '') for f in MEASUREMENT_FIELDS}

        out.append({
            'sample_id': r.get('sample_id') or r.get('id') or r.get('RowID') or str(random.randint(1,1_000_000)),
            'MouldCode': r.get('MouldCode', '5001'),
            'timestamp': r.get('timestamp') or r.get('Ts') or datetime.utcnow().isoformat() + 'Z',
            'setpoints': setpoints,
            'timeseries_summary': ts_summary,
            'measurements': measurements,
            'raw_row': r
        })
    return out


def run():
    pg = PromptGenerator()
    # Ensure we assign to the module-level llm_client when doing lazy import
    global llm_client
    # read L9
    if not L9_CSV.exists():
        raise FileNotFoundError(f"L9 CSV not found: {L9_CSV}")
    trials = []
    with L9_CSV.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            trials.append(row)

    results_summary = []

    # For each trial, sample N examples and run
    for t in trials:
        trial_id = t.get('trial')
        A = int(t.get('A'))
        B = int(t.get('B'))
        C = int(t.get('C'))
        D = int(t.get('D'))
        n = DEFAULT_SAMPLES_PER_TRIAL
        samples = load_samples(n)
        out_file = OUTPUT_DIR / f"run_{trial_id}.ndjson"
        with out_file.open("w", encoding="utf-8") as fout:
            trial_metrics = {'trial': trial_id, 'A': A, 'B': B, 'C': C, 'D': D, 'n': len(samples), 'parsed_ok': 0}
            for s in samples:
                prompt_meta_id = f"{trial_id}-{s['sample_id']}"
                prompt_obj = pg.generate_prompt(s, A, B, C, D, prompt_id=prompt_meta_id)
                try:
                    # support dry-run via environment variable TAGUCHI_DRY_RUN (1/true)
                    dry_flag = os.environ.get('TAGUCHI_DRY_RUN', '').lower() in ('1', 'true', 'yes')
                    # Import llm_client lazily when running live (dry_flag==False)
                    if not dry_flag:
                        if llm_client is None:
                            # local import; will raise if dependencies missing
                            from scripts import llm_client as _llm
                            llm_client = _llm
                    # call llm_client - best-effort signature
                    resp = llm_client.call_llm(prompt_obj['prompt'], provider=os.environ.get('LLM_PROVIDER','gemini'), model=os.environ.get('LLM_MODEL'), dry_run=dry_flag)
                except TypeError:
                    # fallback if llm_client requires different signature
                    resp = llm_client.call_llm(prompt=prompt_obj['prompt'], provider=os.environ.get('LLM_PROVIDER','gemini'))
                except Exception as e:
                    resp = {'error': str(e)}

                parsed = None
                parse_ok = False
                if isinstance(resp, dict):
                    parsed = resp.get('parsed') or resp.get('json') or None
                    if parsed:
                        parse_ok = True
                        trial_metrics['parsed_ok'] += 1

                # Save the full response dict into raw_response so we can debug http/parse issues
                row = {
                    'trial_id': trial_id,
                    'prompt_id': prompt_meta_id,
                    'A': A, 'B': B, 'C': C, 'D': D,
                    'sample_id': s['sample_id'],
                    'prompt': prompt_obj['prompt'],
                    # store the entire response object (may contain raw, raw_text, http_status, parsed, etc.)
                    'raw_response': resp,
                    'parsed': parsed,
                    'parse_ok': parse_ok,
                    # Keep internal metadata for traceability but do NOT include these fields in the prompt
                    'internal_metadata': {
                        'MouldCode': s.get('MouldCode'),
                        'timestamp': s.get('timestamp')
                    },
                    # Redacted raw row to avoid leaking internal ids in stored raw rows
                    'raw_row_redacted': {k: v for k, v in (s.get('raw_row') or {}).items() if k not in ('MouldCode', 'timestamp')}
                }
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

            results_summary.append(trial_metrics)

    # write summary
    summary_path = OUTPUT_DIR / "taguchi_results_summary.json"
    with summary_path.open("w", encoding="utf-8") as sf:
        json.dump({'summary': results_summary, 'run_at': datetime.utcnow().isoformat() + 'Z'}, sf, indent=2, ensure_ascii=False)
    print(f"Completed runs. Outputs: {OUTPUT_DIR}")


if __name__ == '__main__':
    run()
