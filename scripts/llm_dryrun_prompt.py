"""
Dry-run script: builds a quality prompt for the first row in the pruned-with-labels CSV
and writes the prompt and a simulated LLM response to outputs/llm_responses/ for review.
"""
import os
import json
import pandas as pd
from llm_prompt_templates import build_quality_prompt
from llm_client import call_llm

root = 'C:/Users/Beara/OneDrive/Desktop/MCP-Filesystem/Tubitak'
input_csv = os.path.join(root, 'outputs', 'ham_veri_mould_5001_pruned_with_labels.csv')
out_dir = os.path.join(root, 'outputs', 'llm_responses')
os.makedirs(out_dir, exist_ok=True)

print('Loading', input_csv)
df = pd.read_csv(input_csv)
if df.shape[0] == 0:
    raise SystemExit('input CSV is empty')

row = df.iloc[0]
# Prepare a compact sensor dict (select a few key sensors if many exist)
sensor_keys = ['MeasuredCycleDuration','OilTemperature','MaxInjectionPressure','InjectionTime','ActualClampingForce']
sensors = {k: (row[k] if k in row.index else None) for k in sensor_keys}
model_outputs = {k: row[k] for k in ['cluster_k3','anomaly_flag','anomaly_score'] if k in row.index}

prompt = build_quality_prompt(sensors, model_outputs)
print('\n=== PROMPT PREVIEW ===\n')
print(prompt[:4000])

# Call LLM in dry_run mode to get a simulated response
resp = call_llm(prompt, dry_run=True)

# Save prompt and response
with open(os.path.join(out_dir, 'prompt_1.txt'), 'w', encoding='utf-8') as f:
    f.write(prompt)
with open(os.path.join(out_dir, 'response_1.json'), 'w', encoding='utf-8') as f:
    json.dump(resp, f, indent=2, ensure_ascii=False)

print('\nSaved prompt to outputs/llm_responses/prompt_1.txt')
print('Saved simulated response to outputs/llm_responses/response_1.json')
