import json
import glob
import os
from collections import Counter

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'taguchi_runs')
FILES = sorted(glob.glob(os.path.join(OUT_DIR, 'run_T*.ndjson')))

total = 0
parsed_ok = 0
empty_preds = 0
defects_counter = Counter()
examples = []

for f in FILES:
    with open(f, 'r', encoding='utf-8') as fh:
        for line in fh:
            total += 1
            try:
                row = json.loads(line)
            except Exception:
                continue
            if row.get('parse_ok'):
                parsed_ok += 1
                parsed = row.get('parsed') or {}
                preds = parsed.get('predicted_defects') or []
                if not preds:
                    empty_preds += 1
                else:
                    for p in preds:
                        if isinstance(p, dict):
                            t = p.get('type') or json.dumps(p, ensure_ascii=False)
                        else:
                            t = str(p)
                        defects_counter[t] += 1
                    if len(examples) < 10:
                        examples.append({'trial': row.get('trial_id'), 'sample_id': row.get('sample_id'), 'predicted_defects': preds, 'recommended_actions': parsed.get('recommended_actions')})

summary = {
    'total_rows': total,
    'parsed_ok': parsed_ok,
    'empty_predicted_defects': empty_preds,
    'defects_counts': dict(defects_counter),
    'examples': examples,
}

out_path = os.path.join(OUT_DIR, 'predicted_defects_summary.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(json.dumps(summary, ensure_ascii=False, indent=2))
