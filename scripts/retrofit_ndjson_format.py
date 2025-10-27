"""Retrofit existing NDJSON run files to include internal_metadata and raw_row_redacted.

This script scans `outputs/taguchi_runs/*.ndjson`, updates each JSON line to ensure
`internal_metadata` exists (pulling MouldCode/timestamp from top-level or raw_row if present),
and writes a redacted `raw_row_redacted` without MouldCode/timestamp. It writes a new file
`<orig>.patched` and then replaces the original (atomic replace).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / 'outputs' / 'taguchi_runs'

def redact_row(raw_row: dict):
    if not isinstance(raw_row, dict):
        return {}
    return {k: v for k, v in raw_row.items() if k not in ('MouldCode', 'timestamp')}

def ensure_internal(obj):
    # If already present, leave as-is
    if 'internal_metadata' in obj and obj['internal_metadata'] is not None:
        return obj

    mould = None
    ts = None
    # Check top-level fields
    if 'MouldCode' in obj:
        mould = obj.get('MouldCode')
    if 'timestamp' in obj:
        ts = obj.get('timestamp')

    # Try raw_row
    raw = obj.get('raw_row') or {}
    if not mould and isinstance(raw, dict):
        mould = raw.get('MouldCode')
    if not ts and isinstance(raw, dict):
        ts = raw.get('timestamp')

    obj['internal_metadata'] = {'MouldCode': mould, 'timestamp': ts}
    obj['raw_row_redacted'] = redact_row(raw)

    # Optionally remove top-level MouldCode/timestamp to avoid duplication
    obj.pop('MouldCode', None)
    obj.pop('timestamp', None)
    return obj

def process_file(p: Path):
    patched = p.with_suffix(p.suffix + '.patched')
    changed = False
    with p.open('r', encoding='utf-8') as fin, patched.open('w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # write original and continue
                fout.write(line + '\n')
                continue
            before = json.dumps(obj, sort_keys=True)
            obj = ensure_internal(obj)
            after = json.dumps(obj, sort_keys=True)
            if before != after:
                changed = True
            fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
    if changed:
        patched.replace(p)
        print('Patched', p)
    else:
        patched.unlink()
        print('No changes for', p)

def main():
    files = sorted(OUT_DIR.glob('run_*.ndjson'))
    if not files:
        print('No run_*.ndjson files found in', OUT_DIR)
        return 0
    for f in files:
        process_file(f)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
