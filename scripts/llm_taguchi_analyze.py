"""Analyze Taguchi run NDJSON outputs and compute simple S/N metrics.

This script reads outputs/taguchi_runs/*.ndjson and summarizes parse rate,
average confidence, and an S/N proxy (larger-is-better) based on
confidence-weighted parse rate.
"""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "outputs" / "taguchi_runs"


def safe_get_confidence(parsed):
    try:
        if not parsed:
            return None
        # parsed may be dict with 'confidence' or nested
        if isinstance(parsed, dict) and 'confidence' in parsed:
            return float(parsed['confidence'])
        # sometimes parsed may have predicted_defects etc. fallback
        return None
    except Exception:
        return None


def analyze():
    summary = []
    for p in sorted(OUTPUT_DIR.glob('run_*.ndjson')):
        trials = []
        with p.open('r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                trials.append(obj)
        if not trials:
            continue
        trial_id = trials[0].get('trial_id')
        n = len(trials)
        parse_ok = sum(1 for t in trials if t.get('parse_ok'))
        confidences = [safe_get_confidence(t.get('parsed')) for t in trials if t.get('parse_ok')]
        confidences = [c for c in confidences if c is not None]
        avg_conf = statistics.mean(confidences) if confidences else 0.0
        parse_rate = parse_ok / n
        # proxy metric: confidence-weighted parse rate
        proxy = avg_conf * parse_rate
        # S/N larger-is-better: using -10*log10(mean(1/yi^2)) is unstable; use simple dB scale
        sn_db = 10 * (proxy)  # simple scaled proxy for ranking
        summary.append({
            'trial_id': trial_id,
            'n': n,
            'parse_ok': parse_ok,
            'parse_rate': parse_rate,
            'avg_confidence': avg_conf,
            'proxy': proxy,
            'sn_db': sn_db
        })

    out_path = OUTPUT_DIR / 'taguchi_analysis_summary.json'
    with out_path.open('w', encoding='utf-8') as f:
        json.dump({'analysis': summary}, f, indent=2, ensure_ascii=False)
    print(f"Analysis written to {out_path}")


if __name__ == '__main__':
    analyze()
