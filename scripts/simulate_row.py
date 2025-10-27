import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_prompts_taguchi import PromptGenerator

pg = PromptGenerator()
# sample like load_samples would create
s = {
    'sample_id': 'SIM123',
    'MouldCode': '5001',
    'timestamp': '2025-10-27T12:00:00Z',
    'setpoints': 'mold_temp=60C,inj_pressure=120bar,cycle_time=8s',
    'timeseries_summary': 'son 3 okuma: mean=..., std=...'
}
# add measurements
s['measurements'] = {k: str(i) for i,k in enumerate([
    'InjectionStroke','InjectionTime','ActualStrokePosition','MeasuredCycleDuration','cluster3_flag','SliderOutputTimePeriodValue','MoldTemp2','MaxInjectionPressure','SliderInputTimePeriodValue','CoolingTime','OilTemperature','DosingTime','ClosingForceGenerationTimePeriodValue','MoldTemp6','BarrelTemp1'], start=1)}

prompt_obj = pg.generate_prompt(s, A=1, B=1, C=1, D=1, prompt_id='SIM-1')
# Simulate an LLM dry-run response without importing llm_client (requests may not be installed in venv)
resp = {
    'status': 'dry_run',
    'model': 'simulated',
    'simulated_response': {
        'quality': 'Medium',
        'defects': [],
        'reasoning': [],
        'confidence': 0.6,
        'corrective_actions': ['Check mold temp','Review injection pressure']
    }
}
parsed = resp.get('simulated_response')
parse_ok = parsed is not None
row = {
    'trial_id': 'SIM',
    'prompt_id': 'SIM-1',
    'A': 1, 'B': 1, 'C': 1, 'D': 1,
    'sample_id': s['sample_id'],
    'prompt': prompt_obj['prompt'],
    'raw_response': resp,
    'parsed': parsed,
    'parse_ok': parse_ok,
    'internal_metadata': {'MouldCode': s.get('MouldCode'), 'timestamp': s.get('timestamp')},
    'raw_row_redacted': {}
}
print(json.dumps(row, ensure_ascii=False, indent=2))
