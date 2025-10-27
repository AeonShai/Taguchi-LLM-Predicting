import sys
from pathlib import Path
# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.llm_prompts_taguchi import PromptGenerator

pg = PromptGenerator()

s = {
    'sample_id': 'test123',
    'MouldCode': 5001,
    'timestamp': '2025-10-27T00:00:00Z',
    'setpoints': 'mold_temp=60C,inj_pressure=120bar,cycle_time=8s',
    'timeseries_summary': 'son 3 okuma: ...',
    'measurements': {
        'InjectionStroke': '12',
        'InjectionTime': '0.45',
        'ActualStrokePosition': '11',
        'MeasuredCycleDuration': '8.0',
        'cluster3_flag': '',
        'SliderOutputTimePeriodValue': '',
        'MoldTemp2': '58',
        'MaxInjectionPressure': '120',
        'SliderInputTimePeriodValue': '',
        'CoolingTime': '2.3',
        'OilTemperature': '40',
        'DosingTime': '0.12',
        'ClosingForceGenerationTimePeriodValue': '',
        'MoldTemp6': '59',
        'BarrelTemp1': '210'
    }
}

out = pg.generate_prompt(s, A=1, B=2, C=3, D=3, prompt_id='T-TEST')
print(out['prompt'])
