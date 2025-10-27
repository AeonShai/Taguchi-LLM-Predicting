import json
from scripts.llm_client import call_llm

prompt = (
    "Test: Provide a short JSON response with fields 'quality' and 'confidence' for a fictional sensor reading. "
    "Return only JSON if possible."
)

try:
    resp = call_llm(prompt, provider='gemini', model='text-bison', dry_run=False)
    print('STATUS:', resp.get('status'))
    if 'raw' in resp:
        print(json.dumps(resp['raw'], indent=2, ensure_ascii=False))
    else:
        print('RAW_TEXT:', resp.get('raw_text') or resp)
except Exception as e:
    print('ERROR:', e)
