import os
import requests
import json

key = os.getenv('GEMINI_API_KEY')
if not key:
    cred_file = os.path.join(os.getcwd(), '.secrets', 'gemini_key.txt')
    try:
        with open(cred_file, 'r', encoding='utf-8') as f:
            key = f.read().strip()
    except Exception as e:
        print('No GEMINI key found:', e)
        raise SystemExit(1)

endpoint = os.getenv('GEMINI_ENDPOINT')
if not endpoint:
    print('GEMINI_ENDPOINT not set')
    raise SystemExit(1)

headers = {'Content-Type': 'application/json', 'X-goog-api-key': key}
body = {
    'contents': [
        {
            'parts': [
                {'text': 'Test: return a short JSON object with keys "quality" and "confidence" for a sample injection-molding sensor reading.'}
            ]
        }
    ]
}

print('Posting to', endpoint)
resp = requests.post(endpoint, headers=headers, json=body, timeout=30)
print('STATUS:', resp.status_code)
try:
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
except Exception:
    print('TEXT:', resp.text)
    raise SystemExit(1)
