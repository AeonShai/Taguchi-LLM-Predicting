import os
import json
import requests

KEY = os.getenv('GEMINI_API_KEY')
ENDPOINT = os.getenv('GEMINI_ENDPOINT')
PROMPT = "Test probe: return JSON with keys 'quality' and 'confidence' for a sample sensor reading."

if not ENDPOINT:
    print('GEMINI_ENDPOINT not set')
    raise SystemExit(1)

if not KEY:
    print('GEMINI_API_KEY not set')
    raise SystemExit(1)

payload_shapes = [
    {'prompt': {'text': PROMPT}},
    {'prompt': PROMPT},
    {'input': PROMPT},
    {'instances': [{'content': PROMPT}]},
    {'inputs': PROMPT},
]

auth_variants = [
    ('bearer_header', {'Authorization': f'Bearer {KEY}'}),
    ('api_key_query', None),
    ('no_auth', None),
]

suffix_variants = [
    '',
    ':generate',
    ':predict',
]

results = []

for suffix in suffix_variants:
    url = ENDPOINT
    # Ensure suffix added if not present
    if not url.endswith(suffix):
        url_try = url + suffix
    else:
        url_try = url

    for auth_name, headers in auth_variants:
        for pshape in payload_shapes:
            # prepare url and headers
            url_final = url_try
            hdrs = {'Content-Type': 'application/json'}
            if headers:
                hdrs.update(headers)
            # If using api_key_query put key as ?key=...
            if auth_name == 'api_key_query':
                sep = '&' if '?' in url_final else '?'
                url_final = f"{url_final}{sep}key={KEY}"

            try:
                r = requests.post(url_final, headers=hdrs, json=pshape, timeout=20)
                status = r.status_code
                try:
                    j = r.json()
                    text = json.dumps(j)[:1000]
                except Exception:
                    text = r.text[:1000]
                results.append({'url': url_final, 'auth': auth_name, 'payload': list(pshape.keys()), 'status': status, 'text': text})
                print(f"Tried {auth_name} {url_final} payload_keys={list(pshape.keys())} -> {status}")
            except Exception as e:
                results.append({'url': url_final, 'auth': auth_name, 'payload': list(pshape.keys()), 'status': 'ERR', 'text': str(e)})
                print(f"ERROR for {auth_name} {url_final} payload_keys={list(pshape.keys())}: {e}")

# Save results for inspection
outf = 'outputs/llm_responses/probe_results.json'
os.makedirs(os.path.dirname(outf), exist_ok=True)
with open(outf, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('Wrote probe results to', outf)
