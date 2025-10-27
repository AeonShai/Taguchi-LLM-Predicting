import os
import json
import requests

KEY = os.getenv('GEMINI_API_KEY')
ENDPOINT = os.getenv('GEMINI_ENDPOINT')
PROMPT = "Test: return a short JSON object with keys \"quality\" and \"confidence\" for a sample injection-molding sensor reading."

if not ENDPOINT:
    print('GEMINI_ENDPOINT not set')
    raise SystemExit(1)

if not KEY:
    print('GEMINI_API_KEY not set')
    raise SystemExit(1)

headers = {'X-goog-api-key': KEY, 'Content-Type': 'application/json'}

payloads = [
    # the curl example form
    {'contents': [{'parts': [{'text': PROMPT}]}]},
    # include model in body
    {'model': 'gemini-2.0-flash', 'contents': [{'parts': [{'text': PROMPT}]}]},
    # messages style (author + content)
    {'messages': [{'author': 'user', 'content': [{'type': 'text', 'text': PROMPT}]}]},
    # chat-like (role + content)
    {'messages': [{'role': 'user', 'content': PROMPT}]},
    # instances style
    {'instances': [{'content': PROMPT}]},
    # input array
    {'input': [{'text': PROMPT}]},
    # top-level input string
    {'input': PROMPT},
    # simple prompt wrapper
    {'prompt': {'text': PROMPT}},
    # plain text
    {'text': PROMPT},
    # google-style with mimeType
    {'contents': [{'parts': [{'text': PROMPT, 'mimeType': 'text/plain'}]}]},
    # include candidate_count
    {'contents': [{'parts': [{'text': PROMPT}]}], 'candidate_count': 1},
    # include temperature
    {'contents': [{'parts': [{'text': PROMPT}]}], 'temperature': 0.2},
]

results = []
for p in payloads:
    try:
        r = requests.post(ENDPOINT, headers=headers, json=p, timeout=30)
        status = r.status_code
        try:
            body = r.json()
            body_text = json.dumps(body)[:2000]
        except Exception:
            body_text = r.text[:2000]
        results.append({'payload_keys': list(p.keys()), 'status': status, 'body': body_text})
        print(f"Tried payload keys={list(p.keys())} -> {status}")
    except Exception as e:
        results.append({'payload_keys': list(p.keys()), 'status': 'ERR', 'body': str(e)})
        print(f"ERROR for payload keys={list(p.keys())}: {e}")

out_path = 'outputs/llm_responses/generate_probe_results.json'
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print('Wrote results to', out_path)
