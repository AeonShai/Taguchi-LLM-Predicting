"""Lightweight LLM client wrapper.

Features:
- dry_run mode (returns simulated response shell; safe for development)
- OpenAI branch (minimal; uses openai package when available)
- Gemini branch: flexible HTTP caller that accepts a user-supplied endpoint and API key via environment

Notes:
- For Gemini, set GEMINI_API_KEY and GEMINI_ENDPOINT (full URL including model path) in the environment.
- GEMINI_ENDPOINT should be the full POST URL the Gemini/Generative API expects. This file does not assume a single canonical request shape
  — it will send a JSON body containing `prompt` or `input` and include the model if provided. Adjust the payload "shape" in
  the code below if you want a strict Google Generative Language payload.
"""

import json
import os
import logging
from typing import Dict, Any, Optional

import requests

_LOG = logging.getLogger(__name__)


def _read_key_from_file(path: str) -> Optional[str]:
    """Return the key string from path, or None if not readable."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            key = f.read().strip()
            return key if key else None
    except Exception:
        return None


def call_llm(prompt: str, provider: str = 'openai', model: str = 'gpt-4o', dry_run: bool = True, **kwargs) -> Dict[str, Any]:
    """Call an LLM provider and return a structured dict.

    Args:
        prompt: Full prompt string to send to model.
        provider: One of 'openai', 'gemini', 'deepseek' (deepseek not yet implemented).
        model: Model name or alias (passed through to provider when relevant).
        dry_run: If True, returns a simulated response without making network calls.
        **kwargs: Provider-specific options (e.g., max_output_tokens, temperature).

    Returns:
        A dict with at least a 'status' key and provider raw response under 'raw' when successful.
    """

    if dry_run:
        _LOG.debug('LLM dry_run enabled; returning simulated response')
        return {
            'status': 'dry_run',
            'model': model,
            'prompt_excerpt': prompt[:1000].replace('\n', ' '),
            'simulated_response': {
                'quality': 'MEDIUM',
                'defects': [{'type': 'short_shot', 'reason': 'low injection pressure in cycle'}],
                'reasoning': ['pressure dropped at transfer', 'cycle duration shorter than normal'],
                'confidence': 0.65,
                'corrective_actions': ['increase injection pressure by 5%', 'check feed system for blockage'],
                'suggestions_for_measurements': ['capture pressure trace', 'take cavity image']
            }
        }

    # OpenAI / ChatGPT branch
    # Accept provider values 'openai' or 'chatgpt' for clarity.
    if provider in ('openai', 'chatgpt'):
        try:
            import openai
        except Exception:
            raise RuntimeError('openai package not installed; set dry_run=True or install openai')
        key = os.getenv('OPENAI_API_KEY')
        if not key:
            raise RuntimeError('OPENAI_API_KEY not set in environment')
        openai.api_key = key
        # Prefer ChatCompletion API when available; fall back to simple completion when not.
        messages = [{'role': 'user', 'content': prompt}]
        try:
            resp = openai.ChatCompletion.create(model=model or 'gpt-4o', messages=messages, **kwargs)
        except AttributeError:
            # Older openai packages expose Completion API
            resp = openai.Completion.create(model=model or 'gpt-4o', prompt=prompt, **kwargs)
        return {'status': 'ok', 'raw': resp}

    # Gemini flexible HTTP branch. Accepts an API key via env var or local secrets file and a full endpoint URL.
    # Google Generative API expects the X-goog-api-key header and a `contents` payload.
    if provider == 'gemini':
        # API key: env var first, then .secrets/gemini_key.txt (or GEMINI_CRED_FILE)
        gemini_key = os.getenv('GEMINI_API_KEY')
        if not gemini_key:
            cred_file = os.getenv('GEMINI_CRED_FILE', os.path.join(os.getcwd(), '.secrets', 'gemini_key.txt'))
            gemini_key = _read_key_from_file(cred_file)

        gemini_endpoint = os.getenv('GEMINI_ENDPOINT')
        if not gemini_key or not gemini_endpoint:
            raise RuntimeError('GEMINI_API_KEY (or GEMINI_CRED_FILE) and GEMINI_ENDPOINT must be set to use provider=gemini')

        headers = {
            'X-goog-api-key': gemini_key,
            'Content-Type': 'application/json',
        }

        payload = {
            'contents': [
                {
                    'parts': [
                        {'text': prompt}
                    ]
                }
            ]
        }
        # Map a small set of friendly kwargs to the Google Gen API fields.
        # Avoid passing unknown fields which cause a 400 error.
        if 'temperature' in kwargs:
            payload['temperature'] = kwargs['temperature']
        # Map snake_case max_output_tokens -> maxOutputTokens expected by some endpoints
        if 'max_output_tokens' in kwargs:
            payload['maxOutputTokens'] = kwargs['max_output_tokens']
        if 'candidate_count' in kwargs:
            payload['candidateCount'] = kwargs['candidate_count']

        try:
            _LOG.debug('Calling Gemini endpoint %s (model in path=%s)', gemini_endpoint, model)
            resp = requests.post(gemini_endpoint, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            try:
                data = resp.json()
            except ValueError:
                return {'status': 'ok', 'raw_text': resp.text, 'http_status': resp.status_code}

            # Try to parse a JSON payload embedded in the model text (many prompts ask for JSON).
            parsed = None
            try:
                parsed = parse_gemini_response(data)
            except Exception:
                parsed = None

            out = {'status': 'ok', 'raw': data}
            if parsed is not None:
                out['parsed'] = parsed
                # save a copy for inspection
                try:
                    os.makedirs('outputs/llm_responses', exist_ok=True)
                    with open('outputs/llm_responses/last_gemini_parsed.json', 'w', encoding='utf-8') as f:
                        json.dump(parsed, f, ensure_ascii=False, indent=2)
                except Exception:
                    _LOG.exception('Failed to write parsed gemini output')

            return out
        except requests.RequestException as e:
            _LOG.exception('Gemini request failed')
            # If we have a response object, include status/text to help debugging
            if hasattr(e, 'response') and e.response is not None:
                txt = e.response.text
                raise RuntimeError(f'Gemini request failed: {e} RESPONSE_TEXT: {txt}')
            raise RuntimeError(f'Gemini request failed: {e}')

    # Deepseek HTTP branch (simple generic HTTP POST)
    if provider == 'deepseek':
        deepseek_key = os.getenv('DEEPSEEK_API_KEY')
        deepseek_endpoint = os.getenv('DEEPSEEK_ENDPOINT')
        if not deepseek_key or not deepseek_endpoint:
            raise RuntimeError('DEEPSEEK_API_KEY and DEEPSEEK_ENDPOINT must be set to use provider=deepseek')
        headers = {
            'Authorization': f'Bearer {deepseek_key}',
            'Content-Type': 'application/json',
        }
        payload = {'prompt': prompt}
        if model:
            payload['model'] = model
        payload.update({k: v for k, v in kwargs.items() if k in ('temperature', 'max_output_tokens', 'candidate_count')})
        try:
            r = requests.post(deepseek_endpoint, headers=headers, json=payload, timeout=30)
            r.raise_for_status()
            try:
                data = r.json()
            except ValueError:
                return {'status': 'ok', 'raw_text': r.text, 'http_status': r.status_code}

            # try extracting JSON from textual content if present
            parsed = None
            try:
                # reuse generic parser to find JSON in text fields
                parsed = parse_gemini_response(data) or None
            except Exception:
                parsed = None

            out = {'status': 'ok', 'raw': data}
            if parsed is not None:
                out['parsed'] = parsed
            return out
        except requests.RequestException as e:
            _LOG.exception('Deepseek request failed')
            if hasattr(e, 'response') and e.response is not None:
                txt = e.response.text
                raise RuntimeError(f'Deepseek request failed: {e} RESPONSE_TEXT: {txt}')
            raise RuntimeError(f'Deepseek request failed: {e}')

    raise RuntimeError(f'Provider {provider} not supported')


def parse_gemini_response(raw: Dict[str, Any]) -> Dict[str, Any] | None:
    """Parse a Gemini generateContent response and extract a JSON object if present.

    The Gemini response typically includes `candidates` -> first candidate -> content -> parts -> [ { 'text': ... } ].
    The model often returns a fenced JSON block (```json ... ```). This helper extracts and parses it.

    Returns a dict parsed from JSON if found, otherwise None.
    """
    # Defensive navigation
    try:
        candidates = raw.get('candidates') or []
        if not candidates:
            return None
        first = candidates[0]
        content = first.get('content') or {}
        parts = content.get('parts') or []
        if not parts:
            return None
        text = parts[0].get('text') or ''
    except Exception:
        return None

    # Try fenced JSON first
    # look for ```json ... ``` or ``` ... ``` blocks
    import re
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    json_str = None
    if m:
        json_str = m.group(1)
    else:
        # fallback: find first balanced JSON object in the text
        # find first '{' and try to parse until matching '}'
        start = text.find('{')
        if start != -1:
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == '{':
                    depth += 1
                elif text[i] == '}':
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                json_str = text[start:end+1]

    if not json_str:
        return None

    try:
        return json.loads(json_str)
    except Exception:
        return None
