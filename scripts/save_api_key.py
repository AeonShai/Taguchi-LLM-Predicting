"""Helper to save API keys into the local .secrets/ directory.

Usage (PowerShell):
  python .\scripts\save_api_key.py gemini YOUR_KEY
  python .\scripts\save_api_key.py openai YOUR_KEY
  python .\scripts\save_api_key.py deepseek YOUR_KEY

This writes `.secrets/<provider>_key.txt` and ensures the directory exists.
"""
import sys
from pathlib import Path

def main(argv):
    if len(argv) < 3:
        print('Usage: save_api_key.py <provider> <key>')
        print('Providers: gemini, openai, deepseek')
        return 2
    provider = argv[1].lower()
    key = argv[2]
    allowed = {'gemini', 'openai', 'deepseek'}
    if provider not in allowed:
        print('Provider must be one of:', ', '.join(sorted(allowed)))
        return 3
    secret_dir = Path.cwd() / '.secrets'
    secret_dir.mkdir(parents=True, exist_ok=True)
    fname = secret_dir / f'{provider}_key.txt'
    with fname.open('w', encoding='utf-8') as f:
        f.write(key.strip())
    print(f'Wrote key to {fname}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main(sys.argv))
