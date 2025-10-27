"""
Utility to save a Gemini API key to a local secrets file.
Usage (PowerShell):
  python .\scripts\save_gemini_key.py --key "AIza..." --out .secrets/gemini_key.txt
Or interactively:
  python .\scripts\save_gemini_key.py

The script will create the directory if needed and write the key. It does NOT commit the key.
"""
import argparse
import os
import stat

DEFAULT_PATH = os.path.join(os.getcwd(), '.secrets', 'gemini_key.txt')


def save_key(path: str, key: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)
    # Write the key
    with open(path, 'w', encoding='utf-8') as f:
        f.write(key.strip())
    try:
        # Try to set restrictive permissions where possible
        if os.name != 'nt':
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except Exception:
        pass
    print(f'Wrote key to {path}. Be sure this file is ignored by git (.gitignore has .secrets/ by default).')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--key', help='API key to save')
    p.add_argument('--out', default=DEFAULT_PATH, help='Output path (default: .secrets/gemini_key.txt)')
    args = p.parse_args()

    key = args.key
    if not key:
        key = input('Paste your GEMINI API key (it will be saved locally, not committed): ').strip()
    if not key:
        print('No key provided; exiting')
        return
    save_key(args.out, key)


if __name__ == '__main__':
    main()
