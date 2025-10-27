"""Generate an HTML version of the Taguchi LLM Markdown report.
"""
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'reports' / 'Taguchi_LLM_Report.md'
OUT = ROOT / 'reports' / 'Taguchi_LLM_Report.html'

def main():
    if not MD.exists():
        print('Markdown source not found:', MD)
        return 2
    text = MD.read_text(encoding='utf-8')
    html_body = markdown.markdown(text, extensions=['tables','fenced_code','toc'])
    html = '<!doctype html><meta charset="utf-8"><head><style>body{font-family:Arial,Helvetica,sans-serif;margin:30px;}</style></head><body>' + html_body + '</body>'
    OUT.write_text(html, encoding='utf-8')
    print('WROTE', OUT)

if __name__ == '__main__':
    raise SystemExit(main())
