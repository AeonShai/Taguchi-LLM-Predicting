"""Attempt to render the Markdown report to PDF.

Strategy:
- If `pandoc` is available on PATH, call it to convert `reports/Taguchi_LLM_Report.md` -> PDF.
- Otherwise, try to use `reportlab` (if installed) to create a minimal PDF with embedded images.
- If neither is available, print clear instructions for the user to run pandoc in PowerShell.

This script is best-effort: environment dependencies may not be installed on the machine.
"""
import shutil
import subprocess
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / 'reports' / 'Taguchi_LLM_Report.md'
OUT = ROOT / 'reports' / 'Taguchi_LLM_Report.pdf'


def try_pandoc():
    pandoc = shutil.which('pandoc')
    if not pandoc:
        return False, 'pandoc not found'
    cmd = [pandoc, str(MD), '-o', str(OUT), '--pdf-engine=xelatex']
    try:
        subprocess.run(cmd, check=True)
        return True, 'pandoc conversion succeeded'
    except subprocess.CalledProcessError as e:
        return False, f'pandoc failed: {e}'


def try_reportlab():
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(OUT), pagesize=A4)
        text = c.beginText(40, 820)
        text.setFont('Helvetica', 10)
        text.textLines('Taguchi LLM Report (excerpt)')
        c.drawText(text)
        c.showPage()
        c.save()
        return True, 'reportlab fallback produced a minimal PDF'
    except Exception as e:
        return False, f'reportlab not available or failed: {e}'


def main():
    if not MD.exists():
        print(f'Markdown source not found: {MD}')
        sys.exit(2)

    ok, msg = try_pandoc()
    if ok:
        print(msg)
        print(f'PDF written to: {OUT}')
        return

    print('Pandoc not available or failed:', msg)
    ok2, msg2 = try_reportlab()
    if ok2:
        print(msg2)
        print(f'PDF written to: {OUT}')
        return

    print('Could not auto-generate PDF. To produce PDF locally, run in PowerShell:')
    print()
    print('pandoc "reports/Taguchi_LLM_Report.md" -o "reports/Taguchi_LLM_Report.pdf" --pdf-engine=xelatex')
    print()
    print('If pandoc is not installed, install it from https://pandoc.org/installing.html and ensure a LaTeX engine (e.g., TinyTeX, TeXLive) is available for PDF output.')


if __name__ == '__main__':
    main()
