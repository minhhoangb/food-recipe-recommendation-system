"""Render a Markdown report to HTML and print it to PDF with headless Edge or Chrome.

    python scripts/md_to_pdf.py reports/report_part1.md          # -> reports/report_part1.{html,pdf}

Images keep their relative paths, so the HTML sits next to the Markdown file. Needs Microsoft Edge or
Google Chrome installed (the Windows paths below, or `msedge` / `google-chrome` on PATH).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "msedge",
    "google-chrome",
    "chromium",
]
CSS = """
@page { size: A4; margin: 18mm 17mm; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin-top: 16pt; border-bottom: 1px solid #999; }
h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; font-size: 9pt; page-break-inside: avoid; }
th, td { border: 1px solid #aaa; padding: 3pt 5pt; vertical-align: top; text-align: left; }
th { background: #eee; }
img { display: block; max-width: 78%; max-height: 72mm; margin: 6pt auto; page-break-inside: avoid; }
code { font-family: Consolas, monospace; font-size: 9pt; }
"""


def browser() -> str:
    for candidate in BROWSERS:
        found = candidate if Path(candidate).exists() else shutil.which(candidate)
        if found:
            return found
    sys.exit("no Edge/Chrome found; open the .html file and print it to PDF by hand")


def main() -> int:
    source = Path(sys.argv[1]).resolve()
    body = MarkdownIt("commonmark").enable("table").render(source.read_text(encoding="utf-8"))
    title = source.stem.replace("_", " ")
    html = source.with_suffix(".html")
    html.write_text(
        f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{title}</title>'
        f"<style>{CSS}</style></head><body>{body}</body></html>",
        encoding="utf-8",
    )
    pdf = source.with_suffix(".pdf")
    subprocess.run(
        [browser(), "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--print-to-pdf={pdf}"]
        + [html.as_uri()],
        check=True,
        capture_output=True,
    )
    print(f"{html}\n{pdf}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
