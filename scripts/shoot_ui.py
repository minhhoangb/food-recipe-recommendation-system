"""Take the UI screenshots of the task: 390 px (phone) and 1440 px (desktop).

    python scripts/shoot_ui.py --url http://127.0.0.1:8010/ --ingredients chicken, rice, garlic

Headless Chrome is opened on the page with the query in the URL (`?q=...&k=...`), so the
page itself runs the search and the picture shows the real results of the real API.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import quote

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import REPO_ROOT  # noqa: E402

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
VIEWPORTS = {"390": (390, 844), "1440": (1440, 900)}


def shoot(url: str, ingredients: str, top_k: int, out_dir: Path) -> list[Path]:
    """One screenshot per viewport, of the page with the results already on it."""
    target = f"{url}?q={quote(ingredients)}&k={top_k}"
    written: list[Path] = []
    for label, (width, height) in VIEWPORTS.items():
        out = (out_dir / f"ui_{label}.png").resolve()
        if out.exists():
            out.unlink()
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                f"--screenshot={out}",
                "--virtual-time-budget=10000",
                f"--user-data-dir={REPO_ROOT / 'logs' / 'chrome-shot'}",
                target,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=REPO_ROOT,
        )
        for _ in range(20):  # Chrome writes the file a moment after the process exits
            if out.exists():
                break
            time.sleep(0.5)
        written.append(out)
        print(f"{width}x{height} -> {out} ({out.stat().st_size // 1024} KB)")
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8010/")
    parser.add_argument("--ingredients", default="chicken, rice, garlic")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--out-dir", default="reports/figures")
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir) if Path(args.out_dir).is_absolute() else REPO_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    shoot(args.url, args.ingredients, args.top_k, out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
