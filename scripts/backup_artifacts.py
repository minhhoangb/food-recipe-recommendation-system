"""Copy the serving artifacts (corpus + bundle) to a backup folder, so the demo works offline.

    python scripts/backup_artifacts.py --out <folder>

The backup holds everything the API needs to start without the network:

    <folder>/models/<bundle>/...      the serving bundle
    <folder>/processed/main/...       the processed corpus of the main profile
    <folder>/MANIFEST.txt             what was copied, with the hashes

Use `--verify` to check a backup against the working copy without copying anything.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils import sha256_file  # noqa: E402

BUNDLE = "models/minilm_pretrained-E04-main-1f1f4ce5"
CORPUS_FILES = ("corpus.jsonl", "splits.json", "queries.jsonl", "qrels.jsonl", "dataset_manifest.json")


def copy_bundle(bundle: Path, out: Path) -> list[Path]:
    target = out / bundle
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(bundle, target, dirs_exist_ok=True)
    return sorted(p for p in target.rglob("*") if p.is_file())


def copy_corpus(processed: Path, out: Path) -> list[Path]:
    target = out / "processed" / "main"
    target.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in CORPUS_FILES:
        source = processed / name
        if not source.exists():
            print(f"missing: {source}")
            continue
        shutil.copy2(source, target / name)
        copied.append(target / name)
    return copied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="the backup folder")
    parser.add_argument("--bundle", default=BUNDLE, help="the bundle folder to copy")
    parser.add_argument("--processed", default="data/processed/main", help="the corpus folder")
    parser.add_argument("--verify", action="store_true", help="only check the backup")
    args = parser.parse_args(argv)

    bundle = Path(args.bundle)
    processed = Path(args.processed)
    out = Path(args.out)
    if not bundle.is_dir():
        print(f"the bundle {bundle} is not on this machine")
        return 1

    if args.verify:
        return verify(bundle, processed, out)

    out.mkdir(parents=True, exist_ok=True)
    files = copy_bundle(bundle, out) + copy_corpus(processed, out)
    manifest = out / "MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as handle:
        handle.write("Backup of the serving artifacts (food-recipe-recommendation-system)\n")
        handle.write(f"bundle: {bundle}\ncorpus: {processed}\n\n")
        for path in files:
            handle.write(f"{sha256_file(path)}  {path.relative_to(out)}\n")
    print(f"copied {len(files)} files to {out}")
    print(f"manifest: {manifest}")
    return 0


def verify(bundle: Path, processed: Path, out: Path) -> int:
    """Compare the backup with the working copy, file by file."""
    problems = []
    pairs = [(p, out / p) for p in sorted(bundle.rglob("*")) if p.is_file()]
    pairs += [(processed / name, out / "processed" / "main" / name) for name in CORPUS_FILES]
    for source, backup in pairs:
        if not source.exists():
            continue
        if not backup.exists():
            problems.append(f"missing in the backup: {backup}")
        elif sha256_file(source) != sha256_file(backup):
            problems.append(f"differs: {backup}")
    if problems:
        for problem in problems:
            print(f"   - {problem}")
        return 1
    print(f"the backup in {out} matches the working copy ({len(pairs)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
