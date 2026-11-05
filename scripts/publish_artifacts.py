"""Publish an artifact file to GitHub Releases using existing git credentials.

Does not replace existing release assets. Prints only public URLs and hashes.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repo", default="minhhoangb/food-recipe-recommendation-system")
    parser.add_argument("--tag", default="processed-1f1f4ce505")
    parser.add_argument("--name", default="Verified processed RecipeNLG subsets")
    parser.add_argument("--notes-file", type=Path)
    parser.add_argument("--target", help="Commit or branch for a newly created release tag")
    args = parser.parse_args(argv)
    if not args.archive.is_file():
        parser.error(f"Artifact file does not exist: {args.archive}")
    notes = args.notes_file.read_text(encoding="utf-8") if args.notes_file else (
        "Processed main/pilot subsets for the reproducible course project. "
        "Raw CSV and credentials are excluded. Verify SHA-256 before extraction."
    )
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GCM_INTERACTIVE": "Never"}
    result = subprocess.run(["git", "credential", "fill"], input="protocol=https\nhost=github.com\n\n",
                            capture_output=True, text=True, env=env, check=True)
    credentials = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    headers = {"Authorization": f"Bearer {credentials['password']}",
               "Accept": "application/vnd.github+json", "User-Agent": "recipe-artifact-publisher"}

    def request(url, payload=None, content_type="application/json"):
        data = json.dumps(payload).encode() if isinstance(payload, dict) else payload
        req = urllib.request.Request(url, data=data, headers={**headers, "Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=180) as response:
            return json.load(response)

    api = f"https://api.github.com/repos/{args.repo}/releases"
    try:
        release = request(f"{api}/tags/{args.tag}")
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        release = request(api, {"tag_name": args.tag, "name": args.name, "body": notes,
                               **({"target_commitish": args.target} if args.target else {})})
    existing = next((a for a in release["assets"] if a["name"] == args.archive.name), None)
    if existing is None:
        from urllib.parse import quote
        existing = request(release["upload_url"].split("{")[0] + "?name=" + quote(args.archive.name),
                           args.archive.read_bytes(),
                           mimetypes.guess_type(args.archive.name)[0] or "application/octet-stream")
    import hashlib
    print(json.dumps({"url": existing["browser_download_url"],
                      "sha256": hashlib.file_digest(args.archive.open("rb"), "sha256").hexdigest(),
                      "release": release["html_url"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
