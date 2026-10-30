"""Upload huggingface_space/ to the public Space, with the gate and the exclusions enforced.

    set PYTHONUTF8=1
    venv\\Scripts\\python.exe huggingface_space\\tools\\upload_space.py            # dry run: lists files
    venv\\Scripts\\python.exe huggingface_space\\tools\\upload_space.py --push     # uploads

The token is read from the HF_TOKEN environment variable only. It is never a file in this
repo, never an argument, never printed. Set it once per machine from the clipboard without
it ever appearing on screen:

    [Environment]::SetEnvironmentVariable("HF_TOKEN", (Get-Clipboard), "User")

then open a NEW PowerShell window.

Order of operations, each one a hard stop:
  1. `sync_shared.py` must exit 0 (the shared copies are current and the folder is deployable).
  2. The exclusion list below (same as .dockerignore) is applied; a `.env` in the folder is a
     hard error, not a skipped file.
  3. Upload with huggingface_hub, as one commit to the Space.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

SPACE_ID = "ckharche/ProfitBook"
HERE = Path(__file__).resolve().parents[1]          # huggingface_space/
REPO = HERE.parent
IGNORE = [".env", ".env.*", "*.pem", "*.key", "creds*", "__pycache__/*", "*.pyc",
          ".pytest_cache/*", "*_test.py", "verify_fix.py", ".dockerignore"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--push", action="store_true", help="actually upload; default is a dry run")
    a = ap.parse_args(argv)

    secrets = [p for p in HERE.rglob("*") if p.name == ".env" or p.name.startswith(".env.")
               or p.name.startswith("creds") or p.suffix in (".pem", ".key")]
    if secrets:
        print("REFUSING: secret-looking files inside huggingface_space/:",
              *[str(p.relative_to(REPO)) for p in secrets], sep="\n  ")
        return 2

    gate = subprocess.run([sys.executable, str(HERE / "sync_shared.py")], cwd=str(REPO))
    if gate.returncode != 0:
        print(f"REFUSING: sync_shared.py exited {gate.returncode}; fix the drift first")
        return gate.returncode

    files = [p for p in HERE.rglob("*") if p.is_file()
             and not any(p.match(pat) for pat in IGNORE)
             and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts]
    print(f"{len(files)} files would be uploaded to {SPACE_ID}")
    if not a.push:
        for p in sorted(files)[:40]:
            print("  ", p.relative_to(HERE))
        if len(files) > 40:
            print(f"   ... and {len(files) - 40} more")
        print("dry run; add --push to upload")
        return 0

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("REFUSING: HF_TOKEN is not set in this shell (see the docstring)")
        return 2
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.upload_folder(folder_path=str(HERE), repo_id=SPACE_ID, repo_type="space",
                      ignore_patterns=IGNORE,
                      commit_message="Upload rebuilt Space (research showcase + labelled demo)")
    print(f"uploaded to https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
