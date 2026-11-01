#!/usr/bin/env python3
"""
Download / update the InfiAgent-DABench benchmark data.

Usage:
    python scripts/download_dabench.py

What it does:
  1. Clones InfiAgent/InfiAgent into data/dabench/infiagent-repo (depth=1).
     If the directory already exists it runs `git pull` instead.
  2. Loads each known split with DABenchLoader and prints the task count.

Run from the project root or any location — the script resolves paths
relative to its own location on disk.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — always relative to this script's location so the script works
# regardless of the working directory.
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "dabench"
REPO_DIR = DATA_DIR / "infiagent-repo"
REPO_URL = "https://github.com/InfiAgent/InfiAgent"

# Make src/ importable without an editable install.
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from benchmark.loader import DABenchLoader  # noqa: E402  (import after path fix)


def clone_or_pull() -> None:
    """Clone the repo if absent; pull the latest commit if it already exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not REPO_DIR.exists():
        print(f"Cloning {REPO_URL} → {REPO_DIR} (depth=1) …")
        subprocess.run(
            ["git", "clone", REPO_URL, str(REPO_DIR), "--depth", "1"],
            check=True,
        )
        print("Clone complete.")
    else:
        print(f"Repository already exists at {REPO_DIR}.")
        print("Pulling latest changes …")
        result = subprocess.run(
            ["git", "-C", str(REPO_DIR), "pull", "--ff-only"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(result.stdout.strip() or "Already up to date.")
        else:
            # Shallow repos may reject pulls in some edge cases — warn but continue.
            print(
                f"Warning: git pull returned non-zero exit code {result.returncode}.\n"
                f"stderr: {result.stderr.strip()}\n"
                "Continuing with existing data."
            )


def count_tasks(split: str, loader: DABenchLoader) -> int:
    """Return the number of tasks in *split*, or -1 on error."""
    try:
        tasks = loader.load_tasks(split)
        return len(tasks)
    except FileNotFoundError as exc:
        print(f"  [skip] split={split!r} — {exc}")
        return -1


def main() -> None:
    clone_or_pull()

    loader = DABenchLoader(DATA_DIR)

    print("\nTask counts per split:")
    print("-" * 35)
    for split in ("dev", "test"):
        n = count_tasks(split, loader)
        if n >= 0:
            print(f"  {split:<6}  {n:>4} tasks")

    print("\nDone.")


if __name__ == "__main__":
    main()
