"""
End-to-end test: run 5 InfiAgent-DABench tasks through GPT-4o and save rich
JSONL rows to data/results/e2e_test/responses.jsonl.

Usage:
    cd /Users/wang/LLM_eval
    source .venv/bin/activate
    python scripts/test_e2e.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load .env before anything else — fail gracefully if missing
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("WARNING: python-dotenv not installed. Relying on system environment.")

if not os.environ.get("OPENAI_API_KEY"):
    print(
        "ERROR: OPENAI_API_KEY is not set.\n"
        "Create a .env file in the project root with:\n"
        "    OPENAI_API_KEY=sk-...\n"
        "or export the variable in your shell before running this script."
    )
    sys.exit(1)

# ---------------------------------------------------------------------------
# Project imports (must come after environment is ready)
# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so `src` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmark.loader import DABenchLoader
from src.agents.openai_agent import OpenAIAgent

try:
    from tabulate import tabulate
except ImportError:
    print("ERROR: 'tabulate' is not installed. Run: pip install tabulate")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data" / "dabench"
OUTPUT_DIR = PROJECT_ROOT / "data" / "results" / "e2e_test"
OUTPUT_FILE = OUTPUT_DIR / "responses.jsonl"

# Task IDs to run: one easy, one medium, one hard, plus two more for variety.
# Chosen from the dev split (verified against da-dev-questions.jsonl).
#   easy:   0  (mean fare — simple aggregation)
#   easy:   9  (another easy task for coverage)
#   medium: 5  (FamilySize correlation — feature engineering + correlation)
#   medium: 6  (AgeGroup mean fare — feature engineering + groupby)
#   hard:   7  (linear regression with sklearn)
TARGET_IDS = [0, 9, 5, 6, 7]


def _truncate(text: str, max_len: int = 300) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"... [+{len(text) - max_len} chars]"


def main() -> None:
    # --- Load tasks --------------------------------------------------------
    loader = DABenchLoader(DATA_DIR)
    all_tasks = loader.load_tasks(split="dev")
    task_map = {t.id: t for t in all_tasks}

    selected = []
    for tid in TARGET_IDS:
        if tid not in task_map:
            print(f"WARNING: task id={tid} not found in dev split, skipping.")
        else:
            selected.append(task_map[tid])

    if not selected:
        print("ERROR: No tasks selected. Aborting.")
        sys.exit(1)

    print(f"Running {len(selected)} tasks with GPT-4o...")
    print(f"Output: {OUTPUT_FILE}\n")

    # --- Set up output file ------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Run tasks ---------------------------------------------------------
    agent = OpenAIAgent(model="gpt-4o", max_attempts=3, timeout=60)
    responses = []

    for task in selected:
        level_label = task.level.ljust(6)
        print(f"[task {task.id:3d} | {level_label}] {task.question[:80]}")

        response = agent.solve(task)
        responses.append(response)

        # Serialize to JSONL immediately
        row = {}
        # Use dataclasses.asdict for serialisation; convert Path to str
        import dataclasses
        raw = dataclasses.asdict(response)
        for k, v in raw.items():
            row[k] = str(v) if isinstance(v, Path) else v

        with open(OUTPUT_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

        # Print code + stdout for debuggability
        print(f"  Generated code:\n    {_truncate(response.code, 300)}")
        print(f"  Stdout: {_truncate(response.stdout, 300)}")
        if response.stderr:
            print(f"  Stderr: {_truncate(response.stderr, 200)}")
        print(
            f"  exec={response.execution_result}  "
            f"answer={response.answer}  "
            f"builtin_eval={response.builtin_eval_result}  "
            f"runtime={response.runtime_seconds:.1f}s  "
            f"tokens={response.token_usage.get('total_tokens', 0)}"
        )
        print()

    # --- Summary table -----------------------------------------------------
    total_tokens = sum(r.token_usage.get("total_tokens", 0) for r in responses)
    total_runtime = sum(r.runtime_seconds for r in responses)

    table_rows = []
    for r in responses:
        task = task_map[r.task_id]
        answer_str = str(r.answer) if r.answer else "(none)"
        if len(answer_str) > 30:
            answer_str = answer_str[:27] + "..."
        table_rows.append([
            r.task_id,
            task.level,
            r.execution_result,
            answer_str,
            r.builtin_eval_result if r.builtin_eval_result is not None else "N/A",
            f"{r.runtime_seconds:.1f}s",
            r.token_usage.get("total_tokens", 0),
        ])

    headers = ["Task ID", "Level", "Exec result", "Answer", "Builtin eval", "Runtime", "Tokens"]
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(tabulate(table_rows, headers=headers, tablefmt="github"))
    print()

    # --- Final stats -------------------------------------------------------
    passed = [r for r in responses if r.builtin_eval_result == 1.0]
    failed = [r for r in responses if r.builtin_eval_result == 0.0]
    no_gt = [r for r in responses if r.builtin_eval_result is None]
    errors = [r for r in responses if r.execution_result in ("error", "timeout")]
    no_ans = [r for r in responses if r.execution_result == "no_answer"]

    print(f"Passed builtin eval : {len(passed)}/{len(responses)}"
          f"  (IDs: {[r.task_id for r in passed]})")
    print(f"Failed builtin eval : {len(failed)}/{len(responses)}"
          f"  (IDs: {[r.task_id for r in failed]})")
    if no_gt:
        print(f"No ground truth     : {len(no_gt)}"
              f"  (IDs: {[r.task_id for r in no_gt]})")
    if errors:
        print(f"Code errors/timeout : {len(errors)}"
              f"  (IDs: {[r.task_id for r in errors]})")
    if no_ans:
        print(f"No answer extracted : {len(no_ans)}"
              f"  (IDs: {[r.task_id for r in no_ans]})")
    print(f"Total tokens used   : {total_tokens:,}")
    print(f"Total wall-clock    : {total_runtime:.1f}s")
    print(f"\nResults saved to    : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
