#!/usr/bin/env python
"""
Run the IndependentVerifier over saved agent responses (reference-free signal).

Usage:
    python scripts/run_verifier.py \\
        --responses data/results/batch/gpt-4o_dev/agent_responses.jsonl \\
        [--n N] \\
        [--output <dir>]

For each saved agent response this script:

  1. Loads the matching DABench task (dev split) by task_id.
  2. Runs ``IndependentVerifier().verify(task, response)`` which:
       - re-solves the task COLD with Claude Sonnet 4.6 (no agent code/answer,
         no reference), and
       - re-executes the agent's own code to check reproducibility.
  3. Appends the VerifierResult as a JSON line to
       ``<output>/verifier_results.jsonl``.

It is resume-safe: any task_id already present in the output file is skipped.
Default ``--output`` is the directory containing ``--responses``.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_OUTPUT_FILENAME = "verifier_results.jsonl"


def _load_responses(path: Path) -> list[Any]:
    """Load AgentResponse objects from a JSONL file.

    Maps each dict onto ``AgentResponse`` via its ``__dataclass_fields__`` so
    extra keys are dropped and missing keys fall back to ``None``.
    """
    import jsonlines

    from src.benchmark.task import AgentResponse

    field_names = set(AgentResponse.__dataclass_fields__.keys())
    responses: list[AgentResponse] = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            kwargs = {name: obj.get(name) for name in field_names}
            responses.append(AgentResponse(**kwargs))
    return responses


def _existing_task_ids(path: Path) -> set[int]:
    """Return the set of task_ids already recorded in an output JSONL file."""
    if not path.exists():
        return set()
    done: set[int] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            tid = row.get("task_id")
            if tid is not None:
                done.add(tid)
    return done


def _serialize(result: Any) -> dict[str, Any]:
    """Convert a VerifierResult to a JSON-serializable dict.

    Defensively casts any ``Path`` values to ``str`` so serialization never
    fails on a stray path object.
    """
    row = asdict(result)
    for key, value in list(row.items()):
        if isinstance(value, Path):
            row[key] = str(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the independent reference-free verifier over saved "
        "agent responses."
    )
    parser.add_argument(
        "--responses", required=True, help="Path to agent_responses.jsonl"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Only verify the first N responses (default: all).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory (default: directory containing --responses).",
    )
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: responses file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / _OUTPUT_FILENAME

    # --- environment -----------------------------------------------------
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    from tqdm import tqdm

    from src.benchmark.loader import DABenchLoader
    from src.judge.verifier import IndependentVerifier

    # --- load responses --------------------------------------------------
    print(f"Loading responses from {responses_path} ...")
    responses = _load_responses(responses_path)
    if args.n is not None:
        responses = responses[: args.n]
    print(f"  {len(responses)} response(s) to verify.")

    # --- resume state ----------------------------------------------------
    done = _existing_task_ids(out_path)
    if done:
        print(
            f"  Resume: {len(done)} result(s) already in {out_path.name}; "
            "will skip those."
        )

    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    verifier = IndependentVerifier()

    # --- tallies for the end-of-run summary ------------------------------
    n_agree = 0
    n_disagree = 0
    n_cannot_compare = 0
    n_verifier_exec_failed = 0
    n_agent_reproduces = 0
    n_agent_reproduces_total = 0  # responses where reproduction was defined
    skipped_no_task = 0
    n_new = 0

    handle = out_path.open("a", encoding="utf-8")
    try:
        with tqdm(total=len(responses), desc="Verifying", unit="task") as bar:
            for response in responses:
                if response.task_id in done:
                    bar.update(1)
                    continue

                try:
                    task = loader.get_task(response.task_id)
                except KeyError:
                    skipped_no_task += 1
                    bar.update(1)
                    continue

                result = verifier.verify(task, response)

                handle.write(json.dumps(_serialize(result)) + "\n")
                handle.flush()
                done.add(response.task_id)
                n_new += 1

                # --- tally ---
                if result.agrees_with_agent is True:
                    n_agree += 1
                elif result.agrees_with_agent is False:
                    n_disagree += 1
                else:
                    n_cannot_compare += 1

                if not result.verifier_execution_ok:
                    n_verifier_exec_failed += 1

                if result.agent_code_reproduces is not None:
                    n_agent_reproduces_total += 1
                    if result.agent_code_reproduces:
                        n_agent_reproduces += 1

                bar.update(1)
    finally:
        handle.close()

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Verifier run complete.")
    print(f"  New results written: {n_new}")
    if skipped_no_task:
        print(f"  Skipped (task not in dev split): {skipped_no_task}")

    print("\nVerifier vs agent (re-solve agreement):")
    print(f"  agree            {n_agree}")
    print(f"  disagree         {n_disagree}")
    print(f"  unable-to-compare {n_cannot_compare}")
    print(f"  verifier-exec-failed {n_verifier_exec_failed}")

    print("\nAgent-code reproducibility (re-execution):")
    if n_agent_reproduces_total:
        print(
            f"  reproduces       {n_agent_reproduces}/{n_agent_reproduces_total}"
        )
    else:
        print("  (no responses with comparable agent code)")

    print(f"\nOutput file: {out_path}")


if __name__ == "__main__":
    main()
