#!/usr/bin/env python
"""
Run the self-validating LLM judge over saved agent responses.

Usage:
    python scripts/run_self_validating_judge.py \\
        --responses data/results/batch/gpt-4o_dev/agent_responses.jsonl \\
        [--n N] \\
        [--output <dir>]

Unlike the plain rubric judge (which only reads code/logs) or the independent
verifier (a different model that re-solves), the ``SelfValidatingJudge`` is an
agentic Opus 4.7 that recomputes the answer itself with a code-execution tool,
in a THINK -> DO loop, before rendering its verdict.

For each (task, agent response) pair this script:

  1. Loads the DABench task (dev split) by task_id.
  2. Summarizes the task's CSV once per unique file (cached).
  3. Runs the deterministic pre-checks once.
  4. Calls ``SelfValidatingJudge.judge`` (no_reference mode).
  5. Applies the deterministic scoring layer
     (computed_process_quality_score, computed_overall_pass).
  6. Appends each result as a JSONL row to ``judge_results_self_validate.jsonl``.

It is resume-safe: any task_id already present in the output file is skipped.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_OUTPUT_FILENAME = "judge_results_self_validate.jsonl"

# The seven 1-5 score fields the deterministic scoring layer consumes.
_SCORE_FIELDS: tuple[str, ...] = (
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
)


def _load_responses(path: Path) -> list[Any]:
    """Load AgentResponse objects from a JSONL file (tolerant of schema drift)."""
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
    """Return the set of task_ids already recorded in the output JSONL file."""
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the self-validating LLM judge over saved agent responses."
    )
    parser.add_argument(
        "--responses", required=True, help="Path to agent_responses.jsonl"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Only judge the first N responses (default: all).",
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

    # --- environment -----------------------------------------------------
    try:
        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    from tqdm import tqdm

    from src.benchmark.loader import DABenchLoader
    from src.judge.prechecks import run_prechecks
    from src.judge.schema import summarize_csv
    from src.judge.scoring import (
        compute_overall_pass,
        compute_process_quality_score,
    )
    from src.judge.self_validating_judge import SelfValidatingJudge

    # --- load responses --------------------------------------------------
    print(f"Loading responses from {responses_path} ...")
    responses = _load_responses(responses_path)
    if args.n is not None:
        responses = responses[: args.n]
    print(f"  {len(responses)} responses to judge.")

    out_path = output_dir / _OUTPUT_FILENAME
    done = _existing_task_ids(out_path)
    if done:
        print(
            f"  Resume: {len(done)} result(s) already in {out_path.name}; "
            "will skip those."
        )

    # --- loader + judge --------------------------------------------------
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    judge = SelfValidatingJudge()

    # CSV summaries cached by file_name (many tasks reuse the same CSV).
    summary_cache: dict[str, dict] = {}

    def _summary_for(task: Any) -> dict:
        key = task.file_name
        if key not in summary_cache:
            summary_cache[key] = summarize_csv(task.file_path)
        return summary_cache[key]

    # --- tallies for the end-of-run summary ------------------------------
    agree_counts: Counter[str] = Counter()
    n_judged = 0
    n_judge_pass = 0
    total_tool_calls = 0
    skipped_no_task = 0

    handle = out_path.open("a", encoding="utf-8")
    try:
        with tqdm(total=len(responses), desc="Self-validating", unit="task") as bar:
            for response in responses:
                # Resume-safe: skip already-judged task_ids.
                if response.task_id in done:
                    bar.update(1)
                    continue

                # Resolve the task (skip if it isn't in the dev split).
                try:
                    task = loader.get_task(response.task_id)
                except KeyError:
                    skipped_no_task += 1
                    bar.update(1)
                    continue

                dataset_summary = _summary_for(task)
                prechecks = run_prechecks(task, response, dataset_summary)

                verdict = judge.judge(
                    task=task,
                    response=response,
                    dataset_summary=dataset_summary,
                    prechecks=prechecks,
                    mode="no_reference",
                )

                # --- deterministic scoring layer ---
                jr = {field: verdict[field] for field in _SCORE_FIELDS}
                jr["critical_errors"] = verdict["critical_errors"]
                computed_process = compute_process_quality_score(jr)
                computed_pass = compute_overall_pass(jr)

                row: dict[str, Any] = {
                    "task_id": task.id,
                    "judge_model": judge.model,
                    "judge_prompt_version": judge.PROMPT_VERSION,
                    "judge_mode": "no_reference",
                    "task_success_score": verdict["task_success_score"],
                    "code_correctness_score": verdict["code_correctness_score"],
                    "data_handling_score": verdict["data_handling_score"],
                    "statistical_validity_score": verdict[
                        "statistical_validity_score"
                    ],
                    "reproducibility_score": verdict["reproducibility_score"],
                    "final_answer_quality_score": verdict[
                        "final_answer_quality_score"
                    ],
                    "reasoning_quality_score": verdict["reasoning_quality_score"],
                    "critical_errors": verdict["critical_errors"],
                    "overall_pass": verdict["overall_pass"],
                    "confidence": verdict["confidence"],
                    "judge_explanation": verdict["judge_explanation"],
                    "self_derived_answer": verdict["self_derived_answer"],
                    "self_validation_agrees": verdict["self_validation_agrees"],
                    "validation_summary": verdict["validation_summary"],
                    "n_tool_calls": verdict["n_tool_calls"],
                    "computed_process_quality_score": computed_process,
                    "computed_overall_pass": computed_pass,
                    "parse_success": verdict["parse_success"],
                }

                handle.write(json.dumps(row) + "\n")
                handle.flush()
                done.add(task.id)

                # --- tally ---
                agree_counts[verdict["self_validation_agrees"]] += 1
                total_tool_calls += verdict["n_tool_calls"]
                n_judged += 1
                if computed_pass:
                    n_judge_pass += 1

                bar.update(1)
    finally:
        handle.close()

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Self-validating judge run complete.")
    print(f"  New judgments written: {n_judged}")
    if skipped_no_task:
        print(f"  Skipped (task not in dev split): {skipped_no_task}")

    print("\nSelf-validation agreement (new judgments):")
    if agree_counts:
        for label in ("agree", "disagree", "could_not_validate"):
            print(f"  {label:<22} {agree_counts.get(label, 0)}")
    else:
        print("  (none — all requested judgments already present)")

    if n_judged:
        mean_tool_calls = total_tool_calls / n_judged
        print(f"\nMean run_python calls per task: {mean_tool_calls:.2f}")
        judge_rate = n_judge_pass / n_judged
        print(
            f"Judge pass rate (computed_overall_pass): "
            f"{judge_rate:.1%}  ({n_judge_pass}/{n_judged})"
        )

    print(f"\nOutput file: {out_path}")


if __name__ == "__main__":
    main()
