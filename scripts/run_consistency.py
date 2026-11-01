#!/usr/bin/env python
"""
Self-consistency test runner for the rubric LLM judge (§19.1).

Calls the SAME judge on the SAME fixed inputs N times per task and records
every result, so verdict flips, score variance, and confidence swings can be
detected downstream.

Usage:
    python scripts/run_consistency.py \\
        --responses data/results/batch/gpt-4o_dev/agent_responses.jsonl \\
        [--verifier data/results/batch/gpt-4o_dev/verifier_results.jsonl] \\
        [--repeats 5] \\
        [--n N] \\
        [--mode no_reference] \\
        [--prompt-version v1] \\
        [--output <dir>]

For each (task, agent response) pair, the FIXED inputs are computed exactly
once and held constant across repeats:

  * the CSV ``dataset_summary`` (cached per unique file_name),
  * the deterministic ``prechecks``,
  * the per-task verifier dict (when --verifier is given).

Only the ``RubricJudge.judge(...)`` call repeats; a single ``RubricJudge``
instance is constructed up front and reused.  The judge model is Opus 4.7
(the RubricJudge default) and temperature is deliberately NOT passed (Opus
rejects sampling parameters) — its natural run-to-run variation is exactly
what this runner measures.

Each repeat's result has the deterministic scoring layer applied
(``compute_process_quality_score`` + ``compute_overall_pass``, matching
run_rubric_judge) and is written as ONE JSONL row to
``<output>/consistency_results.jsonl`` under a LOCKED schema.

It is resume-safe: any (task_id, repeat_idx) pair already present in the
output file is skipped, so the script can be re-run to fill in gaps.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

_OUTPUT_FILENAME = "consistency_results.jsonl"

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
    """Load AgentResponse objects from a JSONL file.

    Maps each dict onto ``AgentResponse`` via its ``__dataclass_fields__`` so
    extra keys are dropped and missing keys fall back to ``None``.  This is
    tolerant of schema drift in the saved responses file.
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


def _load_verifier_map(path: Path) -> dict[int, dict[str, Any]]:
    """Load a verifier_results.jsonl into a ``{task_id: verifier_dict}`` map.

    The verifier result is treated as a plain dict per the locked JSONL
    contract (NOT the verifier dataclass).  Rows without a ``task_id`` are
    skipped; later rows for the same task_id win.
    """
    verifier_map: dict[int, dict[str, Any]] = {}
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
                verifier_map[tid] = row
    return verifier_map


def _existing_pairs(path: Path) -> set[tuple[int, int]]:
    """Return the set of (task_id, repeat_idx) pairs already in an output file."""
    if not path.exists():
        return set()
    done: set[tuple[int, int]] = set()
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
            ridx = row.get("repeat_idx")
            if tid is not None and ridx is not None:
                done.add((tid, ridx))
    return done


def _judge_result_dict(result: Any) -> dict[str, Any]:
    """Pull the 7 score fields + critical_errors off a RubricJudgeResult."""
    d = {field: getattr(result, field) for field in _SCORE_FIELDS}
    d["critical_errors"] = result.critical_errors
    return d


def _row_for(
    result: Any,
    repeat_idx: int,
    computed_process_quality_score: float,
    computed_overall_pass: bool,
) -> dict[str, Any]:
    """Build the LOCKED consistency_results.jsonl row from a scored result.

    The schema here is depended on by a sibling analyzer agent — do not change
    keys or types.
    """
    return {
        "task_id": result.task_id,
        "repeat_idx": repeat_idx,
        "judge_model": result.judge_model,
        "judge_prompt_version": result.judge_prompt_version,
        "task_success_score": result.task_success_score,
        "code_correctness_score": result.code_correctness_score,
        "data_handling_score": result.data_handling_score,
        "statistical_validity_score": result.statistical_validity_score,
        "reproducibility_score": result.reproducibility_score,
        "final_answer_quality_score": result.final_answer_quality_score,
        "reasoning_quality_score": result.reasoning_quality_score,
        "critical_errors": result.critical_errors,
        "overall_pass": result.overall_pass,
        "confidence": result.confidence,
        "computed_process_quality_score": computed_process_quality_score,
        "computed_overall_pass": computed_overall_pass,
        "parse_success": result.parse_success,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-consistency test runner for the rubric LLM judge."
    )
    parser.add_argument(
        "--responses", required=True, help="Path to agent_responses.jsonl"
    )
    parser.add_argument(
        "--verifier",
        default=None,
        help=(
            "Optional path to a verifier_results.jsonl. When given, the matching "
            "per-task verifier dict is fed into the judge prompt (FIXED across "
            "repeats)."
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Number of judge calls per task (default: 5).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Only run the first N responses (default: all).",
    )
    parser.add_argument(
        "--mode",
        choices=["no_reference", "with_reference"],
        default="no_reference",
        help="Judge mode (default: no_reference, the production default).",
    )
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help="Judge prompt version (default: v1, the production default).",
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

    if args.repeats < 1:
        print("Error: --repeats must be >= 1", file=sys.stderr)
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
    from src.judge.rubric_judge import RubricJudge
    from src.judge.schema import summarize_csv
    from src.judge.scoring import compute_overall_pass, compute_process_quality_score

    # --- load responses --------------------------------------------------
    print(f"Loading responses from {responses_path} ...")
    responses = _load_responses(responses_path)
    if args.n is not None:
        responses = responses[: args.n]
    print(f"  {len(responses)} response(s) to judge x {args.repeats} repeat(s).")

    # --- optional independent-verifier signal ----------------------------
    verifier_map: dict[int, dict[str, Any]] = {}
    if args.verifier:
        verifier_path = Path(args.verifier)
        if not verifier_path.exists():
            print(
                f"Error: verifier file not found: {verifier_path}", file=sys.stderr
            )
            sys.exit(1)
        verifier_map = _load_verifier_map(verifier_path)
        print(
            f"  Loaded {len(verifier_map)} verifier result(s) from "
            f"{verifier_path.name}."
        )

    # --- resume state: which (task_id, repeat_idx) pairs are already done -
    out_path = output_dir / _OUTPUT_FILENAME
    done_pairs = _existing_pairs(out_path)
    if done_pairs:
        print(
            f"  Resume: {len(done_pairs)} result row(s) already in "
            f"{out_path.name}; will skip those."
        )

    # --- loader + a SINGLE reused judge instance -------------------------
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    judge = RubricJudge(prompt_version=args.prompt_version)

    # CSV summaries cached by file_name (many tasks reuse the same CSV).
    summary_cache: dict[str, dict] = {}

    def _summary_for(task: Any) -> dict:
        key = task.file_name
        if key not in summary_cache:
            summary_cache[key] = summarize_csv(task.file_path)
        return summary_cache[key]

    # Track per-task computed_overall_pass values to detect non-unanimity.
    pass_by_task: dict[int, set[bool]] = {}

    n_written = 0
    skipped_no_task = 0

    handle = out_path.open("a", encoding="utf-8")
    try:
        total_units = len(responses) * args.repeats
        with tqdm(total=total_units, desc="Consistency", unit="judgment") as bar:
            for response in responses:
                # Resolve the task (skip if it isn't in the dev split).
                try:
                    task = loader.get_task(response.task_id)
                except KeyError:
                    skipped_no_task += 1
                    bar.update(args.repeats)
                    continue

                # --- FIXED inputs: computed ONCE per task ---
                dataset_summary = _summary_for(task)
                prechecks = run_prechecks(task, response, dataset_summary)
                verifier_dict: Optional[dict[str, Any]] = verifier_map.get(
                    response.task_id
                )

                for repeat_idx in range(args.repeats):
                    # Resume-safe: skip already-recorded (task_id, repeat_idx).
                    if (response.task_id, repeat_idx) in done_pairs:
                        bar.update(1)
                        continue

                    result = judge.judge(
                        task=task,
                        response=response,
                        dataset_summary=dataset_summary,
                        prechecks=prechecks,
                        mode=args.mode,
                        verifier_result=verifier_dict,
                    )

                    # --- deterministic scoring layer (per repeat) ---
                    jr = _judge_result_dict(result)
                    computed_pq = compute_process_quality_score(jr)
                    computed_pass = compute_overall_pass(jr)

                    row = _row_for(result, repeat_idx, computed_pq, computed_pass)
                    handle.write(json.dumps(row) + "\n")
                    handle.flush()

                    done_pairs.add((response.task_id, repeat_idx))
                    pass_by_task.setdefault(response.task_id, set()).add(
                        computed_pass
                    )
                    n_written += 1
                    bar.update(1)
    finally:
        handle.close()

    # --- summary ---------------------------------------------------------
    flip_tasks = [tid for tid, passes in pass_by_task.items() if len(passes) > 1]

    print("\n" + "=" * 60)
    print("Consistency run complete.")
    print(f"  Mode: {args.mode} | prompt: judge_{args.prompt_version} "
          f"| repeats: {args.repeats}")
    print(f"  Total judgments written: {n_written}")
    if skipped_no_task:
        print(f"  Skipped (task not in dev split): {skipped_no_task}")
    print(
        f"  Tasks with non-unanimous computed_overall_pass (flips): "
        f"{len(flip_tasks)}"
    )
    if flip_tasks:
        print(f"    Flipped task_ids: {sorted(flip_tasks)}")
    print(f"\nOutput file: {out_path}")


if __name__ == "__main__":
    main()
