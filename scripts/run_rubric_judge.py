#!/usr/bin/env python
"""
Run the rubric LLM judge over saved agent responses (Steps 11, §16/§21).

Usage:
    python scripts/run_rubric_judge.py \\
        --responses data/results/batch/gpt-4o_dev/agent_responses.jsonl \\
        --mode {no_reference|with_reference|both} \\
        [--n N] \\
        [--output <dir>]

For each (task, agent response) pair this script:

  1. Loads the DABench task (dev split) by task_id.
  2. Summarizes the task's CSV once per unique file (cached).
  3. Runs the deterministic pre-checks once.
  4. Calls ``RubricJudge.judge`` for each requested mode.
  5. Fills the deterministic scoring layer on the result:
       computed_process_quality_score, computed_overall_pass,
       builtin_pass, classification.
  6. Appends each result as a JSONL row to the per-mode output file.

It is resume-safe: any (task_id, mode) pair already present in the target
output file is skipped, so the script can be re-run to fill in gaps.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

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

_OUTPUT_FILENAMES: dict[str, str] = {
    "no_reference": "judge_results_no_reference.jsonl",
    "with_reference": "judge_results_with_reference.jsonl",
}


def _modes_for(mode: str) -> list[str]:
    """Resolve the --mode argument into the concrete list of judge modes."""
    if mode == "both":
        return ["no_reference", "with_reference"]
    return [mode]


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


def _judge_result_dict(result: Any) -> dict[str, Any]:
    """Pull the 7 score fields + critical_errors off a RubricJudgeResult."""
    d = {field: getattr(result, field) for field in _SCORE_FIELDS}
    d["critical_errors"] = result.critical_errors
    return d


def _serialize(result: Any) -> dict[str, Any]:
    """Convert a RubricJudgeResult to a JSON-serializable dict.

    Uses ``dataclasses.asdict`` and defensively casts any ``Path`` values to
    ``str`` so serialization never fails on a stray path object.
    """
    row = asdict(result)
    for key, value in list(row.items()):
        if isinstance(value, Path):
            row[key] = str(value)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rubric LLM judge over saved agent responses."
    )
    parser.add_argument(
        "--responses", required=True, help="Path to agent_responses.jsonl"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=["no_reference", "with_reference", "both"],
        help="Judge mode(s) to run.",
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
    parser.add_argument(
        "--prompt-version",
        choices=["v1", "v2"],
        default="v1",
        help=(
            "Which judge prompt version to use (default: v1). Selects the "
            "judge_{mode}_{version}.txt templates; result rows are stamped "
            "judge_<version>."
        ),
    )
    parser.add_argument(
        "--verifier",
        default=None,
        help=(
            "Optional path to a verifier_results.jsonl produced by the "
            "independent verifier. When given, the matching per-task verifier "
            "dict is fed into the judge prompt and verifier/judge conflicts "
            "are recorded."
        ),
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
    from src.judge.rubric_judge import RubricJudge
    from src.judge.schema import summarize_csv
    from src.judge.scoring import (
        classify_run,
        compute_overall_pass,
        compute_process_quality_score,
        detect_verifier_judge_conflict,
    )

    modes = _modes_for(args.mode)

    # --- load responses --------------------------------------------------
    print(f"Loading responses from {responses_path} ...")
    responses = _load_responses(responses_path)
    if args.n is not None:
        responses = responses[: args.n]
    print(f"  {len(responses)} responses to judge.")

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

    # --- resume state: which (task_id, mode) pairs are already done ------
    done_by_mode: dict[str, set[int]] = {}
    for mode in modes:
        out_path = output_dir / _OUTPUT_FILENAMES[mode]
        done_by_mode[mode] = _existing_task_ids(out_path)
        if done_by_mode[mode]:
            print(
                f"  Resume: {len(done_by_mode[mode])} {mode} result(s) already "
                f"in {out_path.name}; will skip those."
            )

    # --- loader + judge --------------------------------------------------
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    judge = RubricJudge(prompt_version=args.prompt_version)

    # CSV summaries cached by file_name (many tasks reuse the same CSV).
    summary_cache: dict[str, dict] = {}

    def _summary_for(task: Any) -> dict:
        key = task.file_name
        if key not in summary_cache:
            summary_cache[key] = summarize_csv(task.file_path)
        return summary_cache[key]

    # Open each mode's output file in append mode (resume-safe).
    handles = {
        mode: (output_dir / _OUTPUT_FILENAMES[mode]).open("a", encoding="utf-8")
        for mode in modes
    }

    # --- tallies for the end-of-run summary ------------------------------
    classification_counts: Counter[str] = Counter()
    conflict_counts: Counter[str] = Counter()  # verifier/judge conflicts
    n_judged = 0          # number of (task, mode) judgments newly written
    n_judge_pass = 0      # judgments whose computed_overall_pass is True
    builtin_seen: set[int] = set()
    builtin_pass_count = 0
    skipped_no_task = 0

    try:
        total_units = len(responses) * len(modes)
        with tqdm(total=total_units, desc="Judging", unit="judgment") as bar:
            for response in responses:
                # Resolve the task (skip if it isn't in the dev split).
                try:
                    task = loader.get_task(response.task_id)
                except KeyError:
                    skipped_no_task += 1
                    bar.update(len(modes))
                    continue

                # Track builtin pass-rate denominator (once per response).
                builtin = response.builtin_eval_result
                builtin_pass: Optional[bool]
                if builtin is None:
                    builtin_pass = None
                else:
                    builtin_pass = bool(builtin == 1.0)
                    if response.task_id not in builtin_seen:
                        builtin_seen.add(response.task_id)
                        if builtin_pass:
                            builtin_pass_count += 1

                dataset_summary = _summary_for(task)
                prechecks = run_prechecks(task, response, dataset_summary)

                # Independent-verifier dict for this task (None when absent).
                verifier_dict = verifier_map.get(response.task_id)

                for mode in modes:
                    # Resume-safe: skip already-judged (task_id, mode) pairs.
                    if response.task_id in done_by_mode[mode]:
                        bar.update(1)
                        continue

                    result = judge.judge(
                        task=task,
                        response=response,
                        dataset_summary=dataset_summary,
                        prechecks=prechecks,
                        mode=mode,
                        verifier_result=verifier_dict,
                    )

                    # --- deterministic scoring layer ---
                    jr = _judge_result_dict(result)
                    result.computed_process_quality_score = (
                        compute_process_quality_score(jr)
                    )
                    result.computed_overall_pass = compute_overall_pass(jr)
                    result.builtin_pass = builtin_pass
                    result.classification = classify_run(
                        builtin_pass,
                        result.computed_overall_pass,
                        result.computed_process_quality_score,
                    )

                    # --- verifier/judge conflict (None when no --verifier) ---
                    conflict = detect_verifier_judge_conflict(
                        verifier_dict, result.computed_overall_pass
                    )

                    # --- persist ---
                    row = _serialize(result)
                    row["verifier_conflict"] = conflict
                    handles[mode].write(json.dumps(row) + "\n")
                    handles[mode].flush()
                    done_by_mode[mode].add(response.task_id)

                    if conflict is not None:
                        conflict_counts[conflict] += 1

                    # --- tally ---
                    classification_counts[result.classification] += 1
                    n_judged += 1
                    if result.computed_overall_pass:
                        n_judge_pass += 1

                    bar.update(1)
    finally:
        for handle in handles.values():
            handle.close()

    # --- summary ---------------------------------------------------------
    print("\n" + "=" * 60)
    print("Judge run complete.")
    print(f"  Modes: {', '.join(modes)}")
    print(f"  New judgments written: {n_judged}")
    if skipped_no_task:
        print(f"  Skipped (task not in dev split): {skipped_no_task}")

    print("\nClassification counts (new judgments):")
    if classification_counts:
        for label, count in classification_counts.most_common():
            print(f"  {label:<28} {count}")
    else:
        print("  (none — all requested judgments already present)")

    if args.verifier:
        print("\nVerifier/judge conflict counts (new judgments):")
        for label in ("unconfirmed_pass", "nonreproducible_pass", "confirmed_fail"):
            print(f"  {label:<28} {conflict_counts.get(label, 0)}")

    if n_judged:
        judge_rate = n_judge_pass / n_judged
        print(f"\nJudge pass rate (computed_overall_pass): "
              f"{judge_rate:.1%}  ({n_judge_pass}/{n_judged})")
    n_builtin = len(builtin_seen)
    if n_builtin:
        builtin_rate = builtin_pass_count / n_builtin
        print(f"Built-in pass rate:                      "
              f"{builtin_rate:.1%}  ({builtin_pass_count}/{n_builtin})")

    print("\nOutput files:")
    for mode in modes:
        print(f"  {mode:<16} -> {output_dir / _OUTPUT_FILENAMES[mode]}")


if __name__ == "__main__":
    main()
