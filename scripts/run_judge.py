#!/usr/bin/env python
"""
Judge saved agent responses.

Usage:
    python scripts/run_judge.py \\
        --responses data/results/run1/agent_responses.jsonl \\
        [--judge composite|programmatic|llm] \\
        [--output data/results/run1]

Reads task ground truth from the DABench dev loader.
Writes judge_scores.jsonl to --output (defaults to the same directory as --responses).
Prints overall accuracy and a breakdown by difficulty level.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_responses(path: Path):
    """Load AgentResponse objects from a JSONL file."""
    import jsonlines
    from src.benchmark.task import AgentResponse

    responses = []
    with jsonlines.open(path, mode="r") as reader:
        for obj in reader:
            responses.append(
                AgentResponse(
                    task_id=obj["task_id"],
                    model=obj["model"],
                    code=obj["code"],
                    stdout=obj["stdout"],
                    stderr=obj["stderr"],
                    answer=obj.get("answer", {}),
                    error=obj.get("error"),
                    attempts=obj.get("attempts", 1),
                )
            )
    return responses


def main() -> None:
    parser = argparse.ArgumentParser(description="Judge saved agent responses.")
    parser.add_argument("--responses", required=True, help="Path to agent_responses.jsonl")
    parser.add_argument(
        "--judge",
        default="programmatic",
        choices=["composite", "programmatic", "llm"],
        help="Which judge to use (default: programmatic)",
    )
    parser.add_argument("--output", default=None, help="Output directory (default: same as responses)")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: responses file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output) if args.output else responses_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load environment for LLM judges
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    import yaml
    with open(REPO_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    # Load responses
    print(f"Loading responses from {responses_path} ...")
    responses = _load_responses(responses_path)
    print(f"  {len(responses)} responses loaded.")

    # Load tasks and build id → task map
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    task_map = {t.id: t for t in tasks}

    # Build judge
    judge_name = args.judge
    if judge_name == "programmatic":
        from src.judge.programmatic import ProgrammaticJudge
        judge = ProgrammaticJudge()
    elif judge_name == "llm":
        from src.judge.claude_judge import ClaudeJudge
        judge = ClaudeJudge(model=cfg["judge"]["model"])
    else:  # composite
        from src.judge.claude_judge import ClaudeJudge
        from src.judge.composite import CompositeJudge
        judge = CompositeJudge(llm_judge=ClaudeJudge(model=cfg["judge"]["model"]))

    print(f"Judging {len(responses)} responses with judge={judge_name!r} ...")

    scores = []
    skipped = 0
    with open(output_dir / "judge_scores.jsonl", "w", encoding="utf-8") as fh:
        for resp in responses:
            task = task_map.get(resp.task_id)
            if task is None:
                skipped += 1
                continue
            score = judge.score(task, resp)
            fh.write(json.dumps(asdict(score)) + "\n")
            scores.append(score)

    if skipped:
        print(f"  Warning: {skipped} responses skipped (task not found in dev set).")

    # --- summary ---
    overall_acc = sum(s.score for s in scores) / len(scores) if scores else 0.0
    print(f"\nOverall accuracy: {overall_acc:.3f}  ({sum(s.score for s in scores):.0f}/{len(scores)})")

    # Breakdown by level
    level_scores: dict[str, list[float]] = defaultdict(list)
    for resp, score in zip(responses, scores):
        task = task_map.get(resp.task_id)
        if task:
            level_scores[task.level].append(score.score)

    print("\nBreakdown by level:")
    for level in ["easy", "medium", "hard"]:
        lvl = level_scores.get(level, [])
        if lvl:
            acc = sum(lvl) / len(lvl)
            print(f"  {level:<8}  {acc:.3f}  ({sum(lvl):.0f}/{len(lvl)})")
        else:
            print(f"  {level:<8}  (no tasks)")

    print(f"\nScores saved to {output_dir / 'judge_scores.jsonl'}")


if __name__ == "__main__":
    main()
