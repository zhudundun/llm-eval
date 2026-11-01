#!/usr/bin/env python
"""
Judge calibration: compares ClaudeJudge against ProgrammaticJudge as oracle.

Usage:
    python scripts/calibrate.py \\
        --responses data/results/run1/agent_responses.jsonl \\
        [--n 100] \\
        [--seed 42]

Samples N responses, scores each with both the programmatic judge (oracle)
and the Claude LLM judge, computes calibration metrics, and prints a report
including a bootstrap CI on Claude judge accuracy.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_responses(path: Path):
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
    parser = argparse.ArgumentParser(description="Calibrate ClaudeJudge against ProgrammaticJudge.")
    parser.add_argument("--responses", required=True, help="Path to agent_responses.jsonl")
    parser.add_argument("--n", type=int, default=100, help="Number of responses to sample (default: 100)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    responses_path = Path(args.responses)
    if not responses_path.exists():
        print(f"Error: responses file not found: {responses_path}", file=sys.stderr)
        sys.exit(1)

    # Load environment
    try:
        from dotenv import load_dotenv
        load_dotenv(REPO_ROOT / ".env")
    except ImportError:
        pass

    import yaml
    with open(REPO_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    # Load and sample responses
    print(f"Loading responses from {responses_path} ...")
    all_responses = _load_responses(responses_path)
    rng = random.Random(args.seed)
    sample = rng.sample(all_responses, min(args.n, len(all_responses)))
    print(f"  Sampled {len(sample)} responses (seed={args.seed}).")

    # Load tasks
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    task_map = {t.id: t for t in tasks}

    # Build judges
    from src.judge.programmatic import ProgrammaticJudge
    from src.judge.claude_judge import ClaudeJudge

    prog_judge = ProgrammaticJudge()
    claude_judge = ClaudeJudge(model=cfg["judge"]["model"])

    print(f"Running ProgrammaticJudge (oracle) and ClaudeJudge on {len(sample)} responses...")

    prog_scores: list[float] = []
    claude_scores: list[float] = []
    skipped = 0

    for i, resp in enumerate(sample, 1):
        task = task_map.get(resp.task_id)
        if task is None:
            skipped += 1
            continue

        ps = prog_judge.score(task, resp)
        cs = claude_judge.score(task, resp)
        prog_scores.append(ps.score)
        claude_scores.append(cs.score)

        if i % 10 == 0:
            print(f"  [{i}/{len(sample)}] prog={ps.score:.0f}  claude={cs.score:.0f}")

    if skipped:
        print(f"  Warning: {skipped} responses skipped (task not found in dev set).")

    if not prog_scores:
        print("No scoreable responses found.", file=sys.stderr)
        sys.exit(1)

    # Calibration report
    from src.calibration.metrics import compute_calibration, print_calibration_report
    from src.reliability.bootstrap import bootstrap_ci

    print("\n--- Calibration: ClaudeJudge vs ProgrammaticJudge (oracle) ---")
    results = compute_calibration(claude_scores, prog_scores)
    print_calibration_report(results)

    # Bootstrap CI on Claude judge accuracy
    rel_cfg = cfg["reliability"]
    mean, lower, upper = bootstrap_ci(
        claude_scores,
        n_bootstrap=rel_cfg["n_bootstrap"],
        ci=rel_cfg["ci"],
        random_seed=args.seed,
    )
    ci_pct = int(rel_cfg["ci"] * 100)
    print(f"\nClaude judge accuracy:  {mean:.3f}  [{ci_pct}% CI: {lower:.3f} – {upper:.3f}]")

    # Agreement summary
    agreements = sum(p == c for p, c in zip(prog_scores, claude_scores))
    print(f"Agreement:              {agreements}/{len(prog_scores)}  ({agreements/len(prog_scores):.1%})")


if __name__ == "__main__":
    main()
