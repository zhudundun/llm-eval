#!/usr/bin/env python
"""
Print a summary of judge scores.

Usage:
    python scripts/analyze_results.py --scores data/results/run1/judge_scores.jsonl

Prints:
  - Overall accuracy with 95% bootstrap CI
  - Accuracy breakdown by difficulty level (easy / medium / hard)
  - Accuracy breakdown by concept category
  - Number of tasks with no answer extracted (error tasks)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))


def _load_scores(path: Path) -> list[dict]:
    scores = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                scores.append(json.loads(line))
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize judge score results.")
    parser.add_argument("--scores", required=True, help="Path to judge_scores.jsonl")
    args = parser.parse_args()

    scores_path = Path(args.scores)
    if not scores_path.exists():
        print(f"Error: scores file not found: {scores_path}", file=sys.stderr)
        sys.exit(1)

    import yaml
    with open(REPO_ROOT / "config.yaml") as fh:
        cfg = yaml.safe_load(fh)

    # Load scores
    records = _load_scores(scores_path)
    if not records:
        print("No scores found in file.", file=sys.stderr)
        sys.exit(1)

    score_by_id = {r["task_id"]: r["score"] for r in records}

    # Load tasks (for level + concepts)
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    task_map = {t.id: t for t in tasks}

    # Merge scores with task metadata
    all_scores: list[float] = []
    level_scores: dict[str, list[float]] = defaultdict(list)
    concept_scores: dict[str, list[float]] = defaultdict(list)
    n_errors = 0

    for task_id, score in score_by_id.items():
        all_scores.append(score)
        task = task_map.get(task_id)
        if task:
            level_scores[task.level].append(score)
            for concept in task.concepts:
                concept_scores[concept].append(score)
        # Count error tasks (score 0 and no answer key in responses — we proxy
        # errors as tasks that scored 0 due to reasoning containing "no parseable")
        rec = next((r for r in records if r["task_id"] == task_id), None)
        if rec and "no parseable" in rec.get("reasoning", "").lower():
            n_errors += 1

    # Bootstrap CI
    from src.reliability.bootstrap import bootstrap_ci
    rel_cfg = cfg["reliability"]
    mean, lower, upper = bootstrap_ci(
        all_scores,
        n_bootstrap=rel_cfg["n_bootstrap"],
        ci=rel_cfg["ci"],
    )
    ci_pct = int(rel_cfg["ci"] * 100)

    n_correct = sum(1 for s in all_scores if s == 1.0)
    n_total = len(all_scores)

    print("=" * 55)
    print("DABench Evaluation Results")
    print("=" * 55)
    print(f"Tasks evaluated:  {n_total}")
    print(f"Overall accuracy: {mean:.3f}  ({n_correct}/{n_total})")
    print(f"{ci_pct}% Bootstrap CI:  [{lower:.3f}, {upper:.3f}]")
    print(f"Tasks with no answer extracted (errors): {n_errors}")

    # Breakdown by level
    print("\nBreakdown by difficulty level:")
    print(f"  {'Level':<10}  {'Accuracy':>8}  {'Correct':>8}  {'Total':>6}")
    print(f"  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*6}")
    for level in ["easy", "medium", "hard"]:
        lvl = level_scores.get(level, [])
        if lvl:
            acc = sum(lvl) / len(lvl)
            print(f"  {level:<10}  {acc:>8.3f}  {sum(lvl):>8.0f}  {len(lvl):>6}")
        else:
            print(f"  {level:<10}  {'N/A':>8}  {'—':>8}  {'0':>6}")

    # Breakdown by concept category (top 10 by frequency)
    print("\nBreakdown by concept category (top 10 by task count):")
    concept_sorted = sorted(concept_scores.items(), key=lambda kv: -len(kv[1]))[:10]
    print(f"  {'Concept':<35}  {'Accuracy':>8}  {'Total':>6}")
    print(f"  {'-'*35}  {'-'*8}  {'-'*6}")
    for concept, cscores in concept_sorted:
        acc = sum(cscores) / len(cscores)
        print(f"  {concept:<35}  {acc:>8.3f}  {len(cscores):>6}")

    print("=" * 55)


if __name__ == "__main__":
    main()
