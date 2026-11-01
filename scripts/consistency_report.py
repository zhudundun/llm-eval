#!/usr/bin/env python
"""
Build the LLM-judge self-consistency report (spec §19.1).

Usage:
    python scripts/consistency_report.py \\
        --results <consistency_results.jsonl> \\
        [--out reports/consistency_report.md]

Reads the repeated-judgment JSONL (one row per (task_id, repeat_idx)) produced
by the sibling runner, aggregates per task via ``analyze_consistency``, prints
the summary dict, and writes a markdown report with:

  * a Summary block (headline: pass/fail flip_rate),
  * Table 1 — Pass/Fail flips (same answer got both pass and fail),
  * Table 2 — High score variance (any dimension var > 1.0),
  * Table 3 — Wide confidence (confidence range > 0.2),
  * a note listing tasks flagged by ALL three criteria (most unstable).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.reliability.consistency_analysis import (  # noqa: E402
    CONFIDENCE_RANGE_THRESHOLD,
    SCORE_VAR_THRESHOLD,
    analyze_consistency,
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load consistency-result JSONL rows into a list of dicts."""
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
    return rows


def _verdict_str(verdicts: list[bool]) -> str:
    """Render a verdict list as e.g. 'P,P,F,P,F'."""
    return ",".join("P" if v else "F" for v in verdicts)


def _flip_table(per_task: dict[Any, dict[str, Any]]) -> str:
    """Table 1 — pass/fail flips, sorted worst-first."""
    flipped = [a for a in per_task.values() if a["pass_fail_flip"]]
    if not flipped:
        return "_No pass/fail flips._"
    # Worst-first: closer to a 50/50 split is more unstable, then higher
    # process-score variance, then wider confidence range.
    flipped.sort(
        key=lambda a: (
            abs(a["pass_rate"] - 0.5),
            -a["process_score_variance"],
            -a["confidence_range"],
        )
    )
    headers = [
        "task_id",
        "verdicts",
        "pass_rate",
        "process_var",
        "conf_range",
    ]
    table = [
        [
            a["task_id"],
            _verdict_str(a["verdicts"]),
            f"{a['pass_rate']:.2f}",
            f"{a['process_score_variance']:.4f}",
            f"{a['confidence_range']:.4f}",
        ]
        for a in flipped
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def _high_var_table(per_task: dict[Any, dict[str, Any]]) -> str:
    """Table 2 — high score variance, sorted worst-first."""
    flagged = [a for a in per_task.values() if a["high_score_variance"]]
    if not flagged:
        return "_No tasks with dimension variance above threshold._"
    flagged.sort(key=lambda a: -a["max_dim_variance"])
    headers = ["task_id", "offending_dimensions (variance)", "max_dim_variance"]
    table = []
    for a in flagged:
        offenders = [
            f"{dim}={var:.4f}"
            for dim, var in sorted(
                a["dim_variances"].items(), key=lambda kv: -kv[1]
            )
            if var > SCORE_VAR_THRESHOLD
        ]
        table.append(
            [
                a["task_id"],
                "; ".join(offenders),
                f"{a['max_dim_variance']:.4f}",
            ]
        )
    return tabulate(table, headers=headers, tablefmt="github")


def _wide_conf_table(per_task: dict[Any, dict[str, Any]]) -> str:
    """Table 3 — wide confidence, sorted worst-first."""
    flagged = [a for a in per_task.values() if a["wide_confidence"]]
    if not flagged:
        return "_No tasks with confidence range above threshold._"
    flagged.sort(key=lambda a: -a["confidence_range"])
    headers = ["task_id", "min_confidence", "max_confidence", "conf_range"]
    table = [
        [
            a["task_id"],
            f"{min(a['confidences']):.4f}",
            f"{max(a['confidences']):.4f}",
            f"{a['confidence_range']:.4f}",
        ]
        for a in flagged
    ]
    return tabulate(table, headers=headers, tablefmt="github")


def _build_report(analysis: dict[str, Any], results_path: Path) -> str:
    """Assemble the full markdown report string."""
    summary = analysis["summary"]
    per_task = analysis["per_task"]

    lines: list[str] = ["# LLM-Judge Self-Consistency Report", ""]
    lines.append(f"Source: `{results_path}`")
    lines.append("")

    # --- Summary ---------------------------------------------------------
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- **Pass/fail flip rate (headline): {summary['flip_rate']:.1%}** "
        f"({summary['n_flipped']} / {summary['n_tasks']} tasks)"
    )
    lines.append(f"- Tasks: {summary['n_tasks']}")
    lines.append(f"- Repeats per task (mode): {summary['repeats_per_task']}")
    lines.append(
        f"- High score variance (any dim var > {SCORE_VAR_THRESHOLD}): "
        f"{summary['n_high_score_variance']}"
    )
    lines.append(
        f"- Wide confidence (range > {CONFIDENCE_RANGE_THRESHOLD}): "
        f"{summary['n_wide_confidence']}"
    )
    lines.append(
        f"- Tasks flagged by >=1 criterion: {summary['n_any_flag']}"
    )
    lines.append(
        f"- Mean process-score variance: "
        f"{summary['mean_process_score_variance']:.4f}"
    )
    lines.append(
        f"- Median process-score variance: "
        f"{summary['median_process_score_variance']:.4f}"
    )
    lines.append(
        f"- Mean confidence range: {summary['mean_confidence_range']:.4f}"
    )
    lines.append(
        f"- Mean max dimension variance: "
        f"{summary['mean_max_dim_variance']:.4f}"
    )
    lines.append("")

    # --- Table 1 ---------------------------------------------------------
    lines.append("## Table 1 — Pass/Fail flips (same answer got both)")
    lines.append("")
    lines.append(_flip_table(per_task))
    lines.append("")

    # --- Table 2 ---------------------------------------------------------
    lines.append(
        f"## Table 2 — High score variance (any dimension var > "
        f"{SCORE_VAR_THRESHOLD})"
    )
    lines.append("")
    lines.append(_high_var_table(per_task))
    lines.append("")

    # --- Table 3 ---------------------------------------------------------
    lines.append(
        f"## Table 3 — Wide confidence (range > {CONFIDENCE_RANGE_THRESHOLD})"
    )
    lines.append("")
    lines.append(_wide_conf_table(per_task))
    lines.append("")

    # --- Most-unstable note ---------------------------------------------
    all_three = sorted(
        a["task_id"]
        for a in per_task.values()
        if a["pass_fail_flip"]
        and a["high_score_variance"]
        and a["wide_confidence"]
    )
    lines.append("## Most unstable (flagged by ALL three criteria)")
    lines.append("")
    if all_three:
        lines.append(
            "Tasks: "
            + ", ".join(str(t) for t in all_three)
            + f" ({len(all_three)} total)"
        )
    else:
        lines.append("None — no task tripped all three criteria.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the LLM-judge self-consistency report."
    )
    parser.add_argument(
        "--results",
        required=True,
        help="Path to a consistency_results JSONL file.",
    )
    parser.add_argument(
        "--out",
        default="reports/consistency_report.md",
        help=(
            "Path for the markdown report "
            "(default: reports/consistency_report.md)."
        ),
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path

    rows = _load_rows(results_path)
    print(f"Loaded {len(rows)} consistency-result rows from {results_path}")

    analysis = analyze_consistency(rows)

    print("\nSummary dict:")
    print(json.dumps(analysis["summary"], indent=2))

    report = _build_report(analysis, results_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print(f"\nReport written to: {out_path}")


if __name__ == "__main__":
    main()
