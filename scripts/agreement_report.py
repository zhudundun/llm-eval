#!/usr/bin/env python
"""
Build the agreement report from rubric judge results (Step 12, §17).

Usage:
    python scripts/agreement_report.py \\
        --results data/results/batch/gpt-4o_dev/judge_results_no_reference.jsonl \\
        [--out reports/agreement_report.md] \\
        [--label no_reference]

Reads the JSONL produced by ``run_rubric_judge.py``, reshapes each row into the
``judged_run`` dict the agreement functions expect, then:

  * prints the agreement dict (accuracy/precision/recall/F1/kappa, confusion
    matrix, suspicious-pass / near-miss counts, top failure types) and the
    per-dimension Spearman correlations,
  * writes the markdown agreement report via ``generate_agreement_report``,
  * writes two side files in the same reports directory:
      - suspicious_passes.md  (classification == "suspicious_pass")
      - near_misses.md        (classification == "near_miss")
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The seven 1-5 score fields plus critical_errors form the judge_result sub-dict.
_SCORE_FIELDS: tuple[str, ...] = (
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    """Load judge-result JSONL rows into a list of dicts."""
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


def _to_judged_run(row: dict[str, Any]) -> dict[str, Any]:
    """Reshape a flat judge-result row into the agreement-function shape.

    The agreement functions need ``task_id``, ``builtin_pass``,
    ``computed_overall_pass``, ``classification`` and a nested ``judge_result``
    holding the 7 score fields + ``critical_errors``.
    """
    judge_result = {field: row.get(field) for field in _SCORE_FIELDS}
    judge_result["critical_errors"] = row.get("critical_errors") or []
    return {
        "task_id": row.get("task_id"),
        "builtin_pass": row.get("builtin_pass"),
        "computed_overall_pass": row.get("computed_overall_pass"),
        "judge_result": judge_result,
        "classification": row.get("classification"),
    }


def _format_critical_errors(errors: list[Any]) -> str:
    """Render a list of critical-error dicts as a compact markdown sub-list."""
    if not errors:
        return "  - (none)"
    lines: list[str] = []
    for err in errors:
        if isinstance(err, dict):
            etype = err.get("type", "unknown")
            severity = err.get("severity", "?")
            explanation = err.get("explanation", "")
            lines.append(f"  - **{etype}** ({severity}): {explanation}")
        else:
            lines.append(f"  - {err}")
    return "\n".join(lines)


def _write_detail_file(
    path: Path,
    title: str,
    rows: list[dict[str, Any]],
    label: str,
) -> int:
    """Write a detail markdown file for a subset of rows; return the count."""
    lines: list[str] = [f"# {title}", ""]
    lines.append(f"Source label: `{label}`")
    lines.append("")
    lines.append(f"Count: {len(rows)}")
    lines.append("")
    if not rows:
        lines.append("None.")
    else:
        for row in rows:
            task_id = row.get("task_id")
            lines.append(f"## Task {task_id}")
            lines.append("")
            explanation = row.get("judge_explanation", "")
            lines.append(f"**Judge explanation:** {explanation}")
            lines.append("")
            lines.append("**Critical errors:**")
            lines.append(_format_critical_errors(row.get("critical_errors") or []))
            lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the agreement report from rubric judge results."
    )
    parser.add_argument(
        "--results", required=True, help="Path to a judge_results JSONL file."
    )
    parser.add_argument(
        "--out",
        default="reports/agreement_report.md",
        help="Path for the markdown agreement report (default: reports/agreement_report.md).",
    )
    parser.add_argument(
        "--label",
        default="no_reference",
        help="Label describing this results set (used in the side files).",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file not found: {results_path}", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    reports_dir = out_path.parent

    from src.calibration.agreement import (
        compute_agreement,
        generate_agreement_report,
        spearman_scores,
    )

    # --- load + reshape --------------------------------------------------
    rows = _load_rows(results_path)
    print(f"Loaded {len(rows)} judge-result rows from {results_path}")
    judged_runs = [_to_judged_run(row) for row in rows]

    # --- agreement metrics ----------------------------------------------
    agreement = compute_agreement(judged_runs)
    spearman = spearman_scores(judged_runs)

    print("\nAgreement dict:")
    print(json.dumps(agreement, indent=2))

    print("\nSpearman (dimension -> [rho, p-value]):")
    if spearman:
        for dim, (rho, pvalue) in spearman.items():
            rho_s = "N/A" if rho is None else f"{rho:.4f}"
            p_s = "N/A" if pvalue is None else f"{pvalue:.4f}"
            print(f"  {dim:<28} rho={rho_s:<10} p={p_s}")
    else:
        print("  (too few labeled samples; need >= 3)")

    # --- main markdown report -------------------------------------------
    generate_agreement_report(judged_runs, out_path)

    # --- side detail files ----------------------------------------------
    suspicious_rows = [r for r in rows if r.get("classification") == "suspicious_pass"]
    near_miss_rows = [r for r in rows if r.get("classification") == "near_miss"]

    # Suffix side files by label so different judge modes don't overwrite each other.
    suffix = f"_{args.label}" if args.label else ""
    suspicious_path = reports_dir / f"suspicious_passes{suffix}.md"
    near_misses_path = reports_dir / f"near_misses{suffix}.md"

    n_susp = _write_detail_file(
        suspicious_path,
        "Suspicious Passes (built-in pass, judge fail)",
        suspicious_rows,
        args.label,
    )
    n_near = _write_detail_file(
        near_misses_path,
        "Near Misses (built-in fail, decent process quality)",
        near_miss_rows,
        args.label,
    )

    # --- where everything went ------------------------------------------
    print("\nFiles written:")
    print(f"  agreement report : {out_path}")
    print(f"  suspicious passes: {suspicious_path}  ({n_susp} rows)")
    print(f"  near misses      : {near_misses_path}  ({n_near} rows)")


if __name__ == "__main__":
    main()
