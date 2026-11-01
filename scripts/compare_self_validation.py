#!/usr/bin/env python
"""
Head-to-head comparison of a self-validating LLM judge against the plain
no-verifier baseline judge.

Usage:
    python scripts/compare_self_validation.py \\
        --new      <self_validate.jsonl> \\
        --baseline <plain.jsonl> \\
        --responses <agent_responses.jsonl> \\
        [--out reports/self_validation_compare.md]

Both judge files carry ``task_id`` and ``computed_overall_pass``; the built-in
answer key lives in the agent-responses file as ``builtin_eval_result`` (1.0 =
correct, 0.0 = wrong).  The "new" self-validating file additionally carries
``self_derived_answer``, ``self_validation_agrees``, ``validation_summary`` and
``n_tool_calls`` -- but this script degrades gracefully when those fields are
absent (e.g. when a plain run is used as a stand-in for smoke testing).

Metrics treat FAIL as the positive class because we are measuring failure
detection:

    TP = judge fail & builtin fail      FP = judge fail & builtin pass
    FN = judge pass & builtin fail      TN = judge pass & builtin pass
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score
from tabulate import tabulate

REPO_ROOT = Path(__file__).resolve().parent.parent

# Sentinel returned when a self-validation field is missing from the input.
NA = "n/a"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts, skipping blank lines."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def key_by_task_id(rows: list[dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    """Index rows by their ``task_id``."""
    return {row["task_id"]: row for row in rows}


def builtin_pass_map(responses: dict[Any, dict[str, Any]]) -> dict[Any, bool]:
    """Map task_id -> builtin pass (True) / fail (False) from ``builtin_eval_result``."""
    out: dict[Any, bool] = {}
    for task_id, row in responses.items():
        out[task_id] = float(row["builtin_eval_result"]) == 1.0
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
@dataclass
class JudgeMetrics:
    """Failure-detection metrics for one judge vs. the built-in key.

    FAIL is the positive class.
    """

    label: str
    n: int
    tp: int
    fp: int
    fn: int
    tn: int
    accuracy: float
    precision_failure: float
    recall_failure: float
    fnr: float
    fpr: float
    kappa: float
    judge_pass_rate: float


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def compute_metrics(
    label: str,
    task_ids: list[Any],
    judge_pass: dict[Any, bool],
    builtin_pass: dict[Any, bool],
) -> JudgeMetrics:
    """Build the FAIL-positive confusion matrix and derived metrics."""
    tp = fp = fn = tn = 0
    for task_id in task_ids:
        jp = judge_pass[task_id]  # judge says pass
        bp = builtin_pass[task_id]  # key says pass
        judge_fail = not jp
        builtin_fail = not bp
        if judge_fail and builtin_fail:
            tp += 1
        elif judge_fail and not builtin_fail:
            fp += 1
        elif (not judge_fail) and builtin_fail:
            fn += 1
        else:
            tn += 1

    n = len(task_ids)
    accuracy = _safe_div(tp + tn, n)
    precision_failure = _safe_div(tp, tp + fp)
    recall_failure = _safe_div(tp, tp + fn)
    fnr = _safe_div(fn, fn + tp)
    fpr = _safe_div(fp, fp + tn)
    judge_pass_rate = _safe_div(sum(1 for t in task_ids if judge_pass[t]), n)

    # Cohen's kappa on the *pass* labels (builtin-pass vs judge-pass), as
    # requested.  Degenerate when one rater has a single class -> report 0.0.
    builtin_labels = [builtin_pass[t] for t in task_ids]
    judge_labels = [judge_pass[t] for t in task_ids]
    try:
        kappa = float(cohen_kappa_score(builtin_labels, judge_labels))
    except Exception:
        kappa = float("nan")
    if kappa != kappa:  # NaN guard (single-class agreement)
        kappa = 0.0

    return JudgeMetrics(
        label=label,
        n=n,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        accuracy=accuracy,
        precision_failure=precision_failure,
        recall_failure=recall_failure,
        fnr=fnr,
        fpr=fpr,
        kappa=kappa,
        judge_pass_rate=judge_pass_rate,
    )


def metrics_side_by_side(base: JudgeMetrics, new: JudgeMetrics) -> list[list[Any]]:
    """Rows for a baseline-vs-new metrics table."""
    rows: list[list[Any]] = [
        ["n (shared tasks)", base.n, new.n],
        ["TP (judge fail & builtin fail)", base.tp, new.tp],
        ["FP (judge fail & builtin pass)", base.fp, new.fp],
        ["FN (judge pass & builtin fail)", base.fn, new.fn],
        ["TN (judge pass & builtin pass)", base.tn, new.tn],
        ["accuracy", f"{base.accuracy:.3f}", f"{new.accuracy:.3f}"],
        ["precision(failure)", f"{base.precision_failure:.3f}", f"{new.precision_failure:.3f}"],
        ["recall(failure)", f"{base.recall_failure:.3f}", f"{new.recall_failure:.3f}"],
        ["FNR", f"{base.fnr:.3f}", f"{new.fnr:.3f}"],
        ["FPR", f"{base.fpr:.3f}", f"{new.fpr:.3f}"],
        ["Cohen's kappa", f"{base.kappa:.3f}", f"{new.kappa:.3f}"],
        ["judge_pass_rate", f"{base.judge_pass_rate:.3f}", f"{new.judge_pass_rate:.3f}"],
    ]
    return rows


# --------------------------------------------------------------------------- #
# Per-task verdict diffs
# --------------------------------------------------------------------------- #
@dataclass
class VerdictDiff:
    task_id: Any
    builtin: str
    baseline: str
    new: str
    direction: str  # "improvement" | "regression"


def _verdict_str(passed: bool) -> str:
    return "pass" if passed else "fail"


def verdict_diffs(
    task_ids: list[Any],
    base_pass: dict[Any, bool],
    new_pass: dict[Any, bool],
    builtin_pass: dict[Any, bool],
) -> list[VerdictDiff]:
    """Tasks where baseline and new ``computed_overall_pass`` disagree.

    A change is an *improvement* when the new verdict matches the built-in key
    (and the baseline did not); a *regression* when the new verdict moves away
    from the key (baseline matched, new does not).
    """
    diffs: list[VerdictDiff] = []
    for task_id in task_ids:
        b = base_pass[task_id]
        nw = new_pass[task_id]
        if b == nw:
            continue
        key = builtin_pass[task_id]
        # Exactly one of {baseline, new} matches the key, since b != nw.
        direction = "improvement" if nw == key else "regression"
        diffs.append(
            VerdictDiff(
                task_id=task_id,
                builtin=_verdict_str(key),
                baseline=_verdict_str(b),
                new=_verdict_str(nw),
                direction=direction,
            )
        )
    diffs.sort(key=lambda d: (d.direction != "improvement", str(d.task_id)))
    return diffs


def verdict_diff_rows(diffs: list[VerdictDiff]) -> list[list[Any]]:
    return [
        [d.task_id, d.builtin, d.baseline, d.new, d.direction] for d in diffs
    ]


# --------------------------------------------------------------------------- #
# Self-validation specific analysis (new judge only)
# --------------------------------------------------------------------------- #
@dataclass
class SelfValidationAnalysis:
    available: bool
    agree: int
    disagree: int
    could_not_validate: int
    other: dict[str, int]
    disagree_total: int
    disagree_real_failures: int  # builtin == fail among disagree cases
    disagree_precision: float | None  # precision of the disagree signal
    builtin_failures: list[Any]
    failures_caught_by_new: int  # new judge said fail
    failures_disagree_flagged: int  # self_validation_agrees == disagree
    n_tool_calls_mean: float | None
    n_tool_calls_max: int | None


def analyze_self_validation(
    task_ids: list[Any],
    new_rows: dict[Any, dict[str, Any]],
    new_pass: dict[Any, bool],
    builtin_pass: dict[Any, bool],
) -> SelfValidationAnalysis:
    """Compute the self-validation-specific stats, degrading gracefully when
    the ``self_validation_*`` / ``n_tool_calls`` fields are absent."""
    has_agree = any("self_validation_agrees" in new_rows[t] for t in task_ids)
    has_tool_calls = any("n_tool_calls" in new_rows[t] for t in task_ids)

    builtin_failures = [t for t in task_ids if not builtin_pass[t]]

    if not has_agree and not has_tool_calls:
        return SelfValidationAnalysis(
            available=False,
            agree=0,
            disagree=0,
            could_not_validate=0,
            other={},
            disagree_total=0,
            disagree_real_failures=0,
            disagree_precision=None,
            builtin_failures=builtin_failures,
            failures_caught_by_new=sum(1 for t in builtin_failures if not new_pass[t]),
            failures_disagree_flagged=0,
            n_tool_calls_mean=None,
            n_tool_calls_max=None,
        )

    agree = disagree = could_not_validate = 0
    other: dict[str, int] = {}
    disagree_real_failures = 0
    for task_id in task_ids:
        val = new_rows[task_id].get("self_validation_agrees")
        if val is None:
            continue
        if val == "agree":
            agree += 1
        elif val == "disagree":
            disagree += 1
            if not builtin_pass[task_id]:
                disagree_real_failures += 1
        elif val == "could_not_validate":
            could_not_validate += 1
        else:
            other[str(val)] = other.get(str(val), 0) + 1

    disagree_total = disagree
    disagree_precision: float | None = (
        _safe_div(disagree_real_failures, disagree_total) if disagree_total else None
    )

    failures_caught_by_new = sum(1 for t in builtin_failures if not new_pass[t])
    failures_disagree_flagged = sum(
        1
        for t in builtin_failures
        if new_rows[t].get("self_validation_agrees") == "disagree"
    )

    tool_call_vals: list[int] = []
    for task_id in task_ids:
        v = new_rows[task_id].get("n_tool_calls")
        if isinstance(v, (int, float)):
            tool_call_vals.append(int(v))
    n_tool_calls_mean = (
        sum(tool_call_vals) / len(tool_call_vals) if tool_call_vals else None
    )
    n_tool_calls_max = max(tool_call_vals) if tool_call_vals else None

    return SelfValidationAnalysis(
        available=True,
        agree=agree,
        disagree=disagree,
        could_not_validate=could_not_validate,
        other=other,
        disagree_total=disagree_total,
        disagree_real_failures=disagree_real_failures,
        disagree_precision=disagree_precision,
        builtin_failures=builtin_failures,
        failures_caught_by_new=failures_caught_by_new,
        failures_disagree_flagged=failures_disagree_flagged,
        n_tool_calls_mean=n_tool_calls_mean,
        n_tool_calls_max=n_tool_calls_max,
    )


def _fmt(value: Any) -> str:
    if value is None:
        return NA
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def self_validation_rows(sv: SelfValidationAnalysis) -> list[list[Any]]:
    """Rows summarising the self-validation analysis."""
    if not sv.available:
        rows: list[list[Any]] = [
            ["self_validation_agrees == agree", NA],
            ["self_validation_agrees == disagree", NA],
            ["self_validation_agrees == could_not_validate", NA],
            ["disagree cases that are real failures (builtin=fail)", NA],
            ["precision of disagree signal", NA],
            [f"builtin failures caught by new (of {len(sv.builtin_failures)})", sv.failures_caught_by_new],
            ["builtin failures with disagree flag", NA],
            ["mean n_tool_calls", NA],
            ["max n_tool_calls", NA],
        ]
        return rows

    other_str = (
        ", ".join(f"{k}={v}" for k, v in sorted(sv.other.items()))
        if sv.other
        else "-"
    )
    rows = [
        ["self_validation_agrees == agree", sv.agree],
        ["self_validation_agrees == disagree", sv.disagree],
        ["self_validation_agrees == could_not_validate", sv.could_not_validate],
        ["self_validation_agrees == other", other_str],
        [
            "disagree cases that are real failures (builtin=fail)",
            f"{sv.disagree_real_failures} / {sv.disagree_total}",
        ],
        ["precision of disagree signal", _fmt(sv.disagree_precision)],
        [
            f"builtin failures caught by new (of {len(sv.builtin_failures)})",
            sv.failures_caught_by_new,
        ],
        ["builtin failures with disagree flag", sv.failures_disagree_flagged],
        ["mean n_tool_calls", _fmt(sv.n_tool_calls_mean)],
        ["max n_tool_calls", _fmt(sv.n_tool_calls_max)],
    ]
    return rows


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def build_report(
    *,
    base_metrics: JudgeMetrics,
    new_metrics: JudgeMetrics,
    diffs: list[VerdictDiff],
    sv: SelfValidationAnalysis,
    new_path: Path,
    baseline_path: Path,
    responses_path: Path,
) -> str:
    metric_tbl = tabulate(
        metrics_side_by_side(base_metrics, new_metrics),
        headers=["metric", "baseline", "new (self-validate)"],
        tablefmt="github",
    )

    diff_rows = verdict_diff_rows(diffs)
    if diff_rows:
        diff_tbl = tabulate(
            diff_rows,
            headers=["task_id", "builtin", "baseline", "new", "change"],
            tablefmt="github",
        )
    else:
        diff_tbl = "_No verdict differences between baseline and new._"

    sv_tbl = tabulate(
        self_validation_rows(sv),
        headers=["self-validation metric", "value"],
        tablefmt="github",
    )

    n_improve = sum(1 for d in diffs if d.direction == "improvement")
    n_regress = sum(1 for d in diffs if d.direction == "regression")
    sv_note = (
        ""
        if sv.available
        else (
            "\n> The new file lacks the `self_validation_*` / `n_tool_calls` "
            "fields, so self-validation metrics are reported as `n/a`.\n"
        )
    )

    md = f"""# Self-Validating Judge vs. Plain Baseline

**Positive class = FAIL** (we measure failure detection vs. the built-in key).

- new (self-validate): `{new_path}`
- baseline (plain):    `{baseline_path}`
- responses (key):     `{responses_path}`
- shared task_ids:     {base_metrics.n}

## 1. Failure-detection metrics (side by side)

{metric_tbl}

## 2. Per-task verdict diffs (baseline vs new `computed_overall_pass`)

{n_improve} improvement(s), {n_regress} regression(s).

{diff_tbl}

## 3. Self-validation analysis (new judge only)
{sv_note}
{sv_tbl}
"""
    return md


# --------------------------------------------------------------------------- #
# Stdout summary
# --------------------------------------------------------------------------- #
def print_summary(
    base_metrics: JudgeMetrics,
    new_metrics: JudgeMetrics,
    diffs: list[VerdictDiff],
    sv: SelfValidationAnalysis,
) -> None:
    print("=" * 70)
    print("FAILURE-DETECTION METRICS (FAIL = positive class)")
    print("=" * 70)
    print(
        tabulate(
            metrics_side_by_side(base_metrics, new_metrics),
            headers=["metric", "baseline", "new (self-validate)"],
            tablefmt="github",
        )
    )

    n_improve = sum(1 for d in diffs if d.direction == "improvement")
    n_regress = sum(1 for d in diffs if d.direction == "regression")
    print()
    print("=" * 70)
    print(
        f"PER-TASK VERDICT DIFFS: {len(diffs)} total "
        f"({n_improve} improvement, {n_regress} regression)"
    )
    print("=" * 70)
    if diffs:
        print(
            tabulate(
                verdict_diff_rows(diffs),
                headers=["task_id", "builtin", "baseline", "new", "change"],
                tablefmt="github",
            )
        )
    else:
        print("(none)")

    print()
    print("=" * 70)
    print("SELF-VALIDATION ANALYSIS (new judge only)")
    print("=" * 70)
    if not sv.available:
        print("Self-validation fields not present -> reported as n/a.")
    print(
        tabulate(
            self_validation_rows(sv),
            headers=["self-validation metric", "value"],
            tablefmt="github",
        )
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a self-validating LLM judge against the plain no-verifier "
            "baseline, scored against the built-in answer key."
        )
    )
    parser.add_argument(
        "--new", required=True, help="self-validating judge results JSONL"
    )
    parser.add_argument(
        "--baseline", required=True, help="plain baseline judge results JSONL"
    )
    parser.add_argument(
        "--responses",
        required=True,
        help="agent_responses.jsonl containing builtin_eval_result",
    )
    parser.add_argument(
        "--out",
        default="reports/self_validation_compare.md",
        help="markdown report path (resolved vs repo root)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    new_rows = key_by_task_id(load_jsonl(Path(args.new)))
    base_rows = key_by_task_id(load_jsonl(Path(args.baseline)))
    responses = key_by_task_id(load_jsonl(Path(args.responses)))

    # Step 1: restrict to task_ids present in BOTH judge files.
    shared = sorted(
        set(new_rows) & set(base_rows),
        key=lambda x: (isinstance(x, str), x),
    )
    if not shared:
        print("No overlapping task_ids between the two judge files.", file=sys.stderr)
        return 1

    missing_key = [t for t in shared if t not in responses]
    if missing_key:
        print(
            f"WARNING: {len(missing_key)} shared task_id(s) missing from "
            f"responses; dropping them: {missing_key}",
            file=sys.stderr,
        )
        shared = [t for t in shared if t in responses]

    builtin_pass = builtin_pass_map(responses)
    base_pass = {t: bool(base_rows[t]["computed_overall_pass"]) for t in shared}
    new_pass = {t: bool(new_rows[t]["computed_overall_pass"]) for t in shared}

    # Step 2: metrics for each judge vs. built-in key.
    base_metrics = compute_metrics("baseline", shared, base_pass, builtin_pass)
    new_metrics = compute_metrics("new", shared, new_pass, builtin_pass)

    # Step 3: per-task verdict diffs.
    diffs = verdict_diffs(shared, base_pass, new_pass, builtin_pass)

    # Step 4: self-validation specific analysis.
    sv = analyze_self_validation(shared, new_rows, new_pass, builtin_pass)

    # Step 5: report + stdout.
    report = build_report(
        base_metrics=base_metrics,
        new_metrics=new_metrics,
        diffs=diffs,
        sv=sv,
        new_path=Path(args.new),
        baseline_path=Path(args.baseline),
        responses_path=Path(args.responses),
    )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")

    print_summary(base_metrics, new_metrics, diffs, sv)
    print()
    print(f"Markdown report written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
