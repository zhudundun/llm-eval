"""
Agreement analyzer for the LLM judge (Step 12, §17 of the design doc).

This module compares the built-in (benchmark) evaluator against the LLM judge
over a list of judged-run dicts and produces:

* binary-agreement metrics (accuracy, precision, recall, F1, Cohen's kappa)
* a 2x2 confusion matrix laid out as in §17
* counts of "suspicious passes" and "near misses"
* the most common critical-error (failure) types
* per-dimension Spearman correlations between each 1-5 judge score and the
  built-in pass/fail label
* a markdown report matching the §17 layout

Each judged-run dict is expected to carry at least::

    {
      "task_id":               <hashable>,
      "builtin_pass":          bool | None,
      "computed_overall_pass": bool,
      "judge_result":          {<7 score fields> + "critical_errors": [...]},
      "classification":        str,
    }

Runs whose ``builtin_pass`` is ``None`` are excluded from binary-agreement
metrics (there is no ground-truth-ish label to compare against) but their
classifications and failure types are still counted.

Edge cases (n == 0, a single class present, fewer than 3 Spearman samples)
are handled without crashing: undefined metrics return ``None``.
"""
from __future__ import annotations

import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import ConstantInputWarning, spearmanr
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    precision_recall_fscore_support,
)

# The seven 1-5 judge score dimensions (§9 / §11).
_SCORE_DIMENSIONS: tuple[str, ...] = (
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
)


def _judge_result(run: dict) -> dict:
    """Return the judge_result sub-dict for a run (empty dict if absent)."""
    jr = run.get("judge_result")
    return jr if isinstance(jr, dict) else {}


def _collect_failure_types(judged_runs: list[dict]) -> Counter:
    """Count critical-error ``type`` values across all runs.

    Items missing a ``type`` (or that are not dicts) are recorded as
    ``"unknown"`` so nothing is silently dropped.
    """
    counter: Counter = Counter()
    for run in judged_runs:
        errors = _judge_result(run).get("critical_errors") or []
        for err in errors:
            if isinstance(err, dict):
                counter[err.get("type", "unknown")] += 1
            else:
                counter["unknown"] += 1
    return counter


def compute_agreement(judged_runs: list[dict]) -> dict:
    """Compare ``builtin_pass`` (ground-truth-ish) vs ``computed_overall_pass``.

    ``builtin_pass`` is treated as ``y_true`` and ``computed_overall_pass`` as
    ``y_pred`` (the judge "predicts" the built-in result).  Runs with a
    ``None`` ``builtin_pass`` are excluded from the binary metrics but still
    counted for classifications and failure types.

    Returns a dict with keys:
        ``n``                  : int — number of runs with a non-None builtin_pass
        ``builtin_pass_rate``  : float | None
        ``judge_pass_rate``    : float | None
        ``agreement``          : float | None — accuracy
        ``precision``          : float | None
        ``recall``             : float | None
        ``f1``                 : float | None
        ``cohen_kappa``        : float | None
        ``confusion_matrix``   : dict — {"tp","fp","fn","tn"} ints
        ``n_suspicious_passes``: int
        ``n_near_misses``      : int
        ``top_failure_types``  : list[tuple[str, int]] — most-common first
    """
    # Failure types and classification-derived counts use *all* runs.
    failure_counter = _collect_failure_types(judged_runs)
    classifications = Counter(
        run.get("classification") for run in judged_runs
    )
    n_suspicious_passes = int(classifications.get("suspicious_pass", 0))
    n_near_misses = int(classifications.get("near_miss", 0))

    # Binary metrics use only runs with a non-None builtin_pass.
    labeled = [
        run for run in judged_runs
        if run.get("builtin_pass") is not None
    ]
    n = len(labeled)

    y_true = [bool(run["builtin_pass"]) for run in labeled]
    y_pred = [bool(run.get("computed_overall_pass")) for run in labeled]

    # Confusion matrix counts (built explicitly so labels match §17 exactly,
    # regardless of which classes are present).
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if (not t) and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and (not p))
    tn = sum(1 for t, p in zip(y_true, y_pred) if (not t) and (not p))
    confusion_matrix = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    if n == 0:
        return {
            "n": 0,
            "builtin_pass_rate": None,
            "judge_pass_rate": None,
            "agreement": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "cohen_kappa": None,
            "confusion_matrix": confusion_matrix,
            "n_suspicious_passes": n_suspicious_passes,
            "n_near_misses": n_near_misses,
            "top_failure_types": failure_counter.most_common(),
        }

    builtin_pass_rate = float(sum(y_true)) / n
    judge_pass_rate = float(sum(y_pred)) / n

    accuracy = float(accuracy_score(y_true, y_pred))

    # Precision/recall/F1 for the positive (pass) class.  zero_division=0 keeps
    # this defined even when no positives are predicted/present.
    precision_arr, recall_arr, f1_arr, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[True],
        average="binary",
        pos_label=True,
        zero_division=0,
    )
    precision = float(precision_arr)
    recall = float(recall_arr)
    f1 = float(f1_arr)

    # Cohen's kappa is undefined when only one label is present across both
    # raters (sklearn returns nan); surface that as None.
    if len(set(y_true)) < 2 and len(set(y_pred)) < 2:
        cohen_kappa: float | None = None
    else:
        kappa_val = float(cohen_kappa_score(y_true, y_pred))
        cohen_kappa = None if kappa_val != kappa_val else kappa_val  # nan -> None

    return {
        "n": n,
        "builtin_pass_rate": builtin_pass_rate,
        "judge_pass_rate": judge_pass_rate,
        "agreement": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": cohen_kappa,
        "confusion_matrix": confusion_matrix,
        "n_suspicious_passes": n_suspicious_passes,
        "n_near_misses": n_near_misses,
        "top_failure_types": failure_counter.most_common(),
    }


def spearman_scores(judged_runs: list[dict]) -> dict:
    """Spearman correlation of each 1-5 judge dimension vs ``builtin_pass``.

    For runs with a non-None ``builtin_pass``, correlate each score dimension
    (extracted from ``judge_result``) against the built-in label encoded as
    0/1.  Returns ``{dimension: (rho, pvalue)}``.

    Returns an empty dict when there are fewer than 3 labeled samples.  A
    dimension whose correlation is undefined (e.g. a constant column, which
    yields nan) is reported as ``(None, None)``.
    """
    labeled = [
        run for run in judged_runs
        if run.get("builtin_pass") is not None
    ]
    if len(labeled) < 3:
        return {}

    builtin_binary = [1 if run["builtin_pass"] else 0 for run in labeled]

    out: dict[str, tuple[float | None, float | None]] = {}
    for dim in _SCORE_DIMENSIONS:
        scores: list[float | None] = [
            _judge_result(run).get(dim) for run in labeled
        ]
        # Require every run to carry this dimension; otherwise skip it.
        if any(s is None for s in scores):
            continue

        # A constant input column yields nan and a ConstantInputWarning; this
        # is an expected, handled case, so silence the warning here.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConstantInputWarning)
            result = spearmanr(scores, builtin_binary)
        rho = float(result.statistic)
        pvalue = float(result.pvalue)
        if rho != rho or pvalue != pvalue:  # nan (constant input) -> None
            out[dim] = (None, None)
        else:
            out[dim] = (rho, pvalue)

    return out


def _fmt(value: Any, *, pct: bool = False) -> str:
    """Format a metric for the markdown report; ``None`` becomes 'N/A'."""
    if value is None:
        return "N/A"
    if pct:
        return f"{value * 100:.1f}%"
    return f"{value:.4f}"


def generate_agreement_report(
    judged_runs: list[dict],
    out_path: Path | str,
) -> str:
    """Write a markdown agreement report and return the markdown string.

    The layout mirrors §17: total tasks, pass rates, agreement, the confusion
    matrix table, suspicious-pass and near-miss counts, top failure types, plus
    a Spearman-correlation section.  Parent directories of ``out_path`` are
    created if needed.
    """
    agreement = compute_agreement(judged_runs)
    spearman = spearman_scores(judged_runs)

    cm = agreement["confusion_matrix"]
    tp, fp, fn, tn = cm["tp"], cm["fp"], cm["fn"], cm["tn"]

    lines: list[str] = []
    lines.append("# Agreement Report: Built-in Evaluator vs LLM Judge")
    lines.append("")

    # --- Summary ----------------------------------------------------------
    n = agreement["n"]
    lines.append(f"Total tasks (with built-in label): {n}")
    lines.append(f"Total runs analyzed: {len(judged_runs)}")
    lines.append(
        f"Built-in pass rate: {_fmt(agreement['builtin_pass_rate'], pct=True)}"
    )
    lines.append(
        f"Judge pass rate: {_fmt(agreement['judge_pass_rate'], pct=True)}"
    )
    lines.append(f"Agreement: {_fmt(agreement['agreement'], pct=True)}")
    lines.append("")

    # --- Binary metrics ---------------------------------------------------
    lines.append("## Binary Agreement Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Accuracy | {_fmt(agreement['agreement'])} |")
    lines.append(f"| Precision | {_fmt(agreement['precision'])} |")
    lines.append(f"| Recall | {_fmt(agreement['recall'])} |")
    lines.append(f"| F1 | {_fmt(agreement['f1'])} |")
    lines.append(f"| Cohen's Kappa | {_fmt(agreement['cohen_kappa'])} |")
    lines.append("")

    # --- Confusion matrix -------------------------------------------------
    lines.append("## Confusion Matrix")
    lines.append("")
    lines.append("```")
    lines.append(
        f"                 {'Built-in Pass':<15} {'Built-in Fail':<15}"
    )
    lines.append(f"Judge Pass       {tp:<15} {fp:<15}")
    lines.append(f"Judge Fail       {fn:<15} {tn:<15}")
    lines.append("```")
    lines.append("")
    lines.append("| | Built-in Pass | Built-in Fail |")
    lines.append("|---|---|---|")
    lines.append(f"| Judge Pass | {tp} | {fp} |")
    lines.append(f"| Judge Fail | {fn} | {tn} |")
    lines.append("")

    # --- Disagreement buckets --------------------------------------------
    lines.append("## Disagreement Buckets")
    lines.append("")
    lines.append(f"Suspicious passes: {agreement['n_suspicious_passes']}")
    lines.append(f"Near misses: {agreement['n_near_misses']}")
    lines.append("")

    # --- Top failure types -----------------------------------------------
    lines.append("## Top Failure Types")
    lines.append("")
    top_failures = agreement["top_failure_types"]
    if top_failures:
        for ftype, count in top_failures:
            lines.append(f"{ftype}: {count}")
    else:
        lines.append("None")
    lines.append("")

    # --- Spearman ---------------------------------------------------------
    lines.append("## Spearman Correlation (1-5 score vs built-in pass)")
    lines.append("")
    if spearman:
        lines.append("| Dimension | rho | p-value |")
        lines.append("|---|---|---|")
        for dim, (rho, pvalue) in spearman.items():
            lines.append(f"| {dim} | {_fmt(rho)} | {_fmt(pvalue)} |")
    else:
        lines.append("Too few labeled samples (need >= 3) to compute Spearman.")
    lines.append("")

    markdown = "\n".join(lines)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8")

    return markdown
