"""LLM-judge self-consistency analysis (spec §19.1).

Aggregates N repeated judgments per task on fixed inputs and flags instability:

* **pass/fail flip** — ``computed_overall_pass`` is not unanimous across a
  task's repeats (the same answer received both pass and fail).
* **high score variance** — any of the 7 dimension scores has sample variance
  (ddof=1) > 1.0 across repeats.
* **wide confidence** — ``max(confidence) - min(confidence) > 0.2`` across
  repeats.

All functions are pure and return JSON-serializable dicts.
"""
from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

DIMENSIONS: list[str] = [
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
]
SCORE_VAR_THRESHOLD: float = 1.0
CONFIDENCE_RANGE_THRESHOLD: float = 0.2


def _variance(values: list[float]) -> float:
    """Sample variance (ddof=1), guarding n<2 -> 0.0."""
    if len(values) < 2:
        return 0.0
    return float(statistics.variance(values))


def analyze_task(rows_for_one_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate the repeated judgments for ONE task.

    Parameters
    ----------
    rows_for_one_task:
        All rows sharing a single ``task_id`` (one per ``repeat_idx``).

    Returns
    -------
    dict with keys:
        ``task_id``, ``n_repeats``,
        ``verdicts`` (list[bool]), ``pass_fail_flip`` (bool),
        ``pass_rate`` (float), ``majority_verdict`` (bool),
        ``dim_variances`` (dict name->float), ``max_dim_variance`` (float),
        ``high_score_variance`` (bool), ``process_score_variance`` (float),
        ``confidences`` (list[float]), ``confidence_range`` (float),
        ``wide_confidence`` (bool), ``mean_process_score`` (float).
    """
    if not rows_for_one_task:
        raise ValueError("analyze_task requires at least one row")

    # Order repeats deterministically by repeat_idx for stable verdict strings.
    rows = sorted(rows_for_one_task, key=lambda r: r.get("repeat_idx", 0))
    task_id = rows[0]["task_id"]
    n_repeats = len(rows)

    verdicts: list[bool] = [bool(r["computed_overall_pass"]) for r in rows]
    n_pass = sum(1 for v in verdicts if v)
    pass_rate = n_pass / n_repeats if n_repeats else 0.0
    # Unanimous => no flip. Tie/majority handled below for majority_verdict.
    pass_fail_flip = len(set(verdicts)) > 1
    majority_verdict = n_pass * 2 >= n_repeats  # ties resolve to pass

    dim_variances: dict[str, float] = {}
    for dim in DIMENSIONS:
        dim_variances[dim] = _variance([float(r[dim]) for r in rows])
    max_dim_variance = max(dim_variances.values()) if dim_variances else 0.0
    high_score_variance = any(
        v > SCORE_VAR_THRESHOLD for v in dim_variances.values()
    )

    process_scores = [float(r["computed_process_quality_score"]) for r in rows]
    process_score_variance = _variance(process_scores)
    mean_process_score = (
        float(statistics.mean(process_scores)) if process_scores else 0.0
    )

    confidences = [float(r["confidence"]) for r in rows]
    confidence_range = (
        max(confidences) - min(confidences) if confidences else 0.0
    )
    # Range is only meaningful with >=2 repeats.
    wide_confidence = (
        n_repeats >= 2 and confidence_range > CONFIDENCE_RANGE_THRESHOLD
    )

    return {
        "task_id": task_id,
        "n_repeats": n_repeats,
        "verdicts": verdicts,
        "pass_fail_flip": pass_fail_flip,
        "pass_rate": pass_rate,
        "majority_verdict": majority_verdict,
        "dim_variances": dim_variances,
        "max_dim_variance": max_dim_variance,
        "high_score_variance": high_score_variance,
        "process_score_variance": process_score_variance,
        "confidences": confidences,
        "confidence_range": confidence_range,
        "wide_confidence": wide_confidence,
        "mean_process_score": mean_process_score,
    }


def analyze_consistency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Group all rows by ``task_id`` and aggregate each task.

    Returns
    -------
    dict with keys:
        ``per_task`` : dict[task_id, analyze_task(...) result]
        ``summary``  : dict with keys
            ``n_tasks``, ``repeats_per_task`` (mode), ``n_flipped``,
            ``flip_rate``, ``n_high_score_variance``, ``n_wide_confidence``,
            ``mean_process_score_variance``, ``median_process_score_variance``,
            ``mean_confidence_range``, ``mean_max_dim_variance``,
            ``n_any_flag`` (tasks flagged by >=1 criterion).
    """
    grouped: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["task_id"], []).append(row)

    per_task: dict[Any, dict[str, Any]] = {}
    for task_id, task_rows in grouped.items():
        per_task[task_id] = analyze_task(task_rows)

    analyses = list(per_task.values())
    n_tasks = len(analyses)

    repeat_counts = [a["n_repeats"] for a in analyses]
    repeats_per_task = (
        int(Counter(repeat_counts).most_common(1)[0][0]) if repeat_counts else 0
    )

    n_flipped = sum(1 for a in analyses if a["pass_fail_flip"])
    n_high_score_variance = sum(1 for a in analyses if a["high_score_variance"])
    n_wide_confidence = sum(1 for a in analyses if a["wide_confidence"])
    n_any_flag = sum(
        1
        for a in analyses
        if a["pass_fail_flip"] or a["high_score_variance"] or a["wide_confidence"]
    )

    flip_rate = n_flipped / n_tasks if n_tasks else 0.0

    process_variances = [a["process_score_variance"] for a in analyses]
    confidence_ranges = [a["confidence_range"] for a in analyses]
    max_dim_variances = [a["max_dim_variance"] for a in analyses]

    summary = {
        "n_tasks": n_tasks,
        "repeats_per_task": repeats_per_task,
        "n_flipped": n_flipped,
        "flip_rate": flip_rate,
        "n_high_score_variance": n_high_score_variance,
        "n_wide_confidence": n_wide_confidence,
        "mean_process_score_variance": (
            float(statistics.mean(process_variances)) if process_variances else 0.0
        ),
        "median_process_score_variance": (
            float(statistics.median(process_variances)) if process_variances else 0.0
        ),
        "mean_confidence_range": (
            float(statistics.mean(confidence_ranges)) if confidence_ranges else 0.0
        ),
        "mean_max_dim_variance": (
            float(statistics.mean(max_dim_variances)) if max_dim_variances else 0.0
        ),
        "n_any_flag": n_any_flag,
    }

    return {"per_task": per_task, "summary": summary}
