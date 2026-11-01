"""
Deterministic scoring layer for the LLM judge (Step 10, §15 of the design doc).

The LLM's own ``overall_pass`` is stored but not trusted blindly.  This module
recomputes a pass/fail decision and a weighted process-quality score from the
raw judge score fields, then classifies each run against the built-in
evaluator result.

All functions operate on a plain ``dict`` (``r``) holding the judge score
fields — they deliberately do *not* depend on any dataclass so they can be
applied directly to parsed JSON judge output.

Expected keys in ``r``:
    task_success_score, code_correctness_score, data_handling_score,
    statistical_validity_score, reproducibility_score,
    final_answer_quality_score, reasoning_quality_score, critical_errors

``critical_errors`` is a list whose items may be dicts carrying a
``severity`` key (one of "low" | "medium" | "high").
"""
from __future__ import annotations

from typing import Any


def compute_process_quality_score(r: dict) -> float:
    """Return the weighted process-quality score for a judge result.

    Weighted sum per §15::

        0.30 * task_success_score
      + 0.25 * code_correctness_score
      + 0.20 * data_handling_score
      + 0.10 * statistical_validity_score
      + 0.10 * reproducibility_score
      + 0.05 * final_answer_quality_score

    The seven score fields are on a 1-5 scale, so the result lies in [1, 5].
    """
    return (
        0.30 * r["task_success_score"]
        + 0.25 * r["code_correctness_score"]
        + 0.20 * r["data_handling_score"]
        + 0.10 * r["statistical_validity_score"]
        + 0.10 * r["reproducibility_score"]
        + 0.05 * r["final_answer_quality_score"]
    )


def compute_overall_pass(r: dict) -> bool:
    """Return the deterministic pass/fail decision for a judge result.

    The run fails outright if *any* critical error has ``severity == "high"``.
    Otherwise it passes only when the minimum of the four core dimensions
    (task success, code correctness, data handling, final-answer quality) is
    at least 4.
    """
    critical_errors: list[Any] = r["critical_errors"] or []
    high_severity_errors = [
        e for e in critical_errors
        if isinstance(e, dict) and e.get("severity") == "high"
    ]

    if high_severity_errors:
        return False

    required_scores = [
        r["task_success_score"],
        r["code_correctness_score"],
        r["data_handling_score"],
        r["final_answer_quality_score"],
    ]

    return min(required_scores) >= 4


def classify_run(
    builtin_pass: bool | None,
    judge_pass: bool,
    process_score: float,
) -> str:
    """Bucket a run by comparing the built-in evaluator vs the judge.

    Classification rules (per §15, evaluated in order)::

        builtin True  & judge_pass & process >= 4.0  -> "strong_pass"
        builtin True  & not judge_pass               -> "suspicious_pass"
        builtin False & process >= 3.5               -> "near_miss"
        builtin False & process <  3.5               -> "clear_fail"
        builtin None  & judge_pass                   -> "judge_pass_no_reference"
        else                                         -> "judge_fail_no_reference"
    """
    if builtin_pass is True and judge_pass and process_score >= 4.0:
        return "strong_pass"
    if builtin_pass is True and not judge_pass:
        return "suspicious_pass"
    if builtin_pass is False and process_score >= 3.5:
        return "near_miss"
    if builtin_pass is False and process_score < 3.5:
        return "clear_fail"
    if builtin_pass is None and judge_pass:
        return "judge_pass_no_reference"
    return "judge_fail_no_reference"


def detect_verifier_judge_conflict(
    verifier_result: dict | None, judge_pass: bool
) -> str | None:
    """Surface tension between the verifier signal and the judge verdict.

    Returns one of:
      "unconfirmed_pass"     : verifier DISAGREES (agrees_with_agent is False) but judge PASSED  (top review priority — the old blind spot)
      "nonreproducible_pass" : agent_code_reproduces is False but judge PASSED                    (hardcoding suspect)
      "confirmed_fail"       : verifier AGREES (True) but judge FAILED                            (likely a process failure or correlated error)
      None                   : no verifier result, or no conflict
    Precedence: unconfirmed_pass > nonreproducible_pass > confirmed_fail.
    """
    if verifier_result is None:
        return None

    agrees = verifier_result.get("agrees_with_agent")
    reproduces = verifier_result.get("agent_code_reproduces")

    if judge_pass and agrees is False:
        return "unconfirmed_pass"
    if judge_pass and reproduces is False:
        return "nonreproducible_pass"
    if not judge_pass and agrees is True:
        return "confirmed_fail"
    return None
