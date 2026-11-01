"""
JSON parser/validator for LLM-judge output (Step 9, §14 of the design doc).

Never trust raw LLM output.  ``validate_judge_output`` enforces the judge_result
schema: the 11 required keys, integer 1-5 scores, a boolean ``overall_pass``, a
``confidence`` in [0, 1], and a well-formed ``critical_errors`` list whose entries
use only the controlled ``FAILURE_TYPES`` vocabulary and the {low, medium, high}
severity scale.

Any violation raises ``ValueError`` with a clear, specific message.
"""
from __future__ import annotations

from typing import Any

# Controlled failure taxonomy (Step 5, §10).  Exported as a module constant so the
# judge prompt builder, validator, and downstream scoring all share one source.
FAILURE_TYPES: list[str] = [
    "wrong_answer",
    "wrong_filter",
    "wrong_aggregation",
    "wrong_join",
    "wrong_groupby",
    "wrong_sorting",
    "wrong_column",
    "wrong_unit",
    "rounding_error",
    "missing_value_error",
    "data_leakage",
    "code_execution_error",
    "timeout",
    "hallucinated_column",
    "unsupported_claim",
    "incomplete_answer",
    "format_error",
]

_SEVERITIES: frozenset[str] = frozenset({"low", "medium", "high"})

_REQUIRED_FIELDS: list[str] = [
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
    "critical_errors",
    "overall_pass",
    "confidence",
    "judge_explanation",
]

_SCORE_FIELDS: list[str] = [
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
]


def validate_judge_output(obj: dict) -> dict:
    """Validate a judge_result dict in place, returning it on success.

    Raises
    ------
    ValueError
        On a missing key, an out-of-range or wrong-typed score, a non-boolean
        ``overall_pass``, an out-of-range ``confidence``, or a malformed
        ``critical_errors`` entry.
    """
    if not isinstance(obj, dict):
        raise ValueError(f"judge output must be a dict, got {type(obj).__name__}")

    # 1. all required keys present
    for key in _REQUIRED_FIELDS:
        if key not in obj:
            raise ValueError(f"Missing required field: {key}")

    # 2. seven score fields are ints in 1-5.  Reject bool explicitly: in Python
    #    ``isinstance(True, int)`` is True, but a boolean is not a valid score.
    for field in _SCORE_FIELDS:
        value = obj[field]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{field} must be int, got {type(value).__name__}: {value!r}"
            )
        if value < 1 or value > 5:
            raise ValueError(f"{field} must be between 1 and 5, got {value}")

    # 3. overall_pass is a real boolean
    if not isinstance(obj["overall_pass"], bool):
        raise ValueError(
            f"overall_pass must be boolean, got {type(obj['overall_pass']).__name__}"
        )

    # 4. confidence is a number in [0, 1] (bool is not a number here)
    confidence = obj["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError(
            f"confidence must be a number, got {type(confidence).__name__}"
        )
    if not 0 <= confidence <= 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence}")

    # 5. judge_explanation is a string
    if not isinstance(obj["judge_explanation"], str):
        raise ValueError(
            "judge_explanation must be a string, got "
            f"{type(obj['judge_explanation']).__name__}"
        )

    # 6. critical_errors is a list of well-formed {type, severity, explanation}
    critical_errors = obj["critical_errors"]
    if not isinstance(critical_errors, list):
        raise ValueError(
            f"critical_errors must be a list, got {type(critical_errors).__name__}"
        )

    for i, error in enumerate(critical_errors):
        _validate_critical_error(error, i)

    return obj


def _validate_critical_error(error: Any, index: int) -> None:
    """Validate a single ``critical_errors`` entry; raise ValueError if malformed."""
    if not isinstance(error, dict):
        raise ValueError(
            f"critical_errors[{index}] must be a dict, got {type(error).__name__}"
        )

    for key in ("type", "severity", "explanation"):
        if key not in error:
            raise ValueError(f"critical_errors[{index}] missing '{key}'")

    err_type = error["type"]
    if err_type not in FAILURE_TYPES:
        raise ValueError(
            f"critical_errors[{index}].type {err_type!r} is not a valid failure type "
            f"(must be one of FAILURE_TYPES)"
        )

    severity = error["severity"]
    if severity not in _SEVERITIES:
        raise ValueError(
            f"critical_errors[{index}].severity {severity!r} must be one of "
            "low, medium, high"
        )

    if not isinstance(error["explanation"], str):
        raise ValueError(
            f"critical_errors[{index}].explanation must be a string, got "
            f"{type(error['explanation']).__name__}"
        )
