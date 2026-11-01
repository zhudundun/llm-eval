"""Programmatic judge: purely algorithmic, no LLM calls."""
from __future__ import annotations

from src.benchmark.task import AgentResponse, DABenchTask, JudgeScore
from src.judge.base import BaseJudge

_JUDGE_MODEL = "programmatic"


def _numeric_match(predicted: str, expected: str) -> bool:
    """Return True if both strings parse as floats and are within tolerance."""
    try:
        p = float(predicted.strip())
        e = float(expected.strip())
    except ValueError:
        return False

    abs_diff = abs(p - e)
    # Absolute tolerance for near-zero expected values
    if abs(e) < 1e-9:
        return abs_diff <= 0.5
    # 1% relative tolerance; also accept absolute ≤ 0.5 as safety net
    return abs_diff / abs(e) <= 0.01 or abs_diff <= 0.5


def _string_match(predicted: str, expected: str) -> bool:
    """Case-insensitive, whitespace-stripped string comparison."""
    return predicted.strip().lower() == expected.strip().lower()


def _values_match(predicted: str, expected: str) -> bool:
    """Try numeric match first, fall back to string match."""
    return _numeric_match(predicted, expected) or _string_match(predicted, expected)


def _find_answer_value(response_answer: dict[str, str], name: str) -> str | None:
    """Look up *name* in response_answer with case-insensitive key matching."""
    # Exact match first
    if name in response_answer:
        return response_answer[name]
    # Case-insensitive fallback
    name_lower = name.lower()
    for key, value in response_answer.items():
        if key.lower() == name_lower:
            return value
    return None


class ProgrammaticJudge(BaseJudge):
    """Judge that compares answers algorithmically without any LLM calls.

    Scoring strategy
    ----------------
    For each ``(name, expected_value)`` pair in ``task.common_answers``:
      1. Locate the matching key in ``response.answer`` (case-insensitive).
      2. Attempt numeric comparison with 1% relative tolerance (or ±0.5 for
         near-zero values).
      3. Fall back to case-insensitive string comparison.

    A score of 1.0 is returned only when **all** required answers match.
    """

    def score(self, task: DABenchTask, response: AgentResponse) -> JudgeScore:
        # --- guard: missing ground truth ---
        if not task.common_answers:
            return JudgeScore(
                task_id=task.id,
                judge_model=_JUDGE_MODEL,
                score=0.0,
                reasoning=(
                    "No ground-truth answers available for this task; "
                    "cannot evaluate programmatically."
                ),
                method="programmatic",
            )

        # --- guard: no answer extracted from agent output ---
        if not response.answer:
            return JudgeScore(
                task_id=task.id,
                judge_model=_JUDGE_MODEL,
                score=0.0,
                reasoning=(
                    "Agent produced no parseable @name[value] answer tokens in stdout."
                ),
                method="programmatic",
            )

        # --- compare each required answer ---
        mismatches: list[str] = []
        matches: list[str] = []

        for pair in task.common_answers:
            name: str = pair[0]
            expected: str = pair[1]

            predicted = _find_answer_value(response.answer, name)
            if predicted is None:
                mismatches.append(
                    f"'{name}' not found in agent answer (keys: "
                    f"{list(response.answer.keys())})"
                )
                continue

            if _values_match(predicted, expected):
                matches.append(f"'{name}': predicted={predicted!r} ≈ expected={expected!r}")
            else:
                mismatches.append(
                    f"'{name}': predicted={predicted!r} != expected={expected!r}"
                )

        if mismatches:
            reasoning = "Incorrect. Mismatches: " + "; ".join(mismatches)
            if matches:
                reasoning += ". Correct: " + "; ".join(matches)
            return JudgeScore(
                task_id=task.id,
                judge_model=_JUDGE_MODEL,
                score=0.0,
                reasoning=reasoning,
                method="programmatic",
            )

        reasoning = "All answers correct. " + "; ".join(matches)
        return JudgeScore(
            task_id=task.id,
            judge_model=_JUDGE_MODEL,
            score=1.0,
            reasoning=reasoning,
            method="programmatic",
        )
