"""
InfiAgent-DABench built-in scorer.

Based directly on the evaluation logic in:
  infiagent-repo/examples/DA-Agent/eval_closed_form.py

The is_equal() function and the per-sub-question correctness logic are
reproduced faithfully.  A task is correct (1.0) only when ALL sub-questions
(all answer names in common_answers) are individually correct.
"""
from __future__ import annotations

from typing import Optional

from src.benchmark.task import DABenchTask, AgentResponse


def _is_equal(response: str, label: str) -> bool:
    """
    Replicate InfiAgent's is_equal() exactly:
      1. String equality first.
      2. Then try numeric comparison with tolerance < 1e-6.
    """
    if response == label:
        return True
    try:
        return abs(float(response) - float(label)) < 1e-6
    except Exception:
        return False


def infiagent_score(task: DABenchTask, response: AgentResponse) -> Optional[float]:
    """
    Score one AgentResponse against its ground-truth label using InfiAgent's
    closed-form evaluation logic.

    Returns
    -------
    1.0  — all sub-question answers are correct
    0.0  — at least one sub-question answer is wrong or missing
    None — task.common_answers is absent (no ground truth available)
    """
    if task.common_answers is None:
        return None

    # Build label dict: {answer_name: answer_value}
    label_answers: dict[str, str] = {ans[0]: ans[1] for ans in task.common_answers}

    # response.answer is already a {name: value} dict extracted from stdout
    extracted_answers: dict[str, str] = response.answer

    # Check every sub-question: ALL must be correct for score = 1.0
    for ans_name, label_value in label_answers.items():
        predicted_value = extracted_answers.get(ans_name, "")
        if not _is_equal(predicted_value, label_value):
            return 0.0

    return 1.0
