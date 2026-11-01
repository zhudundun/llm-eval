"""Abstract base class for all judges."""
from __future__ import annotations

from abc import ABC, abstractmethod

from src.benchmark.task import AgentResponse, DABenchTask, JudgeScore


class BaseJudge(ABC):
    """All judges must implement ``score``."""

    @abstractmethod
    def score(self, task: DABenchTask, response: AgentResponse) -> JudgeScore:
        """Evaluate a single agent response against the task ground truth.

        Parameters
        ----------
        task:
            The DABench task, including ground-truth ``common_answers``.
        response:
            The agent's response, including the parsed ``answer`` dict.

        Returns
        -------
        JudgeScore
            A dataclass holding task_id, judge_model, score (0.0 or 1.0),
            reasoning, and method ("programmatic" or "llm").
        """
        ...
