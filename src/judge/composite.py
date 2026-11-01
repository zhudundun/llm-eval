"""Composite judge: tries programmatic scoring first, falls back to LLM."""
from __future__ import annotations

from src.benchmark.task import AgentResponse, DABenchTask, JudgeScore
from src.judge.base import BaseJudge
from src.judge.claude_judge import ClaudeJudge
from src.judge.programmatic import ProgrammaticJudge


class CompositeJudge(BaseJudge):
    """Judge that runs programmatic scoring first and falls back to an LLM judge.

    Decision logic
    --------------
    1. If the task has ground-truth answers **and** the agent produced a
       parseable answer, run the programmatic judge and return its score
       immediately — the result is unambiguous.
    2. Otherwise (no ground truth, or no parsed agent answer) fall back to
       the LLM judge, which can reason about partial or missing information.

    Parameters
    ----------
    llm_judge:
        A :class:`ClaudeJudge` instance used for fallback evaluation.
    programmatic_judge:
        An optional :class:`ProgrammaticJudge` instance.  A default instance
        is created automatically when *None* is passed.
    """

    def __init__(
        self,
        llm_judge: ClaudeJudge,
        programmatic_judge: ProgrammaticJudge | None = None,
    ) -> None:
        self.llm_judge = llm_judge
        self.programmatic_judge: ProgrammaticJudge = (
            programmatic_judge if programmatic_judge is not None else ProgrammaticJudge()
        )

    def score(self, task: DABenchTask, response: AgentResponse) -> JudgeScore:
        """Score *response* against *task*, preferring programmatic evaluation.

        Returns
        -------
        JudgeScore
            A score with ``method`` set to ``"programmatic"`` or ``"llm"``
            depending on which judge was ultimately used.
        """
        # Condition for an unambiguous programmatic result:
        #   - ground truth is available (non-empty common_answers)
        #   - agent produced at least one parseable answer
        has_ground_truth = bool(task.common_answers)
        has_parsed_answer = bool(response.answer)

        if has_ground_truth and has_parsed_answer:
            return self.programmatic_judge.score(task, response)

        # Fall back to the LLM judge
        return self.llm_judge.score(task, response)
