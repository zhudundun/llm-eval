"""LLM-based judge using Claude claude-opus-4-7 (or a configurable model)."""
from __future__ import annotations

import re
from typing import Optional

from src.benchmark.task import AgentResponse, DABenchTask, JudgeScore
from src.judge.base import BaseJudge

_DEFAULT_MODEL = "claude-opus-4-7"

_PROMPT_TEMPLATE = """\
You are evaluating whether a data analysis agent correctly answered a question.

## Question
{question}

## Required Answer Format
{format}

## Expected Answer(s)
{expected_answers}

## Agent's Answer
{agent_answer}

## Instructions
Determine if the agent's answer is CORRECT or INCORRECT.
- For numerical answers: allow up to 1% relative error (or ±0.5 for values near zero)
- For categorical/string answers: semantically equivalent matches are correct
- If the agent gave multiple answers, all must be correct

Respond EXACTLY in this format:
JUDGMENT: CORRECT
REASONING: <one sentence>

or

JUDGMENT: INCORRECT
REASONING: <one sentence>\
"""

_JUDGMENT_RE = re.compile(r"JUDGMENT:\s*(CORRECT|INCORRECT)", re.IGNORECASE)
_REASONING_RE = re.compile(r"REASONING:\s*(.+)", re.IGNORECASE | re.DOTALL)


def _format_expected_answers(common_answers: list[list[str]] | None) -> str:
    if not common_answers:
        return "(no ground-truth answers provided)"
    lines = [f"  {pair[0]}: {pair[1]}" for pair in common_answers]
    return "\n".join(lines)


def _format_agent_answer(answer: dict[str, str]) -> str:
    if not answer:
        return "(no answer extracted from agent output)"
    lines = [f"  {k}: {v}" for k, v in answer.items()]
    return "\n".join(lines)


class ClaudeJudge(BaseJudge):
    """Judge that uses a Claude model to evaluate agent answers.

    The Anthropic client is lazily instantiated on first use so the module
    can be imported without an ``ANTHROPIC_API_KEY`` in the environment.

    Parameters
    ----------
    model:
        Anthropic model identifier to use for judging.
    """

    def __init__(self, model: str = _DEFAULT_MODEL) -> None:
        self.model = model
        self._client: Optional[object] = None

    @property
    def client(self):  # type: ignore[return]
        """Return a lazily-created ``anthropic.Anthropic`` client."""
        if self._client is None:
            import anthropic  # local import — keeps module importable without key
            self._client = anthropic.Anthropic()
        return self._client

    def _build_prompt(self, task: DABenchTask, response: AgentResponse) -> str:
        return _PROMPT_TEMPLATE.format(
            question=task.question,
            format=task.format,
            expected_answers=_format_expected_answers(task.common_answers),
            agent_answer=_format_agent_answer(response.answer),
        )

    def _parse_response(self, text: str) -> tuple[float, str]:
        """Parse the model output into (score, reasoning).

        Returns (0.0, ...) if the judgment cannot be parsed.
        """
        judgment_match = _JUDGMENT_RE.search(text)
        reasoning_match = _REASONING_RE.search(text)

        reasoning = reasoning_match.group(1).strip() if reasoning_match else text.strip()

        if judgment_match is None:
            return 0.0, f"Could not parse judgment from model output. Raw: {text[:300]}"

        judgment = judgment_match.group(1).upper()
        score = 1.0 if judgment == "CORRECT" else 0.0
        return score, reasoning

    def score(
        self,
        task: DABenchTask,
        response: AgentResponse,
        temperature: float = 0.0,
    ) -> JudgeScore:
        """Evaluate *response* against *task* using a Claude LLM call.

        Parameters
        ----------
        task:
            The DABench task being evaluated.
        response:
            The agent's response to evaluate.
        temperature:
            Sampling temperature forwarded to the API (default 0.0).
            Pass a non-zero value to probe judge consistency.
        """
        prompt = self._build_prompt(task, response)

        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text: str = message.content[0].text
        except Exception as exc:  # noqa: BLE001
            return JudgeScore(
                task_id=task.id,
                judge_model=self.model,
                score=0.0,
                reasoning=f"LLM call failed: {exc}",
                method="llm",
            )

        score, reasoning = self._parse_response(raw_text)
        return JudgeScore(
            task_id=task.id,
            judge_model=self.model,
            score=score,
            reasoning=reasoning,
            method="llm",
        )
