"""Judge consistency measurement via repeated scoring."""
from __future__ import annotations

from src.benchmark.task import AgentResponse, DABenchTask
from src.judge.base import BaseJudge


def judge_consistency(
    judge: BaseJudge,
    tasks: list[DABenchTask],
    responses: list[AgentResponse],
    n_repeat: int = 3,
    temperature: float = 0.5,
) -> dict:
    """Measure how consistently a judge scores the same (task, response) pairs.

    The judge is called ``n_repeat`` times for each task.  Agreement is
    measured as the fraction of (task, run) pairs where every run agrees with
    the majority vote.

    The ``temperature`` kwarg is forwarded to ``judge.score()`` so that
    stochastic judges (e.g. :class:`~src.judge.claude_judge.ClaudeJudge`) can
    be probed with non-zero temperature.  Deterministic judges such as
    :class:`~src.judge.programmatic.ProgrammaticJudge` ignore the kwarg via
    ``**kwargs`` acceptance (note: ``ProgrammaticJudge.score`` does not accept
    ``temperature``, so we only pass it when the judge supports it).

    Parameters
    ----------
    judge:
        Any :class:`BaseJudge` implementation to evaluate.
    tasks:
        Ordered list of tasks to evaluate.
    responses:
        Ordered list of agent responses aligned with *tasks*.
    n_repeat:
        Number of times to call ``judge.score`` per task.
    temperature:
        Sampling temperature forwarded to the judge when it accepts it
        (detected via ``inspect``).

    Returns
    -------
    dict with keys:
        - ``overall_consistency`` : float — fraction of tasks where all
          ``n_repeat`` runs returned the same score.
        - ``per_task_details``    : list[dict] — one dict per task, each with:
            - ``task_id``        : int
            - ``scores``         : list[float] — raw scores from each run
            - ``majority_vote``  : float — mode of the scores
            - ``is_consistent``  : bool — True iff all runs agreed
            - ``agreement_rate`` : float — fraction of runs matching majority

    Raises
    ------
    ValueError
        If *tasks* and *responses* have different lengths.
    """
    import inspect

    if len(tasks) != len(responses):
        raise ValueError(
            f"tasks and responses must have the same length "
            f"({len(tasks)} vs {len(responses)})"
        )

    # Determine whether the judge's score() method accepts a 'temperature' kwarg
    try:
        sig = inspect.signature(judge.score)
        _judge_accepts_temperature = "temperature" in sig.parameters
    except (ValueError, TypeError):
        _judge_accepts_temperature = False

    per_task_details: list[dict] = []
    consistent_count = 0

    for task, response in zip(tasks, responses):
        run_scores: list[float] = []
        for _ in range(n_repeat):
            if _judge_accepts_temperature:
                result = judge.score(task, response, temperature=temperature)
            else:
                result = judge.score(task, response)
            run_scores.append(result.score)

        # Majority vote (mode); for binary 0/1 this is just the more common value
        ones = run_scores.count(1.0)
        zeros = run_scores.count(0.0)
        majority_vote: float = 1.0 if ones >= zeros else 0.0

        agreement_rate: float = run_scores.count(majority_vote) / n_repeat
        is_consistent: bool = len(set(run_scores)) == 1

        if is_consistent:
            consistent_count += 1

        per_task_details.append(
            {
                "task_id": task.id,
                "scores": run_scores,
                "majority_vote": majority_vote,
                "is_consistent": is_consistent,
                "agreement_rate": agreement_rate,
            }
        )

    n_tasks = len(tasks)
    overall_consistency: float = consistent_count / n_tasks if n_tasks > 0 else 0.0

    return {
        "overall_consistency": overall_consistency,
        "per_task_details": per_task_details,
    }
