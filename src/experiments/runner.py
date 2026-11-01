"""
ExperimentRunner: orchestrates running an agent over all tasks with resume
support so interrupted runs can be continued without re-processing tasks.

Results are written as JSON Lines (one AgentResponse per line) immediately
after each task completes.  The runner detects already-completed task IDs on
startup so the experiment can be safely restarted.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import jsonlines
from tqdm import tqdm

from src.benchmark.task import AgentResponse, DABenchTask


def _response_to_dict(response: AgentResponse) -> dict[str, Any]:
    """
    Convert an AgentResponse to a JSON-serialisable dict.

    Path objects (e.g. if any field ever holds a Path) are converted to str
    so json.dumps never raises a TypeError.
    """
    d = asdict(response)
    # Convert any Path values to str for JSON serialisation
    for key, value in d.items():
        if isinstance(value, Path):
            d[key] = str(value)
    return d


def _dict_to_response(d: dict[str, Any]) -> AgentResponse:
    """
    Reconstruct an AgentResponse from a plain dict (as read from JSONL).

    Handles missing keys gracefully so older JSONL rows without the new
    enriched fields can still be loaded without crashing.
    """
    return AgentResponse(
        task_id=d["task_id"],
        dataset_id=d.get("dataset_id", ""),
        question=d.get("question", ""),
        model=d["model"],
        reasoning_trace=d.get("reasoning_trace"),
        code=d["code"],
        attempts=d.get("attempts", 1),
        stdout=d["stdout"],
        stderr=d["stderr"],
        execution_result=d.get("execution_result", "success" if d.get("answer") else "no_answer"),
        answer=d.get("answer", {}),
        builtin_eval_result=d.get("builtin_eval_result"),
        runtime_seconds=d.get("runtime_seconds", 0.0),
        token_usage=d.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
        error=d.get("error"),
    )


class ExperimentRunner:
    """Run an agent over a list of tasks and persist results incrementally."""

    def __init__(self, agent: Any, output_dir: Path) -> None:
        self.agent = agent
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / "agent_responses.jsonl"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        tasks: list[DABenchTask],
        show_progress: bool = True,
    ) -> list[AgentResponse]:
        """
        Solve *tasks* with ``self.agent``, skipping already-completed ones.

        Completed task IDs are loaded from ``output_path`` at startup.  Each
        new result is appended to the file immediately after the task finishes.

        Returns all responses (previously saved + newly computed).
        """
        completed: dict[int, AgentResponse] = {
            r.task_id: r for r in self.load_responses()
        }

        pending = [t for t in tasks if t.id not in completed]

        if show_progress:
            iterator = tqdm(pending, desc="Running tasks", unit="task")
        else:
            iterator = pending  # type: ignore[assignment]

        with jsonlines.open(self.output_path, mode="a") as writer:
            for task in iterator:
                if show_progress:
                    iterator.set_postfix(task_id=task.id)  # type: ignore[union-attr]
                response = self.agent.solve(task)
                writer.write(_response_to_dict(response))
                completed[response.task_id] = response

        # Return in the same order as *tasks*
        task_ids_ordered = [t.id for t in tasks]
        return [completed[tid] for tid in task_ids_ordered if tid in completed]

    def load_responses(self) -> list[AgentResponse]:
        """Read ``output_path`` and return all saved AgentResponse objects."""
        if not self.output_path.exists():
            return []

        responses: list[AgentResponse] = []
        with jsonlines.open(self.output_path, mode="r") as reader:
            for obj in reader:
                try:
                    responses.append(_dict_to_response(obj))
                except (KeyError, TypeError):
                    # Skip malformed lines rather than crashing the whole run
                    continue
        return responses
