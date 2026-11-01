"""
DABench benchmark loader.

Reads from the InfiAgent-DABench repository layout:

    <data_dir>/
    └── infiagent-repo/
        └── examples/DA-Agent/data/
            ├── da-dev-questions.jsonl
            ├── da-dev-labels.jsonl
            └── da-dev-tables/
                ├── test_ave.csv
                └── ...

Only the "dev" split is present in the repo.  The "test" split structure is
anticipated but not yet released; this loader will raise FileNotFoundError
with a clear message if you request it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator, Optional

from .task import DABenchTask

logger = logging.getLogger(__name__)

# Relative paths inside infiagent-repo — matches the cloned repo layout exactly.
_DA_AGENT_DATA = Path("infiagent-repo") / "examples" / "DA-Agent" / "data"

_SPLIT_FILES: dict[str, dict[str, str]] = {
    "dev": {
        "questions": "da-dev-questions.jsonl",
        "labels": "da-dev-labels.jsonl",
        "tables_dir": "da-dev-tables",
    },
    # test split: questions exist in some release variants; labels are withheld.
    "test": {
        "questions": "da-test-questions.jsonl",
        "labels": None,
        "tables_dir": "da-test-tables",
    },
}


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed dicts."""
    records = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {lineno} of {path}: {exc}"
                ) from exc
    return records


class DABenchLoader:
    """
    Loads DABench tasks from the cloned InfiAgent-DABench repository.

    Parameters
    ----------
    data_dir:
        Root of the dabench data directory, e.g.
        ``Path("/Users/wang/LLM_eval/data/dabench")``.
        The loader expects ``<data_dir>/infiagent-repo/`` to exist.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._repo_data_dir = self.data_dir / _DA_AGENT_DATA

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_tasks(self, split: str = "dev") -> Iterator[DABenchTask]:
        """
        Yield :class:`DABenchTask` objects for the requested split.

        Parameters
        ----------
        split:
            ``"dev"`` (default) or ``"test"``.

        Yields
        ------
        DABenchTask
            One task per line in the questions JSONL.  The ``common_answers``
            field is populated from the labels file when it exists; otherwise
            it is ``None``.  The ``file_path`` field is set to the absolute
            path of the CSV if the file exists; otherwise ``None`` and a
            warning is logged.
        """
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}. Valid splits: {list(_SPLIT_FILES)}"
            )

        split_cfg = _SPLIT_FILES[split]

        # --- questions ---
        questions_path = self._repo_data_dir / split_cfg["questions"]
        if not questions_path.exists():
            raise FileNotFoundError(
                f"Questions file not found for split={split!r}: {questions_path}\n"
                "Make sure the InfiAgent-DABench repo has been cloned into "
                f"{self.data_dir / 'infiagent-repo'}."
            )
        questions = _read_jsonl(questions_path)

        # --- labels (optional) ---
        label_map: dict[int, list[list[str]]] = {}
        labels_filename = split_cfg["labels"]
        if labels_filename is not None:
            labels_path = self._repo_data_dir / labels_filename
            if labels_path.exists():
                for record in _read_jsonl(labels_path):
                    label_map[record["id"]] = record["common_answers"]
            else:
                logger.warning(
                    "Labels file not found for split=%r: %s — "
                    "tasks will have common_answers=None",
                    split,
                    labels_path,
                )

        # --- CSV tables directory ---
        tables_dir = self._repo_data_dir / split_cfg["tables_dir"]
        if not tables_dir.exists():
            logger.warning(
                "Tables directory not found for split=%r: %s — "
                "all file_path values will be None",
                split,
                tables_dir,
            )
            tables_dir = None  # type: ignore[assignment]

        # --- yield tasks ---
        for raw in questions:
            task_id: int = raw["id"]
            file_name: str = raw["file_name"]

            file_path: Optional[Path] = None
            if tables_dir is not None:
                candidate = tables_dir / file_name
                if candidate.exists():
                    file_path = candidate.resolve()
                else:
                    logger.warning(
                        "CSV not found for task id=%d (file_name=%r): %s",
                        task_id,
                        file_name,
                        candidate,
                    )

            yield DABenchTask(
                id=task_id,
                question=raw["question"],
                concepts=raw["concepts"],
                constraints=raw["constraints"],
                format=raw["format"],
                file_name=file_name,
                level=raw["level"],
                common_answers=label_map.get(task_id),
                file_path=file_path,
            )

    def load_tasks(self, split: str = "dev") -> list[DABenchTask]:
        """
        Return all tasks for the requested split as a list.

        Parameters
        ----------
        split:
            ``"dev"`` (default) or ``"test"``.
        """
        return list(self.iter_tasks(split))

    def get_task(self, task_id: int, split: str = "dev") -> DABenchTask:
        """
        Return the single task matching *task_id* in the given split.

        Parameters
        ----------
        task_id:
            The integer ``id`` field from the questions JSONL.
        split:
            ``"dev"`` (default) or ``"test"``.

        Raises
        ------
        KeyError
            If no task with the given id exists in the split.
        """
        for task in self.iter_tasks(split):
            if task.id == task_id:
                return task
        raise KeyError(
            f"No task with id={task_id!r} found in split={split!r}"
        )
