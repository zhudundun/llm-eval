#!/usr/bin/env python
"""
Smoke test for the full LLM_eval pipeline — no API keys required.

Run from the repo root:
    python scripts/smoke_test.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure the repo root is on the path so `src` is importable.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

failures: list[str] = []


def check(label: str, fn):
    """Run fn(); print [PASS] or [FAIL]; collect failures."""
    try:
        fn()
        print(f"[PASS] {label}")
    except Exception:
        print(f"[FAIL] {label}")
        traceback.print_exc()
        failures.append(label)


# ---------------------------------------------------------------------------
# Step 1: import check
# ---------------------------------------------------------------------------
def step_imports():
    import src.benchmark.task       # noqa: F401
    import src.benchmark.loader     # noqa: F401
    import src.agents.executor      # noqa: F401
    import src.agents.openai_agent  # noqa: F401
    import src.judge.programmatic   # noqa: F401
    import src.judge.claude_judge   # noqa: F401
    import src.judge.composite      # noqa: F401
    import src.experiments.runner   # noqa: F401
    import src.calibration.metrics  # noqa: F401
    import src.reliability.bootstrap  # noqa: F401
    print("    All src modules imported successfully.")


check("Import check", step_imports)

# ---------------------------------------------------------------------------
# Step 2: loader check — 257 tasks
# ---------------------------------------------------------------------------
def step_loader():
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    assert len(tasks) == 257, f"Expected 257 tasks, got {len(tasks)}"
    print(f"    Loaded {len(tasks)} tasks.")


check("Loader check (257 tasks)", step_loader)

# ---------------------------------------------------------------------------
# Step 3: file check — >250 tasks have an existing CSV
# ---------------------------------------------------------------------------
def step_file_check():
    from src.benchmark.loader import DABenchLoader
    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    with_csv = [t for t in tasks if t.file_path is not None and t.file_path.exists()]
    print(f"    {len(with_csv)}/{len(tasks)} tasks have existing CSVs.")
    assert len(with_csv) > 250, f"Expected >250 tasks with CSVs, got {len(with_csv)}"


check("File check (>250 CSVs present)", step_file_check)

# ---------------------------------------------------------------------------
# Step 4: executor check — run pandas code on first task's CSV
# ---------------------------------------------------------------------------
def step_executor():
    from src.benchmark.loader import DABenchLoader
    from src.agents.executor import CodeExecutor

    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")

    # Find the first task that has a real CSV
    task = next(t for t in tasks if t.file_path is not None and t.file_path.exists())
    working_dir = task.file_path.parent

    code = (
        "import pandas as pd\n"
        f"df = pd.read_csv('{task.file_name}')\n"
        "print(len(df))\n"
    )
    stdout, stderr = CodeExecutor(timeout=30).run(code, working_dir)
    print(f"    Executor stdout={stdout!r}  stderr={stderr!r}")
    assert stderr == "", f"Expected empty stderr, got: {stderr!r}"
    assert stdout.strip().isdigit(), f"Expected a row count in stdout, got: {stdout!r}"


check("Executor check (pandas read_csv)", step_executor)

# ---------------------------------------------------------------------------
# Step 5: programmatic judge check — empty answer dict → score 0.0
# ---------------------------------------------------------------------------
def step_judge_empty():
    from src.benchmark.loader import DABenchLoader
    from src.benchmark.task import AgentResponse
    from src.judge.programmatic import ProgrammaticJudge

    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")
    # Pick a task that has ground truth so we exercise the "no answer" branch
    task = next(t for t in tasks if t.common_answers)

    response = AgentResponse(
        task_id=task.id,
        model="mock",
        code="",
        stdout="",
        stderr="",
        answer={},
    )
    result = ProgrammaticJudge().score(task, response)
    print(f"    Score={result.score}  reasoning={result.reasoning!r}")
    assert result.score == 0.0, f"Expected score 0.0, got {result.score}"


check("Programmatic judge (empty answer → 0.0)", step_judge_empty)

# ---------------------------------------------------------------------------
# Step 6: end-to-end check — 5 tasks through MockAgent + ProgrammaticJudge
# ---------------------------------------------------------------------------
class MockAgent:
    """Agent that generates simple pandas code without any LLM call."""

    model = "mock"

    def solve(self, task):
        from src.benchmark.task import AgentResponse
        from src.agents.executor import CodeExecutor

        if task.file_path is not None and task.file_path.exists():
            working_dir = task.file_path.parent
            code = (
                "import pandas as pd\n"
                f"df = pd.read_csv('{task.file_name}')\n"
                "print(len(df))\n"
            )
        else:
            working_dir = Path.cwd()
            code = "print(0)\n"

        stdout, stderr = CodeExecutor(timeout=30).run(code, working_dir)
        # MockAgent produces no @name[value] tokens — answer is always {}
        return AgentResponse(
            task_id=task.id,
            model=self.model,
            code=code,
            stdout=stdout,
            stderr=stderr,
            answer={},
        )


def step_end_to_end():
    from src.benchmark.loader import DABenchLoader
    from src.judge.programmatic import ProgrammaticJudge
    from src.experiments.runner import ExperimentRunner
    import tempfile

    loader = DABenchLoader(REPO_ROOT / "data" / "dabench")
    tasks = loader.load_tasks("dev")[:5]

    with tempfile.TemporaryDirectory() as tmpdir:
        runner = ExperimentRunner(MockAgent(), Path(tmpdir))
        responses = runner.run(tasks, show_progress=False)

    assert len(responses) == 5, f"Expected 5 responses, got {len(responses)}"

    judge = ProgrammaticJudge()
    task_map = {t.id: t for t in tasks}
    for resp in responses:
        task = task_map[resp.task_id]
        score = judge.score(task, resp)
        # MockAgent never emits @name[value] so score must be 0.0
        assert score.score == 0.0, f"Unexpected score {score.score} for task {task.id}"

    print(f"    5/5 tasks ran and scored without crashing.")


check("End-to-end check (5 tasks, MockAgent + judge)", step_end_to_end)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print()
if failures:
    print(f"RESULT: {len(failures)} step(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("RESULT: All steps passed.")
    sys.exit(0)
