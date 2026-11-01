"""
OpenAIAgent: calls GPT-4o to generate data-analysis code, executes it, and
retries on failure.

The agent is intentionally lazy about client creation so the module can be
imported safely without an ``OPENAI_API_KEY`` in the environment.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.benchmark.task import AgentResponse, DABenchTask
from src.benchmark.evaluator import infiagent_score
from src.agents.executor import CodeExecutor

# Load .env only if it exists — avoids raising an error in environments where
# the key is already set via the system environment.
load_dotenv()


SYSTEM_PROMPT = """You are a data analyst. Given a CSV file and a question, write Python code that:
1. Loads the CSV with: df = pd.read_csv('<file_name>')
2. Computes the answer
3. Outputs ONLY in this format: @<answer_name>[<answer_value>]

The answer name and format are specified in the question's format field.
Do not print anything else. Do not use print() for intermediate results."""

# Regex that matches @answer_name[answer_value] anywhere in stdout
_ANSWER_RE = re.compile(r"@(\w+)\[(.*?)\]")

# Regex that extracts the first ```python ... ``` fenced block
_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)


def _extract_code(text: str) -> str:
    """Return the first Python fenced code block, or the full text as fallback."""
    match = _CODE_BLOCK_RE.search(text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _parse_answer(stdout: str) -> dict[str, str]:
    """Extract all @name[value] pairs from stdout into a dict."""
    return {name: value for name, value in _ANSWER_RE.findall(stdout)}


def _csv_preview(file_path: Path) -> str:
    """Return a human-readable preview: first 5 rows + column dtypes."""
    try:
        df = pd.read_csv(file_path, nrows=5)
        dtypes_str = "\n".join(
            f"  {col}: {dtype}" for col, dtype in df.dtypes.items()
        )
        preview = df.to_string(index=False)
        return f"Column dtypes:\n{dtypes_str}\n\nFirst 5 rows:\n{preview}"
    except Exception as exc:  # noqa: BLE001
        return f"(Could not read CSV preview: {exc})"


def _determine_execution_result(answer: dict, stderr: str) -> str:
    """
    Determine the execution_result string based on answer and stderr.

    Priority order:
      1. "success"  — answer dict is non-empty
      2. "timeout"  — stderr contains timeout-related text
      3. "error"    — stderr is non-empty
      4. "no_answer" — code ran clean but no @name[value] found
    """
    if answer:
        return "success"
    if "TimeoutError" in stderr or "timed out" in stderr.lower():
        return "timeout"
    if stderr:
        return "error"
    return "no_answer"


class OpenAIAgent:
    """Agent that uses GPT-4o to answer DABench tasks via code generation."""

    def __init__(self, model: str = "gpt-4o", max_attempts: int = 3, timeout: int = 30) -> None:
        self.model = model
        self.max_attempts = max_attempts
        self._client: Optional[object] = None  # lazily initialised
        self._executor = CodeExecutor(timeout=timeout)

    @property
    def client(self):  # type: ignore[return]
        """Lazily create the OpenAI client on first access."""
        if self._client is None:
            import openai  # local import keeps module importable without key
            self._client = openai.OpenAI()
        return self._client

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _call_api(self, messages: list[dict]) -> tuple[str, dict]:
        """
        Call the OpenAI chat completions endpoint with exponential back-off.

        Returns (content, usage_dict) where usage_dict has prompt_tokens,
        completion_tokens, and total_tokens.
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        return content, usage

    def solve(self, task: DABenchTask) -> AgentResponse:
        """
        Attempt to solve *task* by having GPT-4o generate and execute code.

        Retries up to ``max_attempts`` times when execution produces stderr
        but no valid answer.
        """
        start_time = time.perf_counter()

        # Build CSV context
        file_path: Optional[Path] = task.file_path
        if file_path is not None and file_path.exists():
            csv_preview = _csv_preview(file_path)
            working_dir = file_path.parent
        else:
            csv_preview = "(CSV file not found)"
            working_dir = Path.cwd()

        # Initial message list
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"File: {task.file_name}\n\n"
                    f"CSV Preview:\n{csv_preview}\n\n"
                    f"Question: {task.question}\n\n"
                    f"Constraints: {task.constraints}\n\n"
                    f"Answer format: {task.format}"
                ),
            },
        ]

        code = ""
        stdout = ""
        stderr = ""
        error: Optional[str] = None
        reasoning_trace: Optional[str] = None

        # Accumulated token usage across all attempts
        token_usage: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

        attempt = 1
        for attempt in range(1, self.max_attempts + 1):
            # Get (or retry with error feedback) a code response
            raw_response, usage = self._call_api(messages)

            # Accumulate token counts
            for key in token_usage:
                token_usage[key] += usage.get(key, 0)

            # Capture the full first response as the reasoning trace
            if reasoning_trace is None:
                reasoning_trace = raw_response

            code = _extract_code(raw_response)

            stdout, stderr = self._executor.run(code, working_dir)
            answer = _parse_answer(stdout)

            # Success: we have at least one parsed answer and no blocking error
            if answer:
                error = None
                break

            # Failure: build error feedback and loop (unless last attempt)
            error = stderr.splitlines()[0] if stderr else "No @answer[value] found in output."
            if attempt < self.max_attempts:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"The code produced an error or no valid output.\n"
                            f"stderr: {stderr}\n"
                            f"stdout: {stdout}\n\n"
                            "Please fix the code and ensure it prints exactly "
                            "@answer_name[answer_value] and nothing else."
                        ),
                    }
                )
        else:
            # Exhausted all attempts — parse whatever stdout we have
            answer = _parse_answer(stdout)
            if not error:
                error = stderr.splitlines()[0] if stderr else "No @answer[value] found in output."

        runtime_seconds = time.perf_counter() - start_time
        execution_result = _determine_execution_result(answer, stderr)

        # Build response before scoring (scorer needs the answer dict)
        response = AgentResponse(
            task_id=task.id,
            dataset_id=task.file_name,
            question=task.question,
            model=self.model,
            reasoning_trace=reasoning_trace,
            code=code,
            attempts=attempt,
            stdout=stdout,
            stderr=stderr,
            execution_result=execution_result,
            answer=answer,
            builtin_eval_result=None,  # filled in below
            runtime_seconds=runtime_seconds,
            token_usage=token_usage,
            error=error if not answer else None,
        )

        # Run InfiAgent's built-in scorer
        response.builtin_eval_result = infiagent_score(task, response)

        return response
