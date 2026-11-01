"""
IndependentVerifier: a reference-free failure signal for the LLM-judge eval.

The judge in no-reference mode only *reads* the agent's code and logs — it never
recomputes, so it is structurally blind to a confidently-wrong numeric answer.
This verifier closes that blind spot two ways:

  1. RE-SOLVE (cold): a *different-family* model (Claude Sonnet 4.6) writes its
     OWN code to re-derive the answer, having seen ONLY the question, the answer
     format, and a small CSV preview — never the agent's code, the agent's
     answer, or any reference/ground-truth answer.  We then compare its answer
     to the agent's.  Because the two models are from different families their
     errors are largely uncorrelated, so agreement is meaningful evidence.

  2. RE-EXECUTE (reproducibility): we re-run the AGENT's own code and check that
     its stdout reproduces the answer the agent claimed.  A mismatch flags a
     non-reproducible / stale answer.

The Anthropic client is created lazily so this module imports cleanly without an
``ANTHROPIC_API_KEY`` in the environment.
"""
from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_random_exponential

from src.agents.executor import CodeExecutor
from src.benchmark.task import AgentResponse, DABenchTask, VerifierResult

# Load .env only if present — never raises if the key is already in the env.
load_dotenv()


_DEFAULT_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a careful data analyst. You are given a CSV file and a question. Write Python code that:
1. Loads the CSV with: df = pd.read_csv('<file_name>')   (use the exact file name given)
2. Computes the answer to the question from the data.
3. Prints ONLY the answer in this exact format: @<answer_name>[<answer_value>]

Use the EXACT answer name(s) from the task's answer-format specification.
Respect every constraint in the task (rounding, ddof, filters, units, etc.).
Output a single fenced Python code block and nothing else.
Do not print anything other than the @<answer_name>[<answer_value>] line(s).
Do not use print() for intermediate results or debugging."""

# Matches @answer_name[answer_value] anywhere in stdout.
_ANSWER_RE = re.compile(r"@(\w+)\[(.*?)\]")

# Extracts the first ```python ... ``` fenced block.
_CODE_BLOCK_RE = re.compile(r"```python\s*(.*?)```", re.DOTALL)

# Tolerance for confirming the AGENT's code reproduces its own claimed answer.
_REPRO_ABS_TOL = 1e-6


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
    """Return a human-readable preview: column dtypes + first 5 rows.

    Mirrors ``openai_agent._csv_preview`` so the verifier sees the same shape of
    context the agent did.
    """
    try:
        df = pd.read_csv(file_path, nrows=5)
        dtypes_str = "\n".join(
            f"  {col}: {dtype}" for col, dtype in df.dtypes.items()
        )
        preview = df.to_string(index=False)
        return f"Column dtypes:\n{dtypes_str}\n\nFirst 5 rows:\n{preview}"
    except Exception as exc:  # noqa: BLE001
        return f"(Could not read CSV preview: {exc})"


def _try_float(value: object) -> Optional[float]:
    """Best-effort float parse; returns None if the value is not numeric."""
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def compare_answers(
    a: dict, b: dict, rel_tol: float
) -> tuple[Optional[bool], str]:
    """Compare two {name: value} answer dicts (a=agent, b=verifier).

    Rules
    -----
    * For each name in ``keys(a) & keys(b)``: try to float-parse both. If both
      are numeric, match iff ``rel = |x-y| / max(|y|, 1e-9) <= rel_tol`` OR
      ``|x-y| <= 1e-6``. Otherwise compare as case-insensitive stripped strings.
    * ``agrees = True`` iff the key sets are equal AND every shared value
      matches; ``False`` if any shared value mismatches OR the key sets differ;
      ``None`` if there are no shared keys or either dict is empty.
    * ``detail`` is one line per shared name plus a note when key sets differ.

    Returns
    -------
    (agrees, detail)
    """
    keys_a = set(a.keys())
    keys_b = set(b.keys())
    shared = keys_a & keys_b

    # Cannot compare: either side empty, or no overlap in answer names.
    if not a or not b or not shared:
        note_parts = []
        if not a:
            note_parts.append("agent answer empty")
        if not b:
            note_parts.append("verifier answer empty")
        if a and b and not shared:
            note_parts.append(
                f"no shared answer names (agent={sorted(keys_a)} "
                f"verifier={sorted(keys_b)})"
            )
        return None, "; ".join(note_parts) or "no comparable answers"

    lines: list[str] = []
    all_match = True
    for name in sorted(shared):
        av = a[name]
        bv = b[name]
        xa = _try_float(av)
        xb = _try_float(bv)
        if xa is not None and xb is not None:
            diff = abs(xa - xb)
            rel = diff / max(abs(xb), 1e-9)
            matched = rel <= rel_tol or diff <= 1e-6
            lines.append(
                f"{name}: agent={av} verifier={bv} rel={rel:.2%}"
                + ("" if matched else "  <-- MISMATCH")
            )
        else:
            matched = str(av).strip().lower() == str(bv).strip().lower()
            lines.append(
                f"{name}: agent={av} verifier={bv} "
                f"(string {'match' if matched else 'MISMATCH'})"
            )
        if not matched:
            all_match = False

    keys_equal = keys_a == keys_b
    if not keys_equal:
        only_agent = sorted(keys_a - keys_b)
        only_verifier = sorted(keys_b - keys_a)
        lines.append(
            f"NOTE: answer-name sets differ "
            f"(only-agent={only_agent} only-verifier={only_verifier})"
        )

    agrees = keys_equal and all_match
    return agrees, "\n".join(lines)


class IndependentVerifier:
    """Reference-free verifier: re-solve cold with Sonnet + re-execute agent code."""

    SYSTEM_PROMPT = SYSTEM_PROMPT

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        timeout: int = 30,
        max_attempts: int = 3,
        rel_tol: float = 1e-3,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.rel_tol = rel_tol
        self._client: Optional[object] = None  # lazily initialised
        self._executor = CodeExecutor(timeout=timeout)

    @property
    def client(self):  # type: ignore[return]
        """Lazily create the ``anthropic.Anthropic`` client on first access."""
        if self._client is None:
            import anthropic  # local import keeps module importable without key
            self._client = anthropic.Anthropic()
        return self._client

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(5))
    def _call_api(self, messages: list[dict]) -> tuple[str, dict]:
        """Call the Anthropic Messages API and return (text, usage_dict).

        No sampling parameters are sent: Anthropic's newest models (incl. Sonnet
        4.6) reject ``temperature``/``top_p``/``top_k`` with a 400.  Omitting
        them is sufficiently deterministic for this use.  ``usage_dict`` is
        normalised to the OpenAI-style keys the rest of the codebase uses.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=self.SYSTEM_PROMPT,
            messages=messages,
        )
        text_parts = [
            block.text
            for block in message.content
            if getattr(block, "type", None) == "text"
        ]
        content = "\n".join(text_parts)
        usage = {
            "prompt_tokens": getattr(message.usage, "input_tokens", 0) or 0,
            "completion_tokens": getattr(message.usage, "output_tokens", 0) or 0,
            "total_tokens": (
                (getattr(message.usage, "input_tokens", 0) or 0)
                + (getattr(message.usage, "output_tokens", 0) or 0)
            ),
        }
        return content, usage

    def _re_solve(
        self, task: DABenchTask, working_dir: Path, csv_preview: str
    ) -> tuple[str, str, str, dict, dict]:
        """Have Sonnet re-derive the answer cold.

        The prompt contains ONLY the file name, CSV preview, question,
        constraints, and answer format — never the agent's code/answer or any
        reference.  Retries with stderr feedback up to ``max_attempts``.

        Returns (code, stdout, stderr, verifier_answer, token_usage).
        """
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # First user turn — strictly reference-free / answer-free context.
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"File: {task.file_name}\n\n"
                    f"CSV Preview:\n{csv_preview}\n\n"
                    f"Question: {task.question}\n\n"
                    f"Constraints: {task.constraints}\n\n"
                    f"Answer format: {task.format}"
                ),
            }
        ]

        code = ""
        stdout = ""
        stderr = ""
        answer: dict[str, str] = {}

        for attempt in range(1, self.max_attempts + 1):
            raw_response, usage = self._call_api(messages)
            for key in token_usage:
                token_usage[key] += usage.get(key, 0)

            code = _extract_code(raw_response)
            stdout, stderr = self._executor.run(code, working_dir)
            answer = _parse_answer(stdout)

            if answer:
                break

            if attempt < self.max_attempts:
                messages.append({"role": "assistant", "content": raw_response})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The code produced an error or no valid output.\n"
                            f"stderr: {stderr}\n"
                            f"stdout: {stdout}\n\n"
                            "Please fix the code and ensure it prints exactly "
                            "@answer_name[answer_value] (using the exact answer "
                            "name from the format spec) and nothing else."
                        ),
                    }
                )

        return code, stdout, stderr, answer, token_usage

    def _reproduce_agent(
        self, response: AgentResponse, working_dir: Path
    ) -> tuple[dict, Optional[bool]]:
        """Re-run the agent's own code and check it reproduces its claimed answer.

        Returns (reproduced_answer, agent_code_reproduces).  ``agent_code_reproduces``
        is None when the agent code is missing; otherwise it is True iff every
        answer name the agent claimed is reproduced within a STRICT tolerance
        (numeric: abs diff <= 1e-6; string: case-insensitive stripped equality).
        """
        if not response.code or not response.code.strip():
            return {}, None

        stdout, _stderr = self._executor.run(response.code, working_dir)
        reproduced = _parse_answer(stdout)

        claimed = response.answer or {}
        if not claimed:
            # Nothing to compare against; reproduction is undefined.
            return reproduced, bool(reproduced) if reproduced else None

        reproduces = True
        for name, claimed_val in claimed.items():
            if name not in reproduced:
                reproduces = False
                break
            repro_val = reproduced[name]
            xc = _try_float(claimed_val)
            xr = _try_float(repro_val)
            if xc is not None and xr is not None:
                if abs(xc - xr) > _REPRO_ABS_TOL:
                    reproduces = False
                    break
            else:
                if str(claimed_val).strip().lower() != str(repro_val).strip().lower():
                    reproduces = False
                    break

        return reproduced, reproduces

    def verify(self, task: DABenchTask, response: AgentResponse) -> VerifierResult:
        """Run the full reference-free verification for one agent response."""
        start_time = time.perf_counter()

        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        error: Optional[str] = None

        # Resolve the CSV location (same logic the agent used).
        file_path: Optional[Path] = task.file_path
        if file_path is not None and file_path.exists():
            csv_preview = _csv_preview(file_path)
            working_dir = file_path.parent
        else:
            csv_preview = "(CSV file not found)"
            working_dir = Path.cwd()

        # --- 1. RE-SOLVE (cold, different model) ---
        verifier_code = ""
        verifier_stdout = ""
        verifier_stderr = ""
        verifier_answer: dict[str, str] = {}
        try:
            (
                verifier_code,
                verifier_stdout,
                verifier_stderr,
                verifier_answer,
                solve_usage,
            ) = self._re_solve(task, working_dir, csv_preview)
            for key in token_usage:
                token_usage[key] += solve_usage.get(key, 0)
        except Exception as exc:  # noqa: BLE001
            error = f"re-solve failed: {exc}"

        verifier_execution_ok = bool(verifier_answer)

        # --- 2. COMPARE verifier vs agent ---
        agrees_with_agent, agreement_detail = compare_answers(
            response.answer or {}, verifier_answer, self.rel_tol
        )

        # --- 3. RE-EXECUTE the agent's own code ---
        reproduced_answer: dict[str, str] = {}
        agent_code_reproduces: Optional[bool] = None
        try:
            reproduced_answer, agent_code_reproduces = self._reproduce_agent(
                response, working_dir
            )
        except Exception as exc:  # noqa: BLE001
            repro_err = f"agent-code re-execution failed: {exc}"
            error = f"{error}; {repro_err}" if error else repro_err

        runtime_seconds = time.perf_counter() - start_time

        return VerifierResult(
            task_id=task.id,
            verifier_model=self.model,
            verifier_code=verifier_code,
            verifier_stdout=verifier_stdout,
            verifier_stderr=verifier_stderr,
            verifier_answer=verifier_answer,
            verifier_execution_ok=verifier_execution_ok,
            agrees_with_agent=agrees_with_agent,
            agreement_detail=agreement_detail,
            reproduced_answer=reproduced_answer,
            agent_code_reproduces=agent_code_reproduces,
            runtime_seconds=runtime_seconds,
            token_usage=token_usage,
            error=error,
        )
