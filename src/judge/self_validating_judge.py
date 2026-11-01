"""
Self-validating LLM judge for data-science agent outputs.

Unlike ``RubricJudge`` (which only reads the agent's code/logs) or the separate
independent verifier (a different model that re-solves the task), the
``SelfValidatingJudge`` is an *agentic* Opus 4.7 that **recomputes the answer
itself** using a code-execution tool, then renders its verdict.

It follows a THINK -> DO loop:

  * THINK: reason about what to recompute to check the agent's answer.
  * DO:    call the ``run_python`` tool to load the CSV and INDEPENDENTLY
           recompute the answer, comparing the result to the agent's claim.

The loop runs up to ``max_tool_iters`` turns.  On each turn the model may call
``run_python`` (executed locally via ``CodeExecutor`` against the task's CSV
directory) or ``record_judgment`` (the structured verdict, which ends the loop).
If the model stalls without recording a judgment it is nudged; on the final
iteration ``record_judgment`` is forced via ``tool_choice``.

Reliable structured output is obtained the same way as ``RubricJudge``: a
``record_judgment`` tool whose ``input_schema`` is the judge_result schema,
extended with three self-validation fields (``self_derived_answer``,
``self_validation_agrees``, ``validation_summary``).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.agents.executor import CodeExecutor
from src.benchmark.task import AgentResponse, DABenchTask
from src.judge.validate import FAILURE_TYPES, validate_judge_output

# Load .env (if present) so ANTHROPIC_API_KEY is available without breaking
# importability when python-dotenv is absent.
try:  # pragma: no cover - trivial import guard
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-opus-4-7"
_RECORD_TOOL_NAME = "record_judgment"
_RUN_TOOL_NAME = "run_python"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Max characters of stdout / stderr returned to the model per run_python call.
_TOOL_RESULT_TRUNCATE = 4000

# The seven 1-5 score fields, in canonical order.
_SCORE_FIELDS: tuple[str, ...] = (
    "task_success_score",
    "code_correctness_score",
    "data_handling_score",
    "statistical_validity_score",
    "reproducibility_score",
    "final_answer_quality_score",
    "reasoning_quality_score",
)

_SELF_VALIDATION_VERDICTS: tuple[str, ...] = (
    "agree",
    "disagree",
    "could_not_validate",
)

# The three extra self-validation fields the record_judgment tool carries on top
# of the 11 standard judge fields.
_EXTRA_FIELDS: tuple[str, ...] = (
    "self_derived_answer",
    "self_validation_agrees",
    "validation_summary",
)


def _run_python_tool() -> dict[str, Any]:
    """The ``run_python`` tool: executes Python in the CSV's directory."""
    return {
        "name": _RUN_TOOL_NAME,
        "description": (
            "Execute Python code in the directory containing the task's CSV file. "
            "Use it to load the CSV and INDEPENDENTLY recompute the agent's answer. "
            "Only stdout and stderr are returned, so print() any value you want to "
            "inspect. You may call this tool multiple times."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python source to execute.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        },
    }


def _record_judgment_tool() -> dict[str, Any]:
    """The ``record_judgment`` tool: 11 standard fields + 3 self-validation fields."""
    score_props = {
        field: {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
            "description": f"Integer 1-5 score for {field.replace('_', ' ')}.",
        }
        for field in _SCORE_FIELDS
    }

    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            **score_props,
            "critical_errors": {
                "type": "array",
                "description": (
                    "Concrete failures found in the agent's solution. Empty if none."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {
                            "type": "string",
                            "enum": list(FAILURE_TYPES),
                            "description": "Failure type from the controlled taxonomy.",
                        },
                        "severity": {
                            "type": "string",
                            "enum": ["low", "medium", "high"],
                            "description": "Severity of this failure.",
                        },
                        "explanation": {
                            "type": "string",
                            "description": "Specific, evidence-grounded description.",
                        },
                    },
                    "required": ["type", "severity", "explanation"],
                    "additionalProperties": False,
                },
            },
            "overall_pass": {
                "type": "boolean",
                "description": "Whether the answer should be accepted overall.",
            },
            "confidence": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
                "description": "Confidence in this verdict, 0-1.",
            },
            "judge_explanation": {
                "type": "string",
                "description": "Concise justification citing specific evidence.",
            },
            # --- self-validation fields ---
            "self_derived_answer": {
                "type": "string",
                "description": (
                    "The value YOU computed yourself via run_python, e.g. "
                    "'mean_fare=34.65'."
                ),
            },
            "self_validation_agrees": {
                "type": "string",
                "enum": list(_SELF_VALIDATION_VERDICTS),
                "description": (
                    "Whether your independently computed value agrees with the "
                    "agent's answer."
                ),
            },
            "validation_summary": {
                "type": "string",
                "description": (
                    "How you validated: what you loaded, what you computed, and "
                    "the comparison result."
                ),
            },
        },
        "required": [
            *_SCORE_FIELDS,
            "critical_errors",
            "overall_pass",
            "confidence",
            "judge_explanation",
            *_EXTRA_FIELDS,
        ],
        "additionalProperties": False,
    }

    return {
        "name": _RECORD_TOOL_NAME,
        "description": (
            "Record the structured judgment of a data-science agent's solution. "
            "Call this exactly once, AFTER you have independently recomputed the "
            "answer with run_python, with all rubric scores, any critical errors, "
            "the overall pass/fail decision, confidence, an explanation, and the "
            "three self-validation fields."
        ),
        "input_schema": input_schema,
    }


def _neutral_result() -> dict[str, Any]:
    """Neutral-default verdict used when no valid record_judgment is produced."""
    return {
        **{field: 3 for field in _SCORE_FIELDS},
        "critical_errors": [],
        "overall_pass": False,
        "confidence": 0.0,
        "judge_explanation": (
            "Judge did not produce a valid record_judgment after the tool loop; "
            "neutral default scores applied."
        ),
        "self_derived_answer": "",
        "self_validation_agrees": "could_not_validate",
        "validation_summary": "Self-validation did not complete.",
    }


def _truncate(text: str, limit: int = _TOOL_RESULT_TRUNCATE) -> str:
    """Truncate ``text`` to ``limit`` chars, appending a marker when cut."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


class SelfValidatingJudge:
    """Agentic LLM judge that recomputes the answer itself before judging."""

    PROMPT_VERSION = "judge_selfvalidate_v1"

    _VALID_MODES: tuple[str, ...] = ("no_reference",)

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        timeout: int = 30,
        max_tool_iters: int = 6,
    ) -> None:
        self.model = model
        self.timeout = timeout
        self.max_tool_iters = max_tool_iters
        self._client: Optional[Any] = None
        self._run_tool = _run_python_tool()
        self._record_tool = _record_judgment_tool()
        self._template: Optional[str] = None

    @property
    def client(self) -> Any:
        """Lazily-created ``anthropic.Anthropic`` client (keeps module importable)."""
        if self._client is None:
            import anthropic  # local import — no key needed to import this module

            self._client = anthropic.Anthropic()
        return self._client

    # -- prompt construction --------------------------------------------------

    def _template_text(self) -> str:
        if self._template is None:
            path = _PROMPTS_DIR / "judge_self_validate_v1.txt"
            self._template = path.read_text(encoding="utf-8")
        return self._template

    def _build_prompt(
        self,
        task: DABenchTask,
        response: AgentResponse,
        dataset_summary: dict,
        prechecks: dict,
    ) -> str:
        fields: dict[str, str] = {
            "question": task.question,
            "dataset_summary": json.dumps(dataset_summary, ensure_ascii=False),
            "precheck_results": json.dumps(prechecks, ensure_ascii=False),
            "agent_final_answer": json.dumps(response.answer, ensure_ascii=False),
            "agent_code": response.code or "(no code provided)",
            "stdout": response.stdout or "(empty)",
            "stderr": response.stderr or "(empty)",
            "file_name": task.file_name,
        }
        return self._template_text().format(**fields)

    # -- token accounting -----------------------------------------------------

    @staticmethod
    def _accumulate_usage(total: dict[str, int], message: Any) -> None:
        usage = getattr(message, "usage", None)
        if usage is None:
            return
        prompt = int(getattr(usage, "input_tokens", 0) or 0)
        completion = int(getattr(usage, "output_tokens", 0) or 0)
        total["prompt_tokens"] += prompt
        total["completion_tokens"] += completion
        total["total_tokens"] += prompt + completion

    # -- agentic tool loop ----------------------------------------------------

    def _run_python(self, code: str, working_dir: Path) -> str:
        """Execute ``code`` via CodeExecutor; return truncated stdout+stderr."""
        stdout, stderr = CodeExecutor(self.timeout).run(code, working_dir)
        parts = [
            "STDOUT:",
            _truncate(stdout) or "(empty)",
            "",
            "STDERR:",
            _truncate(stderr) or "(empty)",
        ]
        return "\n".join(parts)

    def _agentic_loop(
        self,
        prompt: str,
        working_dir: Path,
    ) -> tuple[dict[str, Any], bool, int, dict[str, int]]:
        """Drive the THINK -> DO -> record_judgment loop.

        Returns ``(verdict_dict, parse_success, n_tool_calls, token_usage)``.
        ``n_tool_calls`` counts ``run_python`` invocations only.
        """
        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        n_tool_calls = 0

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        # Cache the large static prompt prefix; the volatile
                        # tool exchange is appended after it on every turn.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        tools = [self._run_tool, self._record_tool]

        for iteration in range(self.max_tool_iters):
            last_iter = iteration == self.max_tool_iters - 1
            tool_choice: dict[str, Any]
            if last_iter:
                # Force a verdict on the final turn so the loop always ends
                # with a structured judgment when possible.
                tool_choice = {"type": "tool", "name": _RECORD_TOOL_NAME}
            else:
                tool_choice = {"type": "auto"}

            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    tools=tools,
                    tool_choice=tool_choice,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - API/transport error
                logger.warning("Self-validating judge call failed: %s", exc)
                return _neutral_result(), False, n_tool_calls, token_usage

            self._accumulate_usage(token_usage, message)

            # Append the assistant turn verbatim (content blocks as dicts).
            assistant_content = [
                self._block_to_dict(block) for block in message.content
            ]
            messages.append({"role": "assistant", "content": assistant_content})

            # Collect tool_use blocks; record_judgment ends the loop.
            tool_results: list[dict[str, Any]] = []
            record_input: Optional[dict[str, Any]] = None

            for block in message.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == _RECORD_TOOL_NAME:
                    record_input = dict(block.input)
                elif block.name == _RUN_TOOL_NAME:
                    n_tool_calls += 1
                    code = (block.input or {}).get("code", "")
                    result_text = self._run_python(code, working_dir)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_text,
                        }
                    )

            # If the model recorded a judgment, validate and return.
            if record_input is not None:
                # Any run_python calls in the SAME turn still need results, but
                # since we are stopping we don't send them back. Validate now.
                verdict, ok = self._validate_record(record_input)
                return verdict, ok, n_tool_calls, token_usage

            # Feed back any tool_result(s) and continue the loop.
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                continue

            # Model stopped without calling either tool: nudge it (unless this
            # was already the final, forced turn — that case is handled below).
            if not last_iter:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not yet recorded a verdict. First use "
                            "run_python to independently recompute the answer, "
                            "then call record_judgment exactly once with all "
                            "required fields including the three self-validation "
                            "fields."
                        ),
                    }
                )
                continue

        # Exhausted the loop without a valid record_judgment.
        logger.info(
            "Self-validating judge exhausted %d iterations without a verdict.",
            self.max_tool_iters,
        )
        return _neutral_result(), False, n_tool_calls, token_usage

    @staticmethod
    def _block_to_dict(block: Any) -> dict[str, Any]:
        """Convert an SDK content block to a plain dict for message replay."""
        btype = getattr(block, "type", None)
        if btype == "text":
            return {"type": "text", "text": block.text}
        if btype == "tool_use":
            return {
                "type": "tool_use",
                "id": block.id,
                "name": block.name,
                "input": dict(block.input) if block.input else {},
            }
        if btype == "thinking":
            return {
                "type": "thinking",
                "thinking": getattr(block, "thinking", ""),
                "signature": getattr(block, "signature", ""),
            }
        # Best-effort fallback via model_dump for any other block type.
        dump = getattr(block, "model_dump", None)
        if callable(dump):
            return dump()
        return {"type": str(btype)}

    def _validate_record(
        self, record_input: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Validate the 11 standard fields; tolerate/normalize the 3 extras.

        Returns ``(verdict_dict, parse_success)``.  On validation failure of the
        standard fields, returns a neutral-default verdict with
        ``parse_success=False``.
        """
        # validate_judge_output checks only the 11 standard fields and ignores
        # the extra keys, so the 3 self-validation fields are tolerated.
        try:
            validate_judge_output(record_input)
        except ValueError as exc:
            logger.warning("record_judgment failed validation: %s", exc)
            return _neutral_result(), False

        verdict = dict(record_input)
        # Normalize the three extra fields with safe defaults.
        sva = verdict.get("self_validation_agrees")
        if sva not in _SELF_VALIDATION_VERDICTS:
            sva = "could_not_validate"
        verdict["self_validation_agrees"] = sva
        verdict["self_derived_answer"] = str(verdict.get("self_derived_answer", "") or "")
        verdict["validation_summary"] = str(verdict.get("validation_summary", "") or "")
        return verdict, True

    # -- public API -----------------------------------------------------------

    def judge(
        self,
        task: DABenchTask,
        response: AgentResponse,
        dataset_summary: dict,
        prechecks: dict,
        mode: str = "no_reference",
    ) -> dict[str, Any]:
        """Judge one agent response, recomputing the answer first.

        Parameters
        ----------
        task, response:
            The locked DABench task / agent-response contracts.
        dataset_summary:
            Compact CSV summary dict (from ``schema.summarize_csv``).
        prechecks:
            Deterministic pre-check results dict (from ``prechecks.py``).
        mode:
            Only ``"no_reference"`` is supported (no reference answer is shown).

        Returns
        -------
        dict
            The 11 standard judge fields, the 3 self-validation fields
            (``self_derived_answer``, ``self_validation_agrees``,
            ``validation_summary``), plus ``n_tool_calls``, ``parse_success``,
            and ``token_usage``.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"unknown judge mode {mode!r}; SelfValidatingJudge only "
                "supports 'no_reference'"
            )

        prompt = self._build_prompt(task, response, dataset_summary, prechecks)

        # The run_python tool executes in the directory containing the CSV.
        if task.file_path is not None:
            working_dir = Path(task.file_path).parent
        else:
            working_dir = Path.cwd()

        verdict, parse_success, n_tool_calls, token_usage = self._agentic_loop(
            prompt, working_dir
        )

        result: dict[str, Any] = {
            field: verdict[field] for field in _SCORE_FIELDS
        }
        result["critical_errors"] = verdict["critical_errors"]
        result["overall_pass"] = verdict["overall_pass"]
        result["confidence"] = verdict["confidence"]
        result["judge_explanation"] = verdict["judge_explanation"]
        result["self_derived_answer"] = verdict["self_derived_answer"]
        result["self_validation_agrees"] = verdict["self_validation_agrees"]
        result["validation_summary"] = verdict["validation_summary"]
        result["n_tool_calls"] = n_tool_calls
        result["parse_success"] = parse_success
        result["token_usage"] = token_usage
        return result
