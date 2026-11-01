"""
Blind-recompute LLM judge for data-science agent outputs.

This is a variant of :class:`SelfValidatingJudge`.  The self-validating judge
recomputes the answer itself, but it sees the agent's code/answer FIRST, so it
anchors to the agent and can miss subtle wrong answers.  The
``BlindRecomputeJudge`` fixes that by splitting the work into TWO phases inside a
single Opus 4.7 agent:

  * PHASE 1 -- BLIND SOLVE.  The model is framed purely as a data analyst.  It
    sees ONLY the task (question, constraints, format, file_name) and the dataset
    summary -- never the agent's code, answer, stdout, stderr, or prechecks.  It
    uses ``run_python`` to compute, then commits its own answer via
    ``submit_blind_answer``.  That committed value is captured as
    ``self_derived_answer``.

  * PHASE 2 -- REVEAL & JUDGE.  The SAME message history is kept (so the blind
    answer and any computed evidence stay in context), then a new user turn
    reveals the agent's solution and asks the model to evaluate it, comparing the
    agent to the answer IT ALREADY committed.  ``record_judgment`` is forced via
    ``tool_choice``.

By committing an answer before the agent's work is revealed, the judge's
independence is restored: it cannot anchor to a value it has not yet seen.

The structured verdict reuses the same ``record_judgment`` schema as the
self-validating judge (the 11 standard fields plus ``self_validation_agrees`` and
``validation_summary``); ``self_derived_answer`` is captured from phase 1 and is
NOT re-requested in phase 2.  The 11 standard fields are validated with
``validate_judge_output``.
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
_SUBMIT_TOOL_NAME = "submit_blind_answer"
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

# The two extra self-validation fields the record_judgment tool carries on top of
# the 11 standard judge fields.  (self_derived_answer is captured in phase 1, not
# from record_judgment.)
_EXTRA_FIELDS: tuple[str, ...] = (
    "self_validation_agrees",
    "validation_summary",
)

# --- PHASE 1 prompts (short, so kept inline) --------------------------------

_PHASE1_SYSTEM = (
    "You are an expert data analyst. You solve data-analysis tasks correctly by "
    "loading the data and computing the answer yourself with code. You always "
    "verify your numbers against the actual data before committing to an answer."
)

_PHASE1_USER_TEMPLATE = """Solve the following data-analysis task yourself.

Task:
{question}

Constraints:
{constraints}

Required answer format:
{format}

CSV file name (in your working directory):
{file_name}

Dataset summary:
{dataset_summary}

You have a `run_python` tool that executes Python in the directory containing the
CSV. Use it to load the CSV ('{file_name}') and compute the answer from scratch.
- Load the CSV directly, e.g. `import pandas as pd; df = pd.read_csv('{file_name}')`.
- Always `print(...)` the values you want to see; only stdout/stderr are returned.
- You may call `run_python` more than once.

When you are confident in your answer, call `submit_blind_answer` exactly once
with your final computed answer (as a string, in or near the requested format)
and a short reasoning describing how you computed it. Do this independently --
there is no reference answer; rely solely on your own computation."""


def _run_python_tool() -> dict[str, Any]:
    """The ``run_python`` tool: executes Python in the CSV's directory."""
    return {
        "name": _RUN_TOOL_NAME,
        "description": (
            "Execute Python code in the directory containing the task's CSV file. "
            "Use it to load the CSV and compute the answer. Only stdout and stderr "
            "are returned, so print() any value you want to inspect. You may call "
            "this tool multiple times."
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


def _submit_blind_answer_tool() -> dict[str, Any]:
    """The ``submit_blind_answer`` tool used to COMMIT the phase-1 answer."""
    return {
        "name": _SUBMIT_TOOL_NAME,
        "description": (
            "Commit your own, independently-computed answer to the task. Call this "
            "exactly once, after you have computed the answer with run_python."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "Your final computed answer as a string, in or near the "
                        "requested format (e.g. 'mean_fare=34.65')."
                    ),
                },
                "reasoning": {
                    "type": "string",
                    "description": (
                        "Short description of how you computed this answer."
                    ),
                },
            },
            "required": ["answer", "reasoning"],
            "additionalProperties": False,
        },
    }


def _record_judgment_tool() -> dict[str, Any]:
    """The ``record_judgment`` tool: 11 standard fields + 2 self-validation fields."""
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
                "description": "Whether the agent's answer should be accepted overall.",
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
            # --- self-validation fields (self_derived_answer NOT requested here;
            #     it is captured from the phase-1 submit_blind_answer call) ---
            "self_validation_agrees": {
                "type": "string",
                "enum": list(_SELF_VALIDATION_VERDICTS),
                "description": (
                    "Whether the agent's answer agrees with the answer YOU "
                    "committed blindly in phase 1."
                ),
            },
            "validation_summary": {
                "type": "string",
                "description": (
                    "How the agent's answer compares to your blind answer: what "
                    "you computed, what the agent reported, and the comparison."
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
            "Record the structured judgment of the agent's solution. Call this "
            "exactly once, AFTER the agent's solution has been revealed, with all "
            "rubric scores, any critical errors, the overall pass/fail decision, "
            "confidence, an explanation, and the two self-validation fields."
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
        "self_validation_agrees": "could_not_validate",
        "validation_summary": "Phase-2 judgment did not complete.",
    }


def _truncate(text: str, limit: int = _TOOL_RESULT_TRUNCATE) -> str:
    """Truncate ``text`` to ``limit`` chars, appending a marker when cut."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text) - limit} more chars]"


class BlindRecomputeJudge:
    """Agentic LLM judge that solves the task BLIND, then judges the agent."""

    PROMPT_VERSION = "judge_blind_recompute_v1"

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
        self._submit_tool = _submit_blind_answer_tool()
        self._record_tool = _record_judgment_tool()
        self._phase2_template: Optional[str] = None

    @property
    def client(self) -> Any:
        """Lazily-created ``anthropic.Anthropic`` client (keeps module importable)."""
        if self._client is None:
            import anthropic  # local import — no key needed to import this module

            self._client = anthropic.Anthropic()
        return self._client

    # -- prompt construction --------------------------------------------------

    def _phase2_template_text(self) -> str:
        if self._phase2_template is None:
            path = _PROMPTS_DIR / "judge_blind_recompute_v1.txt"
            self._phase2_template = path.read_text(encoding="utf-8")
        return self._phase2_template

    def _build_phase1_prompt(
        self,
        task: DABenchTask,
        dataset_summary: dict,
    ) -> str:
        return _PHASE1_USER_TEMPLATE.format(
            question=task.question,
            constraints=task.constraints or "(none)",
            format=task.format or "(unspecified)",
            file_name=task.file_name,
            dataset_summary=json.dumps(dataset_summary, ensure_ascii=False),
        )

    def _build_phase2_prompt(
        self,
        response: AgentResponse,
        prechecks: dict,
        blind_answer: str,
    ) -> str:
        fields: dict[str, str] = {
            "agent_final_answer": json.dumps(response.answer, ensure_ascii=False),
            "agent_code": response.code or "(no code provided)",
            "stdout": response.stdout or "(empty)",
            "stderr": response.stderr or "(empty)",
            "precheck_results": json.dumps(prechecks, ensure_ascii=False),
            "blind_answer": blind_answer or "(no blind answer was committed)",
        }
        return self._phase2_template_text().format(**fields)

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

    # -- tool execution -------------------------------------------------------

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

    # -- PHASE 1: blind solve -------------------------------------------------

    def _phase1_blind_solve(
        self,
        prompt: str,
        working_dir: Path,
        messages: list[dict[str, Any]],
        token_usage: dict[str, int],
    ) -> tuple[str, int]:
        """Run the blind-solve loop, mutating ``messages`` in place.

        Returns ``(self_derived_answer, n_tool_calls)``.  ``messages`` is left
        ready to receive the phase-2 reveal turn (i.e. every tool_use in the
        history has a matching tool_result already appended).
        """
        messages.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        )
        tools = [self._run_tool, self._submit_tool]
        n_tool_calls = 0
        self_derived_answer = ""

        for iteration in range(self.max_tool_iters):
            last_iter = iteration == self.max_tool_iters - 1
            # Force the blind answer on the last phase-1 iteration so we always
            # capture a committed value.
            if last_iter:
                tool_choice: dict[str, Any] = {"type": "tool", "name": _SUBMIT_TOOL_NAME}
            else:
                tool_choice = {"type": "auto"}

            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    system=_PHASE1_SYSTEM,
                    tools=tools,
                    tool_choice=tool_choice,
                    messages=messages,
                )
            except Exception as exc:  # noqa: BLE001 - API/transport error
                logger.warning("Blind-recompute judge phase-1 call failed: %s", exc)
                return self_derived_answer, n_tool_calls

            self._accumulate_usage(token_usage, message)

            assistant_content = [
                self._block_to_dict(block) for block in message.content
            ]
            messages.append({"role": "assistant", "content": assistant_content})

            tool_results: list[dict[str, Any]] = []
            submitted = False

            for block in message.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                if block.name == _SUBMIT_TOOL_NAME:
                    answer = str((block.input or {}).get("answer", "") or "")
                    self_derived_answer = answer
                    submitted = True
                    # Acknowledge the submit tool_use so the message history is
                    # well-formed for the phase-2 continuation.
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": "Blind answer recorded.",
                        }
                    )
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

            # Append tool_results (if any) so every tool_use is answered.
            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if submitted:
                return self_derived_answer, n_tool_calls

            # Model produced neither tool call: nudge it (unless this was already
            # the final, forced turn).
            if not tool_results and not last_iter:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not yet committed an answer. Use run_python "
                            "to compute it, then call submit_blind_answer exactly "
                            "once with your final answer and reasoning."
                        ),
                    }
                )

        return self_derived_answer, n_tool_calls

    # -- PHASE 2: reveal & judge ----------------------------------------------

    def _phase2_judge(
        self,
        reveal_prompt: str,
        working_dir: Path,
        messages: list[dict[str, Any]],
        token_usage: dict[str, int],
        n_tool_calls: int,
    ) -> tuple[dict[str, Any], bool, int]:
        """Run the reveal-and-judge loop, continuing the SAME ``messages``.

        Returns ``(verdict_dict, parse_success, n_tool_calls)``.
        ``n_tool_calls`` is the running run_python count carried in from phase 1
        and incremented for any further run_python calls here.
        """
        messages.append(
            {
                "role": "user",
                "content": [{"type": "text", "text": reveal_prompt}],
            }
        )
        tools = [self._run_tool, self._record_tool]

        for iteration in range(self.max_tool_iters):
            last_iter = iteration == self.max_tool_iters - 1
            if last_iter:
                tool_choice: dict[str, Any] = {
                    "type": "tool",
                    "name": _RECORD_TOOL_NAME,
                }
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
                logger.warning("Blind-recompute judge phase-2 call failed: %s", exc)
                return _neutral_result(), False, n_tool_calls

            self._accumulate_usage(token_usage, message)

            assistant_content = [
                self._block_to_dict(block) for block in message.content
            ]
            messages.append({"role": "assistant", "content": assistant_content})

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

            if record_input is not None:
                verdict, ok = self._validate_record(record_input)
                return verdict, ok, n_tool_calls

            if tool_results:
                messages.append({"role": "user", "content": tool_results})
                continue

            if not last_iter:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You have not yet recorded a verdict. Compare the "
                            "agent's answer to the one you committed blindly, then "
                            "call record_judgment exactly once with all required "
                            "fields including the two self-validation fields."
                        ),
                    }
                )
                continue

        logger.info(
            "Blind-recompute judge exhausted %d phase-2 iterations without a verdict.",
            self.max_tool_iters,
        )
        return _neutral_result(), False, n_tool_calls

    def _validate_record(
        self, record_input: dict[str, Any]
    ) -> tuple[dict[str, Any], bool]:
        """Validate the 11 standard fields; tolerate/normalize the 2 extras.

        Returns ``(verdict_dict, parse_success)``.  On validation failure of the
        standard fields, returns a neutral-default verdict with
        ``parse_success=False``.
        """
        try:
            validate_judge_output(record_input)
        except ValueError as exc:
            logger.warning("record_judgment failed validation: %s", exc)
            return _neutral_result(), False

        verdict = dict(record_input)
        sva = verdict.get("self_validation_agrees")
        if sva not in _SELF_VALIDATION_VERDICTS:
            sva = "could_not_validate"
        verdict["self_validation_agrees"] = sva
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
        """Judge one agent response after solving the task BLIND first.

        Parameters
        ----------
        task, response:
            The locked DABench task / agent-response contracts.
        dataset_summary:
            Compact CSV summary dict (from ``schema.summarize_csv``).
        prechecks:
            Deterministic pre-check results dict (from ``prechecks.py``).  Only
            revealed in phase 2 -- never in the blind phase.
        mode:
            Only ``"no_reference"`` is supported (no reference answer is shown).

        Returns
        -------
        dict
            The 11 standard judge fields, ``self_derived_answer`` (from phase 1),
            ``self_validation_agrees``, ``validation_summary``, plus
            ``n_tool_calls`` (run_python across BOTH phases), ``parse_success``,
            and ``token_usage``.
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"unknown judge mode {mode!r}; BlindRecomputeJudge only "
                "supports 'no_reference'"
            )

        # The run_python tool executes in the directory containing the CSV.
        if task.file_path is not None:
            working_dir = Path(task.file_path).parent
        else:
            working_dir = Path.cwd()

        token_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        messages: list[dict[str, Any]] = []

        # --- PHASE 1: blind solve (sees ONLY task + dataset summary) ---
        phase1_prompt = self._build_phase1_prompt(task, dataset_summary)
        self_derived_answer, n_tool_calls = self._phase1_blind_solve(
            phase1_prompt, working_dir, messages, token_usage
        )

        # --- PHASE 2: reveal agent + judge (same message history) ---
        reveal_prompt = self._build_phase2_prompt(
            response, prechecks, self_derived_answer
        )
        verdict, parse_success, n_tool_calls = self._phase2_judge(
            reveal_prompt, working_dir, messages, token_usage, n_tool_calls
        )

        result: dict[str, Any] = {
            field: verdict[field] for field in _SCORE_FIELDS
        }
        result["critical_errors"] = verdict["critical_errors"]
        result["overall_pass"] = verdict["overall_pass"]
        result["confidence"] = verdict["confidence"]
        result["judge_explanation"] = verdict["judge_explanation"]
        result["self_derived_answer"] = self_derived_answer
        result["self_validation_agrees"] = verdict["self_validation_agrees"]
        result["validation_summary"] = verdict["validation_summary"]
        result["n_tool_calls"] = n_tool_calls
        result["parse_success"] = parse_success
        result["token_usage"] = token_usage
        return result
