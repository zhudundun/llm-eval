"""
The LLM-judge core (Steps 4, 6, 7, 8, 9 of the design doc).

``RubricJudge`` evaluates one ``AgentResponse`` against a ``DABenchTask`` using
Claude Opus 4.7 as the judge model.  Reliable structured JSON is obtained via
Anthropic tool-use: a single tool ``record_judgment`` whose ``input_schema`` is
the judge_result schema, forced with ``tool_choice={"type": "tool", ...}``.

If the model returns text instead of (or in addition to) a tool call, or the tool
input fails validation, the judge makes ONE repair attempt; if that also fails it
returns a neutral-default ``RubricJudgeResult`` with ``parse_success=False``.

The deterministic-layer fields on ``RubricJudgeResult``
(``computed_process_quality_score``, ``computed_overall_pass``, ``builtin_pass``,
``classification``) are left at their defaults — they are filled in downstream by
``scoring.py``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from src.benchmark.task import AgentResponse, DABenchTask, RubricJudgeResult
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
_TOOL_NAME = "record_judgment"
_PROMPTS_DIR = Path(__file__).parent / "prompts"

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


def _build_tool_schema() -> dict[str, Any]:
    """Construct the ``record_judgment`` tool definition from the 11 fields.

    Scores are integers constrained to 1-5; ``critical_errors`` is an array of
    objects whose ``type`` is restricted to ``FAILURE_TYPES`` and ``severity`` to
    {low, medium, high}.
    """
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
        },
        "required": [
            *_SCORE_FIELDS,
            "critical_errors",
            "overall_pass",
            "confidence",
            "judge_explanation",
        ],
        "additionalProperties": False,
    }

    return {
        "name": _TOOL_NAME,
        "description": (
            "Record the structured judgment of a data-science agent's solution. "
            "Call this exactly once with all rubric scores, any critical errors, "
            "the overall pass/fail decision, confidence, and an explanation."
        ),
        "input_schema": input_schema,
    }


def _neutral_judge_result() -> dict[str, Any]:
    """A neutral-default judge_result dict used when parsing/validation fails."""
    return {
        **{field: 3 for field in _SCORE_FIELDS},
        "critical_errors": [],
        "overall_pass": False,
        "confidence": 0.0,
        "judge_explanation": (
            "Judge output could not be parsed or validated after a repair attempt; "
            "neutral default scores applied."
        ),
    }


class RubricJudge:
    """LLM-as-a-judge for data-science agent outputs (no-reference & reference-aware)."""

    PROMPT_VERSION = "judge_v1"

    _VALID_MODES: tuple[str, ...] = ("no_reference", "with_reference")

    def __init__(
        self,
        model: str = _DEFAULT_MODEL,
        temperature: float = 0.0,
        max_tokens: int = 1500,
        prompt_version: str = "v1",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.prompt_version = prompt_version
        # Per-instance label that overrides the class ``PROMPT_VERSION`` constant
        # so v2 runs are stamped ``judge_v2`` on every result row.
        self.judge_prompt_version = f"judge_{prompt_version}"
        self._client: Optional[Any] = None
        self._tool = _build_tool_schema()
        # Templates are loaded lazily, per mode, on first use (see
        # ``_template_for``).  This keeps construction cheap and tolerant of
        # (mode, version) combinations whose file does not yet exist — only the
        # mode that is actually requested needs its file to be present.
        self._templates: dict[str, str] = {}

    @property
    def client(self) -> Any:
        """Lazily-created ``anthropic.Anthropic`` client (keeps module importable)."""
        if self._client is None:
            import anthropic  # local import — no key needed to import this module

            self._client = anthropic.Anthropic()
        return self._client

    # -- prompt construction --------------------------------------------------

    @staticmethod
    def _format_reference_answer(common_answers: Optional[list[list[str]]]) -> str:
        if not common_answers:
            return "(no reference answer available)"
        return json.dumps(common_answers, ensure_ascii=False)

    @staticmethod
    def _format_builtin_result(builtin: Optional[float]) -> str:
        if builtin is None:
            return "(no built-in evaluator result available)"
        passed = builtin >= 1.0
        return f"{builtin} ({'pass' if passed else 'fail'})"

    @staticmethod
    def _format_verifier_fields(
        verifier_result: Optional[dict[str, Any]],
    ) -> dict[str, str]:
        """Build the three verifier prompt placeholders from a plain dict.

        The ``verifier_result`` is treated as a plain dict per the locked
        JSONL contract (NOT the verifier dataclass).  When it is ``None`` the
        section renders as "(not run)" so the judge knows to ignore it.
        """
        if verifier_result is None:
            return {
                "verifier_answer": "(not run)",
                "verifier_agreement": "(not run)",
                "agent_reproduces": "(not run)",
            }

        verifier_answer = json.dumps(
            verifier_result["verifier_answer"], ensure_ascii=False
        )

        agrees = verifier_result["agrees_with_agent"]
        detail = verifier_result["agreement_detail"]
        if agrees is True:
            verifier_agreement = "AGREE"
        elif agrees is False:
            verifier_agreement = f"DISAGREE — {detail}"
        else:  # agrees is None
            verifier_agreement = f"UNABLE TO COMPARE — {detail}"

        reproduces = verifier_result["agent_code_reproduces"]
        if reproduces is True:
            agent_reproduces = "yes"
        elif reproduces is False:
            agent_reproduces = "NO (answer not reproduced on re-run)"
        else:  # reproduces is None
            agent_reproduces = "(unknown)"

        return {
            "verifier_answer": verifier_answer,
            "verifier_agreement": verifier_agreement,
            "agent_reproduces": agent_reproduces,
        }

    def _template_for(self, mode: str) -> str:
        """Return the prompt template for ``mode`` at the selected version.

        Templates are loaded lazily and cached per mode.  The file is resolved
        as ``judge_{mode}_{prompt_version}.txt`` under ``_PROMPTS_DIR``.  A
        clear ``FileNotFoundError`` is raised only when this specific
        (mode, version) file is actually requested and missing, so constructing
        a judge for a version whose file does not yet exist never crashes.
        """
        if mode not in self._templates:
            path = _PROMPTS_DIR / f"judge_{mode}_{self.prompt_version}.txt"
            try:
                self._templates[mode] = path.read_text(encoding="utf-8")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"prompt template for mode {mode!r} version "
                    f"{self.prompt_version!r} not found: {path}"
                ) from exc
        return self._templates[mode]

    def _build_prompt(
        self,
        task: DABenchTask,
        response: AgentResponse,
        dataset_summary: dict,
        prechecks: dict,
        mode: str,
        verifier_result: Optional[dict[str, Any]] = None,
    ) -> str:
        fields: dict[str, str] = {
            "question": task.question,
            "dataset_summary": json.dumps(dataset_summary, ensure_ascii=False),
            "precheck_results": json.dumps(prechecks, ensure_ascii=False),
            "agent_final_answer": json.dumps(response.answer, ensure_ascii=False),
            "agent_code": response.code or "(no code provided)",
            "stdout": response.stdout or "(empty)",
            "stderr": response.stderr or "(empty)",
        }
        fields.update(self._format_verifier_fields(verifier_result))
        if mode == "with_reference":
            fields["reference_answer"] = self._format_reference_answer(
                task.common_answers
            )
            fields["builtin_result"] = self._format_builtin_result(
                response.builtin_eval_result
            )
        return self._template_for(mode).format(**fields)

    # -- model call -----------------------------------------------------------

    def _call_tool(self, messages: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        """Call Claude forcing ``record_judgment``; return the tool input or None.

        Note: Opus 4.7 removed the ``temperature`` sampling parameter (sending it
        returns a 400), so ``self.temperature`` is intentionally NOT forwarded to
        the API.  It is retained on the instance / ``judge()`` for interface
        compatibility with self-consistency tests on models that accept it.
        """
        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            tools=[self._tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            messages=messages,
        )
        for block in message.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                # block.input is already a parsed dict.
                return dict(block.input)
        return None

    def _judge_with_repair(self, prompt: str) -> tuple[dict[str, Any], bool]:
        """Run the judge call, validate, and retry once on failure.

        Returns ``(judge_result_dict, parse_success)``.  On unrecoverable failure
        returns a neutral-default dict with ``parse_success=False``.
        """
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        # Cache the (large, static-per-call) prompt body. The
                        # volatile parts already live inside this single block;
                        # caching still helps the repair retry below, which
                        # re-sends the identical prefix.
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]

        # First attempt.
        try:
            tool_input = self._call_tool(messages)
            if tool_input is not None:
                return validate_judge_output(tool_input), True
            first_error = "model did not call the record_judgment tool"
        except ValueError as exc:  # validation failure
            first_error = f"validation failed: {exc}"
            tool_input = locals().get("tool_input")
        except Exception as exc:  # noqa: BLE001 - API/transport error
            logger.warning("Judge call failed: %s", exc)
            return _neutral_judge_result(), False

        # One repair attempt: tell the model what was wrong and ask again.
        logger.info("Judge output invalid (%s); attempting one repair retry", first_error)
        repair_messages = messages + [
            {
                "role": "user",
                "content": (
                    "Your previous attempt was invalid: "
                    f"{first_error}. "
                    "Call the record_judgment tool exactly once with valid values: "
                    "all seven score fields must be integers 1-5, confidence a number "
                    "in [0, 1], overall_pass a boolean, and each critical_errors entry "
                    "must use a type from the allowed taxonomy with severity "
                    "low/medium/high."
                ),
            }
        ]
        try:
            tool_input = self._call_tool(repair_messages)
            if tool_input is not None:
                return validate_judge_output(tool_input), True
        except ValueError as exc:
            logger.warning("Judge repair retry still invalid: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Judge repair retry call failed: %s", exc)

        return _neutral_judge_result(), False

    # -- public API -----------------------------------------------------------

    def judge(
        self,
        task: DABenchTask,
        response: AgentResponse,
        dataset_summary: dict,
        prechecks: dict,
        mode: str = "no_reference",
        temperature: float | None = None,
        verifier_result: dict[str, Any] | None = None,
    ) -> RubricJudgeResult:
        """Judge one agent response and return a ``RubricJudgeResult``.

        Parameters
        ----------
        task, response:
            The locked DABench task / agent-response contracts.
        dataset_summary:
            Compact CSV summary dict (from ``schema.summarize_csv``).
        prechecks:
            Deterministic pre-check results dict (from ``prechecks.py``).
        mode:
            ``"no_reference"`` (production) or ``"with_reference"`` (benchmark).
        temperature:
            Optional per-call override of the judge temperature.  Stored on the
            instance for the duration of the call but NOT forwarded to Opus 4.7
            (which rejects sampling parameters); kept for interface parity with
            self-consistency tests against models that accept it.
        verifier_result:
            Optional independent-verifier result for this task, as a plain dict
            from the locked JSONL contract (NOT the verifier dataclass).  When
            omitted/``None`` the verifier prompt section renders as "(not run)".
        """
        if mode not in self._VALID_MODES:
            raise ValueError(
                f"unknown judge mode {mode!r}; expected 'no_reference' or 'with_reference'"
            )

        if temperature is not None:
            self.temperature = temperature

        prompt = self._build_prompt(
            task, response, dataset_summary, prechecks, mode, verifier_result
        )
        judge_result, parse_success = self._judge_with_repair(prompt)

        return RubricJudgeResult(
            task_id=task.id,
            agent_model=response.model,
            judge_model=self.model,
            judge_prompt_version=self.judge_prompt_version,
            judge_mode=mode,
            task_success_score=judge_result["task_success_score"],
            code_correctness_score=judge_result["code_correctness_score"],
            data_handling_score=judge_result["data_handling_score"],
            statistical_validity_score=judge_result["statistical_validity_score"],
            reproducibility_score=judge_result["reproducibility_score"],
            final_answer_quality_score=judge_result["final_answer_quality_score"],
            reasoning_quality_score=judge_result["reasoning_quality_score"],
            critical_errors=judge_result["critical_errors"],
            overall_pass=judge_result["overall_pass"],
            confidence=judge_result["confidence"],
            judge_explanation=judge_result["judge_explanation"],
            parse_success=parse_success,
        )
